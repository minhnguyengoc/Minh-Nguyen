import gymnasium as gym
import numpy as np
import pandas as pd
from collections import deque
from typing import Tuple, Dict, Any, List, Optional
import datetime as dt
from python_bot.reward_engine import RewardEngine

class VNStockTradingEnv(gym.Env):
    """
    Canonical Institutional-Grade Gym Environment for Vietnam Stock Market (HOSE).
    Implements: T+2 mechanism, HOSE Tick size, Lot size 100, and Slippage modeling.
    Integrates decomposed RewardEngine for policy alignment.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                 df: pd.DataFrame, 
                 initial_capital: float = 100_000_000.0,
                 max_slippage_ratio: float = 0.1):
        super(VNStockTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.max_slippage_ratio = max_slippage_ratio
        
        # Action Space: 0: HOLD, 1: BUY 100% NAV, 2: SELL 100% Available
        self.action_space = gym.spaces.Discrete(3)
        
        # Observation Space: Upgraded to match Feature Set (26) + Account Info (4) = 30
        self.feature_columns = [c for c in self.df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        n_features = len(self.feature_columns) + 4
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        
        self.reward_engine = RewardEngine()
        self.reset()

    def _get_tick_size(self, price: float) -> float:
        """Quy định bước giá (Tick size) trên HOSE"""
        if price < 10000:
            return 10.0
        elif price < 50000:
            return 50.0
        else:
            return 100.0

    def _apply_tick_constraints(self, price: float, side: str) -> float:
        tick = self._get_tick_size(price)
        if side == 'buy':
            return np.ceil(price / tick) * tick
        else:
            return np.floor(price / tick) * tick

    def _calculate_slippage(self, base_price: float, trade_vol: float, candle_vol: float, side: str) -> float:
        """Mô phỏng trượt giá dựa trên Volume của nến hiện tại"""
        if candle_vol <= 0: return base_price
        
        # Trượt giá tăng khi khối lượng giao dịch chiếm tỷ trọng lớn trong nến
        impact = (trade_vol / candle_vol) * self.max_slippage_ratio
        if side == 'buy':
            return self._apply_tick_constraints(base_price * (1 + impact), 'buy')
        else:
            return self._apply_tick_constraints(base_price * (1 - impact), 'sell')

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_capital
        self.t_plus_queue = deque([0.0, 0.0], maxlen=2) # [T+0, T+1]
        self.available_shares = 0.0
        self.equity_history = [self.initial_capital]
        self.max_equity = self.initial_capital
        self.trade_count = 0
        self.reward_engine = RewardEngine() # Reset engine state
        
        return self._get_obs(), {}

    def _get_obs(self):
        market_stats = self.df.iloc[self.current_step][self.feature_columns].values.astype(np.float32)
        price_now = self.df.iloc[self.current_step]['close']
        
        account_info = np.array([
            self.cash / self.initial_capital,
            (self.available_shares * price_now) / self.initial_capital,
            (self.t_plus_queue[1] * price_now) / self.initial_capital,
            (self.t_plus_queue[0] * price_now) / self.initial_capital
        ], dtype=np.float32)
        
        obs = np.concatenate([market_stats, account_info])
        
        # 1. Finite verification & clipping (Standard institutional guard)
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        obs = np.clip(obs, -5.0, 5.0)
        
        return obs

    def step(self, action: int):
        current_data = self.df.iloc[self.current_step]
        price = current_data['close']
        volume = current_data['volume']
        
        transaction_cost = 0.0
        
        # 1. Action Execution
        if action == 1: # BUY
            fee_rate = 0.0015
            max_buy_val = self.cash / (1 + fee_rate)
            shares_to_buy = (max_buy_val // (price * 100)) * 100
            
            if shares_to_buy > 0:
                exec_price = self._calculate_slippage(price, shares_to_buy, volume, 'buy')
                cost = shares_to_buy * exec_price
                fee = cost * fee_rate
                if self.cash >= (cost + fee):
                    self.cash -= (cost + fee)
                    self.t_plus_queue[0] += shares_to_buy
                    self.trade_count += 1
                    transaction_cost = fee
                    
        elif action == 2: # SELL 100% Available
            if self.available_shares > 0:
                shares_to_sell = self.available_shares
                exec_price = self._calculate_slippage(price, shares_to_sell, volume, 'sell')
                proceeds = shares_to_sell * exec_price
                fee = proceeds * 0.0015
                tax = proceeds * 0.001
                self.cash += (proceeds - fee - tax)
                self.available_shares = 0
                self.trade_count += 1
                transaction_cost = fee + tax

        # 2. Forward Return Calculation (For Reward Engine only)
        forward_rets = {}
        for window in [5, 10]:
            target_idx = min(self.current_step + window, len(self.df) - 1)
            future_price = self.df.iloc[target_idx]['close']
            forward_rets[window] = (future_price - price) / (price + 1e-9)

        # 3. Step Inventory (T+2 cycle)
        self.available_shares += self.t_plus_queue[1]
        self.t_plus_queue[1] = self.t_plus_queue[0]
        self.t_plus_queue[0] = 0.0
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # 4. Reward Calculation
        val_price = self.df.iloc[self.current_step]['close'] if not done else price
        total_pos = self.available_shares + sum(self.t_plus_queue)
        current_equity = self.cash + (total_pos * val_price)
        self.max_equity = max(self.max_equity, current_equity)
        drawdown = (self.max_equity - current_equity) / (self.max_equity + 1e-9)
        
        # Track position before/after
        pos_before = previous_total_pos if 'previous_total_pos' in locals() else total_pos # This is tricky in this structure
        # Let's just use the ones I calculated
        
        reward, components = self.reward_engine.compute_reward(
            equity=current_equity,
            initial_capital=self.initial_capital,
            drawdown=drawdown,
            action=action,
            position=total_pos,
            forward_returns=forward_rets,
            transaction_cost=transaction_cost,
            terminated=done
        )
        
        self.equity_history.append(current_equity)
        
        # 5. Info Pack
        info = {
            "equity": current_equity,
            "shares": total_pos,
            "cash": self.cash,
            "reward_components": components,
            
            # Stage 4.1 Diagnostics
            "raw_action": action,
            "interpreted_action": action,
            "position_before": getattr(self, '_prev_total_pos', total_pos),
            "position_after": total_pos,
            "position_changed": getattr(self, '_prev_total_pos', total_pos) != total_pos,
            "total_trades": self.trade_count,
            "rejected_reason": "INSUFFICIENT_FUNDS_OR_SHARES" if (action > 0 and getattr(self, '_prev_total_pos', total_pos) == total_pos) else None
        }
        self._prev_total_pos = total_pos
        
        return self._get_obs(), float(reward), done, False, info


    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Equity: {self.equity_history[-1]:,.0f}, Cash: {self.cash:,.0f}, Available: {self.available_shares}")
