import gymnasium as gym
import numpy as np
import pandas as pd
from collections import deque
from typing import Tuple, Dict, Any, List, Optional
import datetime as dt
from python_bot.reward_engine import RewardEngine
from python_bot.features.feature_schema import get_numeric_feature_columns

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
        
        # Observation Space: Dynamic based on numeric features + Account Info (4)
        self.feature_columns = get_numeric_feature_columns(self.df)
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
        self.last_trade_step = -10**9
        self.min_trade_interval = 30
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

        # 2. Schema Validation
        expected_dim = self.observation_space.shape[0]
        if obs.shape[0] != expected_dim:
            raise RuntimeError(f"OBSERVATION_DIM_MISMATCH: got {obs.shape[0]} expected {expected_dim}. "
                               f"Numeric features: {len(self.feature_columns)}, Info: 4.")
        
        return obs.astype(np.float32)

    def step(self, action: int):
        """
        Stable long-only execution contract.

        Action mapping:
        0 = HOLD
        1 = BUY
        2 = SELL/CLOSE

        Design:
        - No early return before done/reward/info are defined.
        - Cooldown blocks execution but still advances env by one step.
        - Reward is always finite and clipped for PPO stability.
        """
        action = int(action)

        # Safe defaults
        reward = 0.0
        components = {}
        transaction_cost = 0.0
        rejected_reason = None
        executed_qty = 0.0
        executed_price = None
        position_changed = False

        current_data = self.df.iloc[self.current_step]
        price = float(current_data["close"])
        volume = float(current_data["volume"]) if "volume" in current_data else 0.0

        fee_rate = 0.0015
        tax_rate = 0.001
        lot_size = 100

        # Ensure cooldown state exists
        if not hasattr(self, "last_trade_step"):
            self.last_trade_step = -10**9
        if not hasattr(self, "min_trade_interval"):
            self.min_trade_interval = 30

        position_before = float(self.available_shares + sum(self.t_plus_queue))
        cash_before = float(self.cash)
        equity_before = float(self.cash + position_before * price)

        # --------------------------------------------------
        # 1. COOLDOWN CHECK
        # --------------------------------------------------
        cooldown_active = False
        if action in (1, 2):
            if (self.current_step - int(self.last_trade_step)) < int(self.min_trade_interval):
                cooldown_active = True
                rejected_reason = "TRADE_COOLDOWN"

        # --------------------------------------------------
        # 2. EXECUTION
        # --------------------------------------------------
        if not cooldown_active:
            if action == 1:  # BUY
                if price <= 0:
                    rejected_reason = "INVALID_PRICE"
                elif self.cash <= 0:
                    rejected_reason = "INSUFFICIENT_CASH"
                elif volume <= 0:
                    rejected_reason = "NO_VOLUME"
                else:
                    max_position_pct = 0.10
                    participation_limit = 0.05

                    budget = min(self.cash * 0.95, equity_before * max_position_pct)

                    budget_shares = int((budget / (price * (1.0 + fee_rate))) // lot_size * lot_size)
                    volume_shares = int((volume * participation_limit) // lot_size * lot_size)
                    shares_to_buy = min(budget_shares, volume_shares)

                    if budget_shares <= 0:
                        rejected_reason = "ORDER_BELOW_LOT_SIZE_BY_BUDGET"
                    elif volume_shares <= 0:
                        rejected_reason = "ORDER_TOO_SMALL_BY_LIQUIDITY"
                    elif shares_to_buy <= 0:
                        rejected_reason = "ORDER_BELOW_LOT_SIZE"
                    else:
                        exec_price = self._calculate_slippage(price, shares_to_buy, volume, "buy")

                        if exec_price <= 0:
                            rejected_reason = "INVALID_EXEC_PRICE"
                        else:
                            cost = shares_to_buy * exec_price
                            fee = cost * fee_rate
                            total_cost = cost + fee

                            if self.cash >= total_cost:
                                self.cash -= total_cost
                                self.t_plus_queue[0] += shares_to_buy

                                self.trade_count += 1
                                self.last_trade_step = self.current_step

                                transaction_cost = fee
                                executed_qty = float(shares_to_buy)
                                executed_price = float(exec_price)
                                rejected_reason = None
                            else:
                                rejected_reason = "INSUFFICIENT_CASH_AFTER_SLIPPAGE"

            elif action == 2:  # SELL/CLOSE
                total_pos_before_sell = float(self.available_shares + sum(self.t_plus_queue))

                if total_pos_before_sell <= 0:
                    rejected_reason = "NO_POSITION_TO_SELL"
                elif price <= 0:
                    rejected_reason = "INVALID_PRICE"
                else:
                    shares_to_sell = total_pos_before_sell
                    exec_price = self._calculate_slippage(price, shares_to_sell, volume, "sell")

                    if exec_price <= 0:
                        rejected_reason = "INVALID_EXEC_PRICE"
                    else:
                        proceeds = shares_to_sell * exec_price
                        fee = proceeds * fee_rate
                        tax = proceeds * tax_rate

                        self.cash += proceeds - fee - tax

                        self.available_shares = 0.0
                        self.t_plus_queue[0] = 0.0
                        self.t_plus_queue[1] = 0.0

                        self.trade_count += 1
                        self.last_trade_step = self.current_step

                        transaction_cost = fee + tax
                        executed_qty = float(shares_to_sell)
                        executed_price = float(exec_price)
                        rejected_reason = None

            elif action == 0:
                rejected_reason = None
            else:
                rejected_reason = f"INVALID_ACTION_{action}"

        # --------------------------------------------------
        # 3. FORWARD RETURNS FOR REWARD ENGINE
        # --------------------------------------------------
        forward_rets = {}
        for window in [5, 10]:
            target_idx = min(self.current_step + window, len(self.df) - 1)
            future_price = float(self.df.iloc[target_idx]["close"])
            forward_rets[window] = (future_price - price) / (price + 1e-9)

        # --------------------------------------------------
        # 4. T+2 INVENTORY CYCLE
        # --------------------------------------------------
        self.available_shares += self.t_plus_queue[1]
        self.t_plus_queue[1] = self.t_plus_queue[0]
        self.t_plus_queue[0] = 0.0

        # --------------------------------------------------
        # 5. ADVANCE STEP
        # --------------------------------------------------
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        val_price = float(self.df.iloc[self.current_step]["close"]) if not done else price
        total_pos = float(self.available_shares + sum(self.t_plus_queue))
        current_equity = float(self.cash + total_pos * val_price)

        self.max_equity = max(self.max_equity, current_equity)
        drawdown = (self.max_equity - current_equity) / (self.max_equity + 1e-9)

        # --------------------------------------------------
        # 6. REWARD
        # --------------------------------------------------
        if current_equity <= 0:
            reward = -1.0
            components = {"terminal_penalty": -1.0}
            done = True
            rejected_reason = rejected_reason or "NEGATIVE_EQUITY"
        else:
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

        # Penalty for invalid / non-executed active actions
        extra_penalty = 0.0
        if rejected_reason == "NO_POSITION_TO_SELL":
            extra_penalty -= 0.05
        elif rejected_reason == "TRADE_COOLDOWN":
            extra_penalty -= 0.03
        elif rejected_reason in ("ORDER_TOO_SMALL_BY_LIQUIDITY", "INVALID_EXEC_PRICE"):
            extra_penalty -= 0.02
        elif action in (1, 2) and executed_qty <= 0:
            extra_penalty -= 0.02

        reward = float(reward) + extra_penalty

        reward = float(np.nan_to_num(reward, nan=-1.0, posinf=1.0, neginf=-1.0))
        reward = float(np.clip(reward, -1.0, 1.0))

        if not isinstance(components, dict):
            components = {}
        components["extra_execution_penalty"] = extra_penalty
        components["final_reward_clipped"] = reward

        self.equity_history.append(current_equity)

        position_after = total_pos
        position_changed = bool(position_after != position_before)

        info = {
            "equity": current_equity,
            "cash": float(self.cash),
            "cash_before": cash_before,

            "position": position_after,
            "shares": position_after,
            "available_shares": float(self.available_shares),
            "pending_t1_shares": float(self.t_plus_queue[1]),
            "pending_t0_shares": float(self.t_plus_queue[0]),

            "raw_action": action,
            "interpreted_action": action,
            "position_before": position_before,
            "position_after": position_after,
            "position_changed": position_changed,
            "total_trades": int(self.trade_count),
            "executed_qty": float(executed_qty),
            "executed_price": executed_price,
            "transaction_cost": float(transaction_cost),
            "rejected_reason": rejected_reason,
            "reward_components": components,
        }

        return self._get_obs(), reward, done, False, info


    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Equity: {self.equity_history[-1]:,.0f}, Cash: {self.cash:,.0f}, Available: {self.available_shares}")
