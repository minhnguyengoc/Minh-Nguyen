import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque
import logging
from python_bot.backtest.cost_model import InstitutionalCostModel, CostMode

class InstitutionalBacktester:
    """
    Event-driven backtester for VNStock.
    Features: 
    - T+2 settlement queue
    - Lot size 100 fulfillment
    - Order execution delay (T+1 fill)
    - Realistic cost modeling (Slippage/Impact)
    """
    def __init__(self, 
                 initial_capital: float = 100_000_000.0,
                 cost_mode: CostMode = CostMode.NORMAL,
                 max_pos_nav: float = 1.0):
        self.initial_capital = initial_capital
        self.cost_model = InstitutionalCostModel(mode=cost_mode)
        self.max_pos_nav = max_pos_nav
        self.logger = logging.getLogger("Backtester")
        self.reset()

    def reset(self):
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        self.positions = {} # {symbol: shares}
        # T+2 Logic: {symbol: deque([T+0, T+1])}
        self.t_plus_queues = {} 
        self.available_shares = {}
        
        self.history = []
        self.trade_log = []
        self.max_equity = self.initial_capital

    def _sync_t_plus(self, symbol: str):
        """Advances T+2 settlement for a symbol."""
        if symbol not in self.t_plus_queues:
            self.t_plus_queues[symbol] = deque([0, 0], maxlen=2)
            self.available_shares[symbol] = 0
            
        settled = self.t_plus_queues[symbol].pop() # T+1 becomes settled
        self.available_shares[symbol] += settled
        self.t_plus_queues[symbol].appendleft(0) # T+0 starts empty for next day

    def run(self, df: pd.DataFrame, agent_policy: Any):
        """
        Executes the backtest bar by bar.
        df should be sorted by timestamp then symbol if multi-ticker.
        """
        self.reset()
        
        # Group data by timestamp to handle multi-ticker cross-section
        grouped = df.groupby('timestamp')
        
        for ts, group in grouped:
            for _, row in group.iterrows():
                symbol = row['symbol']
                price = row['close']
                volume = row['volume']
                
                # 1. Update Settlement (Only at session start/end or per-day change)
                # For simplified 1m backtest, we assume T+2 logic cycles daily
                # In more complex setup, check date change
                
                # 2. Extract Observation
                # obs = ... (This depends on env mapping)
                
                # 3. Get Action
                # action = agent_policy.predict(obs) 
                
                # Placeholder for logic in run_harsh_backtest.py wrapper
                pass

        return self.trade_log

    def execute_trade(self, 
                       ts: Any, 
                       symbol: str, 
                       action: int, 
                       price: float, 
                       volume: float,
                       metadata: Dict[str, Any] = None):
        """
        Executes an trade action.
        Action: 0=HOLD, 1=BUY, 2=SELL
        """
        if action == 0: return None
        
        if symbol not in self.available_shares:
            self.available_shares[symbol] = 0
            self.t_plus_queues[symbol] = deque([0, 0], maxlen=2)
            self.positions[symbol] = 0

        # BUY 
        if action == 1:
            # Re-calculate sizing based on Lot 100
            max_buy_val = self.cash * 0.98 # Buffer 2% for fees
            qty = (max_buy_val // (price * 100)) * 100
            if qty <= 0: return None
            
            exec_res = self.cost_model.calculate_execution('BUY', price, int(qty), volume)
            if exec_res['rejected']:
                self.trade_log.append({"ts": ts, "symbol": symbol, "status": "REJECTED", "reason": exec_res['reason']})
                return None
            
            fill_qty = exec_res['quantity_filled']
            cost = (fill_qty * exec_res['effective_price']) + exec_res['fee']
            
            if self.cash >= cost:
                self.cash -= cost
                self.t_plus_queues[symbol][0] += fill_qty
                self.positions[symbol] += fill_qty
                self.trade_log.append({
                    "ts": ts, "symbol": symbol, "side": "BUY", 
                    "price": price, "exec_price": exec_res['effective_price'],
                    "qty": fill_qty, "cost": exec_res['total_cost'], "regime": metadata.get('regime')
                })

        # SELL
        elif action == 2:
            qty = self.available_shares[symbol]
            if qty <= 100: return None
            
            exec_res = self.cost_model.calculate_execution('SELL', price, int(qty), volume)
            if exec_res['rejected']: return None
            
            fill_qty = exec_res['quantity_filled']
            proceeds = (fill_qty * exec_res['effective_price']) - exec_res['fee'] - exec_res['tax']
            
            self.cash += proceeds
            self.available_shares[symbol] -= fill_qty
            self.positions[symbol] -= fill_qty
            self.trade_log.append({
                "ts": ts, "symbol": symbol, "side": "SELL", 
                "price": price, "exec_price": exec_res['effective_price'],
                "qty": fill_qty, "cost": exec_res['total_cost'], "regime": metadata.get('regime')
            })
