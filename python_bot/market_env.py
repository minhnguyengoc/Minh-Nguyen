import gymnasium as gym
import numpy as np
import logging
import datetime as dt
from typing import Tuple, Dict, Any, List, Optional
from python_bot.common.types import MarketData, FillEvent, OrderSide, OrderRequest, OrderType
from python_bot.engine.ledger import ExposureLedger
try:
    from python_bot.engine.execution import QueueFillSimulator
except ImportError:
    try:
        # Fallback for different directory structures in Colab
        from engine.execution import HybridQueueReactiveExecutionSimulator as QueueFillSimulator
    except ImportError:
        logging.error("CRITICAL: Could not import QueueFillSimulator. Ensure PYTHONPATH includes the repository root.")
        raise

from python_bot.reward_engine import RewardEngine

class VNStockTradingEnv(gym.Env):
    """
    Refactored Institutional-Grade Gym Environment.
    Uses strict causal ledger accounting and realistic queue-based execution.
    """
    def __init__(self, 
                 states: np.ndarray, 
                 ohlcv: np.ndarray,      # [N, 5] Matrix: O, H, L, C, V
                 timestamps: List[dt.datetime],
                 initial_capital: float = 100_000_000.0):
        super().__init__()
        self.states = states
        self.ohlcv = ohlcv
        self.timestamps = timestamps
        self.initial_capital = initial_capital
        
        # Spaces
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(states.shape[1],), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4) # 0:Hold, 1:Long, 2:Short, 3:Close
        
        # Sub-Engines
        self.ledger = ExposureLedger(initial_capital)
        self.sim = QueueFillSimulator()
        self.reward_engine = RewardEngine()
        
        self.current_step = 0
        self.trade_count = 0
        self.max_equity = initial_capital

    @property
    def equity(self) -> float:
        """Returns current liquid equity."""
        # Using the last known close price for valuation
        price = self.ohlcv[self.current_step, 3] if self.current_step < len(self.ohlcv) else self.ohlcv[-1, 3]
        state = self.ledger.get_state(price)
        # Equity = available cash + current value of open positions
        return state.available_cash + (state.position_quantity * price)

    @property
    def position(self) -> float:
        """Returns current position quantity."""
        price = self.ohlcv[self.current_step, 3] if self.current_step < len(self.ohlcv) else self.ohlcv[-1, 3]
        return self.ledger.get_state(price).position_quantity

    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.trade_count = 0
        self.ledger = ExposureLedger(self.initial_capital)
        self.sim = QueueFillSimulator()
        self.reward_engine = RewardEngine() # Reset reward state
        self.max_equity = self.initial_capital
        
        return self.states[0], self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        old_equity = self.equity
        self.current_step += 1
        truncated = self.current_step >= len(self.states) - 1
        
        # 1. Fetch current tick data
        row = self.ohlcv[self.current_step]
        ts = self.timestamps[self.current_step]
        data = MarketData(
            symbol="TICKER",
            timestamp=ts,
            received_at=ts,  # Simulation: assume zero latency
            event_id=f"sim_evt_{self.current_step}",
            open=row[0], high=row[1], low=row[2], close=row[3], volume=row[4]
        )
        
        # 2. Process existing queue fills (Causality: Fills happen BEFORE new orders)
        fills = self.sim.step(data)
        for fill in fills:
            self.ledger.apply_fill(fill)
            
        # 3. Handle Agent Action -> Submitting Orders
        if action == 1: # LONG
            req = OrderRequest(symbol="TICKER", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100, timestamp=ts)
            if self.sim.submit(req, data): self.trade_count += 1
        elif action == 2: # SHORT
            req = OrderRequest(symbol="TICKER", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=100, timestamp=ts)
            if self.sim.submit(req, data): self.trade_count += 1
        elif action == 3: # CLOSE
            pos = self.ledger.get_state(data.close).position_quantity
            if pos != 0:
                side = OrderSide.SELL if pos > 0 else OrderSide.BUY
                req = OrderRequest(symbol="TICKER", side=side, order_type=OrderType.MARKET, quantity=abs(pos), timestamp=ts)
                if self.sim.submit(req, data): self.trade_count += 1

        # 4. Calculation of stats for RewardEngine
        current_equity = self.equity
        self.max_equity = max(self.max_equity, current_equity)
        current_drawdown = (self.max_equity - current_equity) / max(self.max_equity, 1e-9)
        
        # Check for bankruptcy / stop-out
        terminated = current_equity < self.initial_capital * 0.1 # 90% loss
        
        # 5. Advanced Reward Engine Call
        reward, components = self.reward_engine.compute_reward(
            equity=current_equity,
            initial_balance=self.initial_capital,
            drawdown=current_drawdown,
            trade_count=self.trade_count,
            action=action,
            current_position=self.position,
            terminated=terminated or truncated
        )
        
        info = self._get_info()
        info["reward_components"] = components
        
        return self.states[self.current_step], float(reward), terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        p = self.ledger.get_state(self.ohlcv[self.current_step, 3] if self.current_step < len(self.ohlcv) else self.ohlcv[-1, 3])
        return {
            "equity": self.equity,
            "position": self.position,
            "cash": p.available_cash,
            "trade_count": self.trade_count
        }
