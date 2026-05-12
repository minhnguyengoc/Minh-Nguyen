import gymnasium as gym
import numpy as np
import datetime as dt
from typing import Tuple, Dict, Any, List, Optional
from python_bot.common.types import (
    MarketData, PortfolioState, ActionDirective, 
    MarketRegime, SessionState, OrderRequest, OrderSide, OrderType
)
from python_bot.system.gateway import PolicyInferenceGateway
from python_bot.engine.ledger import ExposureLedger
from python_bot.engine.execution import HybridQueueReactiveExecutionSimulator
from python_bot.training.reward import InstitutionalRewardEngine
from python_bot.engine.regime import InstitutionalRegimeDetector
from python_bot.engine.ood import AdvancedOODDetector
from python_bot.training.exploit_guard import PolicyExploitGuard
from python_bot.training.distributed_guard import DistributedRolloutGuard
from python_bot.risk.liquidation import InventoryLiquidationEngine

class VNStockInstitutionalEnv(gym.Env):
    """
    Stage-4 Institutional Gymnasium Environment.
    Event-driven, causal, and microstructure-realistic.
    Enforces deterministic distributed rollouts and exploit detection.
    """
    def __init__(self, 
                 history: List[MarketData],
                 initial_capital: float = 100_000_000.0,
                 worker_id: int = 0):
        super().__init__()
        self.history = history
        self.initial_capital = initial_capital
        
        # 1. Infrastructure & Guards
        self.guard = DistributedRolloutGuard(worker_id)
        self.exploit_detector = PolicyExploitGuard()
        self.regime_detector = InstitutionalRegimeDetector()
        self.ood_detector = AdvancedOODDetector.load_mock_calibration()
        self.liquidator = InventoryLiquidationEngine()
        
        # Gateway (used for normalization & feature extraction)
        def dummy_inf(vec): return 0, 1.0
        self.gateway = PolicyInferenceGateway("TRAIN_SYMBOL", dummy_inf)
        
        # Accounting & Execution
        self.ledger = ExposureLedger(initial_capital)
        self.execution = HybridQueueReactiveExecutionSimulator()
        self.reward_engine = InstitutionalRewardEngine()
        
        # Spaces
        self.observation_space = gym.spaces.Box(low=-4.0, high=4.0, shape=(20,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        
        self.current_idx = 0

    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Enforce deterministic seeding across distributed workers
        target_seed = seed if seed is not None else self.guard.seed
        super().reset(seed=target_seed)
        
        self.current_idx = 0
        self.ledger = ExposureLedger(self.initial_capital)
        self.execution = HybridQueueReactiveExecutionSimulator()
        self.reward_engine.reset()
        self.exploit_detector = PolicyExploitGuard()
        self.regime_detector = InstitutionalRegimeDetector()
        self.liquidator = InventoryLiquidationEngine()
        
        # Initial Observation
        data = self.history[0]
        portfolio = self.ledger.get_state(data.close)
        _, obs = self.gateway.process_tick(data, portfolio)
        
        self.guard.reset_worker()
        
        return obs.vector, self._get_info(obs.metadata)

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_idx += 1
        terminated = self.current_idx >= len(self.history) - 1
        
        data = self.history[self.current_idx]
        action = ActionDirective(action_idx)
        
        # 1. PROCESS FILLS (Causal Confirmation)
        fills = self.execution.step(data)
        for fill in fills:
            self.ledger.apply_fill(fill)
            
        # 2. UPDATED STATE & REGIME
        portfolio = self.ledger.get_state(data.close)
        regime = self.regime_detector.update(data)
        
        # 3. OBSERVATION GENERATION (Normalization + OOD)
        # We need this before action dispatch to know the session state correctly
        _, obs = self.gateway.process_tick(data, portfolio)
        ood_score = self.ood_detector.calculate_score(obs.vector, regime)

        # 4. ACTION DISPATCH & AUCTION GUARD
        session = obs.metadata.session_state
        is_auction = session in [SessionState.ATO, SessionState.ATC]
        
        # Standard Action Logic (Ignored during Auctions to prevent reward hacking)
        if not is_auction:
            if action == ActionDirective.LONG:
                qty = 100
                req = OrderRequest(symbol=data.symbol, side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=qty, timestamp=data.timestamp)
                self.execution.submit(req, data)
            elif action == ActionDirective.SHORT:
                qty = 100
                req = OrderRequest(symbol=data.symbol, side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=qty, timestamp=data.timestamp)
                self.execution.submit(req, data)
            elif action == ActionDirective.CLOSE:
                if portfolio.position_quantity != 0:
                    side = OrderSide.SELL if portfolio.position_quantity > 0 else OrderSide.BUY
                    req = OrderRequest(symbol=data.symbol, side=side, order_type=OrderType.MARKET, quantity=abs(portfolio.position_quantity), timestamp=data.timestamp)
                    self.execution.submit(req, data)

        # Force Liquidation still allowed in ATC for emergency exit
        liquidation_orders = self.liquidator.get_liquidation_orders("TRAIN_SYMBOL", portfolio, data, self.current_idx)
        for req in liquidation_orders:
            self.execution.submit(req, data)
            
        # 5. REWARD & EXPLOIT DETECTION
        # Auction Neutrality: Reward is zero during auctions to prevent overfitting to session gaps
        reward = 0.0
        components = {}
        if not is_auction:
            reward, components = self.reward_engine.calculate(portfolio, data, action, fills)
            
        self.exploit_detector.record_step(action_idx, reward, components)
        
        # 6. DISTRIBUTED FINGERPRINT
        self.guard.fingerprint_interaction(data.event_id, obs.vector, action_idx, reward)

        return obs.vector, reward, terminated, False, self._get_info(obs.metadata, ood_score)

    def _get_info(self, meta: Any, ood_score: float = 0.0) -> Dict[str, Any]:
        portfolio = self.ledger.get_state(self.history[self.current_idx].close)
        info = {
            "metadata": meta.model_dump(),
            "pnl": portfolio.realized_pnl_today + portfolio.unrealized_pnl,
            "position": portfolio.position_quantity,
            "ood_score": ood_score,
            "is_exploiting": self.exploit_detector.audit().get("is_exploiting", False)
        }
        return info
