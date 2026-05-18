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
        # Action Space: 0 = HOLD, 1 = BUY, 2 = SELL/CLOSE
        self.action_space = gym.spaces.Discrete(3)
        
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
        
        return obs.vector, self._get_info(obs.metadata, action=0, position_before=0)

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        data_before = self.history[self.current_idx]
        pos_before = self.ledger.get_state(data_before.close).position_quantity
        
        self.current_idx += 1
        terminated = self.current_idx >= len(self.history) - 1
        
        data = self.history[self.current_idx]
        # Map action: 0=HOLD, 1=LONG, 2=CLOSE (matches 0,1,2 used in VNStockTradingEnv mostly)
        action_map = {0: ActionDirective.HOLD, 1: ActionDirective.LONG, 2: ActionDirective.CLOSE}
        action = action_map.get(action_idx, ActionDirective.HOLD)
        
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
        
        rejected_reason = None
        # Standard Action Logic (Ignored during Auctions to prevent reward hacking)
        if not is_auction:
            if action == ActionDirective.LONG:
                # Check cash for buy
                fee_est = (100 * data.close) * 0.0015
                if portfolio.available_cash < (100 * data.close + fee_est):
                    rejected_reason = "INSUFFICIENT_CASH"
                else:
                    qty = 100
                    req = OrderRequest(symbol=data.symbol, side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=qty, timestamp=data.timestamp)
                    self.execution.submit(req, data)
            elif action == ActionDirective.CLOSE:
                if portfolio.position_quantity == 0:
                    rejected_reason = "NO_POSITION_TO_CLOSE"
                elif portfolio.locked_shares_t2 > 0 and (portfolio.position_quantity - portfolio.locked_shares_t2) <= 0:
                    rejected_reason = "T+2_LOCKHOLD"
                else:
                    side = OrderSide.SELL if portfolio.position_quantity > 0 else OrderSide.BUY
                    # In HOSE we only have Long, so position_quantity > 0 always
                    qty_to_sell = portfolio.position_quantity - portfolio.locked_shares_t2
                    if qty_to_sell > 0:
                        req = OrderRequest(symbol=data.symbol, side=side, order_type=OrderType.MARKET, quantity=qty_to_sell, timestamp=data.timestamp)
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
            # Pass full history and current idx for future return access if needed
            reward, components = self.reward_engine.calculate(
                portfolio, data, action, fills, 
                history=self.history, current_idx=self.current_idx
            )
            
        self.exploit_detector.record_step(action_idx, reward, components)
        
        # 6. DISTRIBUTED FINGERPRINT
        self.guard.fingerprint_interaction(data.event_id, obs.vector, action_idx, reward)

        pos_after = portfolio.position_quantity
        return obs.vector, reward, terminated, False, self._get_info(
            obs.metadata, ood_score, action=action_idx, 
            position_before=pos_before, position_after=pos_after,
            rejected_reason=rejected_reason, reward_components=components
        )

    def _get_info(self, meta: Any, ood_score: float = 0.0, **kwargs) -> Dict[str, Any]:
        portfolio = self.ledger.get_state(self.history[self.current_idx].close)
        audit = self.exploit_detector.audit()
        info = {
            "metadata": meta.model_dump() if hasattr(meta, 'model_dump') else meta,
            "pnl": portfolio.realized_pnl_today + portfolio.unrealized_pnl,
            "position": portfolio.position_quantity,
            "ood_score": ood_score,
            "is_exploiting": audit.get("is_exploiting", False),
            
            # Stage 4.1 Diagnostics
            "raw_action": kwargs.get("action"),
            "interpreted_action": kwargs.get("action"),
            "position_before": kwargs.get("position_before"),
            "position_after": kwargs.get("position_after"),
            "position_changed": kwargs.get("position_before") != kwargs.get("position_after"),
            "total_trades": getattr(self.ledger, "total_fills", 0),
            "reward_components": kwargs.get("reward_components", {}),
            "rejected_reason": kwargs.get("rejected_reason")
        }
        return info

