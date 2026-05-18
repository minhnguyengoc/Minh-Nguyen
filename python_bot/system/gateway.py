import numpy as np
from typing import Tuple, Dict, Any, Optional
from python_bot.common.types import (
    MarketData, PortfolioState, ActionDirective, 
    StandardizedObservation, ObservationMetadata, MarketRegime
)
from python_bot.infrastructure.sequencer import TimestampOrderingBuffer, MonotonicEventSequencer
from python_bot.engine.normalization import FrozenQuantileNormalizer
from python_bot.engine.regime import InstitutionalRegimeDetector
from python_bot.risk.latency import LatencyRiskEngine
from python_bot.infrastructure.clock import ClockTriangulator
from python_bot.infrastructure.session import SessionBoundaryFSM
from python_bot.engine.features import FeatureEngine
from python_bot.engine.execution import DriftMonitor
from python_bot.risk.supervisor import RiskSupervisor

class PolicyInferenceGateway:
    """
    Unified Inference Gateway v2 (Institutional Grade).
    Enforces absolute sequence integrity, latency-aware risk, and bitwise-identical normalizations.
    """
    def __init__(self, ticker: str, model_inference_fn):
        self.ticker = ticker
        self.model_inf = model_inference_fn
        
        # Infrastructure
        self.clock = ClockTriangulator()
        self.sequencer = MonotonicEventSequencer()
        self.ooo_buffer = TimestampOrderingBuffer(tolerance_ms=500)
        self.session_fsm = SessionBoundaryFSM()
        self.latency_monitor = LatencyRiskEngine()
        
        # Engines
        self.features = FeatureEngine()
        self.regime_detector = InstitutionalRegimeDetector()
        self.normalizer = FrozenQuantileNormalizer.load_defaults()
        self.monitor = DriftMonitor(baseline_dist=np.zeros((1, 20)))
        self.risk = RiskSupervisor()
        
        self._last_ts = None

    def process_tick(self, data: MarketData, portfolio: PortfolioState) -> Tuple[ActionDirective, StandardizedObservation]:
        # Out-of-Order Handling & Sequencing
        self.latency_monitor.record(data.timestamp, data.received_at)
        
        seq_id = self.sequencer.process(data)
        if seq_id is None:
            # Event sequence failed or duplicate
            return self._get_safe_fallback(data, portfolio)
            
        self.ooo_buffer.push(data)
        ticks = self.ooo_buffer.release(data.received_at)
        
        if not ticks:
            # No ticks released from buffer yet
            return self._get_safe_fallback(data, portfolio)
            
        last_result = None
        for tick in ticks:
            last_result = self._handle_single_tick(tick, portfolio, seq_id)
            
        return last_result

    def _get_safe_fallback(self, data: MarketData, portfolio: PortfolioState) -> Tuple[ActionDirective, StandardizedObservation]:
        """Returns a safe HOLD action and the last known or zero-filled observation."""
        # Use existing features if possible, or zero if first tick
        raw_vec = self.features.generate() if hasattr(self.features, 'generate') else np.zeros(20)
        regime = MarketRegime.STABLE
        norm_vec = self.normalizer.normalize(raw_vec, regime)
        
        meta = ObservationMetadata(
            is_session_active=True,
            session_state=self.session_fsm.current_state,
            regime=regime,
            is_stale=True,
            kill_switch=False,
            drift_score=0.0,
            confidence_score=0.0,
            policy_abstain=True,
            latency_ms=0.0,
            event_sequence_id=-1
        )
        return ActionDirective.HOLD, StandardizedObservation(vector=norm_vec, metadata=meta)

    def _handle_single_tick(self, data: MarketData, portfolio: PortfolioState, seq_id: int) -> Tuple[ActionDirective, StandardizedObservation]:
        # Update Clock & Session
        self.clock.sync(data.timestamp)
        session = self.session_fsm.update(data.timestamp)
        
        # Deterministic State Reset
        if self.session_fsm.is_interrupted(data.timestamp, self._last_ts):
            self.features.reset()
            
        self._last_ts = data.timestamp
        
        # Feature Generation & Normalization
        self.features.push(data)
        raw_vec = self.features.generate()
        
        # Institutional Regime Sync
        regime = self.regime_detector.update(data)
        
        norm_vec = self.normalizer.normalize(raw_vec, regime)
        
        # Monitoring
        self.monitor.push(raw_vec)
        drift_score = self.monitor.calculate_psi()
        
        # Inference
        action_idx, confidence = self.model_inf(norm_vec)
        agent_action = ActionDirective(action_idx)
        
        # Metadata Assembly with Latency and Seq Context
        latency_ms = (data.received_at - data.timestamp).total_seconds() * 1000
        meta = ObservationMetadata(
            is_session_active=session != session.CLOSED,
            session_state=session,
            regime=regime,
            is_stale=latency_ms > 2000,
            kill_switch=False,
            drift_score=drift_score,
            confidence_score=confidence,
            policy_abstain=(confidence < 0.6) or (latency_ms > 1000),
            latency_ms=latency_ms,
            event_sequence_id=seq_id
        )
        
        obs = StandardizedObservation(vector=norm_vec, metadata=meta)
        
        # Risk Gate
        final_action = self.risk.validate_action(agent_action, portfolio, meta)
        
        return final_action, obs
