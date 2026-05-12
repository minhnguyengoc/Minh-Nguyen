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
        """
        Process one market tick and ALWAYS return:
        (ActionDirective, StandardizedObservation).

        Runtime contract:
        - Never return None.
        - Duplicate ticks return HOLD + safe observation.
        - Out-of-order buffer waiting returns HOLD + safe observation.
        - Empty release returns HOLD + safe observation.
        """
        seq_id = self.sequencer.process(data)
        if seq_id is None:
            return self._safe_hold_observation(data, reason="DEDUPED_OR_REJECTED")

        self.latency_monitor.record(data.timestamp, data.received_at)

        self.ooo_buffer.push(data)
        ticks = self.ooo_buffer.release(data.received_at)

        if not ticks:
            return self._safe_hold_observation(data, reason="OOO_BUFFER_WAITING")

        last_result = None
        for tick in ticks:
            last_result = self._handle_single_tick(tick, portfolio, seq_id)

        if last_result is None:
            return self._safe_hold_observation(data, reason="NO_TICK_RELEASED")

        return last_result


    def _safe_hold_observation(self, data: MarketData, reason: str = "SAFE_HOLD") -> Tuple[ActionDirective, StandardizedObservation]:
        """
        Deterministic safe HOLD fallback used during warmup, dedupe,
        out-of-order buffering, stale ticks, and invalid events.
        """
        import numpy as np

        n_features = getattr(self.features, "n_features", None)
        if n_features is None:
            n_features = getattr(self.features, "feature_dim", 30)

        vec = np.zeros(int(n_features), dtype=np.float32)

        try:
            session = self.session_fsm.update(data.timestamp)
        except Exception:
            session = None

        try:
            regime = self.regime_detector.update(data)
        except Exception:
            regime = None

        try:
            latency_ms = (data.received_at - data.timestamp).total_seconds() * 1000
        except Exception:
            latency_ms = 0.0

        meta = ObservationMetadata(
            is_session_active=False,
            session_state=session,
            regime=regime,
            is_stale=True,
            kill_switch=True,
            drift_score=0.0,
            confidence_score=0.0,
            policy_abstain=True,
            latency_ms=latency_ms,
            event_sequence_id=-1
        )

        obs = StandardizedObservation(vector=vec, metadata=meta)
        return ActionDirective.HOLD, obs

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
