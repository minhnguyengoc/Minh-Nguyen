import logging
from typing import Dict, Optional, Tuple
from python_bot.common.types import MarketData, PortfolioState, ActionDirective, StandardizedObservation
from python_bot.system.gateway import PolicyInferenceGateway
from python_bot.engine.ledger import ExposureLedger

class InstitutionalPaperEngine:
    """
    Orchestrates "Causal Paper Trading".
    Parallel behavior to Live but with strict deterministic ingestion.
    """
    def __init__(self, ticker: str, inference_fn):
        self.gateway = PolicyInferenceGateway(ticker, inference_fn)
        self.ledger = ExposureLedger(initial_cash=1e8)
        self._journal = []

    def on_tick(self, tick: MarketData):
        """Main entry point for incoming market packets."""
        portfolio = self.ledger.get_state(tick.close)
        
        # 1. Deterministic Processing
        result = self.gateway.process_tick(tick, portfolio)
        if not result: return # Packet dropped by sequencer/deduper
        
        action, obs = result
        
        # 2. Journaling for Audit
        self._record_interaction(tick, obs, action)
        
        # 3. Decision Dispatch
        # In paper trading, we feed the directive to the Simulator
        # In live, we'd feed this to the Broker Gateway
        return action

    def _record_interaction(self, tick: MarketData, obs: StandardizedObservation, action: ActionDirective):
        self._journal.append({
            'ts': tick.timestamp,
            'event_id': tick.event_id,
            'action': action,
            'obs_hash': obs.metadata.event_sequence_id,
            'latency': obs.metadata.latency_ms
        })

    def get_audit_trail(self):
        return self._journal
