from datetime import datetime
from typing import Dict, List, Optional
import logging

class LatencyRiskEngine:
    """
    Solves "Stale-State Trading".
    Monitors delta between exchange timestamp and receipt timestamp.
    Forces conservatism if system is blind.
    """
    def __init__(self, critical_threshold_ms: int = 2000, warning_threshold_ms: int = 500):
        self.critical_ms = critical_threshold_ms
        self.warning_ms = warning_threshold_ms
        self._latency_history = []
        self._max_history = 100

    def record(self, exchange_ts: datetime, received_at: datetime):
        latency = (received_at - exchange_ts).total_seconds() * 1000
        self._latency_history.append(latency)
        if len(self._latency_history) > self._max_history:
            self._latency_history.pop(0)

    def get_risk_multiplier(self) -> float:
        """
        Returns a multiplier [0.0, 1.0] to scale sizing based on staleness.
        """
        if not self._latency_history: return 1.0
        
        avg_latency = sum(self._latency_history) / len(self._latency_history)
        
        if avg_latency > self.critical_ms:
            logging.critical(f"LATENCY HALT: Avg latency {avg_latency:.0f}ms exceeds {self.critical_ms}ms")
            return 0.0 # Force Hold/Close
        
        if avg_latency > self.warning_ms:
            # Linear decay from warnings -> critical
            decay = (self.critical_ms - avg_latency) / (self.critical_ms - self.warning_ms)
            return max(0.1, decay)
            
        return 1.0
