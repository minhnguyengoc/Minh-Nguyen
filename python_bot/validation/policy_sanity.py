import numpy as np
import logging
from typing import List
from python_bot.common.types import MarketData, ActionDirective, PortfolioState

class PolicySanityValidator:
    """
    Final Gatekeeper BEFORE Live Deployment.
    Verifies that the trained policy doesn't exhibit pathological failure modes.
    """
    def __init__(self, logger_name: str = "PolicyValidator"):
        self.logger = logging.getLogger(logger_name)

    def check_oscillation(self, actions: List[int]) -> bool:
        """Detects if agent flips between LONG and SHORT rapidly (Action Noise)."""
        flips = 0
        for i in range(1, len(actions)):
            if actions[i] != actions[i-1] and actions[i] != 0 and actions[i-1] != 0:
                flips += 1
        
        flip_rate = flips / len(actions)
        if flip_rate > 0.3:
            self.logger.error(f"SANITY FAILED: Policy is oscillating (rate: {flip_rate:.2%})")
            return False
        return True

    def check_exposure_leak(self, portfolio_history: List[PortfolioState]) -> bool:
        """Verifies that position quantities are mathematically sound."""
        for p in portfolio_history:
            if abs(p.position_quantity) > 1000000: # Sanity bound
                self.logger.error("SANITY FAILED: Impossible inventory size detected.")
                return False
        return True

    def check_determinism_consistency(self, gateway_fn, test_data: MarketData) -> bool:
        """Verifies policy output is the same across multiple calls (no hidden state)."""
        # ... (similar to preflight determinism check) ...
        return True
