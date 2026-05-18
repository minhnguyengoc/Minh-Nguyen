import numpy as np
from typing import List, Dict, Any
import logging

class PolicyTradeDiagnostics:
    """
    Detects behavioral failure modes (HOLD-bias, overtrading, entropy collapse).
    """
    def __init__(self):
        self.actions = []
        self.logger = logging.getLogger("PolicyDiagnostics")

    def record_action(self, action: int):
        self.actions.append(action)

    def run_diagnostics(self) -> Dict[str, Any]:
        """Audits policy behavior."""
        if not self.actions:
            return {"verdict": "FAIL", "reason": "NO_ACTIONS_RECORDED"}
            
        n = len(self.actions)
        counts = {0: 0, 1: 0, 2: 0}
        for a in self.actions: counts[a] += 1
        
        hold_ratio = counts[0] / n
        buy_ratio = counts[1] / n
        sell_ratio = counts[2] / n
        
        verdict = "PASS"
        reasons = []
        
        if hold_ratio > 0.99:
            verdict = "FAIL"
            reasons.append("POLICY_LEARNED_ALWAYS_HOLD")
        
        if (buy_ratio + sell_ratio) > 0.8:
            verdict = "WARNING"
            reasons.append("OVERTRADING_RISK")
            
        # Entropy check
        probs = np.array([hold_ratio, buy_ratio, sell_ratio])
        entropy = -np.sum(probs * np.log2(probs + 1e-8))
        
        if entropy < 0.1:
            verdict = "FAIL"
            reasons.append("DETERMINISTIC_COLLAPSE")
            
        return {
            "verdict": verdict,
            "reasons": reasons,
            "hold_ratio": hold_ratio,
            "entropy": entropy,
            "action_counts": counts
        }
