import sys
import os

# Ensure the project root is in the python path
def _setup_path():
    current_file = os.path.abspath(__file__)
    # Go up from /python_bot/evaluation/policy_behavior.py
    # 1: /python_bot/evaluation/
    # 2: /python_bot/
    # 3: / (project root containing python_bot)
    root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root

_project_root = _setup_path()

import numpy as np
from typing import List, Dict, Any, Optional
import logging

class PolicyBehaviorAnalyzer:
    """
    Adversarial Analytics for RL Policies.
    Detects pathological patterns: overtrading, reward farming, and simulator exploitation.
    """
    def __init__(self, history_len: int = 1000):
        self.history_len = history_len
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._reward_components: List[Dict[str, float]] = []
        self._holding_times: List[int] = []
        
        self._current_pos_start = -1
        self.logger = logging.getLogger("BehaviorAnalyzer")

    def record_step(self, action: int, reward: float, components: Any, position: float, step: int):
        """Records a single step of interaction."""
        self._actions.append(int(action))
        self._rewards.append(float(reward))
        self._last_pos = getattr(self, "_last_pos", 0.0)
        
        # Ensure components is a dict
        if isinstance(components, dict):
            self._reward_components.append(components)
        else:
            self._reward_components.append({"reward": float(reward)})
        
        # Track holding times accurately (position can be shares float)
        if (position > 0) != (self._last_pos > 0):
            if self._last_pos > 0:
                # Close previous position timing
                if self._current_pos_start >= 0:
                    self._holding_times.append(max(0, step - self._current_pos_start))
                    self._current_pos_start = -1
            
            if position > 0:
                # Start new position timing
                self._current_pos_start = step
        
        self._last_pos = position
        
        # Maintain window
        if len(self._actions) > self.history_len:
            self._actions.pop(0)
            self._rewards.pop(0)
            self._reward_components.pop(0)

    def analyze(self) -> Dict[str, Any]:
        """Performs a forensic audit of the policy behavior."""
        if not self._actions:
            return {"status": "NO_DATA"}
        
        try:
            # 1. Action Entropy
            unique, counts = np.unique(self._actions, return_counts=True)
            probs = counts / len(self._actions)
            entropy = -np.sum(probs * np.log2(probs + 1e-8))
            
            # 2. Turnover (Avg trades per 1000 steps)
            # Action 1: Buy, 2: Sell
            trades = sum(1 for a in self._actions if a in [1, 2])
            turnover = trades / (max(len(self._actions), 1) / 1000)
            
            # 3. Holding Time Stats
            avg_hold = np.mean(self._holding_times) if self._holding_times else 0
            
            # 4. Reward Component Dominance
            comp_sums = {}
            for c in self._reward_components:
                for k, v in c.items():
                    comp_sums[k] = comp_sums.get(k, 0) + abs(v)
            
            total_abs_r = sum(comp_sums.values())
            dominance = {k: v / (total_abs_r + 1e-8) for k, v in comp_sums.items()}
            
            flags = []
            if turnover > 500: flags.append("HYPER_TURNOVER")
            if turnover < 1.0: flags.append("ZERO_TURNOVER")
            if 0 < avg_hold < 5: flags.append("MICRO_SCALPING_EXPLOIT")
            
            # Check for Reward Hacking: High reward, low PnL/Edge ratio
            # If 'pnl_realized' is low relative to 'directional_edge' but total reward is high
            edge_contribution = dominance.get('directional_edge', 0)
            pnl_contribution = dominance.get('pnl_realized', 0)
            if edge_contribution > 0.8 and pnl_contribution < 0.1:
                flags.append("POTENTIAL_REWARD_HACKING")

            status = "HEALTHY"
            if "HYPER_TURNOVER" in flags or "ZERO_TURNOVER" in flags:
                status = "DANGEROUS"
            elif flags:
                status = "WARNING"
            
            return {
                "entropy": float(entropy),
                "trades_per_1k": float(turnover),
                "avg_holding_bars": float(avg_hold),
                "reward_dominance": dominance,
                "anomaly_flags": flags,
                "status": status
            }
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {"status": "ERROR", "message": str(e)}

if __name__ == "__main__":
    # Diagnostic test
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("BehaviorTest")
    
    analyzer = PolicyBehaviorAnalyzer()
    logger.info("🧪 Running Behavioral Diagnostic...")
    
    # Simulate some "healthy" trading
    for i in range(100):
        # 0: Hold, 1: Long, 2: Short, 3: Close
        action = np.random.choice([0, 1, 2, 3], p=[0.7, 0.1, 0.1, 0.1])
        reward = np.random.normal(0.01, 0.05)
        # Mock components - more balanced to avoid monopoly flag (threshold 0.9)
        components = {
            "pnl": reward * 0.6, 
            "risk": -abs(reward)*0.3, 
            "fee": -0.005,
            "consistency": 0.001
        }
        pos = 1 if i % 10 < 5 else 0
        analyzer.record_step(action, reward, components, pos, i)
        
    results = analyzer.analyze()
    print("\n" + "="*40)
    print("📊 BEHAVIORAL ANALYSIS RESULTS")
    print(f"Status: {results.get('status')}")
    print(f"Entropy: {results.get('entropy'):.4f}")
    print(f"Turnover: {results.get('trades_per_1k'):.1f} per 1k bars")
    print(f"Avg Hold: {results.get('avg_holding_bars'):.1f} bars")
    print(f"Flags: {results.get('anomaly_flags')}")
    print("="*40)
