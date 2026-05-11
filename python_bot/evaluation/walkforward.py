import sys
import os

# Ensure the project root is in the python path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Any
from python_bot.market_env import VNStockTradingEnv
from python_bot.evaluation.policy_behavior import PolicyBehaviorAnalyzer

class WalkForwardValidator:
    """
    Executes Institutional Walk-Forward Validation.
    Prevents temporal overfitting and regime-fragility.
    """
    def __init__(self, states: np.ndarray, ohlcv: np.ndarray, timestamps: List, folds: int = 4):
        self.states = states
        self.ohlcv = ohlcv
        self.timestamps = timestamps
        self.folds = folds
        self.logger = logging.getLogger("WalkForward")

    def run_validation(self, agent) -> Dict[str, Any]:
        """Performs rolling walk-forward test (Out-of-sample)."""
        results = []
        data_len = len(self.states)
        fold_len = data_len // (self.folds + 1)
        
        self.logger.info(f"🚀 Starting {self.folds}-Fold Walk-Forward Validation...")
        
        for f in range(self.folds):
            # Anchored Walk-forward:
            # Fold 0: Test on [fold_len : 2*fold_len]
            # Fold 1: Test on [2*fold_len : 3*fold_len]
            start_idx = (f + 1) * fold_len
            end_idx = min((f + 2) * fold_len, data_len)
            
            test_states = self.states[start_idx:end_idx]
            test_ohlcv = self.ohlcv[start_idx:end_idx]
            test_ts = self.timestamps[start_idx:end_idx]
            
            env = VNStockTradingEnv(test_states, test_ohlcv, test_ts)
            analyzer = PolicyBehaviorAnalyzer(history_len=len(test_states))
            
            obs, info = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                analyzer.record_step(
                    action=action,
                    reward=reward,
                    components=info.get("reward_components", {}),
                    position=info["position"],
                    step=env.current_step
                )
                
                total_reward += reward
                done = terminated or truncated
                
            audit = analyzer.analyze()
            results.append({
                "fold": f,
                "reward": total_reward,
                "equity": info["equity"],
                "return": (info["equity"] - 100_000_000) / 100_000_000,
                "audit_status": audit["status"],
                "turnover": audit.get("trades_per_1k", 0)
            })
            self.logger.info(f"✅ Fold {f} complete | Return: {results[-1]['return']*100:.2f}% | Status: {audit['status']}")
            
        return {
            "stable": all(r['audit_status'] == "HEALTHY" for r in results),
            "fold_results": results,
            "avg_return": np.mean([r['return'] for r in results])
        }
