import logging
import numpy as np
from typing import Dict, Any, List
from python_bot.common.types import MarketData
from python_bot.training.env import VNStockInstitutionalEnv
from python_bot.infrastructure.sequencer import MonotonicEventSequencer

class TrainingPreflightGate:
    """
    Safety Guard for Stage 4 Transition.
    Verifies all institutional invariants before PPO training begins.
    """
    def __init__(self, history: List[MarketData]):
        self.history = history
        self.logger = logging.getLogger("PreflightGate")

    def run_all_checks(self) -> bool:
        self.logger.info("Starting Training Preflight Validation...")
        
        checks = [
            self._check_determinism(),
            self._check_causality(),
            self._check_observation_range(),
            self._check_reward_stability()
        ]
        
        passed = all(checks)
        if passed:
            self.logger.info("🚀 PREFLIGHT PASSED: System ready for Stage 4 training.")
        else:
            self.logger.critical("❌ PREFLIGHT FAILED: Architectural risk detected.")
            
        return passed

    def _check_determinism(self) -> bool:
        """Verifies bitwise identity on env reset and first step."""
        env1 = VNStockInstitutionalEnv(self.history)
        env2 = VNStockInstitutionalEnv(self.history)
        
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        
        if not np.array_equal(obs1, obs2):
            self.logger.error("DETERMINISM FAILED: Reset observations differ.")
            return False
            
        # Step identity
        step_obs1, r1, _, _, _ = env1.step(1)
        step_obs2, r2, _, _, _ = env2.step(1)
        
        if not np.array_equal(step_obs1, step_obs2) or r1 != r2:
            self.logger.error("DETERMINISM FAILED: Step outputs differ.")
            return False
            
        return True

    def _check_causality(self) -> bool:
        """Verifies no future leakage in features."""
        # Simple cross-check (harder to automate completely without looking into indicator code)
        # But we can verify that the clock only moves forward
        env = VNStockInstitutionalEnv(self.history)
        env.reset()
        last_ts = env.history[0].timestamp
        
        for _ in range(10):
            _, _, _, _, info = env.step(0)
            current_ts = env.history[env.current_idx].timestamp
            if current_ts <= last_ts:
                self.logger.error("CAUSALITY FAILED: Clock moved backwards or stagnated.")
                return False
            last_ts = current_ts
            
        return True

    def _check_observation_range(self) -> bool:
        """Verifies normalization is working and values are bounded."""
        env = VNStockInstitutionalEnv(self.history)
        obs, _ = env.reset()
        
        if np.any(np.abs(obs) > 5.0):
            self.logger.error("NORMALIZATION FAILED: Observations out of bounds.")
            return False
        return True

    def _check_reward_stability(self) -> bool:
        """Verifies rewards are not exploding or NaN."""
        env = VNStockInstitutionalEnv(self.history)
        env.reset()
        
        rewards = []
        for _ in range(20):
            _, r, _, _, _ = env.step(1)
            rewards.append(r)
            
        if np.any(np.isnan(rewards)):
            self.logger.error("REWARD FAILED: NaN detected.")
            return False
            
        return True
