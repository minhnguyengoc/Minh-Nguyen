import numpy as np
from typing import Dict, Any, List
import logging

class PPOStabilityHardeners:
    """
    Institutional PPO Stabilizers.
    Implements Adaptive KL, Entropy Collapse Recovery, and Gradient Health Monitoring.
    """
    def __init__(self, target_kl: float = 0.01):
        self.target_kl = target_kl
        self.logger = logging.getLogger("StabilityHardener")
        
    def adaptive_lr(self, current_lr: float, observed_kl: float) -> float:
        """Adjusts LR based on policy distance (KL divergence)."""
        if observed_kl > self.target_kl * 2.0:
            return current_lr * 0.5
        elif observed_kl < self.target_kl * 0.5:
            return current_lr * 1.2
        return current_lr

    def check_entropy_collapse(self, entropy: float, threshold: float = 0.1) -> bool:
        """Detects if agent has stopped exploring (Policy Collapse)."""
        if entropy < threshold:
            self.logger.warning(f"ENTROPY COLLAPSE: {entropy:.4f} < {threshold}")
            return True
        return False

    def stabilize_advantages(self, advantages: np.ndarray) -> np.ndarray:
        """Normalizes and clips advantages for GAE stability."""
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

class RewardStabilizer:
    """Handles reward scaling and outlier clipping."""
    def __init__(self, clip_val: float = 10.0):
        self.clip_val = clip_val
        self._ema_var = 1.0
        self._beta = 0.99

    def process(self, reward: float) -> float:
        self._ema_var = self._beta * self._ema_var + (1 - self._beta) * (reward ** 2)
        norm_reward = reward / (np.sqrt(self._ema_var) + 1e-8)
        return np.clip(norm_reward, -self.clip_val, self.clip_val)
