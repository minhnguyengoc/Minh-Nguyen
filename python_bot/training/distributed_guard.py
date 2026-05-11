import hashlib
import numpy as np
from typing import Dict, List, Any
import logging

class DistributedRolloutGuard:
    """
    Ensures bitwise identity across distributed PPO rollout workers.
    Solves "Hidden Non-Determinism" and "Replay Mismatch".
    """
    def __init__(self, worker_id: int, seed_base: int = 42):
        self.worker_id = worker_id
        self.seed = seed_base + worker_id
        self._interaction_log: List[str] = []
        
        # Deterministic Generator
        self.rng = np.random.default_rng(self.seed)

    def fingerprint_interaction(self, 
                                event_id: str,
                                obs: np.ndarray, 
                                action: int, 
                                reward: float) -> str:
        """Creates a bitwise checksum of a single interaction step, keyed by exchange event_id."""
        state_hash = hashlib.md5(obs.tobytes()).hexdigest()
        payload = f"{event_id}_{state_hash}_{action}_{reward:.8f}"
        
        interaction_hash = hashlib.sha256(payload.encode()).hexdigest()
        self._interaction_log.append(interaction_hash)
        return interaction_hash

    def get_rollout_checksum(self) -> str:
        """Calculates the cumulative hash of the entire rollout interaction log."""
        full_buffer = "".join(self._interaction_log).encode()
        return hashlib.sha256(full_buffer).hexdigest()

    def reset_worker(self):
        self._interaction_log.clear()
