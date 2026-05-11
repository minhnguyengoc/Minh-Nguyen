import logging
from typing import Dict, Any, List
import json
from datetime import datetime

class InstitutionalTrainingMonitor:
    """
    Solves "Silent Inference Degradation" and "Policy Collapse".
    Tracks reward decomposition, action distribution, and feature drift.
    """
    def __init__(self, log_path: str = "logs/training_metrics.json"):
        self.log_path = log_path
        self._metrics = []
        self.logger = logging.getLogger("TrainingMonitor")

    def log_step(self, 
                 epoch: int, 
                 reward_components: Dict[str, float], 
                 action_counts: Dict[int, int],
                 drift_stats: Dict[str, float]):
        """Serializes and logs training health."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "rewards": reward_components,
            "actions": action_counts,
            "drift": drift_stats
        }
        
        self._metrics.append(entry)
        
        # Check for Anomaly: Reward Collapse
        total_r = sum(reward_components.values())
        if epoch > 10 and abs(total_r) < 1e-6:
            self.logger.warning(f"ANOMALY DETECTED: Reward Collapse at Epoch {epoch}")
            
        # Check for Anomaly: Action Imbalance (HOLD-only policy)
        hold_rate = action_counts.get(0, 0) / (sum(action_counts.values()) + 1e-8)
        if hold_rate > 0.99:
            self.logger.warning(f"ANOMALY DETECTED: HOLD-only policy convergence at Epoch {epoch}")

    def save(self):
        try:
            with open(self.log_path, 'w') as f:
                json.dump(self._metrics, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
