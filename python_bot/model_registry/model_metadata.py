from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class ModelMetadata:
    """Represents model parameters and feature space specification."""
    checkpoint_name: str
    algorithm: str = "PPO"
    observation_dim: int = 0
    action_dim: int = 0
    feature_columns: Optional[List[str]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
