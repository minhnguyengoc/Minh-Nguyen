import numpy as np
from typing import Dict, Any, Optional
from python_bot.common.types import MarketData

class GeneralizationEngine:
    """
    Enforces Domain Randomization and Adversarial Observation Perturbation.
    Prevents RL agent from "memorizing" historical microstructure noise.
    """
    def __init__(self, noise_std: float = 0.02, latency_jitter_ms: int = 100):
        self.noise_std = noise_std
        self.latency_jitter_ms = latency_jitter_ms

    def randomize_observation(self, vector: np.ndarray) -> np.ndarray:
        """Injects structured observation noise."""
        noise = np.random.normal(0, self.noise_std, vector.shape)
        return vector + noise

    def randomize_microstructure(self, data: MarketData) -> MarketData:
        """Perturbs liquidity and timing parameters to force invariant learning."""
        d = data.model_dump()
        
        # 1. Jitter Volume (Liquidity Noise)
        d['volume'] *= np.random.uniform(0.9, 1.1)
        
        # 2. Jitter Latency (Timing Noise)
        jitter = np.random.randint(-self.latency_jitter_ms, self.latency_jitter_ms)
        d['received_at'] = d['received_at'] + np.timedelta64(jitter, 'ms')
        
        # 3. Book Depth Jitter
        if d.get('bid_depth'):
            d['bid_depth'] = [(p, q * np.random.uniform(0.8, 1.2)) for p, q in d['bid_depth']]
        if d.get('ask_depth'):
            d['ask_depth'] = [(p, q * np.random.uniform(0.8, 1.2)) for p, q in d['ask_depth']]
            
        return MarketData(**d)

class AdversarialCurriculum:
    """
    Manages the intensity of adversarial scenarios during training.
    Stages: EASY -> NORMAL -> HARD -> CHAOS
    """
    def __init__(self):
        self.level = 0
        
    def get_params(self) -> Dict[str, float]:
        configs = [
            {"slippage_mult": 1.0, "latency_ms": 50, "noise": 0.0},
            {"slippage_mult": 1.5, "latency_ms": 200, "noise": 0.01},
            {"slippage_mult": 3.0, "latency_ms": 500, "noise": 0.05},
            {"slippage_mult": 5.0, "latency_ms": 2000, "noise": 0.1} # CHAOS
        ]
        return configs[min(self.level, len(configs)-1)]

    def upgrade(self):
        self.level += 1
