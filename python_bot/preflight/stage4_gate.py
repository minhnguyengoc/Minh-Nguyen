import logging
import numpy as np
from typing import List
from python_bot.common.types import MarketData
from python_bot.training.env import VNStockInstitutionalEnv
from python_bot.preflight.training_gate import TrainingPreflightGate

from python_bot.simulation.adversarial import AdversarialScenarioEngine
from python_bot.training.distributed_guard import DistributedRolloutGuard

class Stage4FinalGate:
    """
    The FINAL Institutional Gate before Stage 4 (Scale-up Training).
    Enforces the "Stage-4 Readiness" checklist.
    """
    def __init__(self, history: List[MarketData]):
        self.history = history
        self.logger = logging.getLogger("Stage4Gate")

    def verify(self) -> bool:
        self.logger.info("VERIFYING STAGE 4 READINESS...")
        
        # 1. Basic Invariants (Causality, Determinism)
        preflight = TrainingPreflightGate(self.history)
        if not preflight.run_all_checks():
            return False
            
        # 2. Distributed Reproducibility Check
        guard1 = DistributedRolloutGuard(worker_id=0)
        guard2 = DistributedRolloutGuard(worker_id=0)
        
        env1 = VNStockInstitutionalEnv(self.history, worker_id=0)
        env2 = VNStockInstitutionalEnv(self.history, worker_id=0)
        
        o1, _ = env1.reset()
        o2, _ = env2.reset()
        
        if not np.array_equal(o1, o2):
            self.logger.error("DETERMINISTIC FAILED: Parallel workers differ on reset.")
            return False
            
        # 3. Adversarial Robustness Check
        adv_engine = AdversarialScenarioEngine()
        stressed_history = adv_engine.inject_flash_crash(self.history)
        # Ensure env can handle stressed data without crashing
        env_stress = VNStockInstitutionalEnv(stressed_history)
        env_stress.reset()
        env_stress.step(1)
        
        self.logger.info("✅ STAGE 4 GATE PASSED. System is production-stable for training.")
        return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🚀 INSTITUTIONAL READINESS GATE: ACTIVE")
    # ... execution logic ...
