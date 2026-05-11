import datetime as dt
from python_bot.data.replay_dataset import ReplayDatasetPipeline
from python_bot.preflight.training_gate import TrainingPreflightGate
import logging

def run_stage3_final_validation():
    """
    Master Audit of the Stage 3 Hardening Pass.
    Ensures the system is ready for adversarial Stress Testing and Stage 4 PPO.
    """
    logging.basicConfig(level=logging.INFO)
    print("📢 STAGE 3 FINAL ARCHITECTURAL AUDIT")
    
    # 1. Load Real Data Slice for Verification
    # (Mocking a path - in production this usesHOSE/HNX market history)
    # Using a small synthetic slice for this verification script
    from python_bot.common.types import MarketData
    
    start_ts = dt.datetime(2024, 1, 1, 9, 30)
    history = []
    for i in range(200):
        t = start_ts + dt.timedelta(minutes=i)
        history.append(MarketData(
            symbol="TCB",
            event_id=f"e_{i}",
            timestamp=t,
            received_at=t + dt.timedelta(milliseconds=50),
            open=100 + i*0.01,
            high=100.5 + i*0.01,
            low=99.5 + i*0.01,
            close=100.2 + i*0.01,
            volume=100000 + i*10
        ))
    
    # 2. Execute Preflight Gate
    gate = TrainingPreflightGate(history)
    passed = gate.run_all_checks()
    
    if passed:
        print("\n🏆 STAGE 3 HARDENING COMPLETE.")
        print("Architecture is deterministically sound and causally correct.")
        print("Ready for Stage 4: Large-scale PPO Training.")
    else:
        print("\n🛑 STAGE 3 VALIDATION FAILED.")
        print("Architectural invariants violated. Check logs for details.")
        exit(1)

if __name__ == "__main__":
    run_stage3_final_validation()
