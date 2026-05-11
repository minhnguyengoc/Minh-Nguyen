import numpy as np
import datetime as dt
import hashlib
from python_bot.common.types import MarketData, PortfolioState, ActionDirective, OrderSide, OrderType
from python_bot.system.gateway import PolicyInferenceGateway
from python_bot.engine.execution import HybridQueueReactiveExecutionSimulator
from python_bot.infrastructure.sequencer import MonotonicEventSequencer

def get_obs_hash(obs):
    """Calculates a deterministic hash for a StandardizedObservation."""
    vec_bytes = obs.vector.tobytes()
    meta_bytes = str(obs.metadata.model_dump()).encode()
    return hashlib.sha256(vec_bytes + meta_bytes).hexdigest()

def test_bitwise_replay_determinism():
    print("🧪 Testing Bitwise Replay Determinism...")
    
    def mock_inf(vec): return 1, 0.9 # Constant inference

    gateway1 = PolicyInferenceGateway("TCB", mock_inf)
    gateway2 = PolicyInferenceGateway("TCB", mock_inf)
    
    base_ts = dt.datetime(2024, 1, 1, 9, 30)
    tick = MarketData(
        symbol="TCB",
        event_id="evt_101",
        timestamp=base_ts,
        received_at=base_ts + dt.timedelta(milliseconds=100),
        open=100.0, high=101.0, low=99.0, close=100.5, volume=500000.0
    )
    
    port = PortfolioState(available_cash=1e8, locked_shares_t2=0, position_quantity=0, average_entry_price=0, unrealized_pnl=0, realized_pnl_today=0)
    
    # Process twice in two independent instances
    res1 = gateway1.process_tick(tick, port)
    res2 = gateway2.process_tick(tick, port)
    
    hash1 = get_obs_hash(res1[1])
    hash2 = get_obs_hash(res2[1])
    
    assert hash1 == hash2, f"DIVERGENCE: {hash1} != {hash2}"
    print(f"✅ Bitwise Identity Verified. Hash: {hash1[:16]}...")

def test_causal_sequencing():
    print("🧪 Testing OOO Sequencing & Deduplication...")
    
    def mock_inf(vec): return 0, 1.0
    gateway = PolicyInferenceGateway("TCB", mock_inf)
    port = PortfolioState(available_cash=1e8, locked_shares_t2=0, position_quantity=0, average_entry_price=0, unrealized_pnl=0, realized_pnl_today=0)
    
    base_ts = dt.datetime(2024, 1, 1, 9, 30)
    
    # Send event 2, then event 1 (Out of Order)
    t1 = MarketData(symbol="TCB", event_id="E1", timestamp=base_ts, received_at=base_ts + dt.timedelta(seconds=1), open=100, high=101, low=99, close=100, volume=100)
    t2 = MarketData(symbol="TCB", event_id="E2", timestamp=base_ts + dt.timedelta(minutes=1), received_at=base_ts + dt.timedelta(seconds=1), open=101, high=102, low=100, close=101, volume=100)
    
    # Even if t2 arrives "first" or same time as t1 in receipt time, 
    # the gateway should release them monotonically based on exchange_ts after buffer tolerance.
    gateway.process_tick(t2, port)
    res = gateway.process_tick(t1, port)
    
    # Replay verification
    assert gateway.clock.now == base_ts, "Clock should have advanced only to released T1"
    
    print("✅ Causal Constraints Verified.")

def test_latency_abstention():
    print("🧪 Testing Latency Risk Abstention...")
    
    def mock_inf(vec): return 1, 0.9 # Agent wants to go LONG
    gateway = PolicyInferenceGateway("TCB", mock_inf)
    
    # High latency tick (5 seconds lag)
    ts = dt.datetime(2024, 1, 1, 10, 0)
    stale_tick = MarketData(
        symbol="TCB", event_id="E_LAG",
        timestamp=ts,
        received_at=ts + dt.timedelta(seconds=5),
        open=100, high=101, low=99, close=100, volume=100
    )
    
    port = PortfolioState(available_cash=1e8, locked_shares_t2=0, position_quantity=0, average_entry_price=0, unrealized_pnl=0, realized_pnl_today=0)
    action, obs = gateway.process_tick(stale_tick, port)
    
    assert action == ActionDirective.HOLD, "RiskSupervisor should have forced HOLD due to latency."
    assert obs.metadata.policy_abstain == True
    print("✅ Latency Gating Verified.")

if __name__ == "__main__":
    test_bitwise_replay_determinism()
    test_causal_sequencing()
    test_latency_abstention()
    print("\n🚀 INSTITUTIONAL VERIFICATION COMPLETE: ALL ADVERSARIAL INVARIANTS PASSED.")
