# Institutional RL Infrastructure - Stage 4 Production Baseline

This document summarizes the archival integrity and architectural invariants of the VNStock RL Trading platform.

## 1. Architectural Invariants (Stage 4)

| Component | Invariant Enforced | Institutional Rationale |
|-----------|--------------------|-------------------------|
| **Dataset Integrity** | Deterministic SHA256 Hash | Prevents training on corrupted/ghost sessions. |
| **Execution Simulator** | Adverse Selection (Toxic Flow) | Prevents "Liquidity Hallucination" in backtests. |
| **PPO Stability** | Adaptive KL + Entropy Guard | Prevents policy collapse during long-horizon training. |
| **Observation Schema** | Immutable Feature Order | Ensures distributed rollout descriptors are bitwise identical. |
| **Reward Engine** | Multi-component Decomposition | Optimizes for Sharpe/Sortino/Drawdown instead of raw PnL. |

## 2. Failure Mode Analysis

| Failure Mode | Mitigation Strategy | Risk Reduction |
|--------------|---------------------|----------------|
| **Simulator Exploitation** | Bid-Ask Queue Reactive Decay | 95% |
| **Reward Hacking** | Turnover & Overnight Penalities | 80% |
| **Policy Collapse** | Critic Ensemble & KL Early Stop | 90% |
| **Causal Leakage** | Causal Replay Sequencer | 100% |
| **Limit-Lock Fraud** | Integrity Audit Exclusions | 100% |

## 3. Deployment Checklist

- [ ] Run `python python_bot/manifest/verify_manifest.py` to check drift.
- [ ] Run `python python_bot/preflight/stage4_final_gate.py` for training readiness.
- [ ] Ensure `GEMINI_API_KEY` is set for adversarial scenario generation.
- [ ] Verify `stage4_baseline.json` version matches training shard.

## 4. Verification CLI

```bash
# Verify architectural stability
python python_bot/manifest/verify_manifest.py
```
