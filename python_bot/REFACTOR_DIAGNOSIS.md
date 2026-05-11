# VNStock RL Architectural Refactor & Audit Report

## 1. Architectural Diagnosis
The previous system exhibited several "Retail-Grade" failure modes unsuitable for live capital deployment:
- **Causality Leakage**: Time was handled via system `datetime.now()`, making replay inconsistent.
- **Exposure Ghosting**: Positions were updated upon *intention* (action) rather than *confirmation* (fill).
- **Session Blindness**: Rolling features crossed lunch breaks and overnight gaps, polluting the state with stale volatility.
- **Microstructure Naivety**: Immediate full fills ignored LOB depth and queue priority.
- **Heuristic Drift**: Monitoring relied on simple EMA deviations rather than statistical distribution distance.

## 2. Refactored System Design
The new architecture is decoupled into immutable logic and stateful ledgers:
- **Infrastructure Layer**: `ClockTriangulator` and `SessionBoundaryFSM` enforce temporal authority.
- **Engine Layer**: `FeatureEngine` (causal only), `QueueFillSimulator` (microstructure aware), and `RegimeNormalizationRouter` (frozen scalers).
- **Accounting Layer**: `ExposureLedger` enforces the "Fill-First" invariant.
- **Risk Layer**: `RiskSupervisor` acts as a final circuit breaker with OOD (Out-of-Distribution) gating.

## 3. Failure-Mode Coverage Table
| Failure Mode | Mitigation Strategy | Implementation |
| :--- | :--- | :--- |
| **Look-ahead Bias** | Strict Causal Processing | `FeatureEngine` (Oldest -> Newest) |
| **Exposure Contamination** | Delta-based Fill Accounting | `ExposureLedger` |
| **Session Boundary Leak** | FSM-triggered Resets | `SessionBoundaryFSM.is_interrupted()` |
| **Microstructure Hubris** | Queue-aware Fill Simulation | `QueueFillSimulator` (Decay model) |
| **Regime Drift** | Statistical Monitoring (PSI) | `DriftMonitor` |
| **Hidden State Mutability** | Deterministic Replay Guard | `ClockTriangulator` |

## 4. Determinism Guarantees
- **Clock Anchor**: `ClockTriangulator` ensures `Gateway` time only advances via data timestamps.
- **State Seeding**: All components (Feature Buffer, Ledger) are reset to bitwise-identical values at session start.
- **Vectorized Determinism**: `pandas`/`numpy` operations use fixed dtypes (`float32`) and handle `NaN`/`inf` explicitly.

## 5. Statistical Justification (Drift Framework)
We replaced heuristic thresholds with **Population Stability Index (PSI)**. 
- **Inference**: If $PSI > 0.1$, the environment is potentially OOD; if $PSI > 0.25$, we trigger the `RiskSupervisor` to abstain from trading.
- **Regimes**: Separate scalers are maintained for BULL/BEAR/SIDEWAYS markets, preventing BEAR-market volatility from skewing BULL-market signals.

## 6. Execution Realism
`QueueFillSimulator` implements a **Queue Decay Model**:
1. Order enters at `Initial_Depth` (LOB depth or Volume Proxy).
2. For each tick, `Queue_Position` is reduced by $Volume \times Tick\_Efficiency$.
3. Fill occurs ONLY when $Queue\_Position \le 0$. 
This accurately models "waiting in line" for passive orders.

## 7. Adversarial Readiness (Institutional Stage 4 - PRODUCTION TRAINING READY)
The system is now fully hardened for adversarial scale-up. Key Stage 4 enhancements:
- **DatasetIntegrityAuditor**: Detects market discontinuities, auction contamination, and OHLC corruption.
- **GeneralizationEngine**: Implements Domain Randomization and structured observation perturbation.
- **PPOStabilityHardeners**: Protects against entropy collapse, gradient explosions, and KL policy drift.
- **HybridReactiveExecution**: Accounts for partial fills and adverse selection (Toxic Flow).
- **InstitutionalRewardEngine**: Optimized for long-term Sharpe expectancy and drawdown survival.
- **PolicyBehaviorAnalyzer**: AI-driven exploit detection (Hyper-turnover & Scalp-farming flags).

## 8. Development & Audit Commands
```bash
# Stage 3: Determinism & Causal Ledger Verification
python python_bot/verify_institutional.py

# Stage 4: Production Training Readiness Gate
python python_bot/preflight/stage4_final_gate.py

# Run Full Invariant Test Suite
python python_bot/tests/invariant_tests.py
```
