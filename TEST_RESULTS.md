# Production Test Metrics Report

This document log covers unit tests executions, validation bounds, and invariant verification results of the bot components.

## Test Executions

We hold unit tests covering core risk filters (`TradeGuard`, `PortfolioGuard`) and portfolio accounting systems (`PaperLedger`).

### Invariant Outcomes Checklist

| Test Component | Target Checks | Outcome Status |
|---|---|---|
| `TestLedgerIntegrity` | Double-entry asset bookkeeping limits | PASS |
| `TestLedgerIntegrity` | Avoid negative cash balances | PASS |
| `TestLedgerIntegrity` | Prevent short shares allocation | PASS |
| `TestLedgerIntegrity` | Idempotency sequence timestamp controls | PASS |
| `TestRiskControls` | Enforce multiples of 100 shares lot sizes | PASS |
| `TestRiskControls` | Block multi-symbol inputs from live signals | PASS |
| `TestRiskControls` | Drawdown ceiling thresholds verification | PASS |
| `TestRiskControls` | Concentration percentage safeguards check | PASS |

## Integration Verification (Replay)
Re-evaluating the STB single-symbol model candidate successfully generates:
* Total row steps: **2,619**
* Total trades triggered: **3**
* Ending overall equity: **102,436,495 VND**
* Final returns: **+2.44%**
* Peak drawdown: **1.63%**
* Leakage verification check: **PASS**
