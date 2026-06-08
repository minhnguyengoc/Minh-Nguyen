# VNStock RL Trading Gaps and System Limits

This document maps out the specific architectural constraints, known risks, and features limits as a security and principal engineer.

## Critical Gaps

### 1. Multi-Symbol Shared Account Constraints
Using multiple concurrent symbols in `paper-live` or real execution is **STRICTLY BLOCKED** and disabled.
* **Reason**: The internal observation and prediction parameters rely on shared cash, position quantities, and settling queues inside the standard simulation environments. Running concurrent inferences for multiple tickets would overwrite shared ledger assets and trigger severe trade collisions (lookahead and accounting state contaminations).
* **Mitigation**: Multi-symbol trading must only run sequentially or within isolated sandbox worker threads. In paper-live modes, only `STB` is permitted.

### 2. T+2 Settling Queue
The Vietnamese stock market (HOSE) requires a strict T+2 clearing sequence.
* **Limitation**: Buying shares on day $T$ deposits them into unsettled queues. They cannot be sold until $T+2$. While this is properly captured in current model states, rapid trend reversals during the settling cycle may create structural, unavoidable drawdowns during adverse high-volatility shifts.

### 3. Slippage Model Assumptions
Our slippage model assumes volume-impact metrics calculated based on historical candles. Live order queue priority, book bid-ask spreads, and fractional fill times may deviate slightly during high-volume market events.
