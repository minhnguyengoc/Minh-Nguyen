# VNStock RL Trading Architecture Specification

This document details the production-grade architecture of the Vietnam stock market reinforcement learning paper-trading platform.

## Repository Layout
```
python_bot/
  ├── core/                 # Standard core parameters, logs configurations, paths, exceptions
  ├── data_pipeline/        # Schemas, data flow validations, and lookahead feature leakage checks
  ├── features/             # Numeric feature definitions and database schema mappings
  ├── live/                 # Real-time update-data data fetchers, signal engine evaluations, and CLI modules
  ├── model_registry/       # Checkpoint validations and weights loader specifications
  ├── monitoring/           # Observability health checks, telemetry analytics, and fault diagnostics
  ├── paper_portfolio/      # Ledger databases, position volumes accounting, and returns metrics
  ├── preflight/            # Historical gating and generalization tests
  ├── risk/                 # Enterprise pre-trade limits, drawdowns guides, and session bounds
  └── security/             # Cryptographic secret screening, path protections, and audit log tracking
```

## System Component Interactions

```
  [Live Market Feed] -> vnstock API
          |
          v
   [Data Fetcher] -> Merges & Sanitizes raw candles -> Writes to CSV
          |
          v
  [Dataset Health] -> Validates monotonicity, duplicates, NaN ratios
          |
          v
   [Signal Engine] -> Invokes ModelLoader -> Runs deterministic PPO inference
          |
          v
    [Risk Engine] -> TradeGuard (Lot, Short limit checks) + PortfolioGuard (Drawdown, Daily loss check)
          |
          v
   [Paper Ledger] -> Process order -> Writes to ledger CSV -> Captures trade snapshot
```

### 1. Data Pipeline
The data pipeline fetches recent Vietnam stock candles using target symbols (STB specifically for paper-live operations), sanitizing dates/times to `Asia/Ho_Chi_Minh` timezone, dropping weekends and holidays, and calculating standard high-quality technical momentum indicator features. Lookahead features are strictly avoided by tracking shift exclusions in automatic static validators.

### 2. Model Management
Loads PPO weights checkpoint files from the registry framework, verifying shape dimensions against actual active environment spaces. Mismatches in state input lengths or outputs completely reject loading.

### 3. Risk Framework
Consists of two pre-trade gate locks:
* **TradeGuard**: Verifies lot-sizing constraints (multiples of 100 shares), forbids naked short-selling of shares, and forces unique instrument symbol bindings.
* **PortfolioGuard**: Checks that current drawdown and daily losses don't breach specified risk limits (such as 5% drawdown or 2% daily loss buffers). Enforces session timings of HOSE.

### 4. Ledger Accounting
The double-entry bookkeeping ledger logs actual successful transactions to a durable CSV database representation with detailed metrics, keeping tracking of unsettled shares across standard T+2 clearing cycles.
