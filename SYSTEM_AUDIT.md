# SYSTEM_AUDIT

## 1. Repository Tree
```
python_bot/
  api_state.py
  backtest_agent.py
  data_loader.py
  launch_training.sh
  live_data_feed.py
  market_env.py
  matching_engine.py
  ppo_agent.py
  reward_engine.py
  stage3_audit.py
  strategy_router.py
  trainer.py
  validator.py
  verify_institutional.py
  vn_stock_live_data_feed.py
  backtest/
    backtester.py
    cost_model.py
  common/
    types.py
  data/
    history_loader_4y.py
    integrity.py
    replay_dataset.py
  engine/
    execution.py
    features.py
    impact.py
    ledger.py
    normalization.py
    ood.py
    regime.py
  evaluation/
    action_path_diagnostic.py
    entry_quality.py
    institutional_metrics.py
    nav_report.py
    plotting.py
    policy_behavior.py
    policy_trade_diagnostics.py
    reporting.py
    walkforward.py
  features/
    feature_schema.py
  infrastructure/
    clock.py
    sequencer.py
    session.py
  live/
    paper_engine.py
  manifest/
    stage4_baseline.json
    verify_manifest.py
  monitoring/
    training_monitor.py
  preflight/
    stage4_final_gate.py
    stage4_gate.py
    training_gate.py
  risk/
    factor_risk.py
    latency.py
    liquidation.py
    supervisor.py
  system/
    gateway.py
  training/
    action_monitor.py
    distributed_guard.py
    entry_labels.py
    env.py
    exploit_guard.py
    generalization.py
    ppo_config.py
    reward.py
    stability.py
  validation/
    microstructure_audit.py
    policy_sanity.py
```

## 2. Dependency Graph
- **Stable-Baselines3** depends on **PyTorch**, **Gymnasium**, **NumPy**, **Pandas**, **scikit-learn**.
- `trainer.py` & `ppo_agent.py` depend on `stable_baselines3`.
- `market_env.py` depends on `gymnasium` (or standard `gym`), `numpy`, `pandas`.
- `VNStockInstitutionalEnv` (`training/env.py`) depends on `gymnasium`/`gym`, `engine/ledger.py`, `training/exploit_guard.py`, `common/types.py`.
- `data_loader.py` depends on `pandas`, `numpy`, `scikit-learn` (`RobustScaler`).

## 3. Execution Graph
1. **Data Loading:** CSV/API -> `data_loader.py` -> Cleaned DataFrame.
2. **Environment Simulation:** DataFrame -> `market_env.py` or `training/env.py` -> Observation Vectors.
3. **Training Control (PPO):** `trainer.py` wraps Environment -> PPO Agent Training -> Metric Callback & `ActionMonitor` checks limits -> Checkpoint saved to `checkpoints/`.
4. **Diagnostic & Replay:** Checkpoint -> `action_path_diagnostic.py` or `evaluation/` metrics.

## 4. Model Loading Path
- Local Zip checkpoints stored in `checkpoints/`.
- Loaded via `ppo_agent.py` using `PPOAgent.load()` or directly using Stable-Baselines3 `PPO.load(path)`.

## 5. Data Loading Path
- Look for tick / 1m candles in `historical_data/<symbol>_1m.csv`.
- If missing, fetch from `vnstock` v4 API (`from vnstock.api.quote import Quote`).
- As a last resort, fall back to high-quality synthetic/validation data generation in `data_loader.py`.

## 6. Replay Path
- Runs test episodes in deterministic mode (`deterministic=True`) using model.
- Evaluates equity curves, position sizing, hit ratios, drawdowns.

## 7. Training Path
- Triggered via `trainer.py` with multi-ticker combined datasets.
- Trains `PPOAgent` on normalized, sanitized features with regular validation callbacks.

## 8. Live Feed Path
- Utilizes `vn_stock_live_data_feed.py` and `live_data_feed.py`.
- Feeds live ticks to the `gateway.py` to produce standardized observations.

## 9. Risk Engine Path
- Intercepts actions in the `risk/` layer (cooldown, factor exposure, liquidation, lot size check).
- Governed by `risk/supervisor.py` wrapper around actions.

## 10. Checkpoint & Data Files
- `checkpoints/ppo_vn30_multi_stage4.zip` (target model)
- `historical_data/STB_1m.csv` (target replay data)

## 11. Known Architecture & Security Issues
- **NO_MULTI_SYMBOL_LIVE:** The simulation uses a single shared cash and position account across symbols, causing multi-symbol accounts to collide. Therefore, multi-symbol live trading is completely disabled.
- **Path Traversal Risk:** File paths could have traversals if user-specified symbol strings are not sanitized.
- **Duplicate Code:** Overlapping logic in `live_data_feed.py` and `vn_stock_live_data_feed.py`.

> ⚠️ **CRITICAL WARNING:** This system is configured with a shared accounting layout. **MULTI-SYMBOL LIVE PORTFOLIO TRADING IS STRICTLY FORBIDDEN** until isolation ledger mechanisms are developed. Use single-symbol (STB) paper trading only!
