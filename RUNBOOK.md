# VNStock Trading Bot Operational Runbook

This guide covers operational task steps, diagnostic commands, and recovery actions of the Vietnam Stock market PPO paper trading container.

## Pre-deployment Diagnostics
Validate system integrity, directory write paths, and standard libraries availability:
```bash
python -m python_bot.live.cli health
```

Check security compliance (lookahead feature leakages, suspicious file path inputs):
```bash
python -m python_bot.live.cli audit
```

## Data Lifecycle Operations
To review current local candlestick data quality, NaN occurrences, and anomalous price movements:
```bash
python -m python_bot.live.cli data-status --ticker STB
```

To fetch recent candles using real-time vnstock stock API feeds:
```bash
python -m python_bot.live.cli update-data --ticker STB
```

## Signal Engine and Replays
Calculate the deterministic PPO strategy prediction based on current CSV entries:
```bash
python -m python_bot.live.cli run-signal --ticker STB
```

Inspect the latest cached signal in the file database:
```bash
python -m python_bot.live.cli latest-signal --ticker STB
```

## Portfolio Accounting and Ledgers
To display recent successful paper orders processed on the ledger:
```bash
python -m python_bot.live.cli ledger --ticker STB
```

To view current asset valuation, cash balance, and unsettled T+2 quantities:
```bash
python -m python_bot.live.cli portfolio --ticker STB
```

## Automated Polling Loop
To trigger real-time polling operations at custom sleep intervals (e.g., 60 seconds):
```bash
python -m python_bot.live.cli loop --ticker STB --poll-seconds 60
```
