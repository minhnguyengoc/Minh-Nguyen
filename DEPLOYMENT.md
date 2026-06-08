# Deployment Architecture Guide

Deploying the VNStock institutional reinforcement learning bot requires configuring active variables, log-level files, and registry pathways.

## Pre-requisites
* **Python v3.10+**
* Stable-Baselines3, PyTorch, Pandas, and sub-libraries.
* API network connectivity to retrieve vnstock quotes.

## Environment Parameters
Copy the sample environment variable template file to `.env`:
```env
BOT_TICKERS=STB
BOT_TIMEFRAME=1m
PAPER_MAX_STEPS=2619
INITIAL_CASH=100000000
LIVE_POLL_SECONDS=60
```

## Logging and Persistence Directories
Ensure the following directories are mapped with full WRITE permission access rights:
* `historical_data/`: Caches candle CSV data feeds.
* `logs/`: Holds transactional and security compliance error logging.
* `paper_live_logs/`: Saves persistent ledger sequences and trade snapshot JSON files.
* `checkpoints/`: Model weights zip repository directory.
