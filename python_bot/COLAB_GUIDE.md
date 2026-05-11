# 🚀 Colab / Jupyter Integration Guide

To ensure high-fidelity institutional results and environment stability, follow these guidelines:

## 1. Environment Synchronization
Always run this script at the start of your session to pull the latest architectural updates and fix pathing:
```bash
!bash python_bot/setup_colab.sh
```

## 2. Running Scripts
You **MUST** prefix shell commands with `!` or `%`. Running `python script.py` in a code cell without a prefix will cause a `SyntaxError`.

### Training
```bash
!python python_bot/trainer.py --ticker FPT --steps 100000
```

### Backtesting (with Behavioral Audit)
```bash
!python python_bot/backtest_agent.py --ticker FPT --model checkpoints/ppo_fpt_intraday
```

### Behavioral Diagnostic (Standalone)
```bash
!python python_bot/evaluation/policy_behavior.py
```

### Multi-Stock VN30 Audit
To ensure the policy isn't over-tuned to a single ticker, run the batch suite across the top 10 liquid VN30 stocks:
```bash
!bash python_bot/batch_vn30_audit.sh
```

## 3. Viewing Logs
The system automatically generates a detailed `Audit Report` (Markdown) in `logs/reports/` after every backtest. You can display it in Colab:

```python
from IPython.display import Markdown, display
import os

# Find most recent report
report_dir = "logs/reports"
reports = sorted([f for f in os.listdir(report_dir) if f.startswith("audit_")])
if reports:
    with open(os.path.join(report_dir, reports[-1]), 'r') as f:
        display(Markdown(f.read()))
```

## 4. Stability Note
If you see warnings about `gym`, rerun `!pip uninstall gym -y` followed by `!pip install gymnasium`. The `setup_colab.sh` script now handles this automatically.
