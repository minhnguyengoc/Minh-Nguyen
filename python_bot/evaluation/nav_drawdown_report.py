import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

ROOT = Path("/content/Minh-Nguyen")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def normalize_action(action):
    arr = np.asarray(action)
    if arr.shape == ():
        return int(arr.item())
    return int(arr.flatten()[0])


def main():
    from python_bot.trainer import load_multi_ticker_data, sanitize_training_dataframe
    from python_bot.market_env import VNStockTradingEnv

    model_path = ROOT / "checkpoints/ppo_vn30_multi_stage4.zip"
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)

    tickers = ["FPT", "MWG", "HPG", "SSI", "VCB", "MBB", "STB"]

    model = PPO.load(str(model_path))

    df = load_multi_ticker_data(tickers)
    df = sanitize_training_dataframe(df)

    env = VNStockTradingEnv(df=df, initial_capital=100_000_000.0)

    obs, info = env.reset()

    rows = []
    trade_rows = []

    done = False
    steps = 0
    max_steps = min(20_000, len(df) - 2)

    prev_equity = 100_000_000.0
    peak_equity = 100_000_000.0

    rejection_counts = Counter()
    action_counts = Counter()

    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        action_int = normalize_action(action)
        action_counts[action_int] += 1

        result = env.step(action_int)

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = result

        equity = float(info.get("equity", prev_equity))
        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / max(peak_equity, 1e-9)

        reason = info.get("rejected_reason")
        if reason:
            rejection_counts[str(reason)] += 1

        position_changed = bool(info.get("position_changed", False))

        row = {
            "step": steps,
            "action": action_int,
            "reward": float(reward),
            "equity": equity,
            "cash": float(info.get("cash", 0.0)),
            "position": float(info.get("position", 0.0)),
            "drawdown": drawdown,
            "position_changed": position_changed,
            "total_trades": int(info.get("total_trades", 0)),
            "executed_qty": float(info.get("executed_qty", 0.0) or 0.0),
            "executed_price": info.get("executed_price"),
            "transaction_cost": float(info.get("transaction_cost", 0.0) or 0.0),
            "rejected_reason": reason,
        }
        rows.append(row)

        if position_changed:
            trade_rows.append(row)

        prev_equity = equity
        steps += 1

        if steps % 1000 == 0:
            print(f"Report progress: {steps}/{max_steps}")

    report = pd.DataFrame(rows)
    trades = pd.DataFrame(trade_rows)

    report_path = out_dir / "nav_drawdown_report.csv"
    trades_path = out_dir / "trade_log.csv"
    summary_path = out_dir / "summary.txt"

    report.to_csv(report_path, index=False)
    trades.to_csv(trades_path, index=False)

    ending_equity = report["equity"].iloc[-1]
    total_return = ending_equity / 100_000_000.0 - 1
    max_drawdown = report["drawdown"].max()
    total_trades = int(report["total_trades"].max())
    executed_trade_frequency = total_trades / max(len(report), 1)

    summary = f"""
NAV / DRAWDOWN REPORT
=====================

Steps evaluated: {len(report)}
Ending equity: {ending_equity:,.2f}
Total return: {total_return:.2%}
Max drawdown: {max_drawdown:.2%}

Total trades: {total_trades}
Executed trade frequency: {executed_trade_frequency:.2%}

Action counts: {dict(action_counts)}
Rejected reasons: {dict(rejection_counts)}

Files:
- {report_path}
- {trades_path}
"""

    summary_path.write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
