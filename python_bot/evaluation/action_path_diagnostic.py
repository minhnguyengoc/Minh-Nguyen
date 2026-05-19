import sys
from pathlib import Path
from collections import Counter

import numpy as np
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

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    tickers = ["FPT", "MWG", "HPG", "SSI", "VCB", "MBB", "STB"]

    print(f"Loading model: {model_path}")
    model = PPO.load(str(model_path))

    print(f"Loading multi-symbol evaluation data: {tickers}")
    df = load_multi_ticker_data(tickers)
    df = sanitize_training_dataframe(df)

    print(f"Sanitized evaluation dataframe shape: {df.shape}")

    env = VNStockTradingEnv(df=df, initial_capital=100_000_000.0)

    model_obs_dim = int(model.observation_space.shape[0])
    env_obs_dim = int(env.observation_space.shape[0])

    print(f"Model observation dim: {model_obs_dim}")
    print(f"Env observation dim  : {env_obs_dim}")
    print(f"Model action space   : {model.action_space}")
    print(f"Env action space     : {env.action_space}")

    if model_obs_dim != env_obs_dim:
        raise RuntimeError(
            f"OBSERVATION_SPACE_MISMATCH: model={model_obs_dim}, env={env_obs_dim}"
        )

    obs, info = env.reset()

    raw_action_counts = Counter()
    position_counts = Counter()
    rejection_counts = Counter()
    executed_position_changes = 0
    trade_count_from_info = 0
    reward_sum = 0.0
    equity_last = None

    prev_position = info.get("position", 0) if isinstance(info, dict) else 0

    done = False
    steps = 0
    max_steps = min(5_000, len(df) - 2)

    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        action_int = normalize_action(action)

        raw_action_counts[action_int] += 1

        result = env.step(action_int)

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = result

        reward_sum += float(reward)
        steps += 1

        if steps % 1000 == 0:
            print(f"Diagnostic progress: {steps}/{max_steps}")

        if isinstance(info, dict):
            pos = info.get("position", info.get("position_after", prev_position))
            position_counts[pos] += 1

            if pos != prev_position:
                executed_position_changes += 1

            prev_position = pos
            equity_last = info.get("equity", equity_last)
            reason = info.get("rejected_reason", None)
            if reason:
                rejection_counts[str(reason)] += 1

            trade_count_from_info = max(
                trade_count_from_info,
                int(info.get("total_trades", 0) or 0)
            )

    total_actions = sum(raw_action_counts.values())
    hold_count = raw_action_counts.get(0, 0)
    buy_count = raw_action_counts.get(1, 0)
    sell_count = raw_action_counts.get(2, 0)

    hold_pct = hold_count / max(total_actions, 1)
    buy_pct = buy_count / max(total_actions, 1)
    sell_pct = sell_count / max(total_actions, 1)

    active_actions = total_actions - hold_count
    raw_action_frequency = active_actions / max(total_actions, 1)
    executed_trade_frequency = trade_count_from_info / max(steps, 1)

    print("\n========== ACTION PATH DIAGNOSTIC ==========")
    print(f"Steps evaluated              : {steps}")
    print(f"Raw action distribution      : {dict(raw_action_counts)}")
    print(f"HOLD count                   : {hold_count}")
    print(f"BUY count                    : {buy_count}")
    print(f"SELL/CLOSE count             : {sell_count}")
    print(f"HOLD percentage              : {hold_pct:.2%}")
    print(f"BUY percentage               : {buy_pct:.2%}")
    print(f"SELL/CLOSE percentage        : {sell_pct:.2%}")
    print(f"Raw action frequency         : {raw_action_frequency:.2%}")
    print(f"Executed trade frequency     : {executed_trade_frequency:.2%}")
    print(f"Position distribution        : {dict(position_counts)}")
    print(f"Executed position changes    : {executed_position_changes}")
    print(f"Trade count from env info    : {trade_count_from_info}")
    print(f"Rejected reasons             : {dict(rejection_counts)}")
    print(f"Total reward                 : {reward_sum:.6f}")
    print(f"Ending equity                : {equity_last}")

    if equity_last is not None and equity_last <= 0:
        diagnosis = "CASE_E_NEGATIVE_EQUITY_OR_EXECUTION_TOO_AGGRESSIVE"
    elif not np.isfinite(reward_sum):
        diagnosis = "CASE_F_NAN_REWARD"
    elif hold_pct > 0.98 and trade_count_from_info == 0:
        diagnosis = "CASE_A_MODEL_ALWAYS_HOLD"
    elif active_actions > 0 and trade_count_from_info == 0:
        diagnosis = "CASE_B_ACTIONS_NOT_EXECUTED_OR_BLOCKED"
    elif trade_count_from_info > 0 and raw_action_frequency > 0.35:
        diagnosis = "CASE_C_RAW_ACTION_OVERTRADING"
    elif trade_count_from_info > 0:
        diagnosis = "CASE_OK_POLICY_TRADES"
    else:
        diagnosis = "CASE_UNKNOWN"

    print(f"\nFINAL DIAGNOSIS              : {diagnosis}")
    print("============================================\n")


if __name__ == "__main__":
    main()
