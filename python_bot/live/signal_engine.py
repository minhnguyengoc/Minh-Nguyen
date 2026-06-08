import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO

from python_bot.core.paths import PAPER_LOGS_DIR, CHECKPOINTS_DIR, HISTORICAL_DATA_DIR
from python_bot.core.exceptions import ModelIncompatibilityError
from python_bot.data_loader import DataLoader
from python_bot.market_env import VNStockTradingEnv
from python_bot.signal_engine.signal_state import SignalStateManager
from python_bot.signal_engine.signal_store import SignalStore
from python_bot.live.paper_ledger import load_live_ledger

logger = logging.getLogger("VNStockBot.SignalEngine")

def run_signal(ticker: str = "STB", timeframe: str = "1m") -> dict:
    """
    Executes a deterministic PPO evaluation over historical and live candles of STB.
    Appends only new executed trade signals to store and double-entry ledger.
    """
    if ticker.upper() != "STB":
        return {
            "status": "FAILED",
            "ticker": ticker.upper(),
            "new_signal": False,
            "latest_market_timestamp": "",
            "last_executed_trade": None,
            "ending_equity": 0.0,
            "final_position": 0.0,
            "replay_log_path": "",
            "snapshot_path": "",
            "signals_path": "",
            # Legacy keys to prevent KeyErrors in calling modules
            "timestamp": "",
            "close": 0.0,
            "action": 0,
            "action_name": "HOLD",
            "equity": 0.0,
            "position": 0.0
        }
        
    checkpoint_path = CHECKPOINTS_DIR / "ppo_vn30_multi_stage4.zip"
    if not checkpoint_path.exists():
        alternative_paths = [
            Path("checkpoints/ppo_vn30_multi_stage4.zip"),
            Path("/checkpoints/ppo_vn30_multi_stage4.zip"),
            Path("ppo_vn30_multi_stage4.zip")
        ]
        for alt in alternative_paths:
            if alt.exists():
                checkpoint_path = alt
                break
                
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PPO training checkpoint was not found at {checkpoint_path}")
        
    data_file = HISTORICAL_DATA_DIR / f"{ticker.upper()}_1m.csv"
    if not data_file.exists():
        alt_data = Path(f"historical_data/{ticker.upper()}_1m.csv")
        if alt_data.exists():
            data_file = alt_data
        else:
            raise FileNotFoundError(f"Market data file was not found at {data_file}")
            
    loader = DataLoader(ticker=ticker.upper(), base_dir=str(HISTORICAL_DATA_DIR))
    loader.fetch_or_load()
    loader.build_features()
    loader.normalize()
    processed_df = loader.get_processed_df()
    
    raw_df = pd.read_csv(data_file)
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
    
    model = PPO.load(str(checkpoint_path))
    env = VNStockTradingEnv(df=processed_df, initial_capital=100000000.0)
    
    # 15. Hard check: Raise ModelIncompatibilityError on observation space mismatch
    if model.observation_space.shape[0] != env.observation_space.shape[0]:
        raise ModelIncompatibilityError(
            f"Observation dimension mismatch: model has {model.observation_space.shape[0]}, "
            f"env has {env.observation_space.shape[0]} dimensions."
        )
        
    paper_ledger = load_live_ledger(ticker)
    state = SignalStateManager.load_state(ticker)
    last_processed_timestamp = state.get("last_processed_timestamp")
    
    # Initialize from previous signals to persist last executed trade
    last_executed_trade = None
    signals_df = SignalStore.load_signals(ticker)
    if not signals_df.empty:
        last_row_df = signals_df.iloc[-1]
        last_executed_trade = {
            "timestamp": str(last_row_df["timestamp"]),
            "ticker": str(last_row_df["ticker"]),
            "close": float(last_row_df["close"]),
            "action": int(last_row_df["action"]),
            "action_name": str(last_row_df["action_name"]),
            "equity": float(last_row_df["equity"]),
            "position": float(last_row_df["position"]),
            "executed_qty": float(last_row_df["executed_qty"]),
            "executed_price": float(last_row_df["executed_price"]) if pd.notna(last_row_df["executed_price"]) else None,
            "transaction_cost": float(last_row_df["transaction_cost"]) if pd.notna(last_row_df["transaction_cost"]) else 0.0,
            "rejected_reason": str(last_row_df["rejected_reason"]) if (pd.notna(last_row_df["rejected_reason"]) and last_row_df["rejected_reason"] != "None") else None
        }
        
    log_rows = []
    new_signal_written = False
    
    obs, _ = env.reset()
    step_count = 0
    total_steps = len(processed_df)
    
    while step_count < total_steps:
        raw_row = processed_df.iloc[step_count]
        current_ts = raw_df['timestamp'].iloc[step_count + (len(raw_df) - len(processed_df))]
        close_price = float(raw_row['close'])
        
        shares_before = env.available_shares + sum(env.t_plus_queue)
        
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        
        # 16. Do not fake or simplify fills. Use env.step info only.
        next_obs, reward, terminated, truncated, info = env.step(action)
        obs = next_obs
        
        shares_after = env.available_shares + sum(env.t_plus_queue)
        
        # 1. Position tracking from step info
        position_changed = info.get("position_changed") is True
        
        # 2. Extract rejected reason without manual string replacement
        rejected_reason = info.get("rejected_reason")
        if rejected_reason is None and position_changed:
            rejected_reason = None
            
        # 3. Retrieve executed price or default to None
        executed_price = info.get("executed_price")
        if not position_changed:
            executed_price = None
            
        # 4. Quantity fallback ONLY if missing from info keys
        executed_qty = info.get("executed_qty")
        if executed_qty is None:
            executed_qty = float(abs(shares_after - shares_before))
            
        # 5. Extract equity from step info
        equity = info.get("equity")
        if equity is None:
            equity = float(env.cash + shares_after * close_price)
            
        action_name = "HOLD"
        if action == 1:
            action_name = "BUY"
        elif action == 2:
            action_name = "SELL/CLOSE"
            
        log_row = {
            "timestamp": str(current_ts),
            "ticker": ticker.upper(),
            "close": close_price,
            "action": action,
            "action_name": action_name,
            "equity": float(equity),
            "position": float(shares_after),
            "executed_qty": float(executed_qty) if position_changed else 0.0,
            "executed_price": executed_price,
            "transaction_cost": float(info.get("transaction_cost", 0.0)),
            "rejected_reason": rejected_reason
        }
        log_rows.append(log_row)
        
        if position_changed:
            trade_signal = {
                "timestamp": str(current_ts),
                "ticker": ticker.upper(),
                "close": float(close_price),
                "action": int(action),
                "action_name": action_name,
                "equity": float(equity),
                "position": float(shares_after),
                "executed_qty": float(executed_qty),
                "executed_price": executed_price,
                "transaction_cost": float(info.get("transaction_cost", 0.0)),
                "rejected_reason": rejected_reason
            }
            last_executed_trade = trade_signal
            
            is_new = False
            if last_processed_timestamp is None:
                is_new = True
            else:
                is_new = pd.to_datetime(current_ts) > pd.to_datetime(last_processed_timestamp)
                
            # 8 & 9. Verify signal is strictly new and append only executed signals
            if is_new:
                SignalStore.append_signal(ticker, trade_signal)
                # 12. Invoke double-entry ledger only on actual execution triggers
                paper_ledger.apply_signal_to_ledger(trade_signal)
                
                state["last_processed_timestamp"] = str(current_ts)
                state["current_position"] = float(shares_after)
                state["cash"] = float(env.cash)
                state["unsettled_t1"] = float(env.t_plus_queue[0])
                state["unsettled_t2"] = float(env.t_plus_queue[1])
                state["cumulative_pnl"] = float(equity - 100000000.0)
                SignalStateManager.save_state(ticker, state)
                new_signal_written = True
                
        # 7. Explicit termination handle
        if terminated or truncated:
            break
            
        step_count += 1
        
    os.makedirs(PAPER_LOGS_DIR, exist_ok=True)
    replay_log_file = PAPER_LOGS_DIR / f"paper_live_replay_log_{ticker.upper()}_1m.csv"
    replay_df = pd.DataFrame(log_rows)
    replay_df.to_csv(replay_log_file, index=False)
    
    last_row = log_rows[-1]
    last_timestamp = last_row["timestamp"]
    
    snapshot_file = PAPER_LOGS_DIR / f"paper_live_snapshot_{ticker.upper()}_1m.json"
    # 10. Write complete status schema metadata snapshot
    snapshot_data = {
        "latest_market_timestamp": last_timestamp,
        "latest_close": last_row["close"],
        "final_position": last_row["position"],
        "ending_equity": last_row["equity"],
        "last_executed_trade": last_executed_trade,
        "no_new_signal": not new_signal_written
    }
    with open(snapshot_file, 'w', encoding='utf-8') as sf:
        json.dump(snapshot_data, sf, indent=4)
        
    # 13. Formulate standard pass result structure combining CLI legacy attributes
    return {
        "status": "PASS",
        "ticker": ticker.upper(),
        "new_signal": new_signal_written,
        "latest_market_timestamp": last_timestamp,
        "last_executed_trade": last_executed_trade,
        "ending_equity": last_row["equity"],
        "final_position": last_row["position"],
        "replay_log_path": str(replay_log_file),
        "snapshot_path": str(snapshot_file),
        "signals_path": str(PAPER_LOGS_DIR / f"paper_live_signals_{ticker.upper()}_1m.csv"),
        
        # Legacy support attributes to ensure compatibility with CLI modules
        "timestamp": last_timestamp,
        "close": last_row["close"],
        "action": last_row["action"],
        "action_name": last_row["action_name"],
        "equity": last_row["equity"],
        "position": last_row["position"]
    }
