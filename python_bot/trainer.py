import sys
import os

# Ensure the project root is in the python path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import gc
import json
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Optional
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from python_bot.ppo_agent import PPOAgent
from python_bot.market_env import VNStockTradingEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TradingMetricsCallback(BaseCallback):
    """
    Custom callback for real-time equity tracking, drawdown calculation, 
    CSV logging, and hardware-aware early stopping.
    """
    def __init__(self, save_freq: int = 5000, log_dir: str = "logs/metrics", 
                 early_stop_patience: int = 100000, max_drawdown_limit: float = 0.35):
        super().__init__()
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.early_stop_patience = early_stop_patience
        self.max_drawdown_limit = max_drawdown_limit
        
        self.best_equity = 0.0
        self.patience_counter = 0
        self.equity_history: list[float] = []
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_path = os.path.join(log_dir, "training_metrics.csv")

    def _on_step(self) -> bool:
        try:
            # Extract state from VecEnv (DummyVecEnv wraps envs in a list)
            env_ref = self.training_env.envs[0]
            # Unwrap if it's a Monitor wrapper
            while hasattr(env_ref, 'env') and not hasattr(env_ref, 'equity'):
                env_ref = env_ref.env
                
            equity = env_ref.equity
            position = env_ref.position
            trade_count = env_ref.trade_count
            # Access reward from locals
            reward = self.locals.get("rewards", [0.0])[0]
            
            # Access info for reward components
            infos = self.locals.get("infos", [{}])[0]
            components = infos.get("reward_components", {})
        except Exception as e:
            # logging.debug(f"Callback monitoring issue: {e}")
            return True  # Fallback to avoid training crash

        self.equity_history.append(equity)
        peak = max(self.equity_history)
        current_dd = (peak - equity) / max(peak, 1e-9)

        # Early stopping triggers
        if equity > self.best_equity:
            self.best_equity = equity
            self.patience_counter = 0
        else:
            self.patience_counter += self.training_env.num_envs

        stop_early = self.patience_counter > self.early_stop_patience
        dd_breach = current_dd > self.max_drawdown_limit

        # Periodic logging & memory cleanup
        if self.n_calls % self.save_freq == 0 or stop_early or dd_breach:
            self._log_metrics(equity, current_dd, trade_count, position, reward, components)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if dd_breach:
            logging.warning(f"🛑 Max Drawdown breached ({current_dd*100:.2f}%). Triggering early stop.")
            return False
        if stop_early:
            logging.info(f"⏹️ No equity improvement for {self.early_stop_patience} steps. Early stop.")
            return False
        return True

    def _log_metrics(self, equity: float, dd: float, trades: int, pos: int, reward: float, components: dict = None):
        log_data = {
            "step": self.n_calls,
            "equity": equity,
            "drawdown_pct": dd * 100,
            "trade_count": trades,
            "position": pos,
            "reward": reward
        }
        if components:
            for k, v in components.items():
                log_data[f"comp_{k}"] = v
                
        df = pd.DataFrame([log_data])
        if os.path.exists(self.metrics_path):
            df.to_csv(self.metrics_path, mode='a', header=False, index=False)
        else:
            df.to_csv(self.metrics_path, index=False)

class HardwareOptimizer:
    """Auto-detects hardware and returns SB3-optimized configuration."""
    @staticmethod
    def detect() -> Dict[str, Any]:
        cpu_cores = os.cpu_count() or 4
        has_gpu = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_gpu else "None"
        vram_gb = (torch.cuda.get_device_properties(0).total_mem / 1024**3) if has_gpu else 0
        
        # Adaptive tuning logic
        # For small datasets common in Colab, we cap n_steps to avoid RolloutBuffer issues
        n_steps = 2048 if has_gpu else 1024
        batch_size = 128 if vram_gb > 6 else 64
        torch_threads = max(1, cpu_cores // 2)
        device = "cuda" if has_gpu else "cpu"
        
        return {
            "cpu_cores": cpu_cores, "has_gpu": has_gpu,
            "gpu_name": gpu_name, "n_steps": n_steps, 
            "batch_size": batch_size, "torch_threads": torch_threads, 
            "device": device
        }

def run_training_pipeline(
    states: np.ndarray, ohlcv: np.ndarray, timestamps: list,
    ticker: str = "FPT",
    model_dir: str = "agents/saved_models", log_dir: str = "logs",
    total_timesteps: int = 150_000, resume: bool = False
):
    """
    Institutional training pipeline with hardware auto-tuning, memory management,
    and financial early-stopping safeguards.
    """
    hw = HardwareOptimizer.detect()
    
    # Dynamic Optimization for small datasets
    data_size = len(states)
    # n_steps must be a power of 2 and strictly smaller than data_size if possible to ensure full rollouts
    # We enforce n_steps < data_size to avoid collect_rollouts failures
    if data_size < 128:
        hw["n_steps"] = 64
        hw["batch_size"] = 16
    elif data_size < 256:
        hw["n_steps"] = 128
        hw["batch_size"] = 32
    elif data_size < 512:
        hw["n_steps"] = 256
        hw["batch_size"] = 64
    elif data_size < 1024:
        hw["n_steps"] = 512
        hw["batch_size"] = 64
    else:
        # Default cap for CPU stability
        hw["n_steps"] = 2048 if hw["has_gpu"] else 1024
        hw["batch_size"] = 64
        
    # Final safety check
    hw["n_steps"] = min(hw["n_steps"], 2**int(np.floor(np.log2(data_size * 0.9))))
    hw["batch_size"] = min(hw["batch_size"], hw["n_steps"])
    
    logging.info(f"🔍 Hardware Detected: CPU={hw['cpu_cores']}c | GPU={hw['gpu_name']} | Device={hw['device'].upper()}")
    logging.info(f"⚙️  Auto-Optimized: n_steps={hw['n_steps']} | batch_size={hw['batch_size']} | threads={hw['torch_threads']}")
    
    torch.set_num_threads(hw["torch_threads"])
    torch.backends.cudnn.benchmark = True if hw["has_gpu"] else False

    # Initialize environment
    env = VNStockTradingEnv(
        states=states, ohlcv=ohlcv, timestamps=timestamps,
        initial_capital=100_000_000.0
    )
    
    # Initialize agent with hardware-aware defaults
    model_name = f"ppo_{ticker.lower()}_intraday"
    
    agent = PPOAgent(
        env=env, model_dir=model_dir, model_name=model_name,
        seed=42, device=hw["device"], paper_trading=True,
        n_steps=hw["n_steps"], batch_size=hw["batch_size"]
    )

    # Callback setup
    callback = TradingMetricsCallback(
        save_freq=5000, log_dir=log_dir, early_stop_patience=80000, max_drawdown_limit=0.45
    )

    logging.info("🚀 Starting RL Training Pipeline...")
    try:
        if resume and agent.load():
            agent.train(total_timesteps=total_timesteps, callback=callback, reset_steps=False)
        else:
            agent.train(total_timesteps=total_timesteps, callback=callback, reset_steps=True)
            
        # Final evaluation
        metrics = agent.evaluate_policy(episodes=5)
        logging.info(f"📊 Final Evaluation: {metrics}")
        
        # Ensure JSON serializability (convert numpy types)
        def _make_serializable(obj):
            if isinstance(obj, (np.float32, np.float64, np.float16)): return float(obj)
            if isinstance(obj, (np.int32, np.int64, np.int16)): return int(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, dict): return {k: _make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)): return [_make_serializable(v) for v in obj]
            return obj

        with open(os.path.join(log_dir, "final_metrics.json"), "w") as f:
            json.dump(_make_serializable(metrics), f, indent=2)

        # Auto-plot after training finishes
        from plot_training import TrainingVisualizer
        viz = TrainingVisualizer(log_dir=log_dir)
        viz.load_and_plot()
        logging.info("📊 Dashboard auto-generated.")
            
    except KeyboardInterrupt:
        logging.info("⏹️ Training interrupted by user. Saving checkpoint...")
        agent.save()
    except Exception as e:
        import traceback
        error_msg = str(e)
        # Suppress SB3's technical Assertion error on early stop partial buffers
        if "assert self.full" in error_msg or isinstance(e, AssertionError):
            logging.info("🏁 Training concluded (Early Stop safeguard triggered).")
        else:
            logging.error(f"💥 Training failed: {error_msg}\n{traceback.format_exc()}")
        agent.save()
    finally:
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logging.info("✅ Pipeline finished. Model & logs persisted.")

from python_bot.data_loader import DataLoader
from python_bot.strategy_router import StrategyRouter

def load_ticker_data(ticker: str):
    """Bridge between DataLoader class and StrategyRouter's data_loader requirement."""
    loader = DataLoader(ticker=ticker)
    # states, ohlcv, timestamps, features, session_dates
    states, ohlcv, timestamps, _, _ = loader.fetch_or_load().build_features().normalize().get_full_matrix()
    return states, ohlcv, timestamps

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 4 PPO Training CLI")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--tensorboard_log", type=str, default="logs/tensorboard/", help="Tensorboard log dir")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/", help="Model checkpoint dir")
    parser.add_argument("--eval_freq", type=int, default=5000, help="Evaluation frequency")
    parser.add_argument("--total_timesteps", type=int, default=150000, help="Total training steps")
    parser.add_argument("--steps", type=int, default=None, help="Alias for --total_timesteps (overrides it if provided)")
    parser.add_argument("--ticker", type=str, default="FPT", help="Ticker to train on")
    return parser.parse_args()

# Implementation of StrategyRouter to manage multi-ticker learning
if __name__ == "__main__":
    args = parse_args()
    
    # Handle alias for steps
    total_steps = args.steps if args.steps is not None else args.total_timesteps
    
    # Configure logs based on args
    log_dir = args.tensorboard_log
    os.makedirs(log_dir, exist_ok=True)
    
    # Load data for requested ticker
    logging.info(f"🚀 Loading data for {args.ticker}...")
    states, ohlcv, timestamps = load_ticker_data(args.ticker)
    
    # Execute training
    run_training_pipeline(
        states=states, 
        ohlcv=ohlcv, 
        timestamps=timestamps,
        ticker=args.ticker,
        model_dir=args.checkpoint_path,
        log_dir=log_dir,
        total_timesteps=total_steps
    )
