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
from python_bot.training.action_monitor import ActionMonitor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TradingMetricsCallback(BaseCallback):
    """
    Custom callback for real-time equity tracking, drawdown calculation, 
    CSV logging, and hardware-aware early stopping.
    """
    def __init__(self, save_freq: int = 5000, log_dir: str = "logs/metrics", 
                 model_dir: str = "agents/saved_models",
                 early_stop_patience: int = 100000, max_drawdown_limit: float = 0.35):
        super().__init__()
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.early_stop_patience = early_stop_patience
        self.max_drawdown_limit = max_drawdown_limit
        
        self.action_monitor = ActionMonitor(logging.getLogger("TradingMetricsCallback"))
        
        self.best_sharpe = -np.inf
        self.best_return = -np.inf
        self.best_drawdown = np.inf
        
        self.best_equity = 0.0
        self.patience_counter = 0
        self.equity_history: list[float] = []
        self.returns_history: list[float] = []
        self.actions_history: list[int] = []
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_path = os.path.join(log_dir, "training_metrics.csv")

    def _on_step(self) -> bool:
        try:
            # Extract state from VecEnv (DummyVecEnv wraps envs in a list)
            env_ref = self.training_env.envs[0]
            # Unwrap if it's a Monitor wrapper
            while hasattr(env_ref, 'env') and not hasattr(env_ref, 'equity') and not hasattr(env_ref, 'ledger'):
                env_ref = env_ref.env
                
            # Compatibility check between VNStockTradingEnv and VNStockInstitutionalEnv (ledger)
            if hasattr(env_ref, 'equity'):
                equity = env_ref.equity
                trade_count = env_ref.trade_count
            elif hasattr(env_ref, 'ledger'):
                data_now = env_ref.history[env_ref.current_idx]
                portfolio = env_ref.ledger.get_state(data_now.close)
                equity = portfolio.available_cash + (portfolio.position_quantity * data_now.close)
                trade_count = env_ref.ledger.total_fills
            else:
                equity = 100_000_000.0
                trade_count = 0

            # Access values from locals
            reward = self.locals.get("rewards", [0.0])[0]
            action = self.locals.get("actions", [0])[0]
            
            # Access info for reward components
            infos = self.locals.get("infos", [{}])[0]
            components = infos.get("reward_components", {})
            
            # Record action
            self.action_monitor.record_action(int(action), infos)
            # Check for failure modes
            self.action_monitor.check_failure_modes(self.num_timesteps)
            
        except Exception as e:
            if "POLICY_LEARNED_ALWAYS_HOLD" in str(e) or "OVERTRADING_POLICY" in str(e):
                logging.error(f"❌ Policy Failure: {e}")
                return False
            # logging.debug(f"Callback monitoring issue: {e}")
            return True  # Fallback to avoid training crash


        self.equity_history.append(equity)
        self.actions_history.append(int(action))
        
        if len(self.equity_history) > 1:
            ret = (self.equity_history[-1] / (self.equity_history[-2] + 1e-9)) - 1.0
            self.returns_history.append(ret)

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

        # Periodic logging & best model saving
        if self.n_calls % self.save_freq == 0:
            self._analyze_and_save_best(equity, current_dd, trade_count)
            self._log_metrics(equity, current_dd, trade_count, reward, components)
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

    def _analyze_and_save_best(self, equity: float, dd: float, trades: int):
        # Calculate Sharpe (proxy)
        if len(self.returns_history) > 100:
            avg_ret = np.mean(self.returns_history[-1000:])
            std_ret = np.std(self.returns_history[-1000:]) + 1e-9
            sharpe = (avg_ret / std_ret) * np.sqrt(252 * 240) # Scaled to annual approx
            
            # Action Diversity check
            stats = self.action_monitor.get_stats()
            hold_ratio = stats["hold_pct"]
            ent_p = stats["action_entropy"]

            if hold_ratio > 0.95 or ent_p < 0.1:
                logging.warning(f"⚠️ Policy Collapse Warning | HOLD: {hold_ratio*100:.1f}% | Ent: {ent_p:.2f} | Step: {self.n_calls}")
            
            # Save logic
            current_return = (equity - 100_000_000) / 100_000_000
            
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                self.model.save(os.path.join(self.model_dir, "best_sharpe.zip"))
            
            if current_return > self.best_return:
                self.best_return = current_return
                self.model.save(os.path.join(self.model_dir, "best_return.zip"))
                
            if dd < self.best_drawdown and trades > 5:
                # Require at least 5 trades to avoid 'best' being a 0-trade flat line
                self.best_drawdown = dd
                self.model.save(os.path.join(self.model_dir, "best_drawdown.zip"))

    def _log_metrics(self, equity: float, dd: float, trades: int, reward: float, components: dict = None):
        stats = self.action_monitor.get_stats()
        log_data = {
            "step": self.n_calls,
            "equity": equity,
            "drawdown_pct": dd * 100,
            "trade_count": trades,
            "reward": reward,
            "hold_pct": stats["hold_pct"],
            "action_entropy": stats["action_entropy"]
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

class CurriculumCallback(BaseCallback):
    """
    Adjusts RewardEngine parameters dynamically to implement institutional curriculum.
    Stage 1 (0-50k): Light costs, high exploration.
    Stage 2 (50k-200k): Mid costs, turnover penalties.
    Stage 3 (200k+): Real-world parameters.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_stage = 0

    def _on_step(self) -> bool:
        env_ref = self.training_env.envs[0]
        while hasattr(env_ref, 'env') and not hasattr(env_ref, 'reward_engine'):
            env_ref = env_ref.env
        
        re = env_ref.reward_engine
        step = self.num_timesteps
        
        if step < 50_000:
            stage = 1
            re.turnover_penalty_scale = 0.0001
            re.drawdown_penalty_scale = 1.0
        elif step < 150_000:
            stage = 2
            re.turnover_penalty_scale = 0.0005
            re.drawdown_penalty_scale = 3.0
        else:
            stage = 3
            re.turnover_penalty_scale = 0.001
            re.drawdown_penalty_scale = 5.0
            
        if stage != self.last_stage:
            logging.info(f"🎓 Curriculum Stage Shift: Stage {stage} (Step {step})")
            self.last_stage = stage
            
        return True

def run_training_pipeline(
    df: pd.DataFrame,
    ticker: str = "MULTI",
    model_dir: str = "agents/saved_models", 
    log_dir: str = "logs",
    total_timesteps: int = 300_000, 
    resume: bool = False
):
    """
    Advanced training pipeline with Curriculum Learning, Hardware Optimization,
    and multi-model checkpointing.
    """
    hw = HardwareOptimizer.detect()
    n_steps = 2048 if hw["has_gpu"] else 1024
    batch_size = 64
    
    logging.info(f"🔍 Hardware: CPU={hw['cpu_cores']} | GPU={hw['gpu_name']} | Device={hw['device'].upper()}")
    
    torch.set_num_threads(hw["torch_threads"])
    torch.backends.cudnn.benchmark = True if hw["has_gpu"] else False

    # Initialize environment
    env = VNStockTradingEnv(df=df, initial_capital=100_000_000.0)
    
    model_name = f"ppo_{ticker.lower()}_stage4"
    agent = PPOAgent(
        env=env, model_dir=model_dir, model_name=model_name,
        seed=42, device=hw["device"], paper_trading=True,
        n_steps=n_steps, batch_size=batch_size
    )

    # Callbacks
    curriculum = CurriculumCallback()
    metrics = TradingMetricsCallback(
        save_freq=5000, log_dir=log_dir, model_dir=model_dir,
        early_stop_patience=100000, max_drawdown_limit=0.45
    )
    
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    checkpoint = CheckpointCallback(save_freq=25000, save_path=model_dir, name_prefix=f"{model_name}_chkpt")
    
    callbacks = CallbackList([curriculum, metrics, checkpoint])

    logging.info(f"🚀 Starting Curriculum Training for {ticker}...")
    success = False
    failure_reason = ""
    try:
        if resume and agent.load():
            agent.train(total_timesteps=total_timesteps, callback=callbacks, reset_steps=False)
        else:
            agent.train(total_timesteps=total_timesteps, callback=callbacks, reset_steps=True)
            
        # Final evaluation
        eval_metrics = agent.evaluate_policy(episodes=5)
        logging.info(f"📊 Final Evaluation: {eval_metrics}")
        
        # Save canonical "latest" and "best" placeholders
        agent.save() 
        agent.model.save(os.path.join(model_dir, "latest.zip"))
        
        # Verify verdict
        stats = metrics.action_monitor.get_stats()
        total_trades = stats["executed_position_changes"]
        hold_pct = stats["hold_pct"]
        trade_freq = stats["trade_frequency"]
        
        print("\n" + "="*50)
        print("POLICY_TRAINING_VERDICT:")
        
        if total_trades > 0 and hold_pct < 0.95 and trade_freq <= 0.25:
            print("- RESULT: PASS")
            success = True
        else:
            print("- RESULT: FAIL")
            if total_trades == 0:
                print("- REASON: ZERO_TRADE_POLICY")
                failure_reason = "ZERO_TRADE_POLICY"
            elif hold_pct >= 0.95:
                print("- REASON: POLICY_LEARNED_ALWAYS_HOLD")
                failure_reason = "POLICY_LEARNED_ALWAYS_HOLD"
            elif trade_freq > 0.25:
                print("- REASON: OVERTRADING_POLICY")
                failure_reason = "OVERTRADING_POLICY"
        
        print(f"- Trade Count: {total_trades}")
        print(f"- HOLD Percentage: {hold_pct*100:.2f}%")
        print(f"- Trade Frequency: {trade_freq*100:.2f}%")
        print("="*50 + "\n")
        
    except Exception as e:
        logging.error(f"💥 Training failed: {e}")
        failure_reason = str(e)
        agent.save()
    finally:
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    if not success and failure_reason:
        sys.exit(1) # Fail the CI/gate if verdict is FAIL


from python_bot.data_loader import DataLoader, load_multi_ticker_data
from python_bot.features.feature_schema import get_numeric_feature_columns
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="PPO Institutional Trainer")
    parser.add_argument("--symbol", type=str, default="FPT")
    parser.add_argument("--tickers", type=str, default="FPT,MWG,HPG,SSI,VCB,MBB,STB")
    parser.add_argument("--mode", type=str, default="PAPER")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=50000)
    parser.add_argument("--data-path", type=str, default="historical_data/FPT_1m.csv")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="ppo_fpt_intraday")
    parser.add_argument("--tensorboard-log", type=str, default="logs/tensorboard")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--save-freq", type=int, default=10000)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--initial-cash", type=float, default=100000000)
    parser.add_argument("--allow-hold-collapse", action="store_true")
    return parser.parse_args()

def ensure_trainer_args(args):
    if args.checkpoint_path is None:
        args.checkpoint_path = args.checkpoint_dir
    return args

def sanitize_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataframe for RL training:
    1. Encodes strings (ticker/symbol) as numeric IDs.
    2. Drops non-numeric metadata.
    3. Converts time into features.
    4. Ensures all remaining values are float32 numbers.
    """
    logging.info(f"🧹 Sanitizing DataFrame | Shape: {df.shape}")
    df_clean = df.copy()
    
    # 1. Encode symbol/ticker columns
    for col in ['ticker', 'symbol', 'code']:
        if col in df_clean.columns:
            logging.info(f"  Encoding {col} to numeric IDs...")
            df_clean[f'{col}_id'] = df_clean[col].astype('category').cat.codes
            df_clean = df_clean.drop(columns=[col])
            
    # 2. Extract Time Features
    time_col = None
    for col in ['timestamp', 'datetime', 'date']:
        if col in df_clean.columns:
            time_col = col
            break
            
    if time_col:
        logging.info(f"  Converting {time_col} to numeric time features...")
        ts = pd.to_datetime(df_clean[time_col])
        df_clean['hour'] = ts.dt.hour
        df_clean['minute'] = ts.dt.minute
        df_clean['dayofweek'] = ts.dt.dayofweek
        df_clean['hour_sin'] = np.sin(2 * np.pi * ts.dt.hour / 24.0)
        df_clean['hour_cos'] = np.cos(2 * np.pi * ts.dt.hour / 24.0)
        df_clean = df_clean.drop(columns=[time_col])

    # 3. Drop known string metadata
    metadata_cols = ['ticker_name', 'name', 'company_name', 'stock_code']
    for col in metadata_cols:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])
            
    # 4. Filter for numeric only
    numeric_df = df_clean.select_dtypes(include=[np.number])
    dropped_cols = set(df_clean.columns) - set(numeric_df.columns)
    if dropped_cols:
        logging.info(f"  Dropped non-numeric columns: {dropped_cols}")
        
    # 5. Final Cleaning
    numeric_df = numeric_df.replace([np.inf, -np.inf], 0).fillna(0)
    numeric_df = numeric_df.astype(np.float32)
    
    logging.info(f"✨ Sanitization complete | New Shape: {numeric_df.shape}")
    return numeric_df

if __name__ == "__main__":
    args = ensure_trainer_args(parse_args())
    total_steps = args.steps if args.steps is not None else args.total_timesteps
    
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    logging.info(f"🚀 Loading Multi-Ticker Data: {tickers}")
    
    try:
        df = load_multi_ticker_data(tickers)
        df = sanitize_training_dataframe(df)
        
        run_training_pipeline(
            df=df,
            ticker="VN30_MULTI",
            model_dir=args.checkpoint_path,
            log_dir=args.tensorboard_log,
            total_timesteps=total_steps
        )
    except Exception as e:
        logging.error(f"Fatal Pipeline Error: {e}")
        import traceback
        traceback.print_exc()
