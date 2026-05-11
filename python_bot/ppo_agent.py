import os
import logging
import random
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Configure institutional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PPOAgent:
    """
    Institutional-grade PPO Agent for Vietnam Intraday Equity Trading.
    
    Why PPO (Proximal Policy Optimization)?
    1. Clipped Updates: Prevents catastrophic policy collapses in high-noise environments (financial TS).
    2. Stochastic Exploration: Effectively explores non-stationary state spaces without manual epsilon scheduling.
    3. Production Reliability: Computationally efficient and less sensitive to hyperparameters than off-policy methods.
    4. GAE (Generalized Advantage Estimation): Properly balances bias vs variance in rewards for sequential decisions.
    """
    
    def __init__(
        self, 
        env: gym.Env, 
        model_dir: str = "agents/saved_models", 
        model_name: str = "ppo_stock_v1", 
        seed: int = 42, 
        device: str = "auto", 
        paper_trading: bool = True,
        tensorboard_log: Optional[str] = None,
        n_steps: int = 2048,
        batch_size: int = 64
    ):
        self.seed = seed
        self.paper_trading = paper_trading
        
        # 1. Deterministic Seeding for Research Reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 2. Workspace Management
        self.model_name = model_name
        self.base_path = os.path.join(model_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # 3. Space Validation & Vectorization
        if not isinstance(env.action_space, gym.spaces.Discrete) or env.action_space.n != 4:
            raise ValueError("Environment must have Discrete(4) action space (HOLD, LONG, SHORT, CLOSE).")
        
        self.raw_env = env  # Keep reference for custom evaluation
        self.env = DummyVecEnv([lambda: env])
        
        # 4. Custom MLP Architecture (Deeper pi/vf heads for complex feature extraction)
        policy_kwargs = dict(
            net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
            activation_fn=torch.nn.Tanh
        )

        # 5. Model Initialization (Tuned for noisy 1m intraday data)
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=1.5e-4,     # Reduced for higher stability in noisy financial series
            n_steps=n_steps,          
            batch_size=batch_size,    
            n_epochs=10,
            gamma=0.99,               
            gae_lambda=0.95,
            clip_range=0.2,           
            ent_coef=0.01,            # Increased exploration pressure
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.015,          # Safety brake for policy updates
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=self.seed,
            device=device,
            tensorboard_log=tensorboard_log
        )
        
        logger.info(f"PPOAgent Initialized | Mode: {'PAPER' if paper_trading else 'SIM'} | Device: {self.model.device}")

    def train(self, total_timesteps: int, callback: Optional[BaseCallback] = None, reset_steps: bool = False):
        """Executes the learning cycle."""
        logger.info(f"Training session started: {total_timesteps:,} steps.")
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                reset_num_timesteps=reset_steps,
                progress_bar=True
            )
        except AssertionError as e:
            # Handle SB3's technical check if a callback stops training before a rollout completes
            # This happens during early stopping via 'dd_breach' or 'stop_early'
            if "self.full" in str(e) or not str(e):
                logger.warning("⏹️ Training stopped mid-rollout. Skipping final optimization batch to avoid Buffer Error.")
            else:
                raise e
        except Exception as e:
            logger.error(f"Unexpected training error: {e}")
            raise e
        finally:
            self.save()

    def save(self):
        """Standardized persistence."""
        self.model.save(self.base_path)
        logger.info(f"Model saved: {self.base_path}.zip")

    def load(self) -> bool:
        """Loads weights into current architecture."""
        zip_path = f"{self.base_path}.zip"
        if os.path.exists(zip_path):
            self.model = PPO.load(zip_path, env=self.env, device=self.model.device)
            logger.info(f"Weights restored: {zip_path}")
            return True
        else:
            logger.warning(f"No checkpoint found at {zip_path}. Using random init.")
            return False

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Calculates optimal action. Handles batch and single observations."""
        obs = observation if observation.ndim > 1 else observation[np.newaxis, :]
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action[0]) if isinstance(action, np.ndarray) else int(action)

    def evaluate_policy(self, episodes: int = 1) -> Dict[str, Any]:
        """
        Custom high-fidelity evaluation loop to extract trading-specific metrics.
        Runs full episodes against the raw environment to capture per-tick info statistics.
        """
        logger.info(f"Evaluating policy over {episodes} episodes...")
        eval_metrics = {
            "ending_equity": [],
            "total_reward": [],
            "total_trades": [],
            "max_drawdown": [],
            "win_rate": []
        }

        for i in range(episodes):
            obs, info = self.raw_env.reset(seed=self.seed + i)
            terminated = False
            truncated = False
            episode_reward = 0.0
            equity_curve = [info.get("equity", 100_000_000)]
            trades_log = []

            while not (terminated or truncated):
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.raw_env.step(action)
                episode_reward += reward
                equity_curve.append(info.get("equity", equity_curve[-1]))

                if "trade_result" in info:
                    trades_log.append(info["trade_result"])

            # Calculate metrics
            final_equity = equity_curve[-1]
            eval_metrics["ending_equity"].append(final_equity)
            eval_metrics["total_reward"].append(episode_reward)

            # Max Drawdown (Peak-to-Trough)
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / np.maximum(peak, 1e-9)
            eval_metrics["max_drawdown"].append(float(np.max(drawdown)))

            # Trade stats
            n_trades = len(trades_log)
            eval_metrics["total_trades"].append(n_trades)
            if n_trades > 0:
                wins = sum(1 for t in trades_log if t.get("pnl", 0) > 0)
                eval_metrics["win_rate"].append(wins / n_trades)
            else:
                eval_metrics["win_rate"].append(0.0)

        # Aggregate & log
        avg_metrics = {k: np.mean(v) for k, v in eval_metrics.items()}
        logger.info(
            f"Evaluation Complete | Avg Equity: {avg_metrics['ending_equity']:,.0f} VND | "
            f"Avg Reward: {avg_metrics['total_reward']:.2f} | Trades: {avg_metrics['total_trades']:.1f} | "
            f"Win Rate: {avg_metrics['win_rate']*100:.1f}% | Max DD: {avg_metrics['max_drawdown']*100:.2f}%"
        )
        return avg_metrics
