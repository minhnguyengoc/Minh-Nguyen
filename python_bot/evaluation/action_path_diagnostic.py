import os
import pandas as pd
import numpy as np
import torch
import logging
from typing import Dict, Any, List
from stable_baselines3 import PPO
from python_bot.market_env import VNStockTradingEnv
from python_bot.ppo_agent import PPOAgent
from python_bot.data_loader import DataLoader

# Configure diagnostic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ActionPathDiagnostic")

class ActionPathDiagnostic:
    """
    Identifies why a policy might be failing to execute trades.
    Audits the decision-to-execution pipeline.
    """
    def __init__(self, model_path: str, df: pd.DataFrame, episodes: int = 5):
        self.model_path = model_path
        self.df = df
        self.episodes = episodes
        
        # Stats
        self.stats = {
            "total_steps": 0,
            "raw_actions": {0: 0, 1: 0, 2: 0},
            "hold_count": 0,
            "buy_signal_count": 0,
            "sell_signal_count": 0,
            "risk_approved_count": 0,
            "risk_rejected_count": 0,
            "executed_trade_count": 0,
            "fill_rejected_count": 0,
            "position_change_count": 0
        }
        
        self.rejection_log = []

    def run(self):
        logger.info(f"🚀 Starting Path Diagnostics for model: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            logger.error(f"❌ Checkpoint not found: {self.model_path}")
            return "CASE_E_CHECKPOINT_NOT_FOUND"

        # Load environment and agent
        env = VNStockTradingEnv(df=self.df)
        
        # We need to load PPO directly to avoid mismatch if PPOAgent wrapper changes
        try:
            model = PPO.load(self.model_path, env=env)
            logger.info("✅ Model loaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return "CASE_F_MODEL_LOAD_ERROR"

        for episode in range(self.episodes):
            obs, _ = env.reset()
            done = False
            prev_shares = 0
            
            while not (terminated or truncated):
                self.stats["total_steps"] += 1
                
                # 1. Action decision
                action = model.predict(obs, deterministic=True)
                action = int(action)
                self.stats["raw_actions"][action] = self.stats["raw_actions"].get(action, 0) + 1
                
                if action == 0:
                    self.stats["hold_count"] += 1
                elif action == 1:
                    self.stats["buy_signal_count"] += 1
                elif action == 2:
                    self.stats["sell_signal_count"] += 1

                # 2. Execute step
                obs, reward, terminated, truncated, info = env.step(action)
                
                pos_changed = info.get("position_changed", False)
                
                # Check for rejections
                if action in [1, 2]: 
                    if not pos_changed:
                        self.stats["risk_rejected_count"] += 1
                    else:
                        self.stats["risk_approved_count"] += 1
                        self.stats["executed_trade_count"] += 1
                        self.stats["position_change_count"] += 1

                
        return self._summarize()

    def _summarize(self):
        print("\n" + "="*50)
        print("🔍 POLICY ACTION PATH DIAGNOSTIC REPORT")
        print("="*50)
        total_steps = max(1, self.stats["total_steps"])
        hold_pct = self.stats["hold_count"] / total_steps * 100
        
        for k, v in self.stats.items():
            if k == "raw_actions":
                print(f"raw_action_distribution....... {v}")
            else:
                print(f"{k:.<30} {v}")
        
        print(f"hold_pct...................... {hold_pct:.2f}%")

        # Diagnosis logic
        res = "CASE_D_ENV_DOES_NOT_COUNT_TRADES_CORRECTLY"
        
        total_signals = self.stats["buy_signal_count"] + self.stats["sell_signal_count"]
        if hold_pct > 98 and self.stats["executed_trade_count"] == 0:
            res = "CASE_A_MODEL_ALWAYS_HOLD"
        elif total_signals > 0 and self.stats["executed_trade_count"] == 0:
            res = "CASE_B_ACTIONS_BLOCKED_BY_ENV_OR_RISK"
        elif self.stats["executed_trade_count"] > 0:
            res = "CASE_OK_POLICY_TRADES"
            
        print(f"\nFINAL DIAGNOSIS: {res}")
        print("="*50)
        return res


if __name__ == "__main__":
    # 1. Load Data
    logger.info("Loading evaluation data for FPT...")
    loader = DataLoader(ticker="FPT")
    # Fetch data and build features
    # Note: Using build_features and normalize to match training observation space
    df = loader.fetch_or_load().build_features().normalize().get_processed_df()
    
    # 2. Define Model Path
    # User requested 'checkpoints/ppo_fpt_intraday.zip'
    model_path = "checkpoints/ppo_fpt_intraday.zip"
    
    # Adjust path if needed
    if not os.path.exists(model_path):
        # Check alternative common locations
        alt_paths = [
            "agents/saved_models/ppo_fpt_intraday.zip",
            "agents/saved_models/ppo_stock_v1.zip",
            "agents/saved_models/ppo_vn30_multi_stage4.zip",
            "agents/saved_models/latest.zip",
            "agents/saved_models/best_sharpe.zip"
        ]
        for p in alt_paths:
            if os.path.exists(p):
                model_path = p
                logger.info(f"Found alternative checkpoint at: {model_path}")
                break

    # 3. Run Diagnostic
    diagnostic = ActionPathDiagnostic(model_path=model_path, df=df)
    verdict = diagnostic.run()
    
    # Write to log file for persistency
    os.makedirs("logs", exist_ok=True)
    with open("logs/action_path_diagnostic.txt", "w") as f:
        f.write(f"Verdict: {verdict}\n")
        f.write(str(diagnostic.stats))
