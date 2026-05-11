# strategy_router.py
import os
import json
import logging
import gc
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Callable, Any
from python_bot.ppo_agent import PPOAgent
from python_bot.market_env import VNStockTradingEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StrategyRouter:
    """
    Institutional multi-ticker manager for RL trading pipeline.
    Handles isolated training, resource cleanup, lazy model loading, and unified inference routing.
    """
    def __init__(self, tickers: List[str], data_loader: Callable[[str], tuple],
                 model_dir: str = "agents/saved_models", device: str = "auto"):
        self.tickers = tickers
        self.data_loader = data_loader
        self.model_dir = model_dir
        self.device = device
        self.agents: Dict[str, PPOAgent] = {}
        self.performance: Dict[str, Dict[str, Any]] = {}
        os.makedirs(model_dir, exist_ok=True)
        
    def train_all(self, total_timesteps: int = 100_000, parallel: bool = False, 
                  reset_steps: bool = True, **kwargs) -> Dict[str, Dict]:
        """Train models across all tickers. Parallel=False recommended for single-GPU stability."""
        logging.info(f"🚀 Routing training for {len(self.tickers)} tickers: {self.tickers}")
        results = {}
        
        for i, ticker in enumerate(self.tickers):
            logging.info(f"▶️ [{i+1}/{len(self.tickers)}] Training {ticker}...")
            try:
                metrics = self._train_single_ticker(ticker, total_timesteps, reset_steps, **kwargs)
                results[ticker] = metrics
                self.performance[ticker] = metrics
            except Exception as e:
                logging.error(f"💥 Failed {ticker}: {e}")
                results[ticker] = {"error": str(e)}
            finally:
                self._cleanup_memory()
                
        self._save_performance_report()
        return results

    def _train_single_ticker(self, ticker: str, total_timesteps: int, reset_steps: bool, **kwargs) -> Dict:
        # 1. Load data
        states, closes, session_dates = self.data_loader(ticker)
        if len(states) < 100:
            raise ValueError(f"{ticker} has insufficient data ({len(states)} steps)")
            
        # 2. Init isolated env & agent
        env = VNStockTradingEnv(states, closes, session_dates)
        agent = PPOAgent(
            env=env, model_dir=self.model_dir, model_name=f"ppo_{ticker}_v1",
            seed=42, device=self.device, paper_trading=True
        )
        
        # 3. Train
        agent.train(total_timesteps=total_timesteps, reset_steps=reset_steps, **kwargs)
        metrics = agent.evaluate_policy(episodes=3)
        
        del env  # Explicit cleanup
        return metrics

    def predict(self, ticker: str, observation: np.ndarray, deterministic: bool = True) -> int:
        """Unified routing interface for live/paper trading."""
        if ticker not in self.agents:
            self._load_single_agent(ticker)
        if ticker not in self.agents:
            raise RuntimeError(f"Model for {ticker} not found. Run train_all() first.")
        return self.agents[ticker].predict(observation, deterministic=deterministic)

    def _load_single_agent(self, ticker: str):
        """Lazy load model into memory with hardware detection."""
        ticker_dir = os.path.join(self.model_dir, f"ppo_{ticker}_v1")
        zip_path = f"{ticker_dir}.zip"
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Checkpoint missing: {zip_path}")
            
        # Dummy env for shape validation (won't be used for training)
        dummy_states = np.zeros((1, 20), dtype=np.float32)
        dummy_closes = np.array([20.0, 21.0])
        dummy_dates = np.array([20240101, 20240101])
        temp_env = VNStockTradingEnv(dummy_states, dummy_closes, dummy_dates)
        
        self.agents[ticker] = PPOAgent(
            env=temp_env, model_dir=self.model_dir, model_name=f"ppo_{ticker}_v1",
            device=self.device
        )
        self.agents[ticker].load()
        logging.info(f"📦 Loaded model for {ticker} into memory.")

    def _cleanup_memory(self):
        """Aggressive VRAM/RAM cleanup between tickers."""
        self.agents.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        time.sleep(0.5)  # Allow OS to reclaim pages

    def _save_performance_report(self):
        report_path = os.path.join(self.model_dir, "multi_ticker_report.json")
        with open(report_path, 'w') as f:
            json.dump(self.performance, f, indent=2, default=str)
        logging.info(f"📊 Multi-ticker performance saved to: {report_path}")

# Example usage
if __name__ == "__main__":
    # Mock data loader matching your pipeline signature
    def mock_data_loader(ticker: str):
        n = 1500
        return (
            np.random.randn(n, 20).astype(np.float32),
            np.linspace(18.0, 22.0, n),
            np.repeat(20240101, n)
        )
        
    router = StrategyRouter(
        tickers=["FPT", "VCB", "HPG"],
        data_loader=mock_data_loader,
        device="auto"
    )
    
    # Sequential training (safest for single GPU)
    results = router.train_all(total_timesteps=30_000, parallel=False)
    print("✅ Routing complete. Models ready for unified predict(ticker, obs)")
