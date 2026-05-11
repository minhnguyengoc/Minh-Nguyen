import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np

# Ensure the project root is in the python path
def _setup_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root

_project_root = _setup_path()

from python_bot.data_loader import DataLoader
from python_bot.market_env import VNStockTradingEnv
from python_bot.ppo_agent import PPOAgent
from python_bot.evaluation.walkforward import WalkForwardValidator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Institutional Walk-Forward Validation Suite")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker (e.g. FPT)")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--folds", type=int, default=3, help="Number of rolling folds")
    args = parser.parse_args()

    logging.info(f"🧪 Starting Validation Suite for {args.ticker} | Model: {args.model}")

    # 1. Load Data
    loader = DataLoader(ticker=args.ticker)
    loader.fetch_or_load().build_features().normalize(train_ratio=0.8)
    states, ohlcv, timestamps, features, _ = loader.get_full_matrix()

    # 2. Initialize Agent (Temporary env for initialization)
    dummy_env = VNStockTradingEnv(states[:10], ohlcv[:10], timestamps[:10])
    agent = PPOAgent(env=dummy_env, model_dir=os.path.dirname(args.model))
    
    if not agent.load(args.model):
        logging.error("❌ Failed to load model. Ensure path exists.")
        return

    # 3. Perform Walk-Forward Validation
    validator = WalkForwardValidator(states, ohlcv, timestamps, folds=args.folds)
    results = validator.run_validation(agent)

    # 4. Summary Output
    logging.info("\n" + "="*50)
    logging.info(f"🏆 VALIDATION COMPLETE: {args.ticker}")
    logging.info(f"Stability: {'✅ PASSED' if results['stable'] else '❌ FAILED'}")
    logging.info(f"Avg Out-of-Sample Return: {results['avg_return']*100:.2f}%")
    logging.info("-" * 50)
    
    for f_res in results['fold_results']:
        logging.info(f"Fold {f_res['fold']}: Return {f_res['return']*100:6.2f}% | Turnover {f_res['turnover']:6.1f} | Audit: {f_res['audit_status']}")
    
    logging.info("="*50)

if __name__ == "__main__":
    main()
