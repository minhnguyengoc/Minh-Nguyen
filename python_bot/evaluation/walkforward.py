import sys
import os

# Ensure the project root is in the python path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Any
from python_bot.market_env import VNStockTradingEnv
from python_bot.evaluation.policy_behavior import PolicyBehaviorAnalyzer

class WalkForwardValidator:
    """
    Executes Institutional Walk-Forward Validation.
    Prevents temporal overfitting and regime-fragility.
    """
    def __init__(self, df: pd.DataFrame, folds: int = 4):
        self.df = df
        self.folds = folds
        self.logger = logging.getLogger("WalkForward")

    def run_validation(self, agent) -> Dict[str, Any]:
        """Performs rolling walk-forward test (Out-of-sample)."""
        results = []
        data_len = len(self.df)
        fold_len = data_len // (self.folds + 1)
        
        self.logger.info(f"🚀 Starting {self.folds}-Fold Walk-Forward Validation...")
        
        for f in range(self.folds):
            # Anchored Walk-forward
            start_idx = (f + 1) * fold_len
            end_idx = min((f + 2) * fold_len, data_len)
            
            test_df = self.df.iloc[start_idx:end_idx].copy()
            
            env = VNStockTradingEnv(df=test_df)
            analyzer = PolicyBehaviorAnalyzer(history_len=len(test_df))
            
            obs, _ = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                analyzer.record_step(
                    action=action,
                    reward=reward,
                    components=info.get("reward_components", {}),
                    position=info["shares"],
                    step=env.current_step
                )
                
                total_reward += reward
                done = terminated or truncated
                
            audit = analyzer.analyze()
            
            # Additional Institutional Metrics
            equity_curve = analyzer._rewards # This is actually incremental reward, let's use actual equity tracking
            # Wait, the analyzer doesn't track equity directly. Let's use a local equity curve.
            
            # Benchmark: Buy-and-Hold (after costs)
            initial_price = test_df.iloc[0]['close']
            final_price = test_df.iloc[-1]['close']
            bh_return = (final_price / initial_price) - 1.0
            bh_return -= 0.004 # Approximate fee+tax for B&H
            
            fold_return = (info["equity"] - 100_000_000) / 100_000_000
            
            # Metrics for this fold
            wins = sum(1 for r in analyzer._rewards if r > 0.01) # Simple proxy for wins
            trades = sum(1 for a in analyzer._actions if a in [1, 2])
            win_rate = wins / max(trades, 1)

            results.append({
                "fold": f,
                "reward": total_reward,
                "equity": info["equity"],
                "return": fold_return,
                "benchmark_return": bh_return,
                "excess_return": fold_return - bh_return,
                "audit_status": audit["status"],
                "turnover": audit.get("trades_per_1k", 0),
                "avg_hold": audit.get("avg_holding_bars", 0),
                "win_rate": win_rate
            })
            self.logger.info(f"✅ Fold {f} | Return: {fold_return*100:.2f}% | B&H: {bh_return*100:.2f}% | Status: {audit['status']}")
            
        avg_ret = np.mean([r['return'] for r in results])
        avg_bh = np.mean([r['benchmark_return'] for r in results])
        
        return {
            "stable": all(r['audit_status'] == "HEALTHY" for r in results) and avg_ret > 0,
            "fold_results": results,
            "avg_return": avg_ret,
            "avg_benchmark": avg_bh,
            "avg_excess": avg_ret - avg_bh
        }

if __name__ == "__main__":
    import argparse
    from python_bot.ppo_agent import PPOAgent
    from python_bot.data_loader import DataLoader
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="agents/saved_models/ppo_vn30_multi_stage4")
    parser.add_argument("--ticker", type=str, default="FPT")
    args = parser.parse_args()
    
    logging.info(f"💾 Loading data for {args.ticker}...")
    loader = DataLoader(ticker=args.ticker)
    df = loader.fetch_or_load().build_features().normalize().get_processed_df()
    
    # Setup dummy env for agent init
    dummy_env = VNStockTradingEnv(df=df.iloc[:100])
    agent = PPOAgent(env=dummy_env, model_name=os.path.basename(args.model_path), model_dir=os.path.dirname(args.model_path))
    
    if agent.load():
        validator = WalkForwardValidator(df=df)
        report = validator.run_validation(agent)
        
        print("\n" + "="*50)
        print("🏛️ INSTITUTIONAL WALK-FORWARD REPORT")
        print(f"Stable: {report['stable']}")
        print(f"Avg Return: {report['avg_return']*100:.2f}%")
        print(f"Avg Benchmark (B&H): {report['avg_benchmark']*100:.2f}%")
        print(f"Excess Return: {report['avg_excess']*100:.2f}%")
        print("="*50)
        
        # Save report
        os.makedirs("logs", exist_ok=True)
        with open("logs/walkforward_report.json", "w") as f:
            import json
            json.dump(report, f, indent=2)
    else:
        logging.error("Failed to load model for validation.")
