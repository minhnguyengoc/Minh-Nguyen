import sys
import os

# Ensure the project root is in the python path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from python_bot.data_loader import DataLoader
from python_bot.market_env import VNStockTradingEnv
from python_bot.ppo_agent import PPOAgent
from python_bot.evaluation.policy_behavior import PolicyBehaviorAnalyzer
from python_bot.evaluation.reporting import InstitutionalReporter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_backtest(ticker: str = "FPT", model_path: str = "checkpoints/ppo_fpt_intraday"):
    """
    Executes a high-fidelity backtest on out-of-sample data.
    """
    logging.info(f"🔍 Initializing Backtest for {ticker}...")
    
    # 1. Load Data
    loader = DataLoader(ticker=ticker)
    loader.fetch_or_load().build_features().normalize(train_ratio=0.8)
    states, ohlcv, timestamps, features, _ = loader.get_full_matrix()
    
    # 2. Split for Test (The last 20% the model NEVER saw)
    split_idx = int(len(states) * 0.8)
    test_states = states[split_idx:]
    test_ohlcv = ohlcv[split_idx:]
    test_timestamps = timestamps[split_idx:]
    
    logging.info(f"📊 Test Data: {len(test_states)} rows")
    
    # 3. Initialize Test Env
    env = VNStockTradingEnv(
        states=test_states,
        ohlcv=test_ohlcv,
        timestamps=test_timestamps,
        initial_capital=100_000_000.0
    )
    
    # 4. Load Agent
    agent = PPOAgent(
        env=env, 
        model_dir=os.path.dirname(model_path), 
        model_name=os.path.basename(model_path),
        paper_trading=False
    )
    
    if not agent.load():
        logging.error(f"❌ Could not load model from {model_path}.")
        return

    # 5. Run Episode with Behavioral Analyzer
    logging.info("🚀 Running Backtest with Behavioral Audit...")
    obs, info = env.reset()
    done = False
    
    history = []
    analyzer = PolicyBehaviorAnalyzer(history_len=len(test_states))
    
    while not done:
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record behavioral data
        analyzer.record_step(
            action=action,
            reward=reward,
            components=info.get("reward_components", {}),
            position=info["position"],
            step=env.current_step
        )
        
        history.append({
            "timestamp": test_timestamps[env.current_step],
            "price": test_ohlcv[env.current_step, 3],
            "action": action,
            "equity": info["equity"],
            "position": info["position"],
            "reward": reward
        })
        
        done = terminated or truncated
        
    # 6. Post-Process Results
    df_results = pd.DataFrame(history)
    output_dir = "logs/backtest"
    os.makedirs(output_dir, exist_ok=True)
    
    res_path = os.path.join(output_dir, f"backtest_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
    df_results.to_csv(res_path, index=False)
    
    # Calculate Metrics
    from python_bot.evaluation.institutional_metrics import InstitutionalMetrics
    
    # Calculate returns and metrics
    returns = df_results["reward"].values / 10.0 # Approximate minutely return normalization
    inst_metrics_calc = InstitutionalMetrics(returns.tolist())
    stats = inst_metrics_calc.calculate_pnl_stats()
    
    metrics = {
        "equity": info["equity"],
        "total_return": stats["total_return"] * 100,
        "max_drawdown": stats["max_drawdown"] * 100,
        "sharpe": stats["sharpe_ratio"],
        "sortino": stats["sortino_ratio"],
        "trade_count": info["trade_count"]
    }
    
    # Behavioral Audit
    audit = analyzer.analyze()
    
    # Visual Charting
    from python_bot.evaluation.plotting import plot_backtest_results
    plot_path = plot_backtest_results(df_results, ticker)
    
    # Generate MD Report
    reporter = InstitutionalReporter(ticker=ticker)
    report_path = reporter.generate_report(metrics, audit, plot_path)
    
    logging.info("\n" + "="*45)
    logging.info(f"🏁 BACKTEST COMPLETE: {ticker}")
    logging.info(f"💰 Final Equity: {metrics['equity']:,.0f} VND")
    logging.info(f"📈 Total Return: {metrics['total_return']:.2f}%")
    logging.info(f"📉 Max Drawdown: {metrics['max_drawdown']:.2f}%")
    logging.info(f"📊 Sharpe: {metrics['sharpe']:.2f} | Sortino: {metrics['sortino']:.2f}")
    logging.info("-" * 45)
    logging.info("🛡️ INSTITUTIONAL BEHAVIORAL AUDIT")
    logging.info(f"Status: {audit.get('status')}")
    logging.info(f"Turnover: {audit.get('trades_per_1k', 0):.1f} per 1k steps")
    logging.info(f"Avg Hold: {audit.get('avg_holding_bars', 0):.1f} bars")
    logging.info(f"Flags: {audit.get('anomaly_flags', [])}")
    logging.info(f"📑 Audit Report: {report_path}")
    logging.info(f"📈 Performance Chart: {plot_path}")
    logging.info("="*45)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="FPT")
    parser.add_argument("--model", type=str, default="checkpoints/ppo_fpt_intraday")
    args = parser.parse_args()
    
    run_backtest(ticker=args.ticker, model_path=args.model)
