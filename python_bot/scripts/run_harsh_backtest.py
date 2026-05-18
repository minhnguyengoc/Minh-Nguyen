import os
import pandas as pd
import numpy as np
import logging
import argparse
from python_bot.data.history_loader_4y import HistoryLoader4Y
from python_bot.backtest.backtester import InstitutionalBacktester, CostMode
from python_bot.evaluation.nav_report import NAVReportGenerator
from python_bot.evaluation.entry_quality import EntryQualityAnalyzer
from python_bot.evaluation.policy_trade_diagnostics import PolicyTradeDiagnostics
from python_bot.ppo_agent import PPOAgent
from python_bot.market_env import VNStockTradingEnv

def run_stress_test(agent, df, mode: CostMode):
    """Runs a full backtest for a specific cost mode."""
    logging.info(f"🛡️  Starting Backtest [Mode: {mode.value}]")
    
    # Initialize components
    bt = InstitutionalBacktester(cost_mode=mode)
    diagnostics = PolicyTradeDiagnostics()
    quality = EntryQualityAnalyzer(df)
    
    # We use the environment to get observations matching the training state
    # but we control the account logic in the backtester for institutional accuracy.
    env = VNStockTradingEnv(df=df)
    obs, _ = env.reset()
    
    nav_series = []
    
    for i in range(len(df) - 1):
        # 1. Action from Agent
        action, _ = agent.model.predict(obs, deterministic=True)
        action = int(action)
        
        # 2. Execution in Backtester
        row = df.iloc[i]
        symbol = row['symbol']
        price = row['close']
        volume = row['volume']
        
        bt.execute_trade(ts=row['timestamp'], symbol=symbol, action=action, price=price, volume=volume, metadata={"regime": row.get('vol_regime')})
        
        # 3. Step Environment (For normalized observation only)
        obs, reward, done, _, info = env.step(action)
        
        # 4. NAV Accounting
        # Valuation based on current backtest positions
        current_equity = bt.cash
        for sym, shares in bt.positions.items():
            current_equity += shares * price
        nav_series.append(current_equity)
        
        diagnostics.record_action(action)
        if action in [1, 2]:
            quality.log_entry(i, symbol, "BUY" if action == 1 else "SELL", price)
            
        if done: break

    return pd.Series(nav_series), bt.trade_log, diagnostics.run_diagnostics(), quality.analyze()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="agents/saved_models/ppo_vn30_multi_stage4")
    parser.add_argument("--symbols", type=str, default="FPT,MWG,SSI")
    parser.add_argument("--mode", type=str, default="HELL")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # 1. Load Data
    symbols = args.symbols.split(',')
    loader = HistoryLoader4Y(symbols=symbols)
    try:
        df = loader.load_combined()
    except Exception as e:
        logging.error(f"FATAL: {e}")
        return

    # 2. Load Agent
    dummy_env = VNStockTradingEnv(df=df.head(100))
    agent = PPOAgent(env=dummy_env, model_name=os.path.basename(args.model_path), model_dir=os.path.dirname(args.model_path))
    if not agent.load():
        logging.error("Failed to load model checkpoint.")
        return

    # 3. Perform Stress Tests
    modes = [CostMode.NORMAL, CostMode.HELL]
    report_gen = NAVReportGenerator()
    
    results = {}
    for m in modes:
        nav, trades, diag, qual = run_stress_test(agent, df, m)
        metrics = report_gen.calculate_metrics(nav, trades)
        results[m] = {"metrics": metrics, "diag": diag, "qual": qual}
        report_gen.plot_nav(nav, symbol_name=f"{m.value}")

    # 4. Final Verdict
    v_hell = results[CostMode.HELL]
    v_norm = results[CostMode.NORMAL]
    
    print("\n" + "="*60)
    print("🏛️  INSTITUTIONAL STAGE 4.1 FINAL VERDICT")
    print("="*60)
    
    is_pass = True
    reason = "ALL_SYSTEMS_OPTIMAL"
    
    if v_hell['metrics']['total_return_pct'] <= 0:
        is_pass = False
        reason = "NEGATIVE_HELL_MODE_RETURN"
    elif v_hell['diag']['verdict'] == "FAIL":
        is_pass = False
        reason = f"POLICY_FAILURE: {v_hell['diag']['reasons']}"
    elif v_norm['metrics']['trade_count'] < 10:
        is_pass = False
        reason = "INSUFFICIENT_TRADE_ACTIVITY"
        
    print(f"VERDICT: {'🟢 PASS' if is_pass else '🔴 FAIL'}")
    print(f"REASON: {reason}")
    print("-" * 60)
    print(f"NORMAL Return: {v_norm['metrics']['total_return_pct']:.2f}% | Sharpe: {v_norm['metrics']['sharpe_ratio']:.2f}")
    print(f"HELL Return:   {v_hell['metrics']['total_return_pct']:.2f}% | Sharpe: {v_hell['metrics']['sharpe_ratio']:.2f}")
    print(f"Alpha Entry Score: {v_hell['qual'].get('entry_efficiency_score', 0):.4f}")
    print("-" * 60)
    
    # Haircut estimation
    haircut_ret = v_hell['metrics']['total_return_pct'] * 0.5
    print(f"🏦 Live Expectation (50% Haircut): {haircut_ret:.2f}% Net Return")
    print("="*60)

if __name__ == "__main__":
    main()
