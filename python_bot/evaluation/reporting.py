import sys
import os

# Ensure the project root is in the python path
def _setup_path():
    current_file = os.path.abspath(__file__)
    # /python_bot/evaluation/reporting.py -> / (project root)
    root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root

_project_root = _setup_path()

import json
import logging
from typing import Dict, Any, List
import datetime as dt

class InstitutionalReporter:
    """ Generates executive summaries of agent behavioral audits and performance. """
    
    def __init__(self, ticker: str, log_dir: str = "logs/reports"):
        self.ticker = ticker
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger("InstitutionalReporter")

    def generate_report(self, backtest_metrics: Dict[str, Any], behavioral_audit: Dict[str, Any], plot_path: str = None) -> str:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = behavioral_audit.get("status", "UNKNOWN")
        
        # Determine relative path for plot if provided
        plot_md = ""
        if plot_path:
            # We assume the user might view this in a repo where logs/reports and logs/plots are siblings
            rel_plot = os.path.join("../plots", os.path.basename(plot_path))
            plot_md = f"\n## 5. PERFORMANCE CHART\n![Performance Chart]({rel_plot})\n"

        report = f"""# INSTITUTIONAL TRADING AUDIT: {self.ticker}
Generated: {timestamp}
Status: {status}

## 1. PERFORMANCE SUMMARY
- Final Equity: {backtest_metrics.get('equity', 0):,.0f} VND
- Total Return: {backtest_metrics.get('total_return', 0):.2f}%
- Max Drawdown: {backtest_metrics.get('max_drawdown', 0):.2f}%
- Sharpe Ratio: {backtest_metrics.get('sharpe', 0):.2f}
- Sortino Ratio: {backtest_metrics.get('sortino', 0):.2f}
- Total Trades: {backtest_metrics.get('trade_count', 0)}

## 2. BEHAVIORAL DIAGNOSTIC
- **Entropy**: {behavioral_audit.get('entropy', 0):.4f} (Measure of action variety)
- **Turnover**: {behavioral_audit.get('trades_per_1k', 0):.1f} trades / 1k bars
- **Average Hold**: {behavioral_audit.get('avg_holding_bars', 0):.1f} bars

### Anomaly Flags
{self._format_flags(behavioral_audit.get('anomaly_flags', []))}

## 3. COMPONENT ATTRIBUTION (Dominance)
{self._format_dominance(behavioral_audit.get('reward_dominance', {}))}

## 4. EXECUTIVE RECOMMENDATION
{self._generate_recommendation(status, behavioral_audit)}
{plot_md}
"""
        filename = f"audit_{self.ticker}_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.md"
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            f.write(report)
        
        self.logger.info(f"📄 Audit Report generated: {path}")
        return path

    def _format_flags(self, flags: List[str]) -> str:
        if not flags: return "✅ No anomalies detected. Policy adheres to institutional stability constraints."
        return "\n".join([f"- ⚠️ {flag}" for flag in flags])

    def _format_dominance(self, dominance: Dict[str, float]) -> str:
        lines = []
        for k, v in dominance.items():
            bar = "█" * int(v * 20)
            lines.append(f"- {k:15}: {bar} {v*100:.1f}%")
        return "\n".join(lines)

    def _generate_recommendation(self, status: str, audit: Dict[str, Any]) -> str:
        if status == "HEALTHY":
            return "PROCEED. The policy shows stable holding patterns and diversified action entropy. Suitable for paper trading."
        
        recs = ["ACTION REQUIRED:"]
        flags = audit.get("anomaly_flags", [])
        if "HYPER_TURNOVER" in flags:
            recs.append("- CRITICAL: Turnover exceeding institutional limits. Increase 'trade_cost_penalty' (current default 0.05) or add turnover cooling.")
        if "MICRO_SCALPING_EXPLOIT" in flags:
            recs.append("- WARNING: Micro-scalping detected. This usually indicates look-ahead bias or missing slippage models.")
        if "REWARD_SIGNAL_MONOPOLY" in flags:
            recs.append("- DANGER: Reward signal dominated by a single component. Adjust weights in RewardEngine to ensure PnL is the primary driver.")
            recs.append("- Advice: Check if drawdown_cost is too high, causing agent to avoid all risk by sitting in cash or oscillating.")
            
        return "\n".join(recs)
