# plot_training.py
import sys
import os

# Ensure the project root is in the python path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrainingVisualizer:
    """Institutional-grade dashboard generator for RL trading metrics."""
    
    def __init__(self, log_dir: str = "logs/metrics", save_dir: str = "logs/visuals"):
        self.log_dir = log_dir
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        sns.set_theme(style="darkgrid")
        
    def _vnd_formatter(self, x, pos=None):
        return f"{x:,.0f}₫"
    
    def load_and_plot(self, csv_path: Optional[str] = None, json_path: Optional[str] = None) -> plt.Figure:
        csv_path = csv_path or os.path.join(self.log_dir, "training_metrics.csv")
        json_path = json_path or os.path.join(self.log_dir, "final_metrics.json")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")
            
        df = pd.read_csv(csv_path)
        self._validate_and_clean(df)
        
        # Compute drawdown if missing
        if 'drawdown_pct' not in df.columns:
            peak = df['equity'].cummax()
            df['drawdown_pct'] = ((peak - df['equity']) / peak) * 100.0
            
        fig, axes = plt.subplots(3, 2, figsize=(18, 16))
        fig.suptitle("RL Trading Pipeline | Institutional Dashboard", fontsize=22, fontweight='bold', y=0.98)
        
        self._plot_equity_curve(df, axes[0, 0])
        self._plot_drawdown(df, axes[0, 1])
        self._plot_reward_trend(df, axes[1, 0])
        self._plot_trade_activity(df, axes[1, 1])
        self._plot_position_exposure(df, axes[2, 0])
        self._plot_final_summary(json_path, axes[2, 1])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(self.save_dir, "rl_training_dashboard.png")
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        plt.close()
        logging.info(f"✅ Enhanced Dashboard exported to: {save_path}")
        return fig

    def _validate_and_clean(self, df: pd.DataFrame):
        required = ['step', 'equity', 'trade_count', 'position']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        df.dropna(subset=['equity', 'step'], inplace=True)
        df.sort_values('step', inplace=True)
        df.reset_index(drop=True, inplace=True)

    def _plot_equity_curve(self, df: pd.DataFrame, ax: plt.Axes):
        ax.plot(df['step'], df['equity'], color='#2ecc71', linewidth=2.5)
        ax.set_title("Portfolio Equity (VND)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Training Steps")
        ax.yaxis.set_major_formatter(FuncFormatter(self._vnd_formatter))
        ax.grid(True, alpha=0.3)

    def _plot_drawdown(self, df: pd.DataFrame, ax: plt.Axes):
        ax.fill_between(df['step'], df['drawdown_pct'], 0, color='#e74c3c', alpha=0.4)
        ax.plot(df['step'], df['drawdown_pct'], color='#c0392b', linewidth=1.5)
        ax.set_title("Max Drawdown Evolution (%)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Drawdown %")
        ax.axhline(30, color='orange', linestyle='--', alpha=0.6, label='Risk Limit')
        ax.invert_yaxis() # Traditional drawdown view
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_reward_trend(self, df: pd.DataFrame, ax: plt.Axes):
        if 'reward' in df.columns:
            # Moving average for smoothing the noisy reward signals
            smoothed = df['reward'].rolling(window=min(20, len(df))).mean()
            ax.plot(df['step'], df['reward'], color='#9b59b6', alpha=0.2, label='Raw Reward')
            ax.plot(df['step'], smoothed, color='#8e44ad', linewidth=2, label='Avg Reward (MA)')
            ax.set_title("Learning Progress (Reward Trend)", fontsize=14, fontweight='bold')
            ax.set_xlabel("Training Steps")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Reward data missing", transform=ax.transAxes)
        ax.grid(True, alpha=0.3)

    def _plot_trade_activity(self, df: pd.DataFrame, ax: plt.Axes):
        ax.step(df['step'], df['trade_count'], where='post', color='#3498db', linewidth=2)
        ax.set_title("Trade Frequency (Turnover)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Cumulative Trades")
        ax.grid(True, alpha=0.3)

    def _plot_position_exposure(self, df: pd.DataFrame, ax: plt.Axes):
        # 0:Hold, 1:Long, 2:Short, 3:Close (but environment stores position quantity)
        # We simplify to Long (>0), Short (<0), Cash (0)
        pos = df['position']
        longs = (pos > 0).astype(int)
        shorts = (pos < 0).astype(int)
        cash = (pos == 0).astype(int)
        
        ax.stackplot(df['step'], cash, longs, shorts, labels=['Cash', 'Long', 'Short'], 
                    colors=['#bdc3c7', '#2ecc71', '#e74c3c'], alpha=0.7)
        ax.set_title("Agent Exposure Bias", fontsize=14, fontweight='bold')
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Count (Binary)")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.2)

    def _plot_final_summary(self, json_path: str, ax: plt.Axes):
        ax.axis('off')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                metrics = json.load(f)
            text = "FINAL EVALUATION METRICS\n" + "="*30 + "\n"
            for k, v in metrics.items():
                if 'equity' in k: text += f"{k}: {v:,.0f}₫\n"
                elif 'rate' in k or 'drawdown' in k: text += f"{k}: {v*100:.2f}%\n"
                else: text += f"{k}: {v:.2f}\n"
            ax.text(0.1, 0.8, text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', fontfamily='monospace')
        else:
            ax.text(0.1, 0.5, "Final metrics not found.\nRun trainer.py first.", 
                    transform=ax.transAxes, fontsize=12, color='gray')

if __name__ == "__main__":
    viz = TrainingVisualizer()
    viz.load_and_plot()
