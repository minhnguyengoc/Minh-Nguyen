import sys
import os

def _setup_path():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root

_setup_path()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def plot_backtest_results(df: pd.DataFrame, ticker: str, report_dir: str = "logs/plots"):
    """
    Generates a high-quality dual-axis chart:
    Top: Price + Buy/Sell Markers
    Bottom: Equity Curve + Drawdown
    """
    os.makedirs(report_dir, exist_ok=True)
    sns.set_theme(style="darkgrid")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. Price chart
    ax1.plot(df.index, df['price'], label='Price', color='black', alpha=0.6)
    
    # Markers (0:Hold, 1:Long, 2:Short, 3:Close)
    longs = df[df['action'] == 1]
    shorts = df[df['action'] == 2]
    closes = df[df['action'] == 3]
    
    ax1.scatter(longs.index, longs['price'], marker='^', color='green', s=100, label='Long Enter', zorder=5)
    ax1.scatter(shorts.index, shorts['price'], marker='v', color='red', s=100, label='Short Enter', zorder=5)
    ax1.scatter(closes.index, closes['price'], marker='x', color='blue', s=80, label='Close', zorder=5)
    
    ax1.set_title(f"Institutional Performance: {ticker}", fontsize=16)
    ax1.set_ylabel("Price (VND)")
    ax1.legend()

    # 2. Equity Curve
    ax2.plot(df.index, df['equity'], color='blue', label='Portfolio Value')
    ax2.set_ylabel("Equity (VND)")
    
    # Fill Drawdown
    peak = df['equity'].cummax()
    ax2.fill_between(df.index, df['equity'], peak, color='red', alpha=0.3, label='Drawdown')
    
    plt.xlabel("Step / Index")
    plt.tight_layout()
    
    filename = f"plot_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
    path = os.path.join(report_dir, filename)
    plt.savefig(path)
    plt.close()
    return path
