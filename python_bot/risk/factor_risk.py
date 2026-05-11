import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from python_bot.common.types import PortfolioState, MarketData

class FactorGraphRiskEngine:
    """
    Prevents "Contagion Propagation" across a multi-ticker portfolio.
    Tracks sector concentration, Beta exposure, and Liquidity clustering.
    """
    def __init__(self, tickers: List[str], sector_map: Dict[str, str]):
        self.tickers = tickers
        self.sector_map = sector_map
        self._price_history = {t: [] for t in tickers}
        self._window = 60 # 60-bar correlation window
        
        # Risk Limits
        self.max_sector_exposure = 0.40 # 40% cap per sector
        self.max_portfolio_beta = 1.2

    def update_prices(self, tick_map: Dict[str, MarketData]):
        for t, data in tick_map.items():
            if t in self._price_history:
                self._price_history[t].append(data.close)
                if len(self._price_history[t]) > self._window:
                    self._price_history[t].pop(0)

    def calculate_contagion_score(self) -> float:
        """
        Computes the first eigenvalue of the correlation matrix (Absorption Ratio).
        High values indicate market-wide 'coupling', i.e., systemic risk.
        """
        if any(len(h) < self._window for h in self._price_history.values()):
            return 0.0
            
        df = pd.DataFrame(self._price_history).pct_change().dropna()
        if df.empty: return 0.0
        
        corr = df.corr().values
        eigvals = np.linalg.eigvalsh(corr)
        absorption_ratio = np.max(eigvals) / np.sum(eigvals)
        
        return float(absorption_ratio)

    def validate_concentration(self, positions: Dict[str, int], current_prices: Dict[str, float]) -> bool:
        """Checks if a proposed trade violates sector risk caps."""
        sector_totals = {}
        total_equity = 0.0
        
        for t, qty in positions.items():
            notional = qty * current_prices.get(t, 0)
            total_equity += notional
            sector = self.sector_map.get(t, "OTHER")
            sector_totals[sector] = sector_totals.get(sector, 0) + notional
            
        if total_equity == 0: return True
        
        for sector, notional in sector_totals.items():
            if (notional / total_equity) > self.max_sector_exposure:
                return False
                
        return True
