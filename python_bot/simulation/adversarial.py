import numpy as np
from typing import List, Dict, Optional
from python_bot.common.types import MarketData
import datetime as dt

class AdversarialScenarioEngine:
    """
    Generates Market Stress Scenarios to test policy robustness.
    Injects synthetic shocks into real historical streams.
    """
    def __init__(self):
        pass

    def inject_flash_crash(self, data_stream: List[MarketData], magnitude: float = 0.05) -> List[MarketData]:
        """Simulates a sudden 5% drop and 3% recovery."""
        if len(data_stream) < 50: return data_stream
        
        mid = len(data_stream) // 2
        new_stream = [d.model_dump() for d in data_stream]
        
        # Crash
        price = new_stream[mid]['close']
        for i in range(mid, mid + 5):
            new_stream[i]['close'] *= (1.0 - magnitude * (i - mid + 1) / 5)
            new_stream[i]['low'] = new_stream[i]['close'] * 0.99
            new_stream[i]['volume'] *= 5.0 # High volume during crash
            
        return [MarketData(**d) for d in new_stream]

    def inject_latency_spike(self, data_stream: List[MarketData], delay_seconds: float = 5.0) -> List[MarketData]:
        """Simulates an exchange packet lag."""
        new_stream = [d.model_dump() for d in data_stream]
        for d in new_stream:
            d['received_at'] = d['timestamp'] + dt.timedelta(seconds=delay_seconds)
        return [MarketData(**d) for d in new_stream]

    def inject_liquidity_evaporation(self, data_stream: List[MarketData]) -> List[MarketData]:
        """Withdraws LOB depth to simulate high slippage."""
        new_stream = [d.model_dump() for d in data_stream]
        for d in new_stream:
            d['bid_depth'] = [(p, q * 0.1) for p, q in (d.get('bid_depth') or [])]
            d['ask_depth'] = [(p, q * 0.1) for p, q in (d.get('ask_depth') or [])]
            d['volume'] *= 0.1
        return [MarketData(**d) for d in new_stream]
