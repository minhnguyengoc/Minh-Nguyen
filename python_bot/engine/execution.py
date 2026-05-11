import numpy as np
from typing import Optional, List, Dict, Tuple
from python_bot.common.types import OrderRequest, FillEvent, OrderSide, OrderType, MarketData
import uuid

from python_bot.engine.impact import InstitutionalImpactModel

class HybridQueueReactiveExecutionSimulator:
    """
    Institutional-Grade Microstructure Simulator.
    Solves "Liquidity Hallucination" and "Queue Hubris".
    Models: 
    - Full & Partial fills
    - Adverse selection (Toxic fills)
    - Participation-based impact
    """
    def __init__(self, slip_beta: float = 0.5, impact_model: Optional[InstitutionalImpactModel] = None):
        self._pending_orders: Dict[str, Dict] = {}
        self.slip_beta = slip_beta
        self.impact_model = impact_model or InstitutionalImpactModel()
        self._rng = np.random.default_rng(42)

    def submit(self, order: OrderRequest, data: MarketData):
        order_id = str(uuid.uuid4())
        
        # 1. Queue Position with "Back of the Queue" bias
        depth = 0
        if order.side == OrderSide.BUY:
            depth = sum([q for p, q in (data.bid_depth or []) if p >= (order.price or data.close)])
        else:
            depth = sum([q for p, q in (data.ask_depth or []) if p <= (order.price or data.close)])
        
        if depth == 0:
            depth = data.volume * self._rng.uniform(0.2, 0.6)
            
        # Institutional Invariant: You are never at the front of the queue
        depth *= 1.2 

        self._pending_orders[order_id] = {
            'order': order,
            'queue_pos': depth,
            'filled': 0,
            'age': 0
        }
        return order_id

    def step(self, data: MarketData) -> List[FillEvent]:
        fills = []
        finished = []
        
        for oid, state in self._pending_orders.items():
            order = state['order']
            state['age'] += 1
            
            # Adverse Selection Logic: Fills are more likely when the market is moving against you.
            # If we are BUYING and price is DROPPING, queue decay is slower (Adverse).
            # If we are BUYING and price is RISING, queue decay is faster but entry is harder.
            
            price_reached = False
            if order.side == OrderSide.BUY:
                if order.order_type == OrderType.MARKET or (order.price and data.low <= order.price):
                    price_reached = True
            else:
                if order.order_type == OrderType.MARKET or (order.price and data.high >= order.price):
                    price_reached = True
                    
            if not price_reached: continue

            # Directional Adverse Selection Model:
            # Decay = Vol * Participation_Rate * Pressure_Bias
            # If we are BUYING and price is DROPPING (data.close < data.open), 
            # it means there is heavy sell pressure -> Our queue decays SLOWER because 
            # we are "catching the knife" (Toxic Fill potential high).
            # If we are BUYING and price is RISING, queue decays FASTER (but entry is harder).
            
            pressure_bias = 1.0
            if order.side == OrderSide.BUY:
                if data.close < data.open: # Bullish trap / Falling knife
                    pressure_bias = 0.5 # Fill is harder (Toxic Flow)
                else:
                    pressure_bias = 1.5 # Fill is easier (Chasing)
            else:
                if data.close > data.open:
                    pressure_bias = 0.5
                else:
                    pressure_bias = 1.5
                    
            decay = data.volume * self._rng.uniform(0.05, 0.1) * pressure_bias
            state['queue_pos'] -= decay
            
            if state['queue_pos'] <= 0 or order.order_type == OrderType.MARKET:
                # COMPUTE PARTIAL OR FULL FILL
                remaining = order.quantity - state['filled']
                # Partial fill probability (higher for large orders relative to interval volume)
                fill_qty = remaining
                if remaining > (data.volume * 0.1):
                    fill_qty = int(remaining * self._rng.uniform(0.3, 0.7))
                
                if fill_qty > 0:
                    # Impact & Slippage
                    f_price, _ = self.impact_model.calculate_fill_impact(
                        fill_qty, order.side, data.volume, data.close
                    )
                    
                    fill_event = FillEvent(
                        order_id=oid, symbol=order.symbol, side=order.side,
                        fill_quantity=fill_qty, fill_price=f_price,
                        fee=fill_qty * f_price * 0.0003, # Tier-1 fee
                        timestamp=data.timestamp
                    )
                    fills.append(fill_event)
                    state['filled'] += fill_qty
                    
                if state['filled'] >= order.quantity:
                    finished.append(oid)
                    
        for oid in finished:
            del self._pending_orders[oid]
            
        return fills


# Alias for backward compatibility with older environment implementations
QueueFillSimulator = HybridQueueReactiveExecutionSimulator


class DriftMonitor:
    """
    Statistical Monitoring of State Distribution.
    Detects when market features deviate significantly from training distribution.
    """
    def __init__(self, baseline_dist: np.ndarray, threshold: float = 0.2):
        self.baseline = baseline_dist
        self.threshold = threshold
        self._current_window = []
        self._window_size = 500

    def push(self, vector: np.ndarray):
        self._current_window.append(vector)
        if len(self._current_window) > self._window_size:
            self._current_window.pop(0)

    def calculate_psi(self) -> float:
        """Population Stability Index (PSI) proxy."""
        if len(self._current_window) < 100: return 0.0
        
        curr = np.mean(self._current_window, axis=0)
        base = np.mean(self.baseline, axis=0)
        
        # Simplified vector drift: Normalized Euclidean Distance of means
        drift = np.linalg.norm(curr - base) / (np.linalg.norm(base) + 1e-8)
        return float(drift)

    def is_drifted(self) -> bool:
        return self.calculate_psi() > self.threshold
