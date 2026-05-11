from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
import numpy as np

class MarketRegime(Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    AUCTION = "AUCTION"
    PANIC = "PANIC"
    LIQUIDITY_CRUNCH = "LIQUIDITY_CRUNCH"
    CLOSED = "CLOSED"

class SessionState(Enum):
    ATO = "ATO"
    CONTINUOUS_MORNING = "CONTINUOUS_MORNING"
    LUNCH_BREAK = "LUNCH_BREAK"
    CONTINUOUS_AFTERNOON = "CONTINUOUS_AFTERNOON"
    ATC = "ATC"
    CLOSED = "CLOSED"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

class ActionDirective(Enum):
    HOLD = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3

class MarketData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    symbol: str
    timestamp: datetime # Exchange Time
    received_at: datetime # Local Receipt Time
    event_id: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    bid_depth: Optional[List[Tuple[float, float]]] = None
    ask_depth: Optional[List[Tuple[float, float]]] = None
    limit_up: bool = False
    limit_down: bool = False

class OrderRequest(BaseModel):
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    timestamp: datetime

class FillEvent(BaseModel):
    order_id: str
    symbol: str
    side: OrderSide
    fill_quantity: int
    fill_price: float
    fee: float
    timestamp: datetime

class PortfolioState(BaseModel):
    available_cash: float
    locked_shares_t2: int
    position_quantity: int
    average_entry_price: float
    unrealized_pnl: float
    realized_pnl_today: float

class ObservationMetadata(BaseModel):
    is_session_active: bool
    session_state: SessionState
    regime: MarketRegime
    is_stale: bool
    kill_switch: bool
    drift_score: float
    confidence_score: float
    policy_abstain: bool
    latency_ms: float
    event_sequence_id: int

class StandardizedObservation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    vector: np.ndarray
    metadata: ObservationMetadata
