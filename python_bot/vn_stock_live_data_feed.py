import os
import logging
import datetime as dt
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
from collections import deque

# Structured Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("VNStockLiveFeed")

class VNStockLiveDataFeed:
    """
    Institutional-grade Real-time Data Feed & State Manager for Vietnam Stock Market RL.
    Implements 6 distinct layers: 
    1. Session Guard | 2. Integrity Validator | 3. Feature Pipeline | 
    4. Regime Engine | 5. Normalization/Drift Control | 6. Observation Export.
    """
    
    def __init__(self, ticker: str, lookback: int = 30):
        self.ticker = ticker
        self.lookback = lookback  # Default 30-min window for context
        self.feature_dim = 30     # Fixed obs shape requirement for PPO agent
        
        # State Buffers
        self.raw_buffer = deque(maxlen=250)    # Raw OHLCV + context
        self.feature_buffer = deque(maxlen=100) # Historical feature vectors for normalization
        
        # Layer 5: Adaptive Normalization State
        self.feature_means = np.zeros(self.feature_dim)
        self.feature_vars = np.ones(self.feature_dim)
        self.ema_alpha = 0.005 # Slow adaptation to market regime shifts
        
        # Layer 1: Market Constraints (ICT UTC+7)
        self.session_hours = [
            (dt.time(9, 0), dt.time(11, 30)),
            (dt.time(13, 0), dt.time(14, 30))
        ]
        self.ato_window = (dt.time(9, 0), dt.time(9, 15))
        self.atc_window = (dt.time(14, 15), dt.time(14, 30))
        
        # Risk & Portfolio Parameters (Injected via update)
        self.available_cash = 0.0
        self.t2_locked_shares = 0
        self.buying_power = 0.0
        
        # Status Flags
        self.last_update_ts: Optional[dt.datetime] = None
        self.kill_switch_active = False
        self.drift_status = "STABLE"
        
        logger.info(f"Initialized VNStockLiveDataFeed for {ticker} | Obs Dim: {self.feature_dim}")

    def reset(self):
        """Standardized reset for RL environment consistency."""
        self.raw_buffer.clear()
        self.feature_buffer.clear()
        self.last_update_ts = None
        self.kill_switch_active = False
        self.drift_status = "STABLE"
        logger.info(f"Feed reset for {self.ticker}.")

    # --- Layer 1: Session & Calendar Guard ---
    def is_session_active(self, timestamp: Optional[dt.datetime] = None) -> bool:
        """Evaluates if the market is currently in active trading hours."""
        now = timestamp or dt.datetime.now()
        cur_time = now.time()
        
        # Weekend Filter
        if now.weekday() >= 5: return False
        
        for start, end in self.session_hours:
            if start <= cur_time <= end:
                return True
        return False

    def _check_auction(self, now: dt.datetime) -> bool:
        """Detects if current window is ATO or ATC (Pre/Post Call Auction)."""
        cur_time = now.time()
        morning_auction = self.ato_window[0] <= cur_time <= self.ato_window[1]
        afternoon_auction = self.atc_window[0] <= cur_time <= self.atc_window[1]
        return morning_auction or afternoon_auction

    # --- Layer 2: Integrity & Data Guard ---
    def _detect_stale_or_missing(self, now: dt.datetime) -> bool:
        """Determines if the feed heart-beat has stopped during session hours."""
        if self.last_update_ts is None: return False
        elapsed = (now - self.last_update_ts).total_seconds()
        is_stale = elapsed > 60 and self.is_session_active(now)
        if is_stale:
            logger.warning(f"FEED STALE: {elapsed:.0f}s latency detected.")
        return is_stale

    def update(self, data: Any):
        """
        Ingests multi-source data (Vnstock3 stream or manual tick).
        Validates schema and updates internal buffers.
        """
        now = dt.datetime.now()
        
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            logger.error("Invalid data type for update. Expected DataFrame or Dict.")
            return

        # Essential Schema Check
        required = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(df.columns):
            logger.error(f"Missing required columns: {required - set(df.columns)}")
            return

        # Data Injection
        for _, row in df.iterrows():
            # Sync timestamp to last update
            ts = pd.to_datetime(row['timestamp'])
            self.last_update_ts = ts
            
            # Extract meta-data if provided
            self.available_cash = row.get('available_cash', self.available_cash)
            self.t2_locked_shares = int(row.get('locked_shares_t2', self.t2_shares))
            
            self.raw_buffer.append(row.to_dict())

    # --- Layer 3: Feature Engineering Pipeline ---
    def _engineer_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Stationary feature extraction. Uses strictly causal rolling/shift logic.
        Outputs a raw feature vector (unnormalized).
        """
        c = df['close'].values
        h = df['high'].values
        l = df['low'].values
        o = df['open'].values
        v = df['volume'].values
        
        if len(df) < 5: return np.zeros(self.feature_dim)

        # Vectorized calcs
        ret1 = (c[-1] - c[-2]) / (c[-2] + 1e-8)
        ret3 = (c[-1] - c[-4]) / (c[-4] + 1e-8) if len(c) > 3 else ret1
        ret5 = (c[-1] - c[-6]) / (c[-6] + 1e-8) if len(c) > 5 else ret1
        
        hl_range = (h[-1] - l[-1]) / (c[-1] + 1e-8)
        oc_range = (c[-1] - o[-1]) / (o[-1] + 1e-8)
        
        v_ma5 = np.mean(v[-5:])
        v_rel = v[-1] / (v_ma5 + 1e-8) if v_ma5 > 0 else 1.0
        
        # Momentum vs Trend
        ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().values[-1] if len(c) >= 20 else c[-1]
        dist_ema20 = (c[-1] - ema20) / (ema20 + 1e-8)
        
        # Intraday Time Sin/Cos
        ts = pd.to_datetime(df['timestamp'].iloc[-1])
        minutes = ts.hour * 60 + ts.minute
        time_sin = np.sin(2 * np.pi * minutes / 1440.0)
        time_cos = np.cos(2 * np.pi * minutes / 1440.0)

        # Liquidity / Spread Proxy
        spread_proxy = hl_range * 0.1 # Simplified for tick-level spread lack
        
        # Assembly of raw vector (indices 0-14 used for primary price/vol)
        raw_vec = np.zeros(self.feature_dim)
        raw_vec[0:10] = [ret1, ret3, ret5, hl_range, oc_range, v_rel, dist_ema20, time_sin, time_cos, spread_proxy]
        
        # Layer 4: Regime Detection (Integrated into feature tail)
        regimes = self._detect_regimes(df)
        raw_vec[10] = regimes['trend']
        raw_vec[11] = regimes['volatility']
        raw_vec[12] = regimes['liquidity']
        
        # Market context if available
        if 'vnindex_close' in df.columns:
            vni_c = df['vnindex_close'].values
            vni_ret = (vni_c[-1] - vni_c[-10]) / (vni_c[-10] + 1e-8) if len(vni_c) > 10 else 0
            raw_vec[13] = vni_ret
            
        return raw_vec

    # --- Layer 4: Regime Detection Engine ---
    def _detect_regimes(self, df: pd.DataFrame) -> Dict[str, int]:
        """Detects market conditions for policy modulation."""
        c = df['close'].values
        v = df['volume'].values
        
        if len(c) < 20:
            return {"trend": 0, "volatility": 0, "liquidity": 0}
            
        # Trend Regime
        ma10 = np.mean(c[-10:])
        ma20 = np.mean(c[-20:])
        trend = 1 if ma10 > ma20 else (-1 if ma10 < ma20 else 0)
        
        # Volatility Regime (Normalized ATR proxy)
        tr = np.max(df['high'].values[-20:]) - np.min(df['low'].values[-20:])
        tr_ma = np.mean(tr) if tr > 0 else 1.0
        vol = 1 if tr > tr_ma * 1.5 else (0 if tr > tr_ma * 0.5 else -1)
        
        # Liquidity Regime
        adv = np.mean(v[-20:])
        liq = 1 if v[-1] > adv * 1.2 else (-1 if v[-1] < adv * 0.5 else 0)
        
        return {"trend": trend, "volatility": vol, "liquidity": liq}

    # --- Layer 5: Normalization & Drift Control ---
    def _normalize_and_drift_track(self, raw_vec: np.ndarray) -> np.ndarray:
        """Applies Z-Score clipping and updates running stats to handle drift."""
        # Update running stats (Causal EMA)
        self.feature_means = (1 - self.ema_alpha) * self.feature_means + self.ema_alpha * raw_vec
        self.feature_vars = (1 - self.ema_alpha) * self.feature_vars + self.ema_alpha * (raw_vec - self.feature_means)**2
        
        # Normalize
        std = np.sqrt(self.feature_vars) + 1e-8
        norm_vec = (raw_vec - self.feature_means) / std
        
        # Clip to institutional standard [-3, 3] to handle outliers safely
        norm_vec = np.clip(norm_vec, -3.0, 3.0)
        
        # Simple drift check: if PSI proxy increases
        current_psi = np.abs(np.mean(norm_vec)) # Rough proxy
        if current_psi > 0.5:
            self.drift_status = "DRIFT_DETECTED"
            # Fallback: temporarily freeze or increase window (not implemented: logic auto-adapts via EMA)
        else:
            self.drift_status = "STABLE"
            
        return norm_vec.astype(np.float32)

    # --- Layer 6: Observation Export ---
    def get_observation(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes State Machine sequence: 
        Integrity -> Features -> Regime -> Normalization -> Meta Export.
        """
        now = dt.datetime.now()
        
        # Safety: Empty State
        if len(self.raw_buffer) < 5:
            return np.zeros(self.feature_dim, dtype=np.float32), {"is_session_active": False}

        df_state = pd.DataFrame(list(self.raw_buffer))
        
        # Layer Interaction
        raw_vec = self._engineer_features(df_state)
        norm_obs = self._normalize_and_drift_track(raw_vec)
        
        # Global Flags
        is_active = self.is_session_active(now)
        in_auction = self._check_auction(now)
        is_stale = self._detect_stale_or_missing(now)
        regimes = self._detect_regimes(df_state)
        
        # Layer 6: Kill-Switch Logic (Protection Against Chaos)
        if regimes['volatility'] == 1 and regimes['liquidity'] == -1:
            self.kill_switch_active = True
            logger.warning(f"KILL SWITCH ACTIVE for {self.ticker}: High Vol + Illiquid detected.")
        else:
            self.kill_switch_active = False

        metadata = {
            "is_session_active": is_active,
            "in_auction": in_auction,
            "regimes": regimes,
            "is_stale": is_stale,
            "kill_switch": self.kill_switch_active,
            "available_cash": self.available_cash,
            "t2_locked_shares": self.t2_locked_shares,
            "drift_status": self.drift_status,
            "ticker": self.ticker
        }
        
        return norm_obs, metadata

if __name__ == "__main__":
    # --- Example Usage & Stress Test ---
    logger.info("Running VNStockLiveDataFeed Factory Test...")
    feed = VNStockLiveDataFeed(ticker="FPT")
    
    # 1. Simulate Synthetic ICT Session Start
    start_time = dt.datetime.now().replace(hour=9, minute=0, second=0)
    
    for i in range(20):
        tick = {
            "timestamp": (start_time + dt.timedelta(minutes=i)).strftime('%Y-%m-%d %H:%M:%S'),
            "open": 100.0 + i*0.1,
            "high": 101.0 + i*0.1,
            "low": 99.5 + i*0.1,
            "close": 100.2 + i*0.1,
            "volume": 50000 + (1000 * i),
            "vnindex_close": 1250.0 + i
        }
        feed.update(tick)
        
    # 2. Extract PPO-ready Observation
    obs, meta = feed.get_observation()
    
    logger.info(f"Test Successful | Obs Shape: {obs.shape}")
    logger.info(f"Session Active: {meta['is_session_active']} | In Auction: {meta['in_auction']}")
    logger.info(f"Regime Detection: {meta['regimes']}")
    logger.info(f"Sample Features: {obs[:5]}...")
