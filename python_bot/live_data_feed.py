import pandas as pd
import numpy as np
import datetime as dt
import time
import logging
from typing import Optional, List
from vnstock import Vnstock

# Configure institutional logging for live data auditing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveDataFeed:
    """
    Institutional-grade live data connector for Vietnam Stock Market (HOSE/HNX).
    Transforms realtime 1m candles into (30, 20) state matrices for RL model inference.
    Designed for zero look-ahead bias and robustness against API latency.
    """
    def __init__(self, ticker: str, lookback: int = 30, cache_size: int = 200):
        self.ticker = ticker
        self.lookback = lookback
        self.cache_size = cache_size
        self._df_cache = pd.DataFrame()
        
        # 20 feature columns following the PPO observation space requirement
        self.feature_columns = [
            'return_1', 'return_3', 'return_5', 'hl_range', 'oc_range',
            'volume_ratio', 'vwap_dev', 'rolling_std_5', 'rolling_std_15', 'momentum_3',
            'rsi_14', 'dist_ema20', 'dist_ema50', 'ema_gap', 'atr_pct',
            'relative_volume_20', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos'
        ]
        
        try:
            # Source: 'VCI' is a reliable source for 1m intraday resolution for VN market
            self.api = Vnstock().stock(symbol=ticker, source='VCI')
            logger.info(f"LiveDataFeed initialized for {ticker}")
        except Exception as e:
            logger.error(f"Failed to initialize vnstock for {ticker}: {e}")
            self.api = None

    def _fetch_recent_candles(self) -> pd.DataFrame:
        """Fetches latest intraday 1m candles via vnstock API."""
        if not self.api:
            return pd.DataFrame()
            
        try:
            # Request latest 1m candles. Frequency '1m'.
            # Note: start_date is today to minimize payload, but robust enough for the session mask.
            today_str = dt.datetime.now().strftime('%Y-%m-%d')
            df = self.api.quote.history(symbol=self.ticker, resolution="1m", start_date=today_str)
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            # Standardize naming to match internal pipeline
            df.columns = [c.lower() for c in df.columns]
            # Handle potential column name variations
            if 'time' not in df.columns and 'timestamp' in df.columns:
                df.rename(columns={'timestamp': 'time'}, inplace=True)
            elif 'date' in df.columns:
                df.rename(columns={'date': 'time'}, inplace=True)
            
            # Normalize datetime to tz-naive (UTC+7 interpreted)
            df['time'] = pd.to_datetime(df['time'])
            if df['time'].dt.tz is not None:
                df['time'] = df['time'].dt.tz_localize(None)
            
            # Sort ascending, drop duplicate timestamps to maintain signal integrity
            df = df.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
            
            return self._filter_vn_session(df)
        except Exception as e:
            logger.error(f"Production API Fetch Error for {self.ticker}: {e}")
            return pd.DataFrame()

    def _filter_vn_session(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keeps only VN valid session times (09:00-11:30 and 13:00-14:30)."""
        if df.empty: return df
        
        t = df['time'].dt.time
        # Morning session: 09:00 - 11:30
        morning = (t >= dt.time(9, 0)) & (t <= dt.time(11, 30))
        # Afternoon session: 13:00 - 14:30
        afternoon = (t >= dt.time(13, 0)) & (t <= dt.time(14, 30))
        
        return df[morning | afternoon].copy()

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rebuilds the EXACT observation features for PPO.
        Uses vectorized pandas operations for performance and no look-ahead bias.
        Total: 20 features, dtype=float32.
        """
        if len(df) < self.cache_size // 2: # Heuristic for warmup
            return pd.DataFrame()
            
        df = df.copy()
        c = df['close']
        h = df['high']
        l = df['low']
        o = df['open']
        v = df['volume']
        
        # 1-10: Primary Momentum and Range Features (Stationary)
        df['return_1'] = c.pct_change(1)
        df['return_3'] = c.pct_change(3)
        df['return_5'] = c.pct_change(5)
        df['hl_range'] = (h - l) / (c + 1e-8)
        df['oc_range'] = (c - o) / (o + 1e-8)
        df['volume_ratio'] = v / (v.rolling(5).mean() + 1e-8)
        
        # vwap_dev: deviation from typical price (HLC3) as a VWAP proxy
        tp = (h + l + c) / 3
        df['vwap_dev'] = (c - tp) / (tp + 1e-8)
        
        df['rolling_std_5'] = c.rolling(5).std() / (c + 1e-8)
        df['rolling_std_15'] = c.rolling(15).std() / (c + 1e-8)
        df['momentum_3'] = (c - c.shift(3)) / (c.shift(3) + 1e-8)
        
        # 11-20: Secondary Trend and Time Features
        # RSI 14 (Relative Strength Index)
        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi_14'] = (100 - (100 / (1 + rs))) / 100.0 # Normalized [0, 1]
        
        # Trend distances
        ema20 = c.ewm(span=20, adjust=False).mean()
        ema50 = c.ewm(span=50, adjust=False).mean()
        df['dist_ema20'] = (ema20 - c) / (c + 1e-8)
        df['dist_ema50'] = (ema50 - c) / (c + 1e-8)
        df['ema_gap'] = (ema20 - ema50) / (c + 1e-8)
        
        # ATR Proxy (Average True Range percentage)
        tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
        df['atr_pct'] = tr.rolling(14).mean() / (c + 1e-8)
        
        # Relative liquidity
        df['relative_volume_20'] = v / (v.rolling(20).mean() + 1e-8)
        
        # Cyclic time features (encoding intraday session rhythm)
        hours = df['time'].dt.hour + df['time'].dt.minute/60.0
        df['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
        
        minutes = df['time'].dt.minute
        df['minute_sin'] = np.sin(2 * np.pi * minutes / 60.0)
        df['minute_cos'] = np.cos(2 * np.pi * minutes / 60.0)
        
        # Final cleanup for RL stability
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        return df[self.feature_columns].astype(np.float32)

    def get_latest_state(self) -> Optional[np.ndarray]:
        """
        Exposes the latest [lookback, 20] float32 state matrix for PPO inference.
        Returns:
            np.ndarray: Matrix of shape (lookback, 20)
            None: If data is insufficient or market is closed
        """
        raw_df = self._fetch_recent_candles()
        if raw_df.empty:
            return None
            
        # Maintain internal rolling cache for feature computation
        self._df_cache = raw_df.tail(self.cache_size)
        
        # Compute 20 features
        features_df = self._compute_features(self._df_cache)
        
        if len(features_df) < self.lookback:
            logger.debug(f"Waiting for more candles for {self.ticker}...")
            return None
            
        # Return the last (lookback) window for the model
        state = features_df.tail(self.lookback).values
        return state # Matrix shape: (30, 20)

if __name__ == "__main__":
    # Test block for intraday connectivity
    TICKER = "FPT"
    feed = LiveDataFeed(ticker=TICKER)
    logger.info(f"--- Live Terminal: {TICKER} ---")
    
    try:
        # One-off check for terminal output
        state = feed.get_latest_state()
        if state is not None:
            logger.info(f"SUCCESS: State Matrix Shape {state.shape}")
            logger.info(f"Last Row:\n{state[-1]}")
        else:
            logger.warning("Market session not active or API failed to return data.")
    except Exception as e:
        logger.error(f"Live Feed Test Failed: {e}")
