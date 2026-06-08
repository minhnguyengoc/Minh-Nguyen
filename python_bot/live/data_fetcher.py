import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from python_bot.core.paths import HISTORICAL_DATA_DIR
from python_bot.core.exceptions import LiveFeedException

logger = logging.getLogger("VNStockBot.DataFetcher")

def update_data(ticker: str, timeframe: str = "1m") -> dict:
    """
    Fetches the latest bars for ticker and appends them to historical CSV file.
    Deduplicates based on 'timestamp' and saves sorted ascending.
    """
    file_path = HISTORICAL_DATA_DIR / f"{ticker}_1m.csv"
    
    # 1. Load existing data if possible
    existing_df = None
    rows_before = 0
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path)
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
            rows_before = len(existing_df)
        except Exception as e:
            logger_err_msg = f"Failed to read existing backup CSV {file_path}: {e}"
            logger.warning(logger_err_msg)
            
    # Determine the date search range (last 5 days to get latest values)
    end_date_str = datetime.now().strftime('%Y-%m-%d')
    start_date = datetime.now() - timedelta(days=5)
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    # 2. Try fetching from multiple sources via vnstock v4
    fetched_df = None
    sources = ['VCI', 'KBS', 'MSN', 'FMP']
    
    for src in sources:
        try:
            logger.info(f"Polling source {src} for {ticker} (Recent updating)...")
            from vnstock.api.quote import Quote
            q = Quote(symbol=ticker, source=src)
            try:
                fetched_df = q.history(start=start_date_str, end=end_date_str, interval=timeframe)
            except Exception:
                try:
                    fetched_df = q.history(start=start_date_str, end=end_date_str, resolution=timeframe)
                except Exception:
                    fetched_df = q.history(start=start_date_str)
                    
            if fetched_df is not None and not fetched_df.empty:
                # Ensure it has required elements
                fetched_df.columns = [c.lower() for c in fetched_df.columns]
                if 'time' in fetched_df.columns:
                    fetched_df = fetched_df.rename(columns={'time': 'timestamp'})
                elif 'date' in fetched_df.columns:
                    fetched_df = fetched_df.rename(columns={'date': 'timestamp'})
                    
                # Clean and check nulls
                if not fetched_df[['open', 'high', 'low', 'close']].isnull().values.any():
                    logger.info(f"Successfully fetched {len(fetched_df)} fresh rows from source: {src}")
                    break
                else:
                    fetched_df = None
        except Exception as e:
            logger.debug(f"Source {src} polling exception: {e}")
            
    if fetched_df is None or fetched_df.empty:
        # If API failed and we have existing file, we can fall back silently, but print NO_NEW_DATA
        if existing_df is not None:
            logger.info("NO_NEW_DATA (Market closed / API unreachable, using existing cache)")
            return {
                "ticker": ticker,
                "status": "NO_NEW_DATA",
                "rows_before": rows_before,
                "rows_after": rows_before,
                "added_rows": 0,
                "latest_close": float(existing_df['close'].iloc[-1]) if len(existing_df) > 0 else 0.0
            }
        else:
            raise LiveFeedException("All live sources failed, and no cached file was found to fall back on.")
            
    # Standardize columns
    fetched_df['timestamp'] = pd.to_datetime(fetched_df['timestamp'])
    fetched_df = fetched_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # 3. Merge & Deduplicate
    if existing_df is not None:
        combined_df = pd.concat([existing_df, fetched_df], ignore_index=True)
    else:
        combined_df = fetched_df
        
    combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    rows_after = len(combined_df)
    added_rows = rows_after - rows_before
    
    # Check if there really is new data
    if added_rows == 0:
        logger.info("NO_NEW_DATA (Latest timestamp is already cached)")
        return {
            "ticker": ticker,
            "status": "NO_NEW_DATA",
            "rows_before": rows_before,
            "rows_after": rows_before,
            "added_rows": 0,
            "latest_close": float(combined_df['close'].iloc[-1])
        }
        
    # Standardize types and write to file
    combined_df['open'] = combined_df['open'].astype(float)
    combined_df['high'] = combined_df['high'].astype(float)
    combined_df['low'] = combined_df['low'].astype(float)
    combined_df['close'] = combined_df['close'].astype(float)
    combined_df['volume'] = combined_df['volume'].astype(float)
    
    # Save back
    HISTORICAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(file_path, index=False)
    
    start_ts = str(combined_df['timestamp'].min())
    end_ts = str(combined_df['timestamp'].max())
    latest_close = float(combined_df['close'].iloc[-1])
    
    logger.info(f"📈 [SUCCESS] {ticker} dataset updated: rows_before={rows_before}, "
                f"rows_after={rows_after}, added_rows={added_rows}, start={start_ts}, "
                f"end={end_ts}, latest_close={latest_close:.2f}")
                
    return {
        "ticker": ticker,
        "status": "SUCCESS",
        "rows_before": rows_before,
        "rows_after": rows_after,
        "added_rows": added_rows,
        "start": start_ts,
        "end": end_ts,
        "latest_close": latest_close
    }
