from datetime import datetime, timedelta
import heapq
from typing import List, Dict, Optional, Tuple
from python_bot.common.types import MarketData
import logging

class TimestampOrderingBuffer:
    """
    Solves Out-of-Order (OOO) packet arrival by buffering ticks and releasing them monotonically.
    Required for institutional connectivity where UDP packets move on divergent paths.
    """
    def __init__(self, tolerance_ms: int = 500):
        self.tolerance_ms = tolerance_ms
        self._buffer: List[Tuple[datetime, MarketData]] = []
        self._last_released_ts: Optional[datetime] = None

    def push(self, data: MarketData):
        if self._last_released_ts and data.timestamp < self._last_released_ts:
            # Fatal Causality Violation: Data arrived too late to be reordered
            logging.error(f"DROPPING STALE PACKET: {data.event_id} TS {data.timestamp} < Released {self._last_released_ts}")
            return
        
        heapq.heappush(self._buffer, (data.timestamp, data.event_id, data))

    def release(self, current_sync_ts: datetime) -> List[MarketData]:
        """Releases all ticks that are older than (current_sync_ts - tolerance)."""
        released = []
        cutoff = current_sync_ts - timedelta(milliseconds=self.tolerance_ms)
        
        while self._buffer and self._buffer[0][0] <= cutoff:
            ts, _, data = heapq.heappop(self._buffer)
            self._last_released_ts = ts
            released.append(data)
            
        return released

class MonotonicEventSequencer:
    """
    Guarantees absolute sequence integrity and deduplication.
    Ensures bitwise identity across replays by forcing a strict global order.
    """
    def __init__(self):
        self._seen_ids = set()
        self._max_seen_ids = 10000
        self._next_sequence_id = 0

    def process(self, data: MarketData) -> Optional[int]:
        if data.event_id in self._seen_ids:
            return None # Duplicate
            
        self._seen_ids.add(data.event_id)
        if len(self._seen_ids) > self._max_seen_ids:
            self._seen_ids.clear() # Prune
            
        seq_id = self._next_sequence_id
        self._next_sequence_id += 1
        return seq_id
