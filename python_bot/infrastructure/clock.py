from datetime import datetime, time, timedelta
import logging
from typing import Optional
from python_bot.common.types import SessionState

class ClockTriangulator:
    """
    Ensures absolute causal consistency by only allowing time to progress via market events.
    Prevents leaking datetime.now() into the core logic.
    """
    def __init__(self, start_ts: Optional[datetime] = None):
        self._current_ts = start_ts
        self._counter = 0

    def sync(self, event_ts: datetime):
        if self._current_ts and event_ts < self._current_ts:
            # Detects temporal backtracking (Non-causal replay error)
            raise ValueError(f"Temporal violation: Received {event_ts} < Current {self._current_ts}")
        
        if self._current_ts == event_ts:
            self._counter += 1
        else:
            self._current_ts = event_ts
            self._counter = 0

    @property
    def now(self) -> datetime:
        if not self._current_ts:
            raise RuntimeError("Clock uninitialized. No market event received.")
        return self._current_ts

    def delta_seconds(self, other_ts: datetime) -> float:
        return (self.now - other_ts).total_seconds()
