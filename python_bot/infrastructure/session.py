from datetime import datetime, time
from typing import Optional
from python_bot.common.types import SessionState

class SessionBoundaryFSM:
    """
    Finite State Machine for Vietnam Market Sessions.
    Handles transitions, breaks, and session-specific logic isolation.
    """
    def __init__(self):
        self._state = SessionState.CLOSED
        
        # VN Market Rules (ICT UTC+7)
        self.RULES = {
            'ATO': (time(9, 0), time(9, 15)),
            'MORNING': (time(9, 15), time(11, 30)),
            'LUNCH': (time(11, 30), time(13, 0)),
            'AFTERNOON': (time(13, 0), time(14, 30)),
            'ATC': (time(14, 30), time(14, 45)), # Optional ATC window variation
            'CLOSED': (time(14, 45), time(23, 59))
        }

    def update(self, ts: datetime) -> SessionState:
        t = ts.time()
        prev_state = self._state
        
        if ts.weekday() >= 5:
            self._state = SessionState.CLOSED
        elif self.RULES['ATO'][0] <= t < self.RULES['ATO'][1]:
            self._state = SessionState.ATO
        elif self.RULES['MORNING'][0] <= t < self.RULES['MORNING'][1]:
            self._state = SessionState.CONTINUOUS_MORNING
        elif self.RULES['LUNCH'][0] <= t < self.RULES['LUNCH'][1]:
            self._state = SessionState.LUNCH_BREAK
        elif self.RULES['AFTERNOON'][0] <= t < self.RULES['AFTERNOON'][1]:
            self._state = SessionState.CONTINUOUS_AFTERNOON
        elif self.RULES['ATC'][0] <= t < self.RULES['ATC'][1]:
            self._state = SessionState.ATC
        else:
            self._state = SessionState.CLOSED
            
        return self._state

    @property
    def state(self) -> SessionState:
        return self._state

    def is_interrupted(self, ts: datetime, prev_ts: Optional[datetime]) -> bool:
        """Detects if we crossed a lunch or overnight boundary."""
        if not prev_ts: return False
        
        # Day break
        if ts.date() > prev_ts.date(): return True
        
        # Lunch break
        prev_time = prev_ts.time()
        curr_time = ts.time()
        lunch_start = self.RULES['LUNCH'][0]
        if prev_time < lunch_start <= curr_time:
            return True
            
        return False
