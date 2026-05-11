import json
import os
import time
from datetime import datetime

class APIState:
    """
    Persistence layer for standardizing and exporting bot state.
    Serves as the data bridge between Python RL core and Dashboard APIs.
    """
    def __init__(self, base_dir="state"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
        # File path definitions
        self.paths = {
            "portfolio": os.path.join(self.base_dir, "portfolio.json"),
            "history": os.path.join(self.base_dir, "history.json"),
            "lessons": os.path.join(self.base_dir, "lessons.json")
        }
        
        self._initialize_files()

    def _initialize_files(self):
        """Ensures state files exist with valid base structures."""
        if not os.path.exists(self.paths["history"]):
            self._write_json(self.paths["history"], [])
        if not os.path.exists(self.paths["lessons"]):
            self._write_json(self.paths["lessons"], [])

    def update_portfolio(self, equity: float, balance: float, active_positions: list):
        """Exports current account snapshot."""
        data = {
            "equity": round(equity, 2),
            "balance": round(balance, 2),
            "margin_used": round(sum(p.get('margin', 0) for p in active_positions), 2),
            "positions": active_positions,
            "last_updated": datetime.now().isoformat()
        }
        self._write_json(self.paths["portfolio"], data)

    def push_trade(self, symbol, side, price, quantity, pnl=0.0):
        """Logs a completed trade event to history."""
        trade = {
            "id": f"T{int(time.time() * 1000)}",
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "side": side,
            "price": float(price),
            "quantity": float(quantity),
            "pnl": round(float(pnl), 2)
        }
        
        history = self._read_json(self.paths["history"])
        history.append(trade)
        # Keep only last 1000 trades to prevent file bloating
        self._write_json(self.paths["history"], history[-1000:])

    def record_learning_event(self, symbol, state_summary, action_taken, outcome_pnl, reward_score):
        """Records a significant RL transition for feedback auditing."""
        lesson = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "state_snapshot": state_summary,
            "action": action_taken,
            "outcome_pnl": round(outcome_pnl, 4),
            "reward": round(reward_score, 4)
        }
        
        lessons = self._read_json(self.paths["lessons"])
        lessons.append(lesson)
        self._write_json(self.paths["lessons"], lessons[-500:])

    def _write_json(self, path, data):
        """Atomic-simulated write to avoid file corruption during concurrent access."""
        temp_path = f"{path}.tmp"
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=4)
        os.replace(temp_path, path)

    def _read_json(self, path):
        """Safely reads JSON files with fallback."""
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

if __name__ == "__main__":
    # Integration test
    state = APIState()
    state.update_portfolio(10500, 10000, [{"symbol": "BTCUSDT", "position": "LONG"}])
    print(f"State persistence finalized in: {state.base_dir}")
