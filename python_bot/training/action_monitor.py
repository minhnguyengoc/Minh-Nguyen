import numpy as np
import logging

class ActionMonitor:
    """
    Tracks action distribution during training and evaluation.
    Identifies policy collapse (HOLD-only) and overtrading failure modes.
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("ActionMonitor")
        self.reset()

    def reset(self):
        self.hold_count = 0
        self.buy_count = 0
        self.sell_count = 0
        self.close_count = 0
        self.executed_position_changes = 0
        self.total_steps = 0
        self.rejections = {}

    def record_action(self, action: int, info: dict):
        self.total_steps += 1
        if action == 0: self.hold_count += 1
        elif action == 1: self.buy_count += 1
        elif action == 2: self.sell_count += 1
        elif action == 3: self.close_count += 1
        
        if info.get("position_changed", False):
            self.executed_position_changes += 1
            
        rej = info.get("rejected_reason")
        if rej:
            self.rejections[rej] = self.rejections.get(rej, 0) + 1

    def get_stats(self) -> dict:
        total = max(1, self.total_steps)
        hold_pct = self.hold_count / total
        trade_freq = (self.buy_count + self.sell_count + self.close_count) / total
        trades_per_1k = (self.executed_position_changes / total) * 1000
        
        # Action Entropy
        probs = np.array([self.hold_count, self.buy_count, self.sell_count, self.close_count]) / total
        entropy = -np.sum(probs * np.log2(probs + 1e-9))
        
        return {
            "hold_pct": hold_pct,
            "trade_frequency": trade_freq,
            "trades_per_1k_steps": trades_per_1k,
            "executed_position_changes": self.executed_position_changes,
            "action_entropy": entropy,
            "rejections": self.rejections
        }

    def check_failure_modes(self, step: int):
        stats = self.get_stats()
        
        if step > 50000:
            if stats["hold_pct"] > 0.98:
                raise RuntimeError(f"POLICY_LEARNED_ALWAYS_HOLD: {stats['hold_pct'] * 100:.2f}% HOLD at step {step}")
            
            if stats["trades_per_1k_steps"] < 5:
                # Still inactive after anti-HOLD phase
                self.logger.warning(f"STILL_INACTIVE Warning: {stats['trades_per_1k_steps']:.2f} trades per 1k steps")

        if stats["trade_frequency"] > 0.25:
            raise RuntimeError(f"OVERTRADING_POLICY: {stats['trade_frequency'] * 100:.2f}% activity")
        
        if stats["trades_per_1k_steps"] > 250:
            raise RuntimeError(f"HYPER_TURNOVER: {stats['trades_per_1k_steps']:.2f} trades per 1k steps")

from typing import Optional
