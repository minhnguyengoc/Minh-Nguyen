import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple

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
        """
        Behavioral hard gate.

        Important:
        - Raw action frequency is NOT executed overtrading.
        - Do not fail training just because the policy emits many BUY/SELL actions.
        - Hard fail only on executed position changes / executed trade rate.
        """
        stats = self.get_stats()

        total_actions = (
            self.hold_count
            + self.buy_count
            + self.sell_count
            + self.close_count
        )

        if total_actions <= 0:
            return stats

        raw_action_frequency = stats.get("trade_frequency", 0.0)
        executed_changes = stats.get("executed_position_changes", 0)

        executed_per_1k = executed_changes / max(step, 1) * 1000

        stats["raw_action_frequency"] = raw_action_frequency
        stats["executed_position_changes"] = executed_changes
        stats["executed_trades_per_1k_steps"] = executed_per_1k

        # Warmup. Do not hard-fail early exploration.
        if step < 30000 or total_actions < 10000:
            return stats

        # Raw activity is only a warning.
        if raw_action_frequency > 0.50:
            self.logger.warning(
                f"RAW_ACTION_OVERACTIVE Warning: {raw_action_frequency * 100:.2f}% raw activity at step {step}"
            )

        # Hard fail only if real executed position changes are excessive.
        if executed_per_1k > 120:
            raise RuntimeError(
                f"EXECUTED_OVERTRADING_POLICY: {executed_per_1k:.2f} executed changes per 1k steps at step {step}"
            )

        # Always-HOLD check only after enough training.
        if step >= 50000:
            if stats.get("hold_pct", 0.0) > 0.98 and executed_changes == 0:
                raise RuntimeError(
                    f"POLICY_LEARNED_ALWAYS_HOLD: {stats['hold_pct'] * 100:.2f}% HOLD at step {step}"
                )

            if executed_per_1k < 2:
                self.logger.warning(
                    f"STILL_INACTIVE Warning: {executed_per_1k:.2f} executed changes per 1k steps"
                )

        return stats

