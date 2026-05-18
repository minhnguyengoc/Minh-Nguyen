import numpy as np
from typing import Dict, Any, Tuple

class RewardEngine:
    """
    Advanced Institutional Reward Engine for Vietnam Stock Market RL.
    Implements a multi-factor decomposition to prevent reward hacking and 
    ensure policy alignment with real-world quantitative edges.
    """
    def __init__(self, 
                 commission_rate: float = 0.0015,
                 tax_rate: float = 0.001,
                 turnover_penalty_scale: float = 0.001,
                 drawdown_penalty_scale: float = 5.0):
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self.turnover_penalty_scale = turnover_penalty_scale
        self.drawdown_penalty_scale = drawdown_penalty_scale
        
        # State tracking for delta rewards
        self.last_equity = None
        self.last_action = 0
        self.hold_count = 0
        self.trade_count = 0

    def compute_reward(self, 
                       equity: float, 
                       initial_capital: float,
                       drawdown: float,
                       action: int,
                       position: float,
                       forward_returns: Dict[int, float], # {window: return_pct}
                       transaction_cost: float,
                       terminated: bool) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates decomposed reward components.
        - directional_edge: Rewards taking positions that align with future price movement.
        - pnl_delta: Log return of equity.
        - turnover_cost: Penalizes excessive switching.
        - drawdown_cost: Quadratic penalty for portfolio risk.
        - inactivity_drag: Prevents holding forever in high-volatility regimes.
        """
        if self.last_equity is None:
            self.last_equity = equity
            return 0.0, {}

        # 1. Realized PnL Reward (Log Equity Return)
        pnl_reward = np.log(equity / (self.last_equity + 1e-9))
        self.last_equity = equity

        # 2. Directional Edge Reward (Predictive alignment)
        # We reward BUY action if future prices go UP, and HOLD/SELL if they stay flat or down.
        # windows: 5, 10, 20
        edge_reward = 0.0
        if action == 1: # BUY
            # Looking for 5-step and 10-step forward edge
            edge_reward += forward_returns.get(5, 0.0) * 0.5
            edge_reward += forward_returns.get(10, 0.0) * 0.3
        elif action == 2: # SELL
            # Reward selling if future prices are expected to drop
            edge_reward -= forward_returns.get(5, 0.0) * 0.4
            edge_reward -= forward_returns.get(10, 0.0) * 0.2

        # 3. Transaction Cost Penalty
        cost_penalty = -(transaction_cost / (initial_capital * 0.01 + 1e-9))

        # 4. Turnover & Churn Penalty
        turnover_penalty = 0.0
        if action != self.last_action and action != 0:
            # Action switch detected (Buy -> Sell or vice-versa)
            turnover_penalty = -self.turnover_penalty_scale * 2.0
            if self.hold_count < 3: # Phạt nặng nếu hold < 3 bars
                turnover_penalty *= 5.0
        
        # 5. Inactivity Drag (Anti-HOLD bias)
        inactivity_penalty = 0.0
        if action == 0:
            self.hold_count += 1
            # Nếu forward edge rất mạnh (>0.5%) mà vẫn HOLD, phạt nhẹ
            if abs(forward_returns.get(10, 0.0)) > 0.005:
                inactivity_penalty = -0.0005
        else:
            self.hold_count = 0
            self.trade_count += 1

        # 6. Drawdown Penalty (Squared and Bounded)
        drawdown_penalty = -(drawdown ** 2) * self.drawdown_penalty_scale
        drawdown_penalty = np.clip(drawdown_penalty, -2.0, 0.0)

        # 7. Terminal Failure Penalty
        terminal_penalty = 0.0
        if terminated and equity < initial_capital * 0.7:
            terminal_penalty = -5.0

        # Assemble Components and Apply Scaling
        components = {
            "directional_edge": np.clip(edge_reward * 5.0, -1.0, 1.0),
            "pnl_realized": np.clip(pnl_reward * 100.0, -2.0, 2.0),
            "cost_penalty": np.clip(cost_penalty, -1.0, 0.0),
            "turnover_penalty": np.clip(turnover_penalty, -1.0, 0.0),
            "drawdown_cost": drawdown_penalty,
            "inactivity_drag": np.clip(inactivity_penalty, -0.5, 0.0),
            "terminal_delta": terminal_penalty
        }

        self.last_action = action
        
        total_reward = sum(components.values())
        # Multiplier of 1.0 because components are already scaled for PPO
        final_reward = float(np.clip(total_reward, -10, 10))

        return final_reward, components

if __name__ == "__main__":
    re = RewardEngine()
    # Mock call
    r, c = re.compute_reward(10100, 10000, 0.02, 1, 100, {5: 0.01, 10: 0.02}, 150, False)
    print(f"Reward: {r}, Components: {c}")

