import numpy as np

class RewardEngine:
    """
    Advanced reward calculation engine for RL trading agents.
    Designed to balance profitability, risk, and trade efficiency.
    """
    def __init__(self, drawdown_multiplier: float = 5.0, trade_cost_penalty: float = 0.005):
        self.drawdown_multiplier = drawdown_multiplier
        self.trade_cost_penalty = trade_cost_penalty
        self.last_equity_ratio = 1.0
        self.consecutive_profits = 0
        self.last_action = 0

    def compute_reward(self, equity, initial_balance, drawdown, trade_count, action, current_position, terminated, execution_penalty=0.0):
        """
        Calculates the scalar reward for a given timestep + breakdown.
        """
        current_equity_ratio = equity / (initial_balance + 1e-9)
        
        # 1. Differential Reward (Log Return based)
        returns = (np.log(current_equity_ratio + 1e-9) - np.log(self.last_equity_ratio + 1e-9))
        pnl_signal = float(returns * 200.0) # Increased importance of returns
        self.last_equity_ratio = current_equity_ratio
        
        # 2. Drawdown Penalty (Convex + Scaled)
        drawdown_penalty = float((drawdown ** 2) * self.drawdown_multiplier * 5.0)
        
        # 3. Trade Friction (Anti-Churn Policy)
        activity_penalty = 0.0
        if action in [1, 2, 3]: 
            # Friction scaled to prevent micro-scalping
            activity_penalty = float(self.trade_cost_penalty)
            # Whiplash penalty: Penalize switching directions without 'Hold' steps
            # action 1: Long, 2: Short, 3: Close
            if self.last_action != 0 and action != self.last_action and action != 3:
                activity_penalty *= 5.0
            
        self.last_action = action
            
        # 4. Consistency Reward
        consistency_bonus = 0.0
        if returns > 0.0001:
            self.consecutive_profits += 1
            consistency_bonus = float(min(self.consecutive_profits * 0.001, 0.05))
        else:
            self.consecutive_profits = 0

        # 5. Liquidation Penalty
        terminal_reward = 0.0
        if terminated:
            if current_equity_ratio < 0.5: 
                terminal_reward = -10.0
            elif current_equity_ratio > 1.1: 
                terminal_reward = 3.0
        
        # 6. Inactivity Penalty
        inactivity_penalty = 0.0
        if current_position == 0 and action == 0:
            inactivity_penalty = 0.0001

        # Assemble Components
        components = {
            "pnl_signal": pnl_signal,
            "drawdown_cost": -drawdown_penalty,
            "trade_friction": -activity_penalty,
            "consistency_bonus": consistency_bonus,
            "terminal_delta": terminal_reward,
            "inactivity_drag": -inactivity_penalty
        }
        
        total_reward = sum(components.values())
        final_reward = float(np.clip(total_reward, -20, 20))
        
        return final_reward, components

if __name__ == "__main__":
    re = RewardEngine()
    r, c = re.compute_reward(10500, 10000, 0.05, 1, 1, 1, False)
    print(f"Total Reward: {r}")
    print(f"Components: {c}")
