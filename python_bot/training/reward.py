import numpy as np
from typing import Dict, List, Optional, Tuple
from python_bot.common.types import FillEvent, PortfolioState, MarketData, ActionDirective

class InstitutionalRewardEngine:
    """
    Advanced Reward Engineering for RL Agents.
    Optimizes for:
    - Risk-Adjusted Expectancy (Sharpe)
    - Execution Quality
    - Survival (Drawdown Prevention)
    - Stability (Turnover Control)
    """
    def __init__(self, 
                 transaction_cost_pct: float = 0.0003,
                 drawdown_penalty_weight: float = 5.0,
                 turnover_penalty_weight: float = 0.1,
                 overnight_penalty_weight: float = 0.5):
        self.tc_pct = transaction_cost_pct
        self.dd_weight = drawdown_penalty_weight
        self.churn_weight = turnover_penalty_weight
        self.overnight_weight = overnight_penalty_weight
        
        self._prev_equity = -1.0
        self._peak_equity = -1.0
        self._last_action = ActionDirective.HOLD
        self._last_date = None

    def calculate(self, 
                  portfolio_state: PortfolioState, 
                  market_data: MarketData, 
                  action: ActionDirective,
                  fills: List[FillEvent]) -> Tuple[float, Dict[str, float]]:
        """
        Computes a multi-component decomposed reward.
        """
        current_equity = portfolio_state.available_cash + (portfolio_state.position_quantity * market_data.close)
        
        if self._prev_equity < 0:
            self._prev_equity = current_equity
            self._peak_equity = current_equity
            self._last_date = market_data.timestamp.date()
            return 0.0, {}
            
        # A. Performance Component (Log returns)
        pnl_raw = np.log(current_equity / (self._prev_equity + 1e-9))
        
        # B. Risk Component (Drawdown Penalty)
        self._peak_equity = max(self._peak_equity, current_equity)
        drawdown = (self._peak_equity - current_equity) / (self._peak_equity + 1e-9)
        risk_penalty = -self.dd_weight * (drawdown ** 2)
        
        # C. Cost Component (Execution & Turnover)
        fill_notional = sum([f.fill_quantity * f.fill_price for f in fills])
        costs = -(fill_notional * self.tc_pct) / (current_equity + 1e-9)
        
        # Action Friction (Internal Penalty)
        churn = 0.0
        if action != self._last_action and action != ActionDirective.HOLD:
            churn = -self.churn_weight
            
        # D. Overnight Risk Component
        overnight = 0.0
        if market_data.timestamp.date() > self._last_date:
            # Charge for holding position across session boundary
            if portfolio_state.position_quantity != 0:
                overnight = -self.overnight_weight * abs(portfolio_state.position_quantity * market_data.close) / current_equity
        
        # Total Reward
        total_reward = pnl_raw + risk_penalty + costs + churn + overnight
        
        components = {
            "pnl": pnl_raw,
            "risk": risk_penalty,
            "costs": costs,
            "churn": churn,
            "overnight": overnight
        }
        
        # Update state
        self._prev_equity = current_equity
        self._last_action = action
        self._last_date = market_data.timestamp.date()
        
        return float(total_reward), components

    def reset(self):
        self._prev_equity = -1.0
        self._peak_equity = -1.0
        self._last_action = ActionDirective.HOLD
