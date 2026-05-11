from python_bot.common.types import ActionDirective, ObservationMetadata, PortfolioState
import logging

class RiskSupervisor:
    """
    Final Gatekeeper for Agent Actions.
    Enforces absolute safety constraints, circuit breakers, and OOD abstention.
    """
    def __init__(self, max_drawdown: float = 0.05, max_leverage: float = 1.0):
        self.max_drawdown = max_drawdown
        self.max_leverage = max_leverage
        self._peak_equity = -1.0
        
        self._halted = False

    def validate_action(self, 
                        action: ActionDirective, 
                        portfolio: PortfolioState, 
                        meta: ObservationMetadata) -> ActionDirective:
        """
        Filters agent directive through risk layers.
        Returns final ActionDirective (Safe).
        """
        if self._halted: return ActionDirective.HOLD
        
        # 1. Drawdown Circuit Breaker
        current_equity = portfolio.available_cash + (portfolio.position_quantity * portfolio.average_entry_price) + portfolio.unrealized_pnl
        if self._peak_equity < 0: self._peak_equity = current_equity
        self._peak_equity = max(self._peak_equity, current_equity)
        
        drawdown = (self._peak_equity - current_equity) / (self._peak_equity + 1e-8)
        if drawdown > self.max_drawdown:
            logging.critical(f"RISK HALT: Max drawdown exceeded ({drawdown:.2%})")
            self._halted = True
            return ActionDirective.CLOSE
            
        # 2. OOD / Confidence Gating
        if meta.policy_abstain or meta.drift_score > 0.5:
            logging.warning("RISK GATING: High drift or Policy Abstention. Forcing HOLD.")
            if portfolio.position_quantity != 0:
                return ActionDirective.HOLD # Freeze, don't necessarily close unless panic
            return ActionDirective.HOLD

        # 3. Session End / Auction Masking
        from python_bot.common.types import SessionState
        is_auction = meta.session_state in [SessionState.ATO, SessionState.ATC]
        if not meta.is_session_active or is_auction:
            if action in [ActionDirective.LONG, ActionDirective.SHORT]:
                return ActionDirective.HOLD
        
        return action
