import numpy as np

class MatchingEngine:
    """
    Institutional-Grade Trade Execution Engine.
    Simulates complex market mechanics:
    - Atomic Reversals with cumulative friction.
    - Stochastic volatility-scaled slippage.
    - Liquidity spike events.
    - Scalping friction (fee penalties for ultra-short duration).
    """
    def __init__(self, fee_rate: float = 0.0004, base_slippage: float = 0.00005):
        # 0.04% Institutional Taker Fee
        self.fee_rate = fee_rate
        self.base_slippage = base_slippage

    def process_action(self, action, current_price, position, position_notional, leverage, spread, 
                       entry_price=0.0, atr_pct=0.002, hold_duration=0):
        """
        Calculates high-fidelity execution result using hard-notional anchoring.
        Returns: Dict {executed, type, fill_price, fee, new_position, execution_penalty, close_fill_price, open_fill_price}
        """
        result = {'executed': False, 'execution_penalty': 0.0}
        
        # 1. Stochastic Slippage Calculation
        # Base stochastic noise between 2bps and 10bps
        stochastic_noise = np.random.uniform(0.00002, 0.0001)
        # Volatility multiplier (relative to baseline 0.1% ATR)
        vol_multiplier = 1 + (atr_pct / 0.001)
        
        # 2. Liquidity Degradation Spike (8% probability)
        liquidity_multiplier = 1.5 if np.random.random() < 0.08 else 1.0
        
        # 3. Abnormal Execution Event (1% probability)
        abnormal_multiplier = 2.0 if np.random.random() < 0.01 else 1.0
        
        # Aggregate Slippage
        final_slippage_pct = stochastic_noise * vol_multiplier * liquidity_multiplier * abnormal_multiplier
        
        # 4. Scalping Friction (Duration Penalty)
        # If closing/reversing under 3 minutes, add 25% to the base fee rate
        effective_fee_rate = self.fee_rate
        if hold_duration > 0 and hold_duration < 3 and action in [1, 2, 3]:
            effective_fee_rate *= 1.25

        # --- EXECUTION LOGIC ---

        # Action: LONG
        if action == 1 and position != 1:
            if position == -1: # REVERSAL: Short -> Long
                # Close Short (Bid) -> Open Long (Ask)
                close_fill_price = (current_price - (spread / 2)) * (1 - final_slippage_pct)
                open_fill_price = (current_price + (spread / 2)) * (1 + final_slippage_pct)
                
                # Apply friction to both legs
                fee = (position_notional * leverage * 2) * effective_fee_rate 
                execution_penalty = (spread) + (current_price * final_slippage_pct * 2)
                
                return {
                    'executed': True,
                    'type': 'REVERSE_TO_LONG',
                    'fill_price': open_fill_price,
                    'close_fill_price': close_fill_price,
                    'open_fill_price': open_fill_price,
                    'fee': fee,
                    'new_position': 1,
                    'execution_penalty': execution_penalty
                }
            
            # Simple Open Long
            fill_price = (current_price + (spread / 2)) * (1 + final_slippage_pct)
            fee = (position_notional * leverage) * effective_fee_rate
            execution_penalty = (spread / 2) + (current_price * final_slippage_pct)
            
            result = {
                'executed': True,
                'type': 'OPEN_LONG',
                'fill_price': fill_price,
                'fee': fee,
                'new_position': 1,
                'execution_penalty': execution_penalty
            }

        # Action: SHORT
        elif action == 2 and position != -1:
            if position == 1: # REVERSAL: Long -> Short
                # Close Long (Ask) -> Open Short (Bid)
                close_fill_price = (current_price + (spread / 2)) * (1 + final_slippage_pct)
                open_fill_price = (current_price - (spread / 2)) * (1 - final_slippage_pct)
                
                # Apply friction to both legs
                fee = (position_notional * leverage * 2) * effective_fee_rate
                execution_penalty = (spread) + (current_price * final_slippage_pct * 2)
                
                return {
                    'executed': True,
                    'type': 'REVERSE_TO_SHORT',
                    'fill_price': open_fill_price,
                    'close_fill_price': close_fill_price,
                    'open_fill_price': open_fill_price,
                    'fee': fee,
                    'new_position': -1,
                    'execution_penalty': execution_penalty
                }
            
            # Simple Open Short
            fill_price = (current_price - (spread / 2)) * (1 - final_slippage_pct)
            fee = (position_notional * leverage) * effective_fee_rate
            execution_penalty = (spread / 2) + (current_price * final_slippage_pct)
            
            result = {
                'executed': True,
                'type': 'OPEN_SHORT',
                'fill_price': fill_price,
                'fee': fee,
                'new_position': -1,
                'execution_penalty': execution_penalty
            }

        # Action: CLOSE
        elif action == 3 and position != 0:
            result = self._calculate_exit(
                current_price, position, position_notional, leverage, spread, 
                final_slippage_pct, effective_fee_rate
            )
            
        return result

    def _calculate_exit(self, price, position, position_notional, leverage, spread, slippage, fee_rate):
        """Internal exit computation with explicit friction models."""
        if position == 1: # Selling Long (Exit at Bid)
            fill_price = (price - (spread / 2)) * (1 - slippage)
        else: # Buying back Short (Exit at Ask)
            fill_price = (price + (spread / 2)) * (1 + slippage)
            
        fee = (position_notional * leverage) * fee_rate
        execution_penalty = (spread / 2) + (price * slippage)
        
        return {
            'executed': True,
            'type': 'CLOSE',
            'fill_price': fill_price,
            'fee': fee,
            'new_position': 0,
            'execution_penalty': execution_penalty
        }

if __name__ == "__main__":
    # Regression check for final hardening
    engine = MatchingEngine()
    # Test scalping friction with position_notional semantics
    res = engine.process_action(action=3, current_price=50000, position=1, position_notional=1000, leverage=1, spread=5, hold_duration=2)
    print(f"Scalp Exit Fee (Duration 2): {res['fee']}")
    
    # Test reversal
    res_rev = engine.process_action(action=2, current_price=50000, position=1, position_notional=1000, leverage=1, spread=5)
    print(f"Reversal Result: {res_rev['type']}, Penalty: {res_rev['execution_penalty']}")
    if 'close_fill_price' in res_rev and 'open_fill_price' in res_rev:
         print(f"Dual Fills: Close@{res_rev['close_fill_price']} Open@{res_rev['open_fill_price']}")
