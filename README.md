# Stage 4.1: Institutional Validation & Harsh Backtesting

The system now includes a specialized validation layer designed to destroy overfitted strategies and quantify real edge.

### 🏛️ Backtest Architecture
- **Event-Driven Execution**: Bar-by-bar simulation with T+1 fill reality.
- **T+2 Settlement**: Accurate tracking of available cash vs. locked T+2 shares.
- **Lot Size 100**: Mandatory HOSE lot constraints on all orders.
- **Square-Root Market Impact**: Larger orders cause realistic price Slippage.

### 🛡️ Stress Modes
| Mode | Purpose | Slippage | Impact | Fill Ratio | Broker Fee |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NORMAL** | Standard institutional trading | 5 bps | 10% | 80%+ | 15 bps |
| **STRESS** | High volatility / Flash crash | 20 bps | 25% | 50%+ | 20 bps |
| **HELL** | Maximum Pessimism (Liquidity Crisis) | 50 bps | 50% | 20%+ | 30 bps |

### 🚀 Running the Validation
To run the harsh backtest suite against a trained checkpoint:
```bash
export PYTHONPATH=$PYTHONPATH:.
python -m python_bot.scripts.run_harsh_backtest --symbols FPT,MWG,SSI --mode HELL
```

### 📊 Understanding the Haircut
Institutional traders never trust raw backtest numbers. We apply a **50% Haircut** to all net returns to account for:
- Latency & Alpha Decay.
- Hidden corporate actions.
- Unmodeled market edge cases.
- Data artifacts.

**A PASS verdict is only granted if the strategy survives HELL mode with positive expectancy.**
