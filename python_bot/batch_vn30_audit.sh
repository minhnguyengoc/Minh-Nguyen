#!/bin/bash

# Institutional VN30 Multi-Stock Validation Suite
# Targets top 10 liquid stocks to prevent single-ticker overfitting.

TICKERS=("FPT" "SSI" "HPG" "VNM" "TCB" "VCB" "MBB" "ACB" "MWG" "VIC")
STEPS=200000
LOG_DIR="logs/batch_audit_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"
mkdir -p "checkpoints"

echo "🚀 Starting VN30 Institutional Batch Audit..."
echo "Output Directory: $LOG_DIR"
echo "-----------------------------------------------"

# Summary file for easy viewing
SUMMARY_FILE="$LOG_DIR/summary_report.md"
echo "# VN30 Batch Audit Summary - $(date)" > "$SUMMARY_FILE"
echo "| Ticker | Stability | Avg OOS Return | Status |" >> "$SUMMARY_FILE"
echo "|--------|-----------|----------------|--------|" >> "$SUMMARY_FILE"

for TICKER in "${TICKERS[@]}"
do
    echo "🔍 Processing $TICKER..."
    
    # 1. Train Model
    echo "   [1/2] Training RL Policy..."
    python3 python_bot/trainer.py --ticker "$TICKER" --steps "$STEPS" > "$LOG_DIR/${TICKER}_train.log" 2>&1
    
    # Check if training succeeded (model file exists)
    # Use tr for lowercase to support broader bash versions (e.g., in some Colab instances)
    TICKER_LOWER=$(echo "$TICKER" | tr '[:upper:]' '[:lower:]')
    MODEL_PATH="checkpoints/ppo_${TICKER_LOWER}_intraday.zip"
    
    if [ -f "$MODEL_PATH" ]; then
        echo "   [2/2] Running Institutional Validation..."
        # 2. Validate Out-of-Sample
        python3 python_bot/validator.py --ticker "$TICKER" --model "$MODEL_PATH" --folds 3 > "$LOG_DIR/${TICKER}_val.log" 2>&1
        
        # Extract summary from log
        STABILITY=$(grep "Stability:" "$LOG_DIR/${TICKER}_val.log" | grep -oE "PASSED|FAILED" || echo "UNKNOWN")
        RETURN=$(grep "Avg Out-of-Sample Return:" "$LOG_DIR/${TICKER}_val.log" | grep -oE "[-+]?[0-9]*\.?[0-9]+%" || echo "N/A")
        
        echo "   ✅ $TICKER Complete | Stability: $STABILITY | Avg Return: $RETURN"
        echo "| $TICKER | $STABILITY | $RETURN | COMPLETED |" >> "$SUMMARY_FILE"
    else
        echo "   ❌ $TICKER Training Failed. Check logs/batch_audit_*/${TICKER}_train.log"
        echo "| $TICKER | ERROR | N/A | FAILED |" >> "$SUMMARY_FILE"
    fi
    echo "-----------------------------------------------"
done

echo "🏆 VN30 Batch Audit Complete."
echo "Summary saved to: $SUMMARY_FILE"
echo "All logs available in $LOG_DIR"
