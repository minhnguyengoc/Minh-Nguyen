#!/bin/bash
# Institutional Training Launcher

# Determine the absolute path of the repository root
SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( dirname "$SCRIPT_PATH" )"
cd "$REPO_ROOT"
export PYTHONPATH=$PYTHONPATH:$REPO_ROOT

echo "🚀 Launching Distributed Training Session..."
echo "📍 Working Directory: $PWD"
echo "🌐 PYTHONPATH: $PYTHONPATH"

# Default parameters
SYMBOL=${1:-"SSI"}
CONFIG=${2:-"python_bot/manifest/stage4_baseline.json"}

# Pre-flight check
python python_bot/preflight/training_gate.py --config $CONFIG

if [ $? -eq 0 ]; then
    echo "🟢 Pre-flight Check Passed. Initializing Trainer..."
    python python_bot/trainer.py --ticker $SYMBOL --config $CONFIG
else
    echo "🔴 Pre-flight Check Failed. Aborting."
    exit 1
fi
