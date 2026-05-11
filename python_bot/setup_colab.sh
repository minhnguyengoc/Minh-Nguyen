#!/bin/bash
# Institutional Setup Script for Google Colab

echo "🚀 Starting Institutional Environment Setup..."

# Determine the absolute path of the repository root
# This script is located in <root>/python_bot/setup_colab.sh
SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( dirname "$SCRIPT_PATH" )"
cd "$REPO_ROOT"

echo "📍 Working Directory: $PWD"

# 1. Sync from GitHub if in a git repo
if [ -d ".git" ]; then
    echo "🔄 Syncing with remote repository..."
    git fetch origin main
    git reset --hard origin/main
fi

# 3. Clean stale artifacts and legacy gym
echo "🧹 Cleaning environment..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
pip uninstall gym -y 2>/dev/null || true

# 4. Install dependencies
echo "📦 Installing Requirements..."
pip install -r python_bot/requirements.txt

# 4. Set PYTHONPATH globally for the session
# We add the REPO_ROOT so that 'import python_bot' works from anywhere
export PYTHONPATH=$PYTHONPATH:$REPO_ROOT
echo "🌐 PYTHONPATH updated: $PYTHONPATH"

# 5. Verify installation
python -c "import torch; import stable_baselines3; print('✅ Core ML Stack Verified')"
python -c "import sys; sys.path.append('$REPO_ROOT'); from python_bot.ppo_agent import PPOAgent; print('✅ python_bot Module Discovery Verified')"

# Create necessary directories
mkdir -p logs/training
mkdir -p logs/evaluation
mkdir -p models/checkpoints

echo "✅ Environment Ready for Training."
