# Institutional PPO Hyperparameters & Stabilizers
# Optimized for Market Microstructure & Event-Driven RL

# 1. PPO Policy Config
PPO_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 2.5e-4,     # Standard stable LR for financial PPO
    "n_steps": 2048,             # Rollout length per iteration
    "batch_size": 64,            # Minibatch size (keep small for noise handling)
    "n_epochs": 10,              # Optimization epochs
    "gamma": 0.99,               # Discount factor (long-term equity growth)
    "gae_lambda": 0.95,          # GAE for variance reduction
    "clip_range": 0.2,           # Policy clipping
    "ent_coef": 0.01,            # Entropy coefficient (prevents premature convergence)
    "vf_coef": 0.5,              # Value function weight
    "max_grad_norm": 0.5,        # Gradient clipping for stability
}

# 2. Institutional Stabilizers
STABILIZER_CONFIG = {
    "reward_scaling": True,      # Normalize rewards using running variance
    "obs_normalization": True,   # Pre-frozen normalization already in Gateway
    "gradient_clipping": True,
    "kl_early_stopping": 0.02,   # Stop update if KL divergence is too high
    "seed": 42,                  # Absolute seed for reproducibility
}

# 3. Diagnostic Metrics to Track
TRACKING_METRICS = [
    "pnl_daily",
    "drawdown_max",
    "sharpe_ratio",
    "action_distribution",
    "fill_rate",
    "slippage_bps",
    "psi_drift",
    "policy_entropy"
]
