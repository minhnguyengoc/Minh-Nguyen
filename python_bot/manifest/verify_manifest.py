import json
import os
import sys
import logging
from python_bot.training.env import VNStockInstitutionalEnv
from python_bot.training.reward import InstitutionalRewardEngine
from python_bot.engine.impact import InstitutionalImpactModel
from python_bot.engine.execution import HybridQueueReactiveExecutionSimulator
from python_bot.training.ppo_config import PPO_CONFIG, STABILIZER_CONFIG
from python_bot.engine.regime import InstitutionalRegimeDetector

def verify_baseline():
    """
    STRICT Institutional Baseline Verifier.
    Checks for any ARCHITECTURAL DRIFT in the trading infrastructure.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger("ManifestVerifier")

    manifest_path = os.path.join(os.path.dirname(__file__), "stage4_baseline.json")
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest not found at {manifest_path}")
        sys.exit(1)

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    logger.info(f"VERIFYING BASELINE: {manifest['manifest_version']} (Created: {manifest['created_timestamp']})")
    
    drift_detected = False
    
    def check(key, current, expected):
        nonlocal drift_detected
        if current != expected:
            logger.error(f"DRIFT ERROR [{key}]: Current({current}) != Expected({expected})")
            drift_detected = True

    # 1. Observation/Action Schema
    from python_bot.common.types import MarketData
    import datetime as dt
    history = [MarketData(symbol="V", event_id="e1", timestamp=dt.datetime.now(), received_at=dt.datetime.now(), open=100, high=101, low=99, close=100, volume=1000)]
    env = VNStockInstitutionalEnv(history)
    
    check("OBSERVATION_DIM", env.observation_space.shape[0], manifest['observation_schema']['dim'])
    check("ACTION_SIZE", env.action_space.n, manifest['action_schema']['size'])
    check("FEATURE_LENGTH", env.observation_space.shape[0], manifest['feature_vector_length'])

    # 2. PPO Hyperparameters
    ppo_expected = manifest['ppa_hyperparameters']
    check("PPO_LR", PPO_CONFIG['learning_rate'], ppo_expected['learning_rate'])
    check("PPO_STEPS", PPO_CONFIG['n_steps'], ppo_expected['n_steps'])
    check("PPO_BATCH", PPO_CONFIG['batch_size'], ppo_expected['batch_size'])
    check("PPO_GAMMA", PPO_CONFIG['gamma'], ppo_expected['gamma'])
    check("KL_LIMIT", STABILIZER_CONFIG['kl_early_stopping'], ppo_expected['kl_early_stopping'])

    # 3. Reward Engine
    reward = InstitutionalRewardEngine()
    re_expected = manifest['reward_decomposition']['weights']
    check("REWARD_DD_WEIGHT", reward.dd_weight, re_expected['drawdown_penalty'])
    check("REWARD_TC_PCT", reward.tc_pct, re_expected['transaction_cost_pct'])
    check("REWARD_CHURN_WEIGHT", reward.churn_weight, re_expected['turnover_penalty'])

    # 4. Execution Simulator
    sim = HybridQueueReactiveExecutionSimulator()
    check("SIM_SLIP_BETA", sim.slip_beta, manifest['execution_simulator']['parameters']['slip_beta'])
    
    impact = InstitutionalImpactModel()
    check("IMPACT_BETA", impact.beta, manifest['execution_simulator']['parameters']['temp_impact_beta'])
    check("IMPACT_GAMMA", impact.gamma, manifest['execution_simulator']['parameters']['perm_impact_gamma'])

    # 5. Regime Detector
    regime = InstitutionalRegimeDetector()
    rd_expected = manifest['regime_detector']
    check("REGIME_WINDOW", regime.window, rd_expected['window'])

    if drift_detected:
        logger.critical("ARCHITECTURE DRIFT DETECTED")
        sys.exit(1)
    else:
        logger.info("BASELINE VERIFIED")

if __name__ == "__main__":
    verify_baseline()
