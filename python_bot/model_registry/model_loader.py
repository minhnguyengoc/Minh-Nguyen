import os
import logging
from pathlib import Path
from stable_baselines3 import PPO
from python_bot.core.exceptions import ModelIncompatibilityError
from python_bot.model_registry.checkpoint_validator import CheckpointValidator
from python_bot.model_registry.model_metadata import ModelMetadata

logger = logging.getLogger("VNStockBot.ModelLoader")

class ModelLoader:
    """Handles loading and compatibility checks for RL models."""
    
    @classmethod
    def load_and_verify(cls, checkpoint_path: Path, env) -> Tuple[PPO, ModelMetadata]:
        """
        Loads the PPO model and checks space and feature dimensions against the env.
        Raises ModelIncompatibilityError on mismatch.
        """
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        is_ok, err = CheckpointValidator.validate_file_structure(checkpoint_path)
        if not is_ok:
            raise ModelIncompatibilityError(f"Checkpoint structure invalid: {err}")
            
        try:
            model = PPO.load(str(checkpoint_path))
        except Exception as e:
            raise ModelIncompatibilityError(f"Failed to execute Stable-Baselines3 model load: {e}")
            
        # Match spaces
        model_obs_shape = model.observation_space.shape[0]
        env_obs_shape = env.observation_space.shape[0]
        
        if model_obs_shape != env_obs_shape:
            raise ModelIncompatibilityError(
                f"OBSERVATION_DIM_MISMATCH: checkpoint expects {model_obs_shape} dimensions, "
                f"but environment defined {env_obs_shape} dimensions."
            )
            
        # Match actions
        try:
            model_act_dim = model.action_space.n
        except AttributeError:
            model_act_dim = model.action_space.shape[0]
            
        try:
            env_act_dim = env.action_space.n
        except AttributeError:
            env_act_dim = env.action_space.shape[0]
            
        if model_act_dim != env_act_dim:
            raise ModelIncompatibilityError(
                f"ACTION_DIM_MISMATCH: checkpoint expects action space n={model_act_dim}, "
                f"but environment defines action space n={env_act_dim}."
            )
            
        # Compile metadata
        meta = ModelMetadata(
            checkpoint_name=checkpoint_path.name,
            algorithm="PPO",
            observation_dim=model_obs_shape,
            action_dim=model_act_dim,
            feature_columns=getattr(env, 'feature_columns', []),
            hyperparameters={
                "learning_rate": getattr(model, 'learning_rate', None),
                "n_steps": getattr(model, 'n_steps', None),
                "batch_size": getattr(model, 'batch_size', None),
                "ent_coef": getattr(model, 'ent_coef', None)
            }
        )
        
        logger.info(f"🎯 Checkpoint compatibility verified. Obs={model_obs_shape}, Act={model_act_dim}")
        return model, meta
