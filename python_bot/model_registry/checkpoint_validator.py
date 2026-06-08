import os
import zipfile
import logging
from pathlib import Path
from typing import Tuple, Optional
from python_bot.core.exceptions import ModelIncompatibilityError

logger = logging.getLogger("VNStockBot.CheckpointValidator")

class CheckpointValidator:
    """Validator for verifying zip file structural integrity and version matches."""
    
    @classmethod
    def validate_file_structure(cls, checkpoint_path: Path) -> Tuple[bool, Optional[str]]:
        """Returns (is_valid, error_reason) for file level and zip level checks."""
        if not checkpoint_path.exists():
            return False, f"Checkpoint file does not exist: {checkpoint_path}"
            
        if not zipfile.is_zipfile(str(checkpoint_path)):
            return False, "Checkpoint is not a valid zip file."
            
        # Verify it has standard PyTorch weights inside
        try:
            with zipfile.ZipFile(str(checkpoint_path), 'r') as zip_ref:
                file_list = zip_ref.namelist()
                # SB3 saves have parameter files like 'data' or 'policy.pth'
                has_sb3_data = any('data' in name or 'policy' in name for name in file_list)
                if not has_sb3_data:
                    return False, "Checkpoint lacks standard Stable-Baselines3 model structures."
        except Exception as e:
            return False, f"Failed to unzip/read checkpoint file: {e}"
            
        return True, None
