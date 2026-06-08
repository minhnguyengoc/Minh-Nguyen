import os
import re
import logging
from typing import Optional

logger = logging.getLogger("VNStockBot.SecretManager")

class SecretManager:
    """Safeguards access to API credentials, keys, and tokens without console exposure."""
    
    SECRET_PATTERN = re.compile(r'(api[_-]?key|password|secret|credential|token|auth)', re.IGNORECASE)
    
    @classmethod
    def get_secured_env(cls, key: str, default: Optional[str] = None) -> Optional[str]:
        """Fetches environment variable safely. Suppresses printing value if it matches secret regex matches."""
        val = os.getenv(key, default)
        if val is not None:
            if cls.SECRET_PATTERN.search(key):
                logger.debug(f"Retrieved sensitive environment parameter: {key}=***MASKED***")
            else:
                logger.debug(f"Retrieved environment parameter: {key}={val}")
        return val

    @classmethod
    def clean_log_message(cls, message: str) -> str:
        """Scrubs accidental plain text secrets inside logging statements."""
        # Simple mask for expressions like api_key=abc123xyz
        scrubbed = re.sub(r'(api_key|password|token|secret)\s*=\s*[a-zA-Z0-9_\-]+', r'\1=***SCREENSCRUBBED***', message, flags=re.IGNORECASE)
        return scrubbed
