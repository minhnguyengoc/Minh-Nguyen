import os
import logging
import datetime
from python_bot.core.paths import LOGS_DIR

class SecurityAuditLogger:
    """Security audit log manager recording trade events and policy-breach alerts."""
    
    @classmethod
    def log_audit_trail(cls, action: str, details: str, status: str = "SUCCESS"):
        """Appends transactional operations and warning flags safely."""
        os.makedirs(LOGS_DIR, exist_ok=True)
        audit_file = LOGS_DIR / "security_audit.log"
        
        timestamp = datetime.datetime.now().isoformat()
        log_line = f"[{timestamp}] - {action.upper()} - {status.upper()} - {details}\n"
        
        try:
            with open(audit_file, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception as e:
            # Fall back to logging
            logging.getLogger("SecurityAuditLogger").error(f"Failed to record audit line: {e}")
            
    @classmethod
    def log_incident(cls, threat_type: str, details: str):
        """Specifically alerts on security and risk boundary escalations."""
        cls.log_audit_trail(action=f"INCIDENT_ALERT_{threat_type}", details=details, status="MALICIOUS_BREACH_WARN")
