"""
Log configuration and rotation settings
"""

class LogConfig:
    """Configuration for log file management and rotation."""
    
    # Log file limits
    MAX_LOG_LINES = 100000  # Maximum lines per log file before rotation
    BACKUP_LOG_LINES = 50000  # Lines to keep in backup when rotating
    
    # Database-only mode: No file-based logging
    MASTER_ARC_TRAINER_LOG = None
    MASTER_ARC_TRAINER_OUTPUT_LOG = None
    
    # Backup file paths
    MASTER_ARC_TRAINER_BACKUP = None
    MASTER_ARC_TRAINER_OUTPUT_BACKUP = None
    
    # Archive directory for old logs
    LOG_ARCHIVE_DIR = None
    
    @classmethod
    def get_max_lines(cls) -> int:
        """Get the maximum number of lines per log file."""
        return cls.MAX_LOG_LINES
    
    @classmethod
    def set_max_lines(cls, max_lines: int) -> None:
        """Set the maximum number of lines per log file."""
        cls.MAX_LOG_LINES = max_lines
    
    @classmethod
    def get_backup_lines(cls) -> int:
        """Get the number of lines to keep in backup."""
        return cls.BACKUP_LOG_LINES
    
    @classmethod
    def set_backup_lines(cls, backup_lines: int) -> None:
        """Set the number of lines to keep in backup."""
        cls.BACKUP_LOG_LINES = backup_lines
