"""
Log rotation utility for managing log file sizes
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
import logging
from datetime import datetime

from src.config.log_config import LogConfig


class LogRotator:
    """Handles log file rotation and cleanup."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = LogConfig()
    
    def rotate_log_file(self, log_path: str, backup_path: str, max_lines: Optional[int] = None) -> bool:
        """
        Rotate a log file by keeping only the most recent lines.
        
        Args:
            log_path: Path to the log file to rotate
            backup_path: Path to store the backup
            max_lines: Maximum lines to keep (uses config default if None)
            
        Returns:
            True if rotation was successful, False otherwise
        """
        try:
            if not os.path.exists(log_path):
                self.logger.warning(f"Log file {log_path} does not exist, skipping rotation")
                return True
            
            # Get line count
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                total_lines = sum(1 for _ in f)
            
            if total_lines <= (max_lines or self.config.get_max_lines()):
                self.logger.debug(f"Log file {log_path} has {total_lines} lines, no rotation needed")
                return True
            
            self.logger.info(f"Rotating log file {log_path} ({total_lines} lines)")
            
            # Create backup of current file
            if os.path.exists(backup_path):
                os.remove(backup_path)
            shutil.copy2(log_path, backup_path)
            
            # Keep only the most recent lines
            lines_to_keep = max_lines or self.config.get_max_lines()
            temp_file = f"{log_path}.tmp"
            
            with open(log_path, 'r', encoding='utf-8', errors='replace') as infile, \
                 open(temp_file, 'w', encoding='utf-8') as outfile:
                
                # Skip to the last N lines
                lines = infile.readlines()
                if len(lines) > lines_to_keep:
                    lines_to_write = lines[-lines_to_keep:]
                else:
                    lines_to_write = lines
                
                outfile.writelines(lines_to_write)
            
            # Replace original with trimmed version
            os.replace(temp_file, log_path)
            
            self.logger.info(f"Log rotation complete: {log_path} now has {len(lines_to_write)} lines")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rotate log file {log_path}: {e}")
            return False
    
    def rotate_all_logs(self, max_lines: Optional[int] = None) -> bool:
        """
        Rotate all configured log files.
        
        Args:
            max_lines: Maximum lines to keep (uses config default if None)
            
        Returns:
            True if all rotations were successful, False otherwise
        """
        success = True
        
        # Rotate master_arc_trainer.log
        if not self.rotate_log_file(
            self.config.MASTER_ARC_TRAINER_LOG,
            self.config.MASTER_ARC_TRAINER_BACKUP,
            max_lines
        ):
            success = False
        
        # Rotate master_arc_trainer_output.log
        if not self.rotate_log_file(
            self.config.MASTER_ARC_TRAINER_OUTPUT_LOG,
            self.config.MASTER_ARC_TRAINER_OUTPUT_BACKUP,
            max_lines
        ):
            success = False
        
        return success
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> bool:
        """
        Clean up old log files and archives.
        
        Args:
            days_to_keep: Number of days to keep log files
            
        Returns:
            True if cleanup was successful, False otherwise
        """
        try:
            log_dir = Path("data/logs")
            if not log_dir.exists():
                return True
            
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
            cleaned_count = 0
            
            for log_file in log_dir.glob("*.log"):
                if log_file.stat().st_mtime < cutoff_time:
                    self.logger.info(f"Removing old log file: {log_file}")
                    log_file.unlink()
                    cleaned_count += 1
            
            # Clean up archive directory
            archive_dir = Path(self.config.LOG_ARCHIVE_DIR)
            if archive_dir.exists():
                for archive_file in archive_dir.glob("*.log"):
                    if archive_file.stat().st_mtime < cutoff_time:
                        self.logger.info(f"Removing old archive: {archive_file}")
                        archive_file.unlink()
                        cleaned_count += 1
            
            self.logger.info(f"Cleaned up {cleaned_count} old log files")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old logs: {e}")
            return False
    
    def get_log_stats(self) -> dict:
        """Get statistics about log files."""
        stats = {}
        
        for log_name, log_path in [
            ("master_arc_trainer", self.config.MASTER_ARC_TRAINER_LOG),
            ("master_arc_trainer_output", self.config.MASTER_ARC_TRAINER_OUTPUT_LOG)
        ]:
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                    line_count = sum(1 for _ in f)
                file_size = os.path.getsize(log_path)
                stats[log_name] = {
                    "lines": line_count,
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "path": log_path
                }
            else:
                stats[log_name] = {"lines": 0, "size_mb": 0, "path": log_path}
        
        return stats
    
    def should_rotate(self, log_path: str, max_lines: Optional[int] = None) -> bool:
        """Check if a log file should be rotated."""
        if not os.path.exists(log_path):
            return False
        
        try:
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                line_count = sum(1 for _ in f)
            return line_count > (max_lines or self.config.get_max_lines())
        except Exception:
            return False
