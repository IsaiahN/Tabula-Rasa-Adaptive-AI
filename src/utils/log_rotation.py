"""
Log rotation utility for managing log file sizes
"""

import os
import shutil
import time
from pathlib import Path
from typing import List, Optional
import logging
from datetime import datetime
import platform

from src.config.log_config import LogConfig


class LogRotator:
    """Handles log file rotation and cleanup."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = LogConfig()
        self.is_windows = platform.system() == "Windows"
    
    def _safe_file_replace_windows(self, temp_file: str, log_path: str, max_retries: int = 3) -> bool:
        """Windows-safe file replacement with retry logic."""
        for attempt in range(max_retries):
            try:
                # Force garbage collection to close any file handles
                import gc
                gc.collect()
                
                # Small delay to let file handles close
                time.sleep(0.1)
                
                # Try to replace the file
                os.replace(temp_file, log_path)
                return True
                
            except (OSError, PermissionError) as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"File replace attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(0.5)  # Wait longer between retries
                else:
                    self.logger.error(f"All file replace attempts failed: {e}")
                    return False
        
        return False
    
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
            # Database-only mode: Skip file-based log rotation
            if log_path is None or backup_path is None:
                return True
                
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
            
            # Replace original with trimmed version (Windows-safe)
            if self.is_windows:
                # Use Windows-specific safe replacement
                if not self._safe_file_replace_windows(temp_file, log_path):
                    # Final fallback: copy method
                    self.logger.warning("Using copy method as final fallback")
                    try:
                        shutil.copy2(temp_file, log_path)
                        os.remove(temp_file)
                    except Exception as copy_error:
                        self.logger.error(f"Copy method failed: {copy_error}")
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                        return False
            else:
                # Unix/Linux: use atomic replace
                try:
                    os.replace(temp_file, log_path)
                except (OSError, PermissionError) as e:
                    self.logger.error(f"File replace failed: {e}")
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                    return False
            
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
        failed_logs = []
        
        # Rotate master_arc_trainer.log
        if not self.rotate_log_file(
            self.config.MASTER_ARC_TRAINER_LOG,
            self.config.MASTER_ARC_TRAINER_BACKUP,
            max_lines
        ):
            success = False
            failed_logs.append("master_arc_trainer.log")
        
        # Rotate master_arc_trainer_output.log
        if not self.rotate_log_file(
            self.config.MASTER_ARC_TRAINER_OUTPUT_LOG,
            self.config.MASTER_ARC_TRAINER_OUTPUT_BACKUP,
            max_lines
        ):
            success = False
            failed_logs.append("master_arc_trainer_output.log")
        
        # Log summary
        if failed_logs:
            self.logger.warning(f"Log rotation partially failed for: {', '.join(failed_logs)}")
            if self.is_windows:
                self.logger.info(" On Windows, this is often due to file locking. The system will continue normally.")
        else:
            self.logger.info(" All log files rotated successfully")
        
        return success
    
    def rotate_with_graceful_fallback(self, max_lines: Optional[int] = None) -> dict:
        """
        Rotate logs with graceful fallback for locked files.
        
        Returns:
            dict with rotation results and recommendations
        """
        results = {
            "success": True,
            "rotated_files": [],
            "failed_files": [],
            "recommendations": []
        }
        
        # Try to rotate each log file individually
        log_files = [
            (self.config.MASTER_ARC_TRAINER_LOG, self.config.MASTER_ARC_TRAINER_BACKUP, "master_arc_trainer.log"),
            (self.config.MASTER_ARC_TRAINER_OUTPUT_LOG, self.config.MASTER_ARC_TRAINER_OUTPUT_BACKUP, "master_arc_trainer_output.log")
        ]
        
        for log_path, backup_path, log_name in log_files:
            if self.rotate_log_file(log_path, backup_path, max_lines):
                results["rotated_files"].append(log_name)
            else:
                results["failed_files"].append(log_name)
                results["success"] = False
        
        # Add recommendations based on results
        if results["failed_files"]:
            if self.is_windows:
                results["recommendations"].append("Consider restarting the application to release file locks")
                results["recommendations"].append("The system will continue normally despite rotation failures")
            else:
                results["recommendations"].append("Check file permissions and disk space")
        
        return results
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> bool:
        """
        Clean up old log files and archives.
        
        Args:
            days_to_keep: Number of days to keep log files
            
        Returns:
            True if cleanup was successful, False otherwise
        """
        try:
            # Database-only mode: Skip file-based log rotation
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
