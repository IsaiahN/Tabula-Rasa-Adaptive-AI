"""
Centralized configuration module to eliminate duplication between master_arc_trainer.py and continuous_learning_loop.py

This module consolidates all shared configuration, constants, and utilities
to prevent code duplication and ensure consistency.
"""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ActionLimits:
    """Centralized action limits configuration."""
    MAX_ACTIONS_PER_GAME = 2000
    MAX_ACTIONS_PER_SESSION = 5000
    MAX_ACTIONS_PER_SCORECARD = 8000
    MAX_ACTIONS_PER_EPISODE = 1500
    MAX_ACTIONS_SCALING_BASE = 800
    MAX_ACTIONS_SCALING_MAX = 3000
    
    @classmethod
    def get_max_actions_per_game(cls) -> int:
        return cls.MAX_ACTIONS_PER_GAME
    
    @classmethod
    def get_max_actions_per_session(cls) -> int:
        return cls.MAX_ACTIONS_PER_SESSION
    
    @classmethod
    def get_max_actions_per_scorecard(cls) -> int:
        return cls.MAX_ACTIONS_PER_SCORECARD


@dataclass
class APIConfig:
    """Centralized API configuration."""
    ARC3_BASE_URL = "https://three.arcprize.org"
    ARC3_SCOREBOARD_URL = "https://three.arcprize.org/scorecards"
    
    # Rate Limiting Configuration for ARC-AGI-3 API
    RATE_LIMIT = {
        'requests_per_minute': 600,
        'requests_per_second': 10,
        'safe_requests_per_second': 8,
        'backoff_base_delay': 1.0,
        'backoff_max_delay': 60.0,
        'request_timeout': 30.0
    }
    
    @classmethod
    def get_api_key(cls) -> Optional[str]:
        """Get API key from environment or return None."""
        return os.getenv('ARC_API_KEY')
    
    @classmethod
    def validate_api_key(cls, api_key: Optional[str]) -> bool:
        """Validate API key format."""
        if not api_key:
            return False
        return len(api_key) > 10  # Basic validation


@dataclass
class MemoryLimits:
    """Centralized memory limits to prevent memory leaks."""
    MAX_PERFORMANCE_HISTORY = 100
    MAX_SESSION_HISTORY = 100
    MAX_GOVERNOR_DECISIONS = 100
    MAX_ARCHITECT_EVOLUTIONS = 100
    MAX_ACTION_HISTORY = 1000
    MAX_COORDINATE_ATTEMPTS = 500
    MAX_FRAME_HISTORY = 50
    
    @classmethod
    def bound_list(cls, data_list: list, max_size: int) -> list:
        """Bound a list to maximum size, keeping most recent items."""
        if len(data_list) > max_size:
            return data_list[-max_size:]
        return data_list
    
    @classmethod
    def bound_dict(cls, data_dict: dict, max_size: int) -> dict:
        """Bound a dictionary to maximum size, keeping most recent items."""
        if len(data_dict) > max_size:
            # Convert to list of items, sort by key (assuming keys are timestamps or similar)
            items = list(data_dict.items())
            items.sort(key=lambda x: str(x[0]), reverse=True)
            return dict(items[:max_size])
        return data_dict


class LoggingConfig:
    """Centralized logging configuration."""
    
    @staticmethod
    def setup_logging(debug_mode: bool = False, verbose: bool = False) -> logging.Logger:
        """Setup centralized logging configuration."""
        log_level = logging.DEBUG if debug_mode else (
            logging.INFO if verbose else logging.WARNING
        )
        
        # Configure the root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        
        # Disable noisy loggers
        for logger_name in ['matplotlib', 'PIL', 'git']:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        
        return logging.getLogger('CentralizedLogger')
    
    @staticmethod
    def setup_windows_logging():
        """Set up logging that works properly on Windows with real-time terminal and file output."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        simple_format = '%(message)s'  # For terminal display
        
        # Create handlers with UTF-8 encoding
        handlers = []
        
        # Console handler with UTF-8 - handle Windows encoding issues
        try:
            import sys
            import codecs
            
            # Try to set console to UTF-8
            if hasattr(sys.stdout, 'reconfigure'):
                try:
                    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
                except:
                    pass
            
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # Use a formatter that handles Unicode gracefully
            class SafeFormatter(logging.Formatter):
                def format(self, record):
                    try:
                        return super().format(record)
                    except UnicodeEncodeError:
                        # Strip problematic characters and retry
                        record.msg = str(record.msg).encode('ascii', errors='replace').decode('ascii')
                        return super().format(record)
            
            console_handler.setFormatter(SafeFormatter(simple_format))  # Simple format for console
            
        except Exception:
            # Fallback: basic console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter(simple_format))
        
        # Create TeeHandler for simultaneous file and console output
        try:
            from master_arc_trainer import TeeHandler
            tee_handler = TeeHandler(
                file_path='data/logs/master_arc_trainer_output.log',
                console_handler=console_handler
            )
            tee_handler.setLevel(logging.INFO)
            tee_handler.setFormatter(SafeFormatter(log_format) if 'SafeFormatter' in locals() else logging.Formatter(log_format))
            handlers.append(tee_handler)
            
            # Also add separate file handler for detailed logs
            file_handler = logging.FileHandler('data/logs/master_arc_trainer.log', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(log_format))
            handlers.append(file_handler)
            
        except Exception as e:
            # Fallback to console-only
            handlers.append(console_handler)
            print(f"Warning: Could not set up file logging: {e}")
        
        # Configure root logger
        try:
            logging.basicConfig(
                level=logging.INFO,
                format=log_format,
                handlers=handlers,
                force=True  # Override existing config
            )
        except Exception:
            # Minimal fallback
            logging.basicConfig(level=logging.INFO, format=log_format)
        
        print("🚀 Enhanced logging initialized - real-time terminal and file output enabled")


class DatabaseConfig:
    """Centralized database configuration and utilities."""
    
    @staticmethod
    def get_database_path() -> Path:
        """Get the database path."""
        return Path("tabula_rasa.db")
    
    @staticmethod
    def ensure_database_ready() -> bool:
        """Ensure database is ready for use."""
        try:
            from src.database.db_initializer import ensure_database_ready
            return ensure_database_ready()
        except ImportError:
            print("WARNING: Database initialization not available")
            return False
    
    @staticmethod
    def get_director_commands():
        """Get director commands for database operations."""
        try:
            from src.database.director_commands import get_director_commands
            return get_director_commands()
        except ImportError:
            print("WARNING: Director commands not available")
            return None
    
    @staticmethod
    def get_system_integration():
        """Get system integration for database operations."""
        try:
            from src.database.system_integration import get_system_integration
            return get_system_integration()
        except ImportError:
            print("WARNING: System integration not available")
            return None


class MemoryManager:
    """Centralized memory management to prevent leaks."""
    
    def __init__(self):
        self.memory_limits = MemoryLimits()
    
    def bound_performance_history(self, history: list) -> list:
        """Bound performance history to prevent memory leaks."""
        return self.memory_limits.bound_list(history, self.memory_limits.MAX_PERFORMANCE_HISTORY)
    
    def bound_session_history(self, history: list) -> list:
        """Bound session history to prevent memory leaks."""
        return self.memory_limits.bound_list(history, self.memory_limits.MAX_SESSION_HISTORY)
    
    def bound_governor_decisions(self, decisions: list) -> list:
        """Bound governor decisions to prevent memory leaks."""
        return self.memory_limits.bound_list(decisions, self.memory_limits.MAX_GOVERNOR_DECISIONS)
    
    def bound_architect_evolutions(self, evolutions: list) -> list:
        """Bound architect evolutions to prevent memory leaks."""
        return self.memory_limits.bound_list(evolutions, self.memory_limits.MAX_ARCHITECT_EVOLUTIONS)
    
    def bound_action_history(self, history: list) -> list:
        """Bound action history to prevent memory leaks."""
        return self.memory_limits.bound_list(history, self.memory_limits.MAX_ACTION_HISTORY)
    
    def cleanup_old_data(self, data_dict: dict, max_age_seconds: int = 3600) -> dict:
        """Clean up old data based on timestamp."""
        import time
        current_time = time.time()
        
        cleaned = {}
        for key, value in data_dict.items():
            # Assume key contains timestamp or value has timestamp
            if isinstance(value, dict) and 'timestamp' in value:
                timestamp = value.get('timestamp', 0)
                if isinstance(timestamp, str):
                    try:
                        from datetime import datetime
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp()
                    except:
                        continue
                
                if current_time - timestamp < max_age_seconds:
                    cleaned[key] = value
            else:
                cleaned[key] = value
        
        return cleaned


# Global instances for easy access
action_limits = ActionLimits()
api_config = APIConfig()
memory_limits = MemoryLimits()
logging_config = LoggingConfig()
database_config = DatabaseConfig()
memory_manager = MemoryManager()
