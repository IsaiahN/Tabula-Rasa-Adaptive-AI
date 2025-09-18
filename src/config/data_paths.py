"""
Data Paths Configuration for Tabula Rasa Training System

This module defines all data storage paths used throughout the system.
Centralized path management for easy maintenance and updates.
"""

from pathlib import Path
import os

# Database-only mode: No file-based data directory
BASE_DATA_DIR = None

class DataPaths:
    """Centralized data path configuration for the training system."""
    
    # === ROOT DIRECTORIES ===
    # Database-only mode: No file-based directories
    BASE = None
    
    # === TRAINING DATA ===
    # Database-only mode: All data stored in database
    TRAINING = None
    TRAINING_SESSIONS = None
    TRAINING_RESULTS = None
    TRAINING_INTELLIGENCE = None
    TRAINING_META_LEARNING = None
    
    # === LOGGING ===
    # Database-only mode: All logs stored in database
    LOGS = None
    LOGS_TRAINING = None
    LOGS_SYSTEM = None
    LOGS_GOVERNOR = None
    LOGS_ARCHIVED = None
    
    # === MEMORY SYSTEM ===
    # Database-only mode: All memory stored in database
    MEMORY = None
    MEMORY_PERSISTENT = None
    MEMORY_BACKUPS = None
    MEMORY_PATTERNS = None
    MEMORY_CROSS_SESSION = None
    
    # === ARCHITECTURE SYSTEM ===
    # Database-only mode: All architecture data stored in database
    ARCHITECTURE = None
    ARCHITECTURE_EVOLUTION = None
    ARCHITECTURE_MUTATIONS = None
    ARCHITECTURE_INSIGHTS = None
    ARCHITECTURE_STRATEGIES = None
    
    # === EXPERIMENTS ===
    # Database-only mode: All experiment data stored in database
    EXPERIMENTS = None
    EXPERIMENTS_RESEARCH = None
    EXPERIMENTS_PHASE0 = None
    EXPERIMENTS_EVALUATIONS = None
    EXPERIMENTS_PERFORMANCE = None
    
    # === CONFIGURATION ===
    # Database-only mode: All config stored in database
    CONFIG = None
    CONFIG_STATES = None
    CONFIG_COUNTERS = None
    CONFIG_CACHE = None
    
    # === BACKUP PATHS ===
    # Database-only mode: No file-based backups
    TRAINING_STATE_BACKUP = None
    
    @classmethod
    def ensure_directories(cls):
        """Database-only mode: No directories to create."""
        # All data is stored in database, no file directories needed
        pass
    
    @classmethod
    def get_legacy_mapping(cls):
        """Database-only mode: No file-based migration needed."""
        return {
            # All data is now stored in database
            "data": "database",
            "data/sessions": "database",
            "data/meta_learning_data": "database",
            "data/logs": "database",
            "data/logs/governor_decisions": "database",
            "data/backups": "database",
            "data/memory_backups": "database",
            "data/architect_evolution_data": "database",
            "data/mutations": "database",
            "data/adaptive_learning_agi_evaluation": "database",
            "data/phase0_experiment_results": "database",
            "data/performance_optimization_data": "database",
            "data/base_meta_learning": "database",
            "data/meta_cognitive": "database",
        }

# === SPECIFIC FILE PATHS ===
class FilePaths:
    """Database-only mode: No file paths needed."""
    
    # Database-only mode: All data stored in database
    MASTER_TRAINER_OUTPUT = None
    MASTER_TRAINER_ERROR = None
    MASTER_TRAINER_LOG = None
    
    # Governor logs
    GOVERNOR_DECISIONS = None
    META_COGNITIVE_GOVERNOR = None
    
    # Session data
    GLOBAL_COUNTERS = None
    LEARNED_PATTERNS = None
    
    # Training state
    PERSISTENT_LEARNING_STATE = None
    TRAINING_STATE_BACKUP = None
    
    # Research and results
    RESEARCH_RESULTS = None
    UNIFIED_TRAINER_RESULTS = None
    TASK_PERFORMANCE = None

def get_data_path(*path_components) -> None:
    """Database-only mode: No file paths."""
    return None

def get_training_path(*path_components) -> None:
    """Database-only mode: No file paths."""
    return None

def get_log_path(*path_components) -> None:
    """Database-only mode: No file paths."""
    return None

def get_memory_path(*path_components) -> None:
    """Database-only mode: No file paths."""
    return None

def get_architecture_path(*path_components) -> None:
    """Database-only mode: No file paths."""
    return None

def get_experiment_path(*path_components) -> None:
    """Database-only mode: No file paths."""
    return None

def get_config_path(*path_components) -> None:
    """Database-only mode: No file paths."""
    return None

# Initialize directories when module is imported
DataPaths.ensure_directories()

# Legacy compatibility - will be removed after migration
LEGACY_BASE_DIR = "data"

def get_migrated_path(old_path: str) -> str:
    """Database-only mode: All data is stored in database."""
    return "database"
