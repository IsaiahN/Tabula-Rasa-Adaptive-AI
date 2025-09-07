"""
Data Paths Configuration for Tabula Rasa Training System

This module defines all data storage paths used throughout the system.
Centralized path management for easy maintenance and updates.
"""

from pathlib import Path
import os

# Base data directory
BASE_DATA_DIR = Path("data")

class DataPaths:
    """Centralized data path configuration for the training system."""
    
    # === ROOT DIRECTORIES ===
    BASE = BASE_DATA_DIR
    
    # === TRAINING DATA ===
    TRAINING = BASE / "training"
    TRAINING_SESSIONS = TRAINING / "sessions"
    TRAINING_RESULTS = TRAINING / "results"
    TRAINING_INTELLIGENCE = TRAINING / "intelligence" 
    TRAINING_META_LEARNING = TRAINING / "meta_learning"
    
    # === LOGGING ===
    LOGS = BASE / "logs"
    LOGS_TRAINING = LOGS / "training"
    LOGS_SYSTEM = LOGS / "system"
    LOGS_GOVERNOR = LOGS / "governor"
    LOGS_ARCHIVED = LOGS / "archived"
    
    # === MEMORY SYSTEM ===
    MEMORY = BASE / "memory"
    MEMORY_PERSISTENT = MEMORY / "persistent"
    MEMORY_BACKUPS = MEMORY / "backups"
    MEMORY_PATTERNS = MEMORY / "patterns"
    MEMORY_CROSS_SESSION = MEMORY / "cross_session"
    
    # === ARCHITECTURE SYSTEM ===
    ARCHITECTURE = BASE / "architecture"
    ARCHITECTURE_EVOLUTION = ARCHITECTURE / "evolution"
    ARCHITECTURE_MUTATIONS = ARCHITECTURE / "mutations"
    ARCHITECTURE_INSIGHTS = ARCHITECTURE / "insights"
    ARCHITECTURE_STRATEGIES = ARCHITECTURE / "strategies"
    
    # === EXPERIMENTS ===
    EXPERIMENTS = BASE / "experiments"
    EXPERIMENTS_RESEARCH = EXPERIMENTS / "research"
    EXPERIMENTS_PHASE0 = EXPERIMENTS / "phase0"
    EXPERIMENTS_EVALUATIONS = EXPERIMENTS / "evaluations"
    EXPERIMENTS_PERFORMANCE = EXPERIMENTS / "performance"
    
    # === CONFIGURATION ===
    CONFIG = BASE / "config"
    CONFIG_STATES = CONFIG / "states"
    CONFIG_COUNTERS = CONFIG / "counters"
    CONFIG_CACHE = CONFIG / "cache"
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all directories exist."""
        directories = [
            cls.TRAINING, cls.TRAINING_SESSIONS, cls.TRAINING_RESULTS, 
            cls.TRAINING_INTELLIGENCE, cls.TRAINING_META_LEARNING,
            cls.LOGS, cls.LOGS_TRAINING, cls.LOGS_SYSTEM, cls.LOGS_GOVERNOR, cls.LOGS_ARCHIVED,
            cls.MEMORY, cls.MEMORY_PERSISTENT, cls.MEMORY_BACKUPS, cls.MEMORY_PATTERNS, cls.MEMORY_CROSS_SESSION,
            cls.ARCHITECTURE, cls.ARCHITECTURE_EVOLUTION, cls.ARCHITECTURE_MUTATIONS, 
            cls.ARCHITECTURE_INSIGHTS, cls.ARCHITECTURE_STRATEGIES,
            cls.EXPERIMENTS, cls.EXPERIMENTS_RESEARCH, cls.EXPERIMENTS_PHASE0, 
            cls.EXPERIMENTS_EVALUATIONS, cls.EXPERIMENTS_PERFORMANCE,
            cls.CONFIG, cls.CONFIG_STATES, cls.CONFIG_COUNTERS, cls.CONFIG_CACHE
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_legacy_mapping(cls):
        """Get mapping from old paths to new paths for migration."""
        return {
            # Root data directory
            "data": str(cls.BASE),
            
            # Training paths
            "data/sessions": str(cls.TRAINING_SESSIONS),
            "data/meta_learning_data": str(cls.TRAINING_META_LEARNING),
            
            # Logging paths  
            "data/logs": str(cls.LOGS_TRAINING),
            "data/logs/governor_decisions": str(cls.LOGS_GOVERNOR),
            
            # Memory paths
            "data/backups": str(cls.MEMORY_BACKUPS),
            "data/memory_backups": str(cls.MEMORY_BACKUPS),
            
            # Architecture paths
            "data/architect_evolution_data": str(cls.ARCHITECTURE_EVOLUTION),
            "data/mutations": str(cls.ARCHITECTURE_MUTATIONS),
            
            # Experiments
            "data/adaptive_learning_agi_evaluation": str(cls.EXPERIMENTS_EVALUATIONS),
            "data/phase0_experiment_results": str(cls.EXPERIMENTS_PHASE0),
            "data/performance_optimization_data": str(cls.EXPERIMENTS_PERFORMANCE),
            
            # Legacy directories
            "data/base_meta_learning": str(cls.TRAINING_META_LEARNING),
            "data/meta_cognitive": str(cls.MEMORY_PERSISTENT),
        }

# === SPECIFIC FILE PATHS ===
class FilePaths:
    """Specific file paths used by the system."""
    
    # Main output logs
    MASTER_TRAINER_OUTPUT = DataPaths.LOGS_TRAINING / "master_arc_trainer_output.log"
    MASTER_TRAINER_ERROR = DataPaths.LOGS_TRAINING / "master_arc_trainer_error.log"
    MASTER_TRAINER_LOG = DataPaths.LOGS_SYSTEM / "master_arc_trainer.log"
    
    # Governor logs
    GOVERNOR_DECISIONS = DataPaths.LOGS_GOVERNOR / "governor_decisions.log"
    META_COGNITIVE_GOVERNOR = DataPaths.LOGS_SYSTEM / "meta_cognitive_governor.log"
    
    # Session data
    GLOBAL_COUNTERS = DataPaths.CONFIG_COUNTERS / "global_counters.json"
    LEARNED_PATTERNS = DataPaths.MEMORY_PATTERNS / "learned_patterns.pkl"
    
    # Training state
    PERSISTENT_LEARNING_STATE = DataPaths.CONFIG_STATES / "persistent_learning_state.json"
    TRAINING_STATE_BACKUP = DataPaths.CONFIG_STATES / "training_state_backup.json"
    
    # Research and results
    RESEARCH_RESULTS = DataPaths.EXPERIMENTS_RESEARCH / "research_results.json"
    UNIFIED_TRAINER_RESULTS = DataPaths.TRAINING_RESULTS / "unified_trainer_results.json"
    TASK_PERFORMANCE = DataPaths.TRAINING_RESULTS / "task_performance.json"

def get_data_path(*path_components) -> Path:
    """Helper function to get data paths relative to base data directory."""
    return DataPaths.BASE.joinpath(*path_components)

def get_training_path(*path_components) -> Path:
    """Helper function to get training data paths.""" 
    return DataPaths.TRAINING.joinpath(*path_components)

def get_log_path(*path_components) -> Path:
    """Helper function to get log paths."""
    return DataPaths.LOGS.joinpath(*path_components)

def get_memory_path(*path_components) -> Path:
    """Helper function to get memory paths."""
    return DataPaths.MEMORY.joinpath(*path_components)

def get_architecture_path(*path_components) -> Path:
    """Helper function to get architecture paths."""
    return DataPaths.ARCHITECTURE.joinpath(*path_components)

def get_experiment_path(*path_components) -> Path:
    """Helper function to get experiment paths."""
    return DataPaths.EXPERIMENTS.joinpath(*path_components)

def get_config_path(*path_components) -> Path:
    """Helper function to get config paths."""
    return DataPaths.CONFIG.joinpath(*path_components)

# Initialize directories when module is imported
DataPaths.ensure_directories()

# Legacy compatibility - will be removed after migration
LEGACY_BASE_DIR = "data"

def get_migrated_path(old_path: str) -> str:
    """Convert old path format to new organized structure."""
    mapping = DataPaths.get_legacy_mapping()
    
    # Direct mapping
    if old_path in mapping:
        return mapping[old_path]
    
    # Pattern matching for common cases
    for old_pattern, new_base in mapping.items():
        if old_path.startswith(old_pattern):
            relative_path = old_path[len(old_pattern):].lstrip('/')
            return str(Path(new_base) / relative_path)
    
    # Fallback - place in appropriate category based on filename
    filename = Path(old_path).name
    
    if 'session' in filename.lower():
        return str(DataPaths.TRAINING_SESSIONS / filename)
    elif 'intelligence' in filename.lower():
        return str(DataPaths.TRAINING_INTELLIGENCE / filename)  
    elif 'meta_learning' in filename.lower():
        return str(DataPaths.TRAINING_META_LEARNING / filename)
    elif filename.endswith('.log'):
        return str(DataPaths.LOGS_TRAINING / filename)
    elif 'governor' in filename.lower():
        return str(DataPaths.LOGS_GOVERNOR / filename)
    elif 'architect' in filename.lower():
        return str(DataPaths.ARCHITECTURE_EVOLUTION / filename)
    elif 'mutation' in filename.lower():
        return str(DataPaths.ARCHITECTURE_MUTATIONS / filename)
    elif 'research' in filename.lower():
        return str(DataPaths.EXPERIMENTS_RESEARCH / filename)
    elif 'backup' in filename.lower():
        return str(DataPaths.MEMORY_BACKUPS / filename)
    else:
        # Default to base training directory
        return str(DataPaths.TRAINING / filename)
