"""
Configuration loading utilities with validation and defaults.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Configuration loader with validation and inheritance."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration with inheritance support.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            config: Loaded and merged configuration
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            # Try relative to config directory
            config_file = self.config_dir / config_path
            
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        # Handle inheritance
        if 'defaults' in config:
            base_configs = config['defaults']
            if isinstance(base_configs, str):
                base_configs = [base_configs]
                
            # Load and merge base configurations
            merged_config = {}
            for base_config in base_configs:
                base_path = self.config_dir / f"{base_config}.yaml"
                if base_path.exists():
                    with open(base_path, 'r') as f:
                        base = yaml.safe_load(f)
                    merged_config = self._deep_merge(merged_config, base)
                else:
                    logger.warning(f"Base config not found: {base_config}")
                    
            # Merge current config over base
            config = self._deep_merge(merged_config, config)
            
            # Remove defaults key
            config.pop('defaults', None)
            
        return config
        
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
        
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration structure and values.
        
        Args:
            config: Configuration to validate
            
        Returns:
            valid: True if configuration is valid
        """
        required_sections = [
            'learning_progress',
            'memory', 
            'energy',
            'environment',
            'training'
        ]
        
        # Check required sections
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required config section: {section}")
                return False
                
        # Validate learning progress config
        lp_config = config['learning_progress']
        if lp_config.get('smoothing_window', 0) <= 0:
            logger.error("Learning progress smoothing_window must be positive")
            return False
            
        # Validate memory config
        memory_config = config['memory']
        if memory_config.get('memory_size', 0) <= 0:
            logger.error("Memory size must be positive")
            return False
            
        # Validate energy config
        energy_config = config['energy']
        if energy_config.get('max_energy', 0) <= 0:
            logger.error("Max energy must be positive")
            return False
            
        return True
        
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'learning_progress': {
                'smoothing_window': 500,
                'derivative_clamp': [-1.0, 1.0],
                'boredom_threshold': 0.01,
                'boredom_steps': 500,
                'lp_weight': 0.7,
                'empowerment_weight': 0.3,
                'use_adaptive_weights': False
            },
            'memory': {
                'memory_size': 512,
                'word_size': 64,
                'num_read_heads': 4,
                'num_write_heads': 1,
                'controller_size': 256
            },
            'energy': {
                'max_energy': 100.0,
                'base_consumption': 0.01,
                'action_multiplier': 0.5,
                'computation_multiplier': 0.001,
                'food_energy_value': 10.0
            },
            'environment': {
                'visual_size': [3, 64, 64],
                'proprioception_size': 12,
                'action_space_size': 8,
                'max_episode_steps': 10000
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'gradient_clip': 1.0,
                'device': 'cpu'
            }
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """Simple function to load configuration file."""
    loader = ConfigLoader()
    return loader.load_config(config_path)