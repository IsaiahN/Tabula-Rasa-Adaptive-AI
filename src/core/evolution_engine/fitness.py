#!/usr/bin/env python3
"""
Fitness Evaluator - Evaluates fitness of system configurations.
"""

import logging
from typing import Dict, Any

class FitnessEvaluator:
    """Evaluates fitness of system configurations."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def evaluate(self, individual: Dict[str, Any]) -> float:
        """Evaluate fitness of an individual configuration."""
        try:
            # Simple fitness function based on configuration parameters
            fitness = 0.0
            
            # Salience threshold optimization (prefer moderate values)
            salience_threshold = individual.get('salience_threshold', 0.5)
            if 0.3 <= salience_threshold <= 0.7:
                fitness += 0.2
            else:
                fitness += 0.1
            
            # Action limit optimization (prefer reasonable values)
            max_actions = individual.get('max_actions_per_game', 500)
            if 300 <= max_actions <= 800:
                fitness += 0.2
            else:
                fitness += 0.1
            
            # Energy decay optimization (prefer moderate values)
            energy_decay = individual.get('energy_decay_rate', 0.02)
            if 0.01 <= energy_decay <= 0.05:
                fitness += 0.2
            else:
                fitness += 0.1
            
            # Feature combination bonuses
            if individual.get('enable_contrarian_strategy', False):
                fitness += 0.1
            
            if individual.get('enable_exploration_strategies', False):
                fitness += 0.1
            
            # Add some randomness for exploration
            import random
            fitness += random.uniform(-0.05, 0.05)
            
            return max(0.0, min(1.0, fitness))  # Clamp between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Fitness evaluation failed: {e}")
            return 0.0
