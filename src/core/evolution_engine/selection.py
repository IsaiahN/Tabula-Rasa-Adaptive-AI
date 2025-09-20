#!/usr/bin/env python3
"""
Selection Strategy - Implements selection strategies for evolution.
"""

import logging
import random
from typing import List, Dict, Any

class SelectionStrategy:
    """Implements selection strategies for evolution."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def select_parents(self, population: List[Dict[str, Any]], 
                      fitness_scores: List[float]) -> List[Dict[str, Any]]:
        """Select parents for reproduction."""
        try:
            # Tournament selection
            parents = []
            tournament_size = 3
            
            for _ in range(len(population) // 2):  # Select half the population as parents
                # Select tournament participants
                tournament_indices = random.sample(range(len(population)), 
                                                min(tournament_size, len(population)))
                
                # Find best in tournament
                best_index = max(tournament_indices, key=lambda i: fitness_scores[i])
                parents.append(population[best_index])
            
            return parents
            
        except Exception as e:
            self.logger.error(f"Parent selection failed: {e}")
            # Fallback: return random parents
            return random.sample(population, min(len(population) // 2, len(population)))
