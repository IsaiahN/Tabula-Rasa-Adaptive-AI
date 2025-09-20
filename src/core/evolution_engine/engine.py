#!/usr/bin/env python3
"""
Evolution Engine - Manages autonomous evolution cycles.
"""

import logging
from typing import Dict, List, Any, Optional
from .fitness import FitnessEvaluator
from .selection import SelectionStrategy

class EvolutionEngine:
    """Manages autonomous evolution cycles."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.fitness_evaluator = FitnessEvaluator(logger)
        self.selection_strategy = SelectionStrategy(logger)
        self.generation = 0
        self.population = []
        self.evolution_history = []
    
    def should_analyze_governor_data(self) -> bool:
        """Check if Governor data should be analyzed for evolution insights."""
        # Simple heuristic: analyze every 10 generations
        return self.generation % 10 == 0
    
    def analyze_governor_intelligence(self, governor_patterns: Dict[str, Any], 
                                    governor_clusters: Dict[str, Any], 
                                    memory_status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze Governor intelligence for architectural insights."""
        insights = []
        
        # Analyze patterns for architectural improvements
        if governor_patterns.get('high_energy_usage', False):
            insights.append({
                'type': 'energy_optimization',
                'description': 'High energy usage detected, suggesting energy system optimization',
                'priority': 'high',
                'suggested_changes': {
                    'energy_decay_rate': 0.015,  # Reduce decay rate
                    'sleep_trigger_energy': 40.0  # Increase sleep threshold
                }
            })
        
        if governor_patterns.get('memory_fragmentation', False):
            insights.append({
                'type': 'memory_optimization',
                'description': 'Memory fragmentation detected, suggesting memory consolidation improvements',
                'priority': 'medium',
                'suggested_changes': {
                    'memory_consolidation_strength': 0.9,
                    'enable_memory_regularization': True
                }
            })
        
        if governor_clusters.get('low_diversity', False):
            insights.append({
                'type': 'exploration_enhancement',
                'description': 'Low cluster diversity detected, suggesting exploration improvements',
                'priority': 'high',
                'suggested_changes': {
                    'enable_exploration_strategies': True,
                    'enable_action_experimentation': True,
                    'contrarian_threshold': 3
                }
            })
        
        return insights
    
    def execute_autonomous_evolution(self) -> Dict[str, Any]:
        """Execute autonomous evolution cycle."""
        try:
            self.logger.info(f"🧬 Executing autonomous evolution cycle {self.generation}")
            
            # Generate new population if empty
            if not self.population:
                self.population = self._generate_initial_population()
            
            # Evaluate fitness of current population
            fitness_scores = []
            for individual in self.population:
                score = self.fitness_evaluator.evaluate(individual)
                fitness_scores.append(score)
            
            # Select parents for next generation
            parents = self.selection_strategy.select_parents(self.population, fitness_scores)
            
            # Generate offspring
            offspring = self._generate_offspring(parents)
            
            # Evaluate offspring
            offspring_fitness = []
            for child in offspring:
                score = self.fitness_evaluator.evaluate(child)
                offspring_fitness.append(score)
            
            # Update population
            self.population = self._update_population(parents, offspring, fitness_scores, offspring_fitness)
            
            # Record evolution
            self.evolution_history.append({
                'generation': self.generation,
                'best_fitness': max(fitness_scores + offspring_fitness),
                'average_fitness': sum(fitness_scores + offspring_fitness) / len(fitness_scores + offspring_fitness),
                'population_size': len(self.population)
            })
            
            self.generation += 1
            
            return {
                'success': True,
                'generation': self.generation - 1,
                'best_fitness': max(fitness_scores + offspring_fitness),
                'population_size': len(self.population)
            }
            
        except Exception as e:
            self.logger.error(f"Evolution cycle failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'generation': self.generation
            }
    
    def _generate_initial_population(self) -> List[Dict[str, Any]]:
        """Generate initial population of system configurations."""
        population = []
        
        # Generate diverse initial configurations
        for i in range(10):  # Small initial population
            config = {
                'salience_threshold': 0.3 + (i * 0.1),
                'max_actions_per_game': 400 + (i * 50),
                'energy_decay_rate': 0.01 + (i * 0.005),
                'enable_contrarian_strategy': i % 2 == 0,
                'enable_exploration_strategies': i % 3 == 0
            }
            population.append(config)
        
        return population
    
    def _generate_offspring(self, parents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate offspring from selected parents."""
        offspring = []
        
        # Simple crossover: combine parent configurations
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1 = parents[i]
                parent2 = parents[i + 1]
                
                # Create child by combining parent attributes
                child = {}
                for key in parent1.keys():
                    if key in parent2:
                        # Randomly choose from parent1 or parent2
                        child[key] = parent1[key] if i % 2 == 0 else parent2[key]
                    else:
                        child[key] = parent1[key]
                
                offspring.append(child)
        
        return offspring
    
    def _update_population(self, parents: List[Dict[str, Any]], 
                          offspring: List[Dict[str, Any]], 
                          parent_fitness: List[float], 
                          offspring_fitness: List[float]) -> List[Dict[str, Any]]:
        """Update population with best individuals."""
        # Combine parents and offspring
        all_individuals = parents + offspring
        all_fitness = parent_fitness + offspring_fitness
        
        # Sort by fitness (descending)
        sorted_individuals = sorted(zip(all_individuals, all_fitness), 
                                  key=lambda x: x[1], reverse=True)
        
        # Keep top 10 individuals
        return [individual for individual, _ in sorted_individuals[:10]]
