#!/usr/bin/env python3
"""
Tree Evaluation Integration

Integrates the Tree Evaluation Simulation Engine with Tabula Rasa's existing
simulation systems, providing enhanced simulation capabilities with space efficiency.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from .tree_evaluation_simulation import TreeEvaluationSimulationEngine, TreeEvaluationConfig
from .enhanced_simulation_agent import EnhancedSimulationAgent
from .enhanced_simulation_config import EnhancedSimulationConfig

logger = logging.getLogger(__name__)

class TreeEvaluationEnhancedSimulationAgent(EnhancedSimulationAgent):
    """
    Enhanced simulation agent that integrates Tree Evaluation for space-efficient simulation.
    
    This extends the existing EnhancedSimulationAgent with:
    - Tree-based path generation using Cook-Mertz algorithm
    - Space-efficient evaluation with O(√t log t) complexity
    - Deeper lookahead with same memory usage
    - Adaptive depth and branching based on available resources
    """
    
    def __init__(self, 
                 predictive_core,
                 config: Optional[EnhancedSimulationConfig] = None,
                 tree_config: Optional[TreeEvaluationConfig] = None,
                 persistence_dir: str = "data/enhanced_simulation_agent"):
        
        # Initialize parent class
        super().__init__(predictive_core, config, persistence_dir)
        
        # Initialize tree evaluation engine
        self.tree_engine = TreeEvaluationSimulationEngine(tree_config)
        
        # Enhanced statistics
        self.tree_evaluation_stats = {
            'tree_evaluations': 0,
            'memory_savings_total': 0,
            'deepest_simulations': 0,
            'cache_hit_improvements': 0
        }
        
        logger.info("Tree Evaluation Enhanced Simulation Agent initialized")
    
    def generate_action_plan(self, 
                           current_state: Dict[str, Any],
                           available_actions: List[int],
                           frame_analysis: Optional[Dict[str, Any]] = None,
                           memory_patterns: Optional[Dict[str, Any]] = None) -> Tuple[int, Optional[Tuple[int, int]], str]:
        """
        Generate action plan using tree evaluation for enhanced simulation.
        
        This method combines the existing simulation capabilities with tree evaluation
        for deeper, more space-efficient planning.
        """
        try:
            # Prepare context for tree evaluation
            evaluation_context = {
                'current_state': current_state,
                'frame_analysis': frame_analysis or {},
                'memory_patterns': memory_patterns or {},
                'timestamp': time.time()
            }
            
            # Use tree evaluation for primary simulation
            tree_result = self.tree_engine.evaluate_simulation_tree(
                current_state=current_state,
                available_actions=available_actions,
                context=evaluation_context
            )
            
            # Update statistics
            self.tree_evaluation_stats['tree_evaluations'] += 1
            self.tree_evaluation_stats['memory_savings_total'] += tree_result.get('memory_savings_bytes', 0)
            self.tree_evaluation_stats['deepest_simulations'] = max(
                self.tree_evaluation_stats['deepest_simulations'],
                tree_result.get('evaluation_depth', 0)
            )
            
            # Extract results
            recommended_action = tree_result.get('recommended_action')
            confidence = tree_result.get('confidence', 0.0)
            reasoning = tree_result.get('reasoning', 'Tree evaluation completed')
            
            # Fallback to original method if tree evaluation fails
            if recommended_action is None or confidence < 0.3:
                logger.warning("Tree evaluation failed, falling back to original simulation")
                try:
                    return super().generate_action_plan(
                        current_state, available_actions, frame_analysis, memory_patterns
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback simulation also failed: {fallback_error}")
                    # Return a safe default
                    return (available_actions[0] if available_actions else 1, None, "Emergency fallback")
            
            # Extract coordinates if available (for coordinate-based actions)
            coordinates = None
            if recommended_action in [1, 2, 3, 4]:  # Coordinate-based actions
                coordinates = self._extract_coordinates_from_state(current_state)
            
            # Enhanced reasoning with tree evaluation details
            enhanced_reasoning = f"{reasoning} | Tree depth: {tree_result.get('evaluation_depth', 0)} | Memory saved: {tree_result.get('memory_savings_bytes', 0)} bytes"
            
            logger.info(f"Tree evaluation action plan: action={recommended_action}, "
                       f"confidence={confidence:.3f}, depth={tree_result.get('evaluation_depth', 0)}")
            
            return recommended_action, coordinates, enhanced_reasoning
            
        except Exception as e:
            logger.error(f"Tree evaluation failed: {e}, falling back to original simulation")
            return super().generate_action_plan(
                current_state, available_actions, frame_analysis, memory_patterns
            )
    
    def _extract_coordinates_from_state(self, state: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """Extract coordinates from state for coordinate-based actions."""
        if 'coordinates' in state:
            coords = state['coordinates']
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                return (int(coords[0]), int(coords[1]))
        return None
    
    def get_enhanced_simulation_stats(self) -> Dict[str, Any]:
        """Get comprehensive simulation statistics including tree evaluation metrics."""
        # Get base stats from parent class
        try:
            base_stats = super().get_simulation_stats()
        except AttributeError:
            # Fallback if parent method doesn't exist
            base_stats = {
                'learning_stats': self.learning_stats,
                'simulation_count': self.simulation_count,
                'successful_simulations': self.successful_simulations
            }
        
        # Add tree evaluation statistics
        tree_stats = self.tree_engine.get_evaluation_stats()
        
        enhanced_stats = {
            **base_stats,
            'tree_evaluation': {
                'tree_evaluations': self.tree_evaluation_stats['tree_evaluations'],
                'memory_savings_total_mb': self.tree_evaluation_stats['memory_savings_total'] / (1024 * 1024),
                'deepest_simulation': self.tree_evaluation_stats['deepest_simulations'],
                'average_memory_savings_per_evaluation': (
                    self.tree_evaluation_stats['memory_savings_total'] / 
                    max(1, self.tree_evaluation_stats['tree_evaluations'])
                ),
                'tree_engine_stats': tree_stats
            }
        }
        
        return enhanced_stats
    
    def cleanup(self):
        """Clean up resources including tree evaluation engine."""
        super().cleanup()
        self.tree_engine.cleanup()
        logger.info("Tree evaluation enhanced simulation agent cleaned up")


class TreeEvaluationPathGenerator:
    """
    Tree-based path generator that replaces the original PathGenerator
    with space-efficient tree evaluation capabilities.
    """
    
    def __init__(self, tree_engine: TreeEvaluationSimulationEngine):
        self.tree_engine = tree_engine
        self.generation_stats = {
            'paths_generated': 0,
            'total_depth': 0,
            'memory_efficiency': 0.0
        }
    
    def generate_paths(self, 
                      initial_state: Dict[str, Any],
                      available_actions: List[int],
                      max_depth: int = 10,
                      max_paths: int = 100) -> List[Dict[str, Any]]:
        """
        Generate simulation paths using tree evaluation.
        
        Returns a list of path dictionaries with enhanced space efficiency.
        """
        try:
            # Use tree evaluation to generate paths
            tree_result = self.tree_engine.evaluate_simulation_tree(
                current_state=initial_state,
                available_actions=available_actions
            )
            
            # Convert tree result to path format
            paths = self._convert_tree_to_paths(tree_result)
            
            # Update statistics
            self.generation_stats['paths_generated'] += len(paths)
            self.generation_stats['total_depth'] += tree_result.get('evaluation_depth', 0)
            
            if tree_result.get('memory_savings_bytes', 0) > 0:
                theoretical_memory = max_depth * len(available_actions) * 64  # Rough estimate
                actual_memory = tree_result.get('memory_usage_mb', 0) * 1024 * 1024
                self.generation_stats['memory_efficiency'] = (
                    (theoretical_memory - actual_memory) / theoretical_memory
                )
            
            return paths
            
        except Exception as e:
            logger.error(f"Tree-based path generation failed: {e}")
            return []
    
    def _convert_tree_to_paths(self, tree_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert tree evaluation result to path format."""
        paths = []
        
        # Create a single best path from tree result
        if tree_result.get('recommended_action') is not None:
            path = {
                'actions': [tree_result['recommended_action']],
                'states': [tree_result.get('current_state', {})],
                'value': tree_result.get('value', 0.0),
                'confidence': tree_result.get('confidence', 0.0),
                'depth': tree_result.get('evaluation_depth', 0),
                'reasoning': tree_result.get('reasoning', ''),
                'memory_efficiency': tree_result.get('memory_savings_bytes', 0)
            }
            paths.append(path)
        
        return paths
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get path generation statistics."""
        return {
            'generation_stats': self.generation_stats,
            'tree_engine_stats': self.tree_engine.get_evaluation_stats()
        }


# Integration helper functions
def create_tree_evaluation_enhanced_agent(predictive_core,
                                        max_depth: int = 10,
                                        branching_factor: int = 5,
                                        memory_limit_mb: float = 100.0) -> TreeEvaluationEnhancedSimulationAgent:
    """Create a tree evaluation enhanced simulation agent with specified parameters."""
    
    # Create tree evaluation config
    tree_config = TreeEvaluationConfig(
        max_depth=max_depth,
        branching_factor=branching_factor,
        memory_limit_mb=memory_limit_mb
    )
    
    # Create enhanced simulation config
    sim_config = EnhancedSimulationConfig()
    
    # Create the enhanced agent
    agent = TreeEvaluationEnhancedSimulationAgent(
        predictive_core=predictive_core,
        config=sim_config,
        tree_config=tree_config
    )
    
    logger.info(f"Created tree evaluation enhanced agent: depth={max_depth}, "
               f"branching={branching_factor}, memory_limit={memory_limit_mb}MB")
    
    return agent


def integrate_with_continuous_learning_loop(learning_loop, 
                                          tree_config: Optional[TreeEvaluationConfig] = None):
    """
    Integrate tree evaluation with the continuous learning loop.
    
    This function modifies the learning loop to use tree evaluation
    for enhanced simulation capabilities.
    """
    if not hasattr(learning_loop, 'simulation_agent'):
        logger.warning("Learning loop does not have simulation_agent attribute")
        return False
    
    try:
        # Create tree evaluation enhanced agent
        tree_agent = create_tree_evaluation_enhanced_agent(
            predictive_core=learning_loop.predictive_core,
            tree_config=tree_config
        )
        
        # Replace the existing simulation agent
        learning_loop.simulation_agent = tree_agent
        
        logger.info("Successfully integrated tree evaluation with continuous learning loop")
        return True
        
    except Exception as e:
        logger.error(f"Failed to integrate tree evaluation: {e}")
        return False
