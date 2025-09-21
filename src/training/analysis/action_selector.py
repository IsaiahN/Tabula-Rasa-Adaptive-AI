#!/usr/bin/env python3
"""
Intelligent Action Selector for ARC-AGI-3
Uses frame analysis, available actions, and system intelligence to select optimal actions.
Integrates OpenCV analysis, pattern matching, and coordinate intelligence.
"""

import random
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from .frame_analyzer import FrameAnalyzer
from ..api.api_manager import APIManager

# Import advanced cognitive systems
try:
    from ...core.tree_evaluation_simulation import TreeEvaluationSimulationEngine
    from ...core.action_sequence_optimizer import ActionSequenceOptimizer
    from ...core.enhanced_exploration_strategies import EnhancedExplorationSystem, ExplorationState
    from ...core.predictive_core import PredictiveCore, SimulationHypothesis
    TREE_EVALUATION_AVAILABLE = True
except ImportError:
    TREE_EVALUATION_AVAILABLE = False
    TreeEvaluationSimulationEngine = None
    ActionSequenceOptimizer = None
    EnhancedExplorationSystem = None
    PredictiveCore = None

logger = logging.getLogger(__name__)

class ActionSelector:
    """Intelligent action selector that uses multiple sources of information and advanced systems."""
    
    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        self.frame_analyzer = FrameAnalyzer()
        self.action_history = []
        self.success_patterns = {}
        self.coordinate_intelligence = {}
        self.performance_history = []
        self.learning_cycles = 0
        
        # Advanced analysis state
        self.last_frame_analysis = None
        self.action_effectiveness = {}
        self.coordinate_success_rates = {}
        self.pattern_matches = []
        
        # Initialize advanced cognitive systems
        self.tree_evaluation_engine = None
        self.action_sequence_optimizer = None
        self.exploration_system = None
        self.predictive_core = None
        
        if TREE_EVALUATION_AVAILABLE:
            try:
                # Initialize Tree Evaluation Simulation Engine
                self.tree_evaluation_engine = TreeEvaluationSimulationEngine()
                logger.info("🌳 Tree Evaluation Simulation Engine initialized")
                
                # Initialize Action Sequence Optimizer
                self.action_sequence_optimizer = ActionSequenceOptimizer()
                logger.info("🎯 Action Sequence Optimizer initialized")
                
                # Initialize Enhanced Exploration System
                self.exploration_system = EnhancedExplorationSystem()
                logger.info("🔍 Enhanced Exploration System initialized")
                
                # Initialize Predictive Core
                self.predictive_core = PredictiveCore(
                    input_size=64*64*3,  # Frame size
                    hidden_size=256,
                    output_size=8,  # Number of actions
                    memory_size=1000
                )
                logger.info("🔮 Predictive Core initialized")
                
            except Exception as e:
                logger.warning(f"Failed to initialize advanced systems: {e}")
                self.tree_evaluation_engine = None
                self.action_sequence_optimizer = None
                self.exploration_system = None
                self.predictive_core = None
        
        logger.info("🧠 Advanced Action Selector initialized with OpenCV, pattern matching, and cognitive systems")
        
    def select_action(self, game_state: Dict[str, Any], available_actions: List[int]) -> Dict[str, Any]:
        """Select the best action using advanced OpenCV analysis and pattern matching."""
        
        # Extract frame data and current state
        frame_data = game_state.get('frame', [])
        current_score = game_state.get('score', 0)
        game_state_status = game_state.get('state', 'NOT_FINISHED')
        
        # 1. ADVANCED FRAME ANALYSIS - OpenCV-based visual analysis
        frame_analysis = self.frame_analyzer.analyze_frame(frame_data)
        self.last_frame_analysis = frame_analysis
        
        # 2. PATTERN MATCHING - Learn from successful patterns
        pattern_suggestions = self._analyze_patterns(frame_analysis, available_actions)
        
        # 3. COORDINATE INTELLIGENCE - Learn from successful coordinates
        coordinate_suggestions = self._get_intelligent_coordinates(frame_analysis, available_actions)
        
        # 4. OPENCV TARGET DETECTION - Find interactive elements
        opencv_suggestions = self._detect_opencv_targets(frame_analysis, available_actions)
        
        # 5. LEARNING-BASED SUGGESTIONS - Use historical success data
        learning_suggestions = self._generate_learning_suggestions(frame_analysis, available_actions)
        
        # 6. EXPLORATION STRATEGY - Balance exploration vs exploitation
        exploration_suggestions = self._generate_exploration_suggestions(available_actions)
        
        # 7. ADVANCED COGNITIVE SYSTEMS - Tree evaluation and sequence optimization
        advanced_suggestions = self._generate_advanced_suggestions(
            frame_analysis, available_actions, current_score, game_state
        )
        
        # 8. COMBINE ALL SUGGESTIONS - Multi-source decision making
        all_suggestions = self._combine_suggestions(
            pattern_suggestions,
            coordinate_suggestions, 
            opencv_suggestions,
            learning_suggestions,
            exploration_suggestions,
            advanced_suggestions
        )
        
        # 8. INTELLIGENT SELECTION - Multi-factor scoring
        best_action = self._select_best_action_intelligent(
            all_suggestions, available_actions, current_score, frame_analysis
        )
        
        # 9. LEARNING UPDATE - Update all learning systems
        self._update_learning_systems(best_action, game_state, frame_analysis)
        
        # 10. PERFORMANCE TRACKING - Track performance
        self._track_performance(best_action, current_score, game_state)
        
        return best_action
    
    def _analyze_patterns(self, frame_analysis: Dict[str, Any], available_actions: List[int]) -> List[Dict[str, Any]]:
        """Analyze patterns from frame data and historical success."""
        suggestions = []
        
        # Look for patterns in the frame
        objects = frame_analysis.get('objects', [])
        interactive_elements = frame_analysis.get('interactive_elements', [])
        
        # Prioritize interactive elements
        for element in interactive_elements:
            if element.get('confidence', 0) > 0.6:
                suggestions.append({
                    'action': 'ACTION6',
                    'x': element['centroid'][0],
                    'y': element['centroid'][1],
                    'confidence': element['confidence'],
                    'reason': f"Pattern-matched interactive element at ({element['centroid'][0]}, {element['centroid'][1]})",
                    'source': 'pattern_matching'
                })
        
        # Add large objects as secondary targets
        for obj in objects:
            if obj.get('area', 0) > 30:
                suggestions.append({
                    'action': 'ACTION6',
                    'x': obj['centroid'][0],
                    'y': obj['centroid'][1],
                    'confidence': min(obj['area'] / 100, 1.0),
                    'reason': f"Pattern-matched large object at ({obj['centroid'][0]}, {obj['centroid'][1]})",
                    'source': 'pattern_matching'
                })
        
        return suggestions
    
    def _get_intelligent_coordinates(self, frame_analysis: Dict[str, Any], available_actions: List[int]) -> List[Dict[str, Any]]:
        """Get intelligent coordinate suggestions based on learning."""
        suggestions = []
        
        if 6 not in available_actions:
            return suggestions
        
        # Use coordinate intelligence from past successes
        for coord, data in self.coordinate_success_rates.items():
            if data.get('success_rate', 0) > 0.3:  # 30% success rate threshold
                suggestions.append({
                    'action': 'ACTION6',
                    'x': coord[0],
                    'y': coord[1],
                    'confidence': data['success_rate'],
                    'reason': f"Intelligent coordinate from learning (success rate: {data['success_rate']:.2f})",
                    'source': 'coordinate_intelligence'
                })
        
        return suggestions
    
    def _detect_opencv_targets(self, frame_analysis: Dict[str, Any], available_actions: List[int]) -> List[Dict[str, Any]]:
        """Detect targets using OpenCV analysis."""
        suggestions = []
        
        if 6 not in available_actions:
            return suggestions
        
        # Use the frame analyzer's OpenCV detection
        interactive_elements = frame_analysis.get('interactive_elements', [])
        
        for element in interactive_elements:
            if element.get('confidence', 0) > 0.7:  # High confidence threshold
                suggestions.append({
                    'action': 'ACTION6',
                    'x': element['centroid'][0],
                    'y': element['centroid'][1],
                    'confidence': element['confidence'],
                    'reason': f"OpenCV detected target at ({element['centroid'][0]}, {element['centroid'][1]})",
                    'source': 'opencv_detection'
                })
        
        return suggestions
    
    def _generate_learning_suggestions(self, frame_analysis: Dict[str, Any], available_actions: List[int]) -> List[Dict[str, Any]]:
        """Generate suggestions based on learning from past actions."""
        suggestions = []
        
        # Learn from action effectiveness
        for action_id in available_actions:
            if action_id in self.action_effectiveness:
                effectiveness = self.action_effectiveness[action_id]
                # Calculate success rate from effectiveness data
                if isinstance(effectiveness, dict):
                    attempts = effectiveness.get('attempts', 0)
                    successes = effectiveness.get('successes', 0)
                    success_rate = successes / attempts if attempts > 0 else 0
                else:
                    success_rate = float(effectiveness) if isinstance(effectiveness, (int, float)) else 0
                
                if success_rate > 0.5:
                    if action_id == 6:
                        # For ACTION6, use learned coordinates
                        for coord, data in self.coordinate_success_rates.items():
                            if data.get('success_rate', 0) > 0.4:
                                suggestions.append({
                                    'action': 'ACTION6',
                                    'x': coord[0],
                                    'y': coord[1],
                                    'confidence': effectiveness * data['success_rate'],
                                    'reason': f"Learning-based ACTION6 at ({coord[0]}, {coord[1]})",
                                    'source': 'learning'
                                })
                    else:
                        suggestions.append({
                            'action': f'ACTION{action_id}',
                            'confidence': effectiveness,
                            'reason': f"Learning-based action {action_id}",
                            'source': 'learning'
                        })
        
        return suggestions
    
    def _generate_exploration_suggestions(self, available_actions: List[int]) -> List[Dict[str, Any]]:
        """Generate exploration suggestions for unknown areas."""
        suggestions = []
        
        # Exploration strategy: try different areas of the grid
        exploration_coords = [
            (16, 16), (32, 16), (48, 16),  # Top row
            (16, 32), (32, 32), (48, 32),  # Middle row
            (16, 48), (32, 48), (48, 48),  # Bottom row
        ]
        
        for coord in exploration_coords:
            if 6 in available_actions:
                suggestions.append({
                    'action': 'ACTION6',
                    'x': coord[0],
                    'y': coord[1],
                    'confidence': 0.3,
                    'reason': f"Exploration at ({coord[0]}, {coord[1]})",
                    'source': 'exploration'
                })
        
        # Add movement actions for exploration
        for action_id in [1, 2, 3, 4, 5, 7]:
            if action_id in available_actions:
                suggestions.append({
                    'action': f'ACTION{action_id}',
                    'confidence': 0.2,
                    'reason': f"Exploration movement {action_id}",
                    'source': 'exploration'
                })
        
        return suggestions
    
    def _generate_advanced_suggestions(self, frame_analysis: Dict[str, Any], 
                                     available_actions: List[int], 
                                     current_score: int, 
                                     game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggestions using advanced cognitive systems."""
        suggestions = []
        
        try:
            # Tree Evaluation Simulation
            if self.tree_evaluation_engine:
                tree_suggestions = self._get_tree_evaluation_suggestions(
                    frame_analysis, available_actions, current_score
                )
                suggestions.extend(tree_suggestions)
            
            # Action Sequence Optimization
            if self.action_sequence_optimizer:
                sequence_suggestions = self._get_sequence_optimization_suggestions(
                    frame_analysis, available_actions, game_state
                )
                suggestions.extend(sequence_suggestions)
            
            # Enhanced Exploration
            if self.exploration_system:
                exploration_suggestions = self._get_enhanced_exploration_suggestions(
                    frame_analysis, available_actions, current_score
                )
                suggestions.extend(exploration_suggestions)
            
            # Predictive Core Simulation
            if self.predictive_core:
                predictive_suggestions = self._get_predictive_suggestions(
                    frame_analysis, available_actions, game_state
                )
                suggestions.extend(predictive_suggestions)
                
        except Exception as e:
            logger.warning(f"Advanced suggestions generation failed: {e}")
        
        return suggestions
    
    def _get_tree_evaluation_suggestions(self, frame_analysis: Dict[str, Any], 
                                       available_actions: List[int], 
                                       current_score: int) -> List[Dict[str, Any]]:
        """Get suggestions from Tree Evaluation Simulation Engine."""
        suggestions = []
        
        try:
            # Create simulation hypothesis
            hypothesis = {
                'actions': available_actions,
                'frame_data': frame_analysis,
                'current_score': current_score,
                'max_depth': 5
            }
            
            # Run tree evaluation
            result = self.tree_evaluation_engine.evaluate_action_sequences(
                hypothesis, max_depth=5, timeout=0.1
            )
            
            if result and 'best_sequence' in result:
                best_action = result['best_sequence'][0] if result['best_sequence'] else available_actions[0]
                suggestions.append({
                    'action': f'ACTION{best_action}',
                    'id': best_action,
                    'confidence': result.get('confidence', 0.7),
                    'reason': f"Tree evaluation suggests ACTION{best_action}",
                    'source': 'tree_evaluation'
                })
                
        except Exception as e:
            logger.debug(f"Tree evaluation failed: {e}")
        
        return suggestions
    
    def _get_sequence_optimization_suggestions(self, frame_analysis: Dict[str, Any], 
                                             available_actions: List[int], 
                                             game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get suggestions from Action Sequence Optimizer."""
        suggestions = []
        
        try:
            # Convert frame to grid format
            frame_data = frame_analysis.get('frame', [])
            if frame_data:
                grid = frame_data[0] if isinstance(frame_data, list) else frame_data
                
                # Optimize action sequence
                result = self.action_sequence_optimizer.optimize_for_action6(
                    current_state=game_state,
                    available_actions=available_actions,
                    grid=grid,
                    game_id=game_state.get('game_id', 'unknown')
                )
                
                action_id, coordinates = result
                if action_id in available_actions:
                    suggestion = {
                        'action': f'ACTION{action_id}',
                        'id': action_id,
                        'confidence': 0.8,
                        'reason': f"Sequence optimizer suggests ACTION{action_id}",
                        'source': 'sequence_optimizer'
                    }
                    
                    if coordinates and action_id == 6:
                        suggestion['x'] = coordinates[0]
                        suggestion['y'] = coordinates[1]
                        suggestion['reason'] += f" at ({coordinates[0]}, {coordinates[1]})"
                    
                    suggestions.append(suggestion)
                    
        except Exception as e:
            logger.debug(f"Sequence optimization failed: {e}")
        
        return suggestions
    
    def _get_enhanced_exploration_suggestions(self, frame_analysis: Dict[str, Any], 
                                            available_actions: List[int], 
                                            current_score: int) -> List[Dict[str, Any]]:
        """Get suggestions from Enhanced Exploration System."""
        suggestions = []
        
        try:
            # Create exploration state
            state = ExplorationState(
                position=(0, 0),  # Default position
                score=current_score,
                frame_data=frame_analysis,
                available_actions=available_actions
            )
            
            # Get exploration result
            result = self.exploration_system.explore(state, available_actions)
            
            if result and result.action in available_actions:
                suggestions.append({
                    'action': f'ACTION{result.action}',
                    'id': result.action,
                    'confidence': result.confidence,
                    'reason': f"Enhanced exploration suggests ACTION{result.action}",
                    'source': 'enhanced_exploration'
                })
                
        except Exception as e:
            logger.debug(f"Enhanced exploration failed: {e}")
        
        return suggestions
    
    def _get_predictive_suggestions(self, frame_analysis: Dict[str, Any], 
                                  available_actions: List[int], 
                                  game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get suggestions from Predictive Core."""
        suggestions = []
        
        try:
            # Create simulation hypothesis
            hypothesis = SimulationHypothesis(
                action_sequence=[(action, None) for action in available_actions[:3]],  # Top 3 actions
                expected_outcome="score_improvement",
                confidence_threshold=0.6
            )
            
            # Run simulation
            result = self.predictive_core.simulate_rollout(
                initial_state=game_state,
                hypothesis=hypothesis,
                max_steps=3,
                timeout=0.1
            )
            
            if result and result.simulation_history:
                # Find best action from simulation
                best_step = max(result.simulation_history, 
                              key=lambda step: step.learning_progress)
                
                if best_step.action in available_actions:
                    suggestions.append({
                        'action': f'ACTION{best_step.action}',
                        'id': best_step.action,
                        'confidence': best_step.confidence,
                        'reason': f"Predictive core suggests ACTION{best_step.action}",
                        'source': 'predictive_core'
                    })
                    
        except Exception as e:
            logger.debug(f"Predictive core failed: {e}")
        
        return suggestions
    
    def _combine_suggestions(self, *suggestion_lists) -> List[Dict[str, Any]]:
        """Combine all suggestion sources into a unified list."""
        all_suggestions = []
        
        for suggestion_list in suggestion_lists:
            if isinstance(suggestion_list, list):
                all_suggestions.extend(suggestion_list)
            elif isinstance(suggestion_list, dict):
                all_suggestions.append(suggestion_list)
        
        # Remove duplicates
        unique_suggestions = []
        seen_actions = set()
        
        for suggestion in all_suggestions:
            if isinstance(suggestion, dict) and 'action' in suggestion:
                action_key = (suggestion.get('action'), suggestion.get('x'), suggestion.get('y'))
                if action_key not in seen_actions:
                    seen_actions.add(action_key)
                    unique_suggestions.append(suggestion)
        
        return unique_suggestions
    
    def _select_best_action_intelligent(self, suggestions: List[Dict[str, Any]], 
                                       available_actions: List[int],
                                       current_score: int,
                                       frame_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best action using intelligent multi-factor analysis."""
        
        if not suggestions:
            return self._get_fallback_action(available_actions)
        
        # Filter suggestions based on available actions
        valid_suggestions = self._filter_valid_actions(suggestions, available_actions)
        
        if not valid_suggestions:
            return self._get_fallback_action(available_actions)
        
        # Multi-factor scoring system
        scored_suggestions = []
        
        for suggestion in valid_suggestions:
            score = self._calculate_intelligent_score(suggestion, current_score, frame_analysis)
            scored_suggestions.append((score, suggestion))
        
        # Sort by score (highest first)
        scored_suggestions.sort(key=lambda x: x[0], reverse=True)
        
        # Select the best suggestion
        best_score, best_suggestion = scored_suggestions[0]
        
        # Convert to action format
        action_id = self._get_action_id(best_suggestion.get('action', 'ACTION1'))
        
        action = {
            'id': action_id,
            'reason': best_suggestion.get('reason', f'Intelligent selection (score: {best_score:.2f})'),
            'confidence': min(best_score, 1.0),
            'source': best_suggestion.get('source', 'unknown')
        }
        
        # Add coordinates for ACTION6
        if action_id == 6:
            action['x'] = best_suggestion.get('x', 32)
            action['y'] = best_suggestion.get('y', 32)
        
        return action
    
    def _calculate_intelligent_score(self, suggestion: Dict[str, Any], 
                                   current_score: int,
                                   frame_analysis: Dict[str, Any]) -> float:
        """Calculate intelligent score for action suggestion."""
        
        base_score = suggestion.get('confidence', 0.5)
        source = suggestion.get('source', 'unknown')
        
        # Factor 1: Base confidence - ensure it's a float
        if isinstance(base_score, (int, float)):
            confidence_factor = float(base_score)
        else:
            confidence_factor = 0.5
        
        # Factor 2: Source reliability
        source_weights = {
            'opencv_detection': 1.0,
            'pattern_matching': 0.9,
            'coordinate_intelligence': 0.8,
            'learning': 0.7,
            'exploration': 0.5
        }
        source_factor = source_weights.get(source, 0.5)
        
        # Factor 3: Learning from past success
        learning_factor = 1.0
        if source == 'coordinate_intelligence' and 'x' in suggestion and 'y' in suggestion:
            coord = (suggestion['x'], suggestion['y'])
            if coord in self.coordinate_success_rates:
                success_data = self.coordinate_success_rates[coord]
                if isinstance(success_data, dict):
                    learning_factor = success_data.get('success_rate', 0.5)
                else:
                    learning_factor = float(success_data) if success_data is not None else 0.5
        
        # Factor 4: Frame analysis alignment
        frame_factor = 1.0
        if 'x' in suggestion and 'y' in suggestion:
            # Check if coordinates align with detected objects
            x, y = suggestion['x'], suggestion['y']
            for obj in frame_analysis.get('objects', []):
                obj_x, obj_y = obj.get('centroid', (0, 0))
                distance = np.sqrt((x - obj_x)**2 + (y - obj_y)**2)
                if distance < 10:  # Close to detected object
                    frame_factor = 1.2
                    break
        
        # Factor 5: Exploration vs exploitation balance
        exploration_factor = 0.8 if source == 'exploration' else 1.0
        
        # Weighted combination
        intelligent_score = (
            confidence_factor * 0.4 +
            source_factor * 0.3 +
            learning_factor * 0.2 +
            frame_factor * 0.1
        ) * exploration_factor
        
        return min(intelligent_score, 1.0)
    
    def _update_learning_systems(self, action: Dict[str, Any], game_state: Dict[str, Any], frame_analysis: Dict[str, Any]):
        """Update all learning systems with new action and results."""
        
        # Update action history
        self.action_history.append({
            'action': action,
            'game_state': game_state,
            'frame_analysis': frame_analysis,
            'timestamp': game_state.get('timestamp', 0)
        })
        
        # Keep only recent history
        if len(self.action_history) > 1000:
            self.action_history = self.action_history[-1000:]
        
        # Update learning cycles
        self.learning_cycles += 1
        
        # Update action effectiveness
        action_id = action.get('id')
        if action_id:
            if action_id not in self.action_effectiveness:
                self.action_effectiveness[action_id] = {'successes': 0, 'attempts': 0}
            
            self.action_effectiveness[action_id]['attempts'] += 1
            
            # Check if action was successful (score increased)
            current_score = game_state.get('score', 0)
            # Handle case where score might be a dictionary or other type
            if isinstance(current_score, dict):
                # If it's a dictionary, try to extract a numeric value
                current_score = current_score.get('value', current_score.get('score', 0))
            if isinstance(current_score, (int, float)):
                current_score = float(current_score)
            else:
                current_score = 0.0
            
            if len(self.performance_history) > 0:
                last_score = self.performance_history[-1].get('score', 0)
                if isinstance(last_score, (int, float)):
                    last_score = float(last_score)
                else:
                    last_score = 0.0
                    
                if current_score > last_score:
                    self.action_effectiveness[action_id]['successes'] += 1
        
        # Update coordinate intelligence for ACTION6
        if action.get('id') == 6:
            coordinate = (action.get('x', 0), action.get('y', 0))
            if coordinate not in self.coordinate_success_rates:
                self.coordinate_success_rates[coordinate] = {'successes': 0, 'attempts': 0}
            
            self.coordinate_success_rates[coordinate]['attempts'] += 1
            
            # Check if coordinate was successful
            current_score = game_state.get('score', 0)
            if isinstance(current_score, (int, float)):
                current_score = float(current_score)
            else:
                current_score = 0.0
                
            if len(self.performance_history) > 0:
                last_score = self.performance_history[-1].get('score', 0)
                if isinstance(last_score, (int, float)):
                    last_score = float(last_score)
                else:
                    last_score = 0.0
                    
                if current_score > last_score:
                    self.coordinate_success_rates[coordinate]['successes'] += 1
            
            # Calculate success rate
            attempts = self.coordinate_success_rates[coordinate]['attempts']
            successes = self.coordinate_success_rates[coordinate]['successes']
            self.coordinate_success_rates[coordinate]['success_rate'] = successes / attempts if attempts > 0 else 0
    
    def _track_performance(self, action: Dict[str, Any], current_score: int, game_state: Dict[str, Any]):
        """Track performance and update metrics."""
        
        # Ensure current_score is a number
        # Handle case where score might be a dictionary or other type
        if isinstance(current_score, dict):
            # If it's a dictionary, try to extract a numeric value
            current_score = current_score.get('value', current_score.get('score', 0))
        if isinstance(current_score, (int, float)):
            current_score = float(current_score)
        else:
            current_score = 0.0
        
        # Update performance history
        self.performance_history.append({
            'action': action,
            'score': current_score,
            'timestamp': game_state.get('timestamp', 0)
        })
        
        # Keep only recent performance data
        if len(self.performance_history) > 500:
            self.performance_history = self.performance_history[-500:]
    
    def _filter_valid_actions(self, suggestions: List[Dict[str, Any]], available_actions: List[int]) -> List[Dict[str, Any]]:
        """Filter suggestions to only include available actions."""
        valid_suggestions = []
        
        for suggestion in suggestions:
            action_id = self._get_action_id(suggestion.get('action', 'ACTION1'))
            if action_id in available_actions:
                valid_suggestions.append(suggestion)
        
        return valid_suggestions
    
    def _get_action_id(self, action_name: str) -> int:
        """Convert action name to action ID."""
        action_map = {
            'ACTION1': 1, 'ACTION2': 2, 'ACTION3': 3, 'ACTION4': 4,
            'ACTION5': 5, 'ACTION6': 6, 'ACTION7': 7
        }
        return action_map.get(action_name, 1)
    
    def _get_fallback_action(self, available_actions: List[int]) -> Dict[str, Any]:
        """Get a fallback action when no intelligent suggestions are available."""
        
        # Prefer simple actions over complex ones
        simple_actions = [1, 2, 3, 4, 5, 7]  # All except ACTION6
        complex_actions = [6]  # ACTION6
        
        # Try simple actions first
        for action_id in simple_actions:
            if action_id in available_actions:
                return {
                    'id': action_id,
                    'reason': f'Fallback to simple action {action_id}',
                    'confidence': 0.1,
                    'source': 'fallback'
                }
        
        # Try complex actions
        for action_id in complex_actions:
            if action_id in available_actions:
                if action_id == 6:
                    return {
                        'id': action_id,
                        'x': random.randint(0, 63),
                        'y': random.randint(0, 63),
                        'reason': f'Fallback to complex action {action_id} with random coordinates',
                        'confidence': 0.1,
                        'source': 'fallback'
                    }
                else:
                    return {
                        'id': action_id,
                        'reason': f'Fallback to complex action {action_id}',
                        'confidence': 0.1,
                        'source': 'fallback'
                    }
        
        # Last resort: random action
        action_id = random.choice(available_actions)
        action = {
            'id': action_id,
            'reason': f'Random fallback action {action_id}',
            'confidence': 0.05,
            'source': 'fallback'
        }
        
        if action_id == 6:
            action['x'] = random.randint(0, 63)
            action['y'] = random.randint(0, 63)
        
        return action
    
    def get_action_statistics(self) -> Dict[str, Any]:
        """Get statistics about action selection."""
        if not self.action_history:
            return {'total_actions': 0, 'action_distribution': {}}
        
        action_counts = {}
        for record in self.action_history:
            action_id = record['action']['id']
            action_counts[action_id] = action_counts.get(action_id, 0) + 1
        
        total_actions = len(self.action_history)
        action_distribution = {
            action_id: count / total_actions 
            for action_id, count in action_counts.items()
        }
        
        return {
            'total_actions': total_actions,
            'action_distribution': action_distribution,
            'recent_actions': self.action_history[-10:] if self.action_history else [],
            'coordinate_success_rates': self.coordinate_success_rates,
            'action_effectiveness': self.action_effectiveness
        }

def create_action_selector(api_manager: APIManager) -> ActionSelector:
    """Create an action selector instance."""
    return ActionSelector(api_manager)
