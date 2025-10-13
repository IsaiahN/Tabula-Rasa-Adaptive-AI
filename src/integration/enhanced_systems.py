"""
Enhanced Systems Integration Module

This module integrates the enhanced pattern detection, causal analysis,
knowledge transfer, and strategic learning systems into a cohesive whole
that works with the main game loop.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import sqlite3
import numpy as np
import asyncio
from dataclasses import dataclass

# Import our enhanced systems
from ..intelligence.enhanced_pattern_detection import (
    EnhancedPatternDetector,
    ColorSequence,
    ShapePattern,
    DirectionalPattern,
    GridPattern,
    SymmetryPattern
)
from ..intelligence.causal_analysis import (
    CausalAnalysisSystem,
    get_causal_analysis_system
)
from ..learning.knowledge_transfer import (
    KnowledgeTransferSystem,
    TransferablePattern,
    get_knowledge_transfer_system
)
from ..learning.strategic_learning import (
    StrategicLearningSystem,
    get_strategic_learning_system,
    Strategy,
    StrategyOutcome
)

# Disable pycache
import sys
sys.dont_write_bytecode = True

logger = logging.getLogger(__name__)

@dataclass
class GameContext:
    """Holds current game context and state."""
    game_id: str
    current_state: Dict[str, Any]
    detected_patterns: List[Any]
    active_strategy: Optional[Strategy]
    last_action: Optional[int]
    last_coordinates: Optional[Tuple[int, int]]
    score_history: List[float]
    action_history: List[int]

class EnhancedSystemsIntegrator:
    """Integrates and coordinates all enhanced systems."""

    def __init__(self, db_connection: Optional[sqlite3.Connection] = None):
        self.db_connection = db_connection
        
        # Initialize component systems
        self.pattern_detector = EnhancedPatternDetector()
        self.causal_system = get_causal_analysis_system(db_connection)
        self.transfer_system = get_knowledge_transfer_system(db_connection)
        self.learning_system = get_strategic_learning_system(db_connection)
        
        # Active game contexts
        self.game_contexts: Dict[str, GameContext] = {}
        
        logger.info("Enhanced systems integrator initialized")

    async def start_game(self, game_id: str, initial_state: Dict[str, Any]) -> None:
        """Initialize systems for a new game."""
        
        try:
            logger.info(f"Starting enhanced systems for game {game_id}")
            
            # Create new game context
            self.game_contexts[game_id] = GameContext(
                game_id=game_id,
                current_state=initial_state,
                detected_patterns=[],
                active_strategy=None,
                last_action=None,
                last_coordinates=None,
                score_history=[initial_state.get('score', 0.0)],
                action_history=[]
            )
            
            # Initial pattern detection
            await self._detect_patterns(game_id)
            
            # Load relevant knowledge
            await self._load_game_knowledge(game_id)
            
        except Exception as e:
            logger.error(f"Error starting enhanced systems: {e}")
            raise

    async def process_action(self,
                           game_id: str,
                           action_id: int,
                           coordinates: Optional[Tuple[int, int]],
                           new_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process an action and its results."""
        
        if game_id not in self.game_contexts:
            raise ValueError(f"No context found for game {game_id}")
            
        context = self.game_contexts[game_id]
        
        try:
            # Record action
            context.last_action = action_id
            context.last_coordinates = coordinates
            context.action_history.append(action_id)
            
            # Calculate score change
            old_score = context.score_history[-1]
            new_score = new_state.get('score', old_score)
            score_change = new_score - old_score
            context.score_history.append(new_score)
            
            # Update causal system
            self.causal_system.record_action_outcome(
                game_id,
                action_id,
                coordinates,
                context.current_state,
                new_state,
                score_change
            )
            
            # Update active strategy if one is being used
            if context.active_strategy:
                outcome = StrategyOutcome(
                    strategy_id=context.active_strategy.strategy_id,
                    success=score_change > 0,
                    score_change=score_change,
                    action_sequence=[action_id],
                    state_changes=self._compute_state_changes(
                        context.current_state,
                        new_state
                    ),
                    duration=0.0,  # TODO: Track duration
                    timestamp=datetime.now()
                )
                self.learning_system.update_strategy(
                    context.active_strategy,
                    outcome
                )
                
            # Update state
            context.current_state = new_state
            
            # Detect new patterns
            await self._detect_patterns(game_id)
            
            # Generate insights
            insights = await self._generate_insights(game_id)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error processing action: {e}")
            raise

    async def get_action_recommendation(self,
                                      game_id: str,
                                      available_actions: List[int]) -> Dict[str, Any]:
        """Get recommended next action."""
        
        if game_id not in self.game_contexts:
            raise ValueError(f"No context found for game {game_id}")
            
        context = self.game_contexts[game_id]
        
        try:
            recommendations = []
            
            # Get strategy recommendation
            strategy = self.learning_system.select_strategy(
                game_id,
                context.current_state,
                available_actions
            )
            
            if strategy:
                context.active_strategy = strategy
                recommendations.append({
                    'source': 'strategy',
                    'action_weights': strategy.action_weights,
                    'confidence': strategy.confidence
                })
                
            # Get causal system recommendation
            desired_effect = {'type': 'score_change', 'data': {'value': 1.0}}
            causal_recommendations = self.causal_system.get_action_recommendations(
                game_id,
                desired_effect,
                context.current_state
            )
            
            if causal_recommendations:
                recommendations.append({
                    'source': 'causal',
                    'actions': causal_recommendations,
                    'confidence': max(r['confidence'] for r in causal_recommendations)
                })
                
            # Combine recommendations
            final_recommendation = self._combine_recommendations(
                recommendations,
                available_actions
            )
            
            return final_recommendation
            
        except Exception as e:
            logger.error(f"Error getting action recommendation: {e}")
            raise

    async def end_game(self,
                      game_id: str,
                      final_state: Dict[str, Any],
                      success: bool) -> None:
        """Handle end of game and update learning."""
        
        if game_id not in self.game_contexts:
            raise ValueError(f"No context found for game {game_id}")
            
        context = self.game_contexts[game_id]
        
        try:
            # Final pattern detection
            await self._detect_patterns(game_id)
            
            # Store successful patterns if game was won
            if success:
                for pattern in context.detected_patterns:
                    self.transfer_system.store_pattern(
                        game_id,
                        pattern.__class__.__name__,
                        self._pattern_to_dict(pattern),
                        True
                    )
                    
            # Update strategic learning
            if context.active_strategy:
                outcome = StrategyOutcome(
                    strategy_id=context.active_strategy.strategy_id,
                    success=success,
                    score_change=final_state.get('score', 0) - context.score_history[0],
                    action_sequence=context.action_history,
                    state_changes=self._compute_state_changes(
                        context.current_state,
                        final_state
                    ),
                    duration=0.0,  # TODO: Track duration
                    timestamp=datetime.now()
                )
                self.learning_system.update_strategy(
                    context.active_strategy,
                    outcome
                )
                
            # Generate new strategies
            if success:
                self.learning_system.generate_new_strategy(
                    game_id,
                    [{
                        'actions': context.action_history,
                        'state_changes': self._compute_state_changes(
                            context.current_state,
                            final_state
                        ),
                        'initial_state': context.current_state
                    }]
                )
                
            # Evolve strategies
            self.learning_system.evolve_strategies(game_id)
            
            # Cleanup
            del self.game_contexts[game_id]
            
        except Exception as e:
            logger.error(f"Error ending game: {e}")
            raise

    async def _detect_patterns(self, game_id: str) -> None:
        """Run pattern detection on current game state."""
        
        context = self.game_contexts[game_id]
        
        # Get game grid from state
        grid = context.current_state.get('grid')
        if grid is None:
            return
            
        # Convert grid to numpy array if needed
        if not isinstance(grid, np.ndarray):
            grid = np.array(grid)
            
        # Detect patterns
        detected = []
        
        # Color sequences
        color_sequences = self.pattern_detector.detect_color_sequences(grid)
        detected.extend(color_sequences)
        
        # Shapes
        shapes = self.pattern_detector.detect_shapes(grid)
        detected.extend(shapes)
        
        # Directional patterns
        directional = self.pattern_detector.detect_directional_patterns(grid)
        detected.extend(directional)
        
        # Grid patterns
        grid_patterns = self.pattern_detector.analyze_grid_structure(grid)
        detected.extend(grid_patterns)
        
        # Symmetry
        symmetry = self.pattern_detector.detect_symmetry(grid)
        detected.extend(symmetry)
        
        # Update context
        context.detected_patterns = detected

    async def _load_game_knowledge(self, game_id: str) -> None:
        """Load relevant knowledge for the game."""
        
        context = self.game_contexts[game_id]
        
        # Find similar patterns
        for pattern in context.detected_patterns:
            pattern_dict = self._pattern_to_dict(pattern)
            similar = self.transfer_system.find_similar_patterns(
                pattern_dict,
                min_similarity=0.7
            )
            
            # Adapt and validate similar patterns
            for similar_pattern, similarity in similar:
                adapted = self.transfer_system.adapt_pattern(
                    similar_pattern,
                    context.current_state
                )
                
                if adapted and self.transfer_system.validate_transfer(
                    similar_pattern,
                    adapted,
                    context.current_state
                ):
                    # Store adapted pattern
                    self.transfer_system.store_pattern(
                        game_id,
                        similar_pattern.pattern_type,
                        adapted,
                        True  # Assume valid until proven otherwise
                    )

    async def _generate_insights(self, game_id: str) -> Dict[str, Any]:
        """Generate insights from current game state and history."""
        
        context = self.game_contexts[game_id]
        
        insights = {
            'patterns': [],
            'causal_relations': [],
            'strategy_progress': None
        }
        
        # Pattern insights
        for pattern in context.detected_patterns:
            insights['patterns'].append({
                'type': pattern.__class__.__name__,
                'data': self._pattern_to_dict(pattern)
            })
            
        # Causal insights
        if context.last_action is not None:
            effects = self.causal_system.get_likely_effects(
                game_id,
                context.last_action,
                context.last_coordinates,
                context.current_state
            )
            insights['causal_relations'].extend(effects)
            
        # Strategy insights
        if context.active_strategy:
            insights['strategy_progress'] = {
                'strategy_name': context.active_strategy.name,
                'success_rate': context.active_strategy.success_rate,
                'confidence': context.active_strategy.confidence
            }
            
        return insights

    def _pattern_to_dict(self, pattern: Any) -> Dict[str, Any]:
        """Convert pattern object to dictionary."""
        if isinstance(pattern, (ColorSequence, ShapePattern,
                              DirectionalPattern, GridPattern, SymmetryPattern)):
            return {
                k: v for k, v in pattern.__dict__.items()
                if not k.startswith('_')
            }
        return {}

    def _compute_state_changes(self,
                             before_state: Dict[str, Any],
                             after_state: Dict[str, Any]) -> Dict[str, Any]:
        """Compute changes between states."""
        
        changes = {}
        
        # Compare all keys
        all_keys = set(before_state.keys()) | set(after_state.keys())
        
        for key in all_keys:
            before = before_state.get(key)
            after = after_state.get(key)
            
            if before != after:
                changes[key] = {
                    'before': before,
                    'after': after
                }
                
        return changes

    def _combine_recommendations(self,
                               recommendations: List[Dict[str, Any]],
                               available_actions: List[int]) -> Dict[str, Any]:
        """Combine recommendations from different sources."""
        
        if not recommendations:
            return {
                'action': np.random.choice(available_actions),
                'confidence': 0.0,
                'source': 'random'
            }
            
        # Sort by confidence
        recommendations.sort(key=lambda r: r['confidence'], reverse=True)
        
        # Use highest confidence recommendation
        best = recommendations[0]
        
        if best['source'] == 'strategy':
            # Choose action based on weights
            weights = np.array([
                best['action_weights'].get(str(a), 0.0)
                for a in available_actions
            ])
            if weights.sum() > 0:
                weights /= weights.sum()
                action = np.random.choice(available_actions, p=weights)
            else:
                action = np.random.choice(available_actions)
                
        elif best['source'] == 'causal':
            # Use highest confidence action
            action = best['actions'][0]['action_id']
            
        else:
            action = np.random.choice(available_actions)
            
        return {
            'action': action,
            'confidence': best['confidence'],
            'source': best['source']
        }

# Create singleton instance
_enhanced_systems_integrator = None

def get_enhanced_systems_integrator(db_connection: Optional[sqlite3.Connection] = None) -> EnhancedSystemsIntegrator:
    """Get or create the enhanced systems integrator singleton."""
    global _enhanced_systems_integrator
    if _enhanced_systems_integrator is None:
        _enhanced_systems_integrator = EnhancedSystemsIntegrator(db_connection)
    return _enhanced_systems_integrator