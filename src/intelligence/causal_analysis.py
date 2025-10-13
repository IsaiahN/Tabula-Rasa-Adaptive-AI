"""
Causal Analysis System for ARC Games

This module implements a sophisticated causal analysis system that tracks and analyzes
cause-effect relationships between actions and outcomes in ARC games. It builds
predictive models of game mechanics and helps inform strategy generation.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import sqlite3
import logging
from datetime import datetime
import json

# Disable pycache
import sys
sys.dont_write_bytecode = True

logger = logging.getLogger(__name__)

@dataclass
class CausalRelation:
    """Represents a cause-effect relationship."""
    cause_type: str
    cause_data: Dict[str, Any]
    effect_type: str
    effect_data: Dict[str, Any]
    confidence: float
    support_count: int
    last_observed: datetime
    context_conditions: Dict[str, Any]

@dataclass
class ActionOutcome:
    """Represents the outcome of an action."""
    action_id: int
    coordinates: Optional[Tuple[int, int]]
    score_change: float
    state_changes: Dict[str, Any]
    timestamp: datetime

class CausalAnalysisSystem:
    """Core system for analyzing cause-effect relationships in games."""

    def __init__(self, db_connection: Optional[sqlite3.Connection] = None):
        self.db_connection = db_connection
        self.causal_relations: Dict[str, CausalRelation] = {}
        self.recent_actions: List[ActionOutcome] = []
        self.min_support_threshold = 3
        self.confidence_threshold = 0.7

    def record_action_outcome(self, 
                            game_id: str,
                            action_id: int,
                            coordinates: Optional[Tuple[int, int]],
                            before_state: Dict[str, Any],
                            after_state: Dict[str, Any],
                            score_change: float) -> None:
        """Record the outcome of an action for causal analysis."""
        
        # Create action outcome record
        outcome = ActionOutcome(
            action_id=action_id,
            coordinates=coordinates,
            score_change=score_change,
            state_changes=self._compute_state_changes(before_state, after_state),
            timestamp=datetime.now()
        )
        
        # Add to recent actions
        self.recent_actions.append(outcome)
        if len(self.recent_actions) > 100:  # Keep last 100 actions
            self.recent_actions.pop(0)
            
        # Analyze for causal relationships
        self._analyze_causal_relations(game_id, outcome, before_state)
        
        # Store in database
        self._store_action_outcome(game_id, outcome, before_state, after_state)

    def get_likely_effects(self, 
                          game_id: str,
                          action_id: int,
                          coordinates: Optional[Tuple[int, int]],
                          current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict likely effects of an action in current game state."""
        
        likely_effects = []
        
        # Get relevant causal relations
        relations = self._get_matching_relations(action_id, coordinates, current_state)
        
        for relation in relations:
            if relation.confidence >= self.confidence_threshold:
                effect = {
                    'effect_type': relation.effect_type,
                    'effect_data': relation.effect_data.copy(),
                    'confidence': relation.confidence,
                    'support': relation.support_count
                }
                likely_effects.append(effect)
                
        return likely_effects

    def get_action_recommendations(self,
                                 game_id: str,
                                 desired_effect: Dict[str, Any],
                                 current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommended actions to achieve desired effect."""
        
        recommendations = []
        
        # Find causal relations that lead to desired effect
        matching_relations = self._find_relations_for_effect(desired_effect)
        
        for relation in matching_relations:
            if (relation.confidence >= self.confidence_threshold and
                self._check_context_conditions(relation, current_state)):
                    
                recommendation = {
                    'action_id': relation.cause_data.get('action_id'),
                    'coordinates': relation.cause_data.get('coordinates'),
                    'confidence': relation.confidence,
                    'support': relation.support_count
                }
                recommendations.append(recommendation)
                
        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return recommendations

    def _compute_state_changes(self,
                             before_state: Dict[str, Any],
                             after_state: Dict[str, Any]) -> Dict[str, Any]:
        """Compute changes between before and after states."""
        
        changes = {}
        
        # Compare grid states
        if 'grid' in before_state and 'grid' in after_state:
            grid_changes = self._analyze_grid_changes(
                before_state['grid'],
                after_state['grid']
            )
            if grid_changes:
                changes['grid_changes'] = grid_changes
                
        # Compare object states
        if 'objects' in before_state and 'objects' in after_state:
            object_changes = self._analyze_object_changes(
                before_state['objects'],
                after_state['objects']
            )
            if object_changes:
                changes['object_changes'] = object_changes
                
        # Compare other state properties
        for key in before_state:
            if key not in ['grid', 'objects']:
                if key in after_state and before_state[key] != after_state[key]:
                    changes[f'{key}_change'] = {
                        'before': before_state[key],
                        'after': after_state[key]
                    }
                    
        return changes

    def _analyze_grid_changes(self,
                            before_grid: np.ndarray,
                            after_grid: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze changes in the game grid."""
        
        changes = []
        
        # Find changed cells
        diff = after_grid != before_grid
        changed_coords = np.where(diff)
        
        for y, x in zip(*changed_coords):
            change = {
                'position': (int(x), int(y)),
                'before_value': int(before_grid[y, x]),
                'after_value': int(after_grid[y, x])
            }
            changes.append(change)
            
        return changes

    def _analyze_object_changes(self,
                              before_objects: List[Dict[str, Any]],
                              after_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze changes in game objects."""
        
        changes = []
        
        # Track objects by ID if available
        before_dict = {obj.get('id', i): obj for i, obj in enumerate(before_objects)}
        after_dict = {obj.get('id', i): obj for i, obj in enumerate(after_objects)}
        
        # Find changed objects
        for obj_id in set(before_dict.keys()) | set(after_dict.keys()):
            before_obj = before_dict.get(obj_id)
            after_obj = after_dict.get(obj_id)
            
            if before_obj and after_obj:
                # Object modified
                changes.append({
                    'type': 'modified',
                    'object_id': obj_id,
                    'before': before_obj,
                    'after': after_obj
                })
            elif before_obj:
                # Object removed
                changes.append({
                    'type': 'removed',
                    'object_id': obj_id,
                    'object': before_obj
                })
            else:
                # Object added
                changes.append({
                    'type': 'added',
                    'object_id': obj_id,
                    'object': after_obj
                })
                
        return changes

    def _analyze_causal_relations(self,
                                game_id: str,
                                outcome: ActionOutcome,
                                context_state: Dict[str, Any]) -> None:
        """Analyze action outcome for causal relations."""
        
        # Create cause identifier
        cause_id = self._create_cause_id(outcome.action_id, outcome.coordinates)
        
        # Extract effects
        effects = self._extract_effects(outcome)
        
        for effect in effects:
            relation_id = f"{cause_id}:{effect['type']}"
            
            if relation_id in self.causal_relations:
                # Update existing relation
                relation = self.causal_relations[relation_id]
                relation.support_count += 1
                relation.confidence = self._update_confidence(relation, True)
                relation.last_observed = outcome.timestamp
                
            else:
                # Create new relation
                relation = CausalRelation(
                    cause_type='action',
                    cause_data={
                        'action_id': outcome.action_id,
                        'coordinates': outcome.coordinates
                    },
                    effect_type=effect['type'],
                    effect_data=effect['data'],
                    confidence=self.confidence_threshold,  # Initial confidence
                    support_count=1,
                    last_observed=outcome.timestamp,
                    context_conditions=self._extract_context_conditions(context_state)
                )
                self.causal_relations[relation_id] = relation
                
        # Store updated relations
        self._store_causal_relations(game_id)

    def _create_cause_id(self,
                        action_id: int,
                        coordinates: Optional[Tuple[int, int]]) -> str:
        """Create unique identifier for a cause."""
        if coordinates:
            return f"action_{action_id}_at_{coordinates[0]}_{coordinates[1]}"
        return f"action_{action_id}"

    def _extract_effects(self, outcome: ActionOutcome) -> List[Dict[str, Any]]:
        """Extract effect patterns from an action outcome."""
        
        effects = []
        
        # Score changes
        if outcome.score_change != 0:
            effects.append({
                'type': 'score_change',
                'data': {'value': outcome.score_change}
            })
            
        # State changes
        for change_type, change_data in outcome.state_changes.items():
            effects.append({
                'type': change_type,
                'data': change_data
            })
            
        return effects

    def _extract_context_conditions(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant context conditions from game state."""
        
        conditions = {}
        
        # Extract relevant state properties that might affect action outcomes
        # This is game-specific and should be customized
        
        return conditions

    def _update_confidence(self, relation: CausalRelation, success: bool) -> float:
        """Update confidence score for a causal relation."""
        
        # Simple exponential moving average
        alpha = 0.1  # Learning rate
        new_evidence = 1.0 if success else 0.0
        
        return relation.confidence * (1 - alpha) + new_evidence * alpha

    def _get_matching_relations(self,
                              action_id: int,
                              coordinates: Optional[Tuple[int, int]],
                              current_state: Dict[str, Any]) -> List[CausalRelation]:
        """Get causal relations matching the action and context."""
        
        matching = []
        cause_id = self._create_cause_id(action_id, coordinates)
        
        for relation in self.causal_relations.values():
            if (relation.cause_data.get('action_id') == action_id and
                self._check_context_conditions(relation, current_state)):
                matching.append(relation)
                
        return matching

    def _find_relations_for_effect(self,
                                 desired_effect: Dict[str, Any]) -> List[CausalRelation]:
        """Find causal relations that lead to desired effect."""
        
        matching = []
        
        for relation in self.causal_relations.values():
            if (relation.effect_type == desired_effect.get('type') and
                self._match_effect_data(relation.effect_data, desired_effect.get('data', {}))):
                matching.append(relation)
                
        return matching

    def _match_effect_data(self,
                          relation_data: Dict[str, Any],
                          desired_data: Dict[str, Any]) -> bool:
        """Check if effect data matches desired effect."""
        
        for key, value in desired_data.items():
            if key not in relation_data or relation_data[key] != value:
                return False
        return True

    def _check_context_conditions(self,
                                relation: CausalRelation,
                                current_state: Dict[str, Any]) -> bool:
        """Check if current state matches relation's context conditions."""
        
        for condition, value in relation.context_conditions.items():
            if condition not in current_state or current_state[condition] != value:
                return False
        return True

    def _store_action_outcome(self,
                            game_id: str,
                            outcome: ActionOutcome,
                            before_state: Dict[str, Any],
                            after_state: Dict[str, Any]) -> None:
        """Store action outcome in database."""
        
        if not self.db_connection:
            return
            
        try:
            cursor = self.db_connection.cursor()
            
            # Store in action_outcomes table
            cursor.execute("""
                INSERT INTO action_outcomes 
                (game_id, action_id, coordinates, score_change, 
                 before_state, after_state, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                game_id,
                outcome.action_id,
                json.dumps(outcome.coordinates) if outcome.coordinates else None,
                outcome.score_change,
                json.dumps(before_state),
                json.dumps(after_state),
                outcome.timestamp.isoformat()
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing action outcome: {e}")

    def _store_causal_relations(self, game_id: str) -> None:
        """Store causal relations in database."""
        
        if not self.db_connection:
            return
            
        try:
            cursor = self.db_connection.cursor()
            
            for relation_id, relation in self.causal_relations.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO causal_relations
                    (game_id, relation_id, cause_type, cause_data,
                     effect_type, effect_data, confidence, support_count,
                     last_observed, context_conditions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    game_id,
                    relation_id,
                    relation.cause_type,
                    json.dumps(relation.cause_data),
                    relation.effect_type,
                    json.dumps(relation.effect_data),
                    relation.confidence,
                    relation.support_count,
                    relation.last_observed.isoformat(),
                    json.dumps(relation.context_conditions)
                ))
                
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing causal relations: {e}")

# Create singleton instance
_causal_analysis_system = None

def get_causal_analysis_system(db_connection: Optional[sqlite3.Connection] = None) -> CausalAnalysisSystem:
    """Get or create the causal analysis system singleton."""
    global _causal_analysis_system
    if _causal_analysis_system is None:
        _causal_analysis_system = CausalAnalysisSystem(db_connection)
    return _causal_analysis_system