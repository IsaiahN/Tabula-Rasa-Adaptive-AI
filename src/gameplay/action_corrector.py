"""
Intelligent Action Correction System

Automatically corrects actions before execution to prevent errors.
"""

import logging
import random
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class CorrectionType(Enum):
    """Types of action corrections."""
    COORDINATE_CLAMP = "coordinate_clamp"
    CONFIDENCE_BOOST = "confidence_boost"
    ACTION_ROTATION = "action_rotation"
    PARAMETER_VALIDATION = "parameter_validation"
    RETRY_STRATEGY = "retry_strategy"

@dataclass
class ActionCorrection:
    """Represents an action correction."""
    correction_type: CorrectionType
    original_action: Dict[str, Any]
    corrected_action: Dict[str, Any]
    confidence: float
    reason: str

class ActionCorrector:
    """Intelligent action correction system."""
    
    def __init__(self):
        self.correction_strategies = {
            CorrectionType.COORDINATE_CLAMP: self._correct_coordinates,
            CorrectionType.CONFIDENCE_BOOST: self._boost_confidence,
            CorrectionType.ACTION_ROTATION: self._rotate_actions,
            CorrectionType.PARAMETER_VALIDATION: self._validate_parameters,
            CorrectionType.RETRY_STRATEGY: self._apply_retry_strategy
        }
        self.action_history = []
        self.correction_history = []
    
    def correct_action(self, action: Dict[str, Any], game_state: Dict[str, Any], 
                      frame_data: Any) -> ActionCorrection:
        """Correct an action before execution."""
        
        # Store original action
        original_action = action.copy()
        
        # Apply corrections based on action type and context
        corrections_applied = []
        
        # 1. Coordinate validation and clamping
        if 'x' in action and 'y' in action:
            corrected_coords = self._correct_coordinates(action, game_state)
            if corrected_coords != (action['x'], action['y']):
                action['x'], action['y'] = corrected_coords
                corrections_applied.append("coordinates_clamped")
        
        # 2. Confidence boosting for low-confidence actions
        if action.get('confidence', 1.0) < 0.5:
            boosted_confidence = self._boost_confidence(action, game_state)
            if boosted_confidence > action.get('confidence', 1.0):
                action['confidence'] = boosted_confidence
                corrections_applied.append("confidence_boosted")
        
        # 3. Action rotation to prevent repetition
        if self._is_repetitive_action(action):
            rotated_action = self._rotate_actions(action, game_state)
            if rotated_action['id'] != action['id']:
                action['id'] = rotated_action['id']
                action['reason'] = f"Rotated from repetitive action: {rotated_action['reason']}"
                corrections_applied.append("action_rotated")
        
        # 4. Parameter validation
        validated_params = self._validate_parameters(action, game_state)
        if validated_params != action:
            action.update(validated_params)
            corrections_applied.append("parameters_validated")
        
        # 5. Add retry strategy if needed
        if self._needs_retry_strategy(action, game_state):
            retry_info = self._apply_retry_strategy(action, game_state)
            action['retry_strategy'] = retry_info
            corrections_applied.append("retry_strategy_added")
        
        # Create correction record
        correction = ActionCorrection(
            correction_type=CorrectionType.PARAMETER_VALIDATION,  # Default type
            original_action=original_action,
            corrected_action=action,
            confidence=self._calculate_correction_confidence(action, original_action),
            reason=f"Applied corrections: {', '.join(corrections_applied)}"
        )
        
        # Store correction
        self.correction_history.append(correction)
        self.action_history.append(action)
        
        logger.info(f"Action corrected: {corrections_applied}")
        
        return correction
    
    def _correct_coordinates(self, action: Dict[str, Any], game_state: Dict[str, Any]) -> Tuple[int, int]:
        """Correct coordinate values to be within valid bounds."""
        x = action.get('x', 0)
        y = action.get('y', 0)
        
        # Clamp to valid range (0-63 for 64x64 grid)
        corrected_x = max(0, min(63, x))
        corrected_y = max(0, min(63, y))
        
        # If coordinates were out of bounds, try to find a nearby valid position
        if corrected_x != x or corrected_y != y:
            # Try to find a better position near the original
            corrected_x, corrected_y = self._find_better_coordinates(
                x, y, game_state, frame_data=None
            )
        
        return corrected_x, corrected_y
    
    def _find_better_coordinates(self, x: int, y: int, game_state: Dict[str, Any], 
                                frame_data: Any) -> Tuple[int, int]:
        """Find better coordinates near the original position."""
        # Simple strategy: try positions in a small radius
        for radius in range(1, 6):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    new_x = x + dx
                    new_y = y + dy
                    
                    if (0 <= new_x <= 63 and 0 <= new_y <= 63 and 
                        (dx != 0 or dy != 0)):
                        # Check if this position looks promising
                        if self._is_promising_coordinate(new_x, new_y, game_state):
                            return new_x, new_y
        
        # Fallback: clamp to valid range
        return max(0, min(63, x)), max(0, min(63, y))
    
    def _is_promising_coordinate(self, x: int, y: int, game_state: Dict[str, Any]) -> bool:
        """Check if a coordinate looks promising for action."""
        # This would integrate with frame analysis to check for interesting features
        # For now, use a simple heuristic
        return True  # Placeholder
    
    def _boost_confidence(self, action: Dict[str, Any], game_state: Dict[str, Any]) -> float:
        """Boost confidence for low-confidence actions."""
        current_confidence = action.get('confidence', 1.0)
        
        # Boost based on action type and context
        boost_factor = 1.0
        
        # Boost for movement actions (usually more reliable)
        if action.get('id') in [1, 2, 3, 4]:
            boost_factor = 1.2
        
        # Boost if we have good frame data
        if game_state.get('frame_quality', 0) > 0.7:
            boost_factor *= 1.1
        
        # Boost if action is well-reasoned
        if len(action.get('reason', '')) > 20:
            boost_factor *= 1.05
        
        new_confidence = min(1.0, current_confidence * boost_factor)
        
        return new_confidence
    
    def _is_repetitive_action(self, action: Dict[str, Any]) -> bool:
        """Check if action is repetitive."""
        if len(self.action_history) < 3:
            return False
        
        recent_actions = self.action_history[-3:]
        action_id = action.get('id')
        
        # Check if same action type repeated
        recent_action_ids = [a.get('id') for a in recent_actions]
        if recent_action_ids.count(action_id) >= 2:
            return True
        
        # Check if same coordinates repeated
        if 'x' in action and 'y' in action:
            recent_coords = [(a.get('x'), a.get('y')) for a in recent_actions if 'x' in a and 'y' in a]
            if (action['x'], action['y']) in recent_coords:
                return True
        
        return False
    
    def _rotate_actions(self, action: Dict[str, Any], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Rotate to a different action type to prevent repetition."""
        current_id = action.get('id')
        available_actions = game_state.get('available_actions', [1, 2, 3, 4, 5, 6])
        
        # Get action types that haven't been used recently
        recent_action_ids = [a.get('id') for a in self.action_history[-5:]]
        unused_actions = [aid for aid in available_actions if aid not in recent_action_ids]
        
        if unused_actions:
            new_id = random.choice(unused_actions)
        else:
            # All actions used recently, pick a different one
            other_actions = [aid for aid in available_actions if aid != current_id]
            new_id = random.choice(other_actions) if other_actions else current_id
        
        # Create rotated action
        rotated_action = action.copy()
        rotated_action['id'] = new_id
        rotated_action['reason'] = f"Rotated from action {current_id} to prevent repetition"
        
        # Adjust coordinates if needed for different action type
        if new_id in [1, 2, 3, 4]:  # Movement actions
            # Keep coordinates for movement
            pass
        elif new_id in [5, 6]:  # Interaction actions
            # Try to find interesting coordinates
            x, y = action.get('x', 32), action.get('y', 32)
            new_x, new_y = self._find_interaction_coordinates(x, y, game_state)
            rotated_action['x'] = new_x
            rotated_action['y'] = new_y
        
        return rotated_action
    
    def _find_interaction_coordinates(self, x: int, y: int, game_state: Dict[str, Any]) -> Tuple[int, int]:
        """Find good coordinates for interaction actions."""
        # Try positions around the center or near interesting areas
        center_x, center_y = 32, 32
        
        # Try positions in a pattern around center
        positions = [
            (center_x, center_y),
            (center_x + 10, center_y),
            (center_x - 10, center_y),
            (center_x, center_y + 10),
            (center_x, center_y - 10),
            (x, y)  # Original position as fallback
        ]
        
        for pos_x, pos_y in positions:
            if 0 <= pos_x <= 63 and 0 <= pos_y <= 63:
                return pos_x, pos_y
        
        # Fallback to original coordinates (clamped)
        return max(0, min(63, x)), max(0, min(63, y))
    
    def _validate_parameters(self, action: Dict[str, Any], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix action parameters."""
        validated_action = action.copy()
        
        # Validate action ID
        available_actions = game_state.get('available_actions', [1, 2, 3, 4, 5, 6])
        if validated_action.get('id') not in available_actions:
            validated_action['id'] = random.choice(available_actions)
            validated_action['reason'] = f"Corrected invalid action ID to {validated_action['id']}"
        
        # Validate confidence
        confidence = validated_action.get('confidence', 1.0)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            validated_action['confidence'] = 0.5  # Default confidence
        
        # Validate coordinates
        if 'x' in validated_action and 'y' in validated_action:
            x = validated_action['x']
            y = validated_action['y']
            
            if not isinstance(x, int) or not isinstance(y, int):
                validated_action['x'] = 32  # Default center
                validated_action['y'] = 32
            else:
                validated_action['x'] = max(0, min(63, x))
                validated_action['y'] = max(0, min(63, y))
        
        # Add missing required fields
        if 'reason' not in validated_action:
            validated_action['reason'] = f"Action {validated_action['id']} with confidence {validated_action.get('confidence', 1.0)}"
        
        return validated_action
    
    def _needs_retry_strategy(self, action: Dict[str, Any], game_state: Dict[str, Any]) -> bool:
        """Check if action needs a retry strategy."""
        # Add retry strategy for high-risk actions
        if action.get('confidence', 1.0) < 0.6:
            return True
        
        if action.get('id') in [5, 6]:  # Interaction actions are riskier
            return True
        
        return False
    
    def _apply_retry_strategy(self, action: Dict[str, Any], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply retry strategy to action."""
        return {
            "max_retries": 3,
            "backoff_factor": 1.5,
            "retry_conditions": ["api_error", "low_confidence", "no_progress"],
            "fallback_action": self._get_fallback_action(action, game_state)
        }
    
    def _get_fallback_action(self, action: Dict[str, Any], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get fallback action if retry fails."""
        fallback = action.copy()
        
        # Use a simpler, more reliable action
        if action.get('id') in [5, 6]:  # Interaction actions
            fallback['id'] = 1  # Simple movement
            fallback['reason'] = "Fallback to movement action"
        else:
            # Try a different action type
            available_actions = game_state.get('available_actions', [1, 2, 3, 4, 5, 6])
            other_actions = [aid for aid in available_actions if aid != action.get('id')]
            if other_actions:
                fallback['id'] = random.choice(other_actions)
                fallback['reason'] = f"Fallback to action {fallback['id']}"
        
        return fallback
    
    def _calculate_correction_confidence(self, corrected_action: Dict[str, Any], 
                                       original_action: Dict[str, Any]) -> float:
        """Calculate confidence in the correction."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence if coordinates were corrected
        if (corrected_action.get('x') != original_action.get('x') or 
            corrected_action.get('y') != original_action.get('y')):
            confidence += 0.2
        
        # Boost confidence if action type was changed
        if corrected_action.get('id') != original_action.get('id'):
            confidence += 0.1
        
        # Boost confidence if confidence was boosted
        if corrected_action.get('confidence', 0) > original_action.get('confidence', 0):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def get_correction_stats(self) -> Dict[str, Any]:
        """Get statistics about corrections made."""
        if not self.correction_history:
            return {"total_corrections": 0}
        
        correction_types = {}
        for correction in self.correction_history:
            correction_type = correction.correction_type.value
            correction_types[correction_type] = correction_types.get(correction_type, 0) + 1
        
        avg_confidence = sum(c.confidence for c in self.correction_history) / len(self.correction_history)
        
        return {
            "total_corrections": len(self.correction_history),
            "correction_types": correction_types,
            "average_confidence": avg_confidence,
            "recent_corrections": len([c for c in self.correction_history[-10:]])
        }

# Global action corrector instance
action_corrector = ActionCorrector()

def correct_action(action: Dict[str, Any], game_state: Dict[str, Any], 
                   frame_data: Any) -> ActionCorrection:
    """Correct an action before execution."""
    return action_corrector.correct_action(action, game_state, frame_data)

def get_correction_stats() -> Dict[str, Any]:
    """Get action correction statistics."""
    return action_corrector.get_correction_stats()
