"""
Gameplay Error Automation System

Automatically detects, analyzes, and fixes common gameplay errors in ARC-AGI-3.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class GameplayErrorType(Enum):
    """Types of gameplay errors that can be automatically fixed."""
    STAGNATION = "stagnation"
    INVALID_ACTION = "invalid_action"
    COORDINATE_OUT_OF_BOUNDS = "coordinate_out_of_bounds"
    REPETITIVE_ACTIONS = "repetitive_actions"
    LOW_CONFIDENCE_ACTIONS = "low_confidence_actions"
    API_VALIDATION_ERROR = "api_validation_error"
    FRAME_ANALYSIS_ERROR = "frame_analysis_error"
    PATTERN_MATCHING_FAILURE = "pattern_matching_failure"

@dataclass
class GameplayError:
    """Represents a gameplay error with context and fix suggestions."""
    error_type: GameplayErrorType
    description: str
    context: Dict[str, Any]
    severity: str  # 'critical', 'warning', 'info'
    auto_fixable: bool
    fix_suggestion: str
    confidence: float  # 0.0 to 1.0

class GameplayErrorDetector:
    """Detects gameplay errors from various sources."""
    
    def __init__(self):
        self.error_patterns = {
            GameplayErrorType.STAGNATION: self._detect_stagnation,
            GameplayErrorType.INVALID_ACTION: self._detect_invalid_actions,
            GameplayErrorType.COORDINATE_OUT_OF_BOUNDS: self._detect_coordinate_errors,
            GameplayErrorType.REPETITIVE_ACTIONS: self._detect_repetitive_actions,
            GameplayErrorType.LOW_CONFIDENCE_ACTIONS: self._detect_low_confidence,
            GameplayErrorType.API_VALIDATION_ERROR: self._detect_api_errors,
            GameplayErrorType.FRAME_ANALYSIS_ERROR: self._detect_frame_errors,
            GameplayErrorType.PATTERN_MATCHING_FAILURE: self._detect_pattern_failures
        }
    
    def detect_errors(self, game_state: Dict[str, Any], action_history: List[Dict], 
                     frame_data: Any, api_responses: List[Dict]) -> List[GameplayError]:
        """Detect all types of gameplay errors."""
        errors = []
        
        for error_type, detector in self.error_patterns.items():
            try:
                detected_errors = detector(game_state, action_history, frame_data, api_responses)
                errors.extend(detected_errors)
            except Exception as e:
                logger.error(f"Error in {error_type.value} detector: {e}")
        
        return errors
    
    def _detect_stagnation(self, game_state: Dict, action_history: List[Dict], 
                          frame_data: Any, api_responses: List[Dict]) -> List[GameplayError]:
        """Detect game stagnation (no progress for extended period)."""
        errors = []
        
        if len(action_history) < 10:
            return errors
        
        # Check for score stagnation
        recent_actions = action_history[-10:]
        scores = [action.get('score_after', 0) for action in recent_actions]
        
        if len(set(scores)) == 1 and scores[0] == 0:
            errors.append(GameplayError(
                error_type=GameplayErrorType.STAGNATION,
                description="No score progress in last 10 actions",
                context={"recent_scores": scores, "action_count": len(recent_actions)},
                severity="warning",
                auto_fixable=True,
                fix_suggestion="Try different action types or coordinates",
                confidence=0.8
            ))
        
        # Check for action repetition
        recent_action_types = [action.get('id') for action in recent_actions]
        if len(set(recent_action_types)) <= 2:
            errors.append(GameplayError(
                error_type=GameplayErrorType.REPETITIVE_ACTIONS,
                description="Limited action diversity in recent actions",
                context={"action_types": recent_action_types},
                severity="info",
                auto_fixable=True,
                fix_suggestion="Explore different action types",
                confidence=0.7
            ))
        
        return errors
    
    def _detect_invalid_actions(self, game_state: Dict, action_history: List[Dict], 
                               frame_data: Any, api_responses: List[Dict]) -> List[GameplayError]:
        """Detect invalid or problematic actions."""
        errors = []
        
        # Check for actions with very low confidence
        for action in action_history[-5:]:
            confidence = action.get('confidence', 1.0)
            if confidence < 0.3:
                errors.append(GameplayError(
                    error_type=GameplayErrorType.LOW_CONFIDENCE_ACTIONS,
                    description=f"Action {action.get('id')} has very low confidence: {confidence}",
                    context={"action": action, "confidence": confidence},
                    severity="warning",
                    auto_fixable=True,
                    fix_suggestion="Increase confidence threshold or improve action selection",
                    confidence=0.9
                ))
        
        return errors
    
    def _detect_coordinate_errors(self, game_state: Dict, action_history: List[Dict], 
                                 frame_data: Any, api_responses: List[Dict]) -> List[GameplayError]:
        """Detect coordinate-related errors."""
        errors = []
        
        for action in action_history[-5:]:
            if 'x' in action and 'y' in action:
                x, y = action['x'], action['y']
                if not (0 <= x <= 63 and 0 <= y <= 63):
                    errors.append(GameplayError(
                        error_type=GameplayErrorType.COORDINATE_OUT_OF_BOUNDS,
                        description=f"Coordinates ({x}, {y}) are out of bounds",
                        context={"coordinates": (x, y), "action": action},
                        severity="critical",
                        auto_fixable=True,
                        fix_suggestion="Clamp coordinates to valid range (0-63)",
                        confidence=1.0
                    ))
        
        return errors
    
    def _detect_repetitive_actions(self, game_state: Dict, action_history: List[Dict], 
                                  frame_data: Any, api_responses: List[Dict]) -> List[GameplayError]:
        """Detect repetitive action patterns."""
        errors = []
        
        if len(action_history) < 5:
            return errors
        
        recent_actions = action_history[-5:]
        action_types = [action.get('id') for action in recent_actions]
        
        # Check for same action repeated
        if len(set(action_types)) == 1:
            errors.append(GameplayError(
                error_type=GameplayErrorType.REPETITIVE_ACTIONS,
                description=f"Same action {action_types[0]} repeated 5 times",
                context={"action_type": action_types[0], "count": len(action_types)},
                severity="warning",
                auto_fixable=True,
                fix_suggestion="Introduce action diversity",
                confidence=0.8
            ))
        
        return errors
    
    def _detect_low_confidence(self, game_state: Dict, action_history: List[Dict], 
                              frame_data: Any, api_responses: List[Dict]) -> List[GameplayError]:
        """Detect low confidence actions."""
        errors = []
        
        for action in action_history[-3:]:
            confidence = action.get('confidence', 1.0)
            if confidence < 0.5:
                errors.append(GameplayError(
                    error_type=GameplayErrorType.LOW_CONFIDENCE_ACTIONS,
                    description=f"Low confidence action: {confidence}",
                    context={"action": action, "confidence": confidence},
                    severity="info",
                    auto_fixable=True,
                    fix_suggestion="Improve action selection algorithm",
                    confidence=0.6
                ))
        
        return errors
    
    def _detect_api_errors(self, game_state: Dict, action_history: List[Dict], 
                          frame_data: Any, api_responses: List[Dict]) -> List[GameplayError]:
        """Detect API-related errors."""
        errors = []
        
        for response in api_responses[-3:]:
            if 'error' in response or response.get('status_code', 200) != 200:
                errors.append(GameplayError(
                    error_type=GameplayErrorType.API_VALIDATION_ERROR,
                    description=f"API error: {response.get('error', 'Unknown error')}",
                    context={"response": response},
                    severity="critical",
                    auto_fixable=True,
                    fix_suggestion="Retry with corrected parameters",
                    confidence=0.9
                ))
        
        return errors
    
    def _detect_frame_errors(self, game_state: Dict, action_history: List[Dict], 
                            frame_data: Any, api_responses: List[Dict]) -> List[GameplayError]:
        """Detect frame analysis errors."""
        errors = []
        
        if frame_data is None:
            errors.append(GameplayError(
                error_type=GameplayErrorType.FRAME_ANALYSIS_ERROR,
                description="Frame data is None",
                context={"frame_data": frame_data},
                severity="critical",
                auto_fixable=True,
                fix_suggestion="Request new frame data",
                confidence=1.0
            ))
        
        return errors
    
    def _detect_pattern_failures(self, game_state: Dict, action_history: List[Dict], 
                                frame_data: Any, api_responses: List[Dict]) -> List[GameplayError]:
        """Detect pattern matching failures."""
        errors = []
        
        # This would integrate with the pattern matching system
        # For now, return empty list
        return errors

class GameplayErrorFixer:
    """Automatically fixes detected gameplay errors."""
    
    def __init__(self):
        self.fix_strategies = {
            GameplayErrorType.STAGNATION: self._fix_stagnation,
            GameplayErrorType.INVALID_ACTION: self._fix_invalid_actions,
            GameplayErrorType.COORDINATE_OUT_OF_BOUNDS: self._fix_coordinate_errors,
            GameplayErrorType.REPETITIVE_ACTIONS: self._fix_repetitive_actions,
            GameplayErrorType.LOW_CONFIDENCE_ACTIONS: self._fix_low_confidence,
            GameplayErrorType.API_VALIDATION_ERROR: self._fix_api_errors,
            GameplayErrorType.FRAME_ANALYSIS_ERROR: self._fix_frame_errors,
            GameplayErrorType.PATTERN_MATCHING_FAILURE: self._fix_pattern_failures
        }
    
    def fix_errors(self, errors: List[GameplayError], game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply fixes for detected errors."""
        fixes_applied = []
        
        for error in errors:
            if error.auto_fixable and error.error_type in self.fix_strategies:
                try:
                    fix_result = self.fix_strategies[error.error_type](error, game_state)
                    if fix_result:
                        fixes_applied.append({
                            "error_type": error.error_type.value,
                            "description": error.description,
                            "fix_applied": fix_result,
                            "confidence": error.confidence
                        })
                        logger.info(f"Applied fix for {error.error_type.value}: {fix_result}")
                except Exception as e:
                    logger.error(f"Failed to apply fix for {error.error_type.value}: {e}")
        
        return fixes_applied
    
    def _fix_stagnation(self, error: GameplayError, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fix stagnation by suggesting different actions."""
        return {
            "action": "explore_different_actions",
            "suggestion": "Try movement actions (1-4) instead of interaction actions (5-6)",
            "parameters": {
                "action_types": [1, 2, 3, 4],
                "exploration_mode": True
            }
        }
    
    def _fix_invalid_actions(self, error: GameplayError, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fix invalid actions by improving validation."""
        return {
            "action": "improve_action_validation",
            "suggestion": "Add pre-action validation checks",
            "parameters": {
                "min_confidence": 0.5,
                "validate_coordinates": True
            }
        }
    
    def _fix_coordinate_errors(self, error: GameplayError, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fix coordinate errors by clamping to valid range."""
        context = error.context
        if 'coordinates' in context:
            x, y = context['coordinates']
            clamped_x = max(0, min(63, x))
            clamped_y = max(0, min(63, y))
            
            return {
                "action": "clamp_coordinates",
                "suggestion": f"Clamp coordinates from ({x}, {y}) to ({clamped_x}, {clamped_y})",
                "parameters": {
                    "original": (x, y),
                    "clamped": (clamped_x, clamped_y)
                }
            }
        return None
    
    def _fix_repetitive_actions(self, error: GameplayError, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fix repetitive actions by introducing diversity."""
        return {
            "action": "introduce_action_diversity",
            "suggestion": "Use action rotation or random selection",
            "parameters": {
                "diversity_factor": 0.3,
                "rotation_enabled": True
            }
        }
    
    def _fix_low_confidence(self, error: GameplayError, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fix low confidence actions by improving selection."""
        return {
            "action": "improve_confidence",
            "suggestion": "Use ensemble methods or multiple sources",
            "parameters": {
                "min_confidence": 0.7,
                "ensemble_size": 3
            }
        }
    
    def _fix_api_errors(self, error: GameplayError, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fix API errors by retrying with corrected parameters."""
        return {
            "action": "retry_api_call",
            "suggestion": "Retry with validated parameters",
            "parameters": {
                "max_retries": 3,
                "backoff_factor": 2.0
            }
        }
    
    def _fix_frame_errors(self, error: GameplayError, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fix frame analysis errors by requesting new data."""
        return {
            "action": "request_new_frame",
            "suggestion": "Request fresh frame data from API",
            "parameters": {
                "retry_count": 1,
                "timeout": 5.0
            }
        }
    
    def _fix_pattern_failures(self, error: GameplayError, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fix pattern matching failures by improving algorithms."""
        return {
            "action": "improve_pattern_matching",
            "suggestion": "Use multiple pattern matching strategies",
            "parameters": {
                "fallback_strategies": ["template", "feature", "neural"],
                "confidence_threshold": 0.6
            }
        }

class GameplayErrorAutomation:
    """Main class for gameplay error automation."""
    
    def __init__(self):
        self.detector = GameplayErrorDetector()
        self.fixer = GameplayErrorFixer()
        self.error_history = []
        self.fix_history = []
    
    async def process_gameplay_cycle(self, game_state: Dict[str, Any], 
                                   action_history: List[Dict], frame_data: Any, 
                                   api_responses: List[Dict]) -> Dict[str, Any]:
        """Process a complete gameplay cycle with error detection and fixing."""
        
        # Detect errors
        errors = self.detector.detect_errors(game_state, action_history, frame_data, api_responses)
        
        # Store errors
        self.error_history.extend(errors)
        
        # Apply fixes
        fixes_applied = self.fixer.fix_errors(errors, game_state)
        
        # Store fixes
        self.fix_history.extend(fixes_applied)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(errors, fixes_applied)
        
        return {
            "errors_detected": len(errors),
            "fixes_applied": len(fixes_applied),
            "errors": [self._error_to_dict(e) for e in errors],
            "fixes": fixes_applied,
            "recommendations": recommendations,
            "system_health": self._calculate_system_health()
        }
    
    def _error_to_dict(self, error: GameplayError) -> Dict[str, Any]:
        """Convert GameplayError to dictionary."""
        return {
            "type": error.error_type.value,
            "description": error.description,
            "severity": error.severity,
            "auto_fixable": error.auto_fixable,
            "confidence": error.confidence,
            "context": error.context
        }
    
    def _generate_recommendations(self, errors: List[GameplayError], 
                                 fixes: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on errors and fixes."""
        recommendations = []
        
        # Count error types
        error_counts = {}
        for error in errors:
            error_type = error.error_type.value
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Generate recommendations based on error patterns
        if error_counts.get('stagnation', 0) > 2:
            recommendations.append("Consider implementing more diverse action selection strategies")
        
        if error_counts.get('repetitive_actions', 0) > 1:
            recommendations.append("Add action rotation or randomization to prevent repetition")
        
        if error_counts.get('low_confidence_actions', 0) > 3:
            recommendations.append("Improve action confidence calculation or increase thresholds")
        
        if error_counts.get('coordinate_out_of_bounds', 0) > 0:
            recommendations.append("Add coordinate validation before action execution")
        
        return recommendations
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health based on error patterns."""
        if not self.error_history:
            return {"status": "healthy", "score": 1.0}
        
        recent_errors = self.error_history[-20:]  # Last 20 errors
        
        critical_errors = len([e for e in recent_errors if e.severity == 'critical'])
        warning_errors = len([e for e in recent_errors if e.severity == 'warning'])
        info_errors = len([e for e in recent_errors if e.severity == 'info'])
        
        # Calculate health score (0.0 to 1.0)
        total_errors = len(recent_errors)
        if total_errors == 0:
            health_score = 1.0
        else:
            health_score = max(0.0, 1.0 - (critical_errors * 0.3 + warning_errors * 0.1 + info_errors * 0.05))
        
        if health_score >= 0.8:
            status = "healthy"
        elif health_score >= 0.6:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "status": status,
            "score": health_score,
            "critical_errors": critical_errors,
            "warning_errors": warning_errors,
            "info_errors": info_errors,
            "total_errors": total_errors
        }
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors and fixes."""
        return {
            "total_errors": len(self.error_history),
            "total_fixes": len(self.fix_history),
            "recent_errors": [self._error_to_dict(e) for e in self.error_history[-10:]],
            "recent_fixes": self.fix_history[-10:],
            "system_health": self._calculate_system_health()
        }

# Global gameplay error automation instance
gameplay_automation = GameplayErrorAutomation()

async def process_gameplay_errors(game_state: Dict[str, Any], action_history: List[Dict], 
                                 frame_data: Any, api_responses: List[Dict]) -> Dict[str, Any]:
    """Process gameplay errors and apply fixes."""
    return await gameplay_automation.process_gameplay_cycle(
        game_state, action_history, frame_data, api_responses
    )

def get_gameplay_health() -> Dict[str, Any]:
    """Get current gameplay system health."""
    return gameplay_automation.get_error_summary()
