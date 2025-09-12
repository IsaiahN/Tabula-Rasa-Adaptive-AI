#!/usr/bin/env python3
"""
Dynamic Action Limits System

This module provides intelligent, adaptive action limit management that allows the
Governor to dynamically adjust action limits based on performance, learning progress,
and system state while respecting user-defined maximum boundaries.

The system has two levels of control:
1. User-defined maximum boundaries (cannot be exceeded)
2. Governor-controlled dynamic adjustments (within those boundaries)
"""

import json
import time
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

class ActionLimitType(Enum):
    """Types of action limits that can be dynamically adjusted."""
    PER_GAME = "per_game"
    PER_SESSION = "per_session" 
    PER_SCORECARD = "per_scorecard"
    PER_EPISODE = "per_episode"

@dataclass
class ActionLimitConfig:
    """Configuration for a specific action limit type."""
    current_value: int
    min_value: int
    max_value: int
    base_value: int
    scaling_factor: float = 1.0
    last_adjusted: float = 0.0
    adjustment_reason: str = ""
    performance_score: float = 0.5

@dataclass
class ActionLimitState:
    """Current state of all action limits."""
    limits: Dict[ActionLimitType, ActionLimitConfig]
    global_efficiency: float = 0.5
    learning_progress: float = 0.0
    system_stress: float = 0.0
    last_global_adjustment: float = 0.0

class DynamicActionLimits:
    """
    Intelligent action limit management system.
    
    This system allows the Governor to dynamically adjust action limits based on:
    - Performance metrics
    - Learning progress
    - System efficiency
    - Resource availability
    - Game complexity
    
    While respecting user-defined maximum boundaries.
    """
    
    def __init__(self, persistence_dir: Path, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(f"{__name__}.DynamicActionLimits")
        self.persistence_dir = Path(persistence_dir)
        self.persistence_dir.mkdir(exist_ok=True)
        
        # Load or initialize configuration
        self.config_file = self.persistence_dir / "dynamic_action_limits.json"
        self.state = self._load_state()
        
        # Performance tracking
        self.performance_history = []
        self.adjustment_history = []
        
        # Learning parameters - optimized for better adaptation
        self.learning_rate = 0.15          # Increased for faster adaptation
        self.adaptation_threshold = 0.03   # Lower threshold for more responsive changes
        self.min_adjustment_interval = 20.0  # Reduced for more frequent adjustments
        
        self.logger.info("Dynamic Action Limits system initialized")
    
    def _load_state(self) -> ActionLimitState:
        """Load the current state from persistence."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                
                # Convert string keys back to ActionLimitType enum
                limits = {}
                for key, config_data in data.get('limits', {}).items():
                    limit_type = ActionLimitType(key)
                    limits[limit_type] = ActionLimitConfig(**config_data)
                
                return ActionLimitState(
                    limits=limits,
                    global_efficiency=data.get('global_efficiency', 0.5),
                    learning_progress=data.get('learning_progress', 0.0),
                    system_stress=data.get('system_stress', 0.0),
                    last_global_adjustment=data.get('last_global_adjustment', 0.0)
                )
            except Exception as e:
                self.logger.warning(f"Failed to load action limits state: {e}")
        
        # Return default state
        return self._get_default_state()
    
    def _get_default_state(self) -> ActionLimitState:
        """Get the default action limits state - optimized for learning."""
        limits = {
            ActionLimitType.PER_GAME: ActionLimitConfig(
                current_value=2000,      # Increased for better exploration
                min_value=100,           # Higher minimum for meaningful learning
                max_value=4000,          # Higher maximum for complex games
                base_value=2000,         # Higher base value
                scaling_factor=1.0
            ),
            ActionLimitType.PER_SESSION: ActionLimitConfig(
                current_value=5000,      # Increased for diverse learning
                min_value=200,           # Higher minimum
                max_value=8000,          # Higher maximum for comprehensive sessions
                base_value=5000,         # Higher base value
                scaling_factor=1.0
            ),
            ActionLimitType.PER_SCORECARD: ActionLimitConfig(
                current_value=8000,      # Increased for comprehensive evaluation
                min_value=500,           # Higher minimum
                max_value=15000,         # Much higher maximum for thorough evaluation
                base_value=8000,         # Higher base value
                scaling_factor=1.0
            ),
            ActionLimitType.PER_EPISODE: ActionLimitConfig(
                current_value=1500,      # Balanced for focused learning
                min_value=100,           # Higher minimum
                max_value=3000,          # Higher maximum
                base_value=1500,         # Higher base value
                scaling_factor=1.0
            )
        }
        
        return ActionLimitState(limits=limits)
    
    def _save_state(self):
        """Save the current state to persistence."""
        try:
            # Convert enum keys to strings for JSON serialization
            limits_data = {}
            for limit_type, config in self.state.limits.items():
                limits_data[limit_type.value] = asdict(config)
            
            data = {
                'limits': limits_data,
                'global_efficiency': self.state.global_efficiency,
                'learning_progress': self.state.learning_progress,
                'system_stress': self.state.system_stress,
                'last_global_adjustment': self.state.last_global_adjustment
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save action limits state: {e}")
    
    def set_user_maximums(self, **kwargs):
        """Set user-defined maximum boundaries that cannot be exceeded.
        
        Args:
            **kwargs: Maximum values for each limit type
                per_game: Maximum actions per game
                per_session: Maximum actions per session
                per_scorecard: Maximum actions per scorecard
                per_episode: Maximum actions per episode
        """
        for limit_type_str, max_value in kwargs.items():
            try:
                limit_type = ActionLimitType(limit_type_str)
                if limit_type in self.state.limits:
                    self.state.limits[limit_type].max_value = max_value
                    self.logger.info(f"Set user maximum for {limit_type_str}: {max_value}")
            except ValueError:
                self.logger.warning(f"Invalid limit type: {limit_type_str}")
        
        self._save_state()
    
    def get_current_limit(self, limit_type: ActionLimitType) -> int:
        """Get the current limit for a specific type."""
        if limit_type in self.state.limits:
            return self.state.limits[limit_type].current_value
        return 1000  # fallback
    
    def update_performance_metrics(self, 
                                 efficiency: float,
                                 learning_progress: float,
                                 system_stress: float,
                                 game_complexity: float = 0.5):
        """Update performance metrics that influence action limit decisions.
        
        Args:
            efficiency: Overall system efficiency (0.0 to 1.0)
            learning_progress: How much the system is learning (0.0 to 1.0)
            system_stress: Current system stress level (0.0 to 1.0)
            game_complexity: Complexity of current game (0.0 to 1.0)
        """
        self.state.global_efficiency = efficiency
        self.state.learning_progress = learning_progress
        self.state.system_stress = system_stress
        
        # Store performance history
        self.performance_history.append({
            'timestamp': time.time(),
            'efficiency': efficiency,
            'learning_progress': learning_progress,
            'system_stress': system_stress,
            'game_complexity': game_complexity
        })
        
        # Keep only recent history (last 100 entries)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Trigger dynamic adjustment if conditions are met
        self._consider_dynamic_adjustment()
    
    def _consider_dynamic_adjustment(self):
        """Consider whether to make dynamic adjustments to action limits."""
        current_time = time.time()
        
        # Don't adjust too frequently
        if current_time - self.state.last_global_adjustment < self.min_adjustment_interval:
            return
        
        # Calculate adjustment factors
        efficiency_factor = self.state.global_efficiency
        learning_factor = self.state.learning_progress
        stress_factor = 1.0 - self.state.system_stress  # Lower stress = higher limits
        
        # Enhanced decision logic for better learning
        should_increase = (efficiency_factor > 0.75 and 
                          learning_factor > 0.2 and 
                          stress_factor > 0.5)
        
        should_decrease = (efficiency_factor < 0.35 or 
                          stress_factor < 0.3)
        
        # More aggressive learning acceleration
        rapid_learning = learning_factor > 0.6 and efficiency_factor > 0.6
        struggling = efficiency_factor < 0.25 or (learning_factor < 0.1 and efficiency_factor < 0.4)
        
        if should_increase or should_decrease or rapid_learning or struggling:
            adjustment_type = "increase" if (should_increase or rapid_learning) else "decrease"
            self._adjust_limits(adjustment_type == "increase", rapid_learning, struggling)
            self.state.last_global_adjustment = current_time
            self._save_state()
    
    def _adjust_limits(self, increase: bool, rapid_learning: bool = False, struggling: bool = False):
        """Adjust action limits based on performance with learning-optimized scaling."""
        if rapid_learning:
            adjustment_factor = 1.5  # Aggressive increase for rapid learning
            reason = "rapid_learning_acceleration"
        elif struggling:
            adjustment_factor = 0.7  # More aggressive decrease when struggling
            reason = "struggling_reduction"
        elif increase:
            adjustment_factor = 1.2  # Moderate increase for good performance
            reason = "performance_improvement"
        else:
            adjustment_factor = 0.8  # Moderate decrease for poor performance
            reason = "performance_degradation"
        
        for limit_type, config in self.state.limits.items():
            # Calculate new value
            new_value = int(config.current_value * adjustment_factor)
            
            # Ensure it stays within bounds
            new_value = max(config.min_value, min(config.max_value, new_value))
            
            # Only adjust if the change is significant
            if abs(new_value - config.current_value) >= 10:
                old_value = config.current_value
                config.current_value = new_value
                config.scaling_factor *= adjustment_factor
                config.last_adjusted = time.time()
                config.adjustment_reason = reason
                
                self.logger.info(f"Adjusted {limit_type.value}: {old_value} -> {new_value} ({reason})")
                
                # Record adjustment
                self.adjustment_history.append({
                    'timestamp': time.time(),
                    'limit_type': limit_type.value,
                    'old_value': old_value,
                    'new_value': new_value,
                    'reason': reason,
                    'efficiency': self.state.global_efficiency,
                    'learning_progress': self.state.learning_progress
                })
    
    def get_adaptive_limit(self, 
                          limit_type: ActionLimitType,
                          game_complexity: float = 0.5,
                          available_actions: int = 6) -> int:
        """Get an adaptive limit based on current context.
        
        Args:
            limit_type: Type of limit to get
            game_complexity: Complexity of current game (0.0 to 1.0)
            available_actions: Number of available actions in current game
            
        Returns:
            Adaptive action limit for the given context
        """
        if limit_type not in self.state.limits:
            return 1000
        
        config = self.state.limits[limit_type]
        base_limit = config.current_value
        
        # Apply complexity scaling
        complexity_factor = 0.5 + (game_complexity * 0.5)  # 0.5 to 1.0
        complexity_adjusted = int(base_limit * complexity_factor)
        
        # Apply available actions scaling
        actions_factor = min(1.0, available_actions / 6.0)  # Scale based on available actions
        actions_adjusted = int(complexity_adjusted * actions_factor)
        
        # Apply efficiency scaling
        efficiency_factor = 0.7 + (self.state.global_efficiency * 0.3)  # 0.7 to 1.0
        efficiency_adjusted = int(actions_adjusted * efficiency_factor)
        
        # Ensure within bounds
        final_limit = max(config.min_value, min(config.max_value, efficiency_adjusted))
        
        return final_limit
    
    def force_adjustment(self, 
                        limit_type: ActionLimitType,
                        new_value: int,
                        reason: str = "manual_override"):
        """Force an immediate adjustment to a specific limit.
        
        Args:
            limit_type: Type of limit to adjust
            new_value: New value to set
            reason: Reason for the adjustment
        """
        if limit_type not in self.state.limits:
            self.logger.warning(f"Unknown limit type: {limit_type}")
            return
        
        config = self.state.limits[limit_type]
        old_value = config.current_value
        
        # Ensure within bounds
        new_value = max(config.min_value, min(config.max_value, new_value))
        
        config.current_value = new_value
        config.last_adjusted = time.time()
        config.adjustment_reason = reason
        
        self.logger.info(f"Forced adjustment {limit_type.value}: {old_value} -> {new_value} ({reason})")
        self._save_state()
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get a comprehensive status report of the action limits system."""
        return {
            'current_limits': {
                limit_type.value: {
                    'current': config.current_value,
                    'min': config.min_value,
                    'max': config.max_value,
                    'base': config.base_value,
                    'scaling_factor': config.scaling_factor,
                    'last_adjusted': config.last_adjusted,
                    'adjustment_reason': config.adjustment_reason,
                    'performance_score': config.performance_score
                }
                for limit_type, config in self.state.limits.items()
            },
            'global_metrics': {
                'efficiency': self.state.global_efficiency,
                'learning_progress': self.state.learning_progress,
                'system_stress': self.state.system_stress,
                'last_global_adjustment': self.state.last_global_adjustment
            },
            'recent_adjustments': self.adjustment_history[-10:],  # Last 10 adjustments
            'performance_trend': self._calculate_performance_trend()
        }
    
    def _calculate_performance_trend(self) -> str:
        """Calculate the performance trend based on recent history."""
        if len(self.performance_history) < 5:
            return "insufficient_data"
        
        recent_efficiency = [p['efficiency'] for p in self.performance_history[-5:]]
        avg_recent = sum(recent_efficiency) / len(recent_efficiency)
        
        if len(self.performance_history) >= 10:
            older_efficiency = [p['efficiency'] for p in self.performance_history[-10:-5]]
            avg_older = sum(older_efficiency) / len(older_efficiency)
            
            if avg_recent > avg_older + 0.1:
                return "improving"
            elif avg_recent < avg_older - 0.1:
                return "declining"
        
        return "stable"
    
    def reset_to_defaults(self):
        """Reset all limits to their default values."""
        self.state = self._get_default_state()
        self.performance_history = []
        self.adjustment_history = []
        self._save_state()
        self.logger.info("Action limits reset to defaults")
