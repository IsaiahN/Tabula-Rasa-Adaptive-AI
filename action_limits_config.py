#!/usr/bin/env python3
"""
Action Limits Configuration

This file contains the centralized configuration for action limits across the entire system.
The Governor can dynamically adjust these limits based on performance, but will never exceed
the maximum boundaries you set here.

To change action limits:
1. Modify the values in the ActionLimits class below
2. Restart the training system
3. The new limits will be applied automatically

The Governor will intelligently adjust limits within these boundaries based on:
- System performance and efficiency
- Learning progress
- Game complexity
- System stress levels

Example:
    To increase the maximum limit to 2000 actions per game:
    MAX_ACTIONS_PER_GAME = 2000
"""

class ActionLimits:
    """Centralized configuration for action limits across the system."""
    
    # =============================================================================
    # PRIMARY ACTION LIMITS - OPTIMIZED FOR LEARNING
    # =============================================================================
    # These are the main limits optimized for ARC learning based on research:
    # - Games need enough actions to explore and learn patterns
    # - Sessions need variety to prevent overfitting
    # - Scorecards need sufficient data for meaningful evaluation
    # - Episodes need balance between exploration and exploitation
    
    MAX_ACTIONS_PER_GAME = 2000          # Increased for better pattern exploration (was 1000)
    MAX_ACTIONS_PER_SESSION = 5000       # Increased for diverse learning (was 1000)
    MAX_ACTIONS_PER_SCORECARD = 8000     # Increased for comprehensive evaluation (was 1000)
    MAX_ACTIONS_PER_EPISODE = 1500       # Balanced for focused learning (was 1000)
    
    # =============================================================================
    # DYNAMIC SCALING LIMITS - OPTIMIZED FOR ADAPTIVE LEARNING
    # =============================================================================
    # These control how the system scales action limits based on performance
    # Optimized for better learning curves and faster adaptation
    
    MAX_ACTIONS_SCALING_BASE = 800       # Higher base for more exploration (was 500)
    MAX_ACTIONS_SCALING_MAX = 3000       # Higher max for complex games (was 1000)
    
    # =============================================================================
    # ADVANCED CONFIGURATION - OPTIMIZED FOR LEARNING
    # =============================================================================
    # These are more advanced settings optimized for better learning outcomes
    
    ENABLE_DYNAMIC_SCALING = True        # Enable intelligent scaling based on performance
    ENABLE_EARLY_TERMINATION = True      # Allow early termination for efficiency (was False)
    MIN_ACTIONS_PER_GAME = 100          # Higher minimum for meaningful exploration (was 50)
    
    # Learning-optimized parameters
    LEARNING_ACCELERATION_FACTOR = 1.5   # Boost limits when learning rapidly
    STRUGGLE_REDUCTION_FACTOR = 0.7      # Reduce limits when struggling
    COMPLEXITY_BONUS_MULTIPLIER = 1.3    # Extra actions for complex games
    EFFICIENCY_THRESHOLD_HIGH = 0.75     # Threshold for high performance
    EFFICIENCY_THRESHOLD_LOW = 0.35      # Threshold for low performance
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    @classmethod
    def get_max_actions_per_game(cls) -> int:
        """Get the maximum actions per game."""
        return cls.MAX_ACTIONS_PER_GAME
    
    @classmethod
    def get_max_actions_per_session(cls) -> int:
        """Get the maximum actions per session."""
        return cls.MAX_ACTIONS_PER_SESSION
    
    @classmethod
    def get_max_actions_per_scorecard(cls) -> int:
        """Get the maximum actions per scorecard."""
        return cls.MAX_ACTIONS_PER_SCORECARD
    
    @classmethod
    def get_max_actions_per_episode(cls) -> int:
        """Get the maximum actions per episode."""
        return cls.MAX_ACTIONS_PER_EPISODE
    
    @classmethod
    def get_scaled_max_actions(cls, efficiency: float) -> int:
        """Get scaled maximum actions based on efficiency (0.0 to 1.0)."""
        if not cls.ENABLE_DYNAMIC_SCALING:
            return cls.MAX_ACTIONS_PER_GAME
            
        scaled = int(cls.MAX_ACTIONS_SCALING_BASE * (1 + (1 - efficiency)))
        return min(cls.MAX_ACTIONS_SCALING_MAX, scaled)
    
    @classmethod
    def get_min_actions_per_game(cls) -> int:
        """Get the minimum actions per game."""
        return cls.MIN_ACTIONS_PER_GAME
    
    @classmethod
    def is_early_termination_enabled(cls) -> bool:
        """Check if early termination is enabled."""
        return cls.ENABLE_EARLY_TERMINATION
    
    @classmethod
    def update_limits(cls, **kwargs):
        """Update action limits dynamically.
        
        Args:
            **kwargs: Keyword arguments with new limit values
        """
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
                print(f"Updated {key} to {value}")
            else:
                print(f"Warning: {key} is not a valid ActionLimits attribute")
    
    @classmethod
    def print_current_limits(cls):
        """Print the current action limits configuration."""
        print("=" * 50)
        print("CURRENT ACTION LIMITS CONFIGURATION")
        print("=" * 50)
        print(f"Max Actions per Game:      {cls.MAX_ACTIONS_PER_GAME:,}")
        print(f"Max Actions per Session:   {cls.MAX_ACTIONS_PER_SESSION:,}")
        print(f"Max Actions per Scorecard: {cls.MAX_ACTIONS_PER_SCORECARD:,}")
        print(f"Max Actions per Episode:   {cls.MAX_ACTIONS_PER_EPISODE:,}")
        print(f"Scaling Base:              {cls.MAX_ACTIONS_SCALING_BASE:,}")
        print(f"Scaling Max:               {cls.MAX_ACTIONS_SCALING_MAX:,}")
        print(f"Dynamic Scaling:           {cls.ENABLE_DYNAMIC_SCALING}")
        print(f"Early Termination:         {cls.ENABLE_EARLY_TERMINATION}")
        print(f"Min Actions per Game:      {cls.MIN_ACTIONS_PER_GAME:,}")
        print("=" * 50)


if __name__ == "__main__":
    # Print current configuration when run directly
    ActionLimits.print_current_limits()
