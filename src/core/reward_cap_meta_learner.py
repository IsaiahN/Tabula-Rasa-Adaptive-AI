"""
Reward Cap Meta-Learner - Architect Integration

This module implements a meta-learning system that allows the Architect to dynamically
adjust reward and penalty caps based on system performance metrics.

The system monitors:
- Learning progress stability
- Exploration vs exploitation balance
- Score improvement rates
- System behavior patterns (lethargic vs manic)

And adjusts caps to find the "Goldilocks zone" for optimal learning.
"""

import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

from ..database.reward_cap_manager import get_reward_cap_manager
from collections import deque
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CapConfiguration:
    """Configuration for reward/penalty caps."""
    productivity_multiplier: float = 25.0
    productivity_max: float = 100.0
    recent_gains_multiplier: float = 15.0
    recent_gains_max: float = 75.0
    recent_losses_multiplier: float = 10.0
    recent_losses_max: float = 50.0
    exploration_bonus: float = 15.0
    movement_bonus: float = 20.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CapConfiguration':
        return cls(**data)

@dataclass
class PerformanceMetrics:
    """Performance metrics for cap adjustment decisions."""
    learning_progress_stability: float  # 0-1, higher = more stable
    exploration_ratio: float  # 0-1, ratio of exploration vs exploitation
    score_improvement_rate: float  # rate of score improvement over time
    behavior_volatility: float  # 0-1, higher = more volatile behavior
    productivity_efficiency: float  # 0-1, how efficiently bonuses are used
    stagnation_indicators: float  # 0-1, higher = more stagnation
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class RewardCapMetaLearner:
    """
    Meta-learning system for dynamic reward/penalty cap adjustment.
    
    Integrates with the Architect system to automatically tune reward parameters
    based on system performance and learning behavior patterns.
    """
    
    def __init__(
        self,
        base_path: str = ".",
        adjustment_interval: int = 1000,  # Adjust caps every 1000 actions
        stability_window: int = 500,  # Look at last 500 actions for stability
        min_adjustment: float = 0.05,  # Minimum 5% adjustment
        max_adjustment: float = 0.25,  # Maximum 25% adjustment
        logger: Optional[logging.Logger] = None
    ):
        self.base_path = Path(base_path)
        self.logger = logger or logging.getLogger(f"{__name__}.RewardCapMetaLearner")
        self.db_manager = get_reward_cap_manager()
        
        # Configuration
        self.adjustment_interval = adjustment_interval
        self.stability_window = stability_window
        self.min_adjustment = min_adjustment
        self.max_adjustment = max_adjustment
        
        # Current cap configuration
        self.current_caps = CapConfiguration()
        self.cap_history = deque(maxlen=100)  # Keep last 100 configurations
        
        # Performance tracking
        self.performance_history = deque(maxlen=stability_window)
        self.action_count = 0
        self.last_adjustment = 0
        
        # Learning state tracking
        self.score_history = deque(maxlen=stability_window)
        self.bonus_usage_history = deque(maxlen=stability_window)
        self.exploration_history = deque(maxlen=stability_window)
        
        # Meta-learning state
        self.adjustment_effectiveness = {}  # Track which adjustments helped
        self.learning_phase = "exploration"  # "exploration", "exploitation", "balanced"
        
        # Load existing configuration if available
        self._load_configuration()
        
        self.logger.info("🎯 Reward Cap Meta-Learner initialized")
        self.logger.info(f"   Current caps: {self.current_caps.to_dict()}")
    
    def update_performance(
        self,
        score_change: float,
        bonus_used: str,  # "productivity", "recent_gains", "recent_losses", "exploration", "movement"
        is_exploration: bool,
        learning_progress: Optional[float] = None
    ):
        """Update performance metrics with new action data."""
        self.action_count += 1
        
        # Update tracking histories
        self.score_history.append(score_change)
        self.bonus_usage_history.append(bonus_used)
        self.exploration_history.append(is_exploration)
        
        # Calculate current performance metrics
        metrics = self._calculate_performance_metrics(learning_progress)
        self.performance_history.append(metrics)
        
        # Check if it's time to adjust caps
        if self.action_count - self.last_adjustment >= self.adjustment_interval:
            self._consider_cap_adjustment()
    
    def _calculate_performance_metrics(self, learning_progress: Optional[float] = None) -> PerformanceMetrics:
        """Calculate current performance metrics."""
        if len(self.score_history) < 10:  # Need minimum data
            return PerformanceMetrics(
                learning_progress_stability=0.5,
                exploration_ratio=0.5,
                score_improvement_rate=0.0,
                behavior_volatility=0.5,
                productivity_efficiency=0.5,
                stagnation_indicators=0.5,
                timestamp=time.time()
            )
        
        # Learning progress stability (based on score variance)
        recent_scores = list(self.score_history)[-min(50, len(self.score_history)):]
        score_variance = np.var(recent_scores) if len(recent_scores) > 1 else 0.0
        learning_stability = max(0.0, 1.0 - min(1.0, score_variance / 10.0))  # Normalize variance
        
        # Exploration ratio
        recent_exploration = list(self.exploration_history)[-min(50, len(self.exploration_history)):]
        exploration_ratio = sum(recent_exploration) / len(recent_exploration) if recent_exploration else 0.5
        
        # Score improvement rate
        if len(self.score_history) >= 20:
            early_scores = list(self.score_history)[:len(self.score_history)//2]
            recent_scores = list(self.score_history)[len(self.score_history)//2:]
            early_avg = np.mean(early_scores) if early_scores else 0.0
            recent_avg = np.mean(recent_scores) if recent_scores else 0.0
            improvement_rate = (recent_avg - early_avg) / max(abs(early_avg), 1.0)
        else:
            improvement_rate = 0.0
        
        # Behavior volatility (based on action pattern consistency)
        recent_bonuses = list(self.bonus_usage_history)[-min(30, len(self.bonus_usage_history)):]
        if len(recent_bonuses) > 5:
            bonus_counts = {}
            for bonus in recent_bonuses:
                bonus_counts[bonus] = bonus_counts.get(bonus, 0) + 1
            # Higher diversity = higher volatility
            volatility = len(bonus_counts) / len(recent_bonuses)
        else:
            volatility = 0.5
        
        # Productivity efficiency (how often bonuses lead to positive outcomes)
        if len(self.bonus_usage_history) >= 10:
            bonus_outcomes = []
            for i, bonus in enumerate(self.bonus_usage_history):
                if i < len(self.score_history):
                    outcome = 1 if self.score_history[i] > 0 else 0
                    bonus_outcomes.append((bonus, outcome))
            
            # Calculate efficiency for each bonus type
            bonus_efficiency = {}
            for bonus_type in set(bonus for bonus, _ in bonus_outcomes):
                type_outcomes = [outcome for bonus, outcome in bonus_outcomes if bonus == bonus_type]
                if type_outcomes:
                    bonus_efficiency[bonus_type] = sum(type_outcomes) / len(type_outcomes)
            
            # Overall productivity efficiency
            productivity_efficiency = np.mean(list(bonus_efficiency.values())) if bonus_efficiency else 0.5
        else:
            productivity_efficiency = 0.5
        
        # Stagnation indicators (long periods without improvement)
        if len(self.score_history) >= 20:
            recent_scores = list(self.score_history)[-20:]
            stagnation_count = sum(1 for score in recent_scores if score <= 0)
            stagnation_indicators = stagnation_count / len(recent_scores)
        else:
            stagnation_indicators = 0.5
        
        return PerformanceMetrics(
            learning_progress_stability=learning_stability,
            exploration_ratio=exploration_ratio,
            score_improvement_rate=improvement_rate,
            behavior_volatility=volatility,
            productivity_efficiency=productivity_efficiency,
            stagnation_indicators=stagnation_indicators,
            timestamp=time.time()
        )
    
    def _consider_cap_adjustment(self):
        """Consider adjusting caps based on current performance."""
        if len(self.performance_history) < 10:
            return
        
        current_metrics = self.performance_history[-1]
        recent_metrics = list(self.performance_history)[-min(10, len(self.performance_history)):]
        
        # Calculate average recent performance
        avg_stability = np.mean([m.learning_progress_stability for m in recent_metrics])
        avg_exploration = np.mean([m.exploration_ratio for m in recent_metrics])
        avg_improvement = np.mean([m.score_improvement_rate for m in recent_metrics])
        avg_volatility = np.mean([m.behavior_volatility for m in recent_metrics])
        avg_efficiency = np.mean([m.productivity_efficiency for m in recent_metrics])
        avg_stagnation = np.mean([m.stagnation_indicators for m in recent_metrics])
        
        # Determine learning phase
        if avg_exploration > 0.7:
            self.learning_phase = "exploration"
        elif avg_exploration < 0.3:
            self.learning_phase = "exploitation"
        else:
            self.learning_phase = "balanced"
        
        # Make adjustment decisions
        adjustments = self._calculate_adjustments(
            avg_stability, avg_exploration, avg_improvement, 
            avg_volatility, avg_efficiency, avg_stagnation
        )
        
        if adjustments:
            self._apply_adjustments(adjustments)
            self.last_adjustment = self.action_count
    
    def _calculate_adjustments(
        self,
        stability: float,
        exploration: float,
        improvement: float,
        volatility: float,
        efficiency: float,
        stagnation: float
    ) -> Dict[str, float]:
        """Calculate cap adjustments based on performance metrics."""
        adjustments = {}
        
        # Rule 1: If system is too lethargic (low exploration, low improvement)
        if exploration < 0.3 and improvement < 0.1:
            # Increase exploration bonus and reduce productivity caps to encourage exploration
            adjustments['exploration_bonus'] = min(self.max_adjustment, 0.15)
            adjustments['productivity_max'] = -min(self.max_adjustment, 0.15)
            self.logger.info("🔧 LETHARGIC DETECTED: Increasing exploration incentives")
        
        # Rule 2: If system is too manic (high volatility, low stability)
        elif volatility > 0.8 and stability < 0.3:
            # Reduce all caps to stabilize behavior
            adjustments['productivity_multiplier'] = -min(self.max_adjustment, 0.20)
            adjustments['recent_gains_multiplier'] = -min(self.max_adjustment, 0.20)
            adjustments['recent_losses_multiplier'] = -min(self.max_adjustment, 0.20)
            self.logger.info("🔧 MANIC DETECTED: Reducing caps to stabilize behavior")
        
        # Rule 3: If productivity efficiency is low
        elif efficiency < 0.4:
            # Adjust multipliers to improve efficiency
            if self.learning_phase == "exploration":
                adjustments['exploration_bonus'] = min(self.max_adjustment, 0.10)
            else:
                adjustments['productivity_multiplier'] = min(self.max_adjustment, 0.10)
            self.logger.info("🔧 LOW EFFICIENCY: Adjusting multipliers for better productivity")
        
        # Rule 4: If stagnation is high
        elif stagnation > 0.7:
            # Increase recent gains bonus to encourage new attempts
            adjustments['recent_gains_multiplier'] = min(self.max_adjustment, 0.15)
            adjustments['recent_gains_max'] = min(self.max_adjustment, 0.15)
            self.logger.info("🔧 STAGNATION DETECTED: Increasing recent gains incentives")
        
        # Rule 5: If system is well-balanced, fine-tune
        elif stability > 0.6 and 0.3 < exploration < 0.7 and improvement > 0.05:
            # Small adjustments to optimize further
            if efficiency < 0.6:
                adjustments['productivity_multiplier'] = min(self.min_adjustment, 0.05)
            self.logger.info("🔧 BALANCED STATE: Fine-tuning for optimization")
        
        return adjustments
    
    def _apply_adjustments(self, adjustments: Dict[str, float]):
        """Apply calculated adjustments to current caps."""
        old_caps = self.current_caps.to_dict()
        
        for cap_name, adjustment in adjustments.items():
            if hasattr(self.current_caps, cap_name):
                current_value = getattr(self.current_caps, cap_name)
                new_value = current_value * (1 + adjustment)
                
                # Apply reasonable bounds
                if 'multiplier' in cap_name:
                    new_value = max(5.0, min(100.0, new_value))  # Multipliers: 5-100
                elif 'max' in cap_name:
                    new_value = max(10.0, min(500.0, new_value))  # Max values: 10-500
                elif 'bonus' in cap_name:
                    new_value = max(5.0, min(50.0, new_value))  # Bonuses: 5-50
                
                setattr(self.current_caps, cap_name, new_value)
                
                self.logger.info(f"   {cap_name}: {current_value:.2f} → {new_value:.2f} ({adjustment:+.1%})")
        
        # Save configuration
        self._save_configuration()
        
        # Track adjustment effectiveness
        self.cap_history.append(self.current_caps.to_dict())
        
        self.logger.info(f"🎯 Cap adjustment complete. New configuration: {self.current_caps.to_dict()}")
    
    def get_current_caps(self) -> CapConfiguration:
        """Get current cap configuration."""
        return self.current_caps
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        if not self.performance_history:
            return {"status": "insufficient_data"}
        
        recent_metrics = list(self.performance_history)[-min(10, len(self.performance_history)):]
        
        return {
            "learning_phase": self.learning_phase,
            "action_count": self.action_count,
            "last_adjustment": self.action_count - self.last_adjustment,
            "avg_stability": np.mean([m.learning_progress_stability for m in recent_metrics]),
            "avg_exploration": np.mean([m.exploration_ratio for m in recent_metrics]),
            "avg_improvement": np.mean([m.score_improvement_rate for m in recent_metrics]),
            "avg_volatility": np.mean([m.behavior_volatility for m in recent_metrics]),
            "avg_efficiency": np.mean([m.productivity_efficiency for m in recent_metrics]),
            "avg_stagnation": np.mean([m.stagnation_indicators for m in recent_metrics]),
            "current_caps": self.current_caps.to_dict()
        }
    
    def _save_configuration(self):
        """Save current configuration to database."""
        try:
            # Save current caps
            self.db_manager.update_current_caps(self.current_caps.to_dict())
            
            # Save cap history
            for history_entry in self.cap_history:
                self.db_manager.update_cap_history(history_entry)
            
            # Save other configuration
            self.db_manager.set_learning_phase(self.learning_phase)
            self.db_manager.set_last_adjustment(self.last_adjustment)
            self.db_manager.set_action_count(self.action_count)
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration to database: {e}")
    
    def _load_configuration(self):
        """Load configuration from database."""
        try:
            # Load current caps
            caps_data = self.db_manager.get_current_caps()
            if caps_data:
                self.current_caps = CapConfiguration.from_dict(caps_data)
            else:
                self.current_caps = CapConfiguration()
            
            # Load cap history
            history_data = self.db_manager.get_cap_history()
            self.cap_history = deque(history_data, maxlen=100)
            
            # Load other configuration
            self.learning_phase = self.db_manager.get_learning_phase()
            self.last_adjustment = self.db_manager.get_last_adjustment()
            self.action_count = self.db_manager.get_action_count()
            
            self.logger.info("📁 Loaded reward cap configuration from database")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load reward cap configuration from database: {e}")
            self.logger.info("🔄 Using default configuration")

