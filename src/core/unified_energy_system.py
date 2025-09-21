"""
Unified Energy Management System - 0-100 scale survival mechanics.

This system provides a unified, consistent energy management system that:
1. Uses a proper 0-100 energy scale throughout the system
2. Implements survival mechanics with death/respawn
3. Provides adaptive energy consumption based on performance
4. Integrates with sleep cycles and memory consolidation
5. Tracks energy metrics and provides detailed reporting
"""

import time
import math
import logging
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class EnergyState(Enum):
    """Energy state enumeration."""
    HEALTHY = "healthy"          # 70-100 energy
    MODERATE = "moderate"        # 40-69 energy
    LOW = "low"                  # 20-39 energy
    CRITICAL = "critical"        # 5-19 energy
    DEAD = "dead"               # 0-4 energy


@dataclass
class EnergyConfig:
    """Configuration for unified energy management."""
    # Energy scale (0-100)
    max_energy: float = 100.0
    min_energy: float = 0.0
    
    # Base consumption rates
    base_consumption_per_second: float = 0.1      # Energy per second when idle
    base_consumption_per_action: float = 1.0      # Base energy per action
    
    # Action-specific costs (0-100 scale)
    action_costs: Dict[int, float] = None
    
    # Performance-based adjustments
    success_energy_bonus: float = 2.0             # Energy gained from successful actions
    failure_energy_penalty: float = 1.5           # Extra energy cost for failed actions
    learning_energy_reward: float = 3.0           # Energy gained from learning progress
    
    # Sleep and recovery
    sleep_trigger_threshold: float = 30.0         # Energy level to trigger sleep
    emergency_sleep_threshold: float = 10.0       # Emergency sleep trigger
    sleep_energy_restoration: float = 100.0       # Energy restored during sleep
    
    # Death mechanics
    death_threshold: float = 0.0                  # Energy level for death
    respawn_energy: float = 100.0                 # Energy on respawn
    death_penalty: float = 10.0                   # Energy penalty for death
    
    # Adaptive parameters
    performance_window: int = 50                  # Actions to consider for performance
    adaptation_rate: float = 0.1                  # How quickly to adapt to performance
    
    def __post_init__(self):
        """Initialize default action costs if not provided."""
        if self.action_costs is None:
            self.action_costs = {
                1: 0.5,   # Movement actions (low cost)
                2: 0.5,
                3: 0.5,
                4: 0.5,
                5: 1.0,   # Interaction actions (medium cost)
                6: 2.0,   # Coordinate actions (high cost)
                7: 0.1,   # Undo actions (very low cost)
                8: 1.5,   # Special actions (high cost)
            }


class UnifiedEnergySystem:
    """
    Unified energy management system with 0-100 scale survival mechanics.
    """
    
    def __init__(self, config: Optional[EnergyConfig] = None):
        self.config = config or EnergyConfig()
        
        # Current energy state
        self.current_energy = self.config.max_energy
        self.energy_state = EnergyState.HEALTHY
        
        # Performance tracking
        self.recent_actions = []
        self.recent_successes = []
        self.recent_learning_progress = []
        self.consecutive_failures = 0
        self.total_actions_taken = 0
        self.total_deaths = 0
        
        # Energy history and metrics
        self.energy_history = []
        self.consumption_history = []
        self.performance_history = []
        
        # Adaptive parameters
        self.current_action_cost_multiplier = 1.0
        self.current_consumption_rate = self.config.base_consumption_per_second
        
        # Session tracking
        self.session_start_time = time.time()
        self.last_action_time = time.time()
        self.last_sleep_time = time.time()
        
        logger.info(f"UnifiedEnergySystem initialized with {self.current_energy:.1f} energy")
    
    def get_energy_state(self) -> EnergyState:
        """Get current energy state based on energy level."""
        if self.current_energy <= self.config.death_threshold:
            return EnergyState.DEAD
        elif self.current_energy < 5:
            return EnergyState.CRITICAL
        elif self.current_energy < 20:
            return EnergyState.LOW
        elif self.current_energy < 40:
            return EnergyState.MODERATE
        else:
            return EnergyState.HEALTHY
    
    def consume_energy_for_action(
        self, 
        action_id: int, 
        success: bool = False, 
        learning_progress: float = 0.0,
        computation_cost: float = 0.0
    ) -> Dict[str, Any]:
        """
        Consume energy for an action with performance-based adjustments.
        
        Args:
            action_id: ID of the action being performed
            success: Whether the action was successful
            learning_progress: Learning progress from this action (0-1)
            computation_cost: Additional computational cost
            
        Returns:
            Dictionary with energy consumption details
        """
        # Calculate base action cost
        base_cost = self.config.action_costs.get(action_id, 1.0)
        
        # Apply adaptive multiplier based on recent performance
        adaptive_cost = base_cost * self.current_action_cost_multiplier
        
        # Apply performance-based adjustments
        if success:
            # Successful actions cost less and may provide energy bonus
            final_cost = adaptive_cost * 0.8  # 20% discount for success
            energy_bonus = self.config.success_energy_bonus
        else:
            # Failed actions cost more
            final_cost = adaptive_cost * self.config.failure_energy_penalty
            energy_bonus = 0.0
        
        # Add learning progress bonus
        if learning_progress > 0:
            energy_bonus += learning_progress * self.config.learning_energy_reward
        
        # Add computational cost
        final_cost += computation_cost * 0.1  # Small computational cost
        
        # Apply energy consumption
        old_energy = self.current_energy
        self.current_energy = max(
            self.config.min_energy, 
            self.current_energy - final_cost + energy_bonus
        )
        
        # Update energy state
        self.energy_state = self.get_energy_state()
        
        # Track performance
        self._update_performance_tracking(action_id, success, learning_progress)
        
        # Update adaptive parameters
        self._update_adaptive_parameters()
        
        # Record consumption
        consumption_record = {
            'timestamp': time.time(),
            'action_id': action_id,
            'base_cost': base_cost,
            'adaptive_cost': adaptive_cost,
            'final_cost': final_cost,
            'energy_bonus': energy_bonus,
            'energy_before': old_energy,
            'energy_after': self.current_energy,
            'success': success,
            'learning_progress': learning_progress,
            'energy_state': self.energy_state.value
        }
        
        self.consumption_history.append(consumption_record)
        self.energy_history.append(self.current_energy)
        
        # Keep history bounded
        if len(self.energy_history) > 10000:
            self.energy_history = self.energy_history[-5000:]
            self.consumption_history = self.consumption_history[-5000:]
        
        self.total_actions_taken += 1
        self.last_action_time = time.time()
        
        return consumption_record
    
    def consume_energy_over_time(self, time_elapsed: float) -> float:
        """
        Consume energy over time (for idle periods).
        
        Args:
            time_elapsed: Time elapsed in seconds
            
        Returns:
            Energy consumed
        """
        energy_consumed = time_elapsed * self.current_consumption_rate
        self.current_energy = max(self.config.min_energy, self.current_energy - energy_consumed)
        
        # Update energy state
        self.energy_state = self.get_energy_state()
        
        # Record time-based consumption
        self.energy_history.append(self.current_energy)
        
        return energy_consumed
    
    def add_energy(self, amount: float, source: str = "unknown") -> float:
        """
        Add energy to the system.
        
        Args:
            amount: Amount of energy to add
            source: Source of the energy (for tracking)
            
        Returns:
            New energy level
        """
        old_energy = self.current_energy
        self.current_energy = min(self.config.max_energy, self.current_energy + amount)
        
        # Update energy state
        self.energy_state = self.get_energy_state()
        
        logger.info(f"Energy added: +{amount:.1f} from {source} "
                   f"({old_energy:.1f} -> {self.current_energy:.1f})")
        
        return self.current_energy
    
    def trigger_sleep(self) -> Dict[str, Any]:
        """
        Trigger sleep cycle to restore energy.
        
        Returns:
            Sleep cycle information
        """
        old_energy = self.current_energy
        energy_restored = self.config.sleep_energy_restoration
        
        # Restore energy
        self.current_energy = min(self.config.max_energy, self.current_energy + energy_restored)
        self.energy_state = self.get_energy_state()
        
        # Update tracking
        self.last_sleep_time = time.time()
        
        sleep_info = {
            'timestamp': time.time(),
            'energy_before': old_energy,
            'energy_after': self.current_energy,
            'energy_restored': energy_restored,
            'sleep_cycle_number': len([h for h in self.consumption_history if h.get('sleep_triggered', False)]) + 1
        }
        
        logger.info(f"Sleep cycle completed: {old_energy:.1f} -> {self.current_energy:.1f} "
                   f"(+{energy_restored:.1f})")
        
        return sleep_info
    
    def handle_death(self) -> Dict[str, Any]:
        """
        Handle agent death and respawn.
        
        Returns:
            Death and respawn information
        """
        self.total_deaths += 1
        
        # Apply death penalty
        death_penalty = self.config.death_penalty
        self.current_energy = max(0, self.current_energy - death_penalty)
        
        # Respawn with full energy
        old_energy = self.current_energy
        self.current_energy = self.config.respawn_energy
        self.energy_state = self.get_energy_state()
        
        # Reset some performance tracking
        self.consecutive_failures = 0
        
        death_info = {
            'timestamp': time.time(),
            'death_number': self.total_deaths,
            'energy_before_death': old_energy,
            'death_penalty': death_penalty,
            'respawn_energy': self.current_energy,
            'total_actions_before_death': self.total_actions_taken
        }
        
        logger.warning(f"Agent death #{self.total_deaths}: {old_energy:.1f} -> {self.current_energy:.1f}")
        
        return death_info
    
    def should_sleep(self) -> Tuple[bool, str]:
        """
        Check if the agent should enter sleep mode.
        
        Returns:
            Tuple of (should_sleep, reason)
        """
        if self.current_energy <= self.config.emergency_sleep_threshold:
            return True, f"emergency_low_energy ({self.current_energy:.1f} <= {self.config.emergency_sleep_threshold})"
        
        if self.current_energy <= self.config.sleep_trigger_threshold:
            return True, f"low_energy ({self.current_energy:.1f} <= {self.config.sleep_trigger_threshold})"
        
        # Time-based sleep trigger (every 5 minutes of activity)
        time_since_last_sleep = time.time() - self.last_sleep_time
        if time_since_last_sleep > 300:  # 5 minutes
            return True, f"time_based ({time_since_last_sleep:.1f}s since last sleep)"
        
        # Performance-based sleep trigger
        if self.consecutive_failures >= 10:
            return True, f"performance_based ({self.consecutive_failures} consecutive failures)"
        
        return False, "no_trigger"
    
    def is_dead(self) -> bool:
        """Check if the agent is dead."""
        return self.current_energy <= self.config.death_threshold
    
    def get_energy_ratio(self) -> float:
        """Get energy as a ratio of maximum (0-1)."""
        return self.current_energy / self.config.max_energy
    
    def _update_performance_tracking(
        self, 
        action_id: int, 
        success: bool, 
        learning_progress: float
    ):
        """Update performance tracking for adaptive energy management."""
        # Track recent actions
        self.recent_actions.append({
            'action_id': action_id,
            'success': success,
            'learning_progress': learning_progress,
            'timestamp': time.time()
        })
        
        # Keep only recent actions
        if len(self.recent_actions) > self.config.performance_window:
            self.recent_actions = self.recent_actions[-self.config.performance_window:]
        
        # Track successes
        self.recent_successes.append(success)
        if len(self.recent_successes) > self.config.performance_window:
            self.recent_successes = self.recent_successes[-self.config.performance_window:]
        
        # Track learning progress
        self.recent_learning_progress.append(learning_progress)
        if len(self.recent_learning_progress) > self.config.performance_window:
            self.recent_learning_progress = self.recent_learning_progress[-self.config.performance_window:]
        
        # Update consecutive failures
        if success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
    
    def _update_adaptive_parameters(self):
        """Update adaptive parameters based on recent performance."""
        if len(self.recent_successes) < 10:  # Need minimum data
            return
        
        # Calculate recent success rate
        recent_success_rate = sum(self.recent_successes) / len(self.recent_successes)
        
        # Calculate recent learning progress
        recent_learning = np.mean(self.recent_learning_progress) if self.recent_learning_progress else 0.0
        
        # Adjust action cost multiplier based on performance
        if recent_success_rate > 0.7 and recent_learning > 0.1:
            # Good performance - reduce costs
            target_multiplier = 0.8
        elif recent_success_rate < 0.3 or self.consecutive_failures > 5:
            # Poor performance - increase costs to encourage more careful action selection
            target_multiplier = 1.2
        else:
            # Neutral performance - maintain current costs
            target_multiplier = 1.0
        
        # Smoothly adapt to target multiplier
        self.current_action_cost_multiplier = (
            (1 - self.config.adaptation_rate) * self.current_action_cost_multiplier +
            self.config.adaptation_rate * target_multiplier
        )
        
        # Adjust consumption rate based on performance
        if recent_success_rate > 0.8:
            # High success rate - reduce idle consumption
            self.current_consumption_rate = self.config.base_consumption_per_second * 0.8
        elif recent_success_rate < 0.2:
            # Low success rate - increase idle consumption to encourage action
            self.current_consumption_rate = self.config.base_consumption_per_second * 1.2
        else:
            # Normal consumption rate
            self.current_consumption_rate = self.config.base_consumption_per_second
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive energy system status."""
        current_time = time.time()
        session_duration = current_time - self.session_start_time
        
        # Calculate recent performance metrics
        recent_success_rate = 0.0
        recent_learning_progress = 0.0
        if self.recent_successes:
            recent_success_rate = sum(self.recent_successes) / len(self.recent_successes)
        if self.recent_learning_progress:
            recent_learning_progress = np.mean(self.recent_learning_progress)
        
        # Calculate energy trend
        energy_trend = 0.0
        if len(self.energy_history) > 10:
            recent_energy = self.energy_history[-10:]
            x = np.arange(len(recent_energy))
            energy_trend = np.polyfit(x, recent_energy, 1)[0]
        
        return {
            'current_energy': self.current_energy,
            'max_energy': self.config.max_energy,
            'energy_ratio': self.get_energy_ratio(),
            'energy_state': self.energy_state.value,
            'is_dead': self.is_dead(),
            'total_actions_taken': self.total_actions_taken,
            'total_deaths': self.total_deaths,
            'consecutive_failures': self.consecutive_failures,
            'recent_success_rate': recent_success_rate,
            'recent_learning_progress': recent_learning_progress,
            'current_action_cost_multiplier': self.current_action_cost_multiplier,
            'current_consumption_rate': self.current_consumption_rate,
            'energy_trend': energy_trend,
            'session_duration_seconds': session_duration,
            'time_since_last_action': current_time - self.last_action_time,
            'time_since_last_sleep': current_time - self.last_sleep_time,
            'should_sleep': self.should_sleep()[0],
            'sleep_reason': self.should_sleep()[1]
        }
    
    def get_energy_metrics(self) -> Dict[str, Any]:
        """Get detailed energy metrics for analysis."""
        if not self.energy_history:
            return {'error': 'No energy history available'}
        
        # Calculate consumption statistics
        recent_consumption = [c['final_cost'] for c in self.consumption_history[-100:]]
        recent_energy = self.energy_history[-100:]
        
        # Calculate action cost statistics by action type
        action_costs = {}
        for action_id in self.config.action_costs.keys():
            action_consumptions = [
                c['final_cost'] for c in self.consumption_history[-100:]
                if c['action_id'] == action_id
            ]
            if action_consumptions:
                action_costs[action_id] = {
                    'count': len(action_consumptions),
                    'avg_cost': np.mean(action_consumptions),
                    'total_cost': np.sum(action_consumptions)
                }
        
        # Calculate performance-based energy efficiency
        successful_actions = [c for c in self.consumption_history[-100:] if c['success']]
        failed_actions = [c for c in self.consumption_history[-100:] if not c['success']]
        
        avg_successful_cost = np.mean([c['final_cost'] for c in successful_actions]) if successful_actions else 0
        avg_failed_cost = np.mean([c['final_cost'] for c in failed_actions]) if failed_actions else 0
        
        return {
            'energy_history_length': len(self.energy_history),
            'consumption_history_length': len(self.consumption_history),
            'avg_energy_level': np.mean(recent_energy),
            'min_energy_level': np.min(recent_energy),
            'max_energy_level': np.max(recent_energy),
            'avg_consumption_per_action': np.mean(recent_consumption) if recent_consumption else 0,
            'total_energy_consumed': np.sum(recent_consumption) if recent_consumption else 0,
            'action_costs': action_costs,
            'avg_successful_action_cost': avg_successful_cost,
            'avg_failed_action_cost': avg_failed_cost,
            'energy_efficiency_ratio': avg_successful_cost / max(avg_failed_cost, 0.1) if avg_failed_cost > 0 else 1.0,
            'recent_energy_trend': np.polyfit(range(len(recent_energy)), recent_energy, 1)[0] if len(recent_energy) > 1 else 0
        }
    
    def reset(self):
        """Reset the energy system for a new session."""
        self.current_energy = self.config.max_energy
        self.energy_state = EnergyState.HEALTHY
        self.recent_actions = []
        self.recent_successes = []
        self.recent_learning_progress = []
        self.consecutive_failures = 0
        self.total_actions_taken = 0
        self.total_deaths = 0
        self.current_action_cost_multiplier = 1.0
        self.current_consumption_rate = self.config.base_consumption_per_second
        self.session_start_time = time.time()
        self.last_action_time = time.time()
        self.last_sleep_time = time.time()
        
        logger.info("Energy system reset for new session")


class EnergySystemIntegration:
    """
    Integration layer for connecting the unified energy system with training loops.
    """
    
    def __init__(self, energy_system: UnifiedEnergySystem):
        self.energy_system = energy_system
        self.integration_active = False
    
    def integrate_with_training_loop(self, training_loop) -> bool:
        """Integrate energy system with a training loop."""
        try:
            # Set up energy system in training loop
            training_loop.energy_system = self.energy_system
            training_loop.current_energy = self.energy_system.current_energy
            
            # Override energy-related methods if they exist
            if hasattr(training_loop, 'consume_energy'):
                training_loop.consume_energy = self._wrap_consume_energy(training_loop.consume_energy)
            
            self.integration_active = True
            logger.info("Energy system integrated with training loop")
            return True
            
        except Exception as e:
            logger.error(f"Failed to integrate energy system: {e}")
            return False
    
    def _wrap_consume_energy(self, original_method):
        """Wrap the original consume_energy method with unified energy management."""
        def wrapped_consume_energy(action_id, success=False, learning_progress=0.0, **kwargs):
            # Use unified energy system
            consumption_record = self.energy_system.consume_energy_for_action(
                action_id, success, learning_progress
            )
            
            # Call original method if it exists
            if original_method:
                original_method(action_id, success, learning_progress, **kwargs)
            
            return consumption_record
        
        return wrapped_consume_energy
    
    def update_during_training(
        self, 
        action_id: int, 
        success: bool = False, 
        learning_progress: float = 0.0
    ) -> Dict[str, Any]:
        """Update energy system during training."""
        if not self.integration_active:
            return {'error': 'Integration not active'}
        
        # Consume energy for the action
        consumption_record = self.energy_system.consume_energy_for_action(
            action_id, success, learning_progress
        )
        
        # Check for sleep trigger
        should_sleep, sleep_reason = self.energy_system.should_sleep()
        
        # Check for death
        is_dead = self.energy_system.is_dead()
        
        return {
            'consumption_record': consumption_record,
            'should_sleep': should_sleep,
            'sleep_reason': sleep_reason,
            'is_dead': is_dead,
            'current_energy': self.energy_system.current_energy,
            'energy_state': self.energy_system.energy_state.value
        }
