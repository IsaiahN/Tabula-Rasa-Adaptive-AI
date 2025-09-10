"""
Adaptive Energy System - Smart energy management based on action limits and performance.

This system adapts energy depletion rates and sleep triggers based on:
1. Maximum actions available in the session
2. Time-based limits when actions are unlimited
3. Performance patterns and success rates
4. Dynamic adjustment based on gameplay progress
"""

import time
import math
import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EnergyConfig:
    """Configuration for adaptive energy management."""
    max_energy: float = 100.0
    base_depletion_rate: float = 0.005  # Reduced from 0.01 to 0.005 - Energy per second when unlimited actions
    action_based_depletion: float = 0.05  # Reduced from 0.1 to 0.05 - Energy per action when actions are limited
    sleep_trigger_threshold: float = 30.0  # Reduced from 40.0 to 30.0 - Default sleep threshold
    min_sleep_threshold: float = 15.0  # Reduced from 20.0 to 15.0 - Minimum threshold (emergency sleep)
    max_sleep_threshold: float = 50.0  # Reduced from 60.0 to 50.0 - Maximum threshold (early strategic sleep)
    
    # Time-based sleep for unlimited actions (in minutes)
    time_based_sleep_interval: float = 5.0  # Sleep every 5 minutes by default
    min_time_interval: float = 2.0  # Minimum 2 minutes
    max_time_interval: float = 15.0  # Maximum 15 minutes
    
    # Performance-based adjustments
    success_rate_multiplier: float = 0.8  # Reduce energy consumption when successful
    failure_rate_multiplier: float = 1.2  # Increase energy consumption when failing


class AdaptiveEnergySystem:
    """
    Adaptive energy system that adjusts based on action limits and performance.
    """
    
    def __init__(self, config: Optional[EnergyConfig] = None):
        self.config = config or EnergyConfig()
        
        # Current energy state
        self.current_energy = self.config.max_energy
        self.start_time = time.time()
        self.last_sleep_time = self.start_time
        self.total_actions_taken = 0
        
        # Session configuration
        self.max_actions_available = None  # Set by training session
        self.session_time_limit = None  # Set by training session (in seconds)
        self.unlimited_actions = True  # Default to unlimited
        
        # Performance tracking
        self.recent_success_rate = 0.5  # Start neutral
        self.recent_failures = 0
        self.consecutive_failures = 0
        self.last_score_improvement = 0.0
        
        # Adaptive parameters (calculated dynamically)
        self.current_sleep_threshold = self.config.sleep_trigger_threshold
        self.current_sleep_interval = self.config.time_based_sleep_interval
        self.current_depletion_rate = self.config.base_depletion_rate
        
        # Tracking
        self.sleep_cycles_completed = 0
        self.energy_history = []
        self.adaptation_history = []
        
        logger.info(f"AdaptiveEnergySystem initialized with config: {self.config}")
    
    def configure_session(
        self, 
        max_actions: Optional[int] = None, 
        time_limit_minutes: Optional[float] = None
    ):
        """Configure the energy system for a specific training session."""
        self.max_actions_available = max_actions
        self.session_time_limit = time_limit_minutes * 60 if time_limit_minutes else None
        self.unlimited_actions = max_actions is None or max_actions >= 100000
        
        # Reset session state
        self.total_actions_taken = 0
        self.start_time = time.time()
        self.last_sleep_time = self.start_time
        
        # Calculate adaptive parameters based on session type
        self._recalculate_energy_parameters()
        
        logger.info(f"Session configured - Actions: {max_actions or 'UNLIMITED'}, "
                   f"Time limit: {time_limit_minutes or 'NONE'} min, "
                   f"Sleep threshold: {self.current_sleep_threshold:.1f}, "
                   f"Sleep interval: {self.current_sleep_interval:.1f} min")
    
    def _recalculate_energy_parameters(self):
        """Recalculate energy parameters based on current session and performance."""
        if self.unlimited_actions:
            # Time-based system for unlimited actions
            self._calculate_time_based_parameters()
        else:
            # Action-based system for limited actions
            self._calculate_action_based_parameters()
        
        # Apply performance-based adjustments
        self._apply_performance_adjustments()
        
        # Log the adaptation
        adaptation_info = {
            'sleep_threshold': self.current_sleep_threshold,
            'sleep_interval': self.current_sleep_interval,
            'depletion_rate': self.current_depletion_rate,
            'success_rate': self.recent_success_rate,
            'consecutive_failures': self.consecutive_failures,
            'timestamp': time.time()
        }
        self.adaptation_history.append(adaptation_info)
        
        logger.info(f"Energy parameters recalculated: threshold={self.current_sleep_threshold:.1f}, "
                   f"interval={self.current_sleep_interval:.1f}min, rate={self.current_depletion_rate:.4f}")
    
    def _calculate_time_based_parameters(self):
        """Calculate parameters for unlimited action sessions (time-based)."""
        # Base sleep interval (5 minutes default)
        self.current_sleep_interval = self.config.time_based_sleep_interval
        
        # Energy depletion rate to trigger sleep at the desired interval
        # Energy should deplete from 100 to sleep_threshold over the interval
        energy_to_deplete = self.config.max_energy - self.current_sleep_threshold
        interval_seconds = self.current_sleep_interval * 60
        self.current_depletion_rate = energy_to_deplete / interval_seconds
        
        logger.debug(f"Time-based: interval={self.current_sleep_interval:.1f}min, "
                    f"rate={self.current_depletion_rate:.4f}/sec")
    
    def _calculate_action_based_parameters(self):
        """Calculate parameters for limited action sessions (action-based)."""
        if self.max_actions_available and self.max_actions_available > 0:
            # Calculate how often to sleep based on actions available
            # For example, sleep every 10% of total actions
            sleep_action_interval = max(100, self.max_actions_available * 0.1)
            
            # Energy depletion per action to trigger sleep at intervals
            energy_per_interval = self.config.max_energy - self.current_sleep_threshold
            self.current_depletion_rate = energy_per_interval / sleep_action_interval
            
            logger.debug(f"Action-based: {self.max_actions_available} max actions, "
                        f"sleep every ~{sleep_action_interval:.0f} actions, "
                        f"rate={self.current_depletion_rate:.4f}/action")
        else:
            # Fallback to default
            self.current_depletion_rate = self.config.action_based_depletion
    
    def _apply_performance_adjustments(self):
        """Adjust parameters based on recent performance."""
        # Adjust sleep threshold based on success rate
        if self.recent_success_rate > 0.7:
            # Doing well - can afford to be more efficient (higher threshold = less frequent sleep)
            adjustment = (self.recent_success_rate - 0.5) * 20  # Up to +10 adjustment
            self.current_sleep_threshold = min(
                self.config.max_sleep_threshold,
                self.config.sleep_trigger_threshold + adjustment
            )
        elif self.recent_success_rate < 0.3:
            # Struggling - need more frequent consolidation (lower threshold = more frequent sleep)
            adjustment = (0.5 - self.recent_success_rate) * 20  # Up to -10 adjustment
            self.current_sleep_threshold = max(
                self.config.min_sleep_threshold,
                self.config.sleep_trigger_threshold - adjustment
            )
        
        # Adjust sleep frequency based on consecutive failures
        if self.consecutive_failures > 5:
            # Too many failures - increase sleep frequency
            failure_multiplier = min(2.0, 1.0 + (self.consecutive_failures - 5) * 0.1)
            if self.unlimited_actions:
                self.current_sleep_interval = max(
                    self.config.min_time_interval,
                    self.current_sleep_interval / failure_multiplier
                )
            self.current_sleep_threshold = max(
                self.config.min_sleep_threshold,
                self.current_sleep_threshold - 5
            )
            
            logger.info(f"High failure rate detected ({self.consecutive_failures} consecutive), "
                       f"increasing sleep frequency: threshold={self.current_sleep_threshold:.1f}")
    
    def update_energy(self, actions_taken: int = 0, time_elapsed: Optional[float] = None) -> float:
        """Update energy based on actions taken or time elapsed."""
        if time_elapsed is None:
            time_elapsed = time.time() - self.start_time
        
        self.total_actions_taken += actions_taken
        
        if self.unlimited_actions:
            # Time-based energy depletion
            energy_consumed = time_elapsed * self.current_depletion_rate
        else:
            # Action-based energy depletion
            energy_consumed = actions_taken * self.current_depletion_rate
        
        self.current_energy = max(0, self.current_energy - energy_consumed)
        
        # Track energy history
        self.energy_history.append({
            'timestamp': time.time(),
            'energy': self.current_energy,
            'actions_taken': self.total_actions_taken,
            'consumed': energy_consumed
        })
        
        return self.current_energy
    
    def should_sleep(self) -> Tuple[bool, str]:
        """Check if the agent should enter sleep mode."""
        current_time = time.time()
        
        # Energy threshold check
        if self.current_energy <= self.current_sleep_threshold:
            return True, f"energy_threshold ({self.current_energy:.1f} <= {self.current_sleep_threshold:.1f})"
        
        # Time-based check for unlimited actions
        if self.unlimited_actions:
            time_since_last_sleep = (current_time - self.last_sleep_time) / 60  # minutes
            if time_since_last_sleep >= self.current_sleep_interval:
                return True, f"time_interval ({time_since_last_sleep:.1f} >= {self.current_sleep_interval:.1f} min)"
        
        # Emergency sleep for critical performance issues
        if self.consecutive_failures >= 10:
            return True, f"emergency_performance ({self.consecutive_failures} consecutive failures)"
        
        # Strategic sleep when near session limits
        if self.max_actions_available:
            actions_remaining = self.max_actions_available - self.total_actions_taken
            if actions_remaining > 0 and actions_remaining <= 50:  # Close to limit
                return True, f"approaching_action_limit ({actions_remaining} actions left)"
        
        return False, "no_trigger"
    
    def trigger_sleep(self) -> Dict[str, Any]:
        """Trigger a sleep cycle and reset energy."""
        current_time = time.time()
        time_since_last_sleep = current_time - self.last_sleep_time
        
        sleep_info = {
            'trigger_time': current_time,
            'energy_before': self.current_energy,
            'time_since_last_sleep': time_since_last_sleep,
            'actions_since_last_sleep': self.total_actions_taken,
            'sleep_cycle_number': self.sleep_cycles_completed + 1
        }
        
        # Reset energy to full
        self.current_energy = self.config.max_energy
        self.last_sleep_time = current_time
        self.sleep_cycles_completed += 1
        
        # Recalculate parameters after sleep
        self._recalculate_energy_parameters()
        
        sleep_info['energy_after'] = self.current_energy
        sleep_info['new_sleep_threshold'] = self.current_sleep_threshold
        
        logger.info(f"Sleep cycle {self.sleep_cycles_completed} completed. "
                   f"Energy restored to {self.current_energy:.1f}. "
                   f"Next sleep threshold: {self.current_sleep_threshold:.1f}")
        
        return sleep_info
    
    def update_performance(
        self, 
        success: bool, 
        score_improvement: float = 0.0, 
        action_effectiveness: Optional[float] = None
    ):
        """Update performance tracking to influence energy management."""
        if success:
            self.consecutive_failures = 0
            score_improvement = max(0, score_improvement)
        else:
            self.consecutive_failures += 1
            self.recent_failures += 1
        
        self.last_score_improvement = score_improvement
        
        # Update rolling success rate (weighted recent history)
        if hasattr(self, '_recent_results'):
            if len(self._recent_results) >= 20:
                self._recent_results.pop(0)
        else:
            self._recent_results = []
        
        self._recent_results.append(success)
        self.recent_success_rate = sum(self._recent_results) / len(self._recent_results)
        
        # Trigger parameter recalculation if performance changed significantly
        if (self.consecutive_failures > 0 and self.consecutive_failures % 3 == 0) or \
           (success and self.consecutive_failures == 0 and len(self._recent_results) % 5 == 0):
            self._recalculate_energy_parameters()
        
        logger.debug(f"Performance updated: success={success}, consecutive_failures={self.consecutive_failures}, "
                    f"success_rate={self.recent_success_rate:.2f}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current energy system status."""
        current_time = time.time()
        session_duration = current_time - self.start_time
        time_since_last_sleep = current_time - self.last_sleep_time
        
        return {
            'current_energy': self.current_energy,
            'max_energy': self.config.max_energy,
            'sleep_threshold': self.current_sleep_threshold,
            'sleep_interval_minutes': self.current_sleep_interval,
            'depletion_rate': self.current_depletion_rate,
            'total_actions_taken': self.total_actions_taken,
            'max_actions_available': self.max_actions_available,
            'unlimited_actions': self.unlimited_actions,
            'session_duration_minutes': session_duration / 60,
            'time_since_last_sleep_minutes': time_since_last_sleep / 60,
            'sleep_cycles_completed': self.sleep_cycles_completed,
            'recent_success_rate': self.recent_success_rate,
            'consecutive_failures': self.consecutive_failures,
            'should_sleep': self.should_sleep()[0],
            'sleep_reason': self.should_sleep()[1]
        }
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of how the system has adapted over time."""
        if not self.adaptation_history:
            return {'adaptations': 0, 'message': 'No adaptations yet'}
        
        initial = self.adaptation_history[0]
        current = self.adaptation_history[-1]
        
        return {
            'adaptations': len(self.adaptation_history),
            'initial_threshold': initial['sleep_threshold'],
            'current_threshold': current['sleep_threshold'],
            'threshold_change': current['sleep_threshold'] - initial['sleep_threshold'],
            'initial_interval': initial['sleep_interval'],
            'current_interval': current['sleep_interval'],
            'interval_change': current['sleep_interval'] - initial['sleep_interval'],
            'performance_trend': 'improving' if current['success_rate'] > initial['success_rate'] else 'declining',
            'total_sleep_cycles': self.sleep_cycles_completed
        }


class EnergySystemIntegration:
    """Integration layer to connect AdaptiveEnergySystem with existing training systems."""
    
    def __init__(self, training_loop, adaptive_energy: AdaptiveEnergySystem):
        self.training_loop = training_loop
        self.adaptive_energy = adaptive_energy
        self.integration_active = False
        
    def integrate_with_training_session(
        self, 
        session_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Integrate adaptive energy with a training session configuration."""
        # Extract session parameters
        max_actions = session_config.get('max_actions_per_session')
        estimated_duration = session_config.get('estimated_duration_minutes')
        
        # Configure adaptive energy system
        self.adaptive_energy.configure_session(
            max_actions=max_actions,
            time_limit_minutes=estimated_duration
        )
        
        # Override sleep system parameters if available
        if hasattr(self.training_loop, 'sleep_system') and self.training_loop.sleep_system is not None:
            sleep_system = self.training_loop.sleep_system
            sleep_system.sleep_trigger_energy = self.adaptive_energy.current_sleep_threshold
            
            logger.info(f"Sleep system threshold updated to: {sleep_system.sleep_trigger_energy}")
        else:
            logger.info("Sleep system not available, skipping threshold update")
        
        self.integration_active = True
        
        # Return updated configuration with adaptive parameters
        updated_config = session_config.copy()
        updated_config['adaptive_sleep_threshold'] = self.adaptive_energy.current_sleep_threshold
        updated_config['adaptive_sleep_interval'] = self.adaptive_energy.current_sleep_interval
        updated_config['energy_management'] = 'adaptive'
        
        return updated_config
    
    def update_during_training(
        self, 
        actions_taken: int = 0, 
        success: bool = False, 
        score_improvement: float = 0.0
    ):
        """Update energy system during training."""
        if not self.integration_active:
            return
        
        # Update energy based on actions
        current_energy = self.adaptive_energy.update_energy(actions_taken)
        
        # Update performance tracking
        self.adaptive_energy.update_performance(success, score_improvement)
        
        # Check if sleep should be triggered
        should_sleep, reason = self.adaptive_energy.should_sleep()
        
        if should_sleep and hasattr(self.training_loop, 'sleep_system') and self.training_loop.sleep_system is not None:
            logger.info(f"Adaptive energy system triggering sleep: {reason}")
            sleep_info = self.adaptive_energy.trigger_sleep()
            
            # Trigger enhanced sleep in training loop if available
            if hasattr(self.training_loop, '_trigger_enhanced_sleep_with_arc_data'):
                return {
                    'sleep_triggered': True,
                    'sleep_reason': reason,
                    'sleep_info': sleep_info,
                    'current_energy': current_energy
                }
        
        return {
            'sleep_triggered': False,
            'current_energy': current_energy,
            'should_sleep': should_sleep,
            'sleep_reason': reason
        }
