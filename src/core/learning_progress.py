"""
Learning Progress Drive - Core intrinsic motivation system.

This module implements the learning progress calculation with extensive
stability measures and validation capabilities.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class LearningProgressDrive:
    """
    Computes intrinsic reward based on learning progress.
    
    CRITICAL: This is the highest-risk component. Extensive validation required.
    """
    
    def __init__(
        self,
        smoothing_window: int = 500,
        derivative_clamp: Tuple[float, float] = (-1.0, 1.0),
        boredom_threshold: float = 0.01,
        boredom_steps: int = 500,
        lp_weight: float = 0.7,
        empowerment_weight: float = 0.3,
        use_adaptive_weights: bool = False
    ):
        self.smoothing_window = smoothing_window
        self.derivative_clamp = derivative_clamp
        self.boredom_threshold = boredom_threshold
        self.boredom_steps = boredom_steps
        self.lp_weight = lp_weight
        self.empowerment_weight = empowerment_weight
        self.use_adaptive_weights = use_adaptive_weights
        
        # Error tracking with per-modality normalization
        self.error_history = deque(maxlen=smoothing_window)
        self.running_mean = 0.0
        self.running_std = 1.0
        self.step_count = 0
        
        # Boredom tracking
        self.low_lp_counter = 0
        
        # Adaptive weighting (if enabled)
        self.lp_history = deque(maxlen=1000)
        self.empowerment_history = deque(maxlen=1000)
        
        # Validation metrics
        self.validation_metrics = {
            'signal_to_noise_ratio': 0.0,
            'stability_score': 0.0,
            'outlier_rate': 0.0
        }
        
    def update_error_statistics(self, prediction_error: float):
        """Update running statistics for error normalization."""
        self.step_count += 1
        
        # Robust running statistics with outlier detection
        if self.step_count == 1:
            self.running_mean = prediction_error
            self.running_std = 1.0
        else:
            # Detect outliers before updating statistics
            z_score = abs(prediction_error - self.running_mean) / max(self.running_std, 1e-6)
            if z_score < 5.0:  # Only update with non-outliers
                alpha = min(0.01, 1.0 / self.step_count)
                self.running_mean = (1 - alpha) * self.running_mean + alpha * prediction_error
                
                # Update variance estimate
                variance = (1 - alpha) * (self.running_std ** 2) + alpha * ((prediction_error - self.running_mean) ** 2)
                self.running_std = max(np.sqrt(variance), 1e-6)
            else:
                logger.warning(f"Outlier detected: error={prediction_error:.4f}, z_score={z_score:.2f}")
                
    def compute_learning_progress(self, prediction_error: float) -> float:
        """
        Compute learning progress signal with stability measures.
        
        Args:
            prediction_error: Current prediction error magnitude
            
        Returns:
            Learning progress signal (higher = more learning)
        """
        # Update error statistics
        self.update_error_statistics(prediction_error)
        
        # Normalize error
        normalized_error = prediction_error / max(self.running_std, 1e-6)
        
        # Add to history
        self.error_history.append(normalized_error)
        
        # Need sufficient history for derivative
        if len(self.error_history) < 10:
            return 0.0
            
        # Compute robust derivative (negative of error change)
        recent_errors = list(self.error_history)[-10:]
        older_errors = list(self.error_history)[-20:-10] if len(self.error_history) >= 20 else recent_errors
        
        recent_mean = np.mean(recent_errors)
        older_mean = np.mean(older_errors)
        
        # Learning progress = negative derivative of error (error decreasing = positive LP)
        raw_lp = older_mean - recent_mean
        
        # Clamp to prevent noise amplification
        clamped_lp = np.clip(raw_lp, self.derivative_clamp[0], self.derivative_clamp[1])
        
        # Update validation metrics
        self._update_validation_metrics(raw_lp, clamped_lp)
        
        # Update boredom counter
        if abs(clamped_lp) < self.boredom_threshold:
            self.low_lp_counter += 1
        else:
            self.low_lp_counter = 0
        
        return float(clamped_lp)
        
    def compute_empowerment_bonus(self, state_history: List[torch.Tensor]) -> float:
        """
        Compute empowerment approximation based on state diversity.
        
        Args:
            state_history: Recent states for diversity calculation
            
        Returns:
            Empowerment bonus (0-1 range)
        """
        if len(state_history) < 2:
            return 0.0
            
        # Take last 10 states for diversity calculation
        recent_states = state_history[-10:]
        
        if len(recent_states) < 2:
            return 0.0
            
        # Compute state diversity as standard deviation
        states_tensor = torch.stack(recent_states)
        state_diversity = torch.std(states_tensor, dim=0).mean()
        
        # Normalize to 0-1 range
        normalized_diversity = torch.clamp(state_diversity, 0, 1)
        
        return float(normalized_diversity)
        
    def compute_reward(
        self, 
        prediction_error: float, 
        state_history: Optional[List[torch.Tensor]] = None
    ) -> float:
        """
        Compute total intrinsic reward signal.
        
        Args:
            prediction_error: Current prediction error
            state_history: Recent states for empowerment calculation
            
        Returns:
            Combined reward signal
        """
        # Compute learning progress
        lp_signal = self.compute_learning_progress(prediction_error)
        
        # Compute empowerment bonus
        empowerment_bonus = 0.0
        if state_history is not None:
            empowerment_bonus = self.compute_empowerment_bonus(state_history)
            
        # Store for adaptive weighting
        self.lp_history.append(lp_signal)
        self.empowerment_history.append(empowerment_bonus)
        
        # Compute weights
        if self.use_adaptive_weights:
            lp_weight, emp_weight = self._compute_adaptive_weights()
        else:
            lp_weight, emp_weight = self.lp_weight, self.empowerment_weight
            
        # Combined reward
        total_reward = lp_weight * lp_signal + emp_weight * empowerment_bonus
        
        # Note: boredom counter is updated in compute_learning_progress
        return total_reward
        
    def is_bored(self) -> bool:
        """Check if agent is experiencing boredom (low LP for extended period)."""
        return self.low_lp_counter >= self.boredom_steps
        
    def reset_boredom_counter(self):
        """Reset boredom counter (called when new goals invented or complexity increases)."""
        self.low_lp_counter = 0
        
    def _compute_adaptive_weights(self) -> Tuple[float, float]:
        """
        Compute adaptive weights between LP and empowerment.
        
        CRITICAL: Only activate if conflicts are empirically observed.
        """
        if len(self.lp_history) < 100:
            return self.lp_weight, self.empowerment_weight
            
        # Detect trends
        lp_trend = self._compute_trend(list(self.lp_history))
        emp_trend = self._compute_trend(list(self.empowerment_history))
        
        # Detect conflicts
        correlation = np.corrcoef(list(self.lp_history), list(self.empowerment_history))[0, 1]
        
        # Adaptive logic
        if abs(lp_trend) < 0.01 and emp_trend > 0.1:
            # LP stagnant, empowerment progressing
            return 0.3, 0.7
        elif correlation < -0.3:
            # Strong conflict, prioritize LP
            return 0.9, 0.1
        else:
            # Default balanced approach
            return 0.7, 0.3
            
    def _compute_trend(self, values: List[float]) -> float:
        """Compute trend (slope) of recent values."""
        if len(values) < 10:
            return 0.0
            
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope
        
    def _update_validation_metrics(self, raw_lp: float, clamped_lp: float):
        """Update validation metrics for signal quality assessment."""
        if len(self.error_history) < self.smoothing_window // 2:
            return
            
        # Signal-to-noise ratio (avoid recursion by using raw LP values)
        recent_errors = list(self.error_history)[-50:]
        if len(recent_errors) > 10:
            # Compute LP manually without recursion
            recent_lp_values = []
            for i in range(10, len(recent_errors)):
                older_mean = np.mean(recent_errors[max(0, i-10):i])
                newer_mean = np.mean(recent_errors[max(0, i-5):i])
                lp_val = older_mean - newer_mean
                recent_lp_values.append(np.clip(lp_val, self.derivative_clamp[0], self.derivative_clamp[1]))
            
            if len(recent_lp_values) > 1:
                signal_power = np.var(recent_lp_values)
                noise_power = np.var(np.diff(recent_lp_values))  # High-frequency noise
                self.validation_metrics['signal_to_noise_ratio'] = signal_power / max(noise_power, 1e-6)
        
        # Stability score (how often we need to clamp)
        clamp_rate = 1.0 if raw_lp != clamped_lp else 0.0
        alpha = 0.01
        self.validation_metrics['stability_score'] = (
            (1 - alpha) * self.validation_metrics['stability_score'] + 
            alpha * (1.0 - clamp_rate)
        )
        
    def get_validation_metrics(self) -> Dict[str, float]:
        """Get current validation metrics for monitoring."""
        return self.validation_metrics.copy()
        
    def validate_signal_quality(self) -> bool:
        """
        Validate LP signal quality.
        
        Returns:
            True if signal quality is acceptable
        """
        metrics = self.get_validation_metrics()
        
        # Quality thresholds
        min_snr = 2.0
        min_stability = 0.8
        
        return (
            metrics['signal_to_noise_ratio'] > min_snr and
            metrics['stability_score'] > min_stability
        )