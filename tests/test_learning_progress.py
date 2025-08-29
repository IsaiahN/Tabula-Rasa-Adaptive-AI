"""
Unit tests for Learning Progress Drive system.
"""

import pytest
import torch
import numpy as np
from src.core.learning_progress import LearningProgressDrive
from src.environment.synthetic_data import LPValidationSuite, SyntheticDataGenerator


class TestLearningProgressDrive:
    """Test suite for Learning Progress Drive."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lp_drive = LearningProgressDrive(
            smoothing_window=100,
            derivative_clamp=(-1.0, 1.0),
            boredom_threshold=0.01,
            boredom_steps=50
        )
        
    def test_initialization(self):
        """Test proper initialization of LP drive."""
        assert self.lp_drive.smoothing_window == 100
        assert self.lp_drive.derivative_clamp == (-1.0, 1.0)
        assert self.lp_drive.step_count == 0
        assert len(self.lp_drive.error_history) == 0
        
    def test_error_statistics_update(self):
        """Test running statistics update."""
        # First error
        self.lp_drive.update_error_statistics(1.0)
        assert self.lp_drive.running_mean == 1.0
        assert self.lp_drive.step_count == 1
        
        # Second error
        self.lp_drive.update_error_statistics(2.0)
        assert self.lp_drive.step_count == 2
        assert self.lp_drive.running_mean > 1.0  # Should be between 1 and 2
        
    def test_outlier_detection(self):
        """Test outlier detection in error statistics."""
        # Establish normal range
        for _ in range(10):
            self.lp_drive.update_error_statistics(1.0)
            
        initial_mean = self.lp_drive.running_mean
        
        # Inject extreme outlier
        self.lp_drive.update_error_statistics(100.0)
        
        # Mean should not be drastically affected
        assert abs(self.lp_drive.running_mean - initial_mean) < 0.5
        
    def test_learning_progress_calculation(self):
        """Test basic learning progress calculation."""
        # Simulate decreasing error (learning) - need more data points
        errors = []
        for i in range(30):
            error = 2.0 - i * 0.05  # Gradually decreasing error
            errors.append(max(0.1, error))  # Don't go below 0.1
        
        lp_signals = []
        for error in errors:
            lp = self.lp_drive.compute_learning_progress(error)
            lp_signals.append(lp)     
   
        # Should eventually show positive learning progress
        # Check that we get some positive LP signals in the later part
        later_signals = lp_signals[-10:]  # Last 10 signals
        assert any(lp > 0.0 for lp in later_signals), f"No positive LP in {later_signals}"
        
    def test_learning_progress_clamping(self):
        """Test that LP signals are properly clamped."""
        # Inject very noisy errors that could cause extreme derivatives
        errors = [0.1, 10.0, 0.1, 15.0, 0.1, 20.0]
        
        for error in errors:
            lp = self.lp_drive.compute_learning_progress(error)
            assert -1.0 <= lp <= 1.0  # Should be clamped
            
    def test_boredom_detection(self):
        """Test boredom detection mechanism."""
        # Generate constant low errors (no learning)
        # Need to generate enough history first
        for i in range(20):
            self.lp_drive.compute_learning_progress(0.1 + i * 0.001)  # Slight variation to build history
            
        # Now generate truly constant errors to trigger boredom
        for _ in range(60):  # More than boredom_steps threshold
            self.lp_drive.compute_learning_progress(0.1)
            
        assert self.lp_drive.is_bored()
        
        # Reset and verify boredom is cleared
        self.lp_drive.reset_boredom_counter()
        assert not self.lp_drive.is_bored()
        
    def test_empowerment_bonus(self):
        """Test empowerment bonus calculation."""
        # Create diverse state history
        diverse_states = [torch.randn(10) for _ in range(10)]
        
        # Create uniform state history
        uniform_states = [torch.ones(10) for _ in range(10)]
        
        diverse_bonus = self.lp_drive.compute_empowerment_bonus(diverse_states)
        uniform_bonus = self.lp_drive.compute_empowerment_bonus(uniform_states)
        
        # Diverse states should have higher empowerment
        assert diverse_bonus > uniform_bonus
        
    def test_reward_computation(self):
        """Test combined reward computation."""
        state_history = [torch.randn(10) for _ in range(5)]
        
        reward = self.lp_drive.compute_reward(1.0, state_history)
        
        # Should return a finite reward
        assert isinstance(reward, float)
        assert not np.isnan(reward)
        assert not np.isinf(reward)


if __name__ == '__main__':
    pytest.main([__file__])