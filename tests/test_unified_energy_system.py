#!/usr/bin/env python3
"""
Test suite for the unified energy management system.
"""

import unittest
import time
import numpy as np
from unittest.mock import Mock, patch

# Add the src directory to the path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.unified_energy_system import (
    UnifiedEnergySystem, EnergyConfig, EnergyState, 
    EnergySystemIntegration
)


class TestUnifiedEnergySystem(unittest.TestCase):
    """Test the unified energy management system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = EnergyConfig(
            max_energy=100.0,
            base_consumption_per_action=1.0,
            sleep_trigger_threshold=30.0,
            emergency_sleep_threshold=10.0
        )
        self.energy_system = UnifiedEnergySystem(self.config)
    
    def test_initialization(self):
        """Test energy system initialization."""
        self.assertEqual(self.energy_system.current_energy, 100.0)
        self.assertEqual(self.energy_system.energy_state, EnergyState.HEALTHY)
        self.assertEqual(self.energy_system.total_actions_taken, 0)
        self.assertEqual(self.energy_system.total_deaths, 0)
    
    def test_energy_state_calculation(self):
        """Test energy state calculation based on energy level."""
        # Test healthy state
        self.energy_system.current_energy = 80.0
        self.assertEqual(self.energy_system.get_energy_state(), EnergyState.HEALTHY)
        
        # Test moderate state
        self.energy_system.current_energy = 50.0
        self.assertEqual(self.energy_system.get_energy_state(), EnergyState.MODERATE)
        
        # Test low state
        self.energy_system.current_energy = 25.0
        self.assertEqual(self.energy_system.get_energy_state(), EnergyState.LOW)
        
        # Test critical state
        self.energy_system.current_energy = 15.0
        self.assertEqual(self.energy_system.get_energy_state(), EnergyState.CRITICAL)
        
        # Test dead state
        self.energy_system.current_energy = 0.0
        self.assertEqual(self.energy_system.get_energy_state(), EnergyState.DEAD)
    
    def test_action_energy_consumption(self):
        """Test energy consumption for actions."""
        # Test successful action
        consumption = self.energy_system.consume_energy_for_action(
            action_id=1, success=True, learning_progress=0.2
        )
        
        self.assertLess(consumption['energy_after'], consumption['energy_before'])
        self.assertGreater(consumption['energy_bonus'], 0)  # Should have bonus for success
        self.assertEqual(consumption['success'], True)
        self.assertEqual(consumption['action_id'], 1)
        
        # Test failed action
        consumption = self.energy_system.consume_energy_for_action(
            action_id=6, success=False, learning_progress=0.0
        )
        
        self.assertLess(consumption['energy_after'], consumption['energy_before'])
        self.assertEqual(consumption['energy_bonus'], 0)  # No bonus for failure
        self.assertEqual(consumption['success'], False)
        self.assertEqual(consumption['action_id'], 6)
    
    def test_energy_restoration(self):
        """Test energy restoration through sleep."""
        # Deplete energy first
        self.energy_system.current_energy = 20.0
        
        # Trigger sleep
        sleep_info = self.energy_system.trigger_sleep()
        
        self.assertEqual(sleep_info['energy_after'], 100.0)  # Should be restored to max
        self.assertGreater(sleep_info['energy_restored'], 0)
        self.assertEqual(self.energy_system.current_energy, 100.0)
    
    def test_death_and_respawn(self):
        """Test death and respawn mechanics."""
        # Set energy to death threshold
        self.energy_system.current_energy = 0.0
        
        # Handle death
        death_info = self.energy_system.handle_death()
        
        self.assertEqual(death_info['death_number'], 1)
        self.assertEqual(self.energy_system.total_deaths, 1)
        self.assertEqual(self.energy_system.current_energy, 100.0)  # Should respawn with full energy
    
    def test_sleep_triggers(self):
        """Test various sleep trigger conditions."""
        # Test low energy trigger
        self.energy_system.current_energy = 25.0
        should_sleep, reason = self.energy_system.should_sleep()
        self.assertTrue(should_sleep)
        self.assertIn("low_energy", reason)
        
        # Test emergency sleep trigger
        self.energy_system.current_energy = 5.0
        should_sleep, reason = self.energy_system.should_sleep()
        self.assertTrue(should_sleep)
        self.assertIn("emergency_low_energy", reason)
        
        # Test no trigger
        self.energy_system.current_energy = 80.0
        should_sleep, reason = self.energy_system.should_sleep()
        self.assertFalse(should_sleep)
        self.assertEqual(reason, "no_trigger")
    
    def test_performance_tracking(self):
        """Test performance tracking and adaptive parameters."""
        # Perform several actions with varying success
        for i in range(20):
            success = i % 3 == 0  # 1/3 success rate
            self.energy_system.consume_energy_for_action(
                action_id=1, success=success, learning_progress=0.1 if success else 0.0
            )
        
        # Check that performance tracking is working
        self.assertEqual(len(self.energy_system.recent_actions), 20)
        self.assertEqual(len(self.energy_system.recent_successes), 20)
        self.assertGreater(self.energy_system.consecutive_failures, 0)
    
    def test_adaptive_parameters(self):
        """Test adaptive parameter adjustment based on performance."""
        # Simulate poor performance
        for i in range(50):
            self.energy_system.consume_energy_for_action(
                action_id=1, success=False, learning_progress=0.0
            )
        
        # Should have increased action cost multiplier due to poor performance
        self.assertGreater(self.energy_system.current_action_cost_multiplier, 1.0)
        
        # Reset and simulate good performance
        self.energy_system.reset()
        for i in range(50):
            self.energy_system.consume_energy_for_action(
                action_id=1, success=True, learning_progress=0.2
            )
        
        # Should have decreased action cost multiplier due to good performance
        self.assertLess(self.energy_system.current_action_cost_multiplier, 1.0)
    
    def test_energy_metrics(self):
        """Test energy metrics calculation."""
        # Perform some actions
        for i in range(10):
            self.energy_system.consume_energy_for_action(
                action_id=1, success=True, learning_progress=0.1
            )
        
        # Get status
        status = self.energy_system.get_status()
        
        self.assertIn('current_energy', status)
        self.assertIn('energy_ratio', status)
        self.assertIn('energy_state', status)
        self.assertIn('total_actions_taken', status)
        self.assertIn('recent_success_rate', status)
        
        # Get detailed metrics
        metrics = self.energy_system.get_energy_metrics()
        
        self.assertIn('energy_history_length', metrics)
        self.assertIn('avg_energy_level', metrics)
        self.assertIn('action_costs', metrics)
        self.assertIn('energy_efficiency_ratio', metrics)
    
    def test_energy_ratio(self):
        """Test energy ratio calculation."""
        self.energy_system.current_energy = 75.0
        self.assertEqual(self.energy_system.get_energy_ratio(), 0.75)
        
        self.energy_system.current_energy = 0.0
        self.assertEqual(self.energy_system.get_energy_ratio(), 0.0)
        
        self.energy_system.current_energy = 100.0
        self.assertEqual(self.energy_system.get_energy_ratio(), 1.0)
    
    def test_time_based_consumption(self):
        """Test energy consumption over time."""
        initial_energy = self.energy_system.current_energy
        
        # Consume energy over 1 second
        energy_consumed = self.energy_system.consume_energy_over_time(1.0)
        
        self.assertGreater(energy_consumed, 0)
        self.assertLess(self.energy_system.current_energy, initial_energy)
    
    def test_energy_addition(self):
        """Test adding energy to the system."""
        # Deplete some energy first
        self.energy_system.current_energy = 50.0
        
        # Add energy
        new_energy = self.energy_system.add_energy(25.0, "test_source")
        
        self.assertEqual(new_energy, 75.0)
        self.assertEqual(self.energy_system.current_energy, 75.0)
        
        # Test energy cap
        new_energy = self.energy_system.add_energy(50.0, "test_source")
        self.assertEqual(new_energy, 100.0)  # Should be capped at max
        self.assertEqual(self.energy_system.current_energy, 100.0)
    
    def test_reset_functionality(self):
        """Test system reset functionality."""
        # Modify some state
        self.energy_system.current_energy = 50.0
        self.energy_system.total_actions_taken = 100
        self.energy_system.total_deaths = 5
        
        # Reset
        self.energy_system.reset()
        
        # Check that everything is reset
        self.assertEqual(self.energy_system.current_energy, 100.0)
        self.assertEqual(self.energy_system.total_actions_taken, 0)
        self.assertEqual(self.energy_system.total_deaths, 0)
        self.assertEqual(len(self.energy_system.recent_actions), 0)
        self.assertEqual(self.energy_system.consecutive_failures, 0)


class TestEnergySystemIntegration(unittest.TestCase):
    """Test energy system integration with training loops."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.energy_system = UnifiedEnergySystem()
        self.integration = EnergySystemIntegration(self.energy_system)
    
    def test_integration_initialization(self):
        """Test integration initialization."""
        self.assertFalse(self.integration.integration_active)
        self.assertEqual(self.integration.energy_system, self.energy_system)
    
    def test_training_loop_integration(self):
        """Test integration with a mock training loop."""
        mock_training_loop = Mock()
        mock_training_loop.consume_energy = Mock()
        
        # Integrate
        success = self.integration.integrate_with_training_loop(mock_training_loop)
        
        self.assertTrue(success)
        self.assertTrue(self.integration.integration_active)
        self.assertEqual(mock_training_loop.energy_system, self.energy_system)
        self.assertEqual(mock_training_loop.current_energy, self.energy_system.current_energy)
    
    def test_training_update(self):
        """Test energy system updates during training."""
        # Activate integration
        self.integration.integration_active = True
        
        # Update during training
        result = self.integration.update_during_training(
            action_id=1, success=True, learning_progress=0.2
        )
        
        self.assertIn('consumption_record', result)
        self.assertIn('should_sleep', result)
        self.assertIn('current_energy', result)
        self.assertIn('energy_state', result)
        
        # Check that energy was consumed
        self.assertLess(result['current_energy'], 100.0)
    
    def test_integration_without_activation(self):
        """Test that integration methods fail when not activated."""
        result = self.integration.update_during_training(
            action_id=1, success=True, learning_progress=0.2
        )
        
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'Integration not active')


class TestEnergyConfig(unittest.TestCase):
    """Test energy configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EnergyConfig()
        
        self.assertEqual(config.max_energy, 100.0)
        self.assertEqual(config.min_energy, 0.0)
        self.assertEqual(config.base_consumption_per_action, 1.0)
        self.assertEqual(config.sleep_trigger_threshold, 30.0)
        self.assertEqual(config.emergency_sleep_threshold, 10.0)
        self.assertIsNotNone(config.action_costs)
        self.assertIn(1, config.action_costs)
        self.assertIn(6, config.action_costs)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = EnergyConfig(
            max_energy=200.0,
            sleep_trigger_threshold=50.0,
            action_costs={1: 2.0, 2: 3.0}
        )
        
        self.assertEqual(config.max_energy, 200.0)
        self.assertEqual(config.sleep_trigger_threshold, 50.0)
        self.assertEqual(config.action_costs[1], 2.0)
        self.assertEqual(config.action_costs[2], 3.0)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
