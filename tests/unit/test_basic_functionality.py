"""
Basic Functionality Test - Validate core components work together.

This script tests the basic integration of all core components without
full training to catch any immediate issues.
"""

import torch
import logging
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.agent import AdaptiveLearningAgent
from environment.survival_environment import SurvivalEnvironment
from core.data_models import SensoryInput

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_predictive_core():
    """Test predictive core functionality."""
    logger.info("Testing Predictive Core...")
    
    try:
        from core.predictive_core import PredictiveCore
        
        # Create predictive core
        core = PredictiveCore(
            visual_size=(3, 64, 64),
            proprioception_size=12,
            hidden_size=256,  # Smaller for testing
            architecture="lstm"
        )
        
        # Create dummy sensory input
        batch_size = 2
        visual = torch.randn(batch_size, 3, 64, 64)
        proprio = torch.randn(batch_size, 12)
        energy = 75.0
        
        sensory_input = SensoryInput(
            visual=visual,
            proprioception=proprio,
            energy_level=energy,
            timestamp=0
        )
        
        # Test forward pass
        visual_pred, proprio_pred, energy_pred, hidden_state, debug_info = core(sensory_input)
        
        # Check outputs
        assert visual_pred.shape == (batch_size, 3, 64, 64), f"Visual pred shape: {visual_pred.shape}"
        assert proprio_pred.shape == (batch_size, 12), f"Proprio pred shape: {proprio_pred.shape}"
        assert energy_pred.shape == (batch_size, 1), f"Energy pred shape: {energy_pred.shape}"
        
        # Test prediction error computation
        errors = core.compute_prediction_error((visual_pred, proprio_pred, energy_pred), sensory_input)
        assert 'total' in errors, "Missing total error"
        
        logger.info("✓ Predictive Core test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Predictive Core test failed: {e}")
        return False


def test_learning_progress():
    """Test learning progress drive functionality."""
    logger.info("Testing Learning Progress Drive...")
    
    try:
        from core.learning_progress import LearningProgressDrive
        
        # Create LP drive
        lp_drive = LearningProgressDrive(
            smoothing_window=100,
            derivative_clamp=(-1.0, 1.0),
            boredom_threshold=0.01,
            boredom_steps=100
        )
        
        # Test with decreasing error (should give positive LP)
        errors = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        
        for error in errors:
            lp_signal = lp_drive.compute_learning_progress(error)
            
        # Should have positive LP after decreasing errors
        final_lp = lp_drive.compute_learning_progress(0.05)
        assert final_lp > 0, f"Expected positive LP, got {final_lp}"
        
        # Test boredom detection
        assert not lp_drive.is_bored(), "Should not be bored with decreasing errors"
        
        logger.info("✓ Learning Progress Drive test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Learning Progress Drive test failed: {e}")
        return False


def test_energy_system():
    """Test energy system functionality."""
    logger.info("Testing Energy System...")
    
    try:
        from core.energy_system import EnergySystem
        
        # Create energy system
        energy_sys = EnergySystem(
            max_energy=100.0,
            base_consumption=0.01,
            action_multiplier=0.5
        )
        
        # Test energy consumption
        initial_energy = energy_sys.get_energy_level()
        remaining = energy_sys.consume_energy(action_cost=0.1, computation_cost=1.0)
        
        assert remaining < initial_energy, "Energy should decrease"
        assert not energy_sys.is_dead(), "Should not be dead yet"
        
        # Test energy addition
        energy_sys.add_energy(20.0)
        new_energy = energy_sys.get_energy_level()
        assert new_energy > remaining, "Energy should increase"
        
        # Test death
        while not energy_sys.is_dead():
            energy_sys.consume_energy(action_cost=10.0, computation_cost=10.0)
            
        assert energy_sys.is_dead(), "Should be dead after consuming all energy"
        
        logger.info("✓ Energy System test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Energy System test failed: {e}")
        return False


def test_goal_system():
    """Test goal system functionality."""
    logger.info("Testing Goal System...")
    
    try:
        from goals.goal_system import GoalInventionSystem, GoalPhase
        from core.data_models import AgentState
        
        # Create goal system
        goal_system = GoalInventionSystem(phase=GoalPhase.SURVIVAL)
        
        # Create dummy agent state
        agent_state = AgentState(
            position=torch.tensor([0.0, 0.0, 1.0]),
            orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
            energy=50.0,
            hidden_state=None,
            active_goals=[],
            memory_state=None,
            timestamp=0
        )
        
        # Test goal generation
        active_goals = goal_system.get_active_goals(agent_state)
        assert len(active_goals) > 0, "Should generate survival goals"
        
        # Check goal types
        goal_types = [goal.goal_type for goal in active_goals]
        assert 'survival' in goal_types, "Should include survival goals"
        
        logger.info("✓ Goal System test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Goal System test failed: {e}")
        return False


def test_environment():
    """Test environment functionality."""
    logger.info("Testing Environment...")
    
    try:
        # Create environment
        env = SurvivalEnvironment(
            world_size=(10, 10, 3),
            num_food_sources=3,
            complexity_level=1
        )
        
        # Test environment state
        env_state = env.get_environment_state()
        assert env_state['active_food_sources'] == 3, "Should have 3 active food sources"
        assert env_state['complexity_level'] == 1, "Should start at complexity 1"
        
        # Test complexity increase
        env.increase_complexity()
        new_state = env.get_environment_state()
        assert new_state['complexity_level'] == 2, "Complexity should increase"
        
        logger.info("✓ Environment test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Environment test failed: {e}")
        return False


def test_agent_integration():
    """Test basic agent integration."""
    logger.info("Testing Agent Integration...")
    
    try:
        # Create minimal config
        config = {
            'predictive_core': {
                'visual_size': [3, 32, 32],  # Smaller for testing
                'proprioception_size': 12,
                'hidden_size': 256,
                'architecture': 'lstm'
            },
            'memory': {
                'enabled': False  # Disable for simple test
            },
            'learning_progress': {
                'smoothing_window': 100,
                'derivative_clamp': [-1.0, 1.0],
                'boredom_threshold': 0.01,
                'boredom_steps': 100,
                'lp_weight': 0.7,
                'empowerment_weight': 0.3,
                'use_adaptive_weights': False
            },
            'energy': {
                'max_energy': 100.0,
                'base_consumption': 0.01,
                'action_multiplier': 0.5,
                'computation_multiplier': 0.001,
                'food_energy_value': 10.0,
                'memory_size': 256,
                'word_size': 64,
                'use_learned_importance': False,
                'preservation_ratio': 0.2
            },
            'goals': {
                'initial_phase': 'survival',
                'environment_bounds': [-5, 5, -5, 5]
            },
            'sleep': {
                'sleep_trigger_energy': 20.0,
                'sleep_trigger_boredom_steps': 1000,
                'sleep_trigger_memory_pressure': 0.9,
                'sleep_duration_steps': 100,
                'replay_batch_size': 32,
                'learning_rate': 0.001
            },
            'environment': {
                'world_size': [10, 10, 3],
                'num_food_sources': 3,
                'food_respawn_time': 30.0,
                'food_energy_value': 10.0,
                'complexity_level': 1,
                'physics_enabled': True
            },
            'monitoring': {
                'log_interval': 100,
                'save_interval': 1000,
                'log_dir': './logs'
            }
        }
        
        # Create agent
        agent = AdaptiveLearningAgent(config, device='cpu')
        
        # Test agent state
        agent_state = agent.get_agent_state()
        assert agent_state.energy == 100.0, "Agent should start with full energy"
        assert len(agent_state.active_goals) == 0, "Should start with no active goals"
        
        # Test performance metrics
        metrics = agent.get_performance_metrics()
        assert 'total_steps' in metrics, "Should have total_steps metric"
        assert 'episodes_completed' in metrics, "Should have episodes_completed metric"
        
        logger.info("✓ Agent Integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Agent Integration test failed: {e}")
        return False


def run_all_tests():
    """Run all basic functionality tests."""
    logger.info("Running Basic Functionality Tests...")
    logger.info("=" * 50)
    
    tests = [
        test_predictive_core,
        test_learning_progress,
        test_energy_system,
        test_goal_system,
        test_environment,
        test_agent_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")
    
    logger.info("=" * 50)
    logger.info(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Basic functionality is working.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Check the logs above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 