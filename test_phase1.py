"""
Phase 1 Test Script - Test the integrated system from root directory.

This script tests the basic functionality of the Phase 1 implementation.
"""

import sys
import os
import torch
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing module imports...")
    
    try:
        from core.data_models import SensoryInput, AgentState
        logger.info("✓ Data models imported successfully")
        
        from core.predictive_core import PredictiveCore
        logger.info("✓ Predictive core imported successfully")
        
        from core.learning_progress import LearningProgressDrive
        logger.info("✓ Learning progress drive imported successfully")
        
        from core.energy_system import EnergySystem
        logger.info("✓ Energy system imported successfully")
        
        from goals.goal_system import GoalInventionSystem, GoalPhase
        logger.info("✓ Goal system imported successfully")
        
        from memory.dnc import DNCMemory
        logger.info("✓ DNC memory imported successfully")
        
        logger.info("✓ All core modules imported successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_predictive_core():
    """Test predictive core functionality."""
    logger.info("Testing Predictive Core...")
    
    try:
        from core.predictive_core import PredictiveCore
        from core.data_models import SensoryInput
        
        # Create predictive core
        core = PredictiveCore(
            visual_size=(3, 32, 32),  # Smaller for testing
            proprioception_size=12,
            hidden_size=256,
            architecture="lstm"
        )
        
        # Create dummy sensory input
        batch_size = 2
        visual = torch.randn(batch_size, 3, 32, 32)
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
        assert visual_pred.shape == (batch_size, 3, 32, 32), f"Visual pred shape: {visual_pred.shape}"
        assert proprio_pred.shape == (batch_size, 12), f"Proprio pred shape: {proprio_pred.shape}"
        assert energy_pred.shape == (batch_size, 1), f"Energy pred shape: {energy_pred.shape}"
        
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
            
        # Need more history for derivative calculation
        for _ in range(20):  # Add more history
            lp_drive.compute_learning_progress(0.05)
            
        # Should have some LP signal after decreasing errors
        final_lp = lp_drive.compute_learning_progress(0.05)
        assert abs(final_lp) > 0.0001, f"Expected non-zero LP, got {final_lp}"
        
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
        from environment.survival_environment import SurvivalEnvironment
        
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


def test_memory_system():
    """Test DNC memory system functionality."""
    logger.info("Testing Memory System...")
    
    try:
        from memory.dnc import DNCMemory
        
        # Create memory system
        memory = DNCMemory(
            memory_size=256,
            word_size=32,
            num_read_heads=2,
            num_write_heads=1,
            controller_size=128
        )
        
        # Test memory forward pass
        batch_size = 2
        input_data = torch.randn(batch_size, 256)
        prev_reads = torch.randn(batch_size, 2 * 32)  # num_read_heads * word_size
        
        memory_reads, controller_output, controller_state, debug_info = memory(
            input_data, prev_reads
        )
        
        # Check outputs
        assert memory_reads.shape[0] == batch_size, "Batch size should match"
        assert 'memory_utilization' in debug_info, "Should have memory metrics"
        
        logger.info("✓ Memory System test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Memory System test failed: {e}")
        return False


def run_all_tests():
    """Run all basic functionality tests."""
    logger.info("Running Phase 1 Basic Functionality Tests...")
    logger.info("=" * 60)
    
    tests = [
        test_imports,
        test_predictive_core,
        test_learning_progress,
        test_energy_system,
        test_goal_system,
        test_environment,
        test_memory_system
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")
    
    logger.info("=" * 60)
    logger.info(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Phase 1 system is ready for training.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Check the logs above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 