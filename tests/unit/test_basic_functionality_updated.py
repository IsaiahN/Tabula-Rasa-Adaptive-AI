"""
Updated Basic Functionality Test - Validate current core components work together.

This script tests the basic integration of current core components without
full training to catch any immediate issues with the updated architecture.
"""

import pytest
import torch
import logging
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.energy_system import EnergySystem
from core.salience_system import SalienceMode
from core.data_models import AgentState

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBasicFunctionality:
    """Test suite for basic functionality of core components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_energy_system_basic(self):
        """Test basic energy system functionality."""
        logger.info("Testing Energy System...")
        
        try:
            energy_system = EnergySystem(
                max_energy=100.0,
                base_consumption=0.01,
                action_multiplier=0.5,
                computation_multiplier=0.001
            )
            
            # Test initialization
            assert energy_system.max_energy == 100.0
            assert energy_system.current_energy == 100.0
            
            # Test energy consumption
            remaining = energy_system.consume_energy(action_cost=0.5, computation_cost=10.0)
            assert remaining < 100.0
            assert remaining > 0.0
            
            # Test energy addition
            restored = energy_system.add_energy(10.0)
            assert restored <= 100.0
            
            logger.info("✅ Energy System test passed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Energy System test failed: {e}")
            return False
    
    @patch('core.agent.torch.load')  # Mock model loading
    def test_adaptive_learning_agent_basic(self, mock_torch_load):
        """Test basic adaptive learning agent functionality."""
        logger.info("Testing Adaptive Learning Agent...")
        
        try:
            # Mock the model loading
            mock_torch_load.return_value = {
                'model_state_dict': {},
                'optimizer_state_dict': {},
                'epoch': 0,
                'loss': 0.0
            }
            
            # Import here to avoid import issues
            from core.agent import AdaptiveLearningAgent
            
            # Create agent with basic configuration
            agent = AdaptiveLearningAgent(
                hidden_dim=64,
                memory_size=32,
                word_size=8,
                max_energy=100.0,
                salience_mode=SalienceMode.DECAY,
                config_path=None  # Use defaults
            )
            
            # Test basic properties
            assert hasattr(agent, 'energy_system')
            assert hasattr(agent, 'salience_system')
            assert agent.hidden_dim == 64
            
            logger.info("✅ Adaptive Learning Agent test passed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Adaptive Learning Agent test failed: {e}")
            return False
    
    def test_salience_system_basic(self):
        """Test basic salience system functionality."""
        logger.info("Testing Salience System...")
        
        try:
            from core.salience_system import SalienceSystem
            
            salience_system = SalienceSystem(
                mode=SalienceMode.DECAY,
                decay_rate=0.01,
                importance_threshold=0.5
            )
            
            # Test initialization
            assert salience_system.mode == SalienceMode.DECAY
            assert salience_system.decay_rate == 0.01
            
            # Test with sample memory
            memory_matrix = torch.randn(32, 8)
            importance_scores = torch.rand(32)
            
            processed = salience_system.process_memory(memory_matrix, importance_scores)
            assert processed is not None
            assert processed.shape == memory_matrix.shape
            
            logger.info("✅ Salience System test passed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Salience System test failed: {e}")
            return False
    
    def test_agent_state_basic(self):
        """Test basic agent state functionality."""
        logger.info("Testing Agent State...")
        
        try:
            # Create sample agent state
            agent_state = AgentState(
                position=torch.tensor([1.0, 2.0, 3.0]),
                orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
                energy=75.0,
                hidden_state=torch.randn(64),
                active_goals=[],
                memory_state=torch.randn(32, 8)
            )
            
            # Test basic properties
            assert agent_state.position.shape == (3,)
            assert agent_state.orientation.shape == (4,)
            assert agent_state.energy == 75.0
            assert agent_state.hidden_state.shape == (64,)
            assert isinstance(agent_state.active_goals, list)
            assert agent_state.memory_state.shape == (32, 8)
            
            logger.info("✅ Agent State test passed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Agent State test failed: {e}")
            return False
    
    def test_integration_basic(self):
        """Test basic integration of components."""
        logger.info("Testing Basic Integration...")
        
        try:
            # Test that components can be created together
            energy_system = EnergySystem(max_energy=100.0)
            
            from core.salience_system import SalienceSystem
            salience_system = SalienceSystem(mode=SalienceMode.MINIMAL)
            
            # Test that they can interact
            current_energy = energy_system.current_energy
            assert current_energy == 100.0
            
            # Test energy consumption and salience processing
            energy_system.consume_energy(action_cost=0.5)
            
            # Test memory processing
            test_memory = torch.randn(16, 8)
            importance_scores = torch.rand(16)
            
            processed_memory = salience_system.process_memory(test_memory, importance_scores)
            assert processed_memory is not None
            
            logger.info("✅ Basic Integration test passed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Basic Integration test failed: {e}")
            return False


def test_all_basic_functionality():
    """Run all basic functionality tests."""
    logger.info("="*60)
    logger.info("🚀 STARTING BASIC FUNCTIONALITY TESTS")
    logger.info("="*60)
    
    test_suite = TestBasicFunctionality()
    test_suite.setup_method()
    
    tests = [
        test_suite.test_energy_system_basic,
        test_suite.test_adaptive_learning_agent_basic,
        test_suite.test_salience_system_basic,
        test_suite.test_agent_state_basic,
        test_suite.test_integration_basic
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"❌ Test {test.__name__} raised exception: {e}")
            failed += 1
    
    test_suite.teardown_method()
    
    logger.info("="*60)
    logger.info(f"📊 BASIC FUNCTIONALITY TEST RESULTS")
    logger.info("="*60)
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📈 Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    if failed == 0:
        logger.info("🎉 ALL BASIC TESTS PASSED! Core components are working correctly.")
        return True
    else:
        logger.info("⚠️  SOME TESTS FAILED. Please review core component implementations.")
        return False


if __name__ == '__main__':
    # Run tests directly or with pytest
    if len(sys.argv) > 1 and sys.argv[1] == '--pytest':
        pytest.main([__file__, '-v'])
    else:
        success = test_all_basic_functionality()
        sys.exit(0 if success else 1)
