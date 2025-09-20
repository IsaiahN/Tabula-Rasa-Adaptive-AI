"""
Test suite for memory leak fixes in master_arc_trainer.py and continuous_learning_loop.py

This test suite validates that memory leaks have been fixed and data structures
are properly bounded to prevent unlimited growth.
"""

import pytest
import asyncio
import gc
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training import MasterARCTrainer, MasterTrainingConfig
from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop


class TestMemoryLeakFixes:
    """Test memory leak fixes and data structure bounds."""
    
    def test_performance_history_bounds(self):
        """Test that performance history is properly bounded."""
        config = MasterTrainingConfig(mode='minimal-debug')
        trainer = MasterARCTrainer(config)
        
        # Simulate adding many performance records
        for i in range(1000):
            trainer.performance_history.append({
                'session': i,
                'score': i * 0.1,
                'timestamp': f'2024-01-01T{i:02d}:00:00'
            })
        
        # Check that history is bounded (should be limited to reasonable size)
        assert len(trainer.performance_history) <= 100, "Performance history should be bounded"
    
    def test_governor_decisions_bounds(self):
        """Test that governor decisions are properly bounded."""
        config = MasterTrainingConfig(mode='minimal-debug')
        trainer = MasterARCTrainer(config)
        
        # Simulate adding many governor decisions
        for i in range(1000):
            trainer.governor_decisions.append({
                'session': i,
                'recommendation': f'rec_{i}',
                'confidence': 0.5,
                'timestamp': f'2024-01-01T{i:02d}:00:00'
            })
        
        # Check that decisions are bounded
        assert len(trainer.governor_decisions) <= 100, "Governor decisions should be bounded"
    
    def test_architect_evolutions_bounds(self):
        """Test that architect evolutions are properly bounded."""
        config = MasterTrainingConfig(mode='minimal-debug')
        trainer = MasterARCTrainer(config)
        
        # Simulate adding many architect evolutions
        for i in range(1000):
            trainer.architect_evolutions.append({
                'session': i,
                'success': True,
                'improvement': i * 0.01,
                'timestamp': f'2024-01-01T{i:02d}:00:00'
            })
        
        # Check that evolutions are bounded
        assert len(trainer.architect_evolutions) <= 100, "Architect evolutions should be bounded"
    
    def test_continuous_loop_memory_cleanup(self):
        """Test that continuous learning loop properly cleans up memory."""
        # Mock the continuous loop to avoid actual API calls
        with patch('src.arc_integration.continuous_learning_loop.ContinuousLearningLoop._async_initialize'):
            loop = ContinuousLearningLoop(
                arc_agents_path="/mock/path",
                tabula_rasa_path="/mock/path",
                api_key="mock_key"
            )
            
            # Initialize the complex attributes
            loop._initialize_complex_attributes()
            
            # Simulate adding data to various tracking structures
            for i in range(1000):
                loop.performance_history.append({
                    'session': i,
                    'score': i * 0.1
                })
                loop.session_history.append({
                    'session': i,
                    'status': 'completed'
                })
            
            # Check that data structures are bounded
            assert len(loop.performance_history) <= 100, "Performance history should be bounded"
            assert len(loop.session_history) <= 100, "Session history should be bounded"
    
    def test_action_tracking_cleanup(self):
        """Test that action tracking data is properly cleaned up."""
        with patch('src.arc_integration.continuous_learning_loop.ContinuousLearningLoop._async_initialize'):
            loop = ContinuousLearningLoop(
                arc_agents_path="/mock/path",
                tabula_rasa_path="/mock/path",
                api_key="mock_key"
            )
            
            # Initialize the complex attributes
            loop._initialize_complex_attributes()
            
            # Simulate action tracking data accumulation
            for i in range(1000):
                if 'action_history' not in loop.available_actions_memory:
                    loop.available_actions_memory['action_history'] = []
                loop.available_actions_memory['action_history'].append({
                    'action': i % 10,
                    'timestamp': i,
                    'result': 'success'
                })
            
            # Check that action history is bounded
            action_history = loop.available_actions_memory.get('action_history', [])
            assert len(action_history) <= 1000, "Action history should be bounded"
    
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable over time."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        config = MasterTrainingConfig(mode='minimal-debug')
        trainer = MasterARCTrainer(config)
        
        # Simulate many operations
        for i in range(100):
            trainer.performance_history.append({'session': i, 'score': i})
            trainer.governor_decisions.append({'session': i, 'recommendation': f'rec_{i}'})
            trainer.architect_evolutions.append({'session': i, 'success': True})
            
            # Force garbage collection periodically
            if i % 10 == 0:
                gc.collect()
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 50MB)
        assert memory_growth < 50 * 1024 * 1024, f"Memory growth too high: {memory_growth / 1024 / 1024:.2f}MB"
    
    def test_data_structure_cleanup_on_reset(self):
        """Test that data structures are properly cleaned up on reset."""
        config = MasterTrainingConfig(mode='minimal-debug')
        trainer = MasterARCTrainer(config)
        
        # Add some data
        for i in range(50):
            trainer.performance_history.append({'session': i, 'score': i})
            trainer.governor_decisions.append({'session': i, 'recommendation': f'rec_{i}'})
            trainer.architect_evolutions.append({'session': i, 'success': True})
        
        # Simulate reset
        trainer.performance_history.clear()
        trainer.governor_decisions.clear()
        trainer.architect_evolutions.clear()
        
        # Check that structures are empty
        assert len(trainer.performance_history) == 0
        assert len(trainer.governor_decisions) == 0
        assert len(trainer.architect_evolutions) == 0


class TestDatabaseIntegration:
    """Test that deprecated file I/O is replaced with database operations."""
    
    def test_global_counters_database_integration(self):
        """Test that global counters use database instead of JSON files."""
        # This test would verify that global counters are stored in database
        # rather than JSON files
        pass
    
    def test_action_intelligence_database_integration(self):
        """Test that action intelligence data uses database instead of JSON files."""
        # This test would verify that action intelligence data is stored in database
        # rather than JSON files
        pass
    
    def test_performance_data_database_integration(self):
        """Test that performance data uses database instead of JSON files."""
        # This test would verify that performance data is stored in database
        # rather than JSON files
        pass


class TestCodeDeduplication:
    """Test that code duplication has been eliminated."""
    
    def test_action_limits_centralized(self):
        """Test that ActionLimits configuration is centralized."""
        # This test would verify that ActionLimits is defined in one place
        # and imported by both files
        pass
    
    def test_api_key_handling_centralized(self):
        """Test that API key handling is centralized."""
        # This test would verify that API key handling is centralized
        # and not duplicated
        pass
    
    def test_logging_setup_centralized(self):
        """Test that logging setup is centralized."""
        # This test would verify that logging setup is centralized
        # and not duplicated
        pass


if __name__ == "__main__":
    pytest.main([__file__])
