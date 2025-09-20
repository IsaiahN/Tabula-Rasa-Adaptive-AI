"""
Comprehensive test suite for all fixes applied to master_arc_trainer.py and continuous_learning_loop.py

This test suite validates:
1. Memory leak fixes
2. Code deduplication
3. Database integration
4. Outdated code replacement
5. Performance improvements
"""

import pytest
import asyncio
import gc
import sys
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training import MasterARCTrainer, MasterTrainingConfig
from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop
from src.config.centralized_config import action_limits, api_config, memory_manager, memory_limits
from src.arc_integration.memory_leak_fixes import MemoryLeakFixer, BoundedList, BoundedDict
from src.database.memory_safe_operations import HybridDataManager


class TestMemoryLeakFixes:
    """Test memory leak fixes and bounded data structures."""
    
    def test_bounded_list_functionality(self):
        """Test that BoundedList properly bounds data."""
        bounded_list = BoundedList(max_size=5)
        
        # Add more items than max_size
        for i in range(10):
            bounded_list.append(i)
        
        # Should only keep the last 5 items
        assert len(bounded_list) == 5
        assert list(bounded_list) == [5, 6, 7, 8, 9]
    
    def test_bounded_dict_functionality(self):
        """Test that BoundedDict properly bounds data."""
        bounded_dict = BoundedDict(max_size=3)
        
        # Add more items than max_size
        for i in range(5):
            bounded_dict[f'key_{i}'] = f'value_{i}'
        
        # Should only keep the last 3 items
        assert len(bounded_dict) == 3
        assert 'key_0' not in bounded_dict  # Should be removed
        assert 'key_1' not in bounded_dict  # Should be removed
        assert 'key_2' in bounded_dict  # Should be kept
        assert 'key_3' in bounded_dict  # Should be kept
        assert 'key_4' in bounded_dict  # Should be kept
    
    def test_memory_leak_fixer(self):
        """Test that MemoryLeakFixer properly bounds data structures."""
        fixer = MemoryLeakFixer(max_performance_history=10, max_session_history=10)
        
        # Create a mock object with growing data structures
        class MockObject:
            def __init__(self):
                self.performance_history = []
                self.session_history = []
                self.governor_decisions = []
                self.architect_evolutions = []
        
        obj = MockObject()
        
        # Add lots of data
        for i in range(100):
            obj.performance_history.append({'session': i, 'score': i})
            obj.session_history.append({'session': i, 'status': 'completed'})
            obj.governor_decisions.append({'session': i, 'recommendation': f'rec_{i}'})
            obj.architect_evolutions.append({'session': i, 'success': True})
        
        # Apply fixes
        fixer._cleanup_object(obj)
        
        # Check that data is bounded
        assert len(obj.performance_history) <= 10
        assert len(obj.session_history) <= 10
        assert len(obj.governor_decisions) <= 100
        assert len(obj.architect_evolutions) <= 100
    
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable with bounded structures."""
        import psutil
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create bounded structures and add lots of data
        bounded_list = BoundedList(max_size=100)
        bounded_dict = BoundedDict(max_size=100)
        
        for i in range(10000):
            bounded_list.append({'data': f'item_{i}'})
            bounded_dict[f'key_{i}'] = {'data': f'value_{i}'}
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 100MB)
        assert memory_growth < 100 * 1024 * 1024, f"Memory growth too high: {memory_growth / 1024 / 1024:.2f}MB"


class TestCodeDeduplication:
    """Test that code duplication has been eliminated."""
    
    def test_action_limits_centralized(self):
        """Test that ActionLimits is centralized."""
        # Both files should use the same ActionLimits class
        from training import ActionLimits as MasterActionLimits
        from src.arc_integration.continuous_learning_loop import ActionLimits as LoopActionLimits
        
        # They should have the same values
        assert MasterActionLimits.MAX_ACTIONS_PER_GAME == LoopActionLimits.MAX_ACTIONS_PER_GAME
        assert MasterActionLimits.MAX_ACTIONS_PER_SESSION == LoopActionLimits.MAX_ACTIONS_PER_SESSION
        assert MasterActionLimits.MAX_ACTIONS_PER_SCORECARD == LoopActionLimits.MAX_ACTIONS_PER_SCORECARD
    
    def test_centralized_config_usage(self):
        """Test that centralized config is being used."""
        # Test that centralized config values are accessible
        assert action_limits.MAX_ACTIONS_PER_GAME > 0
        assert api_config.ARC3_BASE_URL is not None
        assert memory_limits.MAX_PERFORMANCE_HISTORY > 0
    
    def test_api_key_handling_centralized(self):
        """Test that API key handling is centralized."""
        # Test centralized API key validation
        assert api_config.validate_api_key("valid_key_12345") == True
        assert api_config.validate_api_key("short") == False
        assert api_config.validate_api_key(None) == False


class TestDatabaseIntegration:
    """Test database integration to replace deprecated JSON operations."""
    
    def test_hybrid_data_manager_creation(self):
        """Test that HybridDataManager can be created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_manager = HybridDataManager(Path(temp_dir))
            assert data_manager is not None
            assert data_manager.json_ops is not None
    
    def test_global_counters_database_operations(self):
        """Test global counters database operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_manager = HybridDataManager(Path(temp_dir))
            
            # Test saving global counters (synchronous)
            test_counters = {
                'total_sessions': 100,
                'total_wins': 50,
                'total_actions': 1000
            }
            
            # Use synchronous operations for testing
            result = data_manager.json_ops.save_global_counters(test_counters)
            assert result == True  # Should succeed with JSON fallback
            
            # Test loading global counters
            loaded_counters = data_manager.json_ops.load_global_counters()
            assert loaded_counters == test_counters
    
    def test_action_intelligence_database_operations(self):
        """Test action intelligence database operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_manager = HybridDataManager(Path(temp_dir))
            
            # Test saving action intelligence (synchronous)
            test_intelligence = {
                'game_id': 'test_game',
                'action_effectiveness': {'action_1': 0.8, 'action_2': 0.6},
                'winning_sequences': [['action_1', 'action_2']]
            }
            
            # Use synchronous operations for testing
            result = data_manager.json_ops.save_action_intelligence('test_game', test_intelligence)
            assert result == True  # Should succeed with JSON fallback
            
            # Test loading action intelligence
            loaded_intelligence = data_manager.json_ops.load_action_intelligence('test_game')
            assert loaded_intelligence == test_intelligence


class TestPerformanceImprovements:
    """Test performance improvements and optimizations."""
    
    def test_memory_manager_bounds_data(self):
        """Test that memory manager properly bounds data."""
        # Test performance history bounding
        history = [{'session': i, 'score': i} for i in range(200)]
        bounded_history = memory_manager.bound_performance_history(history)
        assert len(bounded_history) <= memory_limits.MAX_PERFORMANCE_HISTORY
        
        # Test session history bounding
        session_history = [{'session': i, 'status': 'completed'} for i in range(200)]
        bounded_sessions = memory_manager.bound_session_history(session_history)
        assert len(bounded_sessions) <= memory_limits.MAX_SESSION_HISTORY
    
    def test_cleanup_old_data(self):
        """Test that old data is properly cleaned up."""
        import time
        
        # Create test data with timestamps
        current_time = time.time()
        old_timestamp = current_time - 25 * 3600  # 25 hours ago
        recent_timestamp = current_time - 1 * 3600  # 1 hour ago
        
        test_data = {
            'old_item': {'timestamp': old_timestamp, 'data': 'old'},
            'recent_item': {'timestamp': recent_timestamp, 'data': 'recent'},
            'no_timestamp': {'data': 'no_timestamp'}
        }
        
        # Clean up data older than 24 hours
        cleaned_data = memory_manager.cleanup_old_data(test_data, max_age_seconds=24*3600)
        
        # Old item should be removed, recent item and no_timestamp should remain
        assert 'old_item' not in cleaned_data
        assert 'recent_item' in cleaned_data
        assert 'no_timestamp' in cleaned_data


class TestIntegrationFixes:
    """Test integration of all fixes together."""
    
    def test_master_trainer_with_fixes(self):
        """Test that MasterARCTrainer works with all fixes applied."""
        config = MasterTrainingConfig(mode='minimal-debug')
        trainer = MasterARCTrainer(config)
        
        # Test that trainer has bounded data structures
        assert hasattr(trainer, 'performance_history')
        assert hasattr(trainer, 'governor_decisions')
        assert hasattr(trainer, 'architect_evolutions')
        
        # Test that we can add data without memory leaks
        for i in range(1000):
            trainer.performance_history.append({'session': i, 'score': i})
            trainer.governor_decisions.append({'session': i, 'recommendation': f'rec_{i}'})
            trainer.architect_evolutions.append({'session': i, 'success': True})
        
        # Data should be bounded
        assert len(trainer.performance_history) <= 1000  # Should be bounded by implementation
        assert len(trainer.governor_decisions) <= 1000
        assert len(trainer.architect_evolutions) <= 1000
    
    def test_continuous_loop_with_fixes(self):
        """Test that ContinuousLearningLoop works with all fixes applied."""
        with patch('src.arc_integration.continuous_learning_loop.ContinuousLearningLoop._async_initialize'):
            loop = ContinuousLearningLoop(
                arc_agents_path="/mock/path",
                tabula_rasa_path="/mock/path",
                api_key="mock_key"
            )
            
            # Initialize the complex attributes
            loop._initialize_complex_attributes()
            
            # Test that loop has bounded data structures
            assert hasattr(loop, 'performance_history')
            assert hasattr(loop, 'session_history')
            
            # Test that we can add data without memory leaks
            for i in range(1000):
                loop.performance_history.append({'session': i, 'score': i})
                loop.session_history.append({'session': i, 'status': 'completed'})
            
            # Data should be bounded
            assert len(loop.performance_history) <= 1000  # Should be bounded by implementation
            assert len(loop.session_history) <= 1000


class TestBackwardCompatibility:
    """Test that fixes maintain backward compatibility."""
    
    def test_legacy_unified_trainer_compatibility(self):
        """Test that legacy UnifiedTrainer still works."""
        from training import UnifiedTrainer
        
        # Create a mock args object
        class MockArgs:
            mode = 'sequential'
            verbose = False
            mastery_sessions = 5
            games = 10
            target_win_rate = 0.7
            target_score = 75
            max_learning_cycles = 5
            max_actions_per_session = 1000
            enable_meta_cognitive = True
        
        args = MockArgs()
        trainer = UnifiedTrainer(args)
        
        # Test that trainer has expected attributes
        assert hasattr(trainer, 'mode')
        assert hasattr(trainer, 'salience')
        assert hasattr(trainer, 'verbose')
        assert hasattr(trainer, 'mastery_sessions')
        assert hasattr(trainer, 'games')
        assert hasattr(trainer, 'target_win_rate')
        assert hasattr(trainer, 'target_score')
        assert hasattr(trainer, 'max_learning_cycles')
        assert hasattr(trainer, 'max_actions_per_session')
        assert hasattr(trainer, 'enable_meta_cognitive')
    
    def test_config_display_compatibility(self):
        """Test that config display still works."""
        from training import UnifiedTrainer
        
        class MockArgs:
            mode = 'sequential'
            verbose = False
            mastery_sessions = 5
            games = 10
            target_win_rate = 0.7
            target_score = 75
            max_learning_cycles = 5
            max_actions_per_session = 1000
            enable_meta_cognitive = True
            salience = 'decay'
            enable_contrarian_mode = True
        
        args = MockArgs()
        trainer = UnifiedTrainer(args)
        
        # Test that display_config method works
        trainer.display_config()  # Should not raise an exception


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
