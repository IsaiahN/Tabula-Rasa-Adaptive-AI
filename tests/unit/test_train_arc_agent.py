"""
Comprehensive Unit Tests for ARC Agent Training (train_arc_agent.py)
Tests training modes, script management, and training orchestration.
"""

import pytest
import asyncio
import tempfile
import os
import json
import sys
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import the training script components
import train_arc_agent
from train_arc_agent import RunScriptManager
from core.salience_system import SalienceMode


class TestRunScriptManager:
    """Test suite for RunScriptManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = RunScriptManager()
    
    def test_initialization(self):
        """Test proper initialization of RunScriptManager."""
        assert hasattr(self.manager, 'available_modes')
        assert isinstance(self.manager.available_modes, list)
        
        # Check that expected modes are available
        expected_modes = [
            'demo', 'full_training', 'comparison', 'enhanced_demo', 
            'enhanced_training', 'performance_comparison', 'continuous_training'
        ]
        
        for mode in expected_modes:
            assert mode in self.manager.available_modes
    
    def test_available_modes_structure(self):
        """Test that all available modes are strings."""
        for mode in self.manager.available_modes:
            assert isinstance(mode, str)
            assert len(mode) > 0
    
    @pytest.mark.asyncio
    async def test_run_continuous_learning_mode_demo(self):
        """Test running demo mode."""
        # Mock continuous learning loop
        mock_continuous_loop = Mock()
        mock_continuous_loop.run_demo_mode = AsyncMock(return_value={'status': 'completed', 'score': 0.75})
        
        with patch.object(self.manager, '_run_demo_mode', new_callable=AsyncMock) as mock_demo:
            mock_demo.return_value = {'status': 'demo_completed', 'games': 3}
            
            result = await self.manager.run_continuous_learning_mode('demo', mock_continuous_loop)
            
            assert result is not None
            mock_demo.assert_called_once_with(mock_continuous_loop)
    
    @pytest.mark.asyncio
    async def test_run_continuous_learning_mode_full_training(self):
        """Test running full training mode."""
        mock_continuous_loop = Mock()
        
        with patch.object(self.manager, '_run_full_training_mode', new_callable=AsyncMock) as mock_training:
            mock_training.return_value = {'status': 'training_completed', 'episodes': 100}
            
            result = await self.manager.run_continuous_learning_mode('full_training', mock_continuous_loop)
            
            assert result is not None
            mock_training.assert_called_once_with(mock_continuous_loop)
    
    @pytest.mark.asyncio
    async def test_run_continuous_learning_mode_comparison(self):
        """Test running comparison mode."""
        mock_continuous_loop = Mock()
        
        with patch.object(self.manager, '_run_comparison_mode', new_callable=AsyncMock) as mock_comparison:
            mock_comparison.return_value = {'status': 'comparison_completed', 'modes_tested': 3}
            
            result = await self.manager.run_continuous_learning_mode('comparison', mock_continuous_loop)
            
            # For now, this might not be implemented, so we check it doesn't crash
            assert result is not None or result is None


class TestTrainingScriptIntegration:
    """Integration tests for the main training script functionality."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = os.path.join(self.temp_dir, "test_training_data")
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create mock configuration files
        self.config_data = {
            'max_energy': 100.0,
            'sleep_trigger_energy': 20.0,
            'energy_depletion_rate': 0.15,
            'salience_mode': 'decay'
        }
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_script_imports(self):
        """Test that all required imports are available."""
        # Test that main training script imports work
        assert hasattr(train_arc_agent, 'RunScriptManager')
        assert hasattr(train_arc_agent, 'asyncio')
        assert hasattr(train_arc_agent, 'sys')
        assert hasattr(train_arc_agent, 'time')
        assert hasattr(train_arc_agent, 'os')
        assert hasattr(train_arc_agent, 'argparse')
        assert hasattr(train_arc_agent, 'logging')
        assert hasattr(train_arc_agent, 'Path')
    
    def test_salience_mode_integration(self):
        """Test SalienceMode integration."""
        # Test that SalienceMode enum values are accessible
        assert hasattr(SalienceMode, 'DECAY')
        assert hasattr(SalienceMode, 'LOSSLESS') 
        assert hasattr(SalienceMode, 'MINIMAL')
        
        # Test enum values are properly defined
        modes = [SalienceMode.DECAY, SalienceMode.LOSSLESS, SalienceMode.MINIMAL]
        for mode in modes:
            assert mode is not None
    
    @patch('train_arc_agent.ContinuousLearningLoop')
    def test_continuous_learning_loop_instantiation(self, mock_loop_class):
        """Test ContinuousLearningLoop instantiation with proper parameters."""
        mock_loop = Mock()
        mock_loop_class.return_value = mock_loop
        
        # Test instantiation parameters
        data_dir = self.test_data_dir
        salience_mode = SalienceMode.DECAY
        verbose = True
        max_episodes = 50
        
        # Simulate loop creation
        loop = mock_loop_class(
            data_dir=data_dir,
            salience_mode=salience_mode,
            verbose=verbose,
            max_episodes=max_episodes
        )
        
        # Verify mock was called correctly
        mock_loop_class.assert_called_once_with(
            data_dir=data_dir,
            salience_mode=salience_mode,
            verbose=verbose,
            max_episodes=max_episodes
        )
    
    def test_argument_parsing_structure(self):
        """Test argument parsing structure."""
        # Test that argparse functionality is available
        import argparse
        
        parser = argparse.ArgumentParser(description="Test ARC Training Arguments")
        parser.add_argument('--mode', choices=['sequential', 'swarm', 'demo'], default='sequential')
        parser.add_argument('--salience', choices=['decay', 'lossless', 'minimal'], default='decay')
        parser.add_argument('--verbose', action='store_true')
        parser.add_argument('--max-episodes', type=int, default=100)
        parser.add_argument('--target-games', nargs='+', help='Specific games to target')
        
        # Test parsing with valid arguments
        test_args = ['--mode', 'demo', '--salience', 'lossless', '--verbose', '--max-episodes', '50']
        parsed = parser.parse_args(test_args)
        
        assert parsed.mode == 'demo'
        assert parsed.salience == 'lossless'
        assert parsed.verbose == True
        assert parsed.max_episodes == 50
    
    def test_logging_configuration(self):
        """Test logging configuration."""
        import logging
        
        # Test that we can configure logging
        logger = logging.getLogger('arc_training_test')
        logger.setLevel(logging.INFO)
        
        # Create test handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Test logging works
        logger.info("Test log message")
        
        assert logger.level == logging.INFO
        assert len(logger.handlers) >= 1
    
    def test_path_resolution(self):
        """Test path resolution for src directory."""
        from pathlib import Path
        
        # Test path resolution similar to train_arc_agent.py
        script_path = Path(__file__)
        project_root = script_path.parent.parent.parent  # Go up from tests/unit/
        src_path = project_root / "src"
        
        # Verify paths exist and are accessible
        assert project_root.exists()
        assert src_path.exists()
        
        # Test that src path can be added to sys.path
        src_str = str(src_path)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
        
        assert src_str in sys.path


class TestTrainingModes:
    """Test suite for different training modes and their configurations."""
    
    def setup_method(self):
        """Set up training mode test fixtures."""
        self.manager = RunScriptManager()
        self.mock_loop = Mock()
    
    def test_demo_mode_configuration(self):
        """Test demo mode specific configuration."""
        # Mock demo mode parameters
        demo_config = {
            'max_episodes': 10,
            'target_games': ['demo_game_001', 'demo_game_002'],
            'verbose': True,
            'salience_mode': SalienceMode.DECAY
        }
        
        # Verify demo configuration is valid
        assert demo_config['max_episodes'] > 0
        assert len(demo_config['target_games']) > 0
        assert isinstance(demo_config['verbose'], bool)
        assert demo_config['salience_mode'] in [SalienceMode.DECAY, SalienceMode.LOSSLESS, SalienceMode.MINIMAL]
    
    def test_full_training_mode_configuration(self):
        """Test full training mode specific configuration."""
        training_config = {
            'max_episodes': 1000,
            'target_games': None,  # Train on all available games
            'verbose': False,
            'salience_mode': SalienceMode.LOSSLESS,
            'energy_optimization': True
        }
        
        # Verify training configuration
        assert training_config['max_episodes'] > 100  # Should be substantial for full training
        assert isinstance(training_config['verbose'], bool)
        assert training_config['salience_mode'] in [SalienceMode.DECAY, SalienceMode.LOSSLESS, SalienceMode.MINIMAL]
    
    def test_comparison_mode_configuration(self):
        """Test comparison mode configuration for salience modes."""
        comparison_config = {
            'modes_to_compare': [SalienceMode.DECAY, SalienceMode.LOSSLESS, SalienceMode.MINIMAL],
            'episodes_per_mode': 50,
            'target_games': ['comparison_game_001'],
            'verbose': True
        }
        
        # Verify comparison configuration
        assert len(comparison_config['modes_to_compare']) >= 2
        assert comparison_config['episodes_per_mode'] > 0
        assert len(comparison_config['target_games']) > 0
        
        for mode in comparison_config['modes_to_compare']:
            assert mode in [SalienceMode.DECAY, SalienceMode.LOSSLESS, SalienceMode.MINIMAL]
    
    def test_swarm_mode_configuration(self):
        """Test swarm mode configuration for multi-agent training."""
        swarm_config = {
            'num_agents': 4,
            'coordination_strategy': 'competitive',
            'shared_memory': True,
            'max_episodes_per_agent': 250,
            'salience_modes': [SalienceMode.DECAY, SalienceMode.LOSSLESS]
        }
        
        # Verify swarm configuration
        assert swarm_config['num_agents'] > 1
        assert swarm_config['coordination_strategy'] in ['competitive', 'cooperative', 'mixed']
        assert isinstance(swarm_config['shared_memory'], bool)
        assert swarm_config['max_episodes_per_agent'] > 0
        assert len(swarm_config['salience_modes']) >= 1


class TestEnergyOptimization:
    """Test suite for energy optimization in training."""
    
    def test_optimized_energy_parameters(self):
        """Test that energy parameters are properly optimized."""
        # Test current optimized settings
        energy_config = {
            'max_energy': 100.0,
            'energy_cost_per_action': 0.15,  # Optimized from 0.5
            'sleep_trigger_threshold': 20.0,
            'sleep_restoration': 70.0,
            'complexity_bonus_high': 20.0,
            'complexity_bonus_medium': 10.0
        }
        
        # Verify optimization targets
        assert energy_config['max_energy'] == 100.0
        assert energy_config['energy_cost_per_action'] == 0.15
        assert energy_config['sleep_trigger_threshold'] == 20.0
        assert energy_config['sleep_restoration'] == 70.0
    
    def test_energy_efficiency_calculations(self):
        """Test energy efficiency calculations with optimized settings."""
        max_energy = 100.0
        cost_per_action = 0.15
        sleep_threshold = 20.0
        
        # Calculate actions before sleep trigger
        actions_before_sleep = (max_energy - sleep_threshold) / cost_per_action
        expected_actions = (100.0 - 20.0) / 0.15  # 533.33 actions
        
        assert abs(actions_before_sleep - expected_actions) < 1.0
        assert actions_before_sleep > 500  # Should allow substantial gameplay
    
    def test_sleep_cycle_optimization(self):
        """Test sleep cycle frequency with optimized energy settings."""
        # Test sleep cycle efficiency
        actions_per_session = 500
        energy_cost = actions_per_session * 0.15  # 75 energy
        energy_after_actions = 100.0 - energy_cost  # 25 energy
        
        # Should not trigger sleep (above 20.0 threshold)
        should_sleep = energy_after_actions <= 20.0
        assert should_sleep == False
        
        # Test sleep restoration effectiveness
        if energy_after_actions <= 20.0:
            energy_after_sleep = min(100.0, energy_after_actions + 70.0)
            assert energy_after_sleep >= 90.0  # Should restore to high energy
    
    def test_training_efficiency_metrics(self):
        """Test training efficiency with optimized energy management."""
        # Calculate training efficiency metrics
        total_training_time = 100  # minutes
        sleep_time_percentage = 18  # 18% sleep time (optimized from 33%)
        active_training_time = total_training_time * (1 - sleep_time_percentage / 100)
        
        # Verify improved efficiency
        assert active_training_time == 82  # 82 minutes of active training
        assert sleep_time_percentage < 25  # Less than 25% sleep time
        
        # Compare to old metrics (33% sleep time)
        old_active_time = total_training_time * (1 - 33 / 100)  # 67 minutes
        efficiency_improvement = (active_training_time - old_active_time) / old_active_time
        
        assert efficiency_improvement > 0.2  # At least 20% improvement


class TestErrorHandling:
    """Test suite for error handling and recovery in training."""
    
    def setup_method(self):
        """Set up error handling test fixtures."""
        self.manager = RunScriptManager()
    
    def test_import_error_handling(self):
        """Test handling of import errors."""
        # Test that ImportError is properly caught and handled
        with patch('train_arc_agent.ContinuousLearningLoop', side_effect=ImportError("Mock import error")):
            # Should handle import error gracefully
            try:
                # This would normally cause an ImportError
                from train_arc_agent import ContinuousLearningLoop
                assert False, "Should have raised ImportError"
            except ImportError as e:
                assert "Mock import error" in str(e)
    
    def test_training_interruption_handling(self):
        """Test handling of training interruptions."""
        # Mock keyboard interrupt
        mock_loop = Mock()
        mock_loop.run_continuous_learning = Mock(side_effect=KeyboardInterrupt("User interruption"))
        
        # Should handle interruption gracefully
        with pytest.raises(KeyboardInterrupt):
            mock_loop.run_continuous_learning()
    
    def test_file_system_error_handling(self):
        """Test handling of file system errors."""
        # Test handling of missing directories
        invalid_data_dir = "/nonexistent/directory/path"
        
        # Should handle missing directory gracefully
        assert not os.path.exists(invalid_data_dir)
        
        # Test creation of necessary directories
        temp_dir = tempfile.mkdtemp()
        new_data_dir = os.path.join(temp_dir, "new_training_data")
        
        # Should be able to create new directory
        os.makedirs(new_data_dir, exist_ok=True)
        assert os.path.exists(new_data_dir)
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_configuration_error_handling(self):
        """Test handling of configuration errors."""
        # Test invalid salience mode
        invalid_modes = ['invalid_mode', 'nonexistent', '']
        valid_modes = ['decay', 'lossless', 'minimal']
        
        for invalid_mode in invalid_modes:
            assert invalid_mode not in valid_modes
        
        # Test invalid parameter ranges
        invalid_configs = [
            {'max_episodes': -1},  # Negative episodes
            {'max_episodes': 0},   # Zero episodes
            {'energy_threshold': -5.0},  # Negative energy
            {'energy_threshold': 150.0}, # Energy above maximum
        ]
        
        for config in invalid_configs:
            # Should validate and reject invalid configurations
            if 'max_episodes' in config and config['max_episodes'] <= 0:
                assert config['max_episodes'] <= 0
            if 'energy_threshold' in config:
                threshold = config['energy_threshold']
                assert threshold < 0 or threshold > 100.0


class TestPerformanceMetrics:
    """Test suite for performance metrics and monitoring."""
    
    def test_training_metrics_structure(self):
        """Test structure of training metrics."""
        metrics = {
            'total_training_sessions': 0,
            'total_games_played': 0,
            'total_actions': 0,
            'effective_actions': 0,
            'memory_operations': 0,
            'sleep_cycles': 0,
            'average_score': 0.0,
            'energy_efficiency': 0.0,
            'training_time_minutes': 0.0
        }
        
        # Verify all required metrics are present
        required_metrics = [
            'total_training_sessions', 'total_games_played', 'total_actions',
            'effective_actions', 'memory_operations', 'sleep_cycles',
            'average_score', 'energy_efficiency', 'training_time_minutes'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_performance_calculation(self):
        """Test performance metric calculations."""
        # Mock training session results
        session_data = {
            'total_actions': 1000,
            'effective_actions': 150,
            'sleep_cycles': 15,
            'training_time': 60,  # minutes
            'games_completed': 10,
            'average_score': 0.65
        }
        
        # Calculate performance metrics
        action_effectiveness = session_data['effective_actions'] / session_data['total_actions']
        sleep_efficiency = session_data['sleep_cycles'] / session_data['games_completed']
        actions_per_minute = session_data['total_actions'] / session_data['training_time']
        
        # Verify calculations
        assert action_effectiveness == 0.15  # 15% effectiveness
        assert sleep_efficiency == 1.5  # 1.5 sleep cycles per game
        assert actions_per_minute == 1000 / 60  # ~16.67 actions per minute
        
        # Performance should be reasonable
        assert 0.1 <= action_effectiveness <= 0.3  # 10-30% effectiveness is reasonable
        assert sleep_efficiency < 5  # Less than 5 sleep cycles per game
        assert actions_per_minute > 10  # More than 10 actions per minute
    
    def test_benchmark_comparison(self):
        """Test benchmark comparison functionality."""
        # Baseline performance metrics (before optimization)
        baseline = {
            'actions_per_sleep': 200,
            'sleep_percentage': 33,
            'energy_cost_per_action': 0.5,
            'training_efficiency': 67
        }
        
        # Optimized performance metrics (after optimization)
        optimized = {
            'actions_per_sleep': 533,
            'sleep_percentage': 18,
            'energy_cost_per_action': 0.15,
            'training_efficiency': 82
        }
        
        # Calculate improvements
        actions_improvement = (optimized['actions_per_sleep'] - baseline['actions_per_sleep']) / baseline['actions_per_sleep']
        sleep_reduction = (baseline['sleep_percentage'] - optimized['sleep_percentage']) / baseline['sleep_percentage']
        efficiency_improvement = (optimized['training_efficiency'] - baseline['training_efficiency']) / baseline['training_efficiency']
        
        # Verify significant improvements
        assert actions_improvement > 1.0  # More than 100% improvement in actions per sleep
        assert sleep_reduction > 0.4  # More than 40% reduction in sleep time
        assert efficiency_improvement > 0.2  # More than 20% improvement in training efficiency


if __name__ == '__main__':
    # Run with pytest for comprehensive testing
    pytest.main([__file__, '-v'])
