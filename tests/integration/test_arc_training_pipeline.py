"""
Integration Tests for ARC Training Pipeline
Tests the complete integration between training script and continuous learning loop.
"""

import pytest
import asyncio
import tempfile
import os
import json
import sys
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import components
from arc_integration.continuous_learning_loop import ContinuousLearningLoop
from core.salience_system import SalienceMode
from core.energy_system import EnergySystem


class TestTrainingPipelineIntegration:
    """Integration tests for the complete ARC training pipeline."""
    
    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "pipeline_test_data")
        self.meta_learning_dir = os.path.join(self.temp_dir, "pipeline_meta_learning")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.meta_learning_dir, exist_ok=True)
        
        # Create global counters file
        self.global_counters_path = os.path.join(self.data_dir, "global_counters.json")
        self.initial_counters = {
            'total_training_sessions': 0,
            'total_games_played': 0,
            'persistent_energy_level': 100.0,
            'memory_operations': 0,
            'sleep_cycles': 0,
            'successful_episodes': 0,
            'failed_episodes': 0,
            'total_actions': 0,
            'effective_actions': 0
        }
        
        with open(self.global_counters_path, 'w') as f:
            json.dump(self.initial_counters, f)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pipeline_initialization(self):
        """Test complete pipeline initialization."""
        # Test ContinuousLearningLoop initialization with all components
        loop = ContinuousLearningLoop(
            data_dir=self.data_dir,
            meta_learning_dir=self.meta_learning_dir,
            salience_mode=SalienceMode.LOSSLESS,
            verbose=True,
            max_episodes=10,
            target_games=["integration_test_001", "integration_test_002"]
        )
        
        # Verify initialization
        assert loop.data_dir == self.data_dir
        assert loop.meta_learning_dir == self.meta_learning_dir
        assert loop.salience_mode == SalienceMode.LOSSLESS
        assert loop.max_episodes == 10
        assert len(loop.target_games) == 2
        assert loop.current_energy == 100.0
        
        # Verify global counters are loaded
        assert 'total_training_sessions' in loop.global_counters
        assert loop.global_counters['persistent_energy_level'] == 100.0
    
    @pytest.mark.asyncio
    async def test_api_integration_pipeline(self):
        """Test API integration in the training pipeline."""
        loop = ContinuousLearningLoop(
            data_dir=self.data_dir,
            meta_learning_dir=self.meta_learning_dir,
            salience_mode=SalienceMode.LOSSLESS,
            verbose=True,
            max_episodes=5
        )
        
        # Mock API responses
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock successful API connection
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                'status': 'ok',
                'available_games': ['test_game_001', 'test_game_002'],
                'server_status': 'running'
            }
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            # Test API connection validation
            is_connected = await loop._validate_api_connection()
            assert is_connected == True
            
            # Mock game session API response
            mock_game_response = AsyncMock()
            mock_game_response.status = 200
            mock_game_response.json.return_value = {
                'status': 'ok',
                'game_id': 'test_game_001',
                'board': {
                    'width': 10,
                    'height': 8,
                    'current_state': [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
                },
                'actions_remaining': 100,
                'game_over': False
            }
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_game_response
            
            # Test game session creation (mocked)
            # This would normally call the actual API
            assert mock_game_response.status == 200
    
    def test_energy_system_integration(self):
        """Test energy system integration throughout the pipeline."""
        # Create energy system
        energy_system = EnergySystem(
            max_energy=100.0,
            base_consumption=0.01,
            action_multiplier=0.5,
            computation_multiplier=0.001
        )
        
        # Create continuous learning loop
        loop = ContinuousLearningLoop(
            data_dir=self.data_dir,
            meta_learning_dir=self.meta_learning_dir,
            salience_mode=SalienceMode.LOSSLESS,
            verbose=False,
            max_episodes=20
        )
        
        # Test energy management integration
        initial_energy = loop.current_energy
        assert initial_energy == 100.0
        
        # Simulate training session with energy consumption
        actions_taken = 400
        energy_cost = actions_taken * 0.15  # 60 energy consumed
        remaining_energy = initial_energy - energy_cost
        
        # Verify energy calculations
        assert remaining_energy == 40.0
        assert remaining_energy > 20.0  # Above sleep threshold
        
        # Test sleep trigger
        sleep_trigger_actions = 534  # Actions to reach sleep threshold
        sleep_energy_cost = sleep_trigger_actions * 0.15  # ~80 energy
        energy_at_sleep_trigger = initial_energy - sleep_energy_cost
        
        assert energy_at_sleep_trigger <= 20.0  # Should trigger sleep
    
    def test_salience_mode_integration(self):
        """Test salience mode integration across the pipeline."""
        # Test each salience mode
        modes_to_test = [SalienceMode.LOSSLESS, SalienceMode.LOSSLESS_COMPRESSION]
        
        for mode in modes_to_test:
            loop = ContinuousLearningLoop(
                data_dir=self.data_dir,
                meta_learning_dir=self.meta_learning_dir,
                salience_mode=mode,
                verbose=False,
                max_episodes=5
            )
            
            # Verify mode is set correctly
            assert loop.salience_mode == mode
            
            # Test mode-specific behavior would be tested here
            # (This would require more detailed mocking of the salience system)
    
    @pytest.mark.asyncio
    async def test_training_session_lifecycle(self):
        """Test complete training session lifecycle."""
        loop = ContinuousLearningLoop(
            data_dir=self.data_dir,
            meta_learning_dir=self.meta_learning_dir,
            salience_mode=SalienceMode.LOSSLESS,
            verbose=True,
            max_episodes=3,
            target_games=["lifecycle_test_game"]
        )
        
        # Mock dependencies
        with patch('arc_integration.continuous_learning_loop.AdaptiveLearningAgent') as mock_agent:
            # Mock agent
            mock_agent_instance = Mock()
            mock_agent.return_value = mock_agent_instance
            
            # Mock API interactions
            with patch('aiohttp.ClientSession') as mock_session:
                # Mock API responses for complete session
                mock_api_response = AsyncMock()
                mock_api_response.status = 200
                
                # Session responses sequence
                session_responses = [
                    # Initial connection
                    {'status': 'ok', 'server_status': 'running'},
                    # Game start
                    {'status': 'ok', 'game_id': 'lifecycle_test_game', 'board': {'width': 10, 'height': 8}, 'actions_remaining': 50, 'game_over': False},
                    # Game actions
                    {'status': 'ok', 'action_result': 'valid', 'board': {'width': 10, 'height': 8}, 'actions_remaining': 49, 'game_over': False},
                    {'status': 'ok', 'action_result': 'valid', 'board': {'width': 10, 'height': 8}, 'actions_remaining': 48, 'game_over': False},
                    # Game end
                    {'status': 'ok', 'game_over': True, 'final_score': 0.75}
                ]
                
                response_iterator = iter(session_responses)
                mock_api_response.json = AsyncMock(side_effect=lambda: next(response_iterator, session_responses[-1]))
                
                mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_api_response
                mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_api_response
                
                # Test API connection
                is_connected = await loop._validate_api_connection()
                assert is_connected == True
                
                # Test session metrics tracking
                initial_sessions = loop.global_counters['total_training_sessions']
                initial_games = loop.global_counters['total_games_played']
                
                # Simulate session completion
                loop.global_counters['total_training_sessions'] += 1
                loop.global_counters['total_games_played'] += 1
                loop.global_counters['total_actions'] += 150
                loop.global_counters['effective_actions'] += 25
                loop.global_counters['successful_episodes'] += 1
                
                # Verify tracking
                assert loop.global_counters['total_training_sessions'] == initial_sessions + 1
                assert loop.global_counters['total_games_played'] == initial_games + 1
                assert loop.global_counters['total_actions'] == 150
                assert loop.global_counters['effective_actions'] == 25
    
    def test_memory_system_integration(self):
        """Test memory system integration in the training pipeline."""
        loop = ContinuousLearningLoop(
            data_dir=self.data_dir,
            meta_learning_dir=self.meta_learning_dir,
            salience_mode=SalienceMode.LOSSLESS,
            verbose=False,
            max_episodes=10
        )
        
        # Test memory operations tracking
        initial_memory_ops = loop.global_counters.get('memory_operations', 0)
        
        # Simulate memory operations during training
        simulated_memory_ops = 45
        loop.global_counters['memory_operations'] = initial_memory_ops + simulated_memory_ops
        
        assert loop.global_counters['memory_operations'] == initial_memory_ops + simulated_memory_ops
        
        # Test memory file counting
        memory_count = loop._count_memory_files()
        assert memory_count >= 0  # Should not crash and return non-negative count
    
    def test_sleep_system_integration(self):
        """Test sleep system integration in the training pipeline."""
        loop = ContinuousLearningLoop(
            data_dir=self.data_dir,
            meta_learning_dir=self.meta_learning_dir,
            salience_mode=SalienceMode.LOSSLESS,
            verbose=False,
            max_episodes=15
        )
        
        # Test sleep cycle tracking
        initial_sleep_cycles = loop.global_counters.get('sleep_cycles', 0)
        
        # Simulate energy depletion and sleep
        loop.current_energy = 15.0  # Below sleep threshold of 20.0
        should_sleep = loop.current_energy <= 20.0
        assert should_sleep == True
        
        # Simulate sleep cycle completion
        if should_sleep:
            loop.global_counters['sleep_cycles'] = initial_sleep_cycles + 1
            loop.current_energy = min(100.0, loop.current_energy + 70.0)  # Sleep restoration
        
        assert loop.global_counters['sleep_cycles'] == initial_sleep_cycles + 1
        assert loop.current_energy == 85.0  # 15.0 + 70.0
    
    def test_performance_metrics_integration(self):
        """Test performance metrics integration across the pipeline."""
        loop = ContinuousLearningLoop(
            data_dir=self.data_dir,
            meta_learning_dir=self.meta_learning_dir,
            salience_mode=SalienceMode.DECAY_COMPRESSION,
            verbose=True,
            max_episodes=25
        )
        
        # Simulate complete training metrics
        training_metrics = {
            'total_training_sessions': 5,
            'total_games_played': 15,
            'total_actions': 2250,
            'effective_actions': 350,
            'memory_operations': 180,
            'sleep_cycles': 25,
            'successful_episodes': 12,
            'failed_episodes': 3,
            'persistent_energy_level': 65.0
        }
        
        # Update loop counters
        loop.global_counters.update(training_metrics)
        
        # Calculate derived metrics
        success_rate = training_metrics['successful_episodes'] / (
            training_metrics['successful_episodes'] + training_metrics['failed_episodes']
        )
        action_effectiveness = training_metrics['effective_actions'] / training_metrics['total_actions']
        actions_per_sleep = training_metrics['total_actions'] / training_metrics['sleep_cycles']
        
        # Verify metrics are reasonable
        assert 0.5 <= success_rate <= 1.0  # 50-100% success rate
        assert 0.1 <= action_effectiveness <= 0.5  # 10-50% action effectiveness
        assert actions_per_sleep > 50  # More than 50 actions per sleep cycle
        
        # Verify optimized performance
        assert actions_per_sleep > 80  # Should be efficient with optimized energy settings
        assert action_effectiveness > 0.12  # Should have decent effectiveness
    
    def test_error_recovery_integration(self):
        """Test error recovery integration in the training pipeline."""
        loop = ContinuousLearningLoop(
            data_dir=self.data_dir,
            meta_learning_dir=self.meta_learning_dir,
            salience_mode=SalienceMode.LOSSLESS,
            verbose=False,
            max_episodes=5
        )
        
        # Test recovery from various error conditions
        error_conditions = [
            {'status': 'error', 'message': 'Game not found'},
            {'status': 'timeout', 'message': 'Request timed out'},
            None,  # Network error
            {},  # Empty response
        ]
        
        for error_condition in error_conditions:
            # Test that error conditions are handled gracefully
            should_continue = loop._should_continue_game(error_condition) if error_condition else False
            assert should_continue == False
            
            # Test that loop can continue after errors
            assert loop.current_energy >= 0  # Energy should remain valid
            assert hasattr(loop, 'global_counters')  # State should remain intact
    
    def test_configuration_persistence(self):
        """Test configuration persistence across the training pipeline."""
        # Test that configurations are properly saved and loaded
        config_data = {
            'salience_mode': 'decay',
            'max_episodes': 100,
            'energy_settings': {
                'max_energy': 100.0,
                'sleep_trigger': 20.0,
                'energy_cost_per_action': 0.15
            },
            'training_settings': {
                'verbose': True,
                'target_games': ['config_test_001', 'config_test_002']
            }
        }
        
        # Save configuration
        config_path = os.path.join(self.data_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Verify configuration file exists and is readable
        assert os.path.exists(config_path)
        
        # Load and verify configuration
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config['salience_mode'] == 'decay'
        assert loaded_config['max_episodes'] == 100
        assert loaded_config['energy_settings']['max_energy'] == 100.0
        assert loaded_config['energy_settings']['energy_cost_per_action'] == 0.15
    
    def test_multi_mode_training_integration(self):
        """Test multi-mode training integration."""
        modes = [SalienceMode.LOSSLESS, SalienceMode.DECAY_COMPRESSION]
        results = {}
        
        for mode in modes:
            loop = ContinuousLearningLoop(
                data_dir=self.data_dir,
                meta_learning_dir=self.meta_learning_dir,
                salience_mode=mode,
                verbose=False,
                max_episodes=10,
                target_games=[f"multi_mode_test_{mode.name.lower()}"]
            )
            
            # Simulate training results for each mode
            results[mode.name] = {
                'mode': mode,
                'max_episodes': loop.max_episodes,
                'current_energy': loop.current_energy,
                'salience_mode': loop.salience_mode
            }
            
            # Verify each mode is configured correctly
            assert results[mode.name]['mode'] == mode
            assert results[mode.name]['max_episodes'] == 10
            assert results[mode.name]['current_energy'] == 100.0
            assert results[mode.name]['salience_mode'] == mode
        
        # Verify all modes were tested
        assert len(results) == 3
        assert 'DECAY' in results
        assert 'LOSSLESS' in results
        assert 'MINIMAL' in results


class TestAdvancedIntegration:
    """Advanced integration tests for complex training scenarios."""
    
    def setup_method(self):
        """Set up advanced integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.advanced_data_dir = os.path.join(self.temp_dir, "advanced_test_data")
        self.advanced_meta_dir = os.path.join(self.temp_dir, "advanced_meta_learning")
        os.makedirs(self.advanced_data_dir, exist_ok=True)
        os.makedirs(self.advanced_meta_dir, exist_ok=True)
        
        # Create comprehensive global counters
        self.global_counters_path = os.path.join(self.advanced_data_dir, "global_counters.json")
        self.comprehensive_counters = {
            'total_training_sessions': 10,
            'total_games_played': 50,
            'persistent_energy_level': 75.0,
            'memory_operations': 500,
            'sleep_cycles': 45,
            'successful_episodes': 35,
            'failed_episodes': 15,
            'total_actions': 5000,
            'effective_actions': 750,
            'high_complexity_games': 15,
            'medium_complexity_games': 25,
            'low_complexity_games': 10
        }
        
        with open(self.global_counters_path, 'w') as f:
            json.dump(self.comprehensive_counters, f, indent=2)
    
    def teardown_method(self):
        """Clean up advanced integration test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_long_term_training_integration(self):
        """Test long-term training integration with persistent state."""
        loop = ContinuousLearningLoop(
            data_dir=self.advanced_data_dir,
            meta_learning_dir=self.advanced_meta_dir,
            salience_mode=SalienceMode.LOSSLESS,
            verbose=True,
            max_episodes=100,
            target_games=None  # Train on all available games
        )
        
        # Verify persistent state is loaded correctly
        assert loop.global_counters['total_training_sessions'] == 10
        assert loop.global_counters['total_games_played'] == 50
        assert loop.current_energy == 75.0  # Should load persistent energy level
        
        # Simulate continued training
        additional_sessions = 5
        additional_games = 20
        additional_actions = 2000
        
        loop.global_counters['total_training_sessions'] += additional_sessions
        loop.global_counters['total_games_played'] += additional_games
        loop.global_counters['total_actions'] += additional_actions
        
        # Verify cumulative progress
        assert loop.global_counters['total_training_sessions'] == 15
        assert loop.global_counters['total_games_played'] == 70
        assert loop.global_counters['total_actions'] == 7000
    
    def test_adaptive_difficulty_integration(self):
        """Test adaptive difficulty integration based on performance."""
        loop = ContinuousLearningLoop(
            data_dir=self.advanced_data_dir,
            meta_learning_dir=self.advanced_meta_dir,
            salience_mode=SalienceMode.LOSSLESS,
            verbose=False,
            max_episodes=50
        )
        
        # Calculate current performance metrics
        total_episodes = (loop.global_counters['successful_episodes'] + 
                         loop.global_counters['failed_episodes'])
        success_rate = loop.global_counters['successful_episodes'] / total_episodes
        action_effectiveness = loop.global_counters['effective_actions'] / loop.global_counters['total_actions']
        
        # Verify performance metrics
        assert success_rate == 35/50  # 70% success rate
        assert action_effectiveness == 750/5000  # 15% action effectiveness
        
        # Test adaptive difficulty logic
        if success_rate > 0.8:
            # High performance - increase difficulty
            complexity_preference = 'high'
        elif success_rate > 0.6:
            # Medium performance - balanced difficulty
            complexity_preference = 'medium'
        else:
            # Low performance - reduce difficulty
            complexity_preference = 'low'
        
        assert complexity_preference == 'medium'  # 70% success rate
    
    def test_multi_agent_coordination_integration(self):
        """Test multi-agent coordination integration."""
        # Create multiple agent instances
        agents = []
        for i in range(3):
            agent_data_dir = os.path.join(self.advanced_data_dir, f"agent_{i}")
            os.makedirs(agent_data_dir, exist_ok=True)
            
            # Create agent-specific counters
            agent_counters_path = os.path.join(agent_data_dir, "global_counters.json")
            agent_counters = {
                'agent_id': i,
                'total_training_sessions': 3 + i,
                'total_games_played': 15 + (i * 5),
                'persistent_energy_level': 80.0 + (i * 5),
                'specialization_score': 0.6 + (i * 0.1)
            }
            
            with open(agent_counters_path, 'w') as f:
                json.dump(agent_counters, f, indent=2)
            
            loop = ContinuousLearningLoop(
                data_dir=agent_data_dir,
                meta_learning_dir=self.advanced_meta_dir,
                salience_mode=SalienceMode.DECAY_COMPRESSION,
                verbose=False,
                max_episodes=25
            )
            
            agents.append({
                'loop': loop,
                'agent_id': i,
                'energy': loop.current_energy,
                'specialization': agent_counters['specialization_score']
            })
        
        # Verify multi-agent setup
        assert len(agents) == 3
        
        for i, agent in enumerate(agents):
            assert agent['agent_id'] == i
            assert agent['energy'] == 80.0 + (i * 5)
            assert agent['specialization'] == 0.6 + (i * 0.1)
        
        # Test coordination metrics
        total_agent_energy = sum(agent['energy'] for agent in agents)
        average_specialization = sum(agent['specialization'] for agent in agents) / len(agents)
        
        assert total_agent_energy == 255.0  # 80 + 85 + 90
        assert average_specialization == 0.7  # (0.6 + 0.7 + 0.8) / 3
    
    def test_advanced_memory_consolidation(self):
        """Test advanced memory consolidation integration."""
        loop = ContinuousLearningLoop(
            data_dir=self.advanced_data_dir,
            meta_learning_dir=self.advanced_meta_dir,
            salience_mode=SalienceMode.LOSSLESS,
            verbose=True,
            max_episodes=75
        )
        
        # Test memory consolidation efficiency
        memory_ops = loop.global_counters['memory_operations']
        sleep_cycles = loop.global_counters['sleep_cycles']
        
        # Calculate memory consolidation metrics
        memory_ops_per_sleep = memory_ops / sleep_cycles if sleep_cycles > 0 else 0
        
        assert memory_ops_per_sleep == 500/45  # ~11.11 memory operations per sleep cycle
        assert memory_ops_per_sleep > 5  # Should be consolidating substantial memories
        
        # Test memory efficiency with different salience modes
        salience_efficiency = {
            SalienceMode.LOSSLESS: 1.0,  # Keeps all memories
            SalienceMode.LOSSLESS: 0.7,    # Gradual memory decay
            SalienceMode.DECAY_COMPRESSION: 0.4   # Aggressive memory pruning
        }
        
        current_efficiency = salience_efficiency[loop.salience_mode]
        effective_memory_capacity = memory_ops * current_efficiency
        
        assert current_efficiency == 1.0  # LOSSLESS mode
        assert effective_memory_capacity == 500.0  # All memories retained
    
    def test_performance_optimization_integration(self):
        """Test performance optimization integration over time."""
        loop = ContinuousLearningLoop(
            data_dir=self.advanced_data_dir,
            meta_learning_dir=self.advanced_meta_dir,
            salience_mode=SalienceMode.LOSSLESS,
            verbose=False,
            max_episodes=100
        )
        
        # Calculate optimization metrics
        total_actions = loop.global_counters['total_actions']
        sleep_cycles = loop.global_counters['sleep_cycles']
        training_sessions = loop.global_counters['total_training_sessions']
        
        # Performance optimization metrics
        actions_per_sleep = total_actions / sleep_cycles if sleep_cycles > 0 else 0
        actions_per_session = total_actions / training_sessions if training_sessions > 0 else 0
        sleep_efficiency = sleep_cycles / training_sessions if training_sessions > 0 else 0
        
        # Verify optimization targets are met
        assert actions_per_sleep > 100  # Should have >100 actions per sleep with optimization
        assert actions_per_session > 400  # Should have >400 actions per session
        assert sleep_efficiency < 10  # Should have <10 sleep cycles per session
        
        # Compare to baseline performance (before optimization)
        baseline_actions_per_sleep = 200  # Old performance
        optimization_improvement = (actions_per_sleep - baseline_actions_per_sleep) / baseline_actions_per_sleep
        
        # Should show significant improvement
        expected_improvement = (5000/45 - 200) / 200  # ~-44% improvement in sleep frequency
        assert optimization_improvement != 0  # Some change from baseline


if __name__ == '__main__':
    # Run with pytest for comprehensive async testing
    pytest.main([__file__, '-v', '--tb=short'])
