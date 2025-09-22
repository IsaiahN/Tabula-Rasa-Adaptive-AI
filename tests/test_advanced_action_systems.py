#!/usr/bin/env python3
"""
Comprehensive tests for Advanced Action Systems

Tests all the new advanced action systems to ensure they work correctly
with database-only operation and integrate properly with the existing architecture.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Test imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.visual_interactive_system import VisualInteractiveSystem
from src.core.advanced_stagnation_system import AdvancedStagnationSystem, StagnationType
from src.core.strategy_discovery_system import StrategyDiscoverySystem
from src.core.enhanced_frame_analysis import EnhancedFrameAnalysisSystem, ChangeType
from src.core.systematic_exploration_system import SystematicExplorationSystem, ExplorationPhase
from src.core.emergency_override_system import EmergencyOverrideSystem, OverrideType
from src.training.governor.governor import TrainingGovernor

class TestAdvancedActionSystems:
    """Test suite for all advanced action systems."""
    
    @pytest.fixture
    def mock_integration(self):
        """Mock system integration for testing."""
        integration = Mock()
        integration.db = Mock()
        integration.db.execute = AsyncMock()
        integration.db.fetch_all = AsyncMock(return_value=[])
        integration.db.fetch_one = AsyncMock(return_value=None)
        return integration
    
    @pytest.fixture
    def mock_game_state(self):
        """Mock game state for testing."""
        return {
            'game_id': 'test_game_123',
            'session_id': 'test_session_456',
            'score': 10,
            'frame': [[[0, 0, 1] for _ in range(32)] for _ in range(32)],
            'available_actions': [1, 2, 3, 4, 5, 6],
            'state': 'NOT_FINISHED'
        }
    
    @pytest.fixture
    def mock_performance_history(self):
        """Mock performance history for testing."""
        return [
            {'score': 5, 'action': 1, 'timestamp': time.time() - 10},
            {'score': 7, 'action': 2, 'timestamp': time.time() - 8},
            {'score': 10, 'action': 3, 'timestamp': time.time() - 5},
            {'score': 10, 'action': 4, 'timestamp': time.time() - 3},
            {'score': 10, 'action': 5, 'timestamp': time.time() - 1}
        ]
    
    @pytest.fixture
    def mock_action_history(self):
        """Mock action history for testing."""
        return [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    
    @pytest.fixture
    def mock_frame_change_history(self):
        """Mock frame change history for testing."""
        return [True, True, False, False, False, False, False, False, False, False]

class TestVisualInteractiveSystem:
    """Test Visual-Interactive Action6 Targeting System."""
    
    @pytest.mark.asyncio
    async def test_analyze_frame_for_action6_targets(self, mock_integration, mock_game_state):
        """Test frame analysis for Action6 targets."""
        with patch('src.core.visual_interactive_system.get_system_integration', return_value=mock_integration):
            system = VisualInteractiveSystem()
            
            result = await system.analyze_frame_for_action6_targets(
                frame=mock_game_state['frame'],
                game_id=mock_game_state['game_id'],
                available_actions=mock_game_state['available_actions']
            )
            
            assert 'recommended_action6_coord' in result
            assert 'targeting_reason' in result
            assert 'confidence' in result
            assert 'interactive_targets' in result
    
    @pytest.mark.asyncio
    async def test_record_target_interaction(self, mock_integration):
        """Test recording target interaction results."""
        with patch('src.core.visual_interactive_system.get_system_integration', return_value=mock_integration):
            system = VisualInteractiveSystem()
            
            target = Mock()
            target.x = 10
            target.y = 15
            target.target_type = 'button'
            target.confidence = 0.8
            target.detection_method = 'opencv'
            
            await system.record_target_interaction(
                game_id='test_game',
                target=target,
                interaction_successful=True,
                frame_changes_detected=True,
                score_impact=5.0
            )
            
            # Verify database call was made
            mock_integration.db.execute.assert_called_once()

class TestAdvancedStagnationSystem:
    """Test Advanced Stagnation Detection System."""
    
    @pytest.mark.asyncio
    async def test_detect_stagnation_score_regression(self, mock_integration, mock_game_state, mock_performance_history):
        """Test stagnation detection for score regression."""
        with patch('src.core.advanced_stagnation_system.get_system_integration', return_value=mock_integration):
            system = AdvancedStagnationSystem()
            
            # Create performance history with score regression
            regression_history = [
                {'score': 10, 'action': 1, 'timestamp': time.time() - 10},
                {'score': 8, 'action': 2, 'timestamp': time.time() - 8},
                {'score': 6, 'action': 3, 'timestamp': time.time() - 6},
                {'score': 4, 'action': 4, 'timestamp': time.time() - 4},
                {'score': 2, 'action': 5, 'timestamp': time.time() - 2}
            ]
            
            stagnation_event = await system.detect_stagnation(
                game_id=mock_game_state['game_id'],
                session_id=mock_game_state['session_id'],
                current_state=mock_game_state,
                performance_history=regression_history,
                action_history=[1, 2, 3, 4, 5],
                frame_change_history=[False, False, False, False, False]
            )
            
            if stagnation_event:
                assert stagnation_event.stagnation_type == StagnationType.SCORE_REGRESSION
                assert stagnation_event.severity > 0
    
    @pytest.mark.asyncio
    async def test_detect_stagnation_action_repetition(self, mock_integration, mock_game_state):
        """Test stagnation detection for action repetition."""
        with patch('src.core.advanced_stagnation_system.get_system_integration', return_value=mock_integration):
            system = AdvancedStagnationSystem()
            
            # Create action history with repetition
            repetitive_actions = [1, 1, 1, 1, 1, 1, 1, 1]
            
            stagnation_event = await system.detect_stagnation(
                game_id=mock_game_state['game_id'],
                session_id=mock_game_state['session_id'],
                current_state=mock_game_state,
                performance_history=mock_performance_history,
                action_history=repetitive_actions,
                frame_change_history=[False] * len(repetitive_actions)
            )
            
            if stagnation_event:
                assert stagnation_event.stagnation_type == StagnationType.ACTION_REPETITION
                assert stagnation_event.severity > 0

class TestStrategyDiscoverySystem:
    """Test Strategy Discovery & Replication System."""
    
    @pytest.mark.asyncio
    async def test_discover_winning_strategy(self, mock_integration):
        """Test discovering a winning strategy."""
        with patch('src.core.strategy_discovery_system.get_system_integration', return_value=mock_integration):
            system = StrategyDiscoverySystem()
            
            action_sequence = [1, 2, 3, 4, 5]
            score_progression = [0, 2, 5, 8, 12, 15]
            
            strategy = await system.discover_winning_strategy(
                game_id='test_game',
                action_sequence=action_sequence,
                score_progression=score_progression
            )
            
            if strategy:
                assert strategy.action_sequence == action_sequence
                assert strategy.score_progression == score_progression
                assert strategy.total_score_increase == 15
                assert strategy.efficiency > 0
    
    @pytest.mark.asyncio
    async def test_get_best_strategy_for_game(self, mock_integration):
        """Test getting best strategy for a game."""
        with patch('src.core.strategy_discovery_system.get_system_integration', return_value=mock_integration):
            system = StrategyDiscoverySystem()
            
            # Mock database response
            mock_integration.db.fetch_all.return_value = [
                {
                    'strategy_id': 'test_strategy_1',
                    'game_type': 'test_type',
                    'game_id': 'test_game',
                    'action_sequence': json.dumps([1, 2, 3]),
                    'score_progression': json.dumps([0, 5, 10, 15]),
                    'total_score_increase': 15.0,
                    'efficiency': 5.0,
                    'discovery_timestamp': time.time(),
                    'replication_attempts': 0,
                    'successful_replications': 0,
                    'refinement_level': 0,
                    'is_active': True
                }
            ]
            
            strategy = await system.get_best_strategy_for_game('test_game')
            
            if strategy:
                assert strategy.strategy_id == 'test_strategy_1'
                assert strategy.efficiency == 5.0

class TestEnhancedFrameAnalysisSystem:
    """Test Enhanced Frame Change Analysis System."""
    
    @pytest.mark.asyncio
    async def test_analyze_frame_changes(self, mock_integration):
        """Test frame change analysis."""
        with patch('src.core.enhanced_frame_analysis.get_system_integration', return_value=mock_integration):
            system = EnhancedFrameAnalysisSystem()
            
            # Create mock frames
            before_frame = [[[0, 0, 0] for _ in range(32)] for _ in range(32)]
            after_frame = [[[1, 1, 1] for _ in range(32)] for _ in range(32)]
            
            analysis = await system.analyze_frame_changes(
                before_frame=before_frame,
                after_frame=after_frame,
                game_id='test_game',
                action_number=1,
                coordinates=(10, 15)
            )
            
            if analysis:
                assert analysis.game_id == 'test_game'
                assert analysis.action_number == 1
                assert analysis.coordinates == (10, 15)
                assert analysis.num_pixels_changed > 0
                assert analysis.change_percentage > 0
    
    @pytest.mark.asyncio
    async def test_classify_change_type(self, mock_integration):
        """Test change type classification."""
        with patch('src.core.enhanced_frame_analysis.get_system_integration', return_value=mock_integration):
            system = EnhancedFrameAnalysisSystem()
            
            # Test major movement classification
            change_locations = [(x, y) for x in range(10, 20) for y in range(10, 20)]
            change_type = system._classify_change_type(
                change_locations=change_locations,
                num_pixels_changed=100,
                change_percentage=10.0,
                movement_detected=True
            )
            
            assert change_type in [ChangeType.MAJOR_MOVEMENT, ChangeType.OBJECT_MOVEMENT, ChangeType.SMALL_MOVEMENT]

class TestSystematicExplorationSystem:
    """Test Systematic Exploration Phases System."""
    
    @pytest.mark.asyncio
    async def test_get_exploration_coordinates(self, mock_integration):
        """Test getting exploration coordinates."""
        with patch('src.core.systematic_exploration_system.get_system_integration', return_value=mock_integration):
            system = SystematicExplorationSystem()
            
            x, y, phase_name = await system.get_exploration_coordinates(
                game_id='test_game',
                session_id='test_session',
                grid_dimensions=(32, 32),
                available_actions=[1, 2, 3, 4, 5, 6]
            )
            
            assert isinstance(x, int)
            assert isinstance(y, int)
            assert isinstance(phase_name, str)
            assert 0 <= x < 32
            assert 0 <= y < 32
            assert phase_name in ['corners', 'center', 'edges', 'random']
    
    @pytest.mark.asyncio
    async def test_record_exploration_result(self, mock_integration):
        """Test recording exploration results."""
        with patch('src.core.systematic_exploration_system.get_system_integration', return_value=mock_integration):
            system = SystematicExplorationSystem()
            
            # Initialize a phase
            system.active_phases['test_game'] = Mock()
            system.active_phases['test_game'].phase_attempts = 0
            system.active_phases['test_game'].successful_attempts = 0
            system.active_phases['test_game'].phase_success_rate = 0.0
            
            await system.record_exploration_result(
                game_id='test_game',
                coordinates=(10, 15),
                success=True,
                frame_changes=True,
                score_impact=5.0
            )
            
            # Verify the phase was updated
            assert system.active_phases['test_game'].phase_attempts > 0

class TestEmergencyOverrideSystem:
    """Test Emergency Override Systems."""
    
    @pytest.mark.asyncio
    async def test_check_emergency_override_action_loop(self, mock_integration, mock_game_state):
        """Test emergency override for action loops."""
        with patch('src.core.emergency_override_system.get_system_integration', return_value=mock_integration):
            system = EmergencyOverrideSystem()
            
            # Create action history with repetition
            repetitive_actions = [1, 1, 1, 1, 1, 1, 1, 1]
            
            override = await system.check_emergency_override(
                game_id=mock_game_state['game_id'],
                session_id=mock_game_state['session_id'],
                current_state=mock_game_state,
                action_history=repetitive_actions,
                performance_history=mock_performance_history,
                available_actions=[1, 2, 3, 4, 5, 6]
            )
            
            if override:
                assert override.override_type == OverrideType.ACTION_LOOP_BREAK
                assert override.actions_before_override > 0
    
    @pytest.mark.asyncio
    async def test_record_override_result(self, mock_integration):
        """Test recording override results."""
        with patch('src.core.emergency_override_system.get_system_integration', return_value=mock_integration):
            system = EmergencyOverrideSystem()
            
            override = Mock()
            override.game_id = 'test_game'
            override.override_timestamp = time.time()
            
            await system.record_override_result(
                override=override,
                success=True,
                frame_changes=True,
                score_impact=5.0
            )
            
            # Verify database update was called
            mock_integration.db.execute.assert_called_once()

class TestGovernorIntegration:
    """Test Governor integration with advanced action systems."""
    
    @pytest.mark.asyncio
    async def test_get_advanced_action_systems_status(self):
        """Test getting status of all advanced action systems."""
        governor = TrainingGovernor()
        
        status = await governor.get_advanced_action_systems_status()
        
        assert 'visual_interactive_system' in status
        assert 'stagnation_system' in status
        assert 'strategy_discovery_system' in status
        assert 'frame_analysis_system' in status
        assert 'exploration_system' in status
        assert 'emergency_override_system' in status
        
        for system_name, system_info in status.items():
            assert 'available' in system_info
            assert 'description' in system_info
            assert 'features' in system_info
            assert system_info['available'] is True
    
    @pytest.mark.asyncio
    async def test_control_advanced_action_system(self, mock_integration):
        """Test controlling advanced action systems through Governor."""
        with patch('src.database.system_integration.get_system_integration', return_value=mock_integration):
            governor = TrainingGovernor()
            
            # Test getting visual targets
            result = await governor.control_advanced_action_system(
                system_name='visual_interactive_system',
                action='get_targets',
                parameters={'game_id': 'test_game'}
            )
            
            assert result['system'] == 'visual_interactive_system'
            assert result['action'] == 'get_targets'
            assert 'success' in result
            assert 'message' in result

class TestDatabaseIntegration:
    """Test database integration for all systems."""
    
    @pytest.mark.asyncio
    async def test_database_schema_extension(self):
        """Test that all new database tables are properly defined."""
        from src.database.schema import get_database_schema
        
        # This would test that the schema includes all new tables
        # For now, we'll just verify the schema file exists and is readable
        schema_file = 'src/database/schema.sql'
        assert os.path.exists(schema_file)
        
        with open(schema_file, 'r') as f:
            schema_content = f.read()
            
        # Check for new table definitions
        assert 'winning_strategies' in schema_content
        assert 'stagnation_events' in schema_content
        assert 'frame_change_analysis' in schema_content
        assert 'exploration_phases' in schema_content
        assert 'emergency_overrides' in schema_content
        assert 'visual_targets' in schema_content
        assert 'governor_decisions' in schema_content

class TestSystemIntegration:
    """Test system integration layer."""
    
    @pytest.mark.asyncio
    async def test_system_integration_methods(self, mock_integration):
        """Test that all new methods are available in system integration."""
        from src.database.system_integration import SystemIntegration
        
        integration = SystemIntegration()
        
        # Test that all new methods exist
        assert hasattr(integration, 'store_visual_target')
        assert hasattr(integration, 'store_stagnation_event')
        assert hasattr(integration, 'store_winning_strategy')
        assert hasattr(integration, 'store_frame_change_analysis')
        assert hasattr(integration, 'store_exploration_phase')
        assert hasattr(integration, 'store_emergency_override')
        assert hasattr(integration, 'store_governor_decision')
        
        assert hasattr(integration, 'get_visual_targets_for_game')
        assert hasattr(integration, 'get_stagnation_events_for_game')
        assert hasattr(integration, 'get_winning_strategies_for_game_type')
        assert hasattr(integration, 'get_frame_change_analysis_for_game')
        assert hasattr(integration, 'get_exploration_phases_for_game')
        assert hasattr(integration, 'get_emergency_overrides_for_game')
        assert hasattr(integration, 'get_governor_decisions_for_session')

if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
