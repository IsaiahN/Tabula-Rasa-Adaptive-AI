"""
Comprehensive tests for AI-enhanced CORE_GAME_MECHANICS system.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any, Tuple

# Add CORE_GAME_MECHANICS to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CORE_GAME_MECHANICS'))

try:
    from ai_orchestration import AIOrchestrator
    from vision_coordinator import VisionCoordinator
    from pattern_action_selector import PatternActionSelector
    from knowledge_integration import KnowledgeIntegrator
    from core_gameplay import CoreGameplay
    from action_handler import ActionHandler
    from arc_api_client import GameState
except ImportError as e:
    pytest.skip(f"CORE_GAME_MECHANICS modules not available: {e}", allow_module_level=True)


class TestAIOrchestrator:
    """Test AI orchestration system."""

    @pytest.fixture
    def mock_db(self):
        """Mock database interface."""
        db = Mock()
        db.get_action_patterns.return_value = [
            {'pattern_data': {'colors': [1, 2, 3]}, 'success_action': 'ACTION1', 'confidence': 0.8}
        ]
        db.get_similar_games.return_value = [
            {'game_id': 'test_game', 'similarity': 0.9, 'successful_actions': ['ACTION2']}
        ]
        db.get_gan_prediction.return_value = {'predicted_action': 'ACTION3', 'confidence': 0.7}
        return db

    @pytest.fixture
    def orchestrator(self, mock_db):
        """Create AI orchestrator with mocked dependencies."""
        return AIOrchestrator(mock_db, "test_game")

    @pytest.mark.asyncio
    async def test_action_selection_integration(self, orchestrator):
        """Test complete action selection with all AI systems."""
        frame = [[1, 2], [3, 4]]
        game_state = Mock()
        game_state.score = 100
        game_state.frame = frame
        available_actions = ["ACTION1", "ACTION2", "ACTION3"]

        action, metadata = await orchestrator.select_action(frame, game_state, available_actions)

        assert action in available_actions
        assert isinstance(metadata, dict)
        assert 'pattern_confidence' in metadata
        assert 'knowledge_confidence' in metadata
        assert 'gan_confidence' in metadata
        assert 'vision_confidence' in metadata

    def test_frame_analysis(self, orchestrator):
        """Test frame analysis functionality."""
        frame = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        analysis = orchestrator._analyze_frame(frame)

        assert 'colors' in analysis
        assert 'dimensions' in analysis
        assert 'patterns' in analysis
        assert analysis['dimensions'] == (3, 3)
        assert len(analysis['colors']) == 9

    def test_performance_tracking(self, orchestrator):
        """Test AI performance statistics."""
        # Simulate some decisions
        orchestrator.ai_decisions = [
            {'timestamp': '2024-01-01', 'action': 'ACTION1', 'success': True},
            {'timestamp': '2024-01-01', 'action': 'ACTION2', 'success': False}
        ]

        stats = orchestrator.get_performance_stats()

        assert stats['total_decisions'] == 2
        assert stats['success_rate'] == 0.5
        assert 'action_distribution' in stats


class TestVisionCoordinator:
    """Test vision coordination system."""

    @pytest.fixture
    def vision_coordinator(self):
        """Create vision coordinator."""
        return VisionCoordinator("test.db")

    @pytest.mark.asyncio
    async def test_optimal_coordinates_selection(self, vision_coordinator):
        """Test vision-guided coordinate selection."""
        frame = [[1, 2, 3], [4, 0, 6], [7, 8, 9]]
        game_id = "test_game"

        x, y = await vision_coordinator.get_optimal_coordinates(frame, game_id)

        assert 0 <= x < 3
        assert 0 <= y < 3
        assert isinstance(x, int)
        assert isinstance(y, int)

    def test_object_detection(self, vision_coordinator):
        """Test basic object detection."""
        frame = [[1, 1, 2], [1, 0, 2], [3, 3, 3]]

        objects = vision_coordinator._detect_objects(frame)

        assert isinstance(objects, list)
        for obj in objects:
            assert 'x' in obj
            assert 'y' in obj
            assert 'color' in obj

    def test_attention_mechanism(self, vision_coordinator):
        """Test attention-based coordinate selection."""
        frame = [[0, 1, 0], [1, 2, 1], [0, 1, 0]]

        x, y = vision_coordinator._attention_based_selection(frame)

        assert 0 <= x < 3
        assert 0 <= y < 3
        # Center should have highest attention (color 2)
        assert (x, y) == (1, 1)

    def test_coordinate_success_tracking(self, vision_coordinator):
        """Test coordinate success tracking."""
        vision_coordinator.update_coordinate_success(1, 1, "test_game", True, 10.0)
        vision_coordinator.update_coordinate_success(2, 2, "test_game", False, -5.0)

        stats = vision_coordinator.get_performance_stats()

        assert stats['total_attempts'] == 2
        assert stats['success_rate'] == 0.5


class TestPatternActionSelector:
    """Test pattern-based action selection."""

    @pytest.fixture
    def mock_db(self):
        """Mock database with pattern data."""
        db = Mock()
        db.get_action_patterns.return_value = [
            {
                'pattern_data': {'colors': [1, 2, 3], 'dimensions': (3, 3)},
                'success_action': 'ACTION1',
                'confidence': 0.8,
                'success_rate': 0.7
            },
            {
                'pattern_data': {'colors': [4, 5, 6], 'dimensions': (3, 3)},
                'success_action': 'ACTION2',
                'confidence': 0.6,
                'success_rate': 0.5
            }
        ]
        return db

    @pytest.fixture
    def selector(self, mock_db):
        """Create pattern action selector."""
        return PatternActionSelector(mock_db)

    def test_pattern_matching(self, selector):
        """Test pattern matching functionality."""
        frame_analysis = {
            'colors': [1, 2, 3, 4],
            'dimensions': (3, 3),
            'patterns': ['horizontal_line']
        }

        action, confidence = selector.select_action_from_patterns(
            frame_analysis, Mock(), ["ACTION1", "ACTION2"]
        )

        assert action in ["ACTION1", "ACTION2"]
        assert 0 <= confidence <= 1

    def test_pattern_similarity_calculation(self, selector):
        """Test pattern similarity calculation."""
        pattern1 = {'colors': [1, 2, 3], 'dimensions': (3, 3)}
        pattern2 = {'colors': [1, 2, 4], 'dimensions': (3, 3)}

        similarity = selector._calculate_pattern_similarity(pattern1, pattern2)

        assert 0 <= similarity <= 1
        assert similarity > 0  # Should have some similarity

    def test_performance_stats(self, selector):
        """Test pattern selector performance statistics."""
        stats = selector.get_performance_stats()

        assert isinstance(stats, dict)
        assert 'total_patterns' in stats


class TestKnowledgeIntegrator:
    """Test knowledge integration system."""

    @pytest.fixture
    def mock_db(self):
        """Mock database with knowledge data."""
        db = Mock()
        db.get_similar_games.return_value = [
            {
                'game_id': 'similar_game_1',
                'similarity': 0.9,
                'successful_actions': ['ACTION1', 'ACTION2'],
                'final_score': 85.0
            }
        ]
        db.get_game_knowledge.return_value = {
            'patterns': [{'type': 'rotation', 'confidence': 0.8}],
            'strategies': [{'name': 'corner_focus', 'success_rate': 0.7}]
        }
        return db

    @pytest.fixture
    def knowledge_integration(self, mock_db):
        """Create knowledge integration system."""
        return KnowledgeIntegrator(mock_db)

    @pytest.mark.asyncio
    async def test_knowledge_loading(self, knowledge_integration):
        """Test pre-game knowledge loading."""
        target_game_id = "new_game"
        context = {'frame_analysis': {'colors': [1, 2, 3]}}

        knowledge = await knowledge_integration.load_game_knowledge(target_game_id, context)

        assert isinstance(knowledge, dict)
        assert 'similar_games' in knowledge
        assert 'recommended_actions' in knowledge
        assert 'confidence_scores' in knowledge

    def test_knowledge_extraction(self, knowledge_integration):
        """Test post-game knowledge extraction."""
        game_data = {
            'game_id': 'completed_game',
            'final_score': 90.0,
            'action_sequence': ['ACTION1', 'ACTION6', 'ACTION3'],
            'frame_changes': [True, True, False]
        }

        extracted = knowledge_integration.extract_game_knowledge(game_data)

        assert isinstance(extracted, dict)
        assert 'successful_patterns' in extracted
        assert 'effective_actions' in extracted

    def test_similarity_calculation(self, knowledge_integration):
        """Test game similarity calculation."""
        game1_context = {'colors': [1, 2, 3], 'dimensions': (3, 3)}
        game2_context = {'colors': [1, 2, 4], 'dimensions': (3, 3)}

        similarity = knowledge_integration._calculate_game_similarity(game1_context, game2_context)

        assert 0 <= similarity <= 1


class TestEnhancedCoreGameplay:
    """Test enhanced core gameplay with AI integration."""

    @pytest.fixture
    def mock_session_manager(self):
        """Mock session manager."""
        manager = Mock()
        manager.current_game_id = "test_game"
        manager.send_action = AsyncMock(return_value=Mock(
            frame=[[1, 2], [3, 4]],
            score=100.0,
            available_actions=["ACTION1", "ACTION2", "ACTION3"]
        ))
        manager.db = Mock()
        manager.db.db_path = "test.db"
        return manager

    @pytest.fixture
    def core_gameplay(self, mock_session_manager):
        """Create enhanced core gameplay."""
        return CoreGameplay(mock_session_manager)

    @pytest.mark.asyncio
    async def test_ai_enhanced_game_execution(self, core_gameplay):
        """Test complete AI-enhanced game execution."""
        # Mock the initial game state
        initial_state = Mock()
        initial_state.frame = [[1, 2], [3, 4]]
        initial_state.score = 0.0
        initial_state.available_actions = ["ACTION1", "ACTION2", "ACTION3"]
        initial_state.status = "active"

        with patch.object(core_gameplay.session_manager, 'get_current_state', return_value=initial_state):
            result = await core_gameplay.play_game_ai_enhanced(max_actions=3)

        assert isinstance(result, dict)
        assert 'final_score' in result
        assert 'total_actions' in result
        assert 'ai_performance' in result

    def test_ai_orchestrator_initialization(self, core_gameplay):
        """Test AI orchestrator initialization."""
        assert core_gameplay.ai_orchestrator is not None
        assert core_gameplay.ai_available is True


class TestEnhancedActionHandler:
    """Test enhanced action handler with vision integration."""

    @pytest.fixture
    def mock_session_manager(self):
        """Mock session manager."""
        manager = Mock()
        manager.current_game_id = "test_game"
        manager.send_action = AsyncMock(return_value=Mock(
            frame=[[1, 2], [3, 4]],
            score=100.0
        ))
        manager.db = Mock()
        manager.db.db_path = "test.db"
        return manager

    @pytest.fixture
    def action_handler(self, mock_session_manager):
        """Create enhanced action handler."""
        return ActionHandler(mock_session_manager)

    @pytest.mark.asyncio
    async def test_vision_guided_action6(self, action_handler):
        """Test vision-guided ACTION6 execution."""
        frame = [[1, 2, 3], [4, 0, 6], [7, 8, 9]]

        result = await action_handler.send_action_6(frame=frame)

        assert result is not None
        # Verify that coordinates were selected (either vision-guided or fallback)
        assert action_handler.session_manager.send_action.called

    def test_coordinate_validation(self, action_handler):
        """Test coordinate validation."""
        frame = [[1, 2], [3, 4]]

        # Valid coordinates
        assert action_handler._validate_coordinates(0, 0, frame) is True
        assert action_handler._validate_coordinates(1, 1, frame) is True

        # Invalid coordinates
        assert action_handler._validate_coordinates(-1, 0, frame) is False
        assert action_handler._validate_coordinates(2, 0, frame) is False
        assert action_handler._validate_coordinates(0, 2, frame) is False

    def test_frame_change_detection(self, action_handler):
        """Test frame change detection."""
        old_frame = [[1, 2], [3, 4]]
        new_frame = [[1, 5], [3, 4]]  # One change

        changed, num_changes = action_handler._detect_frame_changes(old_frame, new_frame)

        assert changed is True
        assert num_changes == 1

    @pytest.mark.asyncio
    async def test_smart_action_selection(self, action_handler):
        """Test smart action selection strategies."""
        game_state = Mock()
        game_state.available_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION6"]
        game_state.frame = [[1, 2], [3, 4]]

        # Test different strategies
        for strategy in ["random", "conservative", "vision_guided", "balanced"]:
            action = await action_handler.smart_action_selection(game_state, strategy)
            assert action in game_state.available_actions


class TestIntegrationScenarios:
    """Test integration scenarios between AI components."""

    @pytest.fixture
    def complete_system(self):
        """Set up complete AI-enhanced system."""
        mock_db = Mock()
        mock_db.get_action_patterns.return_value = []
        mock_db.get_similar_games.return_value = []
        mock_db.get_gan_prediction.return_value = None
        mock_db.db_path = "test.db"

        mock_session = Mock()
        mock_session.current_game_id = "integration_test"
        mock_session.db = mock_db
        mock_session.send_action = AsyncMock(return_value=Mock(
            frame=[[1, 2], [3, 4]],
            score=100.0,
            available_actions=["ACTION1", "ACTION2", "ACTION3"]
        ))

        return {
            'orchestrator': AIOrchestrator(mock_db, "integration_test"),
            'vision': VisionCoordinator("test.db"),
            'patterns': PatternActionSelector(mock_db),
            'knowledge': KnowledgeIntegrator(mock_db),
            'gameplay': CoreGameplay(mock_session),
            'actions': ActionHandler(mock_session)
        }

    @pytest.mark.asyncio
    async def test_full_game_simulation(self, complete_system):
        """Test complete game simulation with all AI systems."""
        frame = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        game_state = Mock()
        game_state.frame = frame
        game_state.score = 0.0
        game_state.available_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION6"]
        game_state.status = "active"

        # Test orchestrator decision making
        action, metadata = await complete_system['orchestrator'].select_action(
            frame, game_state, game_state.available_actions
        )

        assert action in game_state.available_actions
        assert isinstance(metadata, dict)

        # Test vision coordinate selection for ACTION6
        if action == "ACTION6":
            x, y = await complete_system['vision'].get_optimal_coordinates(frame, "integration_test")
            assert 0 <= x < 3
            assert 0 <= y < 3

    def test_performance_monitoring_integration(self, complete_system):
        """Test integrated performance monitoring."""
        # Get stats from each component
        orchestrator_stats = complete_system['orchestrator'].get_performance_stats()
        vision_stats = complete_system['vision'].get_performance_stats()
        pattern_stats = complete_system['patterns'].get_performance_stats()

        # Verify all components provide performance data
        assert isinstance(orchestrator_stats, dict)
        assert isinstance(vision_stats, dict)
        assert isinstance(pattern_stats, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])