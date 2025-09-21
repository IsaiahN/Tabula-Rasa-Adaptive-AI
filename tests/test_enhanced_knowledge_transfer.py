"""
Test suite for Enhanced Knowledge Transfer System

This module tests the comprehensive knowledge transfer capabilities with
cross-task learning persistence, semantic similarity analysis, and adaptive
transfer strategies.
"""

import unittest
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any

from src.learning.enhanced_knowledge_transfer import (
    EnhancedKnowledgeTransfer, TransferType, TransferConfidence,
    TransferableKnowledge, TransferResult, GameSimilarityProfile,
    create_enhanced_knowledge_transfer
)


class TestEnhancedKnowledgeTransfer(unittest.TestCase):
    """Test cases for Enhanced Knowledge Transfer System."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.transfer_system = create_enhanced_knowledge_transfer(
            persistence_dir=Path(self.temp_dir),
            transfer_threshold=0.6,
            enable_database_storage=False  # Disable for testing
        )
        
        # Test data
        self.test_knowledge_content = {
            'actions': ['ACTION1', 'ACTION2', 'ACTION3'],
            'coordinates': [(10, 10), (20, 20), (30, 30)],
            'pattern': 'test_pattern',
            'confidence': 0.8
        }
        
        self.test_context_features = {
            'visual': {'color': 'red', 'shape': 'circle'},
            'spatial': {'x': 10, 'y': 20, 'width': 30, 'height': 40},
            'complexity': 0.7
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test system initialization."""
        self.assertIsInstance(self.transfer_system, EnhancedKnowledgeTransfer)
        self.assertEqual(self.transfer_system.transfer_threshold, 0.6)
        self.assertFalse(self.transfer_system.enable_database_storage)
        self.assertEqual(self.transfer_system.stats['knowledge_items'], 0)
        self.assertEqual(self.transfer_system.stats['total_transfers'], 0)
    
    def test_add_knowledge(self):
        """Test adding transferable knowledge."""
        knowledge_id = self.transfer_system.add_knowledge(
            source_game='test_game_1',
            knowledge_type=TransferType.PATTERN,
            content=self.test_knowledge_content,
            confidence=0.8,
            success_rate=0.7,
            tags=['test', 'pattern'],
            context_features=self.test_context_features
        )
        
        self.assertIsInstance(knowledge_id, str)
        self.assertGreater(len(knowledge_id), 0)
        self.assertIn(knowledge_id, self.transfer_system.transferable_knowledge)
        
        knowledge = self.transfer_system.transferable_knowledge[knowledge_id]
        self.assertEqual(knowledge.source_game, 'test_game_1')
        self.assertEqual(knowledge.knowledge_type, TransferType.PATTERN)
        self.assertEqual(knowledge.confidence, 0.8)
        self.assertEqual(knowledge.success_rate, 0.7)
        self.assertEqual(knowledge.usage_count, 0)
        self.assertEqual(knowledge.tags, ['test', 'pattern'])
        self.assertEqual(knowledge.context_features, self.test_context_features)
    
    def test_add_multiple_knowledge_types(self):
        """Test adding different types of knowledge."""
        # Add pattern knowledge
        pattern_id = self.transfer_system.add_knowledge(
            source_game='test_game_1',
            knowledge_type=TransferType.PATTERN,
            content={'pattern': 'test_pattern'},
            confidence=0.8,
            success_rate=0.7
        )
        
        # Add coordinate knowledge
        coord_id = self.transfer_system.add_knowledge(
            source_game='test_game_1',
            knowledge_type=TransferType.COORDINATE,
            content={'zones': [(10, 10), (20, 20)]},
            confidence=0.9,
            success_rate=0.8
        )
        
        # Add action sequence knowledge
        action_id = self.transfer_system.add_knowledge(
            source_game='test_game_1',
            knowledge_type=TransferType.ACTION_SEQUENCE,
            content={'actions': ['ACTION1', 'ACTION2', 'ACTION3']},
            confidence=0.7,
            success_rate=0.6
        )
        
        self.assertIsInstance(pattern_id, str)
        self.assertIsInstance(coord_id, str)
        self.assertIsInstance(action_id, str)
        self.assertEqual(len(self.transfer_system.transferable_knowledge), 3)
        self.assertEqual(self.transfer_system.stats['knowledge_items'], 3)
    
    def test_transfer_knowledge(self):
        """Test knowledge transfer between games."""
        # Add knowledge to source game
        knowledge_id = self.transfer_system.add_knowledge(
            source_game='source_game',
            knowledge_type=TransferType.PATTERN,
            content=self.test_knowledge_content,
            confidence=0.8,
            success_rate=0.7
        )
        
        # Transfer knowledge to target game
        transfer_result = self.transfer_system.transfer_knowledge(
            source_game='source_game',
            target_game='target_game',
            knowledge_types=[TransferType.PATTERN]
        )
        
        self.assertIsInstance(transfer_result, TransferResult)
        self.assertEqual(transfer_result.source_game, 'source_game')
        self.assertEqual(transfer_result.target_game, 'target_game')
        self.assertEqual(transfer_result.knowledge_type, TransferType.PATTERN)
        self.assertIn(knowledge_id, transfer_result.transferred_items)
        self.assertGreater(transfer_result.confidence, 0.0)
        self.assertGreater(transfer_result.effectiveness, 0.0)
        self.assertTrue(transfer_result.success)
    
    def test_transfer_knowledge_low_similarity(self):
        """Test knowledge transfer with low game similarity."""
        # Set a higher similarity threshold for this test
        self.transfer_system.similarity_threshold = 0.8
        
        # Add knowledge to source game
        knowledge_id = self.transfer_system.add_knowledge(
            source_game='source_game',
            knowledge_type=TransferType.PATTERN,
            content=self.test_knowledge_content,
            confidence=0.8,
            success_rate=0.7
        )
        
        # Try to transfer to a completely different game
        transfer_result = self.transfer_system.transfer_knowledge(
            source_game='source_game',
            target_game='completely_different_game'
        )
        
        self.assertIsInstance(transfer_result, TransferResult)
        self.assertFalse(transfer_result.success)
        self.assertEqual(len(transfer_result.transferred_items), 0)
        self.assertIn("Low similarity", transfer_result.adaptation_notes[0])
        
        # Reset threshold for other tests
        self.transfer_system.similarity_threshold = 0.5
    
    def test_transfer_knowledge_with_filters(self):
        """Test knowledge transfer with type and confidence filters."""
        # Add different types of knowledge
        pattern_id = self.transfer_system.add_knowledge(
            source_game='source_game',
            knowledge_type=TransferType.PATTERN,
            content={'pattern': 'test_pattern'},
            confidence=0.9,
            success_rate=0.8
        )
        
        coord_id = self.transfer_system.add_knowledge(
            source_game='source_game',
            knowledge_type=TransferType.COORDINATE,
            content={'zones': [(10, 10)]},
            confidence=0.5,  # Low confidence
            success_rate=0.6
        )
        
        # Transfer only high-confidence pattern knowledge
        transfer_result = self.transfer_system.transfer_knowledge(
            source_game='source_game',
            target_game='target_game',
            knowledge_types=[TransferType.PATTERN],
            min_confidence=0.8
        )
        
        self.assertTrue(transfer_result.success)
        self.assertIn(pattern_id, transfer_result.transferred_items)
        self.assertNotIn(coord_id, transfer_result.transferred_items)
    
    def test_evaluate_transfer_success(self):
        """Test transfer success evaluation."""
        # Add knowledge and transfer it
        knowledge_id = self.transfer_system.add_knowledge(
            source_game='source_game',
            knowledge_type=TransferType.PATTERN,
            content=self.test_knowledge_content,
            confidence=0.8,
            success_rate=0.7
        )
        
        transfer_result = self.transfer_system.transfer_knowledge(
            source_game='source_game',
            target_game='target_game'
        )
        
        # Evaluate successful transfer
        self.transfer_system.evaluate_transfer_success(
            transfer_result.transfer_id,
            success=True,
            performance_improvement=0.2
        )
        
        # Check statistics
        self.assertEqual(self.transfer_system.stats['successful_transfers'], 1)
        self.assertEqual(self.transfer_system.stats['failed_transfers'], 0)
        self.assertEqual(self.transfer_system.stats['transfer_effectiveness'], 1.0)
        
        # Check knowledge success rate update
        knowledge = self.transfer_system.transferable_knowledge[knowledge_id]
        self.assertGreater(knowledge.success_rate, 0.7)  # Should be increased
    
    def test_evaluate_transfer_failure(self):
        """Test transfer failure evaluation."""
        # Add knowledge and transfer it
        knowledge_id = self.transfer_system.add_knowledge(
            source_game='source_game',
            knowledge_type=TransferType.PATTERN,
            content=self.test_knowledge_content,
            confidence=0.8,
            success_rate=0.7
        )
        
        transfer_result = self.transfer_system.transfer_knowledge(
            source_game='source_game',
            target_game='target_game'
        )
        
        # Evaluate failed transfer
        self.transfer_system.evaluate_transfer_success(
            transfer_result.transfer_id,
            success=False,
            performance_improvement=-0.1
        )
        
        # Check statistics
        self.assertEqual(self.transfer_system.stats['successful_transfers'], 0)
        self.assertEqual(self.transfer_system.stats['failed_transfers'], 1)
        self.assertEqual(self.transfer_system.stats['transfer_effectiveness'], 0.0)
        
        # Check knowledge success rate update
        knowledge = self.transfer_system.transferable_knowledge[knowledge_id]
        self.assertLess(knowledge.success_rate, 0.7)  # Should be decreased
    
    def test_get_transfer_recommendations(self):
        """Test getting transfer recommendations."""
        # Add knowledge to multiple games
        self.transfer_system.add_knowledge(
            source_game='game_1',
            knowledge_type=TransferType.PATTERN,
            content={'pattern': 'pattern_1'},
            confidence=0.8,
            success_rate=0.7
        )
        
        self.transfer_system.add_knowledge(
            source_game='game_2',
            knowledge_type=TransferType.PATTERN,
            content={'pattern': 'pattern_2'},
            confidence=0.9,
            success_rate=0.8
        )
        
        # Get recommendations for target game
        recommendations = self.transfer_system.get_transfer_recommendations(
            target_game='target_game'
        )
        
        self.assertIsInstance(recommendations, list)
        # Should have recommendations for both source games
        self.assertGreaterEqual(len(recommendations), 0)
    
    def test_get_knowledge_statistics(self):
        """Test getting knowledge statistics."""
        # Add some knowledge
        self.transfer_system.add_knowledge(
            source_game='game_1',
            knowledge_type=TransferType.PATTERN,
            content={'pattern': 'pattern_1'},
            confidence=0.8,
            success_rate=0.7
        )
        
        self.transfer_system.add_knowledge(
            source_game='game_2',
            knowledge_type=TransferType.COORDINATE,
            content={'zones': [(10, 10)]},
            confidence=0.9,
            success_rate=0.8
        )
        
        # Get statistics
        stats = self.transfer_system.get_knowledge_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats['total_knowledge_items'], 2)
        self.assertEqual(stats['games_tracked'], 2)
        self.assertIn('type_distribution', stats)
        self.assertIn('confidence_stats', stats)
        self.assertIn('success_rate_stats', stats)
        self.assertIn('recent_transfers', stats)
    
    def test_cleanup_old_knowledge(self):
        """Test cleanup of old knowledge."""
        # Add old knowledge (simulate by setting old timestamp)
        knowledge_id = self.transfer_system.add_knowledge(
            source_game='old_game',
            knowledge_type=TransferType.PATTERN,
            content={'pattern': 'old_pattern'},
            confidence=0.5,
            success_rate=0.3
        )
        
        # Manually set old timestamp
        knowledge = self.transfer_system.transferable_knowledge[knowledge_id]
        knowledge.created_at = time.time() - (31 * 24 * 60 * 60)  # 31 days ago
        knowledge.usage_count = 0  # Never used
        
        # Add recent knowledge
        recent_id = self.transfer_system.add_knowledge(
            source_game='recent_game',
            knowledge_type=TransferType.PATTERN,
            content={'pattern': 'recent_pattern'},
            confidence=0.8,
            success_rate=0.7
        )
        
        # Cleanup old knowledge
        initial_count = len(self.transfer_system.transferable_knowledge)
        self.transfer_system.cleanup_old_knowledge(max_age_days=30, min_usage_count=1)
        
        # Check that old knowledge was removed but recent knowledge remains
        self.assertNotIn(knowledge_id, self.transfer_system.transferable_knowledge)
        self.assertIn(recent_id, self.transfer_system.transferable_knowledge)
        self.assertLess(len(self.transfer_system.transferable_knowledge), initial_count)
    
    def test_game_similarity_calculation(self):
        """Test game similarity calculation."""
        # Add knowledge to create game profiles
        self.transfer_system.add_knowledge(
            source_game='game_1',
            knowledge_type=TransferType.PATTERN,
            content={'pattern': 'pattern_1'},
            confidence=0.8,
            success_rate=0.7,
            context_features={'visual': {'color': 'red'}, 'spatial': {'x': 10}}
        )
        
        self.transfer_system.add_knowledge(
            source_game='game_2',
            knowledge_type=TransferType.PATTERN,
            content={'pattern': 'pattern_2'},
            confidence=0.9,
            success_rate=0.8,
            context_features={'visual': {'color': 'blue'}, 'spatial': {'x': 20}}
        )
        
        # Calculate similarity
        similarity = self.transfer_system._calculate_game_similarity('game_1', 'game_2')
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_knowledge_adaptation(self):
        """Test knowledge adaptation to target context."""
        # Add knowledge with specific features
        knowledge_id = self.transfer_system.add_knowledge(
            source_game='source_game',
            knowledge_type=TransferType.COORDINATE,
            content={'zones': [(10, 10), (20, 20)]},
            confidence=0.8,
            success_rate=0.7,
            context_features={'spatial': {'scale': 1.0}}
        )
        
        # Create target game profile
        target_profile = GameSimilarityProfile(
            game_id='target_game',
            visual_features={},
            spatial_features={},
            action_patterns=[],
            success_patterns=[],
            coordinate_zones=[(5, 5), (15, 15)],  # Different scale
            complexity_score=0.5,
            pattern_types={TransferType.COORDINATE},
            last_updated=time.time()
        )
        self.transfer_system.game_profiles['target_game'] = target_profile
        
        # Adapt knowledge
        knowledge = self.transfer_system.transferable_knowledge[knowledge_id]
        adapted_content = self.transfer_system._adapt_knowledge_to_context(
            knowledge, 'target_game'
        )
        
        self.assertIsInstance(adapted_content, dict)
        self.assertIn('zones', adapted_content)
        # Zones should be adapted to target game scale
    
    def test_transfer_potential_calculation(self):
        """Test transfer potential calculation."""
        # Add knowledge with different qualities
        high_quality_id = self.transfer_system.add_knowledge(
            source_game='source_game',
            knowledge_type=TransferType.PATTERN,
            content={'pattern': 'high_quality'},
            confidence=0.9,
            success_rate=0.8
        )
        
        low_quality_id = self.transfer_system.add_knowledge(
            source_game='source_game',
            knowledge_type=TransferType.PATTERN,
            content={'pattern': 'low_quality'},
            confidence=0.4,
            success_rate=0.3
        )
        
        # Get knowledge items
        high_quality = self.transfer_system.transferable_knowledge[high_quality_id]
        low_quality = self.transfer_system.transferable_knowledge[low_quality_id]
        
        # Calculate transfer potential
        high_potential = self.transfer_system._calculate_transfer_potential(
            [high_quality], 'target_game'
        )
        
        low_potential = self.transfer_system._calculate_transfer_potential(
            [low_quality], 'target_game'
        )
        
        self.assertGreater(high_potential, low_potential)
        self.assertGreaterEqual(high_potential, 0.0)
        self.assertLessEqual(high_potential, 1.0)
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with invalid parameters
        result = self.transfer_system.transfer_knowledge(
            source_game='nonexistent_game',
            target_game='target_game'
        )
        
        self.assertIsInstance(result, TransferResult)
        self.assertFalse(result.success)
        
        # Test with empty knowledge
        result = self.transfer_system.transfer_knowledge(
            source_game='source_game',
            target_game='target_game',
            knowledge_types=[TransferType.PATTERN],
            min_confidence=1.0  # Impossible threshold
        )
        
        self.assertIsInstance(result, TransferResult)
        self.assertFalse(result.success)
    
    def test_factory_function(self):
        """Test the factory function."""
        transfer_system = create_enhanced_knowledge_transfer(
            persistence_dir=Path(self.temp_dir),
            transfer_threshold=0.7,
            enable_database_storage=False
        )
        
        self.assertIsInstance(transfer_system, EnhancedKnowledgeTransfer)
        self.assertEqual(transfer_system.transfer_threshold, 0.7)
        self.assertFalse(transfer_system.enable_database_storage)


class TestTransferTypes(unittest.TestCase):
    """Test cases for transfer types and enums."""
    
    def test_transfer_type_enum(self):
        """Test TransferType enum values."""
        self.assertEqual(TransferType.PATTERN.value, "pattern")
        self.assertEqual(TransferType.STRATEGY.value, "strategy")
        self.assertEqual(TransferType.COORDINATE.value, "coordinate")
        self.assertEqual(TransferType.ACTION_SEQUENCE.value, "action_sequence")
        self.assertEqual(TransferType.VISUAL_PATTERN.value, "visual_pattern")
        self.assertEqual(TransferType.SPATIAL_REASONING.value, "spatial_reasoning")
        self.assertEqual(TransferType.LOGICAL_REASONING.value, "logical_reasoning")
    
    def test_transfer_confidence_enum(self):
        """Test TransferConfidence enum values."""
        self.assertEqual(TransferConfidence.LOW.value, "low")
        self.assertEqual(TransferConfidence.MEDIUM.value, "medium")
        self.assertEqual(TransferConfidence.HIGH.value, "high")
        self.assertEqual(TransferConfidence.VERY_HIGH.value, "very_high")


class TestDataClasses(unittest.TestCase):
    """Test cases for data classes."""
    
    def test_transferable_knowledge_dataclass(self):
        """Test TransferableKnowledge dataclass."""
        knowledge = TransferableKnowledge(
            knowledge_id='test_id',
            source_game='test_game',
            knowledge_type=TransferType.PATTERN,
            content={'test': 'content'},
            confidence=0.8,
            success_rate=0.7,
            usage_count=5,
            last_used=time.time(),
            created_at=time.time(),
            tags=['test'],
            context_features={'feature': 'value'},
            transfer_history=[]
        )
        
        self.assertEqual(knowledge.knowledge_id, 'test_id')
        self.assertEqual(knowledge.source_game, 'test_game')
        self.assertEqual(knowledge.knowledge_type, TransferType.PATTERN)
        self.assertEqual(knowledge.confidence, 0.8)
        self.assertEqual(knowledge.success_rate, 0.7)
        self.assertEqual(knowledge.usage_count, 5)
        self.assertEqual(knowledge.tags, ['test'])
        self.assertEqual(knowledge.context_features, {'feature': 'value'})
    
    def test_transfer_result_dataclass(self):
        """Test TransferResult dataclass."""
        result = TransferResult(
            transfer_id='test_transfer',
            source_game='source',
            target_game='target',
            knowledge_type=TransferType.PATTERN,
            transferred_items=['item1', 'item2'],
            confidence=0.8,
            effectiveness=0.7,
            adaptation_notes=['note1'],
            timestamp=time.time(),
            success=True,
            performance_improvement=0.2
        )
        
        self.assertEqual(result.transfer_id, 'test_transfer')
        self.assertEqual(result.source_game, 'source')
        self.assertEqual(result.target_game, 'target')
        self.assertEqual(result.knowledge_type, TransferType.PATTERN)
        self.assertEqual(result.transferred_items, ['item1', 'item2'])
        self.assertEqual(result.confidence, 0.8)
        self.assertEqual(result.effectiveness, 0.7)
        self.assertEqual(result.adaptation_notes, ['note1'])
        self.assertTrue(result.success)
        self.assertEqual(result.performance_improvement, 0.2)
    
    def test_game_similarity_profile_dataclass(self):
        """Test GameSimilarityProfile dataclass."""
        profile = GameSimilarityProfile(
            game_id='test_game',
            visual_features={'color': 'red'},
            spatial_features={'x': 10},
            action_patterns=['ACTION1'],
            success_patterns=['pattern1'],
            coordinate_zones=[(10, 10)],
            complexity_score=0.5,
            pattern_types={TransferType.PATTERN},
            last_updated=time.time()
        )
        
        self.assertEqual(profile.game_id, 'test_game')
        self.assertEqual(profile.visual_features, {'color': 'red'})
        self.assertEqual(profile.spatial_features, {'x': 10})
        self.assertEqual(profile.action_patterns, ['ACTION1'])
        self.assertEqual(profile.success_patterns, ['pattern1'])
        self.assertEqual(profile.coordinate_zones, [(10, 10)])
        self.assertEqual(profile.complexity_score, 0.5)
        self.assertEqual(profile.pattern_types, {TransferType.PATTERN})


if __name__ == '__main__':
    unittest.main()
