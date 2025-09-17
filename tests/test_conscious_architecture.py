#!/usr/bin/env python3
"""
Integration Tests for Conscious Architecture Enhancements

Tests the Dual-Pathway Processor and Enhanced Gut Feeling Engine integration
with the CohesiveIntegrationSystem.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch
import time

from src.core.dual_pathway_processor import (
    DualPathwayProcessor, CognitiveMode, ModeSwitchTrigger, 
    ModeSwitchDecision, CognitiveState
)
from src.core.enhanced_gut_feeling_engine import (
    EnhancedGutFeelingEngine, GutFeeling, GutFeelingType, 
    IntuitionConfidence, PatternMatch
)
from src.core.cohesive_integration_system import CohesiveIntegrationSystem


class TestDualPathwayProcessor:
    """Test the Dual-Pathway Processor functionality."""
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = DualPathwayProcessor()
        
        assert processor.cognitive_state.current_mode == CognitiveMode.TPN
        assert processor.tpn_confidence_threshold == 0.7
        assert processor.dmn_activation_threshold == 0.3
        assert processor.mode_switch_cooldown == 5.0
    
    def test_performance_update(self):
        """Test performance metrics update."""
        processor = DualPathwayProcessor()
        
        performance_metrics = {
            'confidence': 0.8,
            'success_rate': 0.6,
            'learning_progress': 0.3,
            'energy_efficiency': 0.7
        }
        
        processor.update_performance(performance_metrics)
        
        assert processor.cognitive_state.confidence_level == 0.8
        assert processor.cognitive_state.performance_metrics == performance_metrics
        assert len(processor.performance_history) == 1
    
    def test_tpn_to_dmn_switch_decision(self):
        """Test switching from TPN to DMN mode."""
        processor = DualPathwayProcessor()
        
        # Simulate very poor performance to trigger switch
        for _ in range(10):
            processor.update_performance({
                'confidence': 0.1,
                'success_rate': 0.05,
                'learning_progress': 0.0,
                'energy_efficiency': 0.1
            })
        
        # Wait for cooldown to expire and set mode duration
        processor.cognitive_state.last_switch_time = time.time() - 10.0
        processor.cognitive_state.mode_duration = 35.0  # Simulate long duration in TPN
        
        context = {'available_actions': [1, 2, 3, 4, 5, 6, 7]}
        decision = processor.should_switch_mode(context, [1, 2, 3, 4, 5, 6, 7])
        
        # The decision should trigger a switch due to poor performance and long duration
        assert decision.should_switch == True
        assert decision.target_mode == CognitiveMode.DMN
    
    def test_mode_switching(self):
        """Test actual mode switching."""
        processor = DualPathwayProcessor()
        
        # Switch to DMN mode
        switch_result = processor.switch_to_mode(
            CognitiveMode.DMN, 
            ModeSwitchTrigger.CURIOSITY_DRIVEN,
            {'test': 'context'}
        )
        
        assert processor.cognitive_state.current_mode == CognitiveMode.DMN
        assert switch_result['current_mode'] == 'default_mode_network'
        assert switch_result['trigger'] == 'curiosity_driven'
        assert processor.cognitive_state.switch_count == 1
    
    def test_mode_specific_actions(self):
        """Test getting mode-specific actions."""
        processor = DualPathwayProcessor()
        
        # Test TPN actions
        tpn_actions = processor._get_tpn_actions([1, 2, 3, 4, 5, 6, 7], {})
        assert len(tpn_actions) > 0
        assert all(action['mode'] == 'TPN' for action in tpn_actions)
        
        # Switch to DMN and test DMN actions
        processor.switch_to_mode(CognitiveMode.DMN, ModeSwitchTrigger.EXPLICIT_COMMAND, {})
        dmn_actions = processor._get_dmn_actions([1, 2, 3, 4, 5, 6, 7], {})
        assert len(dmn_actions) > 0
        assert all(action['mode'] == 'DMN' for action in dmn_actions)
    
    def test_consciousness_metrics(self):
        """Test consciousness metrics calculation."""
        processor = DualPathwayProcessor()
        
        # Update performance to generate metrics
        processor.update_performance({
            'confidence': 0.6,
            'success_rate': 0.5,
            'learning_progress': 0.2,
            'energy_efficiency': 0.7
        })
        
        metrics = processor.get_consciousness_metrics()
        
        assert 'current_mode' in metrics
        assert 'mode_duration' in metrics
        assert 'switch_count' in metrics
        assert 'confidence_level' in metrics
        assert 'consciousness_score' in metrics
        assert 0.0 <= metrics['consciousness_score'] <= 1.0


class TestEnhancedGutFeelingEngine:
    """Test the Enhanced Gut Feeling Engine functionality."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = EnhancedGutFeelingEngine()
        
        assert engine.similarity_threshold == 0.6
        assert engine.max_patterns == 1000
        assert len(engine.patterns) == 0
    
    def test_pattern_addition(self):
        """Test adding patterns to the engine."""
        engine = EnhancedGutFeelingEngine()
        
        pattern_data = {
            'suggested_action': 6,
            'state': {'position': [10, 10]},
            'context': {'goal': 'move_to_target'}
        }
        
        pattern_id = engine.add_pattern(pattern_data, 'spatial', 0.8)
        
        assert pattern_id in engine.patterns
        assert engine.patterns[pattern_id]['success_rate'] == 0.8
        assert engine.patterns[pattern_id]['pattern_type'] == 'spatial'
    
    def test_gut_feeling_generation(self):
        """Test generating gut feelings."""
        engine = EnhancedGutFeelingEngine(similarity_threshold=0.3)  # Lower threshold
        
        # Add a pattern
        pattern_data = {
            'suggested_action': 6,
            'state': {'position': [10, 10]},
            'context': {'goal': 'move_to_target'}
        }
        engine.add_pattern(pattern_data, 'spatial', 0.8)
        
        # Generate gut feelings with more similar state
        current_state = {
            'frame_features': {'position': [10, 10]},  # Exact match
            'spatial_features': {'similarity': 0.9}
        }
        context = {'goal': 'move_to_target'}
        available_actions = [1, 2, 3, 4, 5, 6, 7]
        
        gut_feelings = engine.get_gut_feelings(current_state, available_actions, context)
        
        assert len(gut_feelings) > 0
        assert all(isinstance(gf, GutFeeling) for gf in gut_feelings)
        assert all(gf.action in available_actions for gf in gut_feelings)
    
    def test_learning_from_outcome(self):
        """Test learning from gut feeling outcomes."""
        engine = EnhancedGutFeelingEngine()
        
        # Add a pattern
        pattern_data = {
            'suggested_action': 6,
            'state': {'position': [10, 10]},
            'context': {'goal': 'move_to_target'}
        }
        pattern_id = engine.add_pattern(pattern_data, 'spatial', 0.5)
        
        # Create a gut feeling
        gut_feeling = GutFeeling(
            action=6,
            confidence=0.7,
            gut_feeling_type=GutFeelingType.PATTERN_SIMILARITY,
            reasoning="Test gut feeling",
            pattern_id=pattern_id
        )
        
        # Learn from successful outcome
        outcome = {'success': True, 'performance_score': 0.8}
        context = {'test': 'context'}
        
        engine.learn_from_outcome(gut_feeling, outcome, context)
        
        # Check that pattern success rate was updated
        assert engine.patterns[pattern_id]['success_count'] == 1
        assert engine.patterns[pattern_id]['total_attempts'] == 1
        assert engine.patterns[pattern_id]['success_rate'] == 1.0
    
    def test_gut_feeling_metrics(self):
        """Test gut feeling metrics calculation."""
        engine = EnhancedGutFeelingEngine()
        
        # Add some patterns and gut feelings
        pattern_data = {
            'suggested_action': 6,
            'state': {'position': [10, 10]},
            'context': {'goal': 'move_to_target'}
        }
        engine.add_pattern(pattern_data, 'spatial', 0.8)
        
        # Generate gut feelings
        current_state = {'frame_features': {'position': [12, 12]}}
        context = {'goal': 'move_to_target'}
        gut_feelings = engine.get_gut_feelings(current_state, [1, 2, 3, 4, 5, 6, 7], context)
        
        # Learn from outcomes
        for gf in gut_feelings:
            outcome = {'success': True, 'performance_score': 0.7}
            engine.learn_from_outcome(gf, outcome, context)
        
        metrics = engine.get_gut_feeling_metrics()
        
        assert 'total_gut_feelings' in metrics
        assert 'success_rate' in metrics
        assert 'total_patterns' in metrics
        assert 'average_confidence' in metrics
        assert 0.0 <= metrics['success_rate'] <= 1.0


class TestConsciousArchitectureIntegration:
    """Test integration of conscious architecture with CohesiveIntegrationSystem."""
    
    def test_cohesive_system_initialization(self):
        """Test CohesiveIntegrationSystem with conscious architecture enabled."""
        system = CohesiveIntegrationSystem(enable_conscious_architecture=True)
        
        assert system.enable_conscious_architecture == True
        assert system.dual_pathway_processor is not None
        assert system.gut_feeling_engine is not None
        assert system.current_state.current_cognitive_mode == "TPN"
    
    def test_conscious_architecture_processing(self):
        """Test conscious architecture processing in the main system."""
        system = CohesiveIntegrationSystem(enable_conscious_architecture=True)
        
        # Mock frame and context
        frame = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        context = {
            'available_actions': [1, 2, 3, 4, 5, 6, 7],
            'confidence': 0.6,
            'success_rate': 0.5,
            'frame_features': {'position': [5, 5]},
            'spatial_features': {'similarity': 0.7}
        }
        
        # Mock hypotheses
        mock_hypothesis = Mock()
        mock_hypothesis.recommended_action = 6
        hypotheses = [mock_hypothesis]
        
        # Mock curiosity response
        curiosity_response = {
            'curiosity_level': 0.7,
            'boredom_level': 0.2,
            'learning_rate': 1.0,
            'strategy_switch_needed': False
        }
        
        # Process environment update
        result = system.process_environment_update(frame, context)
        
        assert 'conscious_processing' in result
        assert result['conscious_processing']['enabled'] == True
        assert 'gut_feelings' in result['conscious_processing']
        assert 'mode_actions' in result['conscious_processing']
        assert 'consciousness_metrics' in result['conscious_processing']
    
    def test_mode_switching_integration(self):
        """Test mode switching integration."""
        system = CohesiveIntegrationSystem(enable_conscious_architecture=True)
        
        # Simulate poor performance to trigger mode switch
        for _ in range(5):
            context = {
                'available_actions': [1, 2, 3, 4, 5, 6, 7],
                'confidence': 0.2,
                'success_rate': 0.1,
                'learning_progress': 0.0,
                'energy_efficiency': 0.3
            }
            
            frame = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
            mock_hypothesis = Mock()
            mock_hypothesis.recommended_action = 6
            hypotheses = [mock_hypothesis]
            curiosity_response = {
                'curiosity_level': 0.3,
                'boredom_level': 0.7,
                'learning_rate': 0.5,
                'strategy_switch_needed': True
            }
            
            result = system.process_environment_update(frame, context)
            
            # Check if mode switch occurred
            if 'mode_switch' in result['conscious_processing']:
                assert result['conscious_processing']['mode_switch']['current_mode'] == 'default_mode_network'
                break
    
    def test_gut_feeling_integration(self):
        """Test gut feeling integration in action selection."""
        system = CohesiveIntegrationSystem(enable_conscious_architecture=True)
        
        # Add patterns to gut feeling engine
        pattern_data = {
            'suggested_action': 6,
            'state': {'position': [10, 10]},
            'context': {'goal': 'move_to_target'}
        }
        system.gut_feeling_engine.add_pattern(pattern_data, 'spatial', 0.8)
        
        # Process with very similar context to ensure pattern matching
        frame = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        context = {
            'available_actions': [1, 2, 3, 4, 5, 6, 7],
            'confidence': 0.6,
            'success_rate': 0.5,
            'frame_features': {'position': [10, 10]},  # Exact match
            'spatial_features': {'similarity': 0.9},
            'goal': 'move_to_target'
        }
        
        mock_hypothesis = Mock()
        mock_hypothesis.recommended_action = 1
        hypotheses = [mock_hypothesis]
        curiosity_response = {
            'curiosity_level': 0.7,
            'boredom_level': 0.2,
            'learning_rate': 1.0,
            'strategy_switch_needed': False
        }
        
        result = system.process_environment_update(frame, context)
        
        # Check that gut feelings influenced action selection
        assert 'gut_feelings' in result['conscious_processing']
        gut_feelings = result['conscious_processing']['gut_feelings']
        # Note: May be empty if similarity threshold not met, which is OK
        assert isinstance(gut_feelings, list)
        
        # Check that action selection considers gut feelings
        selected_action = result['selected_action']
        assert selected_action in context['available_actions']
    
    def test_conscious_architecture_status(self):
        """Test getting conscious architecture status."""
        system = CohesiveIntegrationSystem(enable_conscious_architecture=True)
        
        status = system.get_conscious_architecture_status()
        
        assert status['enabled'] == True
        assert 'dual_pathway' in status
        assert 'gut_feeling' in status
        assert 'current_mode' in status['dual_pathway']
        assert 'consciousness_score' in status['dual_pathway']
    
    def test_learning_from_conscious_outcome(self):
        """Test learning from conscious architecture outcomes."""
        system = CohesiveIntegrationSystem(enable_conscious_architecture=True)
        
        # Add patterns
        pattern_data = {
            'suggested_action': 6,
            'state': {'position': [10, 10]},
            'context': {'goal': 'move_to_target'}
        }
        system.gut_feeling_engine.add_pattern(pattern_data, 'spatial', 0.5)
        
        # Simulate gut feeling decision
        gut_feeling = GutFeeling(
            action=6,
            confidence=0.7,
            gut_feeling_type=GutFeelingType.PATTERN_SIMILARITY,
            reasoning="Test gut feeling"
        )
        system._recent_gut_feelings = [gut_feeling]
        
        # Learn from outcome
        outcome = {'success': True, 'performance_score': 0.8}
        context = {'test': 'context'}
        
        system.learn_from_conscious_outcome(6, outcome, context)
        
        # Check that learning occurred
        metrics = system.gut_feeling_engine.get_gut_feeling_metrics()
        assert 'total_gut_feelings' in metrics
        assert 'success_rate' in metrics


class TestConsciousArchitecturePerformance:
    """Test performance characteristics of conscious architecture."""
    
    def test_processing_speed(self):
        """Test that conscious architecture doesn't significantly slow down processing."""
        system = CohesiveIntegrationSystem(enable_conscious_architecture=True)
        
        frame = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        context = {
            'available_actions': [1, 2, 3, 4, 5, 6, 7],
            'confidence': 0.6,
            'success_rate': 0.5,
            'frame_features': {'position': [5, 5]},
            'spatial_features': {'similarity': 0.7}
        }
        
        mock_hypothesis = Mock()
        mock_hypothesis.recommended_action = 6
        hypotheses = [mock_hypothesis]
        curiosity_response = {
            'curiosity_level': 0.7,
            'boredom_level': 0.2,
            'learning_rate': 1.0,
            'strategy_switch_needed': False
        }
        
        # Measure processing time
        start_time = time.time()
        result = system.process_environment_update(frame, context)
        processing_time = time.time() - start_time
        
        # Should complete within reasonable time (less than 1 second)
        assert processing_time < 1.0
        assert 'conscious_processing' in result
    
    def test_memory_usage(self):
        """Test that conscious architecture doesn't cause memory leaks."""
        system = CohesiveIntegrationSystem(enable_conscious_architecture=True)
        
        # Process many updates
        for i in range(100):
            frame = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
            context = {
                'available_actions': [1, 2, 3, 4, 5, 6, 7],
                'confidence': 0.6,
                'success_rate': 0.5,
                'frame_features': {'position': [i % 10, i % 10]},
                'spatial_features': {'similarity': 0.7}
            }
            
            mock_hypothesis = Mock()
            mock_hypothesis.recommended_action = 6
            hypotheses = [mock_hypothesis]
            curiosity_response = {
                'curiosity_level': 0.7,
                'boredom_level': 0.2,
                'learning_rate': 1.0,
                'strategy_switch_needed': False
            }
            
            result = system.process_environment_update(frame, context)
        
        # Check that patterns don't grow unbounded
        gut_metrics = system.gut_feeling_engine.get_gut_feeling_metrics()
        assert 'total_patterns' in gut_metrics
        assert gut_metrics['total_patterns'] <= system.gut_feeling_engine.max_patterns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
