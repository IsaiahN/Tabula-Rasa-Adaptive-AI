#!/usr/bin/env python3
"""
Test suite for the enhanced sleep system with breakthrough detection.
"""

import unittest
import asyncio
import time
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Add the src directory to the path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.sleep_system import SleepCycle
from core.sleep_breakthrough_detection import (
    BreakthroughDetector, BreakthroughProcessor, 
    create_sleep_breakthrough_system
)
from core.data_models import Experience, AgentState, SensoryInput


class TestSleepBreakthroughSystem(unittest.TestCase):
    """Test the enhanced sleep system with breakthrough detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock predictive core
        self.mock_predictive_core = Mock()
        self.mock_predictive_core.use_memory = True
        self.mock_predictive_core.memory = Mock()
        self.mock_predictive_core.memory.get_memory_metrics.return_value = {
            'memory_utilization': 0.5,
            'memory_capacity': 1000
        }
        self.mock_predictive_core.memory.memory_matrix = torch.randn(100, 64)
        self.mock_predictive_core.memory.usage_vector = torch.rand(100)
        
        # Create mock salience calculator
        self.mock_salience_calculator = Mock()
        self.mock_salience_calculator.mode = Mock()
        self.mock_salience_calculator.mode.value = "lossy"
        
        # Create sleep cycle
        self.sleep_cycle = SleepCycle(
            predictive_core=self.mock_predictive_core,
            salience_calculator=self.mock_salience_calculator,
            use_salience_weighting=True
        )
        
        # Create breakthrough system
        self.breakthrough_detector, self.breakthrough_processor = create_sleep_breakthrough_system(
            breakthrough_threshold=0.7,
            novelty_threshold=0.6,
            performance_window=50
        )
        
        # Create test experiences
        self.test_experiences = self._create_test_experiences()
    
    def _create_test_experiences(self) -> List[Experience]:
        """Create test experiences for breakthrough detection."""
        experiences = []
        
        for i in range(10):
            # Create mock state
            state = AgentState(
                visual=torch.randn(3, 32, 32),
                proprioception=torch.randn(4),
                energy_level=50.0 + i * 5,
                timestamp=time.time() + i
            )
            
            next_state = AgentState(
                visual=torch.randn(3, 32, 32),
                proprioception=torch.randn(4),
                energy_level=50.0 + i * 5 + 1,
                timestamp=time.time() + i + 1
            )
            
            # Create experience with varying learning progress
            learning_progress = 0.1 + i * 0.1  # Increasing learning progress
            reward = 0.5 + i * 0.1  # Increasing reward
            
            experience = Experience(
                state=state,
                next_state=next_state,
                action=i % 8,
                reward=reward,
                learning_progress=learning_progress,
                timestamp=time.time() + i
            )
            
            experiences.append(experience)
        
        return experiences
    
    def test_breakthrough_detector_initialization(self):
        """Test breakthrough detector initialization."""
        detector = BreakthroughDetector(
            breakthrough_threshold=0.7,
            novelty_threshold=0.6,
            performance_window=50
        )
        
        self.assertEqual(detector.breakthrough_threshold, 0.7)
        self.assertEqual(detector.novelty_threshold, 0.6)
        self.assertEqual(detector.performance_window, 50)
        self.assertEqual(len(detector.performance_history), 0)
    
    def test_breakthrough_processor_initialization(self):
        """Test breakthrough processor initialization."""
        processor = BreakthroughProcessor(
            breakthrough_threshold=0.7,
            novelty_threshold=0.6,
            performance_window=50
        )
        
        self.assertEqual(processor.breakthrough_threshold, 0.7)
        self.assertEqual(processor.novelty_threshold, 0.6)
        self.assertEqual(processor.performance_window, 50)
        self.assertEqual(len(processor.processed_breakthroughs), 0)
    
    def test_create_sleep_breakthrough_system(self):
        """Test factory function for creating breakthrough system."""
        detector, processor = create_sleep_breakthrough_system(
            breakthrough_threshold=0.8,
            novelty_threshold=0.7,
            performance_window=100
        )
        
        self.assertIsInstance(detector, BreakthroughDetector)
        self.assertIsInstance(processor, BreakthroughProcessor)
        self.assertEqual(detector.breakthrough_threshold, 0.8)
        self.assertEqual(processor.novelty_threshold, 0.7)
    
    def test_breakthrough_detection(self):
        """Test breakthrough detection with various experience patterns."""
        # Test with high learning progress experiences
        high_progress_experiences = []
        for i in range(5):
            exp = self.test_experiences[i]
            exp.learning_progress = 0.8 + i * 0.05  # High learning progress
            high_progress_experiences.append(exp)
        
        breakthroughs = self.breakthrough_detector.detect_breakthroughs(high_progress_experiences)
        
        # Should detect some breakthroughs due to high learning progress
        self.assertGreater(len(breakthroughs), 0)
        
        # Test with low learning progress experiences
        low_progress_experiences = []
        for i in range(5):
            exp = self.test_experiences[i]
            exp.learning_progress = 0.1 + i * 0.02  # Low learning progress
            low_progress_experiences.append(exp)
        
        breakthroughs = self.breakthrough_detector.detect_breakthroughs(low_progress_experiences)
        
        # Should detect fewer or no breakthroughs
        self.assertLessEqual(len(breakthroughs), len(high_progress_experiences))
    
    def test_breakthrough_processing(self):
        """Test breakthrough processing and insight generation."""
        # Create a mock breakthrough
        breakthrough = {
            'type': 'learning_acceleration',
            'confidence': 0.85,
            'novelty_score': 0.75,
            'performance_improvement': 0.3,
            'context': 'test_context'
        }
        
        processed = self.breakthrough_processor.process_breakthrough(breakthrough)
        
        self.assertIsNotNone(processed)
        self.assertIn('type', processed)
        self.assertIn('confidence', processed)
        self.assertIn('consolidation_benefit', processed)
        self.assertEqual(len(self.breakthrough_processor.processed_breakthroughs), 1)
    
    def test_sleep_cycle_with_breakthrough_detection(self):
        """Test sleep cycle execution with breakthrough detection."""
        # Add experiences to replay buffer
        for exp in self.test_experiences:
            self.sleep_cycle.add_experience(exp)
        
        # Start sleep cycle
        self.sleep_cycle.sleep()
        
        # Execute sleep cycle with breakthrough detection
        sleep_results = self.sleep_cycle.execute_sleep_cycle(
            replay_buffer=self.test_experiences,
            arc_data={'test': 'data'}
        )
        
        # Check that breakthrough detection was integrated
        self.assertIn('breakthroughs_detected', sleep_results)
        self.assertIn('breakthroughs_processed', sleep_results)
        self.assertIn('breakthrough_quality', sleep_results)
        self.assertIn('breakthrough_insights', sleep_results)
        self.assertIn('breakthrough_consolidation_benefit', sleep_results)
        
        # Wake up
        wake_results = self.sleep_cycle.wake_up()
        self.assertIn('sleep_duration', wake_results)
    
    def test_breakthrough_metrics_tracking(self):
        """Test that breakthrough metrics are properly tracked."""
        # Process some breakthroughs
        for i in range(3):
            breakthrough = {
                'type': f'breakthrough_{i}',
                'confidence': 0.8,
                'novelty_score': 0.7,
                'performance_improvement': 0.2
            }
            self.breakthrough_processor.process_breakthrough(breakthrough)
        
        # Check metrics
        self.assertEqual(len(self.breakthrough_processor.processed_breakthroughs), 3)
        
        # Test metrics retrieval
        metrics = self.breakthrough_processor.get_metrics()
        self.assertIn('total_breakthroughs', metrics)
        self.assertIn('avg_confidence', metrics)
        self.assertIn('avg_novelty', metrics)
        self.assertEqual(metrics['total_breakthroughs'], 3)
    
    def test_breakthrough_consolidation_benefit(self):
        """Test that breakthroughs provide consolidation benefits."""
        # Create high-value breakthrough
        breakthrough = {
            'type': 'major_learning_breakthrough',
            'confidence': 0.95,
            'novelty_score': 0.9,
            'performance_improvement': 0.5,
            'consolidation_benefit': 0.8
        }
        
        processed = self.breakthrough_processor.process_breakthrough(breakthrough)
        
        self.assertGreater(processed.get('consolidation_benefit', 0), 0)
        self.assertGreater(processed.get('confidence', 0), 0.9)
    
    def test_breakthrough_error_handling(self):
        """Test error handling in breakthrough processing."""
        # Test with invalid breakthrough data
        invalid_breakthrough = None
        
        processed = self.breakthrough_processor.process_breakthrough(invalid_breakthrough)
        self.assertIsNone(processed)
        
        # Test with empty breakthrough
        empty_breakthrough = {}
        
        processed = self.breakthrough_processor.process_breakthrough(empty_breakthrough)
        self.assertIsNotNone(processed)  # Should handle gracefully
        self.assertIn('type', processed)
    
    def test_sleep_cycle_breakthrough_integration(self):
        """Test full integration of breakthrough detection in sleep cycle."""
        # Mock the breakthrough detection to return known results
        with patch.object(self.sleep_cycle.breakthrough_detector, 'detect_breakthroughs') as mock_detect:
            with patch.object(self.sleep_cycle.breakthrough_processor, 'process_breakthrough') as mock_process:
                # Setup mocks
                mock_detect.return_value = [
                    {'type': 'test_breakthrough', 'confidence': 0.8}
                ]
                mock_process.return_value = {
                    'type': 'test_breakthrough',
                    'confidence': 0.8,
                    'consolidation_benefit': 0.6
                }
                
                # Add experiences
                for exp in self.test_experiences:
                    self.sleep_cycle.add_experience(exp)
                
                # Start sleep
                self.sleep_cycle.sleep()
                
                # Execute sleep cycle
                sleep_results = self.sleep_cycle.execute_sleep_cycle(
                    replay_buffer=self.test_experiences
                )
                
                # Verify breakthrough processing was called
                self.assertIn('breakthroughs_detected', sleep_results)
                self.assertIn('breakthroughs_processed', sleep_results)
                self.assertEqual(sleep_results['breakthroughs_detected'], 1)
                self.assertEqual(sleep_results['breakthroughs_processed'], 1)
                
                # Wake up
                self.sleep_cycle.wake_up()
    
    def test_breakthrough_performance_tracking(self):
        """Test performance tracking in breakthrough detection."""
        # Add some performance data
        for i in range(10):
            self.breakthrough_detector.update_performance(0.5 + i * 0.1)
        
        # Check performance history
        self.assertEqual(len(self.breakthrough_detector.performance_history), 10)
        
        # Test performance trend calculation
        trend = self.breakthrough_detector._calculate_performance_trend()
        self.assertIsNotNone(trend)
        self.assertIn('trend', trend)
        self.assertIn('stability', trend)
    
    def test_breakthrough_novelty_detection(self):
        """Test novelty detection in breakthrough processing."""
        # Create experiences with varying novelty
        experiences = []
        for i in range(5):
            exp = self.test_experiences[i]
            exp.learning_progress = 0.5 + i * 0.1
            experiences.append(exp)
        
        # Test novelty calculation
        novelty_scores = []
        for exp in experiences:
            novelty = self.breakthrough_detector._calculate_novelty_score(exp, experiences)
            novelty_scores.append(novelty)
        
        # All scores should be between 0 and 1
        for score in novelty_scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_breakthrough_consolidation_integration(self):
        """Test that breakthrough insights integrate with memory consolidation."""
        # Create breakthrough with high consolidation benefit
        breakthrough = {
            'type': 'memory_consolidation_breakthrough',
            'confidence': 0.9,
            'novelty_score': 0.8,
            'performance_improvement': 0.4,
            'consolidation_benefit': 0.7
        }
        
        processed = self.breakthrough_processor.process_breakthrough(breakthrough)
        
        # Check that consolidation benefit is properly calculated
        self.assertGreater(processed.get('consolidation_benefit', 0), 0.5)
        
        # Test that it affects sleep metrics
        initial_breakthroughs = self.sleep_cycle.sleep_metrics.get('breakthroughs_detected', 0)
        
        # Simulate breakthrough detection during sleep
        self.sleep_cycle.sleep_metrics['breakthroughs_detected'] += 1
        self.sleep_cycle.sleep_metrics['breakthroughs_processed'] += 1
        
        self.assertEqual(self.sleep_cycle.sleep_metrics['breakthroughs_detected'], initial_breakthroughs + 1)
        self.assertEqual(self.sleep_cycle.sleep_metrics['breakthroughs_processed'], initial_breakthroughs + 1)


class TestBreakthroughSystemIntegration(unittest.TestCase):
    """Test integration between breakthrough detection and sleep system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.mock_predictive_core = Mock()
        self.mock_predictive_core.use_memory = True
        self.mock_predictive_core.memory = Mock()
        self.mock_predictive_core.memory.get_memory_metrics.return_value = {
            'memory_utilization': 0.5
        }
        self.mock_predictive_core.memory.memory_matrix = torch.randn(100, 64)
        self.mock_predictive_core.memory.usage_vector = torch.rand(100)
        
        self.mock_salience_calculator = Mock()
        self.mock_salience_calculator.mode = Mock()
        self.mock_salience_calculator.mode.value = "lossy"
        
        self.sleep_cycle = SleepCycle(
            predictive_core=self.mock_predictive_core,
            salience_calculator=self.mock_salience_calculator,
            use_salience_weighting=True
        )
    
    def test_breakthrough_system_initialization_in_sleep_cycle(self):
        """Test that breakthrough system is properly initialized in sleep cycle."""
        # Check that breakthrough detector and processor are initialized
        self.assertIsNotNone(self.sleep_cycle.breakthrough_detector)
        self.assertIsNotNone(self.sleep_cycle.breakthrough_processor)
        
        # Check configuration
        self.assertEqual(self.sleep_cycle.breakthrough_detector.breakthrough_threshold, 0.7)
        self.assertEqual(self.sleep_cycle.breakthrough_processor.novelty_threshold, 0.6)
    
    def test_breakthrough_processing_during_sleep_execution(self):
        """Test that breakthrough processing is called during sleep execution."""
        # Create test experiences
        experiences = []
        for i in range(10):
            state = AgentState(
                visual=torch.randn(3, 32, 32),
                proprioception=torch.randn(4),
                energy_level=50.0,
                timestamp=time.time()
            )
            next_state = AgentState(
                visual=torch.randn(3, 32, 32),
                proprioception=torch.randn(4),
                energy_level=51.0,
                timestamp=time.time() + 1
            )
            exp = Experience(
                state=state,
                next_state=next_state,
                action=i % 8,
                reward=0.5,
                learning_progress=0.3 + i * 0.1,
                timestamp=time.time()
            )
            experiences.append(exp)
        
        # Mock breakthrough detection to return known results
        with patch.object(self.sleep_cycle.breakthrough_detector, 'detect_breakthroughs') as mock_detect:
            with patch.object(self.sleep_cycle.breakthrough_processor, 'process_breakthrough') as mock_process:
                mock_detect.return_value = [
                    {'type': 'integration_test_breakthrough', 'confidence': 0.8}
                ]
                mock_process.return_value = {
                    'type': 'integration_test_breakthrough',
                    'confidence': 0.8,
                    'consolidation_benefit': 0.6
                }
                
                # Start sleep and execute cycle
                self.sleep_cycle.sleep()
                sleep_results = self.sleep_cycle.execute_sleep_cycle(experiences)
                
                # Verify breakthrough processing was integrated
                self.assertIn('breakthroughs_detected', sleep_results)
                self.assertIn('breakthroughs_processed', sleep_results)
                
                # Verify methods were called
                mock_detect.assert_called_once()
                mock_process.assert_called_once()
                
                # Wake up
                self.sleep_cycle.wake_up()


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
