#!/usr/bin/env python3
"""
Test Enhanced Mutation System - Tests the enhanced mutation engine and sandbox tester.
"""

import unittest
import asyncio
import time
from unittest.mock import Mock, patch
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.mutation_system.mutator import MutationEngine, MutationContext
from src.core.mutation_system.tester import SandboxTester, TestConfiguration, TestAnalytics
from src.core.mutation_system.types import Mutation, MutationType, MutationImpact, TestResult
from src.core.system_design.genome import SystemGenome
from src.core.architect import Architect

class TestEnhancedMutationSystem(unittest.TestCase):
    """Test the enhanced mutation system components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Mock()
        self.genome = SystemGenome()
        self.mutation_engine = MutationEngine(self.genome, self.logger)
        self.sandbox_tester = SandboxTester(Path("."), self.logger)
        self.architect = Architect(logger=self.logger)
    
    def test_mutation_context_creation(self):
        """Test mutation context creation."""
        performance_data = [
            {'score': 50, 'success': True},
            {'score': 60, 'success': True},
            {'score': 55, 'success': False}
        ]
        frame_analysis = {'interactive_targets': [1, 2, 3], 'confidence': 0.8}
        context = {'memory_state': {'fragmentation_ratio': 0.5}, 'energy_state': {'current_energy': 75}}
        
        mutation_context = self.architect._create_mutation_context(performance_data, frame_analysis, context)
        
        self.assertIsInstance(mutation_context, MutationContext)
        self.assertEqual(len(mutation_context.performance_history), 3)
        self.assertEqual(mutation_context.frame_analysis, frame_analysis)
        self.assertEqual(mutation_context.memory_state, {'fragmentation_ratio': 0.5})
        self.assertEqual(mutation_context.energy_state, {'current_energy': 75})
        self.assertIsInstance(mutation_context.learning_progress, float)
        self.assertIsInstance(mutation_context.stagnation_detected, bool)
        self.assertIsInstance(mutation_context.recent_failures, int)
    
    def test_context_aware_mutation_generation(self):
        """Test context-aware mutation generation."""
        # Create a context indicating stagnation
        context = MutationContext(
            performance_history=[{'score': 50, 'success': False}] * 5,
            frame_analysis={'interactive_targets': []},
            memory_state={'fragmentation_ratio': 0.8},
            energy_state={'current_energy': 20},
            learning_progress=0.05,
            stagnation_detected=True,
            recent_failures=4
        )
        
        mutation = self.mutation_engine.generate_context_aware_mutation(context)
        
        self.assertIsInstance(mutation, Mutation)
        self.assertIn('context_aware', mutation.id)
        self.assertIsInstance(mutation.changes, dict)
        self.assertGreater(mutation.expected_improvement, 0)
        self.assertGreater(mutation.confidence, 0)
    
    def test_adaptive_weights_update(self):
        """Test adaptive weights update based on mutation results."""
        # Create a test mutation
        mutation = Mutation(
            id="test_mutation",
            type=MutationType.PARAMETER_ADJUSTMENT,
            impact=MutationImpact.MODERATE,
            changes={'salience_threshold': 0.4},
            rationale="Test mutation",
            expected_improvement=0.1,
            confidence=0.7,
            test_duration_estimate=10.0
        )
        
        # Test successful mutation
        initial_weight = self.mutation_engine.adaptive_weights['parameter_adjustment']
        self.mutation_engine.update_adaptive_weights(mutation, True, 0.15)
        
        # Weight should increase for successful mutation
        new_weight = self.mutation_engine.adaptive_weights['parameter_adjustment']
        self.assertGreater(new_weight, initial_weight)
        
        # Test failed mutation
        self.mutation_engine.update_adaptive_weights(mutation, False, -0.05)
        final_weight = self.mutation_engine.adaptive_weights['parameter_adjustment']
        self.assertLess(final_weight, new_weight)
    
    def test_enhanced_mock_test_result(self):
        """Test enhanced mock test result generation."""
        mutation = Mutation(
            id="test_mutation",
            type=MutationType.ARCHITECTURE_ENHANCEMENT,
            impact=MutationImpact.SIGNIFICANT,
            changes={'enable_meta_learning': True},
            rationale="Test architectural enhancement",
            expected_improvement=0.2,
            confidence=0.8,
            test_duration_estimate=20.0
        )
        
        test_config = TestConfiguration(
            test_games=["test_game_1"],
            max_episodes=3,
            performance_metrics=['win_rate', 'learning_efficiency']
        )
        
        test_result = self.sandbox_tester._create_enhanced_mock_test_result(
            mutation, self.genome, time.time(), test_config
        )
        
        self.assertIsInstance(test_result, TestResult)
        self.assertEqual(test_result.mutation_id, mutation.id)
        self.assertIn('analysis', test_result.detailed_results)
        self.assertIn('enhanced_mock', test_result.detailed_results['mode'])
    
    def test_test_analytics(self):
        """Test test analytics functionality."""
        analytics = TestAnalytics()
        
        # Create test result
        test_result = TestResult(
            mutation_id="test_mutation",
            genome_hash="test_hash",
            success=True,
            performance_metrics={'win_rate': 0.7, 'learning_efficiency': 0.6},
            improvement_over_baseline={'win_rate': 0.1, 'learning_efficiency': 0.05},
            test_duration=15.0
        )
        
        mutation = Mutation(
            id="test_mutation",
            type=MutationType.PARAMETER_ADJUSTMENT,
            impact=MutationImpact.MODERATE,
            changes={'salience_threshold': 0.4},
            rationale="Test mutation",
            expected_improvement=0.1,
            confidence=0.7,
            test_duration_estimate=10.0
        )
        
        analysis = analytics.analyze_test_result(test_result, mutation)
        
        self.assertIn('mutation_id', analysis)
        self.assertIn('performance_breakdown', analysis)
        self.assertIn('trend_indicators', analysis)
        self.assertIn('risk_assessment', analysis)
        
        # Check trend indicators
        trend_indicators = analysis['trend_indicators']
        self.assertIn('consistency', trend_indicators)
        self.assertIn('stability', trend_indicators)
        self.assertIn('scalability', trend_indicators)
        
        # Check risk assessment
        risk_assessment = analysis['risk_assessment']
        self.assertIn('risk_score', risk_assessment)
        self.assertIn('risk_factors', risk_assessment)
        self.assertIn('recommendation', risk_assessment)
    
    def test_architect_enhanced_strategy_evolution(self):
        """Test enhanced strategy evolution in Architect."""
        available_actions = [1, 2, 3, 4, 5, 6]
        context = {
            'game_id': 'test_game',
            'memory_state': {'fragmentation_ratio': 0.3},
            'energy_state': {'current_energy': 80}
        }
        performance_data = [
            {'score': 60, 'success': True},
            {'score': 65, 'success': True},
            {'score': 70, 'success': True}
        ]
        frame_analysis = {
            'interactive_targets': [1, 2, 3],
            'confidence': 0.9
        }
        
        result = self.architect.evolve_strategy(available_actions, context, performance_data, frame_analysis)
        
        self.assertIn('strategy', result)
        self.assertIn('reasoning', result)
        self.assertIn('innovation_score', result)
        self.assertIn('mutation_id', result)
        self.assertIn('mutation_context', result)
        self.assertIn('expected_improvement', result)
        
        # Check strategy details
        strategy = result['strategy']
        self.assertEqual(strategy['actions'], available_actions)
        self.assertIn('mutation_applied', strategy)
        self.assertIn('mutation_type', strategy)
        self.assertIn('expected_improvement', strategy)
    
    def test_learning_progress_calculation(self):
        """Test learning progress calculation."""
        # Test with improving performance
        improving_data = [
            {'score': 50}, {'score': 55}, {'score': 60}, {'score': 65}, {'score': 70}
        ]
        progress = self.architect._calculate_learning_progress(improving_data)
        self.assertGreater(progress, 0.5)  # Should be high for improving trend
        
        # Test with declining performance
        declining_data = [
            {'score': 70}, {'score': 65}, {'score': 60}, {'score': 55}, {'score': 50}
        ]
        progress = self.architect._calculate_learning_progress(declining_data)
        self.assertLess(progress, 0.5)  # Should be low for declining trend
        
        # Test with insufficient data
        insufficient_data = [{'score': 60}]
        progress = self.architect._calculate_learning_progress(insufficient_data)
        self.assertEqual(progress, 0.5)  # Should default to 0.5
    
    def test_enhanced_reasoning_generation(self):
        """Test enhanced reasoning generation."""
        mutation = Mutation(
            id="test_mutation",
            type=MutationType.ARCHITECTURE_ENHANCEMENT,
            impact=MutationImpact.SIGNIFICANT,
            changes={'enable_meta_learning': True},
            rationale="Enable meta-learning for better adaptation",
            expected_improvement=0.25,
            confidence=0.85,
            test_duration_estimate=25.0
        )
        
        context = MutationContext(
            performance_history=[],
            frame_analysis={},
            memory_state={},
            energy_state={},
            learning_progress=0.1,
            stagnation_detected=True,
            recent_failures=5
        )
        
        reasoning = self.architect._generate_enhanced_reasoning(mutation, context, "test_game")
        
        self.assertIn("Enhanced strategy evolution for test_game", reasoning)
        self.assertIn("Stagnation detected", reasoning)
        self.assertIn("Recent failures (5)", reasoning)
        self.assertIn("Low learning progress", reasoning)
        self.assertIn("test_mutation", reasoning)
        self.assertIn("Enable meta-learning", reasoning)
        self.assertIn("25.0%", reasoning)  # Expected improvement
        self.assertIn("85.0%", reasoning)  # Confidence

class TestMutationSystemIntegration(unittest.TestCase):
    """Test integration between mutation system components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Mock()
        self.genome = SystemGenome()
        self.mutation_engine = MutationEngine(self.genome, self.logger)
        self.sandbox_tester = SandboxTester(Path("."), self.logger)
    
    @patch('asyncio.create_subprocess_exec')
    async def test_full_mutation_cycle(self, mock_subprocess):
        """Test full mutation cycle from generation to testing."""
        # Create context indicating need for improvement
        context = MutationContext(
            performance_history=[{'score': 40, 'success': False}] * 3,
            frame_analysis={'interactive_targets': []},
            memory_state={'fragmentation_ratio': 0.6},
            energy_state={'current_energy': 30},
            learning_progress=0.05,
            stagnation_detected=True,
            recent_failures=3
        )
        
        # Generate mutation
        mutation = self.mutation_engine.generate_context_aware_mutation(context)
        self.assertIsInstance(mutation, Mutation)
        
        # Test mutation
        test_result = await self.sandbox_tester.test_mutation(mutation, self.genome)
        self.assertIsInstance(test_result, TestResult)
        
        # Update adaptive weights
        improvement = test_result.get_overall_improvement()
        self.mutation_engine.update_adaptive_weights(mutation, test_result.success, improvement)
        
        # Verify weights were updated
        self.assertIn(mutation.type.value, self.mutation_engine.adaptive_weights)

if __name__ == '__main__':
    # Run async tests
    async def run_async_tests():
        test_suite = unittest.TestLoader().loadTestsFromTestCase(TestMutationSystemIntegration)
        for test in test_suite:
            if hasattr(test, 'test_full_mutation_cycle'):
                await test.test_full_mutation_cycle()
    
    # Run synchronous tests
    unittest.main(verbosity=2)
