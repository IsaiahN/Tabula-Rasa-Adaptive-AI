#!/usr/bin/env python3
"""
Integration Tests for Enhanced Conscious Systems

Tests the new enhanced systems for conscious architecture improvements.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
import time

# Import the new systems
from src.core.enhanced_state_transition_system import (
    EnhancedStateTransitionSystem, CognitiveState, StateTransitionTrigger
)
from src.core.global_workspace_system import (
    GlobalWorkspaceSystem, ModuleType, AttentionType
)
from src.core.aha_moment_simulator import (
    AhaMomentSimulator, RestructuringType, ExplorationStrategy
)
from src.core.conscious_behavior_evaluator import (
    ConsciousBehaviorEvaluator, BehaviorMetric, EvaluationContext, BehaviorObservation
)
from src.core.hybrid_architecture_enhancer import (
    HybridArchitectureEnhancer, ProcessingPath, FeatureType
)
from src.core.interleaved_training_enhancer import (
    InterleavedTrainingEnhancer, TaskType, DifficultyLevel, TrainingMode, TrainingTask
)


class TestEnhancedStateTransitionSystem:
    """Test the enhanced state transition system."""
    
    def test_initialization(self):
        """Test system initialization."""
        system = EnhancedStateTransitionSystem()
        assert system.current_state == CognitiveState.ANALYTICAL
        assert system.insight_threshold == 0.7
        assert system.entropy_threshold == 0.5
    
    def test_decision_processing(self):
        """Test decision processing and state transitions."""
        system = EnhancedStateTransitionSystem()
        
        # Test decision processing
        decision = {'action': 6, 'confidence': 0.8}
        performance = 0.9
        confidence = 0.8
        context = {'difficulty': 0.5, 'novelty': 0.3}
        
        current_state, insight_moment = system.process_decision(
            decision, performance, confidence, context
        )
        
        assert current_state in list(CognitiveState)
        # Insight moment may or may not occur
        if insight_moment:
            assert insight_moment.insight_quality > 0
            assert insight_moment.solution_confidence > 0
    
    def test_entropy_tracking(self):
        """Test entropy tracking functionality."""
        system = EnhancedStateTransitionSystem()
        
        # Add multiple observations
        for i in range(10):
            decision = {'action': i % 7, 'type': f'action_{i}'}
            performance = 0.5 + (i % 3) * 0.2
            confidence = 0.6 + (i % 2) * 0.3
            context = {'difficulty': 0.5}
            
            system.process_decision(decision, performance, confidence, context)
        
        # Check entropy tracking
        stats = system.get_state_statistics()
        assert 'current_state' in stats
        assert 'entropy_trend' in stats
    
    def test_insight_detection(self):
        """Test insight moment detection."""
        system = EnhancedStateTransitionSystem()
        
        # Simulate high entropy situation
        for i in range(5):
            decision = {'action': i, 'type': f'action_{i}'}
            performance = 0.3  # Low performance
            confidence = 0.2  # Low confidence
            context = {'difficulty': 0.8, 'novelty': 0.9}
            
            current_state, insight_moment = system.process_decision(
                decision, performance, confidence, context
            )
        
        # Check if insight was detected
        stats = system.get_state_statistics()
        assert 'insight_moments' in stats


class TestGlobalWorkspaceSystem:
    """Test the global workspace system."""
    
    def test_initialization(self):
        """Test system initialization."""
        system = GlobalWorkspaceSystem()
        assert len(system.specialized_modules) == 3
        assert ModuleType.VISION in system.specialized_modules
        assert ModuleType.REASONING in system.specialized_modules
        assert ModuleType.MEMORY in system.specialized_modules
    
    def test_global_workspace_processing(self):
        """Test global workspace processing."""
        system = GlobalWorkspaceSystem()
        
        # Test processing
        input_data = {
            'frame': np.zeros((32, 32, 3), dtype=np.uint8),
            'context': 'test'
        }
        context = {'task_type': 'test', 'difficulty': 0.5}
        
        workspace_state = system.process_global_workspace(input_data, context)
        
        assert workspace_state is not None
        assert 'active_modules' in workspace_state.__dict__
        assert 'attention_weights' in workspace_state.__dict__
        assert 'coherence_score' in workspace_state.__dict__
    
    def test_attention_mechanisms(self):
        """Test attention mechanisms."""
        system = GlobalWorkspaceSystem()
        
        # Test with different input types
        test_cases = [
            {'frame': np.zeros((32, 32, 3)), 'context': 'visual_task'},
            {'text': 'test input', 'context': 'text_task'},
            {'audio': np.zeros(1000), 'context': 'audio_task'}
        ]
        
        for input_data in test_cases:
            context = {'task_type': 'test'}
            workspace_state = system.process_global_workspace(input_data, context)
            
            assert workspace_state.coherence_score >= 0
            assert workspace_state.coherence_score <= 1
    
    def test_workspace_statistics(self):
        """Test workspace statistics."""
        system = GlobalWorkspaceSystem()
        
        # Process some data
        for i in range(5):
            input_data = {'frame': np.zeros((32, 32, 3)), 'iteration': i}
            context = {'task_type': 'test'}
            system.process_global_workspace(input_data, context)
        
        stats = system.get_workspace_statistics()
        assert 'current_state' in stats
        assert 'module_count' in stats
        assert 'history_length' in stats


class TestAhaMomentSimulator:
    """Test the Aha! moment simulator."""
    
    def test_initialization(self):
        """Test system initialization."""
        simulator = AhaMomentSimulator()
        assert simulator.latent_dim == 128
        assert len(simulator.exploration_strategies) > 0
    
    def test_problem_representation_creation(self):
        """Test problem representation creation."""
        from src.core.aha_moment_simulator import ProblemRepresentation
        
        problem_rep = ProblemRepresentation(
            latent_vector=np.random.randn(128),
            features={'shape': 'circle', 'color': 'red'},
            constraints=['must_be_red'],
            goals=['find_circle'],
            difficulty=0.7
        )
        
        assert problem_rep.latent_vector.shape == (128,)
        assert problem_rep.difficulty == 0.7
        assert len(problem_rep.constraints) == 1
    
    def test_aha_moment_simulation(self):
        """Test Aha! moment simulation."""
        simulator = AhaMomentSimulator()
        
        from src.core.aha_moment_simulator import ProblemRepresentation
        
        problem_rep = ProblemRepresentation(
            latent_vector=np.random.randn(128),
            features={'shape': 'circle', 'color': 'red'},
            constraints=['must_be_red'],
            goals=['find_circle'],
            difficulty=0.7
        )
        
        context = {
            'difficulty': 0.7,
            'novelty': 0.8,
            'complexity': 0.6
        }
        
        aha_moment = simulator.simulate_aha_moment(problem_rep, context)
        
        # Aha moment may or may not occur
        if aha_moment:
            assert aha_moment.insight_quality > 0
            assert aha_moment.solution_confidence > 0
            assert aha_moment.restructuring_event is not None
    
    def test_exploration_strategies(self):
        """Test different exploration strategies."""
        simulator = AhaMomentSimulator()
        
        # Test that all strategies are available
        strategies = simulator.exploration_strategies
        assert ExplorationStrategy.RANDOM_WALK in strategies
        assert ExplorationStrategy.GRADIENT_ASCENT in strategies
        assert ExplorationStrategy.SIMULATED_ANNEALING in strategies
        assert ExplorationStrategy.DIFFUSION_SAMPLING in strategies
    
    def test_simulator_statistics(self):
        """Test simulator statistics."""
        simulator = AhaMomentSimulator()
        
        stats = simulator.get_simulator_statistics()
        assert 'total_aha_moments' in stats
        assert 'total_restructuring_events' in stats
        assert 'exploration_strategies_used' in stats


class TestConsciousBehaviorEvaluator:
    """Test the conscious behavior evaluator."""
    
    def test_initialization(self):
        """Test system initialization."""
        evaluator = ConsciousBehaviorEvaluator()
        assert evaluator.window_size == 50
        assert evaluator.flexibility_metrics is not None
        assert evaluator.introspection_metrics is not None
        assert evaluator.generalization_metrics is not None
        assert evaluator.robustness_metrics is not None
    
    def test_behavior_observation_creation(self):
        """Test behavior observation creation."""
        observation = BehaviorObservation(
            timestamp=time.time(),
            context=EvaluationContext.TASK_SWITCHING,
            input_data={'task': 'test'},
            output_data={'result': 'success'},
            performance_metrics={'overall_score': 0.8},
            confidence_level=0.7,
            strategy_used='test_strategy',
            error_occurred=False
        )
        
        assert observation.context == EvaluationContext.TASK_SWITCHING
        assert observation.confidence_level == 0.7
        assert not observation.error_occurred
    
    def test_behavior_evaluation(self):
        """Test behavior evaluation."""
        evaluator = ConsciousBehaviorEvaluator()
        
        # Add some observations
        for i in range(10):
            observation = BehaviorObservation(
                timestamp=time.time(),
                context=EvaluationContext.TASK_SWITCHING,
                input_data={'task': f'task_{i}'},
                output_data={'result': 'success'},
                performance_metrics={'overall_score': 0.5 + (i % 3) * 0.2},
                confidence_level=0.6 + (i % 2) * 0.3,
                strategy_used=f'strategy_{i % 3}',
                error_occurred=i % 5 == 0
            )
            evaluator.add_observation(observation)
        
        # Evaluate behavior
        profile = evaluator.evaluate_conscious_behavior()
        
        assert profile.overall_score >= 0
        assert profile.overall_score <= 1
        assert len(profile.metric_scores) == 4
        assert len(profile.recommendations) > 0
    
    def test_metric_evaluations(self):
        """Test individual metric evaluations."""
        evaluator = ConsciousBehaviorEvaluator()
        
        # Test flexibility metrics
        for i in range(10):
            observation = BehaviorObservation(
                timestamp=time.time(),
                context=EvaluationContext.TASK_SWITCHING,
                input_data={'task': f'task_{i}'},
                output_data={'result': 'success'},
                performance_metrics={'overall_score': 0.8},
                confidence_level=0.7,
                strategy_used=f'strategy_{i % 3}',
                error_occurred=False
            )
            evaluator.flexibility_metrics.add_observation(observation)
        
        flexibility_eval = evaluator.flexibility_metrics.evaluate_flexibility()
        assert flexibility_eval.metric == BehaviorMetric.FLEXIBILITY
        assert flexibility_eval.score >= 0
        assert flexibility_eval.score <= 1


class TestHybridArchitectureEnhancer:
    """Test the hybrid architecture enhancer."""
    
    def test_initialization(self):
        """Test system initialization."""
        enhancer = HybridArchitectureEnhancer()
        assert enhancer.input_dim == 512
        assert enhancer.feature_dim == 512
        assert enhancer.relational_dim == 128
    
    def test_hybrid_processing(self):
        """Test hybrid architecture processing."""
        enhancer = HybridArchitectureEnhancer()
        
        input_data = {
            'frame': np.zeros((32, 32, 3), dtype=np.uint8),
            'context': 'test'
        }
        context = {'task_type': 'test', 'difficulty': 0.5}
        
        output = enhancer.process_with_hybrid_architecture(input_data, context)
        
        assert output is not None
        assert len(output.feature_results) > 0
        assert len(output.reasoning_results) > 0
        assert output.confidence >= 0
        assert output.confidence <= 1
        assert output.processing_time > 0
    
    def test_feature_extraction(self):
        """Test feature extraction functionality."""
        enhancer = HybridArchitectureEnhancer()
        
        # Test with different input types
        test_cases = [
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.ones((32, 32, 3), dtype=np.uint8) * 128,
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        ]
        
        for frame in test_cases:
            input_data = {'frame': frame}
            context = {'task_type': 'test'}
            
            output = enhancer.process_with_hybrid_architecture(input_data, context)
            
            # Check feature results
            assert len(output.feature_results) > 0
            for feat_result in output.feature_results:
                assert feat_result.feature_type in list(FeatureType)
                assert feat_result.confidence >= 0
                assert feat_result.confidence <= 1
    
    def test_processing_statistics(self):
        """Test processing statistics."""
        enhancer = HybridArchitectureEnhancer()
        
        # Process some data
        for i in range(5):
            input_data = {'frame': np.zeros((32, 32, 3)), 'iteration': i}
            context = {'task_type': 'test'}
            enhancer.process_with_hybrid_architecture(input_data, context)
        
        stats = enhancer.get_processing_statistics()
        assert 'total_processing_time' in stats
        assert 'path_usage_counts' in stats
        assert 'time_distribution' in stats


class TestInterleavedTrainingEnhancer:
    """Test the interleaved training enhancer."""
    
    def test_initialization(self):
        """Test system initialization."""
        enhancer = InterleavedTrainingEnhancer()
        assert enhancer.rehearsal_buffer_size == 1000
        assert enhancer.interleaving_ratio == 0.3
        assert enhancer.curriculum_enabled == True
        assert enhancer.generative_replay_enabled == True
    
    def test_training_task_creation(self):
        """Test training task creation."""
        task = TrainingTask(
            task_id='test_task_1',
            task_type=TaskType.ARC_PUZZLE,
            difficulty=DifficultyLevel.MEDIUM,
            input_data={'frame': np.zeros((32, 32, 3))},
            target_output={'action': 6, 'coordinates': (5, 5)}
        )
        
        assert task.task_type == TaskType.ARC_PUZZLE
        assert task.difficulty == DifficultyLevel.MEDIUM
        assert 'frame' in task.input_data
    
    def test_enhanced_training_schedule(self):
        """Test enhanced training schedule creation."""
        enhancer = InterleavedTrainingEnhancer()
        
        # Create test tasks
        tasks = []
        for i in range(10):
            task = TrainingTask(
                task_id=f'task_{i}',
                task_type=TaskType.ARC_PUZZLE,
                difficulty=DifficultyLevel.MEDIUM,
                input_data={'frame': np.zeros((32, 32, 3))},
                target_output={'action': i % 7}
            )
            tasks.append(task)
        
        # Create enhanced schedule
        schedule = enhancer.create_enhanced_training_schedule(tasks, TrainingMode.INTERLEAVED)
        
        assert schedule is not None
        assert len(schedule.tasks) > 0
        assert schedule.mode == TrainingMode.INTERLEAVED
        assert schedule.interleaving_ratio == 0.3
    
    def test_rehearsal_buffer(self):
        """Test rehearsal buffer functionality."""
        enhancer = InterleavedTrainingEnhancer()
        
        # Create test task
        task = TrainingTask(
            task_id='test_task',
            task_type=TaskType.ARC_PUZZLE,
            difficulty=DifficultyLevel.MEDIUM,
            input_data={'frame': np.zeros((32, 32, 3))},
            target_output={'action': 6}
        )
        
        # Add to rehearsal buffer
        enhancer.add_task_to_rehearsal_buffer(task, importance_score=0.8)
        
        # Check buffer
        assert len(enhancer.rehearsal_buffer.tasks) == 1
        assert enhancer.rehearsal_buffer.importance_scores['test_task'] == 0.8
    
    def test_catastrophic_forgetting_detection(self):
        """Test catastrophic forgetting detection."""
        enhancer = InterleavedTrainingEnhancer()
        
        # Test with performance drop
        old_performance = {'task_1': 0.9, 'task_2': 0.8}
        new_performance = {'task_1': 0.6, 'task_2': 0.5}  # Significant drop
        
        forgetting_detected = enhancer.detect_catastrophic_forgetting(
            old_performance, new_performance
        )
        
        assert forgetting_detected == True
    
    def test_training_statistics(self):
        """Test training statistics."""
        enhancer = InterleavedTrainingEnhancer()
        
        stats = enhancer.get_training_statistics()
        assert 'total_tasks_processed' in stats
        assert 'interleaved_tasks' in stats
        assert 'rehearsal_tasks' in stats
        assert 'forgetting_events' in stats


class TestSystemIntegration:
    """Test integration between all enhanced systems."""
    
    def test_enhanced_conscious_architecture_integration(self):
        """Test integration of all enhanced conscious architecture systems."""
        
        # Initialize all systems
        state_transition = EnhancedStateTransitionSystem()
        global_workspace = GlobalWorkspaceSystem()
        aha_simulator = AhaMomentSimulator()
        behavior_evaluator = ConsciousBehaviorEvaluator()
        hybrid_enhancer = HybridArchitectureEnhancer()
        training_enhancer = InterleavedTrainingEnhancer()
        
        # Test that all systems can be initialized together
        assert state_transition is not None
        assert global_workspace is not None
        assert aha_simulator is not None
        assert behavior_evaluator is not None
        assert hybrid_enhancer is not None
        assert training_enhancer is not None
    
    def test_end_to_end_processing(self):
        """Test end-to-end processing through multiple systems."""
        
        # Initialize systems
        state_transition = EnhancedStateTransitionSystem()
        global_workspace = GlobalWorkspaceSystem()
        hybrid_enhancer = HybridArchitectureEnhancer()
        
        # Test data
        input_data = {
            'frame': np.zeros((32, 32, 3), dtype=np.uint8),
            'context': 'test'
        }
        context = {'task_type': 'test', 'difficulty': 0.5}
        
        # Process through hybrid architecture
        hybrid_output = hybrid_enhancer.process_with_hybrid_architecture(input_data, context)
        
        # Process through global workspace
        workspace_state = global_workspace.process_global_workspace(input_data, context)
        
        # Process through state transition
        decision = {'action': 6, 'confidence': 0.8}
        performance = 0.9
        confidence = 0.8
        current_state, insight_moment = state_transition.process_decision(
            decision, performance, confidence, context
        )
        
        # Verify all systems produced valid outputs
        assert hybrid_output is not None
        assert workspace_state is not None
        assert current_state is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
