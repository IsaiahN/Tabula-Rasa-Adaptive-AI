#!/usr/bin/env python3
"""
Comprehensive Integration Tests for High Priority Enhancements

Tests the Self-Prior Mechanism, Pattern Discovery Curiosity, and Enhanced
Architectural Systems to ensure they work correctly and integrate properly
with existing Tabula Rasa systems.
"""

import pytest
import numpy as np
import torch
import time
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any

# Import the new enhancement modules
from src.core.self_prior_mechanism import (
    SelfPriorManager, MultimodalEncoder, DensityModel, IntrinsicGoalGenerator,
    SensoryExperience, IntrinsicGoal, GoalType, SensoryModality
)
from src.core.pattern_discovery_curiosity import (
    PatternDiscoveryCuriosity, CompressionRewarder, PatternClassifier, UtilityLearner,
    DiscoveredPattern, PatternType, CuriosityLevel, CuriosityEvent
)
from src.core.enhanced_architectural_systems import (
    EnhancedTreeBasedDirector, EnhancedTreeBasedArchitect, EnhancedImplicitMemoryManager,
    EnhancedGoal, ArchitecturalEnhancement, EnhancementType
)

class TestSelfPriorMechanism:
    """Test suite for Self-Prior Mechanism."""
    
    @pytest.fixture
    def self_prior_manager(self):
        """Create a self-prior manager for testing."""
        return SelfPriorManager(
            visual_dim=64,
            proprioceptive_dim=16,
            tactile_dim=8,
            auditory_dim=32,
            latent_dim=32,
            n_density_components=5,
            max_experiences=100
        )
    
    @pytest.fixture
    def sample_sensory_experience(self):
        """Create a sample sensory experience for testing."""
        return {
            'visual_features': np.random.rand(64),
            'proprioceptive_state': np.random.rand(16),
            'tactile_feedback': np.random.rand(8),
            'auditory_input': np.random.rand(32),
            'context': {'test': True}
        }
    
    def test_multimodal_encoder_initialization(self):
        """Test multimodal encoder initialization."""
        encoder = MultimodalEncoder(
            visual_dim=64,
            proprioceptive_dim=16,
            tactile_dim=8,
            auditory_dim=32,
            latent_dim=32
        )
        
        assert encoder.visual_dim == 64
        assert encoder.proprioceptive_dim == 16
        assert encoder.tactile_dim == 8
        assert encoder.auditory_dim == 32
        assert encoder.latent_dim == 32
    
    def test_multimodal_encoder_forward(self):
        """Test multimodal encoder forward pass."""
        encoder = MultimodalEncoder(
            visual_dim=64,
            proprioceptive_dim=16,
            tactile_dim=8,
            auditory_dim=32,
            latent_dim=32
        )
        
        visual = torch.randn(1, 64)
        proprio = torch.randn(1, 16)
        tactile = torch.randn(1, 8)
        auditory = torch.randn(1, 32)
        
        output = encoder(visual, proprio, tactile, auditory)
        
        assert output.shape == (1, 32)
        assert torch.all(torch.isfinite(output))
    
    def test_density_model_operations(self):
        """Test density model operations."""
        density_model = DensityModel(n_components=3, max_samples=100)
        
        # Add samples
        for _ in range(20):
            sample = np.random.rand(10)
            density_model.add_sample(sample)
        
        # Fit model
        assert density_model.fit() == True
        assert density_model.is_fitted == True
        
        # Test log likelihood
        test_sample = np.random.rand(10)
        likelihood = density_model.log_likelihood(test_sample)
        assert isinstance(likelihood, float)
        assert np.isfinite(likelihood)
        
        # Test novelty score
        novelty = density_model.novelty_score(test_sample)
        assert isinstance(novelty, float)
        assert 0.0 <= novelty <= 1.0
    
    def test_intrinsic_goal_generation(self, self_prior_manager):
        """Test intrinsic goal generation."""
        # Create a sample experience
        experience = SensoryExperience(
            visual_features=np.random.rand(64),
            proprioceptive_state=np.random.rand(16),
            tactile_feedback=np.random.rand(8),
            auditory_input=np.random.rand(32),
            prediction_error=0.3
        )
        
        # Generate goals
        goals = self_prior_manager.goal_generator.generate_goals(
            experience, 0.3, {'test': True}
        )
        
        assert isinstance(goals, list)
        for goal in goals:
            assert isinstance(goal, IntrinsicGoal)
            assert goal.confidence >= 0.0
            assert goal.confidence <= 1.0
            assert goal.priority >= 0.0
            assert goal.priority <= 1.0
    
    def test_sensory_experience_processing(self, self_prior_manager, sample_sensory_experience):
        """Test processing of sensory experiences."""
        result = self_prior_manager.process_sensory_experience(**sample_sensory_experience)
        
        assert 'encoding' in result
        assert 'prediction_error' in result
        assert 'intrinsic_goals' in result
        assert 'novelty_score' in result
        assert 'body_schema_confidence' in result
        assert 'self_awareness_score' in result
        
        assert isinstance(result['encoding'], np.ndarray)
        assert isinstance(result['intrinsic_goals'], list)
        assert 0.0 <= result['novelty_score'] <= 1.0
        assert 0.0 <= result['body_schema_confidence'] <= 1.0
        assert 0.0 <= result['self_awareness_score'] <= 1.0
    
    def test_intrinsic_rewards_generation(self, self_prior_manager):
        """Test intrinsic rewards generation."""
        # Process some experiences first
        for _ in range(10):
            experience = {
                'visual_features': np.random.rand(64),
                'proprioceptive_state': np.random.rand(16),
                'tactile_feedback': np.random.rand(8),
                'auditory_input': np.random.rand(32),
                'context': {'test': True}
            }
            self_prior_manager.process_sensory_experience(**experience)
        
        # Get intrinsic rewards
        current_state = {'encoding': np.random.rand(32)}
        rewards = self_prior_manager.get_intrinsic_rewards(current_state)
        
        assert isinstance(rewards, dict)
        assert 'novelty' in rewards
        assert 'self_prior_alignment' in rewards
        assert 'exploration' in rewards
        assert 'prediction_error_minimization' in rewards
        
        for reward_name, reward_value in rewards.items():
            assert isinstance(reward_value, float)
            assert reward_value >= 0.0
    
    def test_self_prior_metrics(self, self_prior_manager):
        """Test self-prior metrics generation."""
        # Process some experiences first
        for _ in range(5):
            experience = {
                'visual_features': np.random.rand(64),
                'proprioceptive_state': np.random.rand(16),
                'tactile_feedback': np.random.rand(8),
                'auditory_input': np.random.rand(32),
                'context': {'test': True}
            }
            self_prior_manager.process_sensory_experience(**experience)
        
        metrics = self_prior_manager.get_self_prior_metrics()
        
        assert isinstance(metrics, dict)
        assert 'body_schema_confidence' in metrics
        assert 'prediction_accuracy' in metrics
        assert 'novelty_level' in metrics
        assert 'exploration_drive' in metrics
        assert 'self_awareness_score' in metrics
        assert 'active_goals_count' in metrics
        assert 'total_experiences' in metrics
        assert 'density_model_fitted' in metrics

class TestPatternDiscoveryCuriosity:
    """Test suite for Pattern Discovery Curiosity."""
    
    @pytest.fixture
    def pattern_discovery_curiosity(self):
        """Create a pattern discovery curiosity system for testing."""
        return PatternDiscoveryCuriosity(
            compression_methods=['gzip', 'pca', 'clustering'],
            learning_rate=0.1,
            curiosity_decay=0.99,
            max_patterns=100
        )
    
    @pytest.fixture
    def sample_observation(self):
        """Create a sample observation for testing."""
        return np.random.rand(10, 10)
    
    def test_compression_rewarder_initialization(self):
        """Test compression rewarder initialization."""
        rewarder = CompressionRewarder(
            compression_methods=['gzip', 'pca'],
            min_compression_ratio=0.1,
            max_compression_ratio=0.9
        )
        
        assert 'gzip' in rewarder.compression_methods
        assert 'pca' in rewarder.compression_methods
        assert rewarder.min_compression_ratio == 0.1
        assert rewarder.max_compression_ratio == 0.9
    
    def test_compression_reward_computation(self):
        """Test compression reward computation."""
        rewarder = CompressionRewarder()
        
        # Test with different data types
        test_data = np.random.rand(5, 5)
        reward, details = rewarder.compute_compression_reward(test_data, PatternType.SYMMETRY)
        
        assert isinstance(reward, float)
        assert isinstance(details, dict)
        assert 'gzip_ratio' in details
        assert 'pca_ratio' in details
        assert 'cluster_ratio' in details
        assert 'pattern_bonus' in details
    
    def test_pattern_classifier_detection(self):
        """Test pattern classifier detection."""
        classifier = PatternClassifier()
        
        # Test symmetry detection
        symmetric_data = np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]])
        pattern_data, confidence = classifier._detect_symmetry(symmetric_data, {})
        assert confidence >= 0.0  # Should detect some symmetry
        
        # Test sequence detection
        sequence_data = np.array([1, 2, 3, 4, 5])
        pattern_data, confidence = classifier._detect_sequence(sequence_data, {})
        assert confidence >= 0.0  # Should detect some sequence pattern
    
    def test_utility_learner_operations(self):
        """Test utility learner operations."""
        learner = UtilityLearner(learning_rate=0.1, decay_rate=0.95)
        
        # Create a sample pattern
        pattern = DiscoveredPattern(
            pattern_type=PatternType.SYMMETRY,
            pattern_data=np.random.rand(5, 5),
            compression_ratio=0.5,
            confidence=0.8,
            utility_score=0.6
        )
        
        # Learn from pattern-action-outcome
        learner.learn_from_pattern(pattern, 1, {'success': True, 'reward': 0.8}, {})
        
        # Get action utility
        utility = learner.get_action_utility(pattern, 1)
        assert isinstance(utility, float)
        assert 0.0 <= utility <= 1.0
    
    def test_observation_processing(self, pattern_discovery_curiosity, sample_observation):
        """Test observation processing for pattern discovery."""
        context = {'test': True, 'temporal_sequence': [1, 2, 3, 1, 2, 3]}
        result = pattern_discovery_curiosity.process_observation(sample_observation, context)
        
        assert 'patterns_discovered' in result
        assert 'pattern_rewards' in result
        assert 'total_curiosity_boost' in result
        assert 'intrinsic_rewards' in result
        assert 'curiosity_levels' in result
        assert 'recent_patterns' in result
        
        assert isinstance(result['patterns_discovered'], int)
        assert isinstance(result['pattern_rewards'], dict)
        assert isinstance(result['total_curiosity_boost'], float)
        assert isinstance(result['intrinsic_rewards'], dict)
        assert isinstance(result['curiosity_levels'], dict)
        assert isinstance(result['recent_patterns'], list)
    
    def test_learning_from_action_outcome(self, pattern_discovery_curiosity):
        """Test learning from action outcomes."""
        # Create a sample pattern
        pattern = DiscoveredPattern(
            pattern_type=PatternType.SEQUENCE,
            pattern_data=np.random.rand(10),
            compression_ratio=0.6,
            confidence=0.7,
            utility_score=0.5
        )
        
        # Learn from outcome
        pattern_discovery_curiosity.learn_from_action_outcome(
            pattern, 2, {'success': True, 'reward': 0.9}, {'test': True}
        )
        
        # Check that utility was updated
        assert pattern.utility_score != 0.5  # Should have been updated
    
    def test_pattern_confidence_scoring(self, pattern_discovery_curiosity):
        """Test pattern confidence scoring for gut feeling engine."""
        pattern_data = np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]])  # Symmetric pattern
        confidence = pattern_discovery_curiosity.get_pattern_confidence(
            pattern_data, PatternType.SYMMETRY
        )
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_curiosity_metrics(self, pattern_discovery_curiosity):
        """Test curiosity metrics generation."""
        # Process some observations first
        for _ in range(5):
            observation = np.random.rand(8, 8)
            pattern_discovery_curiosity.process_observation(observation, {'test': True})
        
        metrics = pattern_discovery_curiosity.get_curiosity_metrics()
        
        assert isinstance(metrics, dict)
        assert 'curiosity_levels' in metrics
        assert 'total_patterns_discovered' in metrics
        assert 'pattern_type_counts' in metrics
        assert 'recent_curiosity_events' in metrics
        assert 'avg_pattern_confidence' in metrics
        assert 'avg_utility_score' in metrics

class TestEnhancedArchitecturalSystems:
    """Test suite for Enhanced Architectural Systems."""
    
    @pytest.fixture
    def mock_original_director(self):
        """Create a mock original director."""
        director = Mock()
        director.decompose_goal.return_value = [
            Mock(priority=0.8, description="subgoal1"),
            Mock(priority=0.6, description="subgoal2")
        ]
        return director
    
    @pytest.fixture
    def mock_original_architect(self):
        """Create a mock original architect."""
        architect = Mock()
        architect.evolve_architecture.return_value = (
            Mock(description="evolved_architecture"),
            {'original_metadata': 'test'}
        )
        return architect
    
    @pytest.fixture
    def mock_original_memory_manager(self):
        """Create a mock original memory manager."""
        memory_manager = Mock()
        memory_manager.store_memory.return_value = "memory_id_123"
        return memory_manager
    
    @pytest.fixture
    def mock_self_prior_manager(self):
        """Create a mock self-prior manager."""
        manager = Mock()
        manager.get_self_prior_metrics.return_value = {
            'self_awareness_score': 0.7,
            'active_goals_count': 3
        }
        return manager
    
    @pytest.fixture
    def mock_pattern_discovery_curiosity(self):
        """Create a mock pattern discovery curiosity system."""
        curiosity = Mock()
        curiosity.get_curiosity_metrics.return_value = {
            'curiosity_levels': {
                'intellectual': 0.8,
                'meta_cognitive': 0.6
            },
            'total_patterns_discovered': 5
        }
        curiosity.process_observation.return_value = {
            'patterns_discovered': 2,
            'total_curiosity_boost': 0.3
        }
        return curiosity
    
    def test_enhanced_tree_based_director_initialization(self, mock_original_director):
        """Test enhanced tree-based director initialization."""
        enhanced_director = EnhancedTreeBasedDirector(
            mock_original_director,
            enhancement_weight=0.3
        )
        
        assert enhanced_director.original_director == mock_original_director
        assert enhanced_director.enhancement_weight == 0.3
        assert enhanced_director.self_prior_manager is None
        assert enhanced_director.pattern_discovery_curiosity is None
    
    def test_enhanced_goal_decomposition(self, mock_original_director, mock_self_prior_manager, mock_pattern_discovery_curiosity):
        """Test enhanced goal decomposition."""
        enhanced_director = EnhancedTreeBasedDirector(
            mock_original_director,
            self_prior_manager=mock_self_prior_manager,
            pattern_discovery_curiosity=mock_pattern_discovery_curiosity
        )
        
        original_goal = Mock(description="test_goal")
        context = {'observation': np.random.rand(10, 10), 'test': True}
        
        enhanced_goals = enhanced_director.enhanced_goal_decomposition(original_goal, context)
        
        assert isinstance(enhanced_goals, list)
        assert len(enhanced_goals) == 2  # Based on mock return value
        
        for goal in enhanced_goals:
            assert isinstance(goal, EnhancedGoal)
            assert hasattr(goal, 'self_prior_alignment')
            assert hasattr(goal, 'curiosity_drive')
            assert hasattr(goal, 'pattern_relevance')
            assert hasattr(goal, 'intrinsic_motivation')
            assert hasattr(goal, 'priority_score')
            assert hasattr(goal, 'reasoning')
    
    def test_enhanced_tree_based_architect_initialization(self, mock_original_architect):
        """Test enhanced tree-based architect initialization."""
        enhanced_architect = EnhancedTreeBasedArchitect(
            mock_original_architect,
            motivation_weight=0.2
        )
        
        assert enhanced_architect.original_architect == mock_original_architect
        assert enhanced_architect.motivation_weight == 0.2
    
    def test_enhanced_architectural_evolution(self, mock_original_architect, mock_self_prior_manager, mock_pattern_discovery_curiosity):
        """Test enhanced architectural evolution."""
        enhanced_architect = EnhancedTreeBasedArchitect(
            mock_original_architect,
            self_prior_manager=mock_self_prior_manager,
            pattern_discovery_curiosity=mock_pattern_discovery_curiosity
        )
        
        current_architecture = Mock(description="current_arch")
        performance_metrics = {'accuracy': 0.8, 'efficiency': 0.7}
        context = {'learning_progress': 0.6, 'test': True}
        
        evolved_architecture, metadata = enhanced_architect.enhanced_architectural_evolution(
            current_architecture, performance_metrics, context
        )
        
        assert evolved_architecture is not None
        assert isinstance(metadata, dict)
        assert 'intrinsic_factors' in metadata
        assert 'enhancement_metadata' in metadata
        assert 'motivation_weight' in metadata
    
    def test_enhanced_implicit_memory_manager_initialization(self, mock_original_memory_manager):
        """Test enhanced implicit memory manager initialization."""
        enhanced_memory = EnhancedImplicitMemoryManager(
            mock_original_memory_manager,
            motivational_weight=0.3
        )
        
        assert enhanced_memory.original_memory_manager == mock_original_memory_manager
        assert enhanced_memory.motivational_weight == 0.3
    
    def test_enhanced_memory_storage(self, mock_original_memory_manager, mock_self_prior_manager, mock_pattern_discovery_curiosity):
        """Test enhanced memory storage with motivational significance."""
        enhanced_memory = EnhancedImplicitMemoryManager(
            mock_original_memory_manager,
            self_prior_manager=mock_self_prior_manager,
            pattern_discovery_curiosity=mock_pattern_discovery_curiosity
        )
        
        memory_data = {'test': 'data'}
        context = {'observation': np.random.rand(8, 8), 'learning_progress': 0.5}
        
        memory_id = enhanced_memory.enhanced_memory_storage(memory_data, context)
        
        assert memory_id == "memory_id_123"  # From mock
        assert memory_id in enhanced_memory.motivational_significance_scores
        assert len(enhanced_memory.motivational_clusters) > 0
    
    def test_enhancement_metrics(self, mock_original_director, mock_original_architect, mock_original_memory_manager):
        """Test enhancement metrics generation."""
        # Test director metrics
        enhanced_director = EnhancedTreeBasedDirector(mock_original_director)
        director_metrics = enhanced_director.get_enhancement_metrics()
        assert isinstance(director_metrics, dict)
        
        # Test architect metrics
        enhanced_architect = EnhancedTreeBasedArchitect(mock_original_architect)
        architect_metrics = enhanced_architect.get_enhancement_metrics()
        assert isinstance(architect_metrics, dict)
        
        # Test memory manager metrics
        enhanced_memory = EnhancedImplicitMemoryManager(mock_original_memory_manager)
        memory_metrics = enhanced_memory.get_motivational_clusters()
        assert isinstance(memory_metrics, dict)

class TestIntegration:
    """Integration tests for all high priority enhancements."""
    
    def test_full_system_integration(self):
        """Test full system integration."""
        # Create all components
        self_prior_manager = SelfPriorManager(
            visual_dim=32,
            proprioceptive_dim=8,
            tactile_dim=4,
            auditory_dim=16,
            latent_dim=16
        )
        
        pattern_discovery_curiosity = PatternDiscoveryCuriosity(
            compression_methods=['gzip', 'pca'],
            learning_rate=0.1
        )
        
        # Create mock original systems
        mock_director = Mock()
        mock_director.decompose_goal.return_value = [Mock(priority=0.8)]
        
        mock_architect = Mock()
        mock_architect.evolve_architecture.return_value = (Mock(), {})
        
        mock_memory = Mock()
        mock_memory.store_memory.return_value = "test_id"
        
        # Create enhanced systems
        enhanced_director = EnhancedTreeBasedDirector(
            mock_director,
            self_prior_manager=self_prior_manager,
            pattern_discovery_curiosity=pattern_discovery_curiosity
        )
        
        enhanced_architect = EnhancedTreeBasedArchitect(
            mock_architect,
            self_prior_manager=self_prior_manager,
            pattern_discovery_curiosity=pattern_discovery_curiosity
        )
        
        enhanced_memory = EnhancedImplicitMemoryManager(
            mock_memory,
            self_prior_manager=self_prior_manager,
            pattern_discovery_curiosity=pattern_discovery_curiosity
        )
        
        # Test integration
        # 1. Process sensory experience
        sensory_result = self_prior_manager.process_sensory_experience(
            visual_features=np.random.rand(32),
            proprioceptive_state=np.random.rand(8),
            tactile_feedback=np.random.rand(4),
            auditory_input=np.random.rand(16)
        )
        assert 'intrinsic_goals' in sensory_result
        
        # 2. Process observation for pattern discovery
        pattern_result = pattern_discovery_curiosity.process_observation(
            np.random.rand(5, 5), {'test': True}
        )
        assert 'patterns_discovered' in pattern_result
        
        # 3. Enhanced goal decomposition
        enhanced_goals = enhanced_director.enhanced_goal_decomposition(
            Mock(), {'observation': np.random.rand(5, 5)}
        )
        assert len(enhanced_goals) > 0
        
        # 4. Enhanced memory storage
        memory_id = enhanced_memory.enhanced_memory_storage(
            {'test': 'data'}, {'observation': np.random.rand(5, 5)}
        )
        assert memory_id is not None
        
        # 5. Get metrics from all systems
        self_prior_metrics = self_prior_manager.get_self_prior_metrics()
        curiosity_metrics = pattern_discovery_curiosity.get_curiosity_metrics()
        director_metrics = enhanced_director.get_enhancement_metrics()
        architect_metrics = enhanced_architect.get_enhancement_metrics()
        memory_metrics = enhanced_memory.get_motivational_clusters()
        
        assert all(isinstance(metrics, dict) for metrics in [
            self_prior_metrics, curiosity_metrics, director_metrics, 
            architect_metrics, memory_metrics
        ])
    
    def test_performance_benchmarking(self):
        """Test performance of the enhanced systems."""
        import time
        
        # Create systems with correct dimensions
        self_prior_manager = SelfPriorManager(
            visual_dim=64,
            proprioceptive_dim=16,
            tactile_dim=8,
            auditory_dim=32,
            latent_dim=32
        )
        pattern_discovery_curiosity = PatternDiscoveryCuriosity()
        
        # Benchmark self-prior processing
        start_time = time.time()
        for _ in range(10):
            self_prior_manager.process_sensory_experience(
                visual_features=np.random.rand(64),
                proprioceptive_state=np.random.rand(16),
                tactile_feedback=np.random.rand(8),
                auditory_input=np.random.rand(32)
            )
        self_prior_time = time.time() - start_time
        
        # Benchmark pattern discovery
        start_time = time.time()
        for _ in range(10):
            pattern_discovery_curiosity.process_observation(
                np.random.rand(8, 8), {'test': True}
            )
        pattern_discovery_time = time.time() - start_time
        
        # Assert reasonable performance (should complete in under 2 seconds for 10 iterations)
        assert self_prior_time < 2.0
        assert pattern_discovery_time < 2.0
        
        print(f"Self-prior processing time: {self_prior_time:.3f}s")
        print(f"Pattern discovery time: {pattern_discovery_time:.3f}s")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
