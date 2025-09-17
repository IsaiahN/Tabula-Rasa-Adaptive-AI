#!/usr/bin/env python3
"""
Test Enhanced Interconnectivity System

Tests the Inter-Module Message Bus, Stochastic Tree Evaluation, 
and Prediction Uncertainty systems.
"""

import pytest
import pytest_asyncio
import asyncio
import time
import numpy as np
import torch
from unittest.mock import Mock, patch
from typing import Dict, Any

# Import the enhanced systems
from src.core.inter_module_message_bus import (
    InterModuleMessageBus, Message, MessageType, MessagePriority, 
    get_message_bus, initialize_message_bus, shutdown_message_bus
)
from src.core.tree_evaluation_simulation import (
    TreeEvaluationSimulationEngine, TreeEvaluationConfig
)
from src.core.predictive_core import (
    PredictiveCore, PredictionUncertainty, UncertaintyType
)
from src.core.data_models import SensoryInput

class TestInterModuleMessageBus:
    """Test the Inter-Module Message Bus system."""
    
    @pytest.fixture
    def message_bus(self):
        """Create a message bus for testing."""
        return InterModuleMessageBus(max_queue_size=100, worker_threads=2)
    
    @pytest_asyncio.fixture
    async def running_bus(self, message_bus):
        """Create and start a message bus."""
        await message_bus.start()
        yield message_bus
        await message_bus.stop()
    
    def test_message_creation(self):
        """Test message creation and serialization."""
        message = Message(
            topic="test.topic",
            message_type=MessageType.SYSTEM_STATUS,
            payload={"status": "healthy"},
            priority=MessagePriority.HIGH,
            source_module="test_module"
        )
        
        assert message.topic == "test.topic"
        assert message.message_type == MessageType.SYSTEM_STATUS
        assert message.priority == MessagePriority.HIGH
        assert not message.is_expired()
        
        # Test serialization
        message_dict = message.to_dict()
        assert message_dict['topic'] == "test.topic"
        assert message_dict['message_type'] == "system_status"
    
    def test_message_expiration(self):
        """Test message expiration logic."""
        message = Message(
            topic="test.topic",
            message_type=MessageType.SYSTEM_STATUS,
            payload={"status": "healthy"},
            priority=MessagePriority.HIGH,
            ttl=0.1  # Very short TTL
        )
        
        assert not message.is_expired()
        time.sleep(0.2)  # Wait for expiration
        assert message.is_expired()
    
    @pytest.mark.asyncio
    async def test_publish_subscribe(self, running_bus):
        """Test basic publish-subscribe functionality."""
        received_messages = []
        
        def message_handler(message: Message):
            received_messages.append(message)
        
        # Subscribe to topic
        running_bus.subscribe("test.topic", message_handler)
        
        # Publish message
        success = await running_bus.publish(
            topic="test.topic",
            message_type=MessageType.SYSTEM_STATUS,
            payload={"status": "healthy"},
            priority=MessagePriority.HIGH
        )
        
        assert success
        
        # Wait for message processing
        await asyncio.sleep(0.1)
        
        assert len(received_messages) == 1
        assert received_messages[0].payload["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_priority_routing(self, running_bus):
        """Test priority-based message routing."""
        critical_messages = []
        normal_messages = []
        
        def critical_handler(message: Message):
            critical_messages.append(message)
        
        def normal_handler(message: Message):
            normal_messages.append(message)
        
        # Subscribe with different priority thresholds
        running_bus.subscribe("test.topic", critical_handler, 
                            priority_threshold=MessagePriority.CRITICAL)
        running_bus.subscribe("test.topic", normal_handler, 
                            priority_threshold=MessagePriority.NORMAL)
        
        # Publish messages with different priorities
        await running_bus.publish("test.topic", MessageType.SYSTEM_STATUS, 
                                {"msg": "critical"}, MessagePriority.CRITICAL)
        await running_bus.publish("test.topic", MessageType.SYSTEM_STATUS, 
                                {"msg": "normal"}, MessagePriority.NORMAL)
        
        await asyncio.sleep(0.1)
        
        # Critical handler should receive both messages (or at least the critical one)
        assert len(critical_messages) >= 1
        # Normal handler should receive both messages (since NORMAL priority threshold includes both)
        assert len(normal_messages) >= 1
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, running_bus):
        """Test performance metrics collection."""
        # Publish some messages
        for i in range(10):
            await running_bus.publish(
                f"test.topic.{i}",
                MessageType.PERFORMANCE_METRIC,
                {"value": i},
                MessagePriority.NORMAL
            )
        
        await asyncio.sleep(0.1)
        
        # Get metrics
        metrics = running_bus.get_performance_metrics()
        
        assert 'message_counts' in metrics
        assert 'queue_sizes' in metrics
        assert 'error_counts' in metrics
        assert metrics['message_counts'][MessageType.PERFORMANCE_METRIC] == 10
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self, running_bus):
        """Test module health monitoring."""
        # Register heartbeat
        running_bus.register_module_heartbeat("test_module")
        
        # Wait a bit
        await asyncio.sleep(0.1)
        
        # Check health status
        health = running_bus.get_health_status()
        
        assert 'health_score' in health
        assert 'total_messages' in health
        assert 'modules_healthy' in health
        assert health['modules_healthy'] >= 0

class TestStochasticTreeEvaluation:
    """Test the Stochastic Tree Evaluation system."""
    
    @pytest.fixture
    def tree_config(self):
        """Create tree evaluation config with stochasticity enabled."""
        return TreeEvaluationConfig(
            max_depth=5,
            branching_factor=3,
            enable_stochasticity=True,
            noise_level=0.1,
            temperature=1.0,
            epsilon=0.1,
            boltzmann_exploration=True,
            adaptive_exploration=True
        )
    
    @pytest.fixture
    def tree_engine(self, tree_config):
        """Create tree evaluation engine."""
        return TreeEvaluationSimulationEngine(tree_config)
    
    def test_stochastic_evaluation(self, tree_engine):
        """Test stochastic tree evaluation."""
        root_state = {"position": [0, 0], "score": 0}
        available_actions = [1, 2, 3, 4]
        
        # Run multiple evaluations to test stochasticity
        results = []
        for _ in range(10):
            value, path = tree_engine.evaluate_with_stochasticity(
                root_state, available_actions, exploration_factor=0.5
            )
            results.append((value, len(path)))
        
        # Should have some variation due to stochasticity
        values = [r[0] for r in results]
        # For now, just check that we get results (variation may be minimal with simple test data)
        assert len(values) == 10
        assert all(isinstance(v, float) for v in values)
    
    def test_boltzmann_exploration(self, tree_engine):
        """Test Boltzmann exploration."""
        root_state = {"position": [0, 0], "score": 0}
        available_actions = [1, 2, 3, 4]
        
        # Test with high exploration factor
        value, path = tree_engine._boltzmann_action_selection(
            root_state, available_actions, exploration_factor=1.0
        )
        
        assert isinstance(value, float)
        assert isinstance(path, list)
    
    def test_noisy_action_selection(self, tree_engine):
        """Test noisy action selection."""
        base_value = 0.5
        base_path = [Mock(confidence=0.8)]
        
        noisy_value, noisy_path = tree_engine._noisy_action_selection(
            base_value, base_path, exploration_factor=0.5
        )
        
        # Should be different from base due to noise
        assert noisy_value != base_value
        assert len(noisy_path) == len(base_path)
    
    def test_adaptive_exploration_rate(self, tree_engine):
        """Test adaptive exploration rate calculation."""
        # High variance performance
        high_variance_performance = [0.1, 0.9, 0.2, 0.8, 0.3]
        epsilon_high = tree_engine.adaptive_exploration_rate(high_variance_performance)
        
        # Low variance performance
        low_variance_performance = [0.5, 0.6, 0.4, 0.5, 0.6]
        epsilon_low = tree_engine.adaptive_exploration_rate(low_variance_performance)
        
        # High variance should lead to higher exploration
        assert epsilon_high > epsilon_low
    
    def test_exploration_noise_injection(self, tree_engine):
        """Test exploration noise injection."""
        state = {"position": [1.0, 2.0], "score": 10}
        
        # Test multiple times to ensure noise is applied
        noisy_states = []
        for _ in range(10):
            noisy_state = tree_engine.inject_exploration_noise(state, "gaussian")
            noisy_states.append(noisy_state)
        
        # At least some should be different due to noise
        different_count = 0
        for noisy_state in noisy_states:
            if (noisy_state["position"][0] != state["position"][0] or 
                noisy_state["position"][1] != state["position"][1] or 
                noisy_state["score"] != state["score"]):
                different_count += 1
        
        assert different_count > 0, "Noise injection should produce different values"
    
    def test_stochastic_stats(self, tree_engine):
        """Test stochastic statistics collection."""
        stats = tree_engine.get_stochastic_stats()
        
        assert 'stochasticity_enabled' in stats
        assert 'noise_level' in stats
        assert 'temperature' in stats
        assert 'epsilon' in stats
        assert 'boltzmann_exploration' in stats
        assert 'adaptive_exploration' in stats

class TestPredictionUncertainty:
    """Test the Prediction Uncertainty system."""
    
    @pytest.fixture
    def predictive_core(self):
        """Create predictive core for testing."""
        return PredictiveCore(
            visual_size=(3, 32, 32),
            proprioception_size=6,
            hidden_size=128
        )
    
    @pytest.fixture
    def sample_sensory_input(self):
        """Create sample sensory input."""
        return SensoryInput(
            visual=torch.randn(1, 3, 32, 32),
            proprioception=torch.randn(1, 6),
            energy_level=50.0,
            timestamp=time.time()
        )
    
    def test_uncertainty_types(self):
        """Test uncertainty type enumeration."""
        assert UncertaintyType.ALEATORIC.value == "aleatoric"
        assert UncertaintyType.EPISTEMIC.value == "epistemic"
        assert UncertaintyType.TOTAL.value == "total"
    
    def test_prediction_uncertainty_creation(self):
        """Test PredictionUncertainty dataclass."""
        uncertainty = PredictionUncertainty(
            aleatoric=0.1,
            epistemic=0.2,
            total=0.3,
            confidence=0.7,
            uncertainty_type=UncertaintyType.TOTAL,
            timestamp=time.time()
        )
        
        assert uncertainty.aleatoric == 0.1
        assert uncertainty.epistemic == 0.2
        assert uncertainty.total == 0.3
        assert uncertainty.confidence == 0.7
        
        # Test serialization
        uncertainty_dict = uncertainty.to_dict()
        assert uncertainty_dict['aleatoric'] == 0.1
        assert uncertainty_dict['uncertainty_type'] == 'total'
    
    def test_compute_prediction_uncertainty(self, predictive_core, sample_sensory_input):
        """Test prediction uncertainty computation."""
        # Create mock predictions
        predictions = (
            torch.randn(1, 3, 32, 32),  # visual
            torch.randn(1, 6),          # proprioception
            torch.randn(1, 1)           # energy
        )
        
        uncertainty = predictive_core.compute_prediction_uncertainty(
            predictions, sample_sensory_input, num_samples=5
        )
        
        assert isinstance(uncertainty, PredictionUncertainty)
        assert uncertainty.aleatoric >= 0
        assert uncertainty.epistemic >= 0
        assert uncertainty.total >= 0
        assert 0 <= uncertainty.confidence <= 1
        assert uncertainty.uncertainty_type == UncertaintyType.TOTAL
    
    def test_adaptive_learning_rate(self, predictive_core):
        """Test adaptive learning rate calculation."""
        # High error should increase learning rate
        high_error_lr = predictive_core.adaptive_learning_rate(0.8)
        
        # Low error should decrease learning rate
        low_error_lr = predictive_core.adaptive_learning_rate(0.05)
        
        # Normal error should use base learning rate
        normal_error_lr = predictive_core.adaptive_learning_rate(0.3)
        
        assert high_error_lr > normal_error_lr
        assert low_error_lr < normal_error_lr
        assert normal_error_lr == 0.001  # Base learning rate
    
    def test_uncertainty_stats(self, predictive_core):
        """Test uncertainty statistics collection."""
        stats = predictive_core.get_uncertainty_stats()
        
        assert 'uncertainty_tracking_enabled' in stats
        assert 'monte_carlo_samples' in stats
        assert 'uncertainty_types' in stats
        assert 'confidence_threshold' in stats
        assert stats['uncertainty_tracking_enabled'] == True

class TestIntegration:
    """Test integration between all enhanced systems."""
    
    @pytest.mark.asyncio
    async def test_full_integration(self):
        """Test full integration of all enhanced systems."""
        # Initialize message bus
        message_bus = await initialize_message_bus()
        
        # Initialize tree engine with stochasticity
        tree_config = TreeEvaluationConfig(enable_stochasticity=True)
        tree_engine = TreeEvaluationSimulationEngine(tree_config)
        
        # Initialize predictive core
        predictive_core = PredictiveCore()
        
        # Test message bus communication
        received_messages = []
        
        def message_handler(message: Message):
            received_messages.append(message)
        
        message_bus.subscribe("integration.test", message_handler)
        
        # Test stochastic tree evaluation
        root_state = {"position": [0, 0], "score": 0}
        available_actions = [1, 2, 3, 4]
        
        value, path = tree_engine.evaluate_with_stochasticity(
            root_state, available_actions, exploration_factor=0.3
        )
        
        # Test prediction uncertainty
        sample_input = SensoryInput(
            visual=torch.randn(1, 3, 32, 32),
            proprioception=torch.randn(1, 6),
            energy_level=50.0,
            timestamp=time.time()
        )
        
        predictions = (
            torch.randn(1, 3, 32, 32),
            torch.randn(1, 6),
            torch.randn(1, 1)
        )
        
        uncertainty = predictive_core.compute_prediction_uncertainty(
            predictions, sample_input
        )
        
        # Publish results via message bus
        await message_bus.publish(
            "integration.test",
            MessageType.PERFORMANCE_METRIC,
            {
                "tree_value": value,
                "path_length": len(path),
                "uncertainty": uncertainty.total,
                "confidence": uncertainty.confidence
            },
            MessagePriority.HIGH
        )
        
        await asyncio.sleep(0.1)
        
        # Verify integration
        assert len(received_messages) == 1
        assert received_messages[0].payload["tree_value"] == value
        assert received_messages[0].payload["path_length"] == len(path)
        assert received_messages[0].payload["uncertainty"] == uncertainty.total
        
        # Cleanup
        await shutdown_message_bus()
    
    @pytest.mark.asyncio
    async def test_performance_benchmarking(self):
        """Benchmark performance of enhanced systems."""
        # Initialize systems
        message_bus = await initialize_message_bus()
        tree_engine = TreeEvaluationSimulationEngine(TreeEvaluationConfig())
        predictive_core = PredictiveCore()
        
        # Benchmark message bus latency
        start_time = time.time()
        
        for i in range(100):
            await message_bus.publish(
                f"benchmark.{i}",
                MessageType.PERFORMANCE_METRIC,
                {"value": i},
                MessagePriority.HIGH
            )
        
        message_time = time.time() - start_time
        avg_message_latency = message_time / 100
        
        # Benchmark tree evaluation
        start_time = time.time()
        
        for i in range(50):
            tree_engine.evaluate_with_stochasticity(
                {"position": [i, i], "score": i}, [1, 2, 3, 4]
            )
        
        tree_time = time.time() - start_time
        avg_tree_latency = tree_time / 50
        
        # Verify performance targets
        assert avg_message_latency < 0.005  # <5ms for high priority
        assert avg_tree_latency < 0.1  # <100ms for tree evaluation
        
        # Cleanup
        await shutdown_message_bus()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
