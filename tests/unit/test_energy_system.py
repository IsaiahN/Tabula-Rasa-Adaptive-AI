"""
Unit tests for Energy and Death System.
"""

import pytest
import torch
import numpy as np
from src.core.energy_system import EnergySystem, DeathManager, HeuristicImportanceScorer
from src.core.data_models import AgentState


class TestEnergySystem:
    """Test suite for Energy System."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.energy_system = EnergySystem(
            max_energy=100.0,
            base_consumption=0.01,
            action_multiplier=0.5,
            computation_multiplier=0.001
        )
        
    def test_initialization(self):
        """Test proper initialization."""
        assert self.energy_system.max_energy == 100.0
        assert self.energy_system.current_energy == 100.0
        assert self.energy_system.base_consumption == 0.01
        
    def test_energy_consumption(self):
        """Test energy consumption mechanics."""
        initial_energy = self.energy_system.current_energy
        
        # Consume energy with action and computation costs
        remaining = self.energy_system.consume_energy(
            action_cost=0.5,  # Medium action
            computation_cost=10.0  # Some computation
        )
        
        # Energy should decrease
        assert remaining < initial_energy
        assert remaining == self.energy_system.current_energy
        
        # Should track consumption
        assert len(self.energy_system.consumption_history) == 1
        assert len(self.energy_system.energy_history) == 1
        
    def test_energy_addition(self):
        """Test energy addition (food consumption)."""
        # Consume some energy first
        self.energy_system.consume_energy(action_cost=1.0)
        energy_after_consumption = self.energy_system.current_energy
        
        # Add energy
        new_energy = self.energy_system.add_energy(10.0)
        
        # Energy should increase
        assert new_energy > energy_after_consumption
        assert new_energy == self.energy_system.current_energy
        
    def test_energy_capping(self):
        """Test that energy is capped at maximum."""
        # Try to add more energy than maximum
        final_energy = self.energy_system.add_energy(200.0)
        
        # Should be capped at max
        assert final_energy == self.energy_system.max_energy
        
    def test_death_detection(self):
        """Test death detection when energy reaches zero."""
        # Consume all energy
        while not self.energy_system.is_dead():
            self.energy_system.consume_energy(action_cost=1.0)
            
        assert self.energy_system.is_dead()
        assert self.energy_system.current_energy <= 0.0
        
    def test_sleep_trigger(self):
        """Test sleep mode trigger."""
        # Consume energy to low level
        while self.energy_system.current_energy > 15.0:
            self.energy_system.consume_energy(action_cost=0.1)
            
        assert self.energy_system.should_sleep()
        
    def test_energy_reset(self):
        """Test energy reset for respawn."""
        # Consume some energy
        self.energy_system.consume_energy(action_cost=1.0)
        
        # Reset
        self.energy_system.reset_energy()
        
        assert self.energy_system.current_energy == self.energy_system.max_energy
        
    def test_energy_metrics(self):
        """Test energy metrics computation."""
        # Generate some energy history
        for i in range(10):
            self.energy_system.consume_energy(action_cost=0.1 * i)
            
        metrics = self.energy_system.get_energy_metrics()
        
        # Check all metrics are present
        expected_metrics = [
            'current_energy',
            'energy_ratio',
            'average_consumption',
            'energy_trend'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))


class TestHeuristicImportanceScorer:
    """Test suite for heuristic importance scoring."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = HeuristicImportanceScorer()
        self.memory_size = 32
        self.word_size = 8
        
    def test_importance_scoring(self):
        """Test basic importance scoring."""
        memory_matrix = torch.randn(self.memory_size, self.word_size)
        usage_history = torch.rand(self.memory_size)
        
        scores = self.scorer.score(memory_matrix, usage_history)
        
        # Should return scores for all memory locations
        assert scores.shape == (self.memory_size,)
        
        # Scores should be non-negative
        assert torch.all(scores >= 0)
        
    def test_usage_preference(self):
        """Test that frequently used memories get higher scores."""
        memory_matrix = torch.randn(self.memory_size, self.word_size)
        
        # Create usage pattern where first half is highly used
        usage_history = torch.zeros(self.memory_size)
        usage_history[:self.memory_size//2] = 1.0  # High usage
        usage_history[self.memory_size//2:] = 0.1  # Low usage
        
        scores = self.scorer.score(memory_matrix, usage_history)
        
        # High usage locations should have higher scores
        high_usage_scores = scores[:self.memory_size//2].mean()
        low_usage_scores = scores[self.memory_size//2:].mean()
        
        assert high_usage_scores > low_usage_scores


class TestDeathManager:
    """Test suite for Death Manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.death_manager = DeathManager(
            memory_size=32,
            word_size=8,
            use_learned_importance=False,
            preservation_ratio=0.2
        )
        
    def test_initialization(self):
        """Test proper initialization."""
        assert self.death_manager.memory_size == 32
        assert self.death_manager.word_size == 8
        assert not self.death_manager.use_learned_importance
        assert self.death_manager.preservation_ratio == 0.2
        assert self.death_manager.death_count == 0
        
    def test_memory_importance_computation(self):
        """Test memory importance computation."""
        memory_matrix = torch.randn(32, 8)
        usage_history = torch.rand(32)
        
        importance = self.death_manager.compute_memory_importance(
            memory_matrix, usage_history
        )
        
        # Should return importance scores for all locations
        assert importance.shape == (32,)
        assert torch.all(importance >= 0)
        
    def test_selective_reset(self):
        """Test selective reset preserving important memories."""
        # Create mock agent state
        agent_state = AgentState(
            position=torch.tensor([1.0, 2.0, 3.0]),
            orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
            energy=50.0,
            hidden_state=torch.randn(64),
            active_goals=[],
            memory_state=torch.randn(32, 8)
        )
        
        # Add memory usage attribute
        agent_state.memory_usage = torch.rand(32)
        
        # Perform selective reset
        new_state = self.death_manager.selective_reset(agent_state)
        
        # Check reset properties
        assert new_state.energy == 100.0  # Full energy
        assert torch.allclose(new_state.hidden_state, torch.zeros_like(agent_state.hidden_state))
        assert len(new_state.active_goals) == 0
        
        # Memory should be partially preserved
        if new_state.memory_state is not None:
            preserved_locations = (new_state.memory_state != 0).any(dim=1).sum()
            expected_preserved = int(32 * 0.2)  # 20% preservation
            assert preserved_locations <= expected_preserved + 2  # Allow some tolerance
            
        # Death count should increment
        assert self.death_manager.death_count == 1
        
    def test_death_metrics(self):
        """Test death metrics computation."""
        # Simulate some deaths
        for _ in range(3):
            agent_state = AgentState(
                position=torch.zeros(3),
                orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
                energy=0.0,
                hidden_state=torch.randn(64),
                active_goals=[],
                memory_state=torch.randn(32, 8)
            )
            self.death_manager.selective_reset(agent_state)
            
        metrics = self.death_manager.get_death_metrics()
        
        # Check metrics
        assert metrics['death_count'] == 3
        assert 'average_recovery_time' in metrics
        assert 'preservation_ratio' in metrics


if __name__ == '__main__':
    pytest.main([__file__])