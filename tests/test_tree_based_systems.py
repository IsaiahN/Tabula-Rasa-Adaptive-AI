#!/usr/bin/env python3
"""
Tests for Tree-Based Systems

Tests the Tree-Based Director, Tree-Based Architect, and Implicit Memory Manager
systems that provide enhanced reasoning, self-improvement, and memory capabilities.
"""

import pytest
import sys
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.tree_based_director import (
    TreeBasedDirector,
    create_tree_based_director,
    ReasoningNodeType,
    GoalPriority
)

from src.core.tree_based_architect import (
    TreeBasedArchitect,
    create_tree_based_architect,
    EvolutionNodeType,
    MutationType,
    FitnessMetric
)

from src.core.implicit_memory_manager import (
    ImplicitMemoryManager,
    create_implicit_memory_manager,
    MemoryType,
    CompressionLevel,
    MemoryPriority
)


class TestTreeBasedDirector:
    """Test the Tree-Based Director functionality."""
    
    def test_director_initialization(self):
        """Test director initialization."""
        director = create_tree_based_director()
        
        assert director is not None
        assert hasattr(director, 'active_traces')
        assert hasattr(director, 'completed_traces')
        assert hasattr(director, 'reasoning_stats')
    
    def test_create_reasoning_trace(self):
        """Test creating a reasoning trace."""
        director = create_tree_based_director()
        
        trace_id = director.create_reasoning_trace(
            root_goal="Win the current game",
            context={'game_id': 'test_game'},
            priority=GoalPriority.HIGH
        )
        
        assert trace_id is not None
        assert trace_id in director.active_traces
        
        trace = director.active_traces[trace_id]
        assert trace.root_goal == "Win the current game"
        assert len(trace.nodes) == 1  # Root node
        assert trace.reasoning_depth == 0
    
    def test_decompose_goal(self):
        """Test goal decomposition."""
        director = create_tree_based_director()
        
        trace_id = director.create_reasoning_trace(
            root_goal="Win the current game",
            context={'game_id': 'test_game'}
        )
        
        decomposition = director.decompose_goal(trace_id, "Win the current game")
        
        assert decomposition.main_goal == "Win the current game"
        assert len(decomposition.sub_goals) > 0
        assert len(decomposition.strategies) > 0
        assert len(decomposition.constraints) > 0
        assert len(decomposition.success_criteria) > 0
        assert decomposition.confidence > 0.0
    
    def test_add_reasoning_step(self):
        """Test adding reasoning steps."""
        director = create_tree_based_director()
        
        trace_id = director.create_reasoning_trace(
            root_goal="Test goal",
            context={}
        )
        
        step_id = director.add_reasoning_step(
            trace_id=trace_id,
            step_type=ReasoningNodeType.STRATEGY,
            content="Use pattern recognition strategy",
            confidence=0.8
        )
        
        assert step_id is not None
        trace = director.active_traces[trace_id]
        assert step_id in trace.nodes
        assert trace.reasoning_depth == 1
    
    def test_synthesize_reasoning(self):
        """Test reasoning synthesis."""
        director = create_tree_based_director()
        
        trace_id = director.create_reasoning_trace(
            root_goal="Test goal",
            context={}
        )
        
        # Add some reasoning steps
        director.add_reasoning_step(trace_id, ReasoningNodeType.STRATEGY, "Strategy 1")
        director.add_reasoning_step(trace_id, ReasoningNodeType.CONSTRAINT, "Constraint 1")
        
        synthesis = director.synthesize_reasoning(trace_id)
        
        assert 'overall_confidence' in synthesis
        assert 'reasoning_depth' in synthesis
        assert 'total_nodes' in synthesis
        assert 'recommendations' in synthesis
        assert synthesis['total_nodes'] > 0
    
    def test_complete_trace(self):
        """Test completing a reasoning trace."""
        director = create_tree_based_director()
        
        trace_id = director.create_reasoning_trace(
            root_goal="Test goal",
            context={}
        )
        
        completion = director.complete_trace(trace_id, success=True)
        
        assert completion['success'] is True
        assert trace_id not in director.active_traces
        assert len(director.completed_traces) == 1
        assert director.reasoning_stats['total_traces'] == 1


class TestTreeBasedArchitect:
    """Test the Tree-Based Architect functionality."""
    
    def test_architect_initialization(self):
        """Test architect initialization."""
        architect = create_tree_based_architect()
        
        assert architect is not None
        assert hasattr(architect, 'active_traces')
        assert hasattr(architect, 'completed_traces')
        assert hasattr(architect, 'evolution_stats')
    
    def test_create_self_model(self):
        """Test creating a self-model."""
        architect = create_tree_based_architect()
        
        current_architecture = {
            'layers': ['input', 'hidden', 'output'],
            'parameters': {'learning_rate': 0.01}
        }
        performance_metrics = {'accuracy': 0.85, 'loss': 0.2}
        learning_patterns = {'convergence_rate': 0.1}
        
        model_id = architect.create_self_model(
            current_architecture=current_architecture,
            performance_metrics=performance_metrics,
            learning_patterns=learning_patterns
        )
        
        assert model_id is not None
        assert architect.current_self_model is not None
        assert architect.current_self_model.model_id == model_id
    
    def test_generate_evolution_trace(self):
        """Test generating an evolution trace."""
        architect = create_tree_based_architect()
        
        # Create self-model first
        architect.create_self_model(
            current_architecture={'test': 'architecture'},
            performance_metrics={'test': 0.5},
            learning_patterns={'test': 'pattern'}
        )
        
        trace_id = architect.generate_evolution_trace(
            target_improvement="Improve learning rate",
            current_performance={'accuracy': 0.8}
        )
        
        assert trace_id is not None
        assert trace_id in architect.active_traces
        
        trace = architect.active_traces[trace_id]
        assert trace.root_mutation == "Improve learning rate"
        assert len(trace.nodes) == 1  # Root node
    
    def test_evolve_architecture(self):
        """Test architecture evolution."""
        architect = create_tree_based_architect()
        
        # Create self-model and evolution trace
        architect.create_self_model(
            current_architecture={'test': 'architecture'},
            performance_metrics={'test': 0.5},
            learning_patterns={'test': 'pattern'}
        )
        
        trace_id = architect.generate_evolution_trace(
            target_improvement="Improve performance",
            current_performance={'accuracy': 0.8}
        )
        
        changes = architect.evolve_architecture(trace_id, mutation_strategy="balanced")
        
        assert 'parameter_changes' in changes
        assert 'structural_changes' in changes
        assert 'strategy_changes' in changes
        assert 'overall_confidence' in changes
        assert 'expected_improvement' in changes
    
    def test_complete_evolution_trace(self):
        """Test completing an evolution trace."""
        architect = create_tree_based_architect()
        
        # Create self-model and evolution trace
        architect.create_self_model(
            current_architecture={'test': 'architecture'},
            performance_metrics={'test': 0.5},
            learning_patterns={'test': 'pattern'}
        )
        
        trace_id = architect.generate_evolution_trace(
            target_improvement="Test improvement",
            current_performance={'accuracy': 0.8}
        )
        
        completion = architect.complete_evolution_trace(trace_id, success=True, final_fitness=0.9)
        
        assert completion['success'] is True
        assert completion['final_fitness'] == 0.9
        assert trace_id not in architect.active_traces
        assert len(architect.completed_traces) == 1


class TestImplicitMemoryManager:
    """Test the Implicit Memory Manager functionality."""
    
    def test_memory_manager_initialization(self):
        """Test memory manager initialization."""
        manager = create_implicit_memory_manager()
        
        assert manager is not None
        assert hasattr(manager, 'memories')
        assert hasattr(manager, 'memory_clusters')
        assert hasattr(manager, 'compression_stats')
    
    def test_store_memory(self):
        """Test storing memories."""
        manager = create_implicit_memory_manager()
        
        memory_id = manager.store_memory(
            content="This is a test memory",
            memory_type=MemoryType.EPISODIC,
            priority=MemoryPriority.HIGH,
            compression_level=CompressionLevel.MEDIUM
        )
        
        assert memory_id is not None
        assert memory_id in manager.memories
        
        memory_node = manager.memories[memory_id]
        assert memory_node.content == "This is a test memory"
        assert memory_node.memory_type == MemoryType.EPISODIC
        assert memory_node.priority == MemoryPriority.HIGH
    
    def test_retrieve_memory(self):
        """Test retrieving memories."""
        manager = create_implicit_memory_manager()
        
        memory_id = manager.store_memory(
            content="Test memory content",
            memory_type=MemoryType.SEMANTIC
        )
        
        retrieved_content = manager.retrieve_memory(memory_id)
        
        assert retrieved_content == "Test memory content"
        
        # Test with decompression
        retrieved_decompressed = manager.retrieve_memory(memory_id, decompress=True)
        assert retrieved_decompressed == "Test memory content"
    
    def test_search_memories(self):
        """Test searching memories."""
        manager = create_implicit_memory_manager()
        
        # Store some test memories
        manager.store_memory("Learning about patterns", MemoryType.LEARNING)
        manager.store_memory("Game strategy for winning", MemoryType.STRATEGY)
        manager.store_memory("Memory management techniques", MemoryType.MEMORY)
        
        # Search for memories
        results = manager.search_memories("patterns", limit=5)
        
        assert len(results) > 0
        assert all('relevance_score' in result for result in results)
        assert all('memory_id' in result for result in results)
    
    def test_cluster_memories(self):
        """Test clustering memories."""
        manager = create_implicit_memory_manager()
        
        # Store multiple memories of the same type
        for i in range(10):
            manager.store_memory(
                f"Learning memory {i} about patterns and strategies",
                MemoryType.LEARNING
            )
        
        clusters = manager.cluster_memories(MemoryType.LEARNING, max_clusters=3)
        
        assert len(clusters) > 0
        assert all(hasattr(cluster, 'cluster_id') for cluster in clusters)
        assert all(hasattr(cluster, 'memories') for cluster in clusters)
    
    def test_compression_levels(self):
        """Test different compression levels."""
        manager = create_implicit_memory_manager()
        
        test_content = "This is a test content for compression testing. " * 100
        
        # Test different compression levels
        for level in CompressionLevel:
            memory_id = manager.store_memory(
                content=test_content,
                memory_type=MemoryType.PATTERN,
                compression_level=level
            )
            
            memory_node = manager.memories[memory_id]
            assert memory_node.compression_level == level
            
            # Verify content can be retrieved
            retrieved = manager.retrieve_memory(memory_id)
            assert retrieved == test_content
    
    def test_memory_limits(self):
        """Test memory limit enforcement."""
        manager = create_implicit_memory_manager(max_memory_mb=0.001)  # Very small limit
        
        # Store memories until limit is reached
        memory_ids = []
        for i in range(100):
            memory_id = manager.store_memory(
                content=f"Large memory content {i} " * 1000,  # Large content
                memory_type=MemoryType.EPISODIC,
                priority=MemoryPriority.LOW
            )
            memory_ids.append(memory_id)
        
        # Check that memory usage is within limits
        stats = manager.get_memory_stats()
        assert stats['memory_usage_mb'] <= manager.max_memory_mb * 1.1  # Allow some tolerance


class TestIntegration:
    """Test integration between the three systems."""
    
    def test_director_architect_integration(self):
        """Test integration between Director and Architect."""
        director = create_tree_based_director()
        architect = create_tree_based_architect()
        
        # Director creates reasoning trace
        trace_id = director.create_reasoning_trace(
            root_goal="Improve system performance",
            context={'current_performance': 0.7}
        )
        
        # Architect creates self-model
        architect.create_self_model(
            current_architecture={'performance': 0.7},
            performance_metrics={'accuracy': 0.7},
            learning_patterns={'convergence': 0.1}
        )
        
        # Architect generates evolution trace
        evolution_trace_id = architect.generate_evolution_trace(
            target_improvement="Improve system performance",
            current_performance={'accuracy': 0.7}
        )
        
        # Both systems should work independently
        assert trace_id in director.active_traces
        assert evolution_trace_id in architect.active_traces
    
    def test_memory_director_integration(self):
        """Test integration between Memory Manager and Director."""
        memory_manager = create_implicit_memory_manager()
        director = create_tree_based_director()
        
        # Store reasoning patterns in memory
        reasoning_pattern = "Use pattern recognition for complex problems"
        memory_id = memory_manager.store_memory(
            content=reasoning_pattern,
            memory_type=MemoryType.META_COGNITIVE,
            priority=MemoryPriority.HIGH
        )
        
        # Director creates reasoning trace
        trace_id = director.create_reasoning_trace(
            root_goal="Solve complex problem",
            context={'problem_type': 'complex'}
        )
        
        # Search for relevant patterns
        results = memory_manager.search_memories("pattern recognition")
        
        assert len(results) > 0
        assert any(result['content'] == reasoning_pattern for result in results)
    
    def test_all_systems_integration(self):
        """Test integration of all three systems."""
        director = create_tree_based_director()
        architect = create_tree_based_architect()
        memory_manager = create_implicit_memory_manager()
        
        # Store learning patterns
        memory_manager.store_memory(
            "Adaptive learning improves performance",
            MemoryType.LEARNING,
            MemoryPriority.HIGH
        )
        
        # Director reasons about improvement
        trace_id = director.create_reasoning_trace(
            root_goal="Improve learning system",
            context={'current_learning_rate': 0.01}
        )
        
        # Architect models self-improvement
        architect.create_self_model(
            current_architecture={'learning_rate': 0.01},
            performance_metrics={'learning_efficiency': 0.6},
            learning_patterns={'adaptation': 0.1}
        )
        
        evolution_trace_id = architect.generate_evolution_trace(
            target_improvement="Improve learning system",
            current_performance={'learning_efficiency': 0.6}
        )
        
        # All systems should work together
        assert trace_id in director.active_traces
        assert evolution_trace_id in architect.active_traces
        assert len(memory_manager.memories) > 0
        
        # Test cross-system data flow
        director_synthesis = director.synthesize_reasoning(trace_id)
        architect_changes = architect.evolve_architecture(evolution_trace_id)
        memory_search = memory_manager.search_memories("learning")
        
        assert director_synthesis['overall_confidence'] > 0
        assert architect_changes['overall_confidence'] > 0
        assert len(memory_search) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
