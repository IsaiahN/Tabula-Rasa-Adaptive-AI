"""
Unit tests for Differentiable Neural Computer memory system.
"""

import pytest
import torch
import torch.nn as nn
from src.memory.dnc import DNCMemory


class TestDNCMemory:
    """Test suite for DNC memory system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.memory_size = 64
        self.word_size = 16
        self.num_read_heads = 2
        self.num_write_heads = 1
        self.controller_size = 32
        
        self.dnc = DNCMemory(
            memory_size=self.memory_size,
            word_size=self.word_size,
            num_read_heads=self.num_read_heads,
            num_write_heads=self.num_write_heads,
            controller_size=self.controller_size
        )
        
    def test_initialization(self):
        """Test proper initialization of DNC."""
        assert self.dnc.memory_size == self.memory_size
        assert self.dnc.word_size == self.word_size
        assert self.dnc.num_read_heads == self.num_read_heads
        assert self.dnc.num_write_heads == self.num_write_heads
        
        # Check memory matrices are initialized to zero
        assert torch.allclose(self.dnc.memory_matrix, torch.zeros(self.memory_size, self.word_size))
        assert torch.allclose(self.dnc.usage_vector, torch.zeros(self.memory_size))
        
    def test_forward_pass_shapes(self):
        """Test that forward pass produces correct output shapes."""
        batch_size = 4
        input_size = 8
        
        input_data = torch.randn(batch_size, input_size)
        prev_reads = torch.zeros(batch_size, self.num_read_heads * self.word_size)
        
        read_vectors, controller_output, new_state, debug_info = self.dnc(
            input_data, prev_reads
        )
        
        # Check output shapes
        expected_read_shape = (batch_size, self.num_read_heads * self.word_size)
        expected_controller_shape = (batch_size, self.controller_size)
        
        assert read_vectors.shape == expected_read_shape
        assert controller_output.shape == expected_controller_shape
        
        # Check debug info
        assert 'memory_usage' in debug_info
        assert 'write_weights' in debug_info
        assert 'read_weights' in debug_info
        
    def test_memory_operations(self):
        """Test basic memory read/write operations."""
        batch_size = 2
        input_size = 8
        
        # Write some data to memory
        input_data = torch.randn(batch_size, input_size)
        prev_reads = torch.zeros(batch_size, self.num_read_heads * self.word_size)
        
        # First forward pass (should write to memory)
        read_vectors1, _, state1, debug1 = self.dnc(input_data, prev_reads)
        
        # Memory should have some content now
        memory_norm_before = torch.norm(self.dnc.memory_matrix)
        
        # Second forward pass with different input
        input_data2 = torch.randn(batch_size, input_size) + 1.0
        read_vectors2, _, state2, debug2 = self.dnc(input_data2, read_vectors1, state1)
        
        memory_norm_after = torch.norm(self.dnc.memory_matrix)
        
        # Memory should have changed
        assert memory_norm_after > memory_norm_before
        
    def test_content_addressing(self):
        """Test content-based addressing mechanism."""
        # Manually set some memory content
        test_pattern = torch.randn(self.word_size)
        self.dnc.memory_matrix[0] = test_pattern
        
        # Create key that matches the pattern
        keys = test_pattern.unsqueeze(0).unsqueeze(0)  # [1, 1, word_size]
        strengths = torch.tensor([[5.0]])  # High strength
        
        weights = self.dnc._content_addressing(keys, strengths)
        
        # Should have high weight for location 0
        assert weights[0, 0, 0] > 0.5
        
    def test_allocation_addressing(self):
        """Test allocation-based addressing for free memory."""
        # Set some locations as highly used
        self.dnc.usage_vector[:10] = 0.9
        
        allocation_weights = self.dnc._allocation_addressing()
        
        # Should prefer unused locations
        unused_weight = allocation_weights[20:].sum()
        used_weight = allocation_weights[:10].sum()
        
        assert unused_weight > used_weight
        
    def test_memory_reset(self):
        """Test memory reset functionality."""
        # Write some data
        self.dnc.memory_matrix.fill_(1.0)
        self.dnc.usage_vector.fill_(0.5)
        
        # Reset
        self.dnc.reset_memory()
        
        # Should be back to zeros
        assert torch.allclose(self.dnc.memory_matrix, torch.zeros_like(self.dnc.memory_matrix))
        assert torch.allclose(self.dnc.usage_vector, torch.zeros_like(self.dnc.usage_vector))
        
    def test_memory_metrics(self):
        """Test memory metrics computation."""
        # Write some data to memory
        batch_size = 1
        input_size = 8
        
        for _ in range(5):
            input_data = torch.randn(batch_size, input_size)
            prev_reads = torch.zeros(batch_size, self.num_read_heads * self.word_size)
            self.dnc(input_data, prev_reads)
            
        metrics = self.dnc.get_memory_metrics()
        
        # Check all metrics are present
        expected_metrics = [
            'memory_utilization',
            'average_usage',
            'max_usage',
            'link_matrix_norm',
            'memory_diversity'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
            assert not torch.isnan(torch.tensor(metrics[metric]))


if __name__ == '__main__':
    pytest.main([__file__])