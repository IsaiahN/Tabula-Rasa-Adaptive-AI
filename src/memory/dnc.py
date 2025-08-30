"""
Differentiable Neural Computer (DNC) implementation for embedded memory.

This provides external memory that integrates with the agent's forward pass,
addressing the "external notebook" critique while maintaining proven stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import numpy as np


class DNCMemory(nn.Module):
    """
    Differentiable Neural Computer memory system.
    
    Based on Graves et al. 2016 with Hebbian-inspired addressing bonuses.
    """
    
    def __init__(
        self,
        memory_size: int = 512,
        word_size: int = 64,
        num_read_heads: int = 4,
        num_write_heads: int = 1,
        controller_size: int = 256
    ):
        super().__init__()
        
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.controller_size = controller_size
        
        # Memory matrix and usage tracking
        self.register_buffer('memory_matrix', torch.zeros(memory_size, word_size))
        self.register_buffer('usage_vector', torch.zeros(memory_size))
        self.register_buffer('write_weights_history', torch.zeros(memory_size))
        self.register_buffer('read_weights_history', torch.zeros(memory_size))
        
        # Temporal linking for sequence memory
        self.register_buffer('link_matrix', torch.zeros(memory_size, memory_size))
        self.register_buffer('precedence_weights', torch.zeros(memory_size))
        
        # Controller networks - input size will be set dynamically
        self.controller_size = controller_size
        self.controller = None  # Will be initialized on first forward pass
        
        # Interface layer will be initialized on first forward pass
        self.interface_layer = None
        
        # Initialize parameters
        self._reset_memory()
        
    def _reset_memory(self):
        """Reset memory to initial state."""
        self.memory_matrix.fill_(0.0)
        self.usage_vector.fill_(0.0)
        self.write_weights_history.fill_(0.0)
        self.read_weights_history.fill_(0.0)
        self.link_matrix.fill_(0.0)
        self.precedence_weights.fill_(0.0)
        
    def forward(
        self, 
        input_data: torch.Tensor,
        prev_reads: torch.Tensor,
        controller_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass through DNC.
        
        Args:
            input_data: Input to controller [batch_size, input_size]
            prev_reads: Previous read vectors [batch_size, num_read_heads * word_size]
            controller_state: Previous LSTM state
            
        Returns:
            read_vectors: Current read vectors [batch_size, num_read_heads * word_size]
            controller_output: Controller hidden state [batch_size, controller_size]
            new_controller_state: New LSTM state
            debug_info: Dictionary with debugging information
        """
        batch_size = input_data.size(0)
        
        # Initialize controller on first forward pass
        if self.controller is None:
            controller_input_size = input_data.size(-1) + prev_reads.size(-1)
            self.controller = nn.LSTM(
                input_size=controller_input_size,
                hidden_size=self.controller_size,
                batch_first=True
            )
            
            # Initialize interface layer
            interface_size = (
                self.num_read_heads * self.word_size +  # Read keys
                self.num_read_heads +              # Read strengths
                self.num_write_heads * self.word_size + # Write keys
                self.num_write_heads +             # Write strengths
                self.num_write_heads * self.word_size + # Write vectors
                self.num_write_heads * self.word_size + # Erase vectors
                self.num_write_heads +             # Free gates
                self.num_write_heads +             # Allocation gates
                self.num_write_heads +             # Write gates
                self.num_read_heads * 3            # Read modes (backward, content, forward)
            )
            self.interface_layer = nn.Linear(self.controller_size, interface_size)
        
        # Controller forward pass
        controller_input = torch.cat([input_data, prev_reads], dim=-1)
        controller_output, new_controller_state = self.controller(
            controller_input.unsqueeze(1), controller_state
        )
        controller_output = controller_output.squeeze(1)
        
        # Generate interface parameters
        interface_params = self.interface_layer(controller_output)
        
        # Parse interface parameters
        params = self._parse_interface_params(interface_params)
        
        # Memory operations
        read_vectors, write_info = self._memory_operations(params, batch_size)
        
        # Update memory state
        self._update_memory_state(write_info)
        
        # Prepare debug info
        debug_info = {
            'memory_usage': self.usage_vector.mean(),
            'write_weights': write_info['write_weights'],
            'read_weights': write_info['read_weights'],
            'memory_utilization': (self.usage_vector > 0.1).float().mean()
        }
        
        return read_vectors, controller_output, new_controller_state, debug_info
        
    def _parse_interface_params(self, interface_params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Parse interface parameters from controller output."""
        batch_size = interface_params.size(0)
        
        # Split interface parameters
        splits = []
        offset = 0
        
        # Read parameters
        read_keys_size = self.num_read_heads * self.word_size
        read_keys = interface_params[:, offset:offset + read_keys_size]
        read_keys = read_keys.view(batch_size, self.num_read_heads, self.word_size)
        offset += read_keys_size
        
        read_strengths = F.softplus(interface_params[:, offset:offset + self.num_read_heads]) + 1
        offset += self.num_read_heads
        
        # Write parameters
        write_keys_size = self.num_write_heads * self.word_size
        write_keys = interface_params[:, offset:offset + write_keys_size]
        write_keys = write_keys.view(batch_size, self.num_write_heads, self.word_size)
        offset += write_keys_size
        
        write_strengths = F.softplus(interface_params[:, offset:offset + self.num_write_heads]) + 1
        offset += self.num_write_heads
        
        write_vectors = interface_params[:, offset:offset + write_keys_size]
        write_vectors = write_vectors.view(batch_size, self.num_write_heads, self.word_size)
        offset += write_keys_size
        
        erase_vectors = torch.sigmoid(interface_params[:, offset:offset + write_keys_size])
        erase_vectors = erase_vectors.view(batch_size, self.num_write_heads, self.word_size)
        offset += write_keys_size
        
        # Gates
        free_gates = torch.sigmoid(interface_params[:, offset:offset + self.num_write_heads])
        offset += self.num_write_heads
        
        allocation_gates = torch.sigmoid(interface_params[:, offset:offset + self.num_write_heads])
        offset += self.num_write_heads
        
        write_gates = torch.sigmoid(interface_params[:, offset:offset + self.num_write_heads])
        offset += self.num_write_heads
        
        # Read modes
        read_modes = F.softmax(
            interface_params[:, offset:offset + self.num_read_heads * 3].view(
                batch_size, self.num_read_heads, 3
            ), dim=-1
        )
        
        return {
            'read_keys': read_keys,
            'read_strengths': read_strengths,
            'write_keys': write_keys,
            'write_strengths': write_strengths,
            'write_vectors': write_vectors,
            'erase_vectors': erase_vectors,
            'free_gates': free_gates,
            'allocation_gates': allocation_gates,
            'write_gates': write_gates,
            'read_modes': read_modes
        }
        
    def _memory_operations(
        self, 
        params: Dict[str, torch.Tensor], 
        batch_size: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Perform memory read and write operations."""
        
        # Content-based addressing
        write_content_weights = self._content_addressing(
            params['write_keys'], params['write_strengths']
        )
        read_content_weights = self._content_addressing(
            params['read_keys'], params['read_strengths']
        )
        
        # Allocation weights for writing
        allocation_weights = self._allocation_addressing()
        
        # Combine content and allocation for write weights
        write_weights = (
            params['write_gates'].unsqueeze(-1) * (
                params['allocation_gates'].unsqueeze(-1) * allocation_weights.unsqueeze(0) +
                (1 - params['allocation_gates'].unsqueeze(-1)) * write_content_weights
            )
        )
        
        # Temporal addressing for reading
        read_weights = self._temporal_addressing(
            read_content_weights, params['read_modes']
        )
        
        # Add Hebbian-inspired co-activation bonus
        read_weights = self._add_hebbian_bonus(read_weights, write_weights)
        
        # Ensure read_weights has correct batch dimension
        if read_weights.size(0) != batch_size:
            # Expand to match batch size if needed
            read_weights = read_weights.expand(batch_size, -1, -1)
        
        # Perform reads
        # The read_weights should be [batch_size, num_read_heads, memory_size]
        # We need to add a dimension for matrix multiplication
        read_vectors = torch.matmul(
            read_weights.unsqueeze(-2),  # [batch_size, num_read_heads, 1, memory_size]
            self.memory_matrix.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_read_heads, -1, -1)  # [batch_size, num_read_heads, memory_size, word_size]
        ).squeeze(-2)  # [batch_size, num_read_heads, word_size]
        
        # Perform writes
        self._perform_writes(
            write_weights, params['erase_vectors'], params['write_vectors'],
            params['free_gates']
        )
        
        write_info = {
            'write_weights': write_weights,
            'read_weights': read_weights,
            'allocation_weights': allocation_weights
        }
        
        return read_vectors.view(batch_size, -1), write_info
        
    def _content_addressing(
        self, 
        keys: torch.Tensor, 
        strengths: torch.Tensor
    ) -> torch.Tensor:
        """Content-based addressing using cosine similarity."""
        batch_size, num_heads, word_size = keys.shape
        
        # Normalize keys and memory
        keys_norm = F.normalize(keys, dim=-1)
        memory_norm = F.normalize(self.memory_matrix, dim=-1)
        
        # Compute cosine similarities
        similarities = torch.matmul(
            keys_norm.view(batch_size * num_heads, word_size),
            memory_norm.t()
        ).view(batch_size, num_heads, self.memory_size)
        
        # Apply strength and softmax
        weights = F.softmax(similarities * strengths.unsqueeze(-1), dim=-1)
        
        return weights
        
    def _allocation_addressing(self) -> torch.Tensor:
        """Allocation-based addressing for finding free memory locations."""
        # Sort usage vector to find least used locations
        sorted_usage, indices = torch.sort(self.usage_vector)
        
        # Create allocation weights (prefer least used locations)
        allocation_weights = torch.zeros_like(self.usage_vector)
        
        # Allocate to least used locations
        num_allocate = min(10, self.memory_size)  # Allocate to top 10 least used
        for i in range(num_allocate):
            idx = indices[i]
            allocation_weights[idx] = (num_allocate - i) / num_allocate
            
        # Normalize
        allocation_weights = allocation_weights / (allocation_weights.sum() + 1e-8)
        
        return allocation_weights
        
    def _temporal_addressing(
        self, 
        content_weights: torch.Tensor, 
        read_modes: torch.Tensor
    ) -> torch.Tensor:
        """Temporal addressing using link matrix."""
        batch_size, num_heads, _ = content_weights.shape
        
        # Forward and backward weights from link matrix
        forward_weights = torch.matmul(
            content_weights.view(batch_size * num_heads, 1, self.memory_size),
            self.link_matrix.unsqueeze(0).expand(batch_size * num_heads, -1, -1)
        ).squeeze(1).view(batch_size, num_heads, self.memory_size)
        
        backward_weights = torch.matmul(
            content_weights.view(batch_size * num_heads, 1, self.memory_size),
            self.link_matrix.t().unsqueeze(0).expand(batch_size * num_heads, -1, -1)
        ).squeeze(1).view(batch_size, num_heads, self.memory_size)
        
        # Combine using read modes
        read_weights = (
            read_modes[:, :, 0:1] * backward_weights +
            read_modes[:, :, 1:2] * content_weights +
            read_modes[:, :, 2:3] * forward_weights
        )
        
        return read_weights
        
    def _add_hebbian_bonus(
        self, 
        read_weights: torch.Tensor, 
        write_weights: torch.Tensor
    ) -> torch.Tensor:
        """Add Hebbian-inspired co-activation bonus to addressing."""
        # For simplicity, just return the original read weights for now
        # The Hebbian bonus can be added later when the system is more stable
        return read_weights
        
    def _perform_writes(
        self,
        write_weights: torch.Tensor,
        erase_vectors: torch.Tensor,
        write_vectors: torch.Tensor,
        free_gates: torch.Tensor
    ):
        """Perform memory write operations."""
        batch_size = write_weights.size(0)
        
        # Average across batch for memory update
        avg_write_weights = write_weights.mean(dim=0)
        avg_erase_vectors = erase_vectors.mean(dim=0)
        avg_write_vectors = write_vectors.mean(dim=0)
        avg_free_gates = free_gates.mean(dim=0)
        
        # Erase operation
        for head in range(self.num_write_heads):
            erase_weights = avg_write_weights[head].unsqueeze(-1)
            erase_vector = avg_erase_vectors[head].unsqueeze(0)
            
            self.memory_matrix = self.memory_matrix * (
                1 - erase_weights * erase_vector
            )
            
            # Write operation
            write_vector = avg_write_vectors[head].unsqueeze(0)
            self.memory_matrix = self.memory_matrix + erase_weights * write_vector
            
    def _update_memory_state(self, write_info: Dict[str, torch.Tensor]):
        """Update memory usage and temporal links."""
        write_weights = write_info['write_weights'].mean(dim=0)  # Average across batch
        read_weights = write_info['read_weights'].mean(dim=0)    # Average across batch
        
        # Update usage vector
        for head in range(self.num_write_heads):
            self.usage_vector = (
                self.usage_vector + write_weights[head] - 
                self.usage_vector * write_weights[head]
            )
            
        # Update precedence weights and link matrix
        for head in range(self.num_write_heads):
            # Update precedence
            self.precedence_weights = (
                (1 - write_weights[head].sum()) * self.precedence_weights +
                write_weights[head]
            )
            
            # Update link matrix
            write_weights_expanded = write_weights[head].unsqueeze(-1)
            precedence_expanded = self.precedence_weights.unsqueeze(0)
            
            self.link_matrix = (
                (1 - write_weights_expanded - write_weights_expanded.t()) * self.link_matrix +
                write_weights_expanded * precedence_expanded
            )
            
        # Decay link matrix to prevent overflow
        self.link_matrix = self.link_matrix * 0.99
        
    def get_memory_metrics(self) -> Dict[str, float]:
        """Get memory usage and health metrics."""
        return {
            'memory_utilization': float((self.usage_vector > 0.1).float().mean().detach()),
            'average_usage': float(self.usage_vector.mean().detach()),
            'max_usage': float(self.usage_vector.max().detach()),
            'link_matrix_norm': float(self.link_matrix.norm().detach()),
            'memory_diversity': float(torch.std(self.memory_matrix, dim=0).mean().detach())
        }
        
    def reset_memory(self):
        """Reset memory to initial state."""
        self._reset_memory()
    
    def retrieve_salient_memories(
        self, 
        current_context: torch.Tensor, 
        salience_threshold: float = 0.6,
        max_retrievals: int = 5
    ) -> List[Tuple[torch.Tensor, float]]:
        """
        Retrieve memories based on context similarity and historical salience.
        
        When the agent is in a situation similar to a past high-salience event,
        those memories are primed and recalled more easily.
        
        Args:
            current_context: Current state/context vector
            salience_threshold: Minimum salience for retrieval
            max_retrievals: Maximum number of memories to retrieve
            
        Returns:
            List of (memory_vector, relevance_score) tuples
        """
        if not hasattr(self, 'memory_salience_map'):
            # Initialize salience tracking if not present
            self.register_buffer('memory_salience_map', torch.zeros(self.memory_size))
            return []
        
        # Compute context similarity with stored memories
        current_context_norm = F.normalize(current_context.unsqueeze(0), dim=-1)
        memory_norm = F.normalize(self.memory_matrix, dim=-1)
        
        # Cosine similarity between current context and all memories
        similarities = torch.matmul(current_context_norm, memory_norm.t()).squeeze(0)
        
        # Weight similarities by salience values
        salience_weighted_similarities = similarities * self.memory_salience_map
        
        # Filter by salience threshold
        valid_mask = self.memory_salience_map >= salience_threshold
        if not valid_mask.any():
            return []
        
        # Get top retrievals
        valid_similarities = salience_weighted_similarities[valid_mask]
        valid_indices = torch.nonzero(valid_mask).squeeze(-1)
        
        # Sort by weighted similarity
        sorted_indices = torch.argsort(valid_similarities, descending=True)
        top_indices = valid_indices[sorted_indices[:max_retrievals]]
        
        # Return memory vectors and their relevance scores
        retrieved_memories = []
        for idx in top_indices:
            memory_vector = self.memory_matrix[idx]
            relevance_score = salience_weighted_similarities[idx].item()
            retrieved_memories.append((memory_vector, relevance_score))
        
        return retrieved_memories
    
    def update_memory_salience(self, memory_indices: torch.Tensor, salience_values: torch.Tensor):
        """
        Update salience values for specific memory locations.
        
        Args:
            memory_indices: Indices of memory locations to update
            salience_values: New salience values for those locations
        """
        if not hasattr(self, 'memory_salience_map'):
            self.register_buffer('memory_salience_map', torch.zeros(self.memory_size))
        
        # Update salience values using exponential moving average
        alpha = 0.1
        for idx, salience in zip(memory_indices, salience_values):
            if 0 <= idx < self.memory_size:
                current_salience = self.memory_salience_map[idx]
                self.memory_salience_map[idx] = (
                    alpha * salience + (1 - alpha) * current_salience
                )
    
    def get_high_salience_memories(self, threshold: float = 0.7) -> List[Tuple[int, torch.Tensor, float]]:
        """
        Get all memories above a salience threshold.
        
        Returns:
            List of (index, memory_vector, salience) tuples
        """
        if not hasattr(self, 'memory_salience_map'):
            return []
        
        high_salience_mask = self.memory_salience_map >= threshold
        if not high_salience_mask.any():
            return []
        
        high_salience_memories = []
        high_salience_indices = torch.nonzero(high_salience_mask).squeeze(-1)
        
        for idx in high_salience_indices:
            memory_vector = self.memory_matrix[idx]
            salience = self.memory_salience_map[idx].item()
            high_salience_memories.append((idx.item(), memory_vector, salience))
        
        # Sort by salience (highest first)
        high_salience_memories.sort(key=lambda x: x[2], reverse=True)
        
        return high_salience_memories