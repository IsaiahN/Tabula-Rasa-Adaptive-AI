"""
Predictive Core Module

Implements the core predictive processing system that integrates memory, attention,
and learning mechanisms for the adaptive learning agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import logging

# Fix relative imports by using absolute imports within the package
from memory.dnc import DNCMemory
from .energy_system import EnergySystem
from .data_models import SensoryInput  # Add missing import

logger = logging.getLogger(__name__)


class PredictiveCore(nn.Module):
    """
    Recurrent state-space model for sensory prediction with integrated memory.
    
    This is the agent's world-model that predicts next sensory states and
    maintains temporal context through recurrent dynamics.
    """
    
    def __init__(
        self,
        visual_size: Tuple[int, int, int] = (3, 64, 64),
        proprioception_size: int = 12,
        hidden_size: int = 512,
        memory_config: Optional[Dict] = None,
        architecture: str = "lstm"  # "lstm", "gru", or "mamba"
    ):
        super().__init__()
        
        self.visual_size = visual_size
        self.proprioception_size = proprioception_size
        self.hidden_size = hidden_size
        self.architecture = architecture
        
        # Calculate input dimensions
        # visual_size is (channels, height, width), not (batch, height, width)
        visual_dim = visual_size[0] * visual_size[1] * visual_size[2]  # channels * height * width
        total_input_dim = visual_dim + proprioception_size + 1  # +1 for energy level
        
        # Input encoder
        self.input_encoder = nn.Sequential(
            nn.Linear(total_input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Recurrent core
        if architecture == "lstm":
            self.recurrent_core = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                batch_first=True
            )
        elif architecture == "gru":
            self.recurrent_core = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                batch_first=True
            )
        else:
            # Fallback to LSTM if mamba not available
            logger.warning(f"Architecture {architecture} not implemented, using LSTM")
            self.recurrent_core = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                batch_first=True
            )
            self.architecture = "lstm"
            
        # Memory system (optional)
        self.use_memory = memory_config is not None
        if self.use_memory:
            self.memory = DNCMemory(**memory_config)
            memory_read_size = memory_config['num_read_heads'] * memory_config['word_size']
        else:
            self.memory = None
            memory_read_size = 0
            
        # Prediction heads - will be calculated dynamically based on memory state
        self.base_hidden_size = hidden_size
        self.memory_read_size = memory_read_size
        
    def get_prediction_input_size(self) -> int:
        """Get the current prediction input size based on memory state."""
        if self.use_memory and self.memory is not None:
            return self.base_hidden_size + self.memory.num_read_heads * self.memory.word_size
        else:
            return self.base_hidden_size
        
    def _create_prediction_heads(self, input_size: int) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
        """Create prediction heads with the correct input size."""
        visual_dim = self.visual_size[0] * self.visual_size[1] * self.visual_size[2]
        
        visual_predictor = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, visual_dim),
            nn.Tanh()
        )
        
        proprio_predictor = nn.Sequential(
            nn.Linear(input_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.proprioception_size)
        )
        
        energy_predictor = nn.Sequential(
            nn.Linear(input_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        confidence_estimator = nn.Sequential(
            nn.Linear(input_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, 3),
            nn.Sigmoid()
        )
        
        return visual_predictor, proprio_predictor, energy_predictor, confidence_estimator
        
        # Store dimensions for dynamic prediction heads
        self.hidden_size = hidden_size
        self.visual_size = visual_size
        self.proprioception_size = proprioception_size
        
    def encode_sensory_input(self, sensory_input: SensoryInput) -> torch.Tensor:
        """
        Encode sensory input into internal representation.
        
        Args:
            sensory_input: Multi-modal sensory data
            
        Returns:
            encoded: Encoded representation
        """
        # Handle case where inputs don't have batch dimension
        if sensory_input.visual.dim() == 3:
            # Add batch dimension if missing
            visual = sensory_input.visual.unsqueeze(0)
            proprio = sensory_input.proprioception.unsqueeze(0) if sensory_input.proprioception.dim() == 1 else sensory_input.proprioception
            batch_size = 1
        else:
            visual = sensory_input.visual
            proprio = sensory_input.proprioception
            batch_size = visual.size(0)
        
        # Flatten visual input
        visual_flat = visual.view(batch_size, -1)
        
        # Ensure proprioception has batch dimension
        if proprio.dim() == 1:
            proprio = proprio.unsqueeze(0)
        
        # Normalize energy to 0-1 range
        energy_norm = torch.tensor([sensory_input.energy_level / 100.0]).expand(batch_size, 1)
        if visual_flat.device != energy_norm.device:
            energy_norm = energy_norm.to(visual_flat.device)
            
        # Concatenate all inputs
        combined_input = torch.cat([
            visual_flat,
            proprio,
            energy_norm
        ], dim=-1)
        
        # Encode
        encoded = self.input_encoder(combined_input)
        
        return encoded
        
    def forward(
        self,
        sensory_input: SensoryInput,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        memory_reads: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict]:
        """
        Forward pass through predictive core.
        
        Args:
            sensory_input: Current sensory input
            hidden_state: Previous recurrent state
            memory_reads: Previous memory reads
            
        Returns:
            visual_pred: Visual prediction
            proprio_pred: Proprioception prediction  
            energy_pred: Energy prediction
            new_hidden_state: Updated recurrent state
            debug_info: Debug information
        """
        batch_size = sensory_input.visual.size(0)
        
        # Encode input
        encoded_input = self.encode_sensory_input(sensory_input)
        
        # Recurrent processing
        if self.architecture == "lstm":
            if hidden_state is None:
                hidden_state = (
                    torch.zeros(1, batch_size, self.hidden_size, device=encoded_input.device),
                    torch.zeros(1, batch_size, self.hidden_size, device=encoded_input.device)
                )
            recurrent_output, new_hidden_state = self.recurrent_core(
                encoded_input.unsqueeze(1), hidden_state
            )
            recurrent_output = recurrent_output.squeeze(1)
        else:  # GRU
            if hidden_state is None:
                hidden_state = torch.zeros(1, batch_size, self.hidden_size, device=encoded_input.device)
            recurrent_output, new_hidden_state = self.recurrent_core(
                encoded_input.unsqueeze(1), hidden_state
            )
            recurrent_output = recurrent_output.squeeze(1)
            new_hidden_state = (new_hidden_state, new_hidden_state)  # Make compatible with LSTM format
            
        # Memory processing
        debug_info = {}
        if self.use_memory and self.memory is not None:
            if memory_reads is None:
                memory_reads = torch.zeros(
                    batch_size, 
                    self.memory.num_read_heads * self.memory.word_size,
                    device=encoded_input.device
                )
                
            # Memory forward pass
            new_memory_reads, controller_output, controller_state, memory_debug = self.memory(
                encoded_input, memory_reads
            )
            
            # Combine recurrent output with memory reads
            combined_output = torch.cat([recurrent_output, new_memory_reads], dim=-1)
            debug_info.update(memory_debug)
        else:
            combined_output = recurrent_output
            new_memory_reads = memory_reads
            
        # Generate predictions using dynamic prediction heads
        prediction_input_size = combined_output.size(-1)
        visual_predictor, proprio_predictor, energy_predictor, confidence_estimator = self._create_prediction_heads(prediction_input_size)
        
        visual_pred = visual_predictor(combined_output)
        proprio_pred = proprio_predictor(combined_output)
        energy_pred = energy_predictor(combined_output)
        
        # Reshape visual prediction
        visual_pred = visual_pred.view(batch_size, *self.visual_size)
        
        # Confidence estimation
        confidence = confidence_estimator(combined_output)
        
        debug_info.update({
            'confidence': confidence,
            'recurrent_norm': torch.norm(recurrent_output, dim=-1).mean(),
            'prediction_norms': {
                'visual': torch.norm(visual_pred.view(batch_size, -1), dim=-1).mean(),
                'proprio': torch.norm(proprio_pred, dim=-1).mean(),
                'energy': torch.norm(energy_pred, dim=-1).mean()
            }
        })
        
        return visual_pred, proprio_pred, energy_pred, new_hidden_state, debug_info
        
    def compute_prediction_error(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        actual: SensoryInput
    ) -> Dict[str, torch.Tensor]:
        """
        Compute prediction errors for each modality.
        
        Args:
            predictions: (visual_pred, proprio_pred, energy_pred)
            actual: Actual sensory input
            
        Returns:
            errors: Dictionary of prediction errors
        """
        visual_pred, proprio_pred, energy_pred = predictions
        batch_size = visual_pred.size(0)
        
        # Visual error (MSE)
        visual_error = F.mse_loss(
            visual_pred, 
            actual.visual, 
            reduction='none'
        ).view(batch_size, -1).mean(dim=-1)
        
        # Proprioception error (MSE)
        proprio_error = F.mse_loss(
            proprio_pred,
            actual.proprioception,
            reduction='none'
        ).mean(dim=-1)
        
        # Energy error (MSE)
        energy_target = torch.tensor([actual.energy_level / 100.0]).expand(batch_size, 1)
        if energy_pred.device != energy_target.device:
            energy_target = energy_target.to(energy_pred.device)
            
        energy_error = F.mse_loss(
            energy_pred,
            energy_target,
            reduction='none'
        ).squeeze(-1)
        
        # Combined error (weighted average)
        total_error = 0.5 * visual_error + 0.3 * proprio_error + 0.2 * energy_error
        
        return {
            'visual': visual_error,
            'proprioception': proprio_error,
            'energy': energy_error,
            'total': total_error
        }
        
    def get_state_representation(
        self,
        sensory_input: SensoryInput,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Get internal state representation for goal clustering.
        
        Args:
            sensory_input: Current sensory input
            hidden_state: Current recurrent state
            
        Returns:
            state_repr: Internal state representation
        """
        with torch.no_grad():
            encoded = self.encode_sensory_input(sensory_input)
            
            if hidden_state is not None:
                if self.architecture == "lstm":
                    # Use hidden state from LSTM
                    state_repr = hidden_state[0].squeeze(0)  # Remove sequence dimension
                else:
                    state_repr = hidden_state.squeeze(0)
            else:
                # Use encoded input if no hidden state
                state_repr = encoded
                
        return state_repr
        
    def reset_memory(self):
        """Reset memory system if present."""
        if self.use_memory and self.memory is not None:
            self.memory.reset_memory()