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
import time
import numpy as np

# Fix relative imports by using absolute imports within the package
from memory.dnc import DNCMemory
from .energy_system import EnergySystem
from .data_models import SensoryInput  # Add missing import
from .simulation_models import (
    SimulationResult, SimulationStep, SimulationStatus, 
    SimulationHypothesis, SimulationContext
)

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
    
    def simulate_rollout(self, 
                        initial_state: Dict[str, Any],
                        hypothesis: SimulationHypothesis,
                        max_steps: int = 10,
                        timeout: float = 1.0) -> SimulationResult:
        """
        Run a multi-step simulation internally without taking real actions.
        
        This is the core of the "imagination" - the AI can now "think ahead"
        by simulating multiple action sequences and their outcomes.
        """
        start_time = time.time()
        simulation_state = initial_state.copy()
        simulation_history = []
        
        try:
            # Initialize simulation state
            current_energy = simulation_state.get('energy', 100.0)
            current_position = simulation_state.get('position', [0.0, 0.0, 0.0])
            current_orientation = simulation_state.get('orientation', [0.0, 0.0, 0.0, 1.0])
            
            # Run simulation steps
            for step, (action, coords) in enumerate(hypothesis.action_sequence[:max_steps]):
                # Check timeout
                if time.time() - start_time > timeout:
                    logger.debug(f"Simulation timeout after {step} steps")
                    break
                
                # Predict next state without taking real action
                predicted_state, confidence = self._predict_simulation_step(
                    simulation_state, action, coords
                )
                
                # Calculate energy change
                new_energy = predicted_state.get('energy', current_energy)
                energy_change = new_energy - current_energy
                
                # Estimate learning progress
                learning_progress = self._estimate_learning_progress(
                    simulation_state, predicted_state, action
                )
                
                # Create simulation step
                sim_step = SimulationStep(
                    step=step,
                    action=action,
                    coordinates=coords,
                    predicted_state=predicted_state,
                    energy_change=energy_change,
                    learning_progress=learning_progress,
                    confidence=confidence,
                    reasoning=f"Simulated action {action} with coords {coords}"
                )
                
                simulation_history.append(sim_step)
                
                # Update simulation state
                simulation_state.update(predicted_state)
                current_energy = new_energy
                
                # Check for early termination conditions
                if self._should_terminate_simulation(simulation_state, step):
                    logger.debug(f"Simulation terminated early at step {step}")
                    break
            
            # Calculate success metrics
            success_metrics = self._calculate_simulation_metrics(simulation_history)
            
            # Create simulation result
            result = SimulationResult(
                hypothesis=hypothesis,
                status=SimulationStatus.COMPLETED,
                final_state=simulation_state,
                simulation_history=simulation_history,
                success_metrics=success_metrics,
                total_energy_cost=sum(step.energy_change for step in simulation_history),
                total_learning_gain=sum(step.learning_progress for step in simulation_history),
                execution_time=time.time() - start_time,
                terminated_early=len(simulation_history) < len(hypothesis.action_sequence),
                termination_reason="timeout" if time.time() - start_time > timeout else ""
            )
            
            logger.debug(f"Simulation completed: {len(simulation_history)} steps, "
                        f"energy cost: {result.total_energy_cost:.2f}, "
                        f"learning gain: {result.total_learning_gain:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return SimulationResult(
                hypothesis=hypothesis,
                status=SimulationStatus.FAILED,
                final_state=simulation_state,
                simulation_history=simulation_history,
                success_metrics={},
                total_energy_cost=0.0,
                total_learning_gain=0.0,
                execution_time=time.time() - start_time,
                terminated_early=True,
                termination_reason=f"error: {str(e)}"
            )
    
    def _predict_simulation_step(self, 
                                current_state: Dict[str, Any],
                                action: int,
                                coords: Optional[Tuple[int, int]]) -> Tuple[Dict[str, Any], float]:
        """
        Predict the next state for a single simulation step.
        This is a simplified prediction that doesn't require full sensory input.
        """
        
        # Create a simplified sensory input for prediction
        sensory_input = self._create_simulation_sensory_input(current_state, action, coords)
        
        # Encode the input
        with torch.no_grad():
            encoded = self.encode_sensory_input(sensory_input)
            
            # Use a simple forward pass for prediction
            if hasattr(self, 'visual_predictor') and self.visual_predictor is not None:
                # Use existing prediction heads if available
                prediction_input_size = self.get_prediction_input_size()
                if encoded.size(-1) != prediction_input_size:
                    # Pad or truncate to match expected size
                    if encoded.size(-1) < prediction_input_size:
                        padding = torch.zeros(encoded.size(0), prediction_input_size - encoded.size(-1))
                        encoded = torch.cat([encoded, padding], dim=-1)
                    else:
                        encoded = encoded[:, :prediction_input_size]
                
                # Get predictions
                visual_pred = self.visual_predictor(encoded)
                proprio_pred = self.proprioception_predictor(encoded)
                energy_pred = self.energy_predictor(encoded)
                
                # Convert predictions to state updates
                predicted_state = self._convert_predictions_to_state(
                    visual_pred, proprio_pred, energy_pred, current_state
                )
            else:
                # Fallback: simple state update based on action
                predicted_state = self._simple_action_prediction(current_state, action, coords)
        
        # Calculate confidence based on prediction consistency
        confidence = self._calculate_prediction_confidence(current_state, predicted_state, action)
        
        return predicted_state, confidence
    
    def _create_simulation_sensory_input(self, 
                                       current_state: Dict[str, Any],
                                       action: int,
                                       coords: Optional[Tuple[int, int]]) -> SensoryInput:
        """Create a simplified sensory input for simulation."""
        
        # Create minimal visual input (zeros for simulation)
        visual_data = torch.zeros(self.visual_size)
        
        # Create proprioception data from current state
        position = current_state.get('position', [0.0, 0.0, 0.0])
        orientation = current_state.get('orientation', [0.0, 0.0, 0.0, 1.0])
        velocity = current_state.get('velocity', [0.0, 0.0, 0.0])
        energy = current_state.get('energy', 100.0)
        
        proprioception_data = torch.tensor([
            *position,
            *orientation,
            *velocity,
            energy / 100.0  # Normalize energy
        ], dtype=torch.float32)
        
        return SensoryInput(
            visual=visual_data,
            proprioception=proprioception_data,
            timestamp=time.time()
        )
    
    def _convert_predictions_to_state(self, 
                                    visual_pred: torch.Tensor,
                                    proprio_pred: torch.Tensor,
                                    energy_pred: torch.Tensor,
                                    current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert prediction outputs to state updates."""
        
        predicted_state = current_state.copy()
        
        # Update energy
        energy_change = energy_pred.item() * 10.0  # Scale energy change
        predicted_state['energy'] = max(0, min(100, current_state.get('energy', 100) + energy_change))
        
        # Update position (simplified)
        if proprio_pred.size(-1) >= 3:
            position_delta = proprio_pred[:3].detach().numpy()
            current_pos = np.array(current_state.get('position', [0.0, 0.0, 0.0]))
            predicted_state['position'] = (current_pos + position_delta * 0.1).tolist()
        
        # Update orientation (simplified)
        if proprio_pred.size(-1) >= 7:
            orientation_delta = proprio_pred[3:7].detach().numpy()
            current_orient = np.array(current_state.get('orientation', [0.0, 0.0, 0.0, 1.0]))
            predicted_state['orientation'] = (current_orient + orientation_delta * 0.1).tolist()
        
        return predicted_state
    
    def _simple_action_prediction(self, 
                                current_state: Dict[str, Any],
                                action: int,
                                coords: Optional[Tuple[int, int]]) -> Dict[str, Any]:
        """Simple fallback prediction based on action type."""
        
        predicted_state = current_state.copy()
        
        # Simple energy consumption model
        energy_consumption = {
            1: 0.5,  # Movement actions
            2: 0.5,
            3: 0.5,
            4: 0.5,
            5: 1.0,  # Interaction actions
            6: 2.0,  # Coordinate actions (more complex)
            7: 0.1   # Undo actions (minimal cost)
        }
        
        energy_cost = energy_consumption.get(action, 1.0)
        predicted_state['energy'] = max(0, current_state.get('energy', 100) - energy_cost)
        
        # Simple position update for movement actions
        if action in [1, 2, 3, 4]:  # Movement actions
            current_pos = np.array(current_state.get('position', [0.0, 0.0, 0.0]))
            movement_delta = {
                1: [0, 0, -0.1],   # Up
                2: [0, 0, 0.1],    # Down
                3: [-0.1, 0, 0],   # Left
                4: [0.1, 0, 0]     # Right
            }
            
            if action in movement_delta:
                delta = movement_delta[action]
                predicted_state['position'] = (current_pos + np.array(delta)).tolist()
        
        return predicted_state
    
    def _estimate_learning_progress(self, 
                                  current_state: Dict[str, Any],
                                  predicted_state: Dict[str, Any],
                                  action: int) -> float:
        """Estimate learning progress from state transition."""
        
        # Simple learning progress estimation
        # Higher progress for actions that lead to energy gain or new positions
        progress = 0.0
        
        # Energy-based learning
        energy_change = predicted_state.get('energy', 0) - current_state.get('energy', 0)
        if energy_change > 0:
            progress += 0.1  # Learning from energy gain
        
        # Position-based learning
        current_pos = np.array(current_state.get('position', [0.0, 0.0, 0.0]))
        predicted_pos = np.array(predicted_state.get('position', [0.0, 0.0, 0.0]))
        position_change = np.linalg.norm(predicted_pos - current_pos)
        if position_change > 0.01:
            progress += 0.05  # Learning from movement
        
        # Action-specific learning
        if action == 6 and coords is not None:  # Coordinate actions
            progress += 0.2  # Higher learning potential for coordinate actions
        
        return min(progress, 1.0)  # Cap at 1.0
    
    def _calculate_prediction_confidence(self, 
                                       current_state: Dict[str, Any],
                                       predicted_state: Dict[str, Any],
                                       action: int) -> float:
        """Calculate confidence in prediction."""
        
        # Simple confidence calculation
        confidence = 0.5  # Base confidence
        
        # Higher confidence for simpler actions
        if action in [1, 2, 3, 4]:  # Movement actions
            confidence += 0.2
        elif action == 7:  # Undo actions
            confidence += 0.1
        elif action == 6:  # Coordinate actions
            confidence -= 0.1  # Lower confidence for complex actions
        
        # Adjust based on energy level (lower energy = lower confidence)
        energy_level = current_state.get('energy', 100)
        if energy_level < 30:
            confidence -= 0.2
        elif energy_level > 70:
            confidence += 0.1
        
        return max(0.1, min(1.0, confidence))  # Clamp between 0.1 and 1.0
    
    def _should_terminate_simulation(self, 
                                   simulation_state: Dict[str, Any],
                                   step: int) -> bool:
        """Check if simulation should be terminated early."""
        
        # Terminate if energy is too low
        if simulation_state.get('energy', 100) < 10:
            return True
        
        # Terminate if we've been running too long
        if step > 20:
            return True
        
        # Terminate if position is out of bounds (simplified check)
        position = simulation_state.get('position', [0.0, 0.0, 0.0])
        if any(abs(coord) > 100 for coord in position):
            return True
        
        return False
    
    def _calculate_simulation_metrics(self, 
                                    simulation_history: List[SimulationStep]) -> Dict[str, float]:
        """Calculate success metrics for the simulation."""
        
        if not simulation_history:
            return {
                'success_rate': 0.0,
                'energy_efficiency': 0.0,
                'learning_efficiency': 0.0,
                'action_diversity': 0.0,
                'confidence_avg': 0.0
            }
        
        # Calculate metrics
        total_energy_cost = sum(step.energy_change for step in simulation_history)
        total_learning_gain = sum(step.learning_progress for step in simulation_history)
        
        # Success rate based on learning progress
        successful_steps = sum(1 for step in simulation_history if step.learning_progress > 0.01)
        success_rate = successful_steps / len(simulation_history)
        
        # Energy efficiency (learning per energy cost)
        energy_efficiency = total_learning_gain / max(abs(total_energy_cost), 0.1)
        
        # Learning efficiency (learning per step)
        learning_efficiency = total_learning_gain / len(simulation_history)
        
        # Action diversity
        actions = [step.action for step in simulation_history]
        action_diversity = len(set(actions)) / len(actions) if actions else 0.0
        
        # Average confidence
        confidence_avg = sum(step.confidence for step in simulation_history) / len(simulation_history)
        
        return {
            'success_rate': success_rate,
            'energy_efficiency': energy_efficiency,
            'learning_efficiency': learning_efficiency,
            'action_diversity': action_diversity,
            'confidence_avg': confidence_avg
        }