"""
Sleep and Dream Cycles - Offline processing for memory consolidation.

This module implements sleep cycles that consolidate memories, prune irrelevant
information, and strengthen important patterns without active sensory input.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional, Tuple
import numpy as np
import logging
from collections import deque
import time

from core.data_models import Experience, AgentState
from core.predictive_core import PredictiveCore

logger = logging.getLogger(__name__)


class SleepCycle:
    """
    Manages sleep cycles for offline learning and memory consolidation.
    """
    
    def __init__(
        self,
        predictive_core: PredictiveCore,
        sleep_trigger_energy: float = 20.0,
        sleep_trigger_boredom_steps: int = 1000,
        sleep_trigger_memory_pressure: float = 0.9,
        sleep_duration_steps: int = 100,
        replay_batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        self.predictive_core = predictive_core
        self.sleep_trigger_energy = sleep_trigger_energy
        self.sleep_trigger_boredom_steps = sleep_trigger_boredom_steps
        self.sleep_trigger_memory_pressure = sleep_trigger_memory_pressure
        self.sleep_duration_steps = sleep_duration_steps
        self.replay_batch_size = replay_batch_size
        
        # Sleep state tracking
        self.is_sleeping = False
        self.sleep_start_time = 0
        self.sleep_cycles_completed = 0
        
        # Experience replay buffer for sleep learning
        self.replay_buffer = deque(maxlen=10000)
        self.high_error_buffer = deque(maxlen=1000)
        
        # Optimizer for offline learning
        self.sleep_optimizer = optim.Adam(
            self.predictive_core.parameters(),
            lr=learning_rate
        )
        
        # Sleep metrics
        self.sleep_metrics = {
            'total_sleep_time': 0,
            'experiences_replayed': 0,
            'memory_consolidations': 0,
            'performance_improvements': []
        }
        
    def should_sleep(
        self,
        agent_state: AgentState,
        boredom_counter: int,
        memory_usage: Optional[float] = None
    ) -> bool:
        """
        Determine if agent should enter sleep mode.
        
        Args:
            agent_state: Current agent state
            boredom_counter: Steps of low learning progress
            memory_usage: Current memory utilization (0-1)
            
        Returns:
            should_sleep: True if sleep should be triggered
        """
        # Don't sleep if already sleeping
        if self.is_sleeping:
            return False
            
        # Energy-based trigger
        if agent_state.energy <= self.sleep_trigger_energy:
            logger.info(f"Sleep triggered by low energy: {agent_state.energy}")
            return True
            
        # Boredom-based trigger
        if boredom_counter >= self.sleep_trigger_boredom_steps:
            logger.info(f"Sleep triggered by boredom: {boredom_counter} steps")
            return True
            
        # Memory pressure trigger
        if memory_usage is not None and memory_usage >= self.sleep_trigger_memory_pressure:
            logger.info(f"Sleep triggered by memory pressure: {memory_usage:.2f}")
            return True
            
        return False
        
    def enter_sleep(self, agent_state: AgentState):
        """Enter sleep mode and begin offline processing."""
        if self.is_sleeping:
            return
            
        self.is_sleeping = True
        self.sleep_start_time = time.time()
        
        logger.info("Agent entering sleep mode")
        
        # Set predictive core to training mode for sleep learning
        self.predictive_core.train()
        
    def execute_sleep_cycle(self, replay_buffer: List[Experience]) -> Dict[str, float]:
        """
        Execute one sleep cycle with offline learning.
        
        Args:
            replay_buffer: Experiences for replay learning
            
        Returns:
            sleep_results: Results of sleep cycle
        """
        if not self.is_sleeping:
            return {}
            
        sleep_results = {
            'experiences_processed': 0,
            'avg_loss': 0.0,
            'memory_operations': 0,
            'consolidation_score': 0.0
        }
        
        # Phase 1: Experience Replay
        replay_results = self._replay_experiences(replay_buffer)
        sleep_results.update(replay_results)
        
        # Phase 2: Memory Consolidation
        if self.predictive_core.use_memory:
            consolidation_results = self._consolidate_memory()
            sleep_results.update(consolidation_results)
            
        # Phase 3: Dream Generation (optional)
        dream_results = self._generate_dreams()
        sleep_results.update(dream_results)
        
        # Update sleep metrics
        self.sleep_metrics['experiences_replayed'] += sleep_results['experiences_processed']
        self.sleep_metrics['memory_consolidations'] += 1
        
        return sleep_results
        
    def _replay_experiences(self, replay_buffer: List[Experience]) -> Dict[str, float]:
        """
        Replay high-error experiences for offline learning.
        
        Args:
            replay_buffer: Buffer of experiences to replay
            
        Returns:
            replay_results: Results of experience replay
        """
        if not replay_buffer:
            return {'experiences_processed': 0, 'avg_loss': 0.0}
            
        # Sample high-error experiences
        high_error_experiences = self._sample_high_error_experiences(replay_buffer)
        
        if not high_error_experiences:
            return {'experiences_processed': 0, 'avg_loss': 0.0}
            
        total_loss = 0.0
        num_batches = 0
        
        # Process experiences in batches
        for i in range(0, len(high_error_experiences), self.replay_batch_size):
            batch = high_error_experiences[i:i + self.replay_batch_size]
            
            # Convert experiences to tensors
            states = [exp.state for exp in batch]
            next_states = [exp.next_state for exp in batch]
            
            if not states:
                continue
                
            # Batch the sensory inputs
            batch_visual = torch.stack([s.visual for s in states])
            batch_proprio = torch.stack([s.proprioception for s in states])
            batch_energy = [s.energy_level for s in states]
            
            # Create batched sensory input
            from .data_models import SensoryInput
            batched_input = SensoryInput(
                visual=batch_visual,
                proprioception=batch_proprio,
                energy_level=batch_energy[0],  # Simplified for batch
                timestamp=states[0].timestamp
            )
            
            # Forward pass
            visual_pred, proprio_pred, energy_pred, _, _ = self.predictive_core(batched_input)
            
            # Compute loss against next states
            target_visual = torch.stack([s.visual for s in next_states])
            target_proprio = torch.stack([s.proprioception for s in next_states])
            target_energy = torch.tensor([[s.energy_level / 100.0] for s in next_states])
            
            # Multi-modal loss
            visual_loss = nn.MSELoss()(visual_pred, target_visual)
            proprio_loss = nn.MSELoss()(proprio_pred, target_proprio)
            energy_loss = nn.MSELoss()(energy_pred, target_energy.to(energy_pred.device))
            
            total_loss_batch = 0.5 * visual_loss + 0.3 * proprio_loss + 0.2 * energy_loss
            
            # Backward pass
            self.sleep_optimizer.zero_grad()
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.predictive_core.parameters(), 1.0)
            
            self.sleep_optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
            
        avg_loss = total_loss / max(num_batches, 1)
        
        return {
            'experiences_processed': len(high_error_experiences),
            'avg_loss': avg_loss
        }
        
    def _sample_high_error_experiences(self, replay_buffer: List[Experience]) -> List[Experience]:
        """
        Sample experiences with high prediction errors for replay.
        
        Args:
            replay_buffer: Full experience buffer
            
        Returns:
            high_error_experiences: Experiences with high learning progress
        """
        if not replay_buffer:
            return []
            
        # Sort by learning progress (higher = more important)
        sorted_experiences = sorted(
            replay_buffer,
            key=lambda x: abs(x.learning_progress),
            reverse=True
        )
        
        # Take top experiences for replay
        num_replay = min(len(sorted_experiences), self.sleep_duration_steps)
        return sorted_experiences[:num_replay]
        
    def _consolidate_memory(self) -> Dict[str, float]:
        """
        Consolidate memory by strengthening important connections.
        
        Returns:
            consolidation_results: Results of memory consolidation
        """
        if not self.predictive_core.use_memory or self.predictive_core.memory is None:
            return {'memory_operations': 0}
            
        # Get memory metrics before consolidation
        memory_metrics_before = self.predictive_core.memory.get_memory_metrics()
        
        # Perform memory consolidation operations
        # This is a simplified version - could be more sophisticated
        
        # 1. Strengthen frequently accessed memories
        usage_threshold = 0.1
        memory_matrix = self.predictive_core.memory.memory_matrix
        usage_vector = self.predictive_core.memory.usage_vector
        
        # Boost memories that are frequently used
        high_usage_mask = usage_vector > usage_threshold
        if high_usage_mask.any():
            # Slightly amplify high-usage memories
            memory_matrix[high_usage_mask] *= 1.05
            
        # 2. Decay low-usage memories
        low_usage_mask = usage_vector < 0.01
        if low_usage_mask.any():
            # Decay low-usage memories
            memory_matrix[low_usage_mask] *= 0.95
            
        # 3. Normalize to prevent overflow
        memory_norm = torch.norm(memory_matrix, dim=-1, keepdim=True)
        memory_matrix = memory_matrix / (memory_norm + 1e-8)
        
        # Get metrics after consolidation
        memory_metrics_after = self.predictive_core.memory.get_memory_metrics()
        
        consolidation_score = (
            memory_metrics_after['memory_utilization'] - 
            memory_metrics_before['memory_utilization']
        )
        
        return {
            'memory_operations': 1,
            'consolidation_score': consolidation_score
        }
        
    def _generate_dreams(self) -> Dict[str, float]:
        """
        Generate synthetic experiences through dreaming.
        
        Returns:
            dream_results: Results of dream generation
        """
        # Simplified dream generation - could be more sophisticated
        dream_sequences = 0
        
        # Generate a few synthetic sequences by sampling from memory
        if self.predictive_core.use_memory and self.predictive_core.memory is not None:
            # Sample from memory to create dream sequences
            for _ in range(5):  # Generate 5 dream sequences
                # This is a placeholder - real implementation would be more complex
                dream_sequences += 1
                
        return {
            'dream_sequences': dream_sequences
        }
        
    def wake_up(self) -> Dict[str, float]:
        """
        Exit sleep mode and return to normal operation.
        
        Returns:
            sleep_summary: Summary of sleep cycle results
        """
        if not self.is_sleeping:
            return {}
            
        sleep_duration = time.time() - self.sleep_start_time
        self.sleep_metrics['total_sleep_time'] += sleep_duration
        self.sleep_cycles_completed += 1
        
        # Set predictive core back to eval mode
        self.predictive_core.eval()
        
        self.is_sleeping = False
        
        logger.info(f"Agent waking up after {sleep_duration:.2f}s sleep")
        
        return {
            'sleep_duration': sleep_duration,
            'sleep_cycles_completed': self.sleep_cycles_completed,
            'total_sleep_time': self.sleep_metrics['total_sleep_time']
        }
        
    def add_experience(self, experience: Experience):
        """
        Add experience to replay buffer for future sleep cycles.
        
        Args:
            experience: Experience to add to buffer
        """
        self.replay_buffer.append(experience)
        
        # Also add to high-error buffer if significant learning progress
        if abs(experience.learning_progress) > 0.1:
            self.high_error_buffer.append(experience)
            
    def get_sleep_metrics(self) -> Dict[str, float]:
        """Get sleep system performance metrics."""
        return {
            'is_sleeping': self.is_sleeping,
            'sleep_cycles_completed': self.sleep_cycles_completed,
            'total_sleep_time': self.sleep_metrics['total_sleep_time'],
            'experiences_replayed': self.sleep_metrics['experiences_replayed'],
            'memory_consolidations': self.sleep_metrics['memory_consolidations'],
            'replay_buffer_size': len(self.replay_buffer),
            'high_error_buffer_size': len(self.high_error_buffer)
        }
        
    def reset(self):
        """Reset sleep system for new episode."""
        self.is_sleeping = False
        self.sleep_start_time = 0
        # Keep replay buffers and metrics across episodes