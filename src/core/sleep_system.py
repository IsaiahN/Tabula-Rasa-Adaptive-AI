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

from .data_models import Experience, AgentState
from .predictive_core import PredictiveCore
from .meta_learning import MetaLearningSystem
from .salience_system import SalienceCalculator, SalienceWeightedReplayBuffer, SalientExperience, SalienceMode, CompressedMemory

logger = logging.getLogger(__name__)


class SleepCycle:
    """
    Manages sleep cycles for offline learning and memory consolidation.
    """
    
    def __init__(
        self,
        predictive_core: PredictiveCore,
        meta_learning: Optional[MetaLearningSystem] = None,
        sleep_trigger_energy: float = 20.0,
        sleep_trigger_boredom_steps: int = 1000,
        sleep_trigger_memory_pressure: float = 0.9,
        sleep_duration_steps: int = 100,
        replay_batch_size: int = 32,
        learning_rate: float = 0.001,
        object_encoding_threshold: float = 0.05,
        use_salience_weighting: bool = True
    ):
        self.predictive_core = predictive_core
        self.meta_learning = meta_learning
        self.sleep_trigger_energy = sleep_trigger_energy
        self.sleep_trigger_boredom_steps = sleep_trigger_boredom_steps
        self.sleep_trigger_memory_pressure = sleep_trigger_memory_pressure
        self.sleep_duration_steps = sleep_duration_steps
        self.replay_batch_size = replay_batch_size
        self.object_encoding_threshold = object_encoding_threshold
        self.use_salience_weighting = use_salience_weighting
        
        # Sleep state tracking
        self.is_sleeping = False
        self.sleep_start_time = 0
        self.sleep_cycles_completed = 0
        
        # Salience-based experience replay system
        if self.use_salience_weighting:
            self.salience_calculator = SalienceCalculator()
            self.salience_replay_buffer = SalienceWeightedReplayBuffer(capacity=10000)
        
        # Traditional experience replay buffer for sleep learning
        # Object encoding tracking
        self.object_encodings = {}  # Track learned object representations
        self.encoding_improvements = deque(maxlen=1000)  # Track encoding quality over time
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
            'performance_improvements': [],
            'high_salience_replays': 0,
            'salience_weighted_consolidations': 0
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
        
        # Phase 1: Memory Decay and Compression (if enabled)
        compression_results = {'decayed': 0, 'compressed': 0, 'merged': 0}
        if hasattr(self, 'salience_calculator'):
            compression_results = self._process_memory_decay_and_compression(
                self.salience_calculator, time.time()
            )
        
        # Phase 2: Salience-Weighted Experience Replay
        if self.use_salience_weighting:
            replay_results = self._salience_weighted_replay()
        else:
            replay_results = self._replay_experiences(replay_buffer)
        sleep_results.update(replay_results)
        
        # Phase 3: Object Encoding Enhancement
        encoding_results = self._enhance_object_encodings(replay_buffer)
        sleep_results.update(encoding_results)
        
        # Add compression results to sleep results
        sleep_results['compression_results'] = compression_results
        
        # Phase 3: Salience-Based Memory Consolidation
        if self.predictive_core.use_memory:
            if self.use_salience_weighting:
                consolidation_results = self._salience_based_memory_consolidation()
            else:
                consolidation_results = self._consolidate_memory_with_meta_learning()
            sleep_results.update(consolidation_results)
            
        # Phase 4: Dream Generation (optional)
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
        
    def _salience_weighted_replay(self) -> Dict[str, float]:
        """
        Perform salience-weighted experience replay during sleep.
        
        The agent dreams most vividly about its most important discoveries
        and life-threatening mistakes, based on salience values.
        
        Returns:
            replay_results: Results of salience-weighted replay
        """
        if not self.use_salience_weighting or not hasattr(self, 'salience_replay_buffer'):
            return {'experiences_processed': 0, 'avg_loss': 0.0}
        
        # Sample high-salience experiences for replay
        high_salience_experiences = self.salience_replay_buffer.sample(self.sleep_duration_steps)
        
        if not high_salience_experiences:
            return {'experiences_processed': 0, 'avg_loss': 0.0}
        
        total_loss = 0.0
        num_batches = 0
        high_salience_count = 0
        
        # Process experiences in batches, prioritizing by salience
        for i in range(0, len(high_salience_experiences), self.replay_batch_size):
            batch_salient = high_salience_experiences[i:i + self.replay_batch_size]
            batch_experiences = [se.experience_data['experience'] for se in batch_salient]
            batch_saliences = [se.salience_value for se in batch_salient]
            
            # Convert experiences to tensors
            states = [exp.state for exp in batch_experiences]
            next_states = [exp.next_state for exp in batch_experiences]
            
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
            
            # Multi-modal loss with salience weighting
            visual_loss = nn.MSELoss(reduction='none')(visual_pred, target_visual)
            proprio_loss = nn.MSELoss(reduction='none')(proprio_pred, target_proprio)
            energy_loss = nn.MSELoss(reduction='none')(energy_pred, target_energy.to(energy_pred.device))
            
            # Weight losses by salience values
            salience_weights = torch.tensor(batch_saliences).to(visual_loss.device)
            
            # Apply salience weighting to each loss component
            weighted_visual_loss = (visual_loss.mean(dim=[1,2,3]) * salience_weights).mean()
            weighted_proprio_loss = (proprio_loss.mean(dim=1) * salience_weights).mean()
            weighted_energy_loss = (energy_loss.squeeze() * salience_weights).mean()
            
            total_loss_batch = 0.5 * weighted_visual_loss + 0.3 * weighted_proprio_loss + 0.2 * weighted_energy_loss
            
            # Backward pass with salience-weighted gradients
            self.sleep_optimizer.zero_grad()
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.predictive_core.parameters(), 1.0)
            
            self.sleep_optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
            
            # Count high-salience experiences (>0.7)
            high_salience_count += sum(1 for s in batch_saliences if s > 0.7)
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Update metrics
        self.sleep_metrics['high_salience_replays'] += high_salience_count
        
        logger.info(f"Salience-weighted replay: {len(high_salience_experiences)} experiences, "
                   f"{high_salience_count} high-salience, avg_loss={avg_loss:.4f}")
        
        return {
            'experiences_processed': len(high_salience_experiences),
            'avg_loss': avg_loss,
            'high_salience_count': high_salience_count
        }
    
    def _process_memory_decay_and_compression(self, salience_calculator: SalienceCalculator, current_time: float) -> Dict[str, int]:
        """
        Process memory decay and compression during sleep cycle.
        
        Args:
            salience_calculator: The salience calculator with decay/compression capabilities
            current_time: Current timestamp for decay calculations
            
        Returns:
            Dictionary with compression statistics
        """
        if not hasattr(self, 'salience_replay_buffer') or salience_calculator.mode == SalienceMode.LOSSLESS:
            return {'decayed': 0, 'compressed': 0, 'merged': 0}
        
        # Get all experiences from replay buffer
        all_experiences = list(self.salience_replay_buffer.experiences)
        
        if not all_experiences:
            return {'decayed': 0, 'compressed': 0, 'merged': 0}
        
        # Apply salience decay
        decayed_experiences = salience_calculator.apply_salience_decay(all_experiences, current_time)
        
        # Compress low-salience memories
        remaining_experiences, compressed_memories = salience_calculator.compress_low_salience_memories(
            decayed_experiences, current_time
        )
        
        # Update replay buffer with remaining experiences
        self.salience_replay_buffer.experiences.clear()
        self.salience_replay_buffer.priorities.clear()
        
        for exp in remaining_experiences:
            self.salience_replay_buffer.add(exp)
        
        # Store compressed memories in salience calculator
        salience_calculator.compressed_memories.extend(compressed_memories)
        
        logger.info(f"Memory processing: {len(decayed_experiences)} decayed, "
                   f"{len(compressed_memories)} compressed, "
                   f"{len(remaining_experiences)} remaining")
        
        return {
            'decayed': len(decayed_experiences),
            'compressed': len(compressed_memories),
            'merged': sum(cm.merged_count for cm in compressed_memories)
        }
        
    def _enhance_object_encodings(self, replay_buffer: List[Experience]) -> Dict[str, float]:
        """
        Enhance object encodings during sleep by analyzing visual patterns.
        
        Args:
            replay_buffer: Experiences containing visual data
            
        Returns:
            encoding_results: Results of object encoding enhancement
        """
        if not replay_buffer:
            return {'objects_encoded': 0, 'encoding_improvement': 0.0}
            
        # Extract visual features from experiences
        visual_features = []
        for exp in replay_buffer:
            if exp.state and exp.state.visual is not None:
                # Extract key visual features (simplified)
                visual_tensor = exp.state.visual
                # Compute feature statistics
                mean_intensity = torch.mean(visual_tensor).item()
                std_intensity = torch.std(visual_tensor).item()
                edge_density = torch.mean(torch.abs(visual_tensor[..., 1:, :] - visual_tensor[..., :-1, :])).item()
                
                feature_vector = np.array([mean_intensity, std_intensity, edge_density])
                visual_features.append(feature_vector)
        
        if not visual_features:
            return {'objects_encoded': 0, 'encoding_improvement': 0.0}
            
        # Simple clustering without sklearn - use basic distance-based grouping
        if len(visual_features) >= 3:
            visual_array = np.array(visual_features)
            
            # Simple clustering: group features by similarity threshold
            clusters = []
            cluster_centers = []
            similarity_threshold = 0.5
            
            for feature in visual_array:
                assigned = False
                for i, center in enumerate(cluster_centers):
                    # Calculate Euclidean distance
                    distance = np.linalg.norm(feature - center)
                    if distance < similarity_threshold:
                        # Add to existing cluster
                        clusters[i].append(feature)
                        # Update cluster center (moving average)
                        cluster_centers[i] = (cluster_centers[i] * (len(clusters[i]) - 1) + feature) / len(clusters[i])
                        assigned = True
                        break
                
                if not assigned:
                    # Create new cluster
                    clusters.append([feature])
                    cluster_centers.append(feature.copy())
            
            # Update object encodings
            objects_encoded = 0
            for i, center in enumerate(cluster_centers):
                object_id = f"object_{i}"
                if object_id not in self.object_encodings:
                    self.object_encodings[object_id] = {
                        'features': center,
                        'confidence': 1.0,
                        'last_seen': time.time()
                    }
                    objects_encoded += 1
                else:
                    # Update existing encoding with exponential moving average
                    alpha = 0.1
                    self.object_encodings[object_id]['features'] = (
                        alpha * center + (1 - alpha) * self.object_encodings[object_id]['features']
                    )
                    self.object_encodings[object_id]['confidence'] = min(
                        self.object_encodings[object_id]['confidence'] + 0.1, 1.0
                    )
                    self.object_encodings[object_id]['last_seen'] = time.time()
            
            # Calculate encoding improvement
            current_quality = len(self.object_encodings) * np.mean([obj['confidence'] for obj in self.object_encodings.values()])
            self.encoding_improvements.append(current_quality)
            
            improvement = 0.0
            if len(self.encoding_improvements) > 1:
                improvement = self.encoding_improvements[-1] - self.encoding_improvements[-2]
            
            logger.info(f"Enhanced {objects_encoded} object encodings during sleep")
            
            return {
                'objects_encoded': objects_encoded,
                'encoding_improvement': improvement,
                'total_objects': len(self.object_encodings)
            }
        
        return {'objects_encoded': 0, 'encoding_improvement': 0.0}
    
    def _consolidate_memory_with_meta_learning(self) -> Dict[str, float]:
        """
        Consolidate memory using meta-learning insights for object-aware consolidation.
        
        Returns:
            consolidation_results: Results of memory consolidation
        """
        if not self.predictive_core.use_memory or self.predictive_core.memory is None:
            return {'memory_operations': 0}
            
        # Get memory metrics before consolidation
        memory_metrics_before = self.predictive_core.memory.get_memory_metrics()
        
        # Enhanced memory consolidation with meta-learning insights
        consolidation_operations = 0
        
        # 1. Use meta-learning insights to prioritize important memories
        if self.meta_learning:
            # Get relevant insights for memory consolidation
            relevant_insights = self.meta_learning.retrieve_relevant_insights(
                "sleep_consolidation", 
                None  # No current state during sleep
            )
            
            # Apply insights to memory consolidation strategy
            for insight in relevant_insights:
                if 'memory_priority' in insight.pattern:
                    # Adjust consolidation based on learned patterns
                    consolidation_operations += 1
        
        # 2. Object-aware memory strengthening
        usage_threshold = 0.1
        memory_matrix = self.predictive_core.memory.memory_matrix
        usage_vector = self.predictive_core.memory.usage_vector
        
        # Boost memories associated with well-encoded objects
        object_boost_factor = 1.0 + len(self.object_encodings) * 0.01
        high_usage_mask = usage_vector > usage_threshold
        if high_usage_mask.any():
            memory_matrix[high_usage_mask] *= object_boost_factor
            
        # 3. Decay low-usage memories more aggressively if we have good object encodings
        decay_factor = 0.95 - len(self.object_encodings) * 0.005
        low_usage_mask = usage_vector < 0.01
        if low_usage_mask.any():
            memory_matrix[low_usage_mask] *= max(decay_factor, 0.85)
            
        # 4. Normalize to prevent overflow
        memory_norm = torch.norm(memory_matrix, dim=-1, keepdim=True)
        memory_matrix = memory_matrix / (memory_norm + 1e-8)
        
        # Get metrics after consolidation
        memory_metrics_after = self.predictive_core.memory.get_memory_metrics()
        
        consolidation_score = (
            memory_metrics_after['memory_utilization'] - 
            memory_metrics_before['memory_utilization']
        )
        
        return {
            'memory_operations': 1 + consolidation_operations,
            'consolidation_score': consolidation_score
        }
        
    def _salience_based_memory_consolidation(self) -> Dict[str, float]:
        """
        Consolidate memory using salience values to determine consolidation strength.
        
        High-salience memories get their neural pathways massively strengthened.
        Low-salience memories are allowed to decay rapidly.
        
        Returns:
            consolidation_results: Results of salience-based consolidation
        """
        if not self.predictive_core.use_memory or self.predictive_core.memory is None:
            return {'memory_operations': 0}
            
        # Get memory metrics before consolidation
        memory_metrics_before = self.predictive_core.memory.get_memory_metrics()
        
        # Get high-salience experiences for memory consolidation
        high_salience_experiences = self.salience_calculator.get_high_salience_experiences(
            threshold=0.5, limit=100
        )
        
        if not high_salience_experiences:
            return {'memory_operations': 0}
        
        consolidation_operations = 0
        total_consolidation_strength = 0.0
        
        # Memory consolidation based on salience
        memory_matrix = self.predictive_core.memory.memory_matrix
        usage_vector = self.predictive_core.memory.usage_vector
        
        # Create salience-based consolidation map
        salience_map = torch.zeros_like(usage_vector)
        
        # Map high-salience experiences to memory locations
        for exp_data in high_salience_experiences:
            salience = exp_data['salience']
            consolidation_strength = self.salience_calculator.get_consolidation_strength(salience)
            
            # Find memory locations with high usage (active memories)
            active_locations = usage_vector > 0.1
            if active_locations.any():
                # Apply consolidation strength to active memory locations
                salience_map[active_locations] = torch.max(
                    salience_map[active_locations],
                    torch.tensor(consolidation_strength)
                )
                consolidation_operations += 1
                total_consolidation_strength += consolidation_strength
        
        # Apply salience-based consolidation
        if consolidation_operations > 0:
            # Strengthen high-salience memories
            high_salience_mask = salience_map > 1.5
            if high_salience_mask.any():
                # Massive strengthening for breakthrough experiences
                memory_matrix[high_salience_mask] *= salience_map[high_salience_mask].unsqueeze(-1)
                logger.info(f"Massively strengthened {high_salience_mask.sum()} high-salience memories")
            
            # Moderate strengthening for important memories
            medium_salience_mask = (salience_map > 1.0) & (salience_map <= 1.5)
            if medium_salience_mask.any():
                memory_matrix[medium_salience_mask] *= salience_map[medium_salience_mask].unsqueeze(-1)
                logger.info(f"Strengthened {medium_salience_mask.sum()} medium-salience memories")
            
            # Allow low-salience memories to decay
            low_salience_mask = (usage_vector > 0) & (salience_map < 0.5)
            if low_salience_mask.any():
                decay_factor = 0.1  # Rapid decay for unimportant memories
                memory_matrix[low_salience_mask] *= decay_factor
                logger.info(f"Allowed {low_salience_mask.sum()} low-salience memories to decay")
        
        # Normalize to prevent overflow
        memory_norm = torch.norm(memory_matrix, dim=-1, keepdim=True)
        memory_matrix = memory_matrix / (memory_norm + 1e-8)
        
        # Update memory usage based on consolidation
        # High-salience locations become more "used"
        usage_vector = torch.clamp(usage_vector + salience_map * 0.1, 0.0, 1.0)
        
        # Get metrics after consolidation
        memory_metrics_after = self.predictive_core.memory.get_memory_metrics()
        
        consolidation_score = (
            memory_metrics_after['memory_utilization'] - 
            memory_metrics_before['memory_utilization']
        )
        
        # Update metrics
        self.sleep_metrics['salience_weighted_consolidations'] += consolidation_operations
        
        avg_consolidation_strength = total_consolidation_strength / max(consolidation_operations, 1)
        
        logger.info(f"Salience-based consolidation: {consolidation_operations} operations, "
                   f"avg_strength={avg_consolidation_strength:.2f}")
        
        return {
            'memory_operations': consolidation_operations,
            'consolidation_score': consolidation_score,
            'avg_consolidation_strength': avg_consolidation_strength,
            'high_salience_consolidations': high_salience_mask.sum().item() if 'high_salience_mask' in locals() else 0
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
        
    def add_experience(self, experience: Experience, energy_change: float = 0.0, current_energy: float = 50.0, context: str = "general"):
        """
        Add experience to replay buffer for future sleep cycles.
        
        Args:
            experience: Experience to add to buffer
            energy_change: Change in energy for salience calculation
            current_energy: Current energy level
            context: Context of the experience
        """
        self.replay_buffer.append(experience)
        
        # Also add to high-error buffer if significant learning progress
        if abs(experience.learning_progress) > 0.1:
            self.high_error_buffer.append(experience)
            
        # Add to salience-weighted buffer if enabled
        if self.use_salience_weighting:
            salient_experience = self.salience_calculator.create_salient_experience(
                experience_data={
                    'experience': experience,
                    'state': experience.state,
                    'next_state': experience.next_state,
                    'action': experience.action,
                    'reward': experience.reward
                },
                learning_progress=experience.learning_progress,
                energy_change=energy_change,
                current_energy=current_energy,
                context=context
            )
            self.salience_replay_buffer.add(salient_experience)
            
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
        
    def get_object_encodings(self) -> Dict[str, Dict]:
        """Get current object encodings learned during sleep."""
        return self.object_encodings.copy()
        
    def get_encoding_quality(self) -> float:
        """Get current encoding quality score."""
        if not self.object_encodings:
            return 0.0
        return len(self.object_encodings) * np.mean([obj['confidence'] for obj in self.object_encodings.values()])