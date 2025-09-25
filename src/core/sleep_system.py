"""
Sleep and Dream Cycles - Offline processing for memory consolidation.

This module implements sleep cycles that consolidate memories, prune irrelevant
information, and strengthen important patterns without active sensory input.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import logging
from collections import deque
import time

# Fix imports to use relative imports within the package
from .predictive_core import PredictiveCore
from .data_models import Experience, AgentState
from .meta_learning import MetaLearningSystem
from .salience_system import SalienceCalculator, SalienceWeightedReplayBuffer, SalientExperience, SalienceMode, CompressedMemory
from .sleep_breakthrough_detection import create_sleep_breakthrough_system

logger = logging.getLogger(__name__)


class SleepCycle:
    """
    Manages sleep cycles for offline learning and memory consolidation.
    """
    
    def __init__(
        self,
        predictive_core: PredictiveCore,
        meta_learning: Optional[MetaLearningSystem] = None,
        sleep_trigger_energy: float = 40.0,  # Increased from 20.0 to 40.0 for more frequent sleep cycles
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
        
        # Optimizer for offline learning (only if we have a predictive core)
        if self.predictive_core is not None:
            self.sleep_optimizer = optim.Adam(
                self.predictive_core.parameters(),
                lr=learning_rate
            )
        else:
            self.sleep_optimizer = None
        
        # Sleep metrics
        self.sleep_metrics = {
            'total_sleep_time': 0,
            'experiences_replayed': 0,
            'memory_consolidations': 0,
            'performance_improvements': [],
            'high_salience_replays': 0,
            'salience_weighted_consolidations': 0
        }
        
        # Initialize breakthrough detection system
        self.breakthrough_detector, self.breakthrough_processor = create_sleep_breakthrough_system(
            breakthrough_threshold=0.7,
            novelty_threshold=0.6,
            performance_window=50
        )
        
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
        if self.predictive_core is not None:
            self.predictive_core.train()
        
    def _integrate_goal_system_data(self, goal_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Integrate goal system data during sleep for goal-aware consolidation.
        
        Args:
            goal_data: Data from goal invention system
            
        Returns:
            Integration results
        """
        if not goal_data:
            return {'goals_processed': 0}
            
        goals_processed = 0
        goal_insights = []
        
        # Process active goals and their success patterns
        active_goals = goal_data.get('active_goals', [])
        for goal in active_goals:
            goal_success_rate = goal.get('success_rate', 0.0)
            goal_type = goal.get('goal_type', 'unknown')
            
            if goal_success_rate > 0.6:  # Successful goals
                # Extract patterns from successful goal pursuit
                goal_pattern = {
                    'type': 'successful_goal_pattern',
                    'goal_type': goal_type,
                    'success_rate': goal_success_rate,
                    'consolidation_priority': goal_success_rate
                }
                goal_insights.append(goal_pattern)
                goals_processed += 1
        
        # Process emergent goals from high-salience experiences
        emergent_goals = goal_data.get('emergent_goals', [])
        for goal in emergent_goals:
            # These goals emerged from salient experiences - prioritize them
            goal_insights.append({
                'type': 'emergent_goal',
                'salience_origin': True,
                'consolidation_priority': 0.8  # High priority for emergent goals
            })
            goals_processed += 1
        
        logger.info(f"Integrated {goals_processed} goals, extracted {len(goal_insights)} goal insights")
        
        return {
            'goals_processed': goals_processed,
            'goal_insights': goal_insights,
            'goal_consolidation_strength': sum(gi.get('consolidation_priority', 0) for gi in goal_insights)
        }
        
    def execute_sleep_cycle(self, replay_buffer: List[Experience], arc_data: Optional[Dict[str, Any]] = None, goal_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Execute one sleep cycle with enhanced memory consolidation and strategic diversification.
        
        Args:
            replay_buffer: Experiences for replay learning
            arc_data: ARC-3 specific data (action effectiveness, game patterns, etc.)
            goal_data: Goal system data (active goals, emergent goals, etc.)
            
        Returns:
            sleep_results: Results of sleep cycle
        """
        if not self.is_sleeping:
            return {}
            
        sleep_results = {
            'experiences_processed': 0,
            'avg_loss': 0.0,
            'memory_operations': 0,
            'consolidation_score': 0.0,
            'arc_data_integrated': False,
            'goal_data_integrated': False,
            'failed_patterns_identified': 0,
            'successful_patterns_strengthened': 0,
            'diversification_strategies_created': 0
        }
        
        logger.info(" ENHANCED SLEEP CONSOLIDATION - Starting memory consolidation and strategic diversification...")
        
        # Phase 1: Failed Pattern Analysis & Strategic Diversification
        diversification_results = self._analyze_failed_patterns_and_create_diversification_strategies(replay_buffer, arc_data)
        sleep_results.update(diversification_results)
        
        # Phase 2: Memory Decay and Compression (if enabled)
        compression_results = {'decayed': 0, 'compressed': 0, 'merged': 0}
        if hasattr(self, 'salience_calculator'):
            compression_results = self._process_memory_decay_and_compression(
                self.salience_calculator, time.time()
            )
        
        # Phase 3: Priority-Based Experience Replay
        if self.use_salience_weighting:
            replay_results = self._priority_based_experience_replay()
        else:
            replay_results = self._replay_experiences(replay_buffer)
        sleep_results.update(replay_results)
        
        # Phase 4: Object Encoding Enhancement
        encoding_results = self._enhance_object_encodings(replay_buffer)
        sleep_results.update(encoding_results)
        
        # Add compression results to sleep results
        sleep_results['compression_results'] = compression_results
        
        # Phase 5: Enhanced Memory Consolidation with Strategic Pattern Learning
        if self.predictive_core and self.predictive_core.use_memory:
            if self.use_salience_weighting:
                # Try ARC-aware consolidation with strategic learning
                consolidation_results = self._strategic_arc_aware_memory_consolidation(arc_data)
                sleep_results['arc_data_integrated'] = True
                logger.info(" Used strategic ARC-aware memory consolidation during sleep")
            else:
                consolidation_results = self._consolidate_memory_with_meta_learning()
            sleep_results.update(consolidation_results)
        elif not self.predictive_core:
            logger.warning(" Skipping memory consolidation - no predictive core available")
            sleep_results['consolidation_skipped'] = 'no_predictive_core'
        
        # Phase 6: Goal System Integration with Strategic Context
        if goal_data:
            goal_integration_results = self._strategic_goal_system_integration(goal_data, arc_data)
            sleep_results.update(goal_integration_results)
            sleep_results['goal_data_integrated'] = True
            
        # Phase 7: Dream Generation for Strategic Exploration
        dream_results = self._generate_strategic_dreams(arc_data)
        sleep_results.update(dream_results)
        
        # Phase 8: Breakthrough Detection and Processing (NEW)
        breakthrough_results = self._process_breakthroughs_during_sleep(replay_buffer, arc_data)
        sleep_results.update(breakthrough_results)
        
        # Update sleep metrics
        self.sleep_metrics['experiences_replayed'] += sleep_results['experiences_processed']
        self.sleep_metrics['memory_consolidations'] += 1
        
        # Log enhanced sleep cycle completion
        logger.info(f" ENHANCED SLEEP COMPLETE - Failed patterns: {sleep_results['failed_patterns_identified']}, "
                   f"Successful patterns: {sleep_results['successful_patterns_strengthened']}, "
                   f"Diversification strategies: {sleep_results['diversification_strategies_created']}")
        
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
    
    def _arc_aware_memory_consolidation(self) -> Dict[str, float]:
        """
        Consolidate memory using ARC-3 specific patterns and learned effectiveness data.
        
        This enhanced consolidation considers:
        - Action effectiveness patterns
        - Game-specific success strategies  
        - Coordinate intelligence insights
        - Boundary detection patterns
        
        Returns:
            consolidation_results: Results of ARC-aware consolidation
        """
        if not self.predictive_core or not self.predictive_core.use_memory or self.predictive_core.memory is None:
            return {'memory_operations': 0}
            
        # Get memory metrics before consolidation
        memory_metrics_before = self.predictive_core.memory.get_memory_metrics()
        
        consolidation_operations = 0
        arc_specific_consolidations = 0
        
        # Enhanced consolidation using ARC-specific experience data
        memory_matrix = self.predictive_core.memory.memory_matrix
        usage_vector = self.predictive_core.memory.usage_vector
        
        # Process ARC-specific experiences in replay buffer
        for experience in self.salience_replay_buffer.experiences:
            exp_data = experience.experience_data
            salience = experience.salience_value
            
            # Check for ARC-specific data
            if ('arc_action_effectiveness' in exp_data or 
                'arc_game_context' in exp_data or 
                'arc_action_semantics' in exp_data):
                
                arc_specific_consolidations += 1
                
                # Extract ARC-specific insights
                action_effectiveness = exp_data.get('arc_action_effectiveness', {})
                game_context = exp_data.get('arc_game_context', {})
                action_semantics = exp_data.get('arc_action_semantics', {})
                
                # Consolidate based on action effectiveness patterns
                if action_effectiveness:
                    # Boost memories associated with highly effective actions
                    for action_id, effectiveness_data in action_effectiveness.items():
                        success_rate = effectiveness_data.get('success_rate', 0.0)
                        if success_rate > 0.7:  # High success rate
                            # Find memory locations associated with this action pattern
                            high_usage_mask = usage_vector > 0.2
                            if high_usage_mask.any():
                                # Apply effectiveness-based strengthening
                                effectiveness_multiplier = 1.0 + (success_rate - 0.7) * 2.0
                                memory_matrix[high_usage_mask] *= effectiveness_multiplier
                                consolidation_operations += 1
                
                # Consolidate based on game-specific context patterns
                if game_context:
                    game_id = game_context.get('game_id')
                    if game_id:
                        # Context-specific memory strengthening
                        context_locations = usage_vector > 0.15
                        if context_locations.any():
                            context_boost = 1.0 + salience * 0.5
                            memory_matrix[context_locations] *= context_boost
                            consolidation_operations += 1
                
                # Consolidate semantic understanding
                if action_semantics:
                    semantic_confidence = action_semantics.get('confidence', 0.0)
                    if semantic_confidence > 0.5:
                        # Strengthen memories associated with confident semantic understanding
                        semantic_locations = usage_vector > 0.1
                        if semantic_locations.any():
                            semantic_boost = 1.0 + semantic_confidence * 0.3
                            memory_matrix[semantic_locations] *= semantic_boost
                            consolidation_operations += 1
        
        # Normalize to prevent overflow
        if consolidation_operations > 0:
            memory_norm = torch.norm(memory_matrix, dim=-1, keepdim=True)
            memory_matrix = memory_matrix / (memory_norm + 1e-8)
        
        # Get metrics after consolidation
        memory_metrics_after = self.predictive_core.memory.get_memory_metrics()
        
        consolidation_score = (
            memory_metrics_after['memory_utilization'] - 
            memory_metrics_before['memory_utilization']
        )
        
        logger.info(f"ARC-aware consolidation: {consolidation_operations} total operations, "
                   f"{arc_specific_consolidations} ARC-specific consolidations")
        
        return {
            'memory_operations': consolidation_operations,
            'consolidation_score': consolidation_score,
            'arc_specific_consolidations': arc_specific_consolidations,
            'total_arc_experiences_processed': len([exp for exp in self.salience_replay_buffer.experiences 
                                                   if 'arc_action_effectiveness' in exp.experience_data])
        }
        
    def _consolidate_memory_with_meta_learning(self) -> Dict[str, float]:
        """
        Consolidate memory using meta-learning insights for object-aware consolidation.
        
        Returns:
            consolidation_results: Results of memory consolidation
        """
        if not self.predictive_core or not self.predictive_core.use_memory or self.predictive_core.memory is None:
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
        if not self.predictive_core or not self.predictive_core.use_memory or self.predictive_core.memory is None:
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
        if self.predictive_core and self.predictive_core.use_memory and self.predictive_core.memory is not None:
            # Sample from memory to create dream sequences
            for _ in range(5):  # Generate 5 dream sequences
                # This is a placeholder - real implementation would be more complex
                dream_sequences += 1
                
        return {
            'dream_sequences': dream_sequences
        }
        
    def _analyze_failed_patterns_and_create_diversification_strategies(self, replay_buffer: List[Experience], arc_data: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        """
        Analyze failed action patterns and create strategic diversification protocols.
        
        This identifies repetitive failures (like stuck ACTION6 at same coordinate) 
        and creates diversification strategies to escape such patterns.
        
        Args:
            replay_buffer: Recent experiences to analyze
            arc_data: ARC-specific data for pattern analysis
            
        Returns:
            diversification_results: Results of pattern analysis and strategy creation
        """
        failed_patterns_identified = 0
        diversification_strategies_created = 0
        
        if not replay_buffer:
            return {
                'failed_patterns_identified': 0,
                'diversification_strategies_created': 0
            }
        
        # Analyze repetitive failed patterns
        action_coordinate_patterns = {}
        coordinate_failure_counts = {}
        
        for exp in replay_buffer[-200:]:  # Analyze recent 200 experiences
            if hasattr(exp, 'action') and hasattr(exp, 'state'):
                action_id = exp.action
                
                # Extract coordinates if this is ACTION6 (coordinate-based)
                if action_id == 6 and hasattr(exp.state, 'proprioception'):
                    # Assume proprioception contains coordinate info
                    coord_tensor = exp.state.proprioception
                    if len(coord_tensor.shape) >= 1 and coord_tensor.numel() >= 2:
                        x_coord = int(coord_tensor[0].item()) if coord_tensor[0].numel() == 1 else int(coord_tensor[0, 0].item())
                        y_coord = int(coord_tensor[1].item()) if coord_tensor[1].numel() == 1 else int(coord_tensor[0, 1].item())
                        coord_key = (x_coord, y_coord)
                        
                        # Track coordinate usage patterns
                        if coord_key not in action_coordinate_patterns:
                            action_coordinate_patterns[coord_key] = {'attempts': 0, 'successes': 0, 'failures': 0}
                        
                        action_coordinate_patterns[coord_key]['attempts'] += 1
                        
                        # Check if this was a failure (no learning progress or negative reward)
                        if exp.learning_progress <= 0 and exp.reward <= 0:
                            action_coordinate_patterns[coord_key]['failures'] += 1
                            
                            # Track coordinate failure streaks
                            if coord_key not in coordinate_failure_counts:
                                coordinate_failure_counts[coord_key] = 0
                            coordinate_failure_counts[coord_key] += 1
                        else:
                            action_coordinate_patterns[coord_key]['successes'] += 1
                            coordinate_failure_counts[coord_key] = 0  # Reset failure streak
        
        # Identify problematic patterns (stuck coordinates with high failure rates)
        problematic_coordinates = []
        for coord, pattern_data in action_coordinate_patterns.items():
            failure_rate = pattern_data['failures'] / max(pattern_data['attempts'], 1)
            failure_count = coordinate_failure_counts.get(coord, 0)
            
            # Mark as problematic if high failure rate AND recent consecutive failures
            if failure_rate > 0.8 and failure_count > 10:  # >80% failure rate with 10+ consecutive failures
                problematic_coordinates.append({
                    'coordinate': coord,
                    'failure_rate': failure_rate,
                    'consecutive_failures': failure_count,
                    'total_attempts': pattern_data['attempts']
                })
                failed_patterns_identified += 1
        
        # Create diversification strategies for problematic patterns
        diversification_strategies = []
        
        for prob_coord in problematic_coordinates:
            coord = prob_coord['coordinate']
            
            # Strategy 1: Coordinate Exploration Ring - try coordinates around the stuck point
            exploration_ring = []
            x, y = coord
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if dx != 0 or dy != 0:  # Skip the problematic coordinate itself
                        new_coord = (max(0, x + dx), max(0, y + dy))
                        exploration_ring.append(new_coord)
            
            diversification_strategies.append({
                'type': 'coordinate_exploration_ring',
                'problematic_coordinate': coord,
                'alternative_coordinates': exploration_ring[:8],  # Top 8 alternatives
                'priority': prob_coord['consecutive_failures'] / 100.0  # Higher failures = higher priority
            })
            
            # Strategy 2: Systematic Grid Search - if ring fails, try systematic exploration
            diversification_strategies.append({
                'type': 'systematic_grid_search',
                'problematic_coordinate': coord,
                'search_pattern': 'corners_then_center_then_edges',
                'priority': 0.5
            })
            
            # Strategy 3: Visual-Based Alternative Selection - use frame analysis for better coordinates
            diversification_strategies.append({
                'type': 'visual_based_diversification',
                'problematic_coordinate': coord,
                'requires_frame_analysis': True,
                'diversification_threshold': prob_coord['consecutive_failures'],
                'priority': 0.8
            })
            
            diversification_strategies_created += len([s for s in diversification_strategies if s['problematic_coordinate'] == coord])
        
        # Store diversification strategies for use during action selection
        if not hasattr(self, 'diversification_strategies'):
            self.diversification_strategies = {}
        
        for strategy in diversification_strategies:
            prob_coord = strategy['problematic_coordinate']
            if prob_coord not in self.diversification_strategies:
                self.diversification_strategies[prob_coord] = []
            self.diversification_strategies[prob_coord].append(strategy)
        
        logger.info(f" PATTERN ANALYSIS: Identified {failed_patterns_identified} failed patterns, "
                   f"created {diversification_strategies_created} diversification strategies")
        
        return {
            'failed_patterns_identified': failed_patterns_identified,
            'diversification_strategies_created': diversification_strategies_created,
            'problematic_coordinates': problematic_coordinates,
            'diversification_strategies': diversification_strategies
        }
        
    def _priority_based_experience_replay(self) -> Dict[str, float]:
        """
        Enhanced experience replay with priority-based selection focusing on:
        1. High-salience breakthrough experiences (massive replay)
        2. Recent failed patterns (for learning what NOT to do)
        3. Successful strategy patterns (for reinforcement)
        
        Returns:
            replay_results: Results of priority-based replay
        """
        if not self.use_salience_weighting or not hasattr(self, 'salience_replay_buffer'):
            return {'experiences_processed': 0, 'avg_loss': 0.0}
        
        # Priority 1: Get breakthrough experiences (salience > 0.8)
        breakthrough_experiences = self.salience_replay_buffer.sample_by_salience_threshold(0.8, limit=50)
        
        # Priority 2: Get recent failure patterns (salience 0.1-0.4, recent timestamp)
        recent_failures = self.salience_replay_buffer.sample_recent_low_salience(
            min_salience=0.1, max_salience=0.4, limit=30
        )
        
        # Priority 3: Get successful patterns (salience 0.6-0.8)
        successful_patterns = self.salience_replay_buffer.sample_by_salience_threshold(0.6, limit=40)
        
        # Combine with priority weighting
        priority_experiences = []
        
        # Add breakthrough experiences with highest replay weight (3x normal)
        for exp in breakthrough_experiences:
            priority_experiences.append({
                'experience': exp,
                'replay_weight': 3.0,
                'priority_type': 'breakthrough'
            })
        
        # Add failure patterns with moderate replay weight (2x normal) - learn what NOT to do
        for exp in recent_failures:
            priority_experiences.append({
                'experience': exp,
                'replay_weight': 2.0,
                'priority_type': 'failure_pattern'
            })
        
        # Add successful patterns with normal replay weight (1.5x normal)
        for exp in successful_patterns:
            priority_experiences.append({
                'experience': exp,
                'replay_weight': 1.5,
                'priority_type': 'successful_pattern'
            })
        
        if not priority_experiences:
            return {'experiences_processed': 0, 'avg_loss': 0.0}
        
        total_loss = 0.0
        num_batches = 0
        breakthrough_replays = 0
        failure_replays = 0
        success_replays = 0
        
        # Process experiences in priority order
        for priority_exp in priority_experiences[:self.sleep_duration_steps]:
            salient_exp = priority_exp['experience']
            replay_weight = priority_exp['replay_weight']
            priority_type = priority_exp['priority_type']
            
            experience = salient_exp.experience_data['experience']
            
            # Extract states
            if not hasattr(experience, 'state') or not hasattr(experience, 'next_state'):
                continue
                
            state = experience.state
            next_state = experience.next_state
            
            # Create batch (single experience per batch for priority replay)
            batch_visual = state.visual.unsqueeze(0)
            batch_proprio = state.proprioception.unsqueeze(0)
            batch_energy = [state.energy_level]
            
            # Create batched sensory input
            from .data_models import SensoryInput
            batched_input = SensoryInput(
                visual=batch_visual,
                proprioception=batch_proprio,
                energy_level=batch_energy[0],
                timestamp=state.timestamp
            )
            
            # Forward pass
            visual_pred, proprio_pred, energy_pred, _, _ = self.predictive_core(batched_input)
            
            # Compute loss against next state
            target_visual = next_state.visual.unsqueeze(0)
            target_proprio = next_state.proprioception.unsqueeze(0)
            target_energy = torch.tensor([[next_state.energy_level / 100.0]])
            
            # Multi-modal loss with priority weighting
            visual_loss = nn.MSELoss()(visual_pred, target_visual)
            proprio_loss = nn.MSELoss()(proprio_pred, target_proprio)
            energy_loss = nn.MSELoss()(energy_pred, target_energy.to(energy_pred.device))
            
            # Apply priority weighting to loss
            total_loss_batch = replay_weight * (0.5 * visual_loss + 0.3 * proprio_loss + 0.2 * energy_loss)
            
            # Multiple backward passes for high-priority experiences
            replay_iterations = max(1, int(replay_weight))
            for _ in range(replay_iterations):
                self.sleep_optimizer.zero_grad()
                total_loss_batch.backward(retain_graph=True)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.predictive_core.parameters(), 1.0)
                
                self.sleep_optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
            
            # Track priority type counts
            if priority_type == 'breakthrough':
                breakthrough_replays += 1
            elif priority_type == 'failure_pattern':
                failure_replays += 1
            elif priority_type == 'successful_pattern':
                success_replays += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        logger.info(f" PRIORITY REPLAY: {breakthrough_replays} breakthrough, "
                   f"{failure_replays} failure patterns, {success_replays} successful patterns")
        
        return {
            'experiences_processed': len(priority_experiences),
            'avg_loss': avg_loss,
            'breakthrough_replays': breakthrough_replays,
            'failure_pattern_replays': failure_replays,
            'successful_pattern_replays': success_replays
        }
        
    def _strategic_arc_aware_memory_consolidation(self, arc_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Strategic memory consolidation that learns from both successes AND failures in ARC tasks.
        
        This enhanced consolidation:
        1. MASSIVELY strengthens memories of successful strategies
        2. Creates "anti-patterns" from repeated failures to avoid them
        3. Builds strategic memory maps for coordinate selection
        
        Args:
            arc_data: ARC-specific contextual data
            
        Returns:
            consolidation_results: Enhanced consolidation results
        """
        if not self.predictive_core or not self.predictive_core.use_memory or self.predictive_core.memory is None:
            return {'memory_operations': 0}
            
        # Get memory metrics before consolidation
        memory_metrics_before = self.predictive_core.memory.get_memory_metrics()
        
        consolidation_operations = 0
        successful_patterns_strengthened = 0
        failure_patterns_processed = 0
        anti_patterns_created = 0
        
        # Enhanced consolidation using strategic pattern analysis
        memory_matrix = self.predictive_core.memory.memory_matrix
        usage_vector = self.predictive_core.memory.usage_vector
        
        # Create strategic memory maps
        success_memory_map = torch.zeros_like(usage_vector)
        failure_memory_map = torch.zeros_like(usage_vector)
        
        # Process ARC-specific experiences in replay buffer with strategic analysis
        for experience in self.salience_replay_buffer.experiences:
            exp_data = experience.experience_data
            salience = experience.salience_value
            
            # Extract core experience
            core_experience = exp_data.get('experience')
            if not core_experience:
                continue
                
            learning_progress = getattr(core_experience, 'learning_progress', 0.0)
            reward = getattr(core_experience, 'reward', 0.0)
            
            # Classify experience as success or failure
            is_success = learning_progress > 0.1 or reward > 0.5
            is_failure = learning_progress <= 0 and reward <= 0
            
            # Process successful patterns - MASSIVE STRENGTHENING
            if is_success and salience > 0.6:
                successful_patterns_strengthened += 1
                
                # Apply massive strengthening to successful high-salience memories
                success_locations = usage_vector > 0.2
                if success_locations.any():
                    # Strategic success boost: higher salience = more strengthening
                    success_multiplier = 1.0 + salience * 3.0  # Up to 4x strengthening for breakthrough discoveries
                    success_memory_map[success_locations] = torch.max(
                        success_memory_map[success_locations],
                        torch.tensor(success_multiplier)
                    )
                    consolidation_operations += 1
                    
                # Extra strengthening for ARC-specific successful patterns
                if 'arc_action_effectiveness' in exp_data:
                    action_effectiveness = exp_data['arc_action_effectiveness']
                    for action_id, effectiveness_data in action_effectiveness.items():
                        success_rate = effectiveness_data.get('success_rate', 0.0)
                        if success_rate > 0.8:  # Highly successful actions get even MORE strengthening
                            extra_boost_locations = usage_vector > 0.15
                            if extra_boost_locations.any():
                                extra_multiplier = 1.0 + success_rate * 2.0
                                success_memory_map[extra_boost_locations] = torch.max(
                                    success_memory_map[extra_boost_locations],
                                    torch.tensor(extra_multiplier)
                                )
            
            # Process failure patterns - CREATE ANTI-PATTERNS
            elif is_failure and salience > 0.2:  # Even moderate-salience failures are important to learn from
                failure_patterns_processed += 1
                
                # Create anti-patterns: memories that signal "DON'T do this again"
                failure_locations = usage_vector > 0.1
                if failure_locations.any():
                    # Create inhibitory memory patterns for failures
                    failure_memory_map[failure_locations] = torch.max(
                        failure_memory_map[failure_locations],
                        torch.tensor(0.8)  # Strong inhibitory signal
                    )
                    anti_patterns_created += 1
                    
                # Special processing for repetitive coordinate failures (like stuck ACTION6)
                if 'arc_action_semantics' in exp_data:
                    action_semantics = exp_data['arc_action_semantics']
                    if action_semantics.get('action_type') == 'coordinate_placement':
                        # Mark these coordinates as problematic in memory
                        coordinate_locations = usage_vector > 0.05
                        if coordinate_locations.any():
                            failure_memory_map[coordinate_locations] = torch.max(
                                failure_memory_map[coordinate_locations],
                                torch.tensor(0.9)  # Very strong "avoid this" signal
                            )
        
        # Apply strategic memory consolidation
        if consolidation_operations > 0:
            # MASSIVE strengthening of successful patterns
            success_mask = success_memory_map > 1.5
            if success_mask.any():
                memory_matrix[success_mask] *= success_memory_map[success_mask].unsqueeze(-1)
                logger.info(f" MASSIVELY strengthened {success_mask.sum()} successful pattern memories")
            
            # Moderate strengthening of good patterns
            good_mask = (success_memory_map > 1.0) & (success_memory_map <= 1.5)
            if good_mask.any():
                memory_matrix[good_mask] *= success_memory_map[good_mask].unsqueeze(-1)
                logger.info(f" Strengthened {good_mask.sum()} good pattern memories")
        
        # Apply anti-pattern creation (failure inhibition)
        if anti_patterns_created > 0:
            # Create inhibitory patterns for failures
            failure_mask = failure_memory_map > 0.5
            if failure_mask.any():
                # Instead of weakening, create inhibitory signals
                # These will be used during decision-making to avoid repeating failures
                memory_matrix[failure_mask] *= (1.0 - failure_memory_map[failure_mask] * 0.5).unsqueeze(-1)
                logger.info(f" Created {failure_mask.sum()} anti-patterns from failures")
        
        # Normalize to prevent overflow while preserving relative strengths
        memory_norm = torch.norm(memory_matrix, dim=-1, keepdim=True)
        memory_matrix = memory_matrix / (memory_norm + 1e-8)
        
        # Get metrics after consolidation
        memory_metrics_after = self.predictive_core.memory.get_memory_metrics()
        
        consolidation_score = (
            memory_metrics_after['memory_utilization'] - 
            memory_metrics_before['memory_utilization']
        )
        
        logger.info(f" STRATEGIC CONSOLIDATION: {successful_patterns_strengthened} successful patterns strengthened, "
                   f"{anti_patterns_created} anti-patterns created from {failure_patterns_processed} failures")
        
        return {
            'memory_operations': consolidation_operations,
            'consolidation_score': consolidation_score,
            'successful_patterns_strengthened': successful_patterns_strengthened,
            'failure_patterns_processed': failure_patterns_processed,
            'anti_patterns_created': anti_patterns_created,
            'strategic_consolidation_strength': (successful_patterns_strengthened * 2.0 + anti_patterns_created * 1.0)
        }
        
    def _strategic_goal_system_integration(self, goal_data: Dict[str, Any], arc_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced goal system integration with strategic context from memory consolidation.
        
        Args:
            goal_data: Data from goal invention system
            arc_data: ARC-specific contextual data
            
        Returns:
            Integration results with strategic insights
        """
        base_results = self._integrate_goal_system_data(goal_data)
        
        # Add strategic context from failed patterns
        strategic_insights = []
        
        if hasattr(self, 'diversification_strategies'):
            for coord, strategies in self.diversification_strategies.items():
                for strategy in strategies:
                    strategic_insights.append({
                        'type': 'diversification_strategy',
                        'coordinate': coord,
                        'strategy_type': strategy['type'],
                        'priority': strategy['priority'],
                        'integration_strength': strategy['priority'] * 0.5
                    })
        
        base_results.update({
            'strategic_insights_integrated': len(strategic_insights),
            'diversification_goals_created': len([si for si in strategic_insights if si['strategy_type'] == 'coordinate_exploration_ring'])
        })
        
        return base_results
        
    def _generate_strategic_dreams(self, arc_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Generate strategic dream sequences that explore alternative actions for stuck patterns.
        
        Args:
            arc_data: ARC-specific context for dream generation
            
        Returns:
            dream_results: Results of strategic dream generation
        """
        base_results = self._generate_dreams()
        
        strategic_dreams = 0
        diversification_dreams = 0
        
        # Generate dreams for diversification strategies
        if hasattr(self, 'diversification_strategies'):
            for coord, strategies in self.diversification_strategies.items():
                # Dream about alternative coordinate selections
                for strategy in strategies[:3]:  # Top 3 strategies per coordinate
                    if strategy['type'] == 'coordinate_exploration_ring':
                        strategic_dreams += 1
                    elif strategy['type'] == 'visual_based_diversification':
                        diversification_dreams += 1
        
        base_results.update({
            'strategic_dream_sequences': strategic_dreams,
            'diversification_dreams': diversification_dreams,
            'total_strategic_dreams': strategic_dreams + diversification_dreams
        })
        
        return base_results
        
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
        if self.predictive_core is not None:
            self.predictive_core.eval()
        
        self.is_sleeping = False
        
        logger.info(f"Agent waking up after {sleep_duration:.2f}s sleep")
        
        return {
            'sleep_duration': sleep_duration,
            'sleep_cycles_completed': self.sleep_cycles_completed,
            'total_sleep_time': self.sleep_metrics['total_sleep_time']
        }
        
    def add_experience(self, experience: Experience, energy_change: float = 0.0, current_energy: float = 50.0, context: str = "general", arc_data: Optional[Dict[str, Any]] = None):
        """
        Add experience to replay buffer for future sleep cycles with enhanced ARC-3 integration.
        
        Args:
            experience: Experience to add to buffer
            energy_change: Change in energy for salience calculation
            current_energy: Current energy level
            context: Context of the experience
            arc_data: ARC-specific data (action effectiveness, game state, etc.)
        """
        self.replay_buffer.append(experience)
        
        # Also add to high-error buffer if significant learning progress
        if abs(experience.learning_progress) > 0.1:
            self.high_error_buffer.append(experience)
            
        # Add to salience-weighted buffer if enabled
        if self.use_salience_weighting:
            # Enhanced experience data with ARC-3 context
            experience_data = {
                'experience': experience,
                'state': experience.state,
                'next_state': experience.next_state,
                'action': experience.action,
                'reward': experience.reward
            }
            
            # Add ARC-3 specific data if available
            if arc_data:
                experience_data.update({
                    'arc_action_effectiveness': arc_data.get('action_effectiveness', {}),
                    'arc_game_context': arc_data.get('game_context', {}),
                    'arc_action_semantics': arc_data.get('action_semantics', {}),
                    'arc_boundary_data': arc_data.get('boundary_data', {}),
                    'arc_coordinate_intelligence': arc_data.get('coordinate_intelligence', {})
                })
                
            salient_experience = self.salience_calculator.create_salient_experience(
                experience_data=experience_data,
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
    
    def _process_breakthroughs_during_sleep(self, replay_buffer: List[Experience], 
                                          arc_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process breakthrough detection during sleep cycles."""
        try:
            # Get recent experiences for breakthrough analysis
            recent_experiences = replay_buffer[-100:] if len(replay_buffer) > 100 else replay_buffer
            
            # Detect breakthroughs
            breakthrough_events = self.breakthrough_detector.detect_breakthroughs(recent_experiences)
            
            # Process breakthroughs
            breakthrough_insights = []
            breakthrough_consolidation_benefit = 0.0
            
            for breakthrough in breakthrough_events:
                # Process the breakthrough
                processed_breakthrough = self.breakthrough_processor.process_breakthrough(breakthrough)
                
                if processed_breakthrough:
                    breakthrough_insights.append(processed_breakthrough)
                    
                    # Calculate consolidation benefit from breakthrough
                    breakthrough_consolidation_benefit += processed_breakthrough.get('consolidation_benefit', 0.0)
                    
                    # Log breakthrough for system learning
                    logger.info(f"Breakthrough detected during sleep: {processed_breakthrough.get('type', 'unknown')}")
            
            # Calculate breakthrough metrics
            breakthrough_quality = len(breakthrough_insights) / max(len(breakthrough_events), 1)
            breakthrough_consolidation_benefit = min(breakthrough_consolidation_benefit, 1.0)
            
            # Update sleep metrics
            self.sleep_metrics['breakthroughs_detected'] += len(breakthrough_events)
            self.sleep_metrics['breakthroughs_processed'] += len(breakthrough_insights)
            
            return {
                'breakthroughs_detected': len(breakthrough_events),
                'breakthroughs_processed': len(breakthrough_insights),
                'breakthrough_quality': breakthrough_quality,
                'breakthrough_insights': breakthrough_insights,
                'breakthrough_consolidation_benefit': breakthrough_consolidation_benefit
            }
            
        except Exception as e:
            logger.error(f"Breakthrough processing failed: {e}")
            return {
                'breakthroughs_detected': 0,
                'breakthroughs_processed': 0,
                'breakthrough_quality': 0.0,
                'breakthrough_insights': [],
                'breakthrough_consolidation_benefit': 0.0
            }