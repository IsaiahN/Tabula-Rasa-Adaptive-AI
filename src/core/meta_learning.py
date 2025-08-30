"""
Meta-Learning System for Adaptive Learning Agent

Implements episodic memory, experience consolidation, and learning insights
extraction to enable the agent to learn from its own learning process.
"""

import torch
import torch.nn as nn
import json
import pickle
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np
from pathlib import Path

from core.data_models import Experience, AgentState

logger = logging.getLogger(__name__)

@dataclass
class LearningInsight:
    """Represents a learned insight that can be applied to future situations."""
    context: str  # Description of the context where this insight applies
    pattern: Dict[str, Any]  # The learned pattern or strategy
    success_rate: float  # How successful this insight has been
    usage_count: int  # How many times this insight has been applied
    confidence: float  # Confidence in this insight (0-1)
    timestamp: int  # When this insight was discovered
    
@dataclass
class EpisodicMemory:
    """Represents a significant learning episode."""
    episode_id: str
    context: str
    initial_state: Dict[str, Any]
    actions_taken: List[torch.Tensor]
    outcomes: List[float]
    learning_progress: List[float]
    final_performance: float
    insights_gained: List[str]
    timestamp: int

class MetaLearningSystem:
    """
    Implements meta-learning capabilities for the adaptive learning agent.
    
    This system allows the agent to:
    1. Save learned insights from experiences
    2. Read and understand previous learning patterns
    3. Apply accumulated knowledge to new situations
    4. Consolidate experiences into generalizable knowledge
    """
    
    def __init__(
        self,
        memory_capacity: int = 1000,
        insight_threshold: float = 0.1,
        consolidation_interval: int = 100,
        save_directory: str = "meta_learning_data"
    ):
        self.memory_capacity = memory_capacity
        self.insight_threshold = insight_threshold
        self.consolidation_interval = consolidation_interval
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(exist_ok=True)
        
        # Memory systems
        self.episodic_memories: deque = deque(maxlen=memory_capacity)
        self.learning_insights: Dict[str, LearningInsight] = {}
        self.experience_buffer: deque = deque(maxlen=10000)
        
        # Consolidation tracking
        self.steps_since_consolidation = 0
        self.total_episodes = 0
        
        # Pattern recognition networks
        self.pattern_encoder = self._create_pattern_encoder()
        self.insight_classifier = self._create_insight_classifier()
        
        # Load existing insights if available
        self._load_persistent_insights()
        
    def _create_pattern_encoder(self) -> nn.Module:
        """Create neural network for encoding experience patterns."""
        return nn.Sequential(
            nn.Linear(128, 64),  # Adjust input size based on state representation
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)  # Compressed pattern representation
        )
    
    def _create_insight_classifier(self) -> nn.Module:
        """Create neural network for classifying and retrieving insights."""
        return nn.Sequential(
            nn.Linear(16, 32),  # Takes encoded pattern
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.learning_insights) + 10)  # Classification over insights
        )
    
    def add_experience(self, experience: Experience, context: str = "general"):
        """Add an experience to the buffer for later consolidation."""
        enhanced_experience = {
            'experience': experience,
            'context': context,
            'timestamp': self.total_episodes
        }
        self.experience_buffer.append(enhanced_experience)
        self.steps_since_consolidation += 1
        
        # Trigger consolidation if needed
        if self.steps_since_consolidation >= self.consolidation_interval:
            self.consolidate_experiences()
    
    def record_episode(
        self,
        episode_id: str,
        context: str,
        initial_state: AgentState,
        actions: List[torch.Tensor],
        outcomes: List[float],
        learning_progress: List[float]
    ):
        """Record a complete learning episode for later analysis."""
        episode_memory = EpisodicMemory(
            episode_id=episode_id,
            context=context,
            initial_state={
                'position': initial_state.position.tolist(),
                'energy': initial_state.energy,
                'timestamp': initial_state.timestamp
            },
            actions_taken=actions,
            outcomes=outcomes,
            learning_progress=learning_progress,
            final_performance=np.mean(outcomes[-10:]) if len(outcomes) >= 10 else np.mean(outcomes),
            insights_gained=[],
            timestamp=self.total_episodes
        )
        
        self.episodic_memories.append(episode_memory)
        self.total_episodes += 1
        
        # Extract insights from this episode
        insights = self._extract_insights_from_episode(episode_memory)
        episode_memory.insights_gained = insights
        
        logger.info(f"Recorded episode {episode_id} with {len(insights)} insights")
    
    def _extract_insights_from_episode(self, episode: EpisodicMemory) -> List[str]:
        """Extract learning insights from a completed episode."""
        insights = []
        
        # Analyze learning progress patterns
        if len(episode.learning_progress) > 10:
            lp_trend = np.polyfit(range(len(episode.learning_progress)), episode.learning_progress, 1)[0]
            
            if lp_trend > self.insight_threshold:
                insight_key = f"{episode.context}_positive_learning"
                if insight_key not in self.learning_insights:
                    self.learning_insights[insight_key] = LearningInsight(
                        context=episode.context,
                        pattern={
                            'type': 'positive_learning_trend',
                            'actions_sequence': [a.tolist() for a in episode.actions_taken[:5]],
                            'initial_conditions': episode.initial_state,
                            'trend_slope': lp_trend
                        },
                        success_rate=1.0,
                        usage_count=1,
                        confidence=min(lp_trend * 5, 1.0),
                        timestamp=episode.timestamp
                    )
                    insights.append(insight_key)
                    logger.info(f"Discovered positive learning pattern in {episode.context}")
                else:
                    # Update existing insight
                    existing = self.learning_insights[insight_key]
                    existing.usage_count += 1
                    existing.success_rate = (existing.success_rate * (existing.usage_count - 1) + 1.0) / existing.usage_count
                    existing.confidence = min(existing.confidence + 0.1, 1.0)
        
        # Analyze action effectiveness patterns
        if len(episode.outcomes) > 5:
            action_outcomes = list(zip(episode.actions_taken, episode.outcomes))
            
            # Find high-performing action sequences
            for i in range(len(action_outcomes) - 2):
                sequence_outcome = np.mean([ao[1] for ao in action_outcomes[i:i+3]])
                if sequence_outcome > np.mean(episode.outcomes) + 0.1:
                    insight_key = f"{episode.context}_effective_sequence_{i}"
                    if insight_key not in self.learning_insights:
                        self.learning_insights[insight_key] = LearningInsight(
                            context=episode.context,
                            pattern={
                                'type': 'effective_action_sequence',
                                'sequence': [ao[0].tolist() for ao in action_outcomes[i:i+3]],
                                'expected_outcome': sequence_outcome
                            },
                            success_rate=1.0,
                            usage_count=1,
                            confidence=min((sequence_outcome - np.mean(episode.outcomes)) * 2, 1.0),
                            timestamp=episode.timestamp
                        )
                        insights.append(insight_key)
        
        return insights
    
    def consolidate_experiences(self):
        """Consolidate recent experiences into generalizable knowledge."""
        logger.info(f"Consolidating {len(self.experience_buffer)} experiences")
        
        # Group experiences by context
        context_groups = {}
        for exp_data in self.experience_buffer:
            context = exp_data['context']
            if context not in context_groups:
                context_groups[context] = []
            context_groups[context].append(exp_data)
        
        # Analyze patterns within each context
        for context, experiences in context_groups.items():
            if len(experiences) < 5:
                continue
                
            # Extract learning progress trends
            lp_values = [exp['experience'].learning_progress for exp in experiences]
            if len(lp_values) > 10:
                trend = np.polyfit(range(len(lp_values)), lp_values, 1)[0]
                
                insight_key = f"{context}_consolidation_trend"
                if abs(trend) > self.insight_threshold:
                    if insight_key not in self.learning_insights:
                        self.learning_insights[insight_key] = LearningInsight(
                            context=context,
                            pattern={
                                'type': 'learning_trend',
                                'trend_slope': trend,
                                'sample_size': len(lp_values),
                                'direction': 'improving' if trend > 0 else 'declining'
                            },
                            success_rate=0.8,
                            usage_count=1,
                            confidence=min(abs(trend) * 3, 1.0),
                            timestamp=self.total_episodes
                        )
                        logger.info(f"Consolidated learning trend insight for {context}: {trend:.3f}")
        
        # Clear buffer and reset counter
        self.experience_buffer.clear()
        self.steps_since_consolidation = 0
        
        # Save insights to persistent storage
        self._save_persistent_insights()
    
    def retrieve_relevant_insights(self, context: str, current_state: AgentState) -> List[LearningInsight]:
        """Retrieve insights relevant to the current context and state."""
        relevant_insights = []
        
        for insight_key, insight in self.learning_insights.items():
            # Check context match
            if context in insight.context or insight.context in context:
                # Check confidence threshold
                if insight.confidence > 0.3:
                    relevant_insights.append(insight)
        
        # Sort by relevance (confidence * success_rate)
        relevant_insights.sort(key=lambda x: x.confidence * x.success_rate, reverse=True)
        
        return relevant_insights[:5]  # Return top 5 most relevant
    
    def apply_insight(self, insight: LearningInsight, current_context: str) -> Optional[torch.Tensor]:
        """Apply a learned insight to generate an action suggestion."""
        if insight.pattern['type'] == 'effective_action_sequence':
            # Return the first action from the effective sequence
            sequence = insight.pattern['sequence']
            if sequence:
                suggested_action = torch.tensor(sequence[0], dtype=torch.float32)
                
                # Update insight usage
                insight.usage_count += 1
                
                logger.info(f"Applied insight {insight.context} in {current_context}")
                return suggested_action
        
        elif insight.pattern['type'] == 'positive_learning_trend':
            # Return actions that led to positive learning
            actions = insight.pattern['actions_sequence']
            if actions:
                suggested_action = torch.tensor(actions[0], dtype=torch.float32)
                insight.usage_count += 1
                return suggested_action
        
        return None
    
    def update_insight_performance(self, insight_key: str, success: bool):
        """Update the performance tracking of an applied insight."""
        if insight_key in self.learning_insights:
            insight = self.learning_insights[insight_key]
            old_rate = insight.success_rate
            old_count = insight.usage_count
            
            new_success_count = old_rate * old_count + (1.0 if success else 0.0)
            insight.success_rate = new_success_count / (old_count + 1)
            
            # Adjust confidence based on performance
            if success:
                insight.confidence = min(insight.confidence + 0.05, 1.0)
            else:
                insight.confidence = max(insight.confidence - 0.1, 0.1)
            
            logger.debug(f"Updated insight {insight_key}: success_rate={insight.success_rate:.3f}, confidence={insight.confidence:.3f}")
    
    def _save_persistent_insights(self):
        """Save insights to persistent storage."""
        insights_file = self.save_directory / "learning_insights.json"
        
        # Convert insights to serializable format
        serializable_insights = {}
        for key, insight in self.learning_insights.items():
            serializable_insights[key] = asdict(insight)
        
        try:
            with open(insights_file, 'w') as f:
                json.dump(serializable_insights, f, indent=2)
            logger.info(f"Saved {len(self.learning_insights)} insights to {insights_file}")
        except Exception as e:
            logger.error(f"Failed to save insights: {e}")
    
    def _load_persistent_insights(self):
        """Load insights from persistent storage."""
        insights_file = self.save_directory / "learning_insights.json"
        
        if insights_file.exists():
            try:
                with open(insights_file, 'r') as f:
                    serializable_insights = json.load(f)
                
                # Convert back to LearningInsight objects
                for key, insight_data in serializable_insights.items():
                    self.learning_insights[key] = LearningInsight(**insight_data)
                
                logger.info(f"Loaded {len(self.learning_insights)} insights from {insights_file}")
            except Exception as e:
                logger.error(f"Failed to load insights: {e}")
    
    def get_meta_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of the meta-learning system's current state."""
        return {
            'total_episodes': self.total_episodes,
            'episodic_memories': len(self.episodic_memories),
            'learning_insights': len(self.learning_insights),
            'experiences_in_buffer': len(self.experience_buffer),
            'insights_by_context': {
                context: len([i for i in self.learning_insights.values() if context in i.context])
                for context in set(i.context for i in self.learning_insights.values())
            },
            'top_insights': [
                {
                    'context': insight.context,
                    'confidence': insight.confidence,
                    'success_rate': insight.success_rate,
                    'usage_count': insight.usage_count
                }
                for insight in sorted(self.learning_insights.values(), 
                                    key=lambda x: x.confidence * x.success_rate, reverse=True)[:5]
            ]
        }
