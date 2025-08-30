"""
Salience System - Experience importance derived from agent's innate drives.

This system calculates the "importance" or "salience" of every experience based on
the agent's own Learning Progress and Energy/Survival drives, creating a perfect
feedback loop where the agent's intrinsic motivations determine memory priority.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class SalienceMetrics:
    """Metrics for tracking salience calculation quality."""
    average_salience: float
    max_salience: float
    salience_distribution: Dict[str, float]  # Low, medium, high percentages
    lp_contribution: float
    energy_contribution: float
    
@dataclass
class SalientExperience:
    """An experience with its calculated salience value."""
    experience_data: Dict
    salience_value: float
    lp_component: float
    energy_component: float
    context: str
    timestamp: int

class SalienceCalculator:
    """
    Calculates experience salience based on Learning Progress and Energy drives.
    
    The core insight: The agent's own drives determine what's important.
    - High LP spike = breakthrough understanding = high salience
    - Major energy change = survival-critical event = high salience
    """
    
    def __init__(
        self,
        lp_weight: float = 0.6,
        energy_weight: float = 0.4,
        lp_spike_threshold: float = 0.1,
        energy_change_threshold: float = 5.0,
        salience_history_size: int = 1000,
        normalization_window: int = 100
    ):
        self.lp_weight = lp_weight
        self.energy_weight = energy_weight
        self.lp_spike_threshold = lp_spike_threshold
        self.energy_change_threshold = energy_change_threshold
        self.salience_history_size = salience_history_size
        self.normalization_window = normalization_window
        
        # Salience tracking
        self.salience_history = deque(maxlen=salience_history_size)
        self.lp_history = deque(maxlen=normalization_window)
        self.energy_history = deque(maxlen=normalization_window)
        
        # Running statistics for normalization
        self.lp_mean = 0.0
        self.lp_std = 1.0
        self.energy_mean = 50.0
        self.energy_std = 20.0
        
        # Salience distribution tracking
        self.high_salience_count = 0
        self.medium_salience_count = 0
        self.low_salience_count = 0
        
    def calculate_salience(
        self,
        learning_progress: float,
        energy_change: float,
        current_energy: float,
        context: str = "general",
        additional_factors: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate salience value for an experience.
        
        Args:
            learning_progress: Current LP signal from Learning Progress Drive
            energy_change: Change in energy from previous step
            current_energy: Current energy level
            context: Context of the experience
            additional_factors: Optional additional salience factors
            
        Returns:
            Salience value (0-1 range, higher = more important)
        """
        # Update running statistics
        self._update_statistics(learning_progress, energy_change)
        
        # Calculate Learning Progress component
        lp_component = self._calculate_lp_salience(learning_progress)
        
        # Calculate Energy/Survival component
        energy_component = self._calculate_energy_salience(energy_change, current_energy)
        
        # Combine components
        base_salience = (
            self.lp_weight * lp_component + 
            self.energy_weight * energy_component
        )
        
        # Apply additional factors if provided
        if additional_factors:
            for factor_name, factor_value in additional_factors.items():
                if factor_name == "novelty_bonus":
                    base_salience += 0.1 * factor_value
                elif factor_name == "goal_relevance":
                    base_salience *= (1.0 + 0.2 * factor_value)
        
        # Normalize to 0-1 range
        final_salience = torch.clamp(torch.tensor(base_salience), 0.0, 1.0).item()
        
        # Track salience distribution
        self._track_salience_distribution(final_salience)
        
        # Store in history
        self.salience_history.append({
            'salience': final_salience,
            'lp_component': lp_component,
            'energy_component': energy_component,
            'context': context,
            'timestamp': len(self.salience_history)
        })
        
        return final_salience
    
    def _calculate_lp_salience(self, learning_progress: float) -> float:
        """
        Calculate salience component from Learning Progress.
        
        Key insight: A massive LP spike indicates a breakthrough in understanding.
        This is BY DEFINITION the most important type of experience.
        """
        # Normalize LP relative to recent history
        if len(self.lp_history) > 10:
            normalized_lp = (learning_progress - self.lp_mean) / max(self.lp_std, 1e-6)
        else:
            normalized_lp = learning_progress
        
        # Detect spikes (both positive and negative are important)
        lp_magnitude = abs(normalized_lp)
        
        # Exponential scaling for spikes
        if lp_magnitude > self.lp_spike_threshold:
            # Major breakthrough or confusion - highly salient
            spike_multiplier = min(lp_magnitude / self.lp_spike_threshold, 5.0)
            lp_salience = 0.2 + 0.6 * (spike_multiplier - 1.0) / 4.0
        else:
            # Normal learning - moderate salience
            lp_salience = 0.1 + 0.1 * (lp_magnitude / self.lp_spike_threshold)
        
        # Positive LP gets slight bonus (learning is good)
        if normalized_lp > 0:
            lp_salience *= 1.1
        
        return min(lp_salience, 1.0)
    
    def _calculate_energy_salience(self, energy_change: float, current_energy: float) -> float:
        """
        Calculate salience component from Energy/Survival drive.
        
        Key insight: Significant energy changes indicate survival-critical events.
        Finding food, avoiding death, successful tool use - all highly salient.
        """
        # Energy change magnitude (both gain and loss are important)
        change_magnitude = abs(energy_change)
        
        # Normalize energy change
        if len(self.energy_history) > 10:
            normalized_change = change_magnitude / max(self.energy_std, 1e-6)
        else:
            normalized_change = change_magnitude / self.energy_change_threshold
        
        # Base salience from energy change magnitude
        if change_magnitude > self.energy_change_threshold:
            # Major energy event - highly salient
            change_multiplier = min(normalized_change, 5.0)
            energy_salience = 0.3 + 0.5 * (change_multiplier - 1.0) / 4.0
        else:
            # Normal energy change - low to moderate salience
            energy_salience = 0.05 + 0.25 * normalized_change
        
        # Critical energy levels increase salience
        energy_ratio = current_energy / 100.0  # Assuming max energy is 100
        if energy_ratio < 0.2:  # Very low energy
            energy_salience *= 1.5  # Near-death experiences are highly salient
        elif energy_ratio > 0.9:  # Very high energy
            energy_salience *= 1.2  # Finding abundant resources is important
        
        # Positive energy changes get bonus (survival success)
        if energy_change > 0:
            energy_salience *= 1.1
        
        return min(energy_salience, 1.0)
    
    def _update_statistics(self, learning_progress: float, energy_change: float):
        """Update running statistics for normalization."""
        self.lp_history.append(learning_progress)
        self.energy_history.append(abs(energy_change))
        
        if len(self.lp_history) >= 10:
            self.lp_mean = np.mean(list(self.lp_history))
            self.lp_std = max(np.std(list(self.lp_history)), 1e-6)
        
        if len(self.energy_history) >= 10:
            self.energy_mean = np.mean(list(self.energy_history))
            self.energy_std = max(np.std(list(self.energy_history)), 1e-6)
    
    def _track_salience_distribution(self, salience: float):
        """Track distribution of salience values for monitoring."""
        if salience > 0.7:
            self.high_salience_count += 1
        elif salience > 0.3:
            self.medium_salience_count += 1
        else:
            self.low_salience_count += 1
    
    def get_salience_metrics(self) -> SalienceMetrics:
        """Get current salience system metrics."""
        if not self.salience_history:
            return SalienceMetrics(
                average_salience=0.0,
                max_salience=0.0,
                salience_distribution={'low': 0.0, 'medium': 0.0, 'high': 0.0},
                lp_contribution=0.0,
                energy_contribution=0.0
            )
        
        recent_saliences = [s['salience'] for s in list(self.salience_history)[-100:]]
        recent_lp_components = [s['lp_component'] for s in list(self.salience_history)[-100:]]
        recent_energy_components = [s['energy_component'] for s in list(self.salience_history)[-100:]]
        
        total_count = self.high_salience_count + self.medium_salience_count + self.low_salience_count
        if total_count == 0:
            distribution = {'low': 0.0, 'medium': 0.0, 'high': 0.0}
        else:
            distribution = {
                'low': self.low_salience_count / total_count,
                'medium': self.medium_salience_count / total_count,
                'high': self.high_salience_count / total_count
            }
        
        return SalienceMetrics(
            average_salience=np.mean(recent_saliences),
            max_salience=max(recent_saliences),
            salience_distribution=distribution,
            lp_contribution=np.mean(recent_lp_components),
            energy_contribution=np.mean(recent_energy_components)
        )
    
    def get_high_salience_experiences(self, threshold: float = 0.7, limit: int = 50) -> List[Dict]:
        """
        Get the most salient experiences for prioritized replay.
        
        This is used by the sleep system for experience replay - the agent
        literally dreams most vividly about its most important discoveries.
        """
        high_salience = [
            exp for exp in self.salience_history 
            if exp['salience'] >= threshold
        ]
        
        # Sort by salience value (highest first)
        high_salience.sort(key=lambda x: x['salience'], reverse=True)
        
        return high_salience[:limit]
    
    def create_salient_experience(
        self,
        experience_data: Dict,
        learning_progress: float,
        energy_change: float,
        current_energy: float,
        context: str = "general"
    ) -> SalientExperience:
        """Create a SalientExperience with calculated salience."""
        salience = self.calculate_salience(
            learning_progress, energy_change, current_energy, context
        )
        
        # Get component values for analysis
        lp_component = self._calculate_lp_salience(learning_progress)
        energy_component = self._calculate_energy_salience(energy_change, current_energy)
        
        return SalientExperience(
            experience_data=experience_data,
            salience_value=salience,
            lp_component=lp_component,
            energy_component=energy_component,
            context=context,
            timestamp=len(self.salience_history)
        )
    
    def should_consolidate_strongly(self, salience: float) -> bool:
        """
        Determine if a memory should be consolidated with high strength.
        
        High-salience memories get their neural pathways massively strengthened
        during sleep. Low-salience memories are allowed to decay rapidly.
        """
        return salience > 0.6
    
    def get_consolidation_strength(self, salience: float) -> float:
        """
        Get memory consolidation strength based on salience.
        
        Returns:
            Consolidation strength multiplier (0.1 to 3.0)
        """
        if salience > 0.8:
            return 3.0  # Massive strengthening for breakthrough experiences
        elif salience > 0.6:
            return 2.0  # Strong consolidation for important experiences
        elif salience > 0.4:
            return 1.0  # Normal consolidation
        elif salience > 0.2:
            return 0.5  # Weak consolidation
        else:
            return 0.1  # Allow to decay rapidly
    
    def reset_statistics(self):
        """Reset salience statistics (for new episodes)."""
        self.lp_history.clear()
        self.energy_history.clear()
        self.lp_mean = 0.0
        self.lp_std = 1.0
        self.energy_mean = 50.0
        self.energy_std = 20.0
        
        # Reset distribution counters
        self.high_salience_count = 0
        self.medium_salience_count = 0
        self.low_salience_count = 0


class SalienceWeightedReplayBuffer:
    """
    Experience replay buffer that samples based on salience values.
    
    This implements the "prioritized dreaming" concept - the agent dreams
    most vividly about its most important discoveries and mistakes.
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.experiences = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def add(self, experience: SalientExperience):
        """Add experience with its salience as priority."""
        self.experiences.append(experience)
        # Use salience^alpha as priority for sampling
        priority = experience.salience_value ** self.alpha
        self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> List[SalientExperience]:
        """Sample experiences with probability proportional to salience."""
        if len(self.experiences) == 0:
            return []
        
        # Convert to numpy for easier sampling
        priorities_array = np.array(list(self.priorities))
        probabilities = priorities_array / priorities_array.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(
            len(self.experiences), 
            size=min(batch_size, len(self.experiences)),
            p=probabilities,
            replace=False
        )
        
        return [list(self.experiences)[i] for i in indices]
    
    def get_top_salient(self, k: int) -> List[SalientExperience]:
        """Get the k most salient experiences."""
        if not self.experiences:
            return []
        
        # Sort by salience value
        sorted_experiences = sorted(
            self.experiences, 
            key=lambda x: x.salience_value, 
            reverse=True
        )
        
        return sorted_experiences[:k]
