"""
Salience System Integration Demo

This demonstrates the complete Salience system creating a perfect feedback loop:
1. Learning Progress and Energy drives determine experience importance
2. High-salience experiences get prioritized replay during sleep
3. Memory consolidation strength is based on salience values
4. Meta-learning uses salience clusters to invent new goals
5. Context-aware retrieval prioritizes salient memories

The agent literally dreams about its most important discoveries and mistakes.
"""

import torch
import numpy as np
import logging
from typing import Dict, List
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.salience_system import SalienceCalculator, SalienceWeightedReplayBuffer, SalientExperience
from core.learning_progress import LearningProgressDrive
from core.energy_system import EnergySystem
from core.sleep_system import SleepCycle
from core.meta_learning import MetaLearningSystem
from core.data_models import Experience, AgentState, SensoryInput
from memory.dnc import DNCMemory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalienceSystemDemo:
    """Demonstrates the complete Salience system integration."""
    
    def __init__(self):
        # Initialize core systems
        self.learning_progress = LearningProgressDrive()
        self.energy_system = EnergySystem()
        self.salience_calculator = SalienceCalculator()
        self.meta_learning = MetaLearningSystem(use_salience_based_goals=True)
        self.memory = DNCMemory(memory_size=128, word_size=32)
        
        # Experience tracking
        self.experiences = []
        self.salient_experiences = []
        
    def simulate_breakthrough_discovery(self, step: int) -> Dict:
        """Simulate a major learning breakthrough - high LP spike."""
        logger.info(f"Step {step}: Simulating breakthrough discovery!")
        
        # Massive learning progress spike (breakthrough understanding)
        learning_progress = 0.8  # Very high LP
        energy_change = -2.0     # Thinking is costly
        current_energy = 70.0
        
        # Calculate salience - this should be very high
        salience = self.salience_calculator.calculate_salience(
            learning_progress=learning_progress,
            energy_change=energy_change,
            current_energy=current_energy,
            context="breakthrough_discovery"
        )
        
        # Create mock experience
        experience = self._create_mock_experience(learning_progress, step)
        
        # Create salient experience
        salient_exp = self.salience_calculator.create_salient_experience(
            experience_data={'experience': experience},
            learning_progress=learning_progress,
            energy_change=energy_change,
            current_energy=current_energy,
            context="breakthrough_discovery"
        )
        
        self.salient_experiences.append(salient_exp)
        self.meta_learning.add_salient_experience(salient_exp)
        
        return {
            'type': 'breakthrough',
            'salience': salience,
            'learning_progress': learning_progress,
            'consolidation_strength': self.salience_calculator.get_consolidation_strength(salience)
        }
    
    def simulate_survival_crisis(self, step: int) -> Dict:
        """Simulate a life-threatening situation - major energy loss."""
        logger.info(f"Step {step}: Simulating survival crisis!")
        
        # Major energy loss (near-death experience)
        learning_progress = 0.1   # Low LP during crisis
        energy_change = -15.0     # Major energy loss
        current_energy = 20.0     # Very low energy
        
        # Calculate salience - should be high due to survival importance
        salience = self.salience_calculator.calculate_salience(
            learning_progress=learning_progress,
            energy_change=energy_change,
            current_energy=current_energy,
            context="survival_crisis"
        )
        
        # Create mock experience
        experience = self._create_mock_experience(learning_progress, step)
        
        # Create salient experience
        salient_exp = self.salience_calculator.create_salient_experience(
            experience_data={'experience': experience},
            learning_progress=learning_progress,
            energy_change=energy_change,
            current_energy=current_energy,
            context="survival_crisis"
        )
        
        self.salient_experiences.append(salient_exp)
        self.meta_learning.add_salient_experience(salient_exp)
        
        return {
            'type': 'survival_crisis',
            'salience': salience,
            'energy_change': energy_change,
            'consolidation_strength': self.salience_calculator.get_consolidation_strength(salience)
        }
    
    def simulate_food_discovery(self, step: int) -> Dict:
        """Simulate finding a rich food source - major energy gain."""
        logger.info(f"Step {step}: Simulating food discovery!")
        
        # Major energy gain (found food)
        learning_progress = 0.3   # Moderate LP from successful foraging
        energy_change = 12.0      # Major energy gain
        current_energy = 85.0     # High energy after eating
        
        # Calculate salience - should be high due to survival success
        salience = self.salience_calculator.calculate_salience(
            learning_progress=learning_progress,
            energy_change=energy_change,
            current_energy=current_energy,
            context="food_discovery"
        )
        
        # Create mock experience
        experience = self._create_mock_experience(learning_progress, step)
        
        # Create salient experience
        salient_exp = self.salience_calculator.create_salient_experience(
            experience_data={'experience': experience},
            learning_progress=learning_progress,
            energy_change=energy_change,
            current_energy=current_energy,
            context="food_discovery"
        )
        
        self.salient_experiences.append(salient_exp)
        self.meta_learning.add_salient_experience(salient_exp)
        
        return {
            'type': 'food_discovery',
            'salience': salience,
            'energy_change': energy_change,
            'consolidation_strength': self.salience_calculator.get_consolidation_strength(salience)
        }
    
    def simulate_mundane_experience(self, step: int) -> Dict:
        """Simulate a boring, low-importance experience."""
        # Low learning progress, minimal energy change
        learning_progress = 0.02  # Very low LP
        energy_change = -0.1      # Minimal energy cost
        current_energy = 60.0     # Normal energy
        
        # Calculate salience - should be very low
        salience = self.salience_calculator.calculate_salience(
            learning_progress=learning_progress,
            energy_change=energy_change,
            current_energy=current_energy,
            context="mundane"
        )
        
        return {
            'type': 'mundane',
            'salience': salience,
            'learning_progress': learning_progress,
            'consolidation_strength': self.salience_calculator.get_consolidation_strength(salience)
        }
    
    def _create_mock_experience(self, learning_progress: float, step: int) -> Experience:
        """Create a mock experience for demonstration."""
        # Create mock sensory input
        visual = torch.randn(3, 64, 64)  # Random visual input
        proprioception = torch.randn(10)  # Random proprioceptive input
        
        state = SensoryInput(
            visual=visual,
            proprioception=proprioception,
            energy_level=50.0,
            timestamp=step
        )
        
        next_state = SensoryInput(
            visual=visual + torch.randn_like(visual) * 0.1,
            proprioception=proprioception + torch.randn_like(proprioception) * 0.1,
            energy_level=50.0,
            timestamp=step + 1
        )
        
        return Experience(
            state=state,
            action=torch.randn(4),  # Random action
            reward=0.1,
            next_state=next_state,
            learning_progress=learning_progress,
            timestamp=step
        )
    
    def demonstrate_salience_weighted_replay(self):
        """Demonstrate how high-salience experiences are prioritized during sleep."""
        logger.info("\n=== SALIENCE-WEIGHTED EXPERIENCE REPLAY ===")
        
        # Create replay buffer with our salient experiences
        replay_buffer = SalienceWeightedReplayBuffer(capacity=1000)
        
        for salient_exp in self.salient_experiences:
            replay_buffer.add(salient_exp)
        
        # Sample experiences - should prioritize high-salience ones
        sampled = replay_buffer.sample(5)
        
        logger.info("Sampled experiences for replay (agent's dreams):")
        for i, exp in enumerate(sampled):
            logger.info(f"  Dream {i+1}: {exp.context} (salience={exp.salience_value:.3f})")
        
        # Get top salient experiences
        top_salient = replay_buffer.get_top_salient(3)
        logger.info("\nMost salient memories (strongest dreams):")
        for i, exp in enumerate(top_salient):
            logger.info(f"  Top {i+1}: {exp.context} (salience={exp.salience_value:.3f})")
    
    def demonstrate_memory_consolidation(self):
        """Demonstrate salience-based memory consolidation."""
        logger.info("\n=== SALIENCE-BASED MEMORY CONSOLIDATION ===")
        
        # Simulate memory consolidation for each experience type
        for salient_exp in self.salient_experiences:
            consolidation_strength = self.salience_calculator.get_consolidation_strength(
                salient_exp.salience_value
            )
            
            should_consolidate = self.salience_calculator.should_consolidate_strongly(
                salient_exp.salience_value
            )
            
            logger.info(f"{salient_exp.context}:")
            logger.info(f"  Salience: {salient_exp.salience_value:.3f}")
            logger.info(f"  Consolidation strength: {consolidation_strength:.1f}x")
            logger.info(f"  Strong consolidation: {should_consolidate}")
    
    def demonstrate_goal_invention(self):
        """Demonstrate how salience clusters lead to goal invention."""
        logger.info("\n=== SALIENCE-BASED GOAL INVENTION ===")
        
        # Check what goals were invented from high-salience experience clusters
        summary = self.meta_learning.get_meta_learning_summary()
        
        logger.info(f"Invented goals: {summary.get('invented_goals', 0)}")
        
        # Get active goals for different contexts
        contexts = ["breakthrough_discovery", "survival_crisis", "food_discovery"]
        
        for context in contexts:
            active_goals = self.meta_learning.get_active_invented_goals(context)
            if active_goals:
                logger.info(f"\nActive goals for {context}:")
                for goal in active_goals:
                    pattern = goal['pattern']
                    logger.info(f"  Goal: {pattern.get('description', 'Unknown')}")
                    logger.info(f"  Type: {pattern.get('type', 'Unknown')}")
                    logger.info(f"  Confidence: {goal['confidence']:.3f}")
    
    def demonstrate_context_aware_retrieval(self):
        """Demonstrate context-aware memory retrieval using salience."""
        logger.info("\n=== CONTEXT-AWARE MEMORY RETRIEVAL ===")
        
        # Simulate being in a situation similar to past high-salience events
        current_context = torch.randn(32)  # Mock current context vector
        
        # Update memory with salience values (mock)
        memory_indices = torch.arange(5)
        salience_values = torch.tensor([exp.salience_value for exp in self.salient_experiences[:5]])
        
        self.memory.update_memory_salience(memory_indices, salience_values)
        
        # Retrieve salient memories
        retrieved = self.memory.retrieve_salient_memories(
            current_context, salience_threshold=0.5, max_retrievals=3
        )
        
        logger.info(f"Retrieved {len(retrieved)} salient memories:")
        for i, (memory_vector, relevance) in enumerate(retrieved):
            logger.info(f"  Memory {i+1}: relevance={relevance:.3f}")
        
        # Get high-salience memories
        high_salience_memories = self.memory.get_high_salience_memories(threshold=0.6)
        logger.info(f"\nHigh-salience memories in storage: {len(high_salience_memories)}")
    
    def run_complete_demo(self):
        """Run the complete salience system demonstration."""
        logger.info("🧠 SALIENCE SYSTEM COMPLETE DEMONSTRATION 🧠")
        logger.info("=" * 60)
        
        # Simulate various experiences with different salience levels
        experiences_log = []
        
        # Step 1-5: Mundane experiences (low salience)
        for step in range(5):
            result = self.simulate_mundane_experience(step)
            experiences_log.append(result)
        
        # Step 6: Major breakthrough (very high salience)
        result = self.simulate_breakthrough_discovery(6)
        experiences_log.append(result)
        
        # Step 7-10: More mundane experiences
        for step in range(7, 11):
            result = self.simulate_mundane_experience(step)
            experiences_log.append(result)
        
        # Step 11: Survival crisis (high salience)
        result = self.simulate_survival_crisis(11)
        experiences_log.append(result)
        
        # Step 12: Food discovery (high salience)
        result = self.simulate_food_discovery(12)
        experiences_log.append(result)
        
        # Step 13-15: More mundane experiences
        for step in range(13, 16):
            result = self.simulate_mundane_experience(step)
            experiences_log.append(result)
        
        # Show salience distribution
        logger.info("\n=== EXPERIENCE SALIENCE SUMMARY ===")
        high_salience = [exp for exp in experiences_log if exp['salience'] > 0.6]
        medium_salience = [exp for exp in experiences_log if 0.3 < exp['salience'] <= 0.6]
        low_salience = [exp for exp in experiences_log if exp['salience'] <= 0.3]
        
        logger.info(f"High salience experiences (>0.6): {len(high_salience)}")
        logger.info(f"Medium salience experiences (0.3-0.6): {len(medium_salience)}")
        logger.info(f"Low salience experiences (≤0.3): {len(low_salience)}")
        
        for exp in high_salience:
            logger.info(f"  🔥 {exp['type']}: salience={exp['salience']:.3f}")
        
        # Demonstrate all components
        self.demonstrate_salience_weighted_replay()
        self.demonstrate_memory_consolidation()
        self.demonstrate_goal_invention()
        self.demonstrate_context_aware_retrieval()
        
        # Show final metrics
        logger.info("\n=== FINAL SALIENCE METRICS ===")
        metrics = self.salience_calculator.get_salience_metrics()
        logger.info(f"Average salience: {metrics.average_salience:.3f}")
        logger.info(f"Max salience: {metrics.max_salience:.3f}")
        logger.info(f"Distribution: {metrics.salience_distribution}")
        logger.info(f"LP contribution: {metrics.lp_contribution:.3f}")
        logger.info(f"Energy contribution: {metrics.energy_contribution:.3f}")
        
        logger.info("\n🎯 KEY INSIGHTS:")
        logger.info("1. Breakthrough discoveries have highest salience (LP spike)")
        logger.info("2. Survival events have high salience (energy change)")
        logger.info("3. Mundane experiences have low salience (minimal change)")
        logger.info("4. High-salience memories get prioritized replay during sleep")
        logger.info("5. Memory consolidation strength scales with salience")
        logger.info("6. Goal invention emerges from salience clusters")
        logger.info("7. Context-aware retrieval prioritizes salient memories")
        logger.info("\n✨ The agent's own drives determine what's important! ✨")

def main():
    """Run the salience system demonstration."""
    demo = SalienceSystemDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()
