"""
Salience Mode Comparison Framework

This module provides a comprehensive testing framework to compare:
1. Lossless Salience Testing (current state)
2. Salience Decay/Memory Decomposition

The framework allows the agent to discover optimal memory strategies through experimentation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.salience_system import (
    SalienceCalculator, SalienceMode, SalientExperience, 
    CompressedMemory, SalienceWeightedReplayBuffer
)
from core.sleep_system import SleepCycle
from core.predictive_core import PredictiveCore
from core.data_models import Experience, AgentState, SensoryInput
# from environment.simple_survival_environment import SimpleSurvivalEnvironment  # Optional dependency

logger = logging.getLogger(__name__)

@dataclass
class SalienceModeResults:
    """Results from testing a specific salience mode."""
    mode: SalienceMode
    total_memories: int
    compressed_memories: int
    memory_efficiency: float  # Compression ratio
    learning_performance: float  # Average learning progress
    survival_rate: float  # Percentage of episodes survived
    average_salience: float
    high_salience_count: int
    processing_time: float
    memory_usage_mb: float

class SalienceModeComparator:
    """
    Framework for comparing different salience modes to determine optimal strategies.
    """
    
    def __init__(
        self,
        environment,  # Mock environment
        predictive_core: PredictiveCore,
        test_episodes: int = 10,
        episode_length: int = 1000,
        sleep_frequency: int = 100
    ):
        self.environment = environment
        self.predictive_core = predictive_core
        self.test_episodes = test_episodes
        self.episode_length = episode_length
        self.sleep_frequency = sleep_frequency
        
        # Results storage
        self.results: Dict[SalienceMode, SalienceModeResults] = {}
        self.detailed_logs: Dict[SalienceMode, List[Dict]] = {}
        
    def run_comparison(self) -> Dict[SalienceMode, SalienceModeResults]:
        """
        Run comprehensive comparison between salience modes.
        
        Returns:
            Dictionary mapping each mode to its results
        """
        logger.info("Starting salience mode comparison...")
        
        # Test Lossless Mode
        logger.info("Testing LOSSLESS mode...")
        lossless_results = self._test_salience_mode(SalienceMode.LOSSLESS)
        self.results[SalienceMode.LOSSLESS] = lossless_results
        
        # Test Decay/Compression Mode
        logger.info("Testing DECAY_COMPRESSION mode...")
        decay_results = self._test_salience_mode(SalienceMode.DECAY_COMPRESSION)
        self.results[SalienceMode.DECAY_COMPRESSION] = decay_results
        
        # Generate comparison report
        self._generate_comparison_report()
        
        return self.results
    
    def _test_salience_mode(self, mode: SalienceMode) -> SalienceModeResults:
        """Test a specific salience mode across multiple episodes."""
        
        # Configure salience calculator for this mode
        salience_calc = SalienceCalculator(
            mode=mode,
            decay_rate=0.01 if mode == SalienceMode.DECAY_COMPRESSION else 0.0,
            salience_min=0.05,
            compression_threshold=0.15
        )
        
        # Configure sleep system
        sleep_system = SleepCycle(
            predictive_core=self.predictive_core,
            use_salience_weighting=True
        )
        sleep_system.salience_calculator = salience_calc
        sleep_system.salience_replay_buffer = SalienceWeightedReplayBuffer()
        
        # Track metrics across episodes
        episode_results = []
        total_memories = 0
        total_compressed = 0
        total_learning_progress = 0.0
        survived_episodes = 0
        total_processing_time = 0.0
        
        for episode in range(self.test_episodes):
            episode_start_time = time.time()
            
            # Reset environment and agent
            state = self.environment.reset()
            episode_experiences = []
            episode_lp = []
            
            for step in range(self.episode_length):
                # Simple random action for testing
                action = np.random.randint(0, self.environment.action_space_size)
                
                # Take step in environment
                next_state, reward, done, info = self.environment.step(action)
                
                # Calculate learning progress (simplified)
                lp = self._calculate_learning_progress(state, next_state, reward)
                episode_lp.append(lp)
                
                # Create experience
                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                
                # Calculate salience and create salient experience
                energy_change = next_state.energy_level - state.energy_level
                salient_exp = salience_calc.create_salient_experience(
                    experience_data={'experience': experience},
                    learning_progress=lp,
                    energy_change=energy_change,
                    current_energy=next_state.energy_level,
                    context=f"episode_{episode}_step_{step}"
                )
                
                episode_experiences.append(salient_exp)
                sleep_system.salience_replay_buffer.add(salient_exp)
                
                # Periodic sleep cycles
                if step % self.sleep_frequency == 0 and step > 0:
                    sleep_system.enter_sleep()
                    sleep_results = sleep_system.execute_sleep_cycle([])
                    sleep_system.wake_up()
                
                state = next_state
                
                if done:
                    if next_state.energy_level > 0:
                        survived_episodes += 1
                    break
            
            episode_time = time.time() - episode_start_time
            total_processing_time += episode_time
            
            # Calculate episode metrics
            avg_lp = np.mean(episode_lp) if episode_lp else 0.0
            total_learning_progress += avg_lp
            
            episode_result = {
                'episode': episode,
                'avg_learning_progress': avg_lp,
                'survived': state.energy_level > 0,
                'total_experiences': len(episode_experiences),
                'processing_time': episode_time
            }
            episode_results.append(episode_result)
            
            total_memories += len(episode_experiences)
        
        # Get final compression stats
        compression_stats = salience_calc.get_compression_stats()
        total_compressed = compression_stats['compressed_memories_count']
        
        # Calculate salience metrics
        salience_metrics = salience_calc.get_salience_metrics()
        
        # Calculate memory usage (simplified estimation)
        memory_usage_mb = self._estimate_memory_usage(total_memories, total_compressed)
        
        # Store detailed logs
        self.detailed_logs[mode] = episode_results
        
        return SalienceModeResults(
            mode=mode,
            total_memories=total_memories,
            compressed_memories=total_compressed,
            memory_efficiency=total_compressed / max(total_memories, 1),
            learning_performance=total_learning_progress / self.test_episodes,
            survival_rate=survived_episodes / self.test_episodes,
            average_salience=salience_metrics.average_salience,
            high_salience_count=int(salience_metrics.salience_distribution.get('high', 0) * total_memories),
            processing_time=total_processing_time,
            memory_usage_mb=memory_usage_mb
        )
    
    def _calculate_learning_progress(self, state: AgentState, next_state: AgentState, reward: float) -> float:
        """Calculate learning progress for an experience (simplified)."""
        # Simple heuristic: positive reward indicates learning
        base_lp = max(reward, 0.0) * 0.1
        
        # Energy efficiency bonus
        energy_efficiency = (next_state.energy_level - state.energy_level) / 100.0
        
        # Add some randomness to simulate prediction error changes
        noise = np.random.normal(0, 0.05)
        
        return max(0.0, base_lp + energy_efficiency * 0.05 + noise)
    
    def _estimate_memory_usage(self, total_memories: int, compressed_memories: int) -> float:
        """Estimate memory usage in MB (simplified)."""
        # Assume each full memory takes ~1KB, compressed memories take ~0.1KB
        full_memories = total_memories - compressed_memories
        usage_bytes = (full_memories * 1024) + (compressed_memories * 102.4)
        return usage_bytes / (1024 * 1024)  # Convert to MB
    
    def _generate_comparison_report(self):
        """Generate a comprehensive comparison report."""
        logger.info("\n" + "="*60)
        logger.info("SALIENCE MODE COMPARISON REPORT")
        logger.info("="*60)
        
        for mode, results in self.results.items():
            logger.info(f"\n{mode.value.upper()} MODE RESULTS:")
            logger.info(f"  Total Memories: {results.total_memories}")
            logger.info(f"  Compressed Memories: {results.compressed_memories}")
            logger.info(f"  Memory Efficiency: {results.memory_efficiency:.3f}")
            logger.info(f"  Learning Performance: {results.learning_performance:.3f}")
            logger.info(f"  Survival Rate: {results.survival_rate:.3f}")
            logger.info(f"  Average Salience: {results.average_salience:.3f}")
            logger.info(f"  High Salience Count: {results.high_salience_count}")
            logger.info(f"  Processing Time: {results.processing_time:.2f}s")
            logger.info(f"  Memory Usage: {results.memory_usage_mb:.2f}MB")
        
        # Determine optimal mode
        optimal_mode = self._determine_optimal_mode()
        logger.info(f"\nRECOMMENDED MODE: {optimal_mode.value.upper()}")
        logger.info("="*60)
    
    def _determine_optimal_mode(self) -> SalienceMode:
        """
        Determine the optimal salience mode based on multiple criteria.
        
        Returns:
            The recommended salience mode
        """
        if len(self.results) < 2:
            return list(self.results.keys())[0]
        
        lossless = self.results[SalienceMode.LOSSLESS]
        decay = self.results[SalienceMode.DECAY_COMPRESSION]
        
        # Scoring criteria (weights can be adjusted)
        criteria_weights = {
            'memory_efficiency': 0.3,
            'learning_performance': 0.4,
            'survival_rate': 0.2,
            'processing_speed': 0.1
        }
        
        # Calculate scores (higher is better)
        lossless_score = (
            criteria_weights['learning_performance'] * lossless.learning_performance +
            criteria_weights['survival_rate'] * lossless.survival_rate +
            criteria_weights['processing_speed'] * (1.0 / max(lossless.processing_time, 1.0)) * 100
        )
        
        decay_score = (
            criteria_weights['memory_efficiency'] * decay.memory_efficiency +
            criteria_weights['learning_performance'] * decay.learning_performance +
            criteria_weights['survival_rate'] * decay.survival_rate +
            criteria_weights['processing_speed'] * (1.0 / max(decay.processing_time, 1.0)) * 100
        )
        
        return SalienceMode.DECAY_COMPRESSION if decay_score > lossless_score else SalienceMode.LOSSLESS
    
    def save_results(self, filepath: str):
        """Save comparison results to JSON file."""
        results_dict = {}
        for mode, results in self.results.items():
            results_dict[mode.value] = {
                'total_memories': results.total_memories,
                'compressed_memories': results.compressed_memories,
                'memory_efficiency': results.memory_efficiency,
                'learning_performance': results.learning_performance,
                'survival_rate': results.survival_rate,
                'average_salience': results.average_salience,
                'high_salience_count': results.high_salience_count,
                'processing_time': results.processing_time,
                'memory_usage_mb': results.memory_usage_mb
            }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def plot_comparison(self, save_path: Optional[str] = None):
        """Generate comparison plots."""
        if len(self.results) < 2:
            logger.warning("Need at least 2 modes to generate comparison plots")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Salience Mode Comparison', fontsize=16)
        
        modes = list(self.results.keys())
        mode_names = [mode.value for mode in modes]
        
        # Memory Efficiency
        memory_effs = [self.results[mode].memory_efficiency for mode in modes]
        axes[0, 0].bar(mode_names, memory_effs)
        axes[0, 0].set_title('Memory Efficiency')
        axes[0, 0].set_ylabel('Compression Ratio')
        
        # Learning Performance
        learning_perfs = [self.results[mode].learning_performance for mode in modes]
        axes[0, 1].bar(mode_names, learning_perfs)
        axes[0, 1].set_title('Learning Performance')
        axes[0, 1].set_ylabel('Average Learning Progress')
        
        # Survival Rate
        survival_rates = [self.results[mode].survival_rate for mode in modes]
        axes[1, 0].bar(mode_names, survival_rates)
        axes[1, 0].set_title('Survival Rate')
        axes[1, 0].set_ylabel('Proportion Survived')
        
        # Memory Usage
        memory_usage = [self.results[mode].memory_usage_mb for mode in modes]
        axes[1, 1].bar(mode_names, memory_usage)
        axes[1, 1].set_title('Memory Usage')
        axes[1, 1].set_ylabel('Memory (MB)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Comparison plot saved to {save_path}")
        else:
            plt.show()


def run_salience_comparison_demo():
    """Run a demonstration of the salience mode comparison."""
    def _create_mock_environment(self):
        """Create a mock environment for testing."""
        # Mock environment class for testing
        class MockEnvironment:
            def __init__(self):
                self.grid_size = (10, 10)
                self.num_food_sources = 5
                self.num_obstacles = 3
            
            def reset(self):
                return {"position": (5, 5), "energy": 100}
            
            def step(self, action):
                # Mock step function
                return {"position": (5, 5), "energy": 95}, 0.1, False, {}
        
        return MockEnvironment()
    
    def _create_mock_environment_standalone():
        """Create a mock environment for testing (standalone function)."""
        # Mock environment class for testing
        class MockEnvironment:
            def __init__(self):
                self.grid_size = (10, 10)
                self.num_food_sources = 5
                self.num_obstacles = 3
            
            def reset(self):
                return {"position": (5, 5), "energy": 100}
            
            def step(self, action):
                # Mock step function
                return {"position": (5, 5), "energy": 95}, 0.1, False, {}
        
        return MockEnvironment()
    
    logging.basicConfig(level=logging.INFO)
    
    # Create environment and predictive core
    env = _create_mock_environment_standalone()
    
    # Simple predictive core for testing
    predictive_core = PredictiveCore(
        visual_input_size=(3, 10, 10),
        proprioception_size=4,
        hidden_size=64,
        memory_size=128,
        use_memory=True
    )
    
    # Create comparator
    comparator = SalienceModeComparator(
        environment=env,
        predictive_core=predictive_core,
        test_episodes=5,  # Reduced for demo
        episode_length=500,
        sleep_frequency=50
    )
    
    # Run comparison
    results = comparator.run_comparison()
    
    # Save results
    comparator.save_results('salience_comparison_results.json')
    
    # Generate plots
    comparator.plot_comparison('salience_comparison.png')
    
    return results


if __name__ == "__main__":
    results = run_salience_comparison_demo()
