"""
Salience Modes Demonstration

This script demonstrates both salience testing modes:
1. Lossless Salience Testing (preserves all memories)
2. Salience Decay/Memory Decomposition (compresses low-salience memories)

Shows how the agent can discover optimal memory strategies through experimentation.
"""

import torch
import numpy as np
import logging
from typing import Dict, List
import time

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.salience_system import (
    SalienceCalculator, SalienceMode, SalientExperience, 
    CompressedMemory, SalienceWeightedReplayBuffer
)
from core.data_models import AgentState, SensoryInput

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_experience(step: int, context: str = "exploration") -> SalientExperience:
    """Create a mock salient experience for demonstration."""
    
    # Create mock sensory data
    visual = torch.randn(3, 10, 10)
    proprioception = torch.randn(4)
    energy_level = max(10, 100 - step * 0.1)  # Gradual energy decrease
    
    # Create mock agent state
    state = AgentState(
        position=torch.zeros(3),
        orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
        energy=energy_level,
        hidden_state=torch.zeros(64),
        active_goals=[],
        timestamp=step
    )
    
    # Simulate learning progress and energy changes
    if step % 50 == 0:  # Occasional breakthrough
        learning_progress = np.random.uniform(0.3, 0.8)
        energy_change = np.random.uniform(-5, 15)
    elif step % 10 == 0:  # Regular learning events
        learning_progress = np.random.uniform(0.1, 0.3)
        energy_change = np.random.uniform(-2, 5)
    else:  # Routine experiences
        learning_progress = np.random.uniform(0.0, 0.1)
        energy_change = np.random.uniform(-1, 1)
    
    # Create experience data
    experience_data = {
        'state': state,
        'action': np.random.randint(0, 4),
        'reward': max(0, energy_change),
        'learning_progress': learning_progress,
        'energy_change': energy_change
    }
    
    return SalientExperience(
        experience_data=experience_data,
        salience_value=0.0,  # Will be calculated
        lp_component=0.0,    # Will be calculated
        energy_component=0.0, # Will be calculated
        context=context,
        timestamp=step,
        last_access_time=time.time(),
        access_count=0
    )

def demonstrate_lossless_mode():
    """Demonstrate the lossless salience mode."""
    logger.info("\n" + "="*50)
    logger.info("DEMONSTRATING LOSSLESS SALIENCE MODE")
    logger.info("="*50)
    
    # Create lossless salience calculator
    salience_calc = SalienceCalculator(
        mode=SalienceMode.LOSSLESS,
        lp_weight=0.6,
        energy_weight=0.4
    )
    
    # Create replay buffer
    replay_buffer = SalienceWeightedReplayBuffer(capacity=1000)
    
    # Simulate 200 experiences
    experiences = []
    for step in range(200):
        exp = create_mock_experience(step, f"lossless_step_{step}")
        
        # Calculate salience
        exp.salience_value = salience_calc.calculate_salience(
            learning_progress=exp.experience_data['learning_progress'],
            energy_change=exp.experience_data['energy_change'],
            current_energy=exp.experience_data['state'].energy,
            context=exp.context
        )
        
        # Update components for tracking
        exp.lp_component = salience_calc._calculate_lp_salience(
            exp.experience_data['learning_progress']
        )
        exp.energy_component = salience_calc._calculate_energy_salience(
            exp.experience_data['energy_change'],
            exp.experience_data['state'].energy
        )
        
        experiences.append(exp)
        replay_buffer.add(exp)
    
    # Get metrics
    metrics = salience_calc.get_salience_metrics()
    compression_stats = salience_calc.get_compression_stats()
    
    logger.info(f"Total experiences: {len(experiences)}")
    logger.info(f"Average salience: {metrics.average_salience:.3f}")
    logger.info(f"Max salience: {metrics.max_salience:.3f}")
    logger.info(f"High salience experiences: {metrics.salience_distribution['high']:.1%}")
    logger.info(f"Compressed memories: {compression_stats['compressed_memories_count']}")
    logger.info(f"Memory saved: {compression_stats['memory_saved']:.2f}%")
    
    # Sample high-salience experiences
    high_salience = replay_buffer.get_top_salient(10)
    logger.info(f"\nTop 10 most salient experiences:")
    for i, exp in enumerate(high_salience[:5]):  # Show top 5
        logger.info(f"  {i+1}. Salience: {exp.salience_value:.3f}, Context: {exp.context}")
    
    return experiences, metrics, compression_stats

def demonstrate_decay_compression_mode():
    """Demonstrate the decay/compression salience mode."""
    logger.info("\n" + "="*50)
    logger.info("DEMONSTRATING DECAY/COMPRESSION SALIENCE MODE")
    logger.info("="*50)
    
    # Create decay/compression salience calculator
    salience_calc = SalienceCalculator(
        mode=SalienceMode.DECAY_COMPRESSION,
        lp_weight=0.6,
        energy_weight=0.4,
        decay_rate=0.02,
        salience_min=0.05,
        compression_threshold=0.15
    )
    
    # Create replay buffer
    replay_buffer = SalienceWeightedReplayBuffer(capacity=1000)
    
    # Simulate 200 experiences with periodic compression
    experiences = []
    current_time = time.time()
    
    for step in range(200):
        exp = create_mock_experience(step, f"decay_step_{step}")
        
        # Calculate salience
        exp.salience_value = salience_calc.calculate_salience(
            learning_progress=exp.experience_data['learning_progress'],
            energy_change=exp.experience_data['energy_change'],
            current_energy=exp.experience_data['state'].energy,
            context=exp.context
        )
        
        # Update components
        exp.lp_component = salience_calc._calculate_lp_salience(
            exp.experience_data['learning_progress']
        )
        exp.energy_component = salience_calc._calculate_energy_salience(
            exp.experience_data['energy_change'],
            exp.experience_data['state'].energy
        )
        
        experiences.append(exp)
        replay_buffer.add(exp)
        
        # Periodic decay and compression (every 50 steps)
        if step > 0 and step % 50 == 0:
            logger.info(f"\n--- Processing decay/compression at step {step} ---")
            
            # Apply decay
            current_time += 50  # Simulate time passage
            decayed_experiences = salience_calc.apply_salience_decay(experiences, current_time)
            
            # Apply compression
            remaining_experiences, compressed_memories = salience_calc.compress_low_salience_memories(
                decayed_experiences, current_time
            )
            
            logger.info(f"Decayed {len(decayed_experiences)} experiences")
            logger.info(f"Compressed {len(compressed_memories)} low-salience memories")
            logger.info(f"Remaining {len(remaining_experiences)} active memories")
            
            # Update experiences list
            experiences = remaining_experiences
    
    # Get final metrics
    metrics = salience_calc.get_salience_metrics()
    compression_stats = salience_calc.get_compression_stats()
    
    logger.info(f"\nFINAL RESULTS:")
    logger.info(f"Total experiences processed: 200")
    logger.info(f"Remaining active experiences: {len(experiences)}")
    logger.info(f"Average salience: {metrics.average_salience:.3f}")
    logger.info(f"Max salience: {metrics.max_salience:.3f}")
    logger.info(f"Compressed memories: {compression_stats['compressed_memories_count']}")
    logger.info(f"Total merged experiences: {compression_stats['total_merged']}")
    logger.info(f"Compression ratio: {compression_stats['compression_ratio']:.2f}")
    
    # Show compressed memories
    logger.info(f"\nCompressed memories:")
    for i, cm in enumerate(salience_calc.compressed_memories[:5]):  # Show first 5
        logger.info(f"  {i+1}. {cm.abstract_concept} (merged {cm.merged_count} experiences)")
    
    return experiences, metrics, compression_stats

def demonstrate_meta_learning_optimization():
    """Demonstrate meta-learning optimization of decay parameters."""
    logger.info("\n" + "="*50)
    logger.info("DEMONSTRATING META-LEARNING PARAMETER OPTIMIZATION")
    logger.info("="*50)
    
    salience_calc = SalienceCalculator(
        mode=SalienceMode.DECAY_COMPRESSION,
        decay_rate=0.01,
        compression_threshold=0.2
    )
    
    # Simulate different performance scenarios
    scenarios = [
        {"name": "High Performance", "learning_progress": 0.8, "survival_rate": 0.9, "memory_pressure": 0.6},
        {"name": "Poor Performance", "learning_progress": 0.2, "survival_rate": 0.3, "memory_pressure": 0.4},
        {"name": "High Memory Pressure", "learning_progress": 0.5, "survival_rate": 0.6, "memory_pressure": 0.9},
        {"name": "Low Memory Pressure", "learning_progress": 0.5, "survival_rate": 0.6, "memory_pressure": 0.2},
    ]
    
    logger.info(f"Initial parameters: decay_rate={salience_calc.decay_rate:.4f}, "
               f"compression_threshold={salience_calc.compression_threshold:.3f}")
    
    for scenario in scenarios:
        logger.info(f"\n--- {scenario['name']} Scenario ---")
        logger.info(f"Learning Progress: {scenario['learning_progress']:.1f}, "
                   f"Survival Rate: {scenario['survival_rate']:.1f}, "
                   f"Memory Pressure: {scenario['memory_pressure']:.1f}")
        
        # Optimize parameters
        optimized_params = salience_calc.optimize_decay_parameters({
            'learning_progress': scenario['learning_progress'],
            'survival_rate': scenario['survival_rate'],
            'memory_pressure': scenario['memory_pressure']
        })
        
        logger.info(f"Optimized: decay_rate={optimized_params['decay_rate']:.4f}, "
                   f"compression_threshold={optimized_params['compression_threshold']:.3f}")

def run_comprehensive_demo():
    """Run comprehensive demonstration of both salience modes."""
    logger.info("Starting Comprehensive Salience Modes Demonstration")
    logger.info("This demo shows how the agent can choose optimal memory strategies")
    
    # Demonstrate lossless mode
    lossless_exp, lossless_metrics, lossless_compression = demonstrate_lossless_mode()
    
    # Demonstrate decay/compression mode
    decay_exp, decay_metrics, decay_compression = demonstrate_decay_compression_mode()
    
    # Demonstrate meta-learning optimization
    demonstrate_meta_learning_optimization()
    
    # Compare results
    logger.info("\n" + "="*50)
    logger.info("MODE COMPARISON SUMMARY")
    logger.info("="*50)
    
    logger.info("LOSSLESS MODE:")
    logger.info(f"  - Preserves all {len(lossless_exp)} experiences")
    logger.info(f"  - Average salience: {lossless_metrics.average_salience:.3f}")
    logger.info(f"  - Memory usage: High (no compression)")
    logger.info(f"  - Best for: Critical learning phases, short-term tasks")
    
    logger.info("\nDECAY/COMPRESSION MODE:")
    logger.info(f"  - Compressed to {len(decay_exp)} active experiences")
    logger.info(f"  - Average salience: {decay_metrics.average_salience:.3f}")
    logger.info(f"  - Compression ratio: {decay_compression['compression_ratio']:.2f}")
    logger.info(f"  - Memory usage: Reduced by compression")
    logger.info(f"  - Best for: Long-term learning, memory-constrained environments")
    
    logger.info("\nRECOMMENDATION:")
    if lossless_metrics.average_salience > decay_metrics.average_salience:
        logger.info("  → LOSSLESS mode maintains higher salience quality")
        logger.info("  → Use when memory is not a constraint")
    else:
        logger.info("  → DECAY/COMPRESSION mode provides efficient memory usage")
        logger.info("  → Use for long-term autonomous learning")
    
    logger.info("\nThe agent can dynamically switch between modes based on:")
    logger.info("  - Available memory resources")
    logger.info("  - Learning phase (exploration vs exploitation)")
    logger.info("  - Task complexity and duration")
    logger.info("  - Performance requirements")

if __name__ == "__main__":
    run_comprehensive_demo()
