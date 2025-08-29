#!/usr/bin/env python3
"""
Phase 0: Energy System and Survival Test

This script validates the energy system and death mechanics in a simple environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import argparse
from pathlib import Path
import logging
from datetime import datetime
import numpy as np

from src.core.energy_system import EnergySystem, DeathManager
from src.core.data_models import AgentState
from src.environment.simple_survival import SimpleSurvivalEnvironment, RandomAgent


def setup_logging(log_dir: Path):
    """Set up logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"survival_test_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_survival_simulation(config: dict, logger: logging.Logger) -> dict:
    """
    Run survival simulation with energy system.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        results: Simulation results
    """
    logger.info("Starting survival simulation...")
    
    # Create environment
    env = SimpleSurvivalEnvironment(
        grid_size=(15, 15),
        num_food_sources=3,
        food_respawn_rate=0.02,
        seed=config['experiment']['seed']
    )
    
    # Create energy system
    energy_config = config['energy']
    energy_system = EnergySystem(**energy_config)
    
    # Create death manager
    death_config = config['death_system']
    death_manager = DeathManager(
        memory_size=32,  # Small for testing
        word_size=8,
        **death_config
    )
    
    # Create random agent
    agent = RandomAgent(action_space_size=2)
    
    # Simulation metrics
    metrics = {
        'steps_survived': [],
        'food_collected': [],
        'deaths': 0,
        'energy_history': [],
        'survival_times': []
    }
    
    # Run multiple lives
    max_lives = 5
    max_steps_per_life = 2000
    
    for life in range(max_lives):
        logger.info(f"Starting life {life + 1}/{max_lives}")
        
        # Reset environment and energy
        observation = env.reset()
        energy_system.reset_energy()
        
        # Create agent state
        agent_state = AgentState(
            position=torch.tensor([7.5, 7.5, 0.0]),  # Center of 15x15 grid
            orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
            energy=energy_system.get_energy_level(),
            hidden_state=torch.randn(32),  # Dummy hidden state
            active_goals=[],
            memory_state=torch.randn(32, 8)  # Dummy memory
        )
        
        # Life metrics
        steps_this_life = 0
        food_this_life = 0
        life_energy_history = []
        
        # Run until death or max steps
        for step in range(max_steps_per_life):
            # Update observation with current energy
            observation.energy_level = energy_system.get_energy_level()
            
            # Agent acts
            action = agent.act(observation)
            action_cost = agent.get_action_cost(action)
            
            # Environment step
            next_observation, env_reward, done, info = env.step(action)
            
            # Energy consumption
            computation_cost = 1.0  # Simulate some computation
            remaining_energy = energy_system.consume_energy(action_cost, computation_cost)
            
            # Food collection gives energy
            if env_reward > 0:
                energy_system.add_energy(energy_config['food_energy_value'])
                food_this_life += 1
                logger.debug(f"Food collected! Energy: {energy_system.get_energy_level():.1f}")
                
            # Track metrics
            life_energy_history.append(remaining_energy)
            steps_this_life += 1
            
            # Check for death
            if energy_system.is_dead():
                logger.info(f"Agent died after {steps_this_life} steps (Life {life + 1})")
                
                # Handle death
                agent_state.energy = 0.0
                new_agent_state = death_manager.selective_reset(agent_state)
                
                metrics['deaths'] += 1
                metrics['survival_times'].append(steps_this_life)
                break
                
            # Update for next step
            observation = next_observation
            
        # Record life metrics
        metrics['steps_survived'].append(steps_this_life)
        metrics['food_collected'].append(food_this_life)
        metrics['energy_history'].extend(life_energy_history)
        
        logger.info(f"Life {life + 1} completed: {steps_this_life} steps, {food_this_life} food")
        
    return metrics


def analyze_survival_results(metrics: dict, logger: logging.Logger) -> dict:
    """Analyze survival simulation results."""
    logger.info("Analyzing survival results...")
    
    # Basic statistics
    total_steps = sum(metrics['steps_survived'])
    total_food = sum(metrics['food_collected'])
    avg_survival_time = np.mean(metrics['survival_times']) if metrics['survival_times'] else 0
    
    # Energy analysis
    energy_history = metrics['energy_history']
    if energy_history:
        avg_energy = np.mean(energy_history)
        min_energy = np.min(energy_history)
        energy_variance = np.var(energy_history)
    else:
        avg_energy = min_energy = energy_variance = 0
        
    # Survival rate
    lives_completed = len(metrics['steps_survived'])
    deaths = metrics['deaths']
    survival_rate = (lives_completed - deaths) / lives_completed if lives_completed > 0 else 0
    
    # Food efficiency
    food_per_step = total_food / total_steps if total_steps > 0 else 0
    
    results = {
        'total_steps': total_steps,
        'total_food': total_food,
        'deaths': deaths,
        'survival_rate': survival_rate,
        'avg_survival_time': avg_survival_time,
        'avg_energy': avg_energy,
        'min_energy': min_energy,
        'energy_variance': energy_variance,
        'food_efficiency': food_per_step
    }
    
    logger.info("Survival Analysis Results:")
    logger.info(f"  Total Steps: {total_steps}")
    logger.info(f"  Deaths: {deaths}")
    logger.info(f"  Survival Rate: {survival_rate:.2%}")
    logger.info(f"  Avg Survival Time: {avg_survival_time:.1f} steps")
    logger.info(f"  Food Efficiency: {food_per_step:.4f} food/step")
    logger.info(f"  Average Energy: {avg_energy:.2f}")
    
    return results


def test_death_mechanics(config: dict, logger: logging.Logger) -> dict:
    """Test death and memory preservation mechanics."""
    logger.info("Testing death mechanics...")
    
    # Create death manager
    death_config = config['death_system']
    death_manager = DeathManager(
        memory_size=16,  # Small for testing
        word_size=4,
        **death_config
    )
    
    # Create test agent state with memory
    original_memory = torch.randn(16, 4)
    agent_state = AgentState(
        position=torch.tensor([5.0, 5.0, 0.0]),
        orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        energy=0.0,  # Dead
        hidden_state=torch.randn(32),
        active_goals=[],
        memory_state=original_memory.clone()
    )
    
    # Add memory usage for importance calculation
    agent_state.memory_usage = torch.rand(16)
    
    # Test selective reset
    new_state = death_manager.selective_reset(agent_state)
    
    # Analyze preservation
    if new_state.memory_state is not None:
        preserved_locations = (new_state.memory_state != 0).any(dim=1).sum().item()
        total_locations = new_state.memory_state.size(0)
        preservation_ratio = preserved_locations / total_locations
    else:
        preservation_ratio = 0.0
        
    # Check reset properties
    energy_reset = new_state.energy == 100.0
    hidden_reset = torch.allclose(new_state.hidden_state, torch.zeros_like(new_state.hidden_state))
    goals_cleared = len(new_state.active_goals) == 0
    
    results = {
        'preservation_ratio': preservation_ratio,
        'energy_reset': energy_reset,
        'hidden_reset': hidden_reset,
        'goals_cleared': goals_cleared,
        'death_count': death_manager.death_count
    }
    
    logger.info("Death Mechanics Results:")
    logger.info(f"  Memory Preservation Ratio: {preservation_ratio:.2%}")
    logger.info(f"  Energy Reset: {energy_reset}")
    logger.info(f"  Hidden State Reset: {hidden_reset}")
    logger.info(f"  Goals Cleared: {goals_cleared}")
    
    return results


def main():
    """Main experiment function."""
    parser = argparse.ArgumentParser(description='Phase 0 Survival Test Experiment')
    parser.add_argument('--config', type=str, default='configs/phase0_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='experiments/results/phase0',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir / 'logs')
    config = load_config(args.config)
    
    # Set random seed
    torch.manual_seed(config['experiment']['seed'])
    
    logger.info("Phase 0: Energy System and Survival Test")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {output_dir}")
    
    try:
        # Run survival simulation
        survival_metrics = run_survival_simulation(config, logger)
        survival_results = analyze_survival_results(survival_metrics, logger)
        
        # Test death mechanics
        death_results = test_death_mechanics(config, logger)
        
        # Evaluate overall results
        logger.info("=" * 50)
        logger.info("Overall Evaluation:")
        
        # Pass criteria
        min_survival_rate = 0.2  # At least 20% should survive to max steps
        min_food_efficiency = 0.001  # Should collect some food
        min_avg_energy = 10.0  # Should maintain some energy on average
        
        survival_ok = survival_results['survival_rate'] >= min_survival_rate
        food_ok = survival_results['food_efficiency'] >= min_food_efficiency
        energy_ok = survival_results['avg_energy'] >= min_avg_energy
        death_mechanics_ok = (
            death_results['energy_reset'] and 
            death_results['hidden_reset'] and 
            death_results['goals_cleared']
        )
        
        logger.info(f"Survival Rate: {survival_results['survival_rate']:.2%} ({'PASS' if survival_ok else 'FAIL'})")
        logger.info(f"Food Efficiency: {survival_results['food_efficiency']:.4f} ({'PASS' if food_ok else 'FAIL'})")
        logger.info(f"Energy Management: {survival_results['avg_energy']:.2f} ({'PASS' if energy_ok else 'FAIL'})")
        logger.info(f"Death Mechanics: {'PASS' if death_mechanics_ok else 'FAIL'}")
        
        overall_passed = survival_ok and food_ok and energy_ok and death_mechanics_ok
        
        # Save results
        results = {
            'survival_simulation': {
                'metrics': survival_metrics,
                'analysis': survival_results
            },
            'death_mechanics': death_results,
            'evaluation': {
                'survival_ok': survival_ok,
                'food_ok': food_ok,
                'energy_ok': energy_ok,
                'death_mechanics_ok': death_mechanics_ok,
                'overall_passed': overall_passed
            },
            'config': config
        }
        
        results_file = output_dir / 'survival_test_results.yaml'
        with open(results_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
            
        logger.info(f"Results saved to {results_file}")
        
        if overall_passed:
            logger.info("✓ Survival system validation PASSED!")
            sys.exit(0)
        else:
            logger.error("✗ Survival system validation FAILED!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()