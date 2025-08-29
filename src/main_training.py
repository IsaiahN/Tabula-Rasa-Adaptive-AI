"""
Main Training Script - Phase 1 implementation.

This script brings together all components to train the adaptive learning agent
in the survival environment.
"""

import torch
import torch.nn as nn
import yaml
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np

from core.agent import AdaptiveLearningAgent
from environment.survival_environment import SurvivalEnvironment
from utils.config_loader import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for Phase 1 training."""
    return {
        'training': {
            'max_episodes': 100,
            'max_steps_per_episode': 10000,
            'checkpoint_interval': 10,
            'evaluation_interval': 5,
            'save_dir': './checkpoints'
        },
        'agent': {
            'device': 'cpu',
            'learning_rate': 0.001
        },
        'predictive_core': {
            'visual_size': [3, 64, 64],
            'proprioception_size': 12,
            'hidden_size': 512,
            'architecture': 'lstm'
        },
        'memory': {
            'enabled': True,
            'input_size': 512,
            'hidden_size': 256,
            'num_read_heads': 4,
            'num_write_heads': 1,
            'memory_size': 512,
            'word_size': 64
        },
        'learning_progress': {
            'smoothing_window': 500,
            'derivative_clamp': [-1.0, 1.0],
            'boredom_threshold': 0.01,
            'boredom_steps': 500,
            'lp_weight': 0.7,
            'empowerment_weight': 0.3,
            'use_adaptive_weights': False
        },
        'energy': {
            'max_energy': 100.0,
            'base_consumption': 0.01,
            'action_multiplier': 0.5,
            'computation_multiplier': 0.001,
            'food_energy_value': 10.0,
            'memory_size': 512,
            'word_size': 64,
            'use_learned_importance': False,
            'preservation_ratio': 0.2
        },
        'goals': {
            'initial_phase': 'survival',
            'environment_bounds': [-10, 10, -10, 10]
        },
        'sleep': {
            'sleep_trigger_energy': 20.0,
            'sleep_trigger_boredom_steps': 1000,
            'sleep_trigger_memory_pressure': 0.9,
            'sleep_duration_steps': 100,
            'replay_batch_size': 32,
            'learning_rate': 0.001
        },
        'environment': {
            'world_size': [20, 20, 5],
            'num_food_sources': 5,
            'food_respawn_time': 30.0,
            'food_energy_value': 10.0,
            'complexity_level': 1,
            'physics_enabled': True
        },
        'monitoring': {
            'log_interval': 100,
            'save_interval': 1000,
            'log_dir': './logs'
        }
    }


def train_agent(
    config: Dict[str, Any],
    checkpoint_path: str = None
) -> AdaptiveLearningAgent:
    """
    Train the adaptive learning agent.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to checkpoint to resume from
        
    Returns:
        trained_agent: The trained agent
    """
    logger.info("Starting Phase 1 training")
    
    # Create environment
    env_config = config['environment']
    environment = SurvivalEnvironment(
        world_size=tuple(env_config['world_size']),
        num_food_sources=env_config['num_food_sources'],
        food_respawn_time=env_config['food_respawn_time'],
        food_energy_value=env_config['food_energy_value'],
        complexity_level=env_config['complexity_level'],
        physics_enabled=env_config['physics_enabled']
    )
    
    # Create agent
    agent = AdaptiveLearningAgent(config, device=config['agent']['device'])
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        agent.load_checkpoint(checkpoint_path)
        
    # Training loop
    training_config = config['training']
    max_episodes = training_config['max_episodes']
    max_steps = training_config['max_steps_per_episode']
    checkpoint_interval = training_config['checkpoint_interval']
    evaluation_interval = training_config['evaluation_interval']
    save_dir = Path(training_config['save_dir'])
    save_dir.mkdir(exist_ok=True)
    
    # Training metrics
    episode_rewards = []
    episode_survival_times = []
    episode_goals_achieved = []
    
    for episode in range(max_episodes):
        logger.info(f"Starting episode {episode + 1}/{max_episodes}")
        
        # Reset environment and agent
        environment.reset()
        agent.reset_episode()
        
        episode_reward = 0.0
        episode_steps = 0
        goals_achieved = 0
        
        # Episode loop
        for step in range(max_steps):
            # Get current agent state
            agent_state = agent.get_agent_state()
            
            # Generate simple action (random for now - would be RL policy in later phases)
            action = generate_random_action()
            
            # Environment step
            sensory_input, action_result, done = environment.step(agent, action)
            
            # Agent step
            step_results = agent.step(sensory_input, action)
            
            # Update episode metrics
            episode_reward += step_results['learning_progress']
            episode_steps += 1
            
            # Check goal achievement
            if action_result.get('food_collected', False):
                goals_achieved += 1
                
            # Log progress
            if step % 100 == 0:
                logger.debug(f"Episode {episode + 1}, Step {step}: "
                           f"LP={step_results['learning_progress']:.4f}, "
                           f"Energy={agent_state.energy:.1f}")
                
            # Check if episode is done
            if done or agent_state.energy <= 0:
                break
                
        # Episode completed
        episode_rewards.append(episode_reward)
        episode_survival_times.append(episode_steps)
        episode_goals_achieved.append(goals_achieved)
        
        # Log episode results
        logger.info(f"Episode {episode + 1} completed: "
                   f"Steps={episode_steps}, "
                   f"Reward={episode_reward:.4f}, "
                   f"Goals={goals_achieved}, "
                   f"Final_Energy={agent_state.energy:.1f}")
        
        # Get performance metrics
        metrics = agent.get_performance_metrics()
        logger.info(f"Survival rate: {metrics['survival_rate']:.2%}")
        
        # Save checkpoint
        if (episode + 1) % checkpoint_interval == 0:
            checkpoint_path = save_dir / f"agent_episode_{episode + 1}.pt"
            agent.save_checkpoint(str(checkpoint_path))
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        # Evaluate performance
        if (episode + 1) % evaluation_interval == 0:
            evaluate_agent(agent, environment, episode + 1)
            
        # Increase environment complexity if agent is performing well
        if should_increase_complexity(episode_rewards, episode_survival_times):
            environment.increase_complexity()
            logger.info("Environment complexity increased")
            
    # Training completed
    logger.info("Training completed")
    
    # Save final checkpoint
    final_checkpoint_path = save_dir / "agent_final.pt"
    agent.save_checkpoint(str(final_checkpoint_path))
    
    # Print final statistics
    print_training_summary(episode_rewards, episode_survival_times, episode_goals_achieved)
    
    return agent


def generate_random_action() -> torch.Tensor:
    """Generate random action for exploration."""
    # Simple random movement in 3D space
    action = torch.randn(3) * 0.5  # Small random movements
    return action


def evaluate_agent(
    agent: AdaptiveLearningAgent,
    environment: SurvivalEnvironment,
    episode: int
):
    """Evaluate agent performance."""
    logger.info(f"Evaluating agent at episode {episode}")
    
    # Get performance metrics
    metrics = agent.get_performance_metrics()
    
    # Log key metrics
    logger.info(f"Evaluation Results:")
    logger.info(f"  Learning Progress Avg: {metrics['learning_progress_avg']:.4f}")
    logger.info(f"  Survival Rate: {metrics['survival_rate']:.2%}")
    logger.info(f"  Goals Achieved: {metrics['goals_achieved']}")
    logger.info(f"  Deaths: {metrics['deaths']}")
    
    # Check if agent is ready for next phase
    if metrics['survival_rate'] > 0.8 and episode > 20:
        logger.info("Agent may be ready for Phase 2 (template goals)")
        
    # Check for potential issues
    if metrics['learning_progress_avg'] < 0.001:
        logger.warning("Low learning progress - potential learning issues")
        
    if metrics['deaths'] > episode * 0.5:
        logger.warning("High death rate - energy system may need tuning")


def should_increase_complexity(
    episode_rewards: list,
    episode_survival_times: list
) -> bool:
    """Determine if environment complexity should increase."""
    if len(episode_rewards) < 10:
        return False
        
    # Check if agent is consistently performing well
    recent_rewards = episode_rewards[-10:]
    recent_survival = episode_survival_times[-10:]
    
    avg_reward = np.mean(recent_rewards)
    avg_survival = np.mean(recent_survival)
    
    # Increase complexity if agent is stable and learning
    return (avg_reward > 0.01 and  # Positive learning progress
            avg_survival > 5000 and  # Good survival time
            np.std(recent_rewards) < 0.01)  # Stable performance


def print_training_summary(
    episode_rewards: list,
    episode_survival_times: list,
    episode_goals_achieved: list
):
    """Print training summary statistics."""
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    
    if not episode_rewards:
        print("No episodes completed")
        return
        
    print(f"Total Episodes: {len(episode_rewards)}")
    print(f"Average Reward: {np.mean(episode_rewards):.4f}")
    print(f"Average Survival Time: {np.mean(episode_survival_times):.1f} steps")
    print(f"Average Goals Achieved: {np.mean(episode_goals_achieved):.1f}")
    print(f"Best Episode Reward: {np.max(episode_rewards):.4f}")
    print(f"Worst Episode Reward: {np.min(episode_rewards):.4f}")
    
    # Learning progress analysis
    if len(episode_rewards) > 10:
        early_rewards = episode_rewards[:10]
        late_rewards = episode_rewards[-10:]
        improvement = np.mean(late_rewards) - np.mean(early_rewards)
        print(f"Learning Improvement: {improvement:.4f}")
        
        if improvement > 0:
            print("✓ Agent shows learning progress")
        else:
            print("⚠ Agent may not be learning effectively")
            
    print("="*50)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Adaptive Learning Agent")
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of training episodes'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
        
    # Override episode count if specified
    if args.episodes:
        config['training']['max_episodes'] = args.episodes
        
    # Start training
    try:
        agent = train_agent(config, args.checkpoint)
        logger.info("Training completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 