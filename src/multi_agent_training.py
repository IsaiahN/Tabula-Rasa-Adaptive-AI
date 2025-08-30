"""
Multi-Agent Training Script - Phase 3 implementation.

This script trains multiple adaptive learning agents in a shared environment
where they can compete for resources and develop social behaviors.
"""

import torch
import numpy as np
import yaml
import logging
import time
import os
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import matplotlib.pyplot as plt

from core.agent import AdaptiveLearningAgent
from environment.multi_agent_environment import MultiAgentSurvivalEnvironment
from core.data_models import SensoryInput, AgentState
from utils.monitoring import MetricsLogger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiAgentTrainer:
    """Manages training of multiple agents in shared environment."""
    
    def __init__(self, config_path: str = "configs/base_config.yaml"):
        """Initialize multi-agent trainer."""
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config.get('device', 'cpu'))
        
        # Training parameters
        self.num_agents = self.config.get('multi_agent', {}).get('num_agents', 2)
        self.max_episodes = self.config.get('multi_agent', {}).get('max_episodes', 100)
        self.max_steps_per_episode = self.config.get('multi_agent', {}).get('max_steps_per_episode', 1000)
        self.evaluation_interval = self.config.get('multi_agent', {}).get('evaluation_interval', 10)
        
        # Initialize environment
        self.environment = MultiAgentSurvivalEnvironment(
            num_agents=self.num_agents,
            world_size=self.config.get('environment', {}).get('world_size', (20, 20, 5)),
            num_food_sources=self.config.get('environment', {}).get('num_food_sources', 3),
            food_respawn_time=self.config.get('environment', {}).get('food_respawn_time', 60.0),
            complexity_level=1
        )
        
        # Initialize agents
        self.agents = self._create_agents()
        
        # Metrics and monitoring
        self.metrics_logger = MetricsLogger()
        self.episode_metrics = []
        self.social_dynamics_history = []
        
        logger.info(f"Multi-agent trainer initialized: {self.num_agents} agents")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for multi-agent training."""
        return {
            'device': 'cpu',
            'multi_agent': {
                'num_agents': 2,
                'max_episodes': 100,
                'max_steps_per_episode': 1000,
                'evaluation_interval': 10
            },
            'environment': {
                'world_size': (20, 20, 5),
                'num_food_sources': 3,
                'food_respawn_time': 60.0
            },
            'agent': {
                'state_dim': 64,
                'action_dim': 6,
                'memory_size': 128,
                'learning_rate': 0.001,
                'initial_energy': 100.0,
                'max_energy': 150.0
            }
        }
        
    def _create_agents(self) -> Dict[str, AdaptiveLearningAgent]:
        """Create and initialize all agents."""
        agents = {}
        
        for i in range(self.num_agents):
            agent_id = f"agent_{i}"
            
            # Create agent with unique configuration
            agent_config = self.config.get('agent', {}).copy()
            agent_config['agent_id'] = agent_id
            
            agent = AdaptiveLearningAgent(
                state_dim=agent_config.get('state_dim', 64),
                action_dim=agent_config.get('action_dim', 6),
                memory_size=agent_config.get('memory_size', 128),
                learning_rate=agent_config.get('learning_rate', 0.001),
                device=self.device
            )
            
            # Initialize agent state
            initial_position = self.environment.agent_positions[agent_id]
            agent.reset(
                initial_energy=agent_config.get('initial_energy', 100.0),
                max_energy=agent_config.get('max_energy', 150.0),
                position=initial_position
            )
            
            # Set goal system to emergent phase for Phase 3 testing
            agent.goal_system.transition_to_phase("emergent")
            
            agents[agent_id] = agent
            
        logger.info(f"Created {len(agents)} agents with emergent goal systems")
        return agents
        
    def train(self):
        """Main training loop for multi-agent system."""
        logger.info("Starting multi-agent training...")
        
        for episode in range(self.max_episodes):
            episode_start_time = time.time()
            
            # Reset environment and agents
            self.environment.reset()
            for agent in self.agents.values():
                agent.reset()
                
            # Episode metrics
            episode_metrics = {
                'episode': episode,
                'agent_metrics': {agent_id: {} for agent_id in self.agents.keys()},
                'environment_metrics': {},
                'social_dynamics': {},
                'emergent_goals': {agent_id: [] for agent_id in self.agents.keys()}
            }
            
            # Run episode
            for step in range(self.max_steps_per_episode):
                step_start_time = time.time()
                
                # Generate actions for all agents
                actions_dict = {}
                for agent_id, agent in self.agents.items():
                    # Get current observation (simplified for now)
                    agent_state = agent.get_agent_state()
                    sensory_input = self._create_sensory_input(agent_id, agent_state)
                    
                    # Generate action
                    action = self._generate_action(agent, sensory_input)
                    actions_dict[agent_id] = action
                    
                # Execute environment step
                observations, action_results, done = self.environment.step(
                    self.agents, actions_dict
                )
                
                # Update agents with results
                for agent_id, agent in self.agents.items():
                    if agent_id in observations:
                        obs = observations[agent_id]
                        result = action_results[agent_id]
                        
                        # Agent step with multi-agent sensory input
                        agent_result = agent.step(obs.sensory_input, actions_dict[agent_id])
                        
                        # Update episode metrics
                        self._update_agent_metrics(
                            episode_metrics['agent_metrics'][agent_id],
                            agent_result, result, step
                        )
                        
                        # Track emergent goals
                        if agent.goal_system.current_phase.value == "emergent":
                            emergent_goals = agent.goal_system.emergent_goals.get_active_goals()
                            episode_metrics['emergent_goals'][agent_id] = [
                                {
                                    'goal_id': goal.goal_id,
                                    'description': goal.description,
                                    'achievement_count': goal.achievement_count,
                                    'spatial_centroid': goal.spatial_centroid.tolist() if hasattr(goal, 'spatial_centroid') else None
                                }
                                for goal in emergent_goals
                            ]
                
                # Check if episode should end
                if done:
                    logger.info(f"Episode {episode} ended at step {step}")
                    break
                    
                # Log step timing
                step_time = time.time() - step_start_time
                if step % 100 == 0:
                    logger.info(f"Episode {episode}, Step {step}, Time: {step_time:.3f}s")
                    
            # Finalize episode metrics
            episode_metrics['environment_metrics'] = self.environment.get_environment_metrics()
            episode_metrics['social_dynamics'] = self._analyze_social_dynamics()
            episode_metrics['episode_duration'] = time.time() - episode_start_time
            episode_metrics['total_steps'] = step + 1
            
            self.episode_metrics.append(episode_metrics)
            
            # Log episode summary
            self._log_episode_summary(episode, episode_metrics)
            
            # Evaluation and checkpointing
            if episode % self.evaluation_interval == 0:
                self._evaluate_agents(episode)
                self._save_checkpoint(episode)
                
            # Adaptive complexity scaling
            if episode > 0 and episode % 20 == 0:
                self._scale_complexity()
                
        logger.info("Multi-agent training completed!")
        self._generate_final_report()
        
    def _create_sensory_input(self, agent_id: str, agent_state: AgentState) -> SensoryInput:
        """Create sensory input for an agent based on current state."""
        # Get agent position
        position = self.environment.agent_positions[agent_id]
        
        # Create basic visual input (simplified)
        visual_input = torch.randn(3, 64, 64)  # RGB image
        
        # Proprioceptive input
        proprioception = torch.cat([
            position,  # 3D position
            torch.tensor([agent_state.energy / 100.0]),  # Normalized energy
            torch.tensor([float(agent_state.is_alive)]),  # Alive status
        ])
        
        return SensoryInput(
            visual=visual_input,
            proprioception=proprioception,
            timestamp=time.time()
        )
        
    def _generate_action(self, agent: AdaptiveLearningAgent, sensory_input: SensoryInput) -> torch.Tensor:
        """Generate action for an agent."""
        # For now, use random actions with some structure
        # In a full implementation, this would use the agent's action selection
        action = torch.randn(6)  # 6D action space
        
        # Normalize movement actions
        action[:3] = torch.tanh(action[:3])  # Movement in x, y, z
        action[3:] = torch.sigmoid(action[3:])  # Other actions (0-1 range)
        
        return action
        
    def _update_agent_metrics(
        self, 
        agent_metrics: Dict[str, Any], 
        agent_result: Dict[str, Any], 
        env_result: Dict[str, Any],
        step: int
    ):
        """Update metrics for a specific agent."""
        if 'steps_alive' not in agent_metrics:
            agent_metrics.update({
                'steps_alive': 0,
                'total_energy_consumed': 0.0,
                'learning_progress_sum': 0.0,
                'food_collected': 0,
                'social_interactions': 0,
                'emergent_goals_discovered': 0,
                'goal_achievements': 0
            })
            
        # Update basic metrics
        agent_metrics['steps_alive'] = step + 1
        agent_metrics['total_energy_consumed'] += agent_result.get('energy_consumed', 0.0)
        agent_metrics['learning_progress_sum'] += agent_result.get('learning_progress', 0.0)
        
        # Update environment interaction metrics
        if env_result.get('food_collected', False):
            agent_metrics['food_collected'] += 1
            
        if 'social_interaction' in env_result:
            agent_metrics['social_interactions'] += len(
                env_result['social_interaction'].get('interacted_with', [])
            )
            
        # Update goal metrics
        if 'goals_achieved' in agent_result:
            agent_metrics['goal_achievements'] += len(agent_result['goals_achieved'])
            
    def _analyze_social_dynamics(self) -> Dict[str, Any]:
        """Analyze social dynamics between agents."""
        env_metrics = self.environment.get_environment_metrics()
        
        # Calculate cooperation vs competition ratios
        total_interactions = env_metrics['social_interactions']
        competition_events = env_metrics['resource_competition_events']
        cooperation_events = env_metrics['cooperation_events']
        
        if total_interactions > 0:
            competition_ratio = competition_events / total_interactions
            cooperation_ratio = cooperation_events / total_interactions
        else:
            competition_ratio = 0.0
            cooperation_ratio = 0.0
            
        return {
            'total_social_interactions': total_interactions,
            'competition_ratio': competition_ratio,
            'cooperation_ratio': cooperation_ratio,
            'resource_competition_events': competition_events,
            'cooperation_events': cooperation_events,
            'dominant_behavior': 'competitive' if competition_ratio > cooperation_ratio else 'cooperative'
        }
        
    def _log_episode_summary(self, episode: int, metrics: Dict[str, Any]):
        """Log summary of episode results."""
        env_metrics = metrics['environment_metrics']
        social_metrics = metrics['social_dynamics']
        
        logger.info(f"\n=== Episode {episode} Summary ===")
        logger.info(f"Duration: {metrics['episode_duration']:.2f}s, Steps: {metrics['total_steps']}")
        logger.info(f"Social interactions: {social_metrics['total_social_interactions']}")
        logger.info(f"Resource competition: {env_metrics['resource_competition_events']}")
        logger.info(f"Dominant behavior: {social_metrics['dominant_behavior']}")
        
        # Agent-specific metrics
        for agent_id, agent_metrics in metrics['agent_metrics'].items():
            logger.info(f"{agent_id}: Alive {agent_metrics.get('steps_alive', 0)} steps, "
                       f"Food: {agent_metrics.get('food_collected', 0)}, "
                       f"Social: {agent_metrics.get('social_interactions', 0)}")
            
        # Emergent goals summary
        total_emergent_goals = sum(
            len(goals) for goals in metrics['emergent_goals'].values()
        )
        logger.info(f"Total emergent goals discovered: {total_emergent_goals}")
        
    def _evaluate_agents(self, episode: int):
        """Evaluate agent performance and learning progress."""
        logger.info(f"\n=== Evaluation at Episode {episode} ===")
        
        # Calculate performance metrics
        recent_episodes = self.episode_metrics[-self.evaluation_interval:]
        
        avg_survival_time = np.mean([
            ep['total_steps'] for ep in recent_episodes
        ])
        
        avg_social_interactions = np.mean([
            ep['social_dynamics']['total_social_interactions'] for ep in recent_episodes
        ])
        
        total_emergent_goals = sum([
            sum(len(goals) for goals in ep['emergent_goals'].values())
            for ep in recent_episodes
        ])
        
        logger.info(f"Average survival time: {avg_survival_time:.1f} steps")
        logger.info(f"Average social interactions per episode: {avg_social_interactions:.1f}")
        logger.info(f"Total emergent goals in last {self.evaluation_interval} episodes: {total_emergent_goals}")
        
        # Agent-specific evaluation
        for agent_id in self.agents.keys():
            agent_performance = self._evaluate_single_agent(agent_id, recent_episodes)
            logger.info(f"{agent_id} performance: {agent_performance}")
            
    def _evaluate_single_agent(self, agent_id: str, recent_episodes: List[Dict]) -> Dict[str, float]:
        """Evaluate performance of a single agent."""
        metrics = {
            'avg_survival_time': 0.0,
            'avg_food_collected': 0.0,
            'avg_social_interactions': 0.0,
            'avg_learning_progress': 0.0,
            'emergent_goals_discovered': 0
        }
        
        for episode in recent_episodes:
            agent_metrics = episode['agent_metrics'].get(agent_id, {})
            metrics['avg_survival_time'] += agent_metrics.get('steps_alive', 0)
            metrics['avg_food_collected'] += agent_metrics.get('food_collected', 0)
            metrics['avg_social_interactions'] += agent_metrics.get('social_interactions', 0)
            metrics['avg_learning_progress'] += agent_metrics.get('learning_progress_sum', 0)
            metrics['emergent_goals_discovered'] += len(episode['emergent_goals'].get(agent_id, []))
            
        # Average over episodes
        num_episodes = len(recent_episodes)
        for key in metrics:
            if key != 'emergent_goals_discovered':
                metrics[key] /= num_episodes
                
        return metrics
        
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint_dir = "checkpoints/multi_agent"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'agents': {},
            'environment_state': self.environment.get_environment_metrics(),
            'training_metrics': self.episode_metrics[-10:],  # Last 10 episodes
            'config': self.config
        }
        
        # Save agent states
        for agent_id, agent in self.agents.items():
            checkpoint['agents'][agent_id] = {
                'state_dict': agent.predictive_core.state_dict(),
                'agent_state': asdict(agent.get_agent_state()),
                'goal_system_phase': agent.goal_system.current_phase.value
            }
            
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode}.pt")
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
    def _scale_complexity(self):
        """Increase environment complexity for curriculum learning."""
        self.environment.increase_complexity()
        logger.info(f"Environment complexity increased to level {self.environment.complexity_level}")
        
    def _generate_final_report(self):
        """Generate comprehensive final training report."""
        logger.info("\n" + "="*50)
        logger.info("MULTI-AGENT TRAINING FINAL REPORT")
        logger.info("="*50)
        
        # Overall statistics
        total_episodes = len(self.episode_metrics)
        total_steps = sum(ep['total_steps'] for ep in self.episode_metrics)
        total_duration = sum(ep['episode_duration'] for ep in self.episode_metrics)
        
        logger.info(f"Total episodes: {total_episodes}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Total training time: {total_duration:.2f}s ({total_duration/3600:.2f}h)")
        
        # Social dynamics summary
        total_social_interactions = sum(
            ep['social_dynamics']['total_social_interactions'] for ep in self.episode_metrics
        )
        total_competition_events = sum(
            ep['environment_metrics']['resource_competition_events'] for ep in self.episode_metrics
        )
        
        logger.info(f"\nSocial Dynamics:")
        logger.info(f"Total social interactions: {total_social_interactions}")
        logger.info(f"Total competition events: {total_competition_events}")
        
        # Emergent goals summary
        total_emergent_goals = sum(
            sum(len(goals) for goals in ep['emergent_goals'].values())
            for ep in self.episode_metrics
        )
        
        logger.info(f"\nEmergent Goals:")
        logger.info(f"Total emergent goals discovered: {total_emergent_goals}")
        
        # Agent performance summary
        logger.info(f"\nAgent Performance Summary:")
        for agent_id in self.agents.keys():
            performance = self._evaluate_single_agent(agent_id, self.episode_metrics)
            logger.info(f"{agent_id}:")
            for metric, value in performance.items():
                logger.info(f"  {metric}: {value:.3f}")
                
        logger.info("\nTraining completed successfully!")


def main():
    """Main function to run multi-agent training."""
    # Create trainer
    trainer = MultiAgentTrainer()
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    main()
