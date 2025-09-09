"""
Continuous Learning Loop for ARC-AGI-3 Training

This module implements a continuous learning system that runs the Adaptive Learning Agent
against ARC-AGI-3 tasks, collecting insights and improving performance over time.
"""
import asyncio
import aiohttp
import json
import logging
import os
import random
import sys
import time
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

class ContinuousLearningLoop:
    """Manages continuous learning sessions for the Adaptive Learning Agent on ARC tasks."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the ContinuousLearningLoop with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.session = None
        self.current_episode = 0
        self.total_episodes = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_score = -float('inf')
        self.start_time = time.time()
        self.last_save_time = time.time()
        self.save_interval = self.config.get('save_interval', 600)  # Default: 10 minutes
        
        # Initialize exploration parameters
        self.exploration_rate = self.config.get('initial_exploration', 1.0)
        self.exploration_min = self.config.get('min_exploration', 0.01)
        self.exploration_decay = self.config.get('exploration_decay', 0.995)
        
        # Initialize game complexity tracking
        self.game_complexity = {
            'low': set(),
            'medium': set(),
            'high': set()
        }
        
        # Initialize action tracking
        self.last_actions = []
        self.action_history = []
        self.replay_buffer = []
        self.q_table = {}
        
        # Initialize energy system
        self.current_energy = 100.0  # Full energy at start
        self.energy_consumption_rate = 0.1
        self.energy_recovery_rate = 0.5
        
        # Initialize other components
        self.memory = None
        self.agent = None
        self.environment = None
        
        # Initialize learning parameters
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 32)
        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.01)
        self.update_every = self.config.get('update_every', 4)
        self.buffer_size = int(self.config.get('buffer_size', 1e5))
        self.seed = self.config.get('seed', 0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize trackers
        self.episode_timesteps = 0
        self.total_timesteps = 0
        self.episode_reward = 0
        self.episode_loss = 0
        self.episode_q_values = []
        self.episode_entropy = []
        
        # Initialize action tracking
        self.action_counts = {i: 0 for i in range(10)}  # Assuming 10 possible actions
        self.action_rewards = {i: [] for i in range(10)}
        
        # Initialize performance metrics
        self.performance_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'mean_rewards': [],
            'best_mean_reward': -float('inf'),
            'total_steps': 0,
            'start_time': time.time(),
            'last_save': time.time()
        }
        
        # Initialize API client
        self.api_key = os.environ.get('ARC_API_KEY')
        if not self.api_key:
            self.logger.warning("No ARC_API_KEY found in environment variables. Some features may be limited.")
    
    def _get_state_key(self, observation: Dict[str, Any]) -> str:
        """Convert observation to a state key for Q-table lookup."""
        # Simple state representation - can be enhanced with more sophisticated features
        grid_hash = hash(str(observation.get('grid', [])))
        return f"{grid_hash}"
    
    def _select_action(self, state_key: str, valid_actions: List[int]) -> int:
        """
        Select an action using epsilon-greedy policy with action masking.
        
        Args:
            state_key: String representation of the current state
            valid_actions: List of valid action indices
            
        Returns:
            int: Selected action index
        """
        if not valid_actions:
            raise ValueError("No valid actions provided")
            
        # Ensure exploration rate is within valid range
        exploration_rate = max(0.0, min(1.0, getattr(self, 'exploration_rate', 1.0)))
        
        # Log action selection details for debugging
        logger.debug(f"Action selection - State: {state_key[:30]}... | "
                   f"Exploration: {exploration_rate:.2f} | "
                   f"Valid actions: {valid_actions}")
        
        # Exploration: random valid action
        if random.random() < exploration_rate:
            action = random.choice(valid_actions)
            logger.debug(f"🔍 Exploring: action {action}")
        else:
            # Initialize Q-table for this state if it doesn't exist
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(10)  # Assuming 10 possible actions
                
            # Exploitation: best action from Q-table
            q_values = self.q_table[state_key]
            # Mask invalid actions by setting their Q-values to -inf
            masked_q = [q if i in valid_actions else -float('inf') for i, q in enumerate(q_values)]
            action = np.argmax(masked_q)
            logger.debug(f"🎯 Exploiting: action {action} (Q={q_values[action]:.2f})")
            
        return action
    
    def _calculate_reward(self, state: Dict[str, Any], done: bool, game_state: str) -> float:
        """Calculate shaped reward based on game state and progress."""
        reward = 0.0
        
        # Base reward for continuing
        reward += 0.01  # Small positive reward for each step
        
        # Check for terminal states
        if done:
            if game_state == 'WIN':
                reward += 10.0  # Large reward for winning
                logger.info("🏆 Won the game!")
            elif game_state == 'GAME_OVER':
                reward -= 5.0  # Penalty for losing
                logger.warning("💥 Game over!")
        
        # Penalize action repetition
        if len(self.last_actions) > 1 and len(set(self.last_actions)) == 1:
            reward -= 0.5  # Penalty for repeating the same action
            
        return reward
    
    def _update_q_table(self, state_key: str, action: int, reward: float, 
                       next_state_key: str, done: bool) -> None:
        """Update Q-table using Q-learning update rule."""
        # Initialize Q-table for this state if it doesn't exist
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(10)  # Assuming 10 possible actions
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(10)  # Assuming 10 possible actions
            
        # Q-learning update
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key]) if not done else 0
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
    
    def _update_exploration_rate(self, episode: int) -> None:
        """Decay exploration rate over time."""
        self.exploration_rate = max(
            self.exploration_min, 
            self.exploration_rate * self.exploration_decay
        )
    
    def _update_energy(self) -> None:
        """Update energy level based on recent actions."""
        # Consume energy for taking an action
        self.current_energy = max(0, self.current_energy - self.energy_consumption_rate)
        
        # Recover energy over time (when not taking actions)
        if len(self.last_actions) == 0 or random.random() < 0.1:  # 10% chance to recover energy
            self.current_energy = min(100.0, self.current_energy + self.energy_recovery_rate)
    
    async def _take_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Take an action in the environment based on the current state.
        
        Args:
            state: Current state of the environment
            
        Returns:
            Dict containing action results (reward, game_state, done, etc.)
        """
        try:
            # Initialize last_actions if it doesn't exist
            if not hasattr(self, 'last_actions'):
                self.last_actions = []
                
            # Get valid actions (simplified - in practice, this would come from the environment)
            valid_actions = list(range(10))  # Assuming 10 possible actions
            
            # Get state key for Q-table lookup
            state_key = self._get_state_key(state)
            
            # Select action using epsilon-greedy policy
            action = self._select_action(state_key, valid_actions)
            
            # Track last actions for repetition penalty
            self.last_actions = (self.last_actions + [action])[-5:]  # Keep last 5 actions
            
            # Simulate environment step (in a real implementation, this would call the ARC environment)
            done = random.random() < 0.01  # 1% chance of episode ending
            game_state = 'WIN' if done and random.random() < 0.3 else 'GAME_OVER' if done else 'IN_PROGRESS'
            
            # Calculate shaped reward
            reward = self._calculate_reward(state, done, game_state)
            
            # Get next state (in a real implementation, this would be the new observation)
            next_state_key = self._get_state_key(state)  # Simplified - would be different in practice
            
            # Initialize replay buffer if it doesn't exist
            if not hasattr(self, 'replay_buffer'):
                self.replay_buffer = []
                
            # Store transition in replay buffer
            self.replay_buffer.append((state_key, action, reward, next_state_key, done))
            
            # Update Q-table if it exists
            if hasattr(self, 'q_table'):
                self._update_q_table(state_key, action, reward, next_state_key, done)
            
            # Update exploration rate if needed
            if hasattr(self, 'episode_rewards'):
                self._update_exploration_rate(len(self.episode_rewards))
            
            # Update energy level
            self._update_energy()
            
            return {
                'reward': reward,
                'game_state': game_state,
                'done': done,
                'action': action,
                'next_state': next_state_key
            }
            
        except Exception as e:
            logger.error(f"Error in _take_action: {str(e)}", exc_info=True)
            return {
                'reward': -1.0,
                'game_state': 'ERROR',
                'done': True,
                'error': str(e)
            }
    
    def _get_current_agent_state(self) -> Dict[str, Any]:
        """
        Get the current state of the agent including energy, learning progress, and other metrics.
        
        Returns:
            Dict containing the current agent state
        """
        # Initialize default state
        state = {
            'energy': getattr(self, 'current_energy', 100.0),  # Default to full energy if not set
            'learning_progress': getattr(self, 'learning_progress', 0.0),
            'exploration_rate': getattr(self, 'exploration_rate', 1.0),
            'episode_count': getattr(self, 'current_episode', 0),
            'total_timesteps': getattr(self, 'total_timesteps', 0),
            'consecutive_failures': getattr(self, 'consecutive_failures', 0),
            'last_action': getattr(self, 'last_action', None),
            'last_reward': getattr(self, 'last_reward', 0.0),
            'game_state': getattr(self, 'current_game_state', 'IDLE'),
            'memory_usage': len(getattr(self, 'memory', [])),
            'sleep_cycles': getattr(self, 'sleep_cycles', 0),
            'game_complexity': getattr(self, 'current_game_complexity', 'medium'),
            'available_actions': getattr(self, 'available_actions', list(range(10))),  # Assuming 10 possible actions
            'action_history': getattr(self, 'action_history', []),
            'learning_progress': getattr(self, 'learning_progress', 0.0),
            'consecutive_failures': getattr(self, 'consecutive_failures', 0),
            'consecutive_successes': getattr(self, 'consecutive_successes', 0),
            'total_episodes': getattr(self, 'total_episodes', 0),
            'success_rate': getattr(self, 'success_rate', 0.0),
            'last_action_taken': getattr(self, 'last_action_taken', None),
            'last_action_success': getattr(self, 'last_action_success', None),
            'current_goals': getattr(self, 'current_goals', []),
            'active_learning_strategies': getattr(self, 'active_learning_strategies', [])
        }
        return state
    
    def _estimate_game_complexity(self, game_id: str) -> str:
        """
        Estimate the complexity of a game based on its ID and previous performance.
        
        Args:
            game_id: The ID of the game to estimate complexity for
            
        Returns:
            str: 'low', 'medium', or 'high' complexity
        """
        # Check if we have a cached complexity for this game
        for complexity, games in self.game_complexity.items():
            if game_id in games:
                return complexity
                
        # Default to medium complexity for unknown games
        return 'medium'
    
    async def start_training_with_direct_control(self, game_id: str, max_actions: int, session_id: int) -> Dict[str, Any]:
        """
        Start training with direct control over the agent's actions.
        
        Args:
            game_id: ID of the game to train on
            max_actions: Maximum number of actions to take in this session
            session_id: Unique identifier for this training session
            
        Returns:
            Dict containing training results
        """
        self.logger.info(f"🚀 Starting training session {session_id} for game {game_id}")
        start_time = time.time()
        
        try:
            # Initialize state
            state = {'grid': np.zeros((10, 10))}  # Simplified initial state
            total_reward = 0
            done = False
            action_count = 0
            
            # Run training loop
            while not done and action_count < max_actions:
                # Take action
                action_result = await self._take_action(state)
                
                # Update state and track metrics
                total_reward += action_result.get('reward', 0)
                done = action_result.get('done', False)
                action_count += 1
                
                # Log progress
                if action_count % 100 == 0:
                    self.logger.info(f"Action {action_count}/{max_actions} | "
                                  f"Reward: {total_reward:.2f} | "
                                  f"Exploration: {self.exploration_rate:.3f}")
            
            # Calculate metrics
            duration = time.time() - start_time
            actions_per_second = action_count / duration if duration > 0 else 0
            
            # Return results
            return {
                'success': True,
                'game_id': game_id,
                'session_id': session_id,
                'total_reward': total_reward,
                'action_count': action_count,
                'duration_seconds': duration,
                'actions_per_second': actions_per_second,
                'exploration_rate': self.exploration_rate,
                'energy_level': self.current_energy
            }
            
        except Exception as e:
            self.logger.error(f"Error in training session {session_id}: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'game_id': game_id,
                'session_id': session_id
            }
    
    async def _run_real_arc_mastery_session_enhanced(self, game_id: str, session_count: int) -> Dict[str, Any]:
        """
        Enhanced version that runs COMPLETE mastery sessions with up to 100K actions until WIN/GAME_OVER.
        """
        logger.info(f"🚀 Starting mastery session {session_count} for game {game_id}")
        start_time = time.time()
        
        try:
            # Run training session with direct control
            result = await self.start_training_with_direct_control(
                game_id=game_id,
                max_actions=100000,  # Large number to allow for complete sessions
                session_id=session_count
            )
            
            # Calculate metrics
            duration = time.time() - start_time
            
            # Return results
            return {
                'success': result.get('success', False),
                'game_id': game_id,
                'session_id': session_count,
                'duration_seconds': duration,
                'total_actions': result.get('action_count', 0),
                'total_reward': result.get('total_reward', 0),
                'actions_per_second': result.get('actions_per_second', 0),
                'exploration_rate': result.get('exploration_rate', 1.0),
                'energy_level': result.get('energy_level', 100.0),
                'error': result.get('error')
            }
            
        except Exception as e:
            logger.error(f"Error in mastery session {session_count}: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'game_id': game_id,
                'session_id': session_count
            }
