"""
Adaptive Learning ARC Agent

Integrates the Adaptive Learning Agent with ARC-AGI-3 testing framework.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)

# ARC-AGI-3 imports (assuming they're available)
try:
    from agents.agent import Agent
    from agents.structs import FrameData, GameAction, GameState
except ImportError:
    # Fallback for development
    class Agent:
        pass
    class FrameData:
        pass
    class GameAction:
        pass
    class GameState:
        pass

class AdaptiveLearningARCAgent(Agent):
    """
    Enhanced Adaptive Learning Agent for ARC-AGI-3 integration.
    
    This agent combines the Adaptive Learning Agent with ARC-specific
    visual processing and action mapping capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Adaptive Learning ARC Agent."""
        super().__init__()
        
        self.config = config or {}
        self.game_id = self.config.get('game_id', 'unknown')
        
        # Initialize components
        self.visual_processor = self._initialize_visual_processor()
        self.action_mapper = self._initialize_action_mapper()
        self.learning_system = self._initialize_learning_system()
        
        # State tracking
        self.current_state = None
        self.action_history = []
        self.learning_history = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_actions': 0,
            'successful_actions': 0,
            'learning_episodes': 0,
            'average_reward': 0.0
        }
    
    def _initialize_visual_processor(self):
        """Initialize visual processor."""
        try:
            from ..visual_processing import ARCVisualProcessor
            return ARCVisualProcessor()
        except ImportError as e:
            logger.warning(f"Could not import visual processor: {e}")
            return None
    
    def _initialize_action_mapper(self):
        """Initialize action mapper."""
        try:
            from ..action_mapping import ARCActionMapper
            return ARCActionMapper(self.game_id)
        except ImportError as e:
            logger.warning(f"Could not import action mapper: {e}")
            return None
    
    def _initialize_learning_system(self):
        """Initialize learning system."""
        try:
            # This would integrate with the actual learning system
            # For now, return a simple placeholder
            return SimpleLearningSystem()
        except Exception as e:
            logger.warning(f"Could not initialize learning system: {e}")
            return None
    
    def act(self, frame_data: FrameData) -> GameAction:
        """Main action selection method."""
        try:
            # Process visual input
            visual_features = self._process_visual_input(frame_data)
            
            # Get current state
            current_state = self._get_current_state(frame_data, visual_features)
            
            # Select action using learning system
            action = self._select_action(current_state)
            
            # Record action
            self._record_action(action, current_state)
            
            # Update performance metrics
            self.performance_metrics['total_actions'] += 1
            
            return action
            
        except Exception as e:
            logger.error(f"Error in act method: {e}")
            return GameAction.RESET
    
    def _process_visual_input(self, frame_data: FrameData) -> Dict[str, Any]:
        """Process visual input from frame data."""
        try:
            if not self.visual_processor:
                return {'error': 'Visual processor not available'}
            
            # Extract grid data
            grid_data = self._extract_grid_data(frame_data)
            
            # Process with visual processor
            visual_tensor = self.visual_processor.forward(grid_data)
            features = self.visual_processor.get_feature_maps(grid_data)
            
            return {
                'visual_tensor': visual_tensor,
                'features': features,
                'grid_data': grid_data
            }
            
        except Exception as e:
            logger.error(f"Error processing visual input: {e}")
            return {'error': str(e)}
    
    def _extract_grid_data(self, frame_data: FrameData) -> List[List[int]]:
        """Extract grid data from frame data."""
        try:
            # This would extract the actual grid data from frame_data
            # For now, return a placeholder
            if hasattr(frame_data, 'grid'):
                return frame_data.grid
            elif hasattr(frame_data, 'input'):
                return frame_data.input
            else:
                # Return empty grid as fallback
                return [[0] * 64 for _ in range(64)]
                
        except Exception as e:
            logger.error(f"Error extracting grid data: {e}")
            return [[0] * 64 for _ in range(64)]
    
    def _get_current_state(self, frame_data: FrameData, visual_features: Dict[str, Any]) -> Dict[str, Any]:
        """Get current state representation."""
        try:
            state = {
                'game_id': self.game_id,
                'frame_data': frame_data,
                'visual_features': visual_features,
                'timestamp': time.time(),
                'action_count': len(self.action_history)
            }
            
            # Add learning context if available
            if self.learning_system:
                state['learning_context'] = self.learning_system.get_context()
            
            return state
            
        except Exception as e:
            logger.error(f"Error getting current state: {e}")
            return {'error': str(e)}
    
    def _select_action(self, state: Dict[str, Any]) -> GameAction:
        """Select action using learning system."""
        try:
            if not self.learning_system:
                # Fallback to random action selection
                return self._select_random_action()
            
            # Use learning system to select action
            action = self.learning_system.select_action(state)
            
            # Convert to GameAction if needed
            if isinstance(action, str):
                return self._string_to_game_action(action)
            elif isinstance(action, GameAction):
                return action
            else:
                logger.warning(f"Unknown action type: {type(action)}")
                return GameAction.RESET
                
        except Exception as e:
            logger.error(f"Error selecting action: {e}")
            return GameAction.RESET
    
    def _select_random_action(self) -> GameAction:
        """Select random action as fallback."""
        try:
            actions = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, 
                      GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6]
            return np.random.choice(actions)
        except Exception as e:
            logger.error(f"Error selecting random action: {e}")
            return GameAction.RESET
    
    def _string_to_game_action(self, action_str: str) -> GameAction:
        """Convert string to GameAction."""
        try:
            action_map = {
                'RESET': GameAction.RESET,
                'ACTION1': GameAction.ACTION1,
                'ACTION2': GameAction.ACTION2,
                'ACTION3': GameAction.ACTION3,
                'ACTION4': GameAction.ACTION4,
                'ACTION5': GameAction.ACTION5,
                'ACTION6': GameAction.ACTION6,
                'ACTION7': GameAction.ACTION7
            }
            return action_map.get(action_str, GameAction.RESET)
        except Exception as e:
            logger.error(f"Error converting string to GameAction: {e}")
            return GameAction.RESET
    
    def _record_action(self, action: GameAction, state: Dict[str, Any]):
        """Record action for learning."""
        try:
            action_record = {
                'action': action,
                'state': state,
                'timestamp': time.time()
            }
            
            self.action_history.append(action_record)
            
            # Keep only recent history
            if len(self.action_history) > 1000:
                self.action_history = self.action_history[-500:]
                
        except Exception as e:
            logger.error(f"Error recording action: {e}")
    
    def update_reward(self, reward: float, done: bool = False):
        """Update agent with reward signal."""
        try:
            if self.learning_system:
                self.learning_system.update_reward(reward, done)
            
            # Update performance metrics
            if reward > 0:
                self.performance_metrics['successful_actions'] += 1
            
            # Update average reward
            total_rewards = sum(record.get('reward', 0) for record in self.learning_history)
            self.performance_metrics['average_reward'] = total_rewards / max(len(self.learning_history), 1)
            
            # Record learning episode
            if done:
                self.performance_metrics['learning_episodes'] += 1
                self.learning_history.append({
                    'episode': self.performance_metrics['learning_episodes'],
                    'total_reward': reward,
                    'timestamp': time.time()
                })
                
        except Exception as e:
            logger.error(f"Error updating reward: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    def save_model(self, path: str):
        """Save agent model."""
        try:
            model_data = {
                'config': self.config,
                'performance_metrics': self.performance_metrics,
                'action_history': self.action_history[-100:],  # Keep only recent history
                'learning_history': self.learning_history[-50:]  # Keep only recent history
            }
            
            with open(path, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, path: str):
        """Load agent model."""
        try:
            with open(path, 'r') as f:
                model_data = json.load(f)
            
            self.config = model_data.get('config', {})
            self.performance_metrics = model_data.get('performance_metrics', self.performance_metrics)
            self.action_history = model_data.get('action_history', [])
            self.learning_history = model_data.get('learning_history', [])
            
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")

class SimpleLearningSystem:
    """Simple learning system placeholder."""
    
    def __init__(self):
        self.context = {}
        self.reward_history = []
    
    def select_action(self, state: Dict[str, Any]) -> str:
        """Select action based on state."""
        # Simple random selection for now
        actions = ['ACTION1', 'ACTION2', 'ACTION3', 'ACTION4', 'ACTION5', 'ACTION6']
        return np.random.choice(actions)
    
    def update_reward(self, reward: float, done: bool = False):
        """Update with reward signal."""
        self.reward_history.append(reward)
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-500:]
    
    def get_context(self) -> Dict[str, Any]:
        """Get learning context."""
        return {
            'reward_history_length': len(self.reward_history),
            'average_reward': np.mean(self.reward_history) if self.reward_history else 0.0
        }
