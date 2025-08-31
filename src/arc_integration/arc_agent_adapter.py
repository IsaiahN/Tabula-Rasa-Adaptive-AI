"""
ARC-AGI-3 Agent Adapter for Adaptive Learning Agent

This module integrates the Adaptive Learning Agent with the ARC-AGI-3 testing framework,
allowing the agent to learn from abstract reasoning tasks and develop better cognitive skills.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import time

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

# Tabula Rasa imports
from core.agent import AdaptiveLearningAgent
from core.data_models import SensoryInput, AgentState
from core.meta_learning import MetaLearningSystem

logger = logging.getLogger(__name__)


class ARCVisualProcessor(nn.Module):
    """
    Processes ARC visual grids into format compatible with Adaptive Learning Agent.
    
    ARC grids are typically small (up to 64x64) with discrete color values (0-9).
    This processor converts them to the expected visual format.
    """
    
    def __init__(self, target_size: Tuple[int, int, int] = (3, 64, 64)):
        super().__init__()
        self.target_size = target_size
        self.color_embedding = nn.Embedding(10, 3)  # 10 colors -> 3 channels
        
    def forward(self, arc_frame: List[List[List[int]]]) -> torch.Tensor:
        """
        Convert ARC frame to visual tensor.
        
        Args:
            arc_frame: ARC frame data [height][width][channels] with values 0-9
            
        Returns:
            visual_tensor: [batch_size, channels, height, width]
        """
        if not arc_frame:
            # Return empty frame
            return torch.zeros(1, *self.target_size)
            
        # Convert to numpy array
        frame_array = np.array(arc_frame)
        
        # Handle different input formats
        if len(frame_array.shape) == 2:
            # Single channel grid
            frame_array = frame_array[:, :, np.newaxis]
        elif len(frame_array.shape) == 3 and frame_array.shape[2] == 1:
            # Already has channel dimension
            pass
        else:
            logger.warning(f"Unexpected frame shape: {frame_array.shape}")
            
        # Convert to tensor
        frame_tensor = torch.tensor(frame_array, dtype=torch.long)
        
        # Embed colors to RGB-like representation
        if frame_tensor.dim() == 3:
            embedded = self.color_embedding(frame_tensor)  # [H, W, C, 3]
            embedded = embedded.mean(dim=2)  # Average across original channels
            embedded = embedded.permute(2, 0, 1)  # [3, H, W]
        else:
            embedded = self.color_embedding(frame_tensor)  # [H, W, 3]
            embedded = embedded.permute(2, 0, 1)  # [3, H, W]
            
        # Resize to target size if needed
        if embedded.shape[1:] != self.target_size[1:]:
            embedded = torch.nn.functional.interpolate(
                embedded.unsqueeze(0), 
                size=self.target_size[1:], 
                mode='nearest'
            ).squeeze(0)
            
        return embedded.unsqueeze(0)  # Add batch dimension


class ARCActionMapper:
    """
    Maps between ARC actions and Adaptive Learning Agent actions.
    """
    
    def __init__(self):
        # Map ARC actions to continuous action space
        self.action_mapping = {
            GameAction.RESET: torch.tensor([0.0, 0.0, 0.0]),
            GameAction.ACTION1: torch.tensor([1.0, 0.0, 0.0]),
            GameAction.ACTION2: torch.tensor([0.0, 1.0, 0.0]),
            GameAction.ACTION3: torch.tensor([0.0, 0.0, 1.0]),
            GameAction.ACTION4: torch.tensor([-1.0, 0.0, 0.0]),
            GameAction.ACTION5: torch.tensor([0.0, -1.0, 0.0]),
            GameAction.ACTION6: torch.tensor([0.0, 0.0, -1.0]),  # Complex action
            GameAction.ACTION7: torch.tensor([0.5, 0.5, 0.0]),
        }
        
    def arc_to_agent_action(self, arc_action: GameAction, x: int = 0, y: int = 0) -> torch.Tensor:
        """Convert ARC action to agent action."""
        base_action = self.action_mapping.get(arc_action, torch.zeros(3))
        
        if arc_action == GameAction.ACTION6:  # Complex action with coordinates
            # Normalize coordinates to [-1, 1] range
            norm_x = (x / 32.0) - 1.0
            norm_y = (y / 32.0) - 1.0
            return torch.tensor([norm_x, norm_y, 0.0])
            
        return base_action
        
    def agent_to_arc_action(self, agent_action: torch.Tensor) -> Tuple[GameAction, Dict[str, int]]:
        """Convert agent action to ARC action."""
        action_vec = agent_action.detach().cpu()
        
        # Find closest action mapping
        best_action = GameAction.RESET
        best_distance = float('inf')
        
        for arc_action, mapped_vec in self.action_mapping.items():
            distance = torch.norm(action_vec - mapped_vec).item()
            if distance < best_distance:
                best_distance = distance
                best_action = arc_action
                
        # Handle complex actions
        action_data = {}
        if best_action == GameAction.ACTION6:
            # Convert back to grid coordinates
            x = int((action_vec[0].item() + 1.0) * 32.0)
            y = int((action_vec[1].item() + 1.0) * 32.0)
            action_data = {"x": max(0, min(63, x)), "y": max(0, min(63, y))}
            
        return best_action, action_data


class AdaptiveLearningARCAgent(Agent):
    """
    ARC-AGI-3 Agent that uses the Adaptive Learning Agent for reasoning.
    
    This adapter allows the sophisticated learning mechanisms of the Adaptive Learning Agent
    to be applied to abstract reasoning tasks in the ARC framework.
    """
    
    MAX_ACTIONS = 20000  # Allow enough actions for complex puzzles (matching leaderboard agents)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize Adaptive Learning Agent
        self.config = self._create_arc_config()
        self.learning_agent = AdaptiveLearningAgent(self.config, device="cpu")
        
        # ARC-specific components
        self.visual_processor = ARCVisualProcessor()
        self.action_mapper = ARCActionMapper()
        
        # Learning and meta-cognition
        self.task_insights = []
        self.performance_history = []
        self.reasoning_patterns = {}
        
        # Episode tracking
        self.current_episode_data = {
            'frames': [],
            'actions': [],
            'reasoning': [],
            'learning_progress': []
        }
        
        logger.info(f"Adaptive Learning ARC Agent initialized for game {self.game_id}")
        
    def _create_arc_config(self) -> Dict[str, Any]:
        """Create configuration optimized for ARC tasks."""
        return {
            'agent': {
                'device': 'cpu',
                'learning_rate': 0.001
            },
            'predictive_core': {
                'visual_size': [3, 64, 64],  # Match ARC grid processing
                'proprioception_size': 8,  # Reduced for ARC context
                'hidden_size': 256,  # Smaller for faster learning
                'architecture': 'lstm'
            },
            'memory': {
                'enabled': True,
                'memory_size': 256,  # Smaller memory for focused learning
                'word_size': 32,
                'num_read_heads': 2,
                'num_write_heads': 1,
                'controller_size': 128
            },
            'learning_progress': {
                'smoothing_window': 50,  # Shorter window for ARC tasks
                'derivative_clamp': [-2.0, 2.0],
                'boredom_threshold': 0.05,
                'boredom_steps': 20,
                'lp_weight': 0.8,
                'empowerment_weight': 0.2,
                'use_adaptive_weights': True
            },
            'energy': {
                'max_energy': 100.0,
                'base_consumption': 0.001,  # Lower consumption for reasoning tasks
                'action_multiplier': 0.1,
                'computation_multiplier': 0.0001,
                'food_energy_value': 5.0,
                'memory_size': 256,
                'word_size': 32,
                'use_learned_importance': True,
                'preservation_ratio': 0.3
            },
            'goals': {
                'initial_phase': 'template',
                'environment_bounds': [0, 64, 0, 64]  # ARC grid bounds
            },
            'action_selection': {
                'hidden_size': 128,
                'action_size': 3,  # 3D action space for ARC
                'num_goals': 3,
                'max_velocity': 1.0,
                'action_noise': 0.05,
                'exploration_rate': 0.2,  # Higher exploration for reasoning
                'curiosity_weight': 0.5,
                'learning_rate': 0.002
            },
            'sleep': {
                'sleep_trigger_energy': 10.0,
                'sleep_trigger_boredom_steps': 50,
                'sleep_trigger_memory_pressure': 0.8,
                'sleep_duration_steps': 20,
                'replay_batch_size': 16,
                'learning_rate': 0.001
            },
            'meta_learning': {
                'memory_capacity': 500,
                'insight_threshold': 0.2,
                'consolidation_interval': 25,
                'save_directory': 'arc_meta_learning_data'
            }
        }
        
    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        """Determine if the agent should stop playing."""
        # Continue until win or explicit game over
        if latest_frame.state == GameState.WIN:
            self._record_success(frames, latest_frame)
            return True
        elif latest_frame.state == GameState.GAME_OVER:
            self._record_failure(frames, latest_frame)
            return False  # Allow retry for learning
        
        return False
        
    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        """Choose action using the Adaptive Learning Agent."""
        try:
            # Handle game state transitions
            if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                return self._handle_reset(frames, latest_frame)
                
            # Convert ARC frame to sensory input
            sensory_input = self._convert_frame_to_sensory_input(latest_frame)
            
            # Get agent's current state
            agent_state = self.learning_agent.get_agent_state()
            
            # Generate action using the learning agent
            agent_action = self._generate_agent_action(sensory_input, agent_state)
            
            # Convert to ARC action
            arc_action, action_data = self.action_mapper.agent_to_arc_action(agent_action)
            
            # Set action data if needed
            if action_data:
                arc_action.set_data(action_data)
                
            # Generate reasoning
            reasoning = self._generate_reasoning(sensory_input, agent_action, arc_action)
            arc_action.reasoning = reasoning
            
            # Record for learning
            self._record_step(latest_frame, arc_action, reasoning)
            
            return arc_action
            
        except Exception as e:
            logger.error(f"Error in choose_action: {e}")
            # Fallback to reset
            return GameAction.RESET
            
    def _handle_reset(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        """Handle game reset."""
        if latest_frame.state == GameState.GAME_OVER:
            # Learn from the previous attempt
            self._consolidate_episode_learning()
            
        # Reset episode data
        self.current_episode_data = {
            'frames': [],
            'actions': [],
            'reasoning': [],
            'learning_progress': []
        }
        
        # Reset learning agent for new episode
        self.learning_agent.reset_episode()
        
        return GameAction.RESET
        
    def _convert_frame_to_sensory_input(self, frame: FrameData) -> SensoryInput:
        """Convert ARC frame to sensory input for the learning agent."""
        # Process visual data
        visual_tensor = self.visual_processor(frame.frame)
        
        # Create proprioception data (simplified for ARC)
        proprioception = torch.tensor([
            float(frame.score) / 100.0,  # Normalized score
            1.0 if frame.state == GameState.WIN else 0.0,  # Win state
            1.0 if frame.state == GameState.GAME_OVER else 0.0,  # Game over state
            float(len(frame.available_actions)) / 10.0,  # Available actions
            float(self.action_counter) / 100.0,  # Progress through game
            0.0, 0.0, 0.0  # Padding to match expected size
        ])
        
        return SensoryInput(
            visual=visual_tensor,
            proprioception=proprioception.unsqueeze(0),
            energy_level=90.0,  # High energy for reasoning tasks
            timestamp=time.time()
        )
        
    def _generate_agent_action(self, sensory_input: SensoryInput, agent_state: AgentState) -> torch.Tensor:
        """Generate action using the learning agent."""
        # Step the learning agent
        step_results = self.learning_agent.step(sensory_input, torch.zeros(3))
        
        # Extract action from predictions or use exploration
        if 'predictions' in step_results:
            # Use predictive core output to inform action
            predictions = step_results['predictions']
            if len(predictions) > 0:
                # Convert prediction to action (simplified)
                pred_tensor = predictions[0] if isinstance(predictions[0], torch.Tensor) else torch.tensor(predictions[0])
                action = torch.tanh(pred_tensor.mean(dim=0))[:3]  # Limit to 3D action space
            else:
                action = torch.randn(3) * 0.5  # Random exploration
        else:
            action = torch.randn(3) * 0.5  # Random exploration
            
        return action
        
    def _generate_reasoning(self, sensory_input: SensoryInput, agent_action: torch.Tensor, arc_action: GameAction) -> Dict[str, Any]:
        """Generate reasoning explanation for the action."""
        # Get learning agent metrics
        metrics = self.learning_agent.get_performance_metrics()
        
        reasoning = {
            "action_type": arc_action.name,
            "agent_action_vector": agent_action.tolist(),
            "learning_progress": metrics.get('learning_progress_avg', 0.0),
            "energy_level": sensory_input.energy_level,
            "step_count": self.action_counter,
            "exploration_factor": 0.2,  # Could be dynamic
            "confidence": min(1.0, metrics.get('learning_progress_avg', 0.0) + 0.5)
        }
        
        return reasoning
        
    def _record_step(self, frame: FrameData, action: GameAction, reasoning: Dict[str, Any]):
        """Record step data for learning."""
        self.current_episode_data['frames'].append(frame)
        self.current_episode_data['actions'].append(action)
        self.current_episode_data['reasoning'].append(reasoning)
        
        # Record learning progress
        lp = reasoning.get('learning_progress', 0.0)
        self.current_episode_data['learning_progress'].append(lp)
        
    def _record_success(self, frames: List[FrameData], final_frame: FrameData):
        """Record successful completion for meta-learning."""
        success_data = {
            'game_id': self.game_id,
            'final_score': final_frame.score,
            'actions_taken': self.action_counter,
            'success': True,
            'patterns': self._extract_patterns(),
            'timestamp': time.time()
        }
        
        self.performance_history.append(success_data)
        self._consolidate_episode_learning()
        
        logger.info(f"Game {self.game_id} completed successfully with score {final_frame.score}")
        
    def _record_failure(self, frames: List[FrameData], final_frame: FrameData):
        """Record failure for learning."""
        failure_data = {
            'game_id': self.game_id,
            'final_score': final_frame.score,
            'actions_taken': self.action_counter,
            'success': False,
            'patterns': self._extract_patterns(),
            'timestamp': time.time()
        }
        
        self.performance_history.append(failure_data)
        
        logger.info(f"Game {self.game_id} ended in failure with score {final_frame.score}")
        
    def _extract_patterns(self) -> Dict[str, Any]:
        """Extract patterns from current episode for meta-learning."""
        if not self.current_episode_data['reasoning']:
            return {}
            
        patterns = {
            'avg_confidence': np.mean([r.get('confidence', 0.5) for r in self.current_episode_data['reasoning']]),
            'exploration_ratio': np.mean([r.get('exploration_factor', 0.2) for r in self.current_episode_data['reasoning']]),
            'action_diversity': len(set(a.name for a in self.current_episode_data['actions'])),
            'learning_trend': np.mean(self.current_episode_data['learning_progress'][-10:]) if len(self.current_episode_data['learning_progress']) >= 10 else 0.0
        }
        
        return patterns
        
    def _consolidate_episode_learning(self):
        """Consolidate learning from the episode."""
        if not self.current_episode_data['reasoning']:
            return
            
        # Create insight for meta-learning system
        episode_insight = {
            'context': f"ARC_game_{self.game_id}",
            'patterns': self._extract_patterns(),
            'performance': self.performance_history[-1] if self.performance_history else {},
            'episode_data': {
                'total_actions': len(self.current_episode_data['actions']),
                'avg_learning_progress': np.mean(self.current_episode_data['learning_progress']),
                'action_sequence': [a.name for a in self.current_episode_data['actions'][-10:]]  # Last 10 actions
            }
        }
        
        # Add to meta-learning system
        if hasattr(self.learning_agent, 'meta_learning'):
            self.learning_agent.meta_learning.add_insight(episode_insight)
            
        # Save learning data
        self._save_learning_data(episode_insight)
        
    def _save_learning_data(self, insight: Dict[str, Any]):
        """Save learning data for future analysis."""
        save_dir = Path("arc_learning_data")
        save_dir.mkdir(exist_ok=True)
        
        filename = save_dir / f"{self.game_id}_episode_{len(self.performance_history)}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(insight, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save learning data: {e}")
            
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress."""
        if not self.performance_history:
            return {"status": "no_data"}
            
        recent_performance = self.performance_history[-10:]  # Last 10 episodes
        
        return {
            "total_episodes": len(self.performance_history),
            "success_rate": sum(1 for p in recent_performance if p['success']) / len(recent_performance),
            "avg_score": np.mean([p['final_score'] for p in recent_performance]),
            "avg_actions": np.mean([p['actions_taken'] for p in recent_performance]),
            "learning_trend": "improving" if len(recent_performance) > 5 and 
                           recent_performance[-1]['final_score'] > recent_performance[0]['final_score'] else "stable",
            "agent_metrics": self.learning_agent.get_performance_metrics()
        }
