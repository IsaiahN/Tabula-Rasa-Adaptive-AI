"""
Main Agent Class - Integrates all core components for autonomous learning.

This module implements the complete agent that combines predictive core,
learning progress drive, memory system, goal system, and energy management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging
from collections import deque
import time

from core.data_models import SensoryInput, Experience, AgentState, Goal
from core.predictive_core import PredictiveCore
from core.learning_progress import LearningProgressDrive
from core.energy_system import EnergySystem, DeathManager
from core.sleep_system import SleepCycle
from core.action_selection import ActionSelectionNetwork, ActionExecutor, ExplorationStrategy
from goals.goal_system import GoalInventionSystem, GoalPhase
from memory.dnc import DNCMemory
from monitoring.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class AdaptiveLearningAgent:
    """
    Main agent class that integrates all core components.
    
    Implements the complete sensory-prediction-action cycle with
    intrinsic motivation, memory, and survival mechanics.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cpu"
    ):
        self.config = config
        self.device = device
        
        # Initialize core components
        self._init_predictive_core()
        self._init_memory_system()
        self._init_learning_progress()
        self._init_energy_system()
        self._init_goal_system()
        self._init_action_selection()
        self._init_sleep_system()
        self._init_monitoring()
        
        # Agent state
        self.agent_state = self._create_initial_state()
        self.step_count = 0
        self.episode_count = 0
        
        # Experience tracking
        self.experience_buffer = deque(maxlen=10000)
        self.state_history = deque(maxlen=100)
        
        # Learning state
        self.optimizer = optim.Adam(
            self.predictive_core.parameters(),
            lr=config.get('learning_rate', 0.001)
        )
        
        # Bootstrap protection
        self.bootstrap_manager = BootstrapManager(
            protection_steps=config.get('bootstrap_protection_steps', 10000)
        )
        
        # Performance tracking
        self.performance_metrics = {
            'total_steps': 0,
            'episodes_completed': 0,
            'deaths': 0,
            'goals_achieved': 0,
            'learning_progress_avg': 0.0,
            'survival_rate': 0.0
        }
        
        logger.info("Adaptive Learning Agent initialized successfully")
        
    def _init_predictive_core(self):
        """Initialize the predictive core system."""
        core_config = self.config.get('predictive_core', {})
        
        self.predictive_core = PredictiveCore(
            visual_size=core_config.get('visual_size', (3, 64, 64)),
            proprioception_size=core_config.get('proprioception_size', 12),
            hidden_size=core_config.get('hidden_size', 512),
            architecture=core_config.get('architecture', 'lstm')
        ).to(self.device)
        
        logger.info(f"Predictive core initialized with {core_config.get('architecture', 'lstm')} architecture")
        
    def _init_memory_system(self):
        """Initialize the DNC memory system."""
        memory_config = self.config.get('memory', {})
        
        if memory_config.get('enabled', True):
            self.memory = DNCMemory(
                memory_size=memory_config.get('memory_size', 512),
                word_size=memory_config.get('word_size', 64),
                num_read_heads=memory_config.get('num_read_heads', 4),
                num_write_heads=memory_config.get('num_write_heads', 1),
                controller_size=memory_config.get('controller_size', 256)
            ).to(self.device)
            
            # Update predictive core to use memory
            self.predictive_core.memory = self.memory
            self.predictive_core.use_memory = True
            
            logger.info("DNC memory system initialized and integrated with predictive core")
        else:
            self.memory = None
            logger.info("Memory system disabled")
            
    def _init_learning_progress(self):
        """Initialize the learning progress drive."""
        lp_config = self.config.get('learning_progress', {})
        
        self.lp_drive = LearningProgressDrive(
            smoothing_window=lp_config.get('smoothing_window', 500),
            derivative_clamp=lp_config.get('derivative_clamp', (-1.0, 1.0)),
            boredom_threshold=lp_config.get('boredom_threshold', 0.01),
            boredom_steps=lp_config.get('boredom_steps', 500),
            lp_weight=lp_config.get('lp_weight', 0.7),
            empowerment_weight=lp_config.get('empowerment_weight', 0.3),
            use_adaptive_weights=lp_config.get('use_adaptive_weights', False)
        )
        
        logger.info("Learning progress drive initialized")
        
    def _init_energy_system(self):
        """Initialize the energy and death management system."""
        energy_config = self.config.get('energy', {})
        
        self.energy_system = EnergySystem(
            max_energy=energy_config.get('max_energy', 100.0),
            base_consumption=energy_config.get('base_consumption', 0.01),
            action_multiplier=energy_config.get('action_multiplier', 0.5),
            computation_multiplier=energy_config.get('computation_multiplier', 0.001),
            food_energy_value=energy_config.get('food_energy_value', 10.0)
        )
        
        self.death_manager = DeathManager(
            memory_size=energy_config.get('memory_size', 512),
            word_size=energy_config.get('word_size', 64),
            use_learned_importance=energy_config.get('use_learned_importance', False),
            preservation_ratio=energy_config.get('preservation_ratio', 0.2)
        )
        
        logger.info("Energy and death management system initialized")
        
    def _init_goal_system(self):
        """Initialize the goal invention system."""
        goal_config = self.config.get('goals', {})
        
        self.goal_system = GoalInventionSystem(
            phase=GoalPhase(goal_config.get('initial_phase', 'survival')),
            environment_bounds=goal_config.get('environment_bounds', (-10, 10, -10, 10))
        )
        
        logger.info(f"Goal system initialized in {goal_config.get('initial_phase', 'survival')} phase")
        
    def _init_action_selection(self):
        """Initialize the action selection system."""
        action_config = self.config.get('action_selection', {})
        
        # Calculate input size for action selection network
        # This should match the output size of the predictive core
        input_size = self.predictive_core.hidden_size
        
        self.action_network = ActionSelectionNetwork(
            input_size=input_size,
            hidden_size=action_config.get('hidden_size', 256),
            action_size=action_config.get('action_size', 8),
            num_goals=action_config.get('num_goals', 5)
        ).to(self.device)
        
        self.action_executor = ActionExecutor(
            max_velocity=action_config.get('max_velocity', 2.0),
            action_noise=action_config.get('action_noise', 0.1)
        )
        
        self.exploration_strategy = ExplorationStrategy(
            exploration_rate=action_config.get('exploration_rate', 0.1),
            curiosity_weight=action_config.get('curiosity_weight', 0.3)
        )
        
        # Action selection optimizer
        self.action_optimizer = optim.Adam(
            self.action_network.parameters(),
            lr=action_config.get('learning_rate', 0.001)
        )
        
        logger.info("Action selection system initialized")
        
    def _init_sleep_system(self):
        """Initialize the sleep and dream cycle system."""
        sleep_config = self.config.get('sleep', {})
        
        self.sleep_system = SleepCycle(
            predictive_core=self.predictive_core,
            sleep_trigger_energy=sleep_config.get('sleep_trigger_energy', 20.0),
            sleep_trigger_boredom_steps=sleep_config.get('sleep_trigger_boredom_steps', 1000),
            sleep_trigger_memory_pressure=sleep_config.get('sleep_trigger_memory_pressure', 0.9),
            sleep_duration_steps=sleep_config.get('sleep_duration_steps', 100),
            replay_batch_size=sleep_config.get('replay_batch_size', 32),
            learning_rate=sleep_config.get('learning_rate', 0.001)
        )
        
        logger.info("Sleep and dream cycle system initialized")
        
    def _init_monitoring(self):
        """Initialize the monitoring and metrics collection system."""
        monitoring_config = self.config.get('monitoring', {})
        
        self.metrics_collector = MetricsCollector(
            buffer_size=monitoring_config.get('buffer_size', 10000),
            logging_mode=monitoring_config.get('logging_mode', 'minimal')
        )
        
        logger.info("Monitoring and metrics system initialized")
        
    def _create_initial_state(self) -> AgentState:
        """Create initial agent state."""
        return AgentState(
            position=torch.tensor([0.0, 0.0, 1.0]),  # Spawn at origin
            orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),  # Identity quaternion
            energy=100.0,  # Full energy
            hidden_state=None,  # Will be initialized on first forward pass
            active_goals=[],
            memory_state=None,  # Will be initialized if memory system enabled
            timestamp=0
        )
        
    def step(self, sensory_input: SensoryInput, action: torch.Tensor) -> Dict[str, Any]:
        """
        Execute one agent step.
        
        Args:
            sensory_input: Current sensory input
            action: Action to take
            
        Returns:
            step_results: Results of the step
        """
        self.step_count += 1
        
        # Bootstrap protection
        is_protected = self.bootstrap_manager.is_protected()
        
        # 1. Update agent state with sensory input
        self.agent_state.energy = sensory_input.energy_level
        self.agent_state.timestamp = sensory_input.timestamp
        
        # 2. Generate predictions
        predictions, hidden_state, debug_info = self._generate_predictions(sensory_input)
        
        # 3. Compute learning progress and reward
        prediction_errors = self.predictive_core.compute_prediction_error(
            predictions, sensory_input
        )
        
        total_error = prediction_errors['total'].mean().item()
        lp_signal = self.lp_drive.compute_reward(total_error, list(self.state_history))
        
        # 4. Consume energy
        action_cost = torch.norm(action).item() if action is not None else 0.0
        computation_cost = 1.0  # One forward pass
        remaining_energy = self.energy_system.consume_energy(
            action_cost * (0.1 if is_protected else 1.0),  # Bootstrap protection
            computation_cost * (0.1 if is_protected else 1.0)
        )
        
        # 5. Update agent state
        self.agent_state.hidden_state = hidden_state
        
        # Update memory state if memory system is enabled
        if self.predictive_core.use_memory and self.memory is not None and debug_info:
            if 'memory_usage' in debug_info:
                if self.agent_state.memory_state is None:
                    self.agent_state.memory_state = {}
                self.agent_state.memory_state.update({
                    'read_weights': debug_info.get('read_weights', torch.zeros_like(action[:1])),
                    'write_weights': debug_info.get('write_weights', torch.zeros_like(action[:1])),
                    'controller_state': debug_info.get('controller_state', None)
                })
        
        self.agent_state.energy = remaining_energy
        
        # 6. Check for death
        if self.energy_system.is_dead():
            self._handle_death()
            
        # 7. Update goals
        active_goals = self.goal_system.get_active_goals(self.agent_state)
        self.agent_state.active_goals = active_goals
        
        # 8. Check sleep triggers (disabled for Phase 2 testing)
        should_sleep = False  # Disable sleep for Phase 2 testing
        
        # if should_sleep:
        #     self._enter_sleep_mode()
            
        # 9. Record experience
        experience = Experience(
            state=sensory_input,
            action=action,
            next_state=sensory_input,  # Simplified - would come from environment
            learning_progress=lp_signal,
            energy_change=remaining_energy - sensory_input.energy_level,
            timestamp=sensory_input.timestamp
        )
        
        self.experience_buffer.append(experience)
        self.sleep_system.add_experience(experience)
        
        # 10. Update state history for empowerment calculation
        state_repr = self.predictive_core.get_state_representation(sensory_input, hidden_state)
        self.state_history.append(state_repr)
        
        # 11. Feed experience to emergent goal system if in emergent phase
        if self.goal_system.current_phase.value == "emergent":
            self.goal_system.emergent_goals.add_experience(
                state_repr, lp_signal, self.agent_state
            )
        
        # 12. Collect metrics
        self._update_metrics(lp_signal, total_error, action_cost)
        
        # 13. Check phase transitions
        phase_changed = self.goal_system.check_phase_transition()
        if phase_changed:
            logger.info(f"Goal system advanced to {self.goal_system.current_phase.value} phase")
            
        return {
            'predictions': predictions,
            'learning_progress': lp_signal,
            'prediction_errors': prediction_errors,
            'energy': remaining_energy,
            'active_goals': active_goals,
            'should_sleep': should_sleep,
            'is_protected': is_protected,
            'phase_changed': phase_changed,
            'debug_info': debug_info
        }
        
    def _generate_predictions(
        self, 
        sensory_input: SensoryInput
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Dict]:
        """Generate predictions using the predictive core."""
        # Initialize hidden state if needed
        if self.agent_state.hidden_state is None:
            batch_size = sensory_input.visual.size(0)
            logger.info(f"Initializing hidden state with batch_size: {batch_size}, visual shape: {sensory_input.visual.shape}")
            if self.predictive_core.architecture == "lstm":
                hidden_state = (
                    torch.zeros(1, batch_size, self.predictive_core.hidden_size, device=self.device),
                    torch.zeros(1, batch_size, self.predictive_core.hidden_size, device=self.device)
                )
            else:
                hidden_state = torch.zeros(1, batch_size, self.predictive_core.hidden_size, device=self.device)
        else:
            hidden_state = self.agent_state.hidden_state
            # Ensure hidden state has correct batch size
            if isinstance(hidden_state, tuple):
                batch_size = sensory_input.visual.size(0)
                if hidden_state[0].size(1) != batch_size:
                    logger.warning(f"Hidden state batch size mismatch: expected {batch_size}, got {hidden_state[0].size(1)}")
                    # Resize hidden state to match current batch size
                    if self.predictive_core.architecture == "lstm":
                        hidden_state = (
                            hidden_state[0][:, :batch_size, :],
                            hidden_state[1][:, :batch_size, :]
                        )
                    else:
                        hidden_state = hidden_state[:, :batch_size, :]
            
        # Initialize memory reads if needed
        memory_reads = None
        if self.predictive_core.use_memory and self.memory is not None:
            if self.agent_state.memory_state is None:
                # Initialize memory state
                batch_size = sensory_input.visual.size(0)
                memory_read_size = self.memory.num_read_heads * self.memory.word_size
                memory_reads = torch.zeros(batch_size, memory_read_size, device=self.device)
                self.agent_state.memory_state = {
                    'read_weights': torch.zeros(batch_size, self.memory.num_read_heads, self.memory.memory_size, device=self.device),
                    'write_weights': torch.zeros(batch_size, self.memory.num_write_heads, self.memory.memory_size, device=self.device),
                    'controller_state': None
                }
            else:
                # Use existing memory state
                memory_read_size = self.memory.num_read_heads * self.memory.word_size
                memory_reads = torch.zeros(sensory_input.visual.size(0), memory_read_size, device=self.device)
            
        # Forward pass
        predictions = self.predictive_core(sensory_input, hidden_state, memory_reads)
        
        return predictions[:3], predictions[3], predictions[4]  # predictions, hidden_state, debug_info
        
    def _handle_death(self):
        """Handle agent death and respawn."""
        logger.info("Agent died - initiating respawn")
        
        # Perform selective reset
        new_state = self.death_manager.selective_reset(self.agent_state)
        
        # Update agent state
        self.agent_state = new_state
        
        # Reset energy system
        self.energy_system.reset_energy()
        
        # Reset sleep system
        self.sleep_system.reset()
        
        # Reset learning progress drive
        self.lp_drive.reset_boredom_counter()
        
        # Update episode tracking
        self.episode_count += 1
        self.performance_metrics['deaths'] += 1
        
        # Record episode data
        episode_data = {
            'episode': self.episode_count,
            'steps': self.step_count,
            'died': True,
            'final_energy': 0.0
        }
        self.goal_system.reset_episode(episode_data)
        
        logger.info(f"Agent respawned. Episode {self.episode_count} completed")
        
    def _enter_sleep_mode(self):
        """Enter sleep mode for offline learning."""
        if self.sleep_system.is_sleeping:
            return
            
        logger.info("Agent entering sleep mode")
        
        # Enter sleep
        self.sleep_system.enter_sleep(self.agent_state)
        
        # Execute sleep cycle
        sleep_results = self.sleep_system.execute_sleep_cycle(list(self.experience_buffer))
        
        # Wake up
        wake_results = self.sleep_system.wake_up()
        
        logger.info(f"Sleep cycle completed: {sleep_results}")
        
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage if memory system is enabled."""
        if self.memory is not None:
            metrics = self.memory.get_memory_metrics()
            return metrics.get('memory_utilization', 0.0)
        return None
        
    def _update_metrics(self, lp_signal: float, total_error: float, action_cost: float):
        """Update performance metrics."""
        self.performance_metrics['total_steps'] = self.step_count
        self.performance_metrics['learning_progress_avg'] = (
            (self.performance_metrics['learning_progress_avg'] * (self.step_count - 1) + lp_signal) / 
            self.step_count
        )
        
        # Update metrics collector
        self.metrics_collector.log_step(
            agent_state=self.agent_state,
            prediction_error=total_error,
            lp_signal=lp_signal,
            memory_usage=self._get_memory_usage(),
            additional_metrics={
                'step': self.step_count,
                'episode': self.episode_count,
                'action_cost': action_cost,
                'active_goals': len(self.agent_state.active_goals),
                'is_sleeping': self.sleep_system.is_sleeping
            }
        )
        
    def get_agent_state(self) -> AgentState:
        """Get current agent state."""
        return self.agent_state
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = self.performance_metrics.copy()
        
        # Add component-specific metrics
        metrics.update({
            'energy_metrics': self.energy_system.get_energy_metrics(),
            'goal_metrics': self.goal_system.get_goal_metrics(),
            'sleep_metrics': self.sleep_system.get_sleep_metrics(),
            'lp_validation': self.lp_drive.get_validation_metrics(),
            'death_metrics': self.death_manager.get_death_metrics()
        })
        
        return metrics
        
    def reset_episode(self):
        """Reset for new episode."""
        # Reset step counter
        self.step_count = 0
        
        # Record episode data
        episode_data = {
            'episode': self.episode_count,
            'steps': 0,
            'died': False,
            'final_energy': self.agent_state.energy
        }
        self.goal_system.reset_episode(episode_data)
        
        # Reset sleep system
        self.sleep_system.reset()
        
        # Reset learning progress drive
        self.lp_drive.reset_boredom_counter()
        
        logger.info(f"Episode {self.episode_count} reset")
        
    def save_checkpoint(self, filepath: str):
        """Save agent checkpoint."""
        checkpoint = {
            'agent_state': self.agent_state,
            'predictive_core_state': self.predictive_core.state_dict(),
            'memory_state': self.memory.state_dict() if self.memory else None,
            'optimizer_state': self.optimizer.state_dict(),
            'performance_metrics': self.performance_metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
        
    def load_checkpoint(self, filepath: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.agent_state = checkpoint['agent_state']
        self.predictive_core.load_state_dict(checkpoint['predictive_core_state'])
        
        if self.memory and checkpoint['memory_state']:
            self.memory.load_state_dict(checkpoint['memory_state'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.performance_metrics = checkpoint['performance_metrics']
        
        logger.info(f"Checkpoint loaded from {filepath}")


class BootstrapManager:
    """Manages bootstrap protection for newborn agents."""
    
    def __init__(self, protection_steps: int = 10000):
        self.protection_steps = protection_steps
        self.steps_taken = 0
        
    def is_protected(self) -> bool:
        """Check if agent is still in bootstrap protection period."""
        return self.steps_taken < self.protection_steps
        
    def step(self):
        """Increment step counter."""
        self.steps_taken += 1
        
    def reset(self):
        """Reset bootstrap protection (for new episodes)."""
        self.steps_taken = 0 