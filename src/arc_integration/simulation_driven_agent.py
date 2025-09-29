#!/usr/bin/env python3
"""
Simulation-Driven ARC Agent

This module integrates the simulation-driven intelligence system with the ARC-AGI-3
integration, replacing reactive action selection with proactive multi-step planning.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Import simulation components
from src.core.simulation_agent import SimulationAgent
from src.core.simulation_models import SimulationContext, SimulationConfig
from src.core.predictive_core import PredictiveCore

logger = logging.getLogger(__name__)

class SimulationDrivenARCAgent:
    """
    ARC agent that uses simulation-driven intelligence for action selection.
    
    This replaces the reactive (S → A → S+1) approach with proactive
    simulation (S → [A1→S+1→A2→S+2...] → Best_A).
    """
    
    def __init__(self, 
                 predictive_core: PredictiveCore,
                 config: Optional[SimulationConfig] = None,
                 persistence_dir: str = None):  # Database-only mode
        
        self.simulation_agent = SimulationAgent(
            predictive_core=predictive_core,
            config=config,
            persistence_dir=persistence_dir
        )
        
        # ARC-specific state tracking
        self.current_game_id: Optional[str] = None
        self.action_count = 0
        self.game_state_history: List[Dict[str, Any]] = []
        self.frame_analysis_cache: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.simulation_decisions = 0
        self.strategy_hits = 0
        self.fallback_decisions = 0
        
        logger.debug("Simulation-Driven ARC Agent initialized")
    
    def select_action_with_simulation(self, 
                                    response_data: Dict[str, Any], 
                                    game_id: str,
                                    frame_analyzer: Optional[Any] = None) -> Tuple[int, Optional[Tuple[int, int]], str]:
        """
        Select action using simulation-driven intelligence.
        
        This is the main entry point that replaces simple action selection
        with multi-step strategic planning.
        """
        
        self.current_game_id = game_id
        self.action_count += 1
        
        # Extract current state from response data
        current_state = self._extract_current_state(response_data, game_id)
        
        # Get available actions
        available_actions = response_data.get('available_actions', [])
        if not available_actions:
            logger.warning(f"No available actions for game {game_id}")
            return 6, (32, 32), "Fallback: No available actions"
        
        # Perform frame analysis if analyzer is available
        frame_analysis = None
        if frame_analyzer and 'frame' in response_data:
            try:
                frame_analysis = frame_analyzer.analyze_frame(response_data['frame'], game_id)
                self.frame_analysis_cache[game_id] = frame_analysis
            except Exception as e:
                logger.warning(f"Frame analysis failed: {e}")
        
        # Get memory patterns from game state history
        memory_patterns = self._extract_memory_patterns(game_id)
        
        # Use simulation agent to generate action plan
        try:
            action, coordinates, reasoning = self.simulation_agent.generate_action_plan(
                current_state=current_state,
                available_actions=available_actions,
                frame_analysis=frame_analysis,
                memory_patterns=memory_patterns
            )
            
            self.simulation_decisions += 1
            
            # Check if this was a strategy-based decision
            if "strategy" in reasoning.lower():
                self.strategy_hits += 1
            
            logger.debug(f"Simulation decision: action={action}, coords={coordinates}, reasoning={reasoning}")
            
            return action, coordinates, reasoning
            
        except Exception as e:
            logger.error(f"Simulation-driven action selection failed: {e}")
            # Fallback to simple action selection
            action, coordinates = self._fallback_action_selection(available_actions, current_state)
            self.fallback_decisions += 1
            return action, coordinates, f"Fallback due to error: {str(e)}"
    
    def update_with_action_outcome(self, 
                                 action: int,
                                 coordinates: Optional[Tuple[int, int]],
                                 response_data: Dict[str, Any],
                                 game_id: str):
        """Update the simulation system with the outcome of the selected action."""
        
        # Extract outcome information
        outcome = self._extract_action_outcome(response_data, action, coordinates)
        
        # Update simulation agent
        self.simulation_agent.update_with_real_outcome(action, coordinates, outcome)
        
        # Update game state history
        self._update_game_state_history(game_id, action, coordinates, outcome)
        
        logger.debug(f"Updated simulation system with outcome: {outcome}")
    
    def _extract_current_state(self, response_data: Dict[str, Any], game_id: str) -> Dict[str, Any]:
        """Extract current state information from ARC response data."""
        
        # Basic state extraction
        state = {
            'game_id': game_id,
            'action_count': self.action_count,
            'timestamp': time.time(),
            'energy': 100.0 - (self.action_count * 0.1),  # Simple energy model
            'learning_drive': 0.5,  # Default learning drive
            'boredom_level': min(1.0, self.action_count / 100.0),  # Simple boredom model
            'recent_actions': self._get_recent_actions(game_id),
            'success_history': self._get_success_history(game_id)
        }
        
        # Add frame analysis if available
        if game_id in self.frame_analysis_cache:
            state['frame_analysis'] = self.frame_analysis_cache[game_id]
        
        # Add grid information if available
        if 'frame' in response_data and response_data['frame']:
            frame = response_data['frame']
            if isinstance(frame, list) and len(frame) > 0:
                state['grid_height'] = len(frame)
                if isinstance(frame[0], list):
                    state['grid_width'] = len(frame[0])
                else:
                    state['grid_width'] = 1
        
        return state
    
    def _extract_memory_patterns(self, game_id: str) -> Dict[str, Any]:
        """Extract memory patterns from game state history."""
        
        patterns = {
            'successful_actions': [],
            'successful_coordinates': [],
            'action_effectiveness': {},
            'coordinate_success_zones': {}
        }
        
        # Analyze recent game history
        recent_states = self.game_state_history[-20:]  # Last 20 states
        
        for state in recent_states:
            if state.get('game_id') != game_id:
                continue
            
            # Extract successful actions
            if state.get('success', False):
                action = state.get('action')
                if action:
                    patterns['successful_actions'].append(action)
                
                coords = state.get('coordinates')
                if coords:
                    patterns['successful_coordinates'].append(coords)
            
            # Track action effectiveness
            action = state.get('action')
            if action:
                if action not in patterns['action_effectiveness']:
                    patterns['action_effectiveness'][action] = {'successes': 0, 'attempts': 0}
                
                patterns['action_effectiveness'][action]['attempts'] += 1
                if state.get('success', False):
                    patterns['action_effectiveness'][action]['successes'] += 1
        
        # Calculate success rates
        for action, data in patterns['action_effectiveness'].items():
            if data['attempts'] > 0:
                data['success_rate'] = data['successes'] / data['attempts']
            else:
                data['success_rate'] = 0.0
        
        return patterns
    
    def _extract_action_outcome(self, 
                              response_data: Dict[str, Any], 
                              action: int,
                              coordinates: Optional[Tuple[int, int]]) -> Dict[str, Any]:
        """Extract outcome information from ARC response data."""
        
        outcome = {
            'action': action,
            'coordinates': coordinates,
            'timestamp': time.time(),
            'success': False,
            'energy_change': 0.0,
            'learning_gain': 0.0
        }
        
        # Determine success based on response data
        if 'reward' in response_data:
            outcome['success'] = response_data['reward'] > 0
        elif 'score' in response_data:
            outcome['success'] = response_data['score'] > 0
        elif 'done' in response_data:
            outcome['success'] = response_data['done']
        
        # Estimate energy change (simplified)
        energy_costs = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 1.0, 6: 2.0, 7: 0.1}
        outcome['energy_change'] = -energy_costs.get(action, 1.0)
        
        # Estimate learning gain
        if outcome['success']:
            outcome['learning_gain'] = 0.1
        else:
            outcome['learning_gain'] = 0.01  # Small learning from failures
        
        return outcome
    
    def _update_game_state_history(self, 
                                 game_id: str,
                                 action: int,
                                 coordinates: Optional[Tuple[int, int]],
                                 outcome: Dict[str, Any]):
        """Update the game state history with new action and outcome."""
        
        state_entry = {
            'game_id': game_id,
            'action': action,
            'coordinates': coordinates,
            'timestamp': time.time(),
            'success': outcome.get('success', False),
            'energy_change': outcome.get('energy_change', 0.0),
            'learning_gain': outcome.get('learning_gain', 0.0)
        }
        
        self.game_state_history.append(state_entry)
        
        # Keep only recent history
        if len(self.game_state_history) > 1000:
            self.game_state_history = self.game_state_history[-1000:]
    
    def _get_recent_actions(self, game_id: str) -> List[int]:
        """Get recent actions for the current game."""
        
        recent_actions = []
        for state in reversed(self.game_state_history[-50:]):  # Last 50 states
            if state.get('game_id') == game_id:
                action = state.get('action')
                if action:
                    recent_actions.append(action)
                if len(recent_actions) >= 10:  # Last 10 actions
                    break
        
        return list(reversed(recent_actions))  # Return in chronological order
    
    def _get_success_history(self, game_id: str) -> List[Dict[str, Any]]:
        """Get success history for the current game."""
        
        success_history = []
        for state in reversed(self.game_state_history[-100:]):  # Last 100 states
            if state.get('game_id') == game_id and state.get('success', False):
                success_entry = {
                    'action': state.get('action'),
                    'coordinates': state.get('coordinates'),
                    'timestamp': state.get('timestamp'),
                    'energy_change': state.get('energy_change', 0.0),
                    'learning_gain': state.get('learning_gain', 0.0)
                }
                success_history.append(success_entry)
                if len(success_history) >= 20:  # Last 20 successes
                    break
        
        return list(reversed(success_history))  # Return in chronological order
    
    def _fallback_action_selection(self, 
                                 available_actions: List[int],
                                 current_state: Dict[str, Any]) -> Tuple[int, Optional[Tuple[int, int]]]:
        """Fallback action selection when simulation fails."""
        
        # Prefer movement actions
        movement_actions = [a for a in [1, 2, 3, 4] if a in available_actions]
        if movement_actions:
            action = np.random.choice(movement_actions)
            return action, None
        
        # Prefer interaction actions
        if 5 in available_actions:
            return 5, None
        
        # Use coordinate action with random coordinates
        if 6 in available_actions:
            coords = (np.random.randint(0, 64), np.random.randint(0, 64))
            return 6, coords
        
        # Use any available action
        if available_actions:
            return available_actions[0], None
        
        # Default fallback
        return 1, None
    
    def get_simulation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the simulation system."""
        
        base_stats = self.simulation_agent.get_simulation_statistics()
        
        # Add ARC-specific statistics
        arc_stats = {
            'arc_specific': {
                'simulation_decisions': self.simulation_decisions,
                'strategy_hits': self.strategy_hits,
                'fallback_decisions': self.fallback_decisions,
                'strategy_hit_rate': self.strategy_hits / max(1, self.simulation_decisions),
                'current_game_id': self.current_game_id,
                'action_count': self.action_count,
                'game_state_history_size': len(self.game_state_history)
            }
        }
        
        # Combine statistics
        combined_stats = {**base_stats, **arc_stats}
        
        return combined_stats
    
    def reset_for_new_game(self, game_id: str):
        """Reset the agent state for a new game."""
        
        self.current_game_id = game_id
        self.action_count = 0
        
        # Clear game-specific caches
        if game_id in self.frame_analysis_cache:
            del self.frame_analysis_cache[game_id]
        
        logger.info(f"Reset simulation agent for new game: {game_id}")
    
    def enable_simulation_mode(self, enabled: bool = True):
        """Enable or disable simulation mode."""
        
        if enabled:
            logger.info("Simulation mode enabled - using multi-step planning")
        else:
            logger.info("Simulation mode disabled - using fallback action selection")
        
        # This could be implemented by modifying the simulation agent's behavior
        # For now, we'll just log the change
    
    def get_imagination_status(self) -> Dict[str, Any]:
        """Get the current status of the imagination system."""
        
        base_status = self.simulation_agent.get_imagination_status()
        
        # Add ARC-specific status
        arc_status = {
            'arc_imagination': {
                'current_game': self.current_game_id,
                'actions_taken': self.action_count,
                'simulation_mode': True,
                'frame_analysis_cached': len(self.frame_analysis_cache),
                'game_states_tracked': len(self.game_state_history)
            }
        }
        
        return {**base_status, **arc_status}
