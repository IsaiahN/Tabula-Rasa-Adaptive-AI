"""
Multi-Agent Environment - Phase 3 implementation.

This module implements a shared environment where multiple agents can interact,
compete for resources, and potentially develop social behaviors.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import time

from core.data_models import SensoryInput, AgentState
from environment.survival_environment import SurvivalEnvironment

logger = logging.getLogger(__name__)


@dataclass
class MultiAgentObservation:
    """Observation for a single agent in multi-agent environment."""
    agent_id: str
    sensory_input: SensoryInput
    other_agents_visible: List[Dict[str, Any]]
    shared_resources: Dict[str, Any]
    social_signals: Dict[str, float]


class MultiAgentSurvivalEnvironment:
    """
    Multi-agent survival environment with shared resources and social dynamics.
    
    Features:
    - Multiple agents in shared 3D space
    - Competition for limited food sources
    - Visual observation of other agents
    - Potential for cooperation/competition emergence
    """
    
    def __init__(
        self,
        num_agents: int = 2,
        world_size: Tuple[int, int, int] = (20, 20, 5),
        num_food_sources: int = 3,  # Reduced for competition
        food_respawn_time: float = 60.0,  # Slower respawn
        food_energy_value: float = 15.0,
        agent_visibility_range: float = 8.0,
        social_interaction_range: float = 3.0,
        complexity_level: int = 1
    ):
        self.num_agents = num_agents
        self.world_size = world_size
        self.num_food_sources = num_food_sources
        self.food_respawn_time = food_respawn_time
        self.food_energy_value = food_energy_value
        self.agent_visibility_range = agent_visibility_range
        self.social_interaction_range = social_interaction_range
        self.complexity_level = complexity_level
        
        # Create base survival environment
        self.base_env = SurvivalEnvironment(
            world_size=world_size,
            num_food_sources=num_food_sources,
            food_respawn_time=food_respawn_time,
            food_energy_value=food_energy_value,
            complexity_level=complexity_level,
            physics_enabled=True
        )
        
        # Multi-agent specific state
        self.agents = {}  # agent_id -> agent_info
        self.agent_positions = {}  # agent_id -> position
        self.agent_last_actions = {}  # agent_id -> last_action
        self.social_interaction_history = []
        
        # Competition metrics
        self.resource_competition_events = []
        self.cooperation_events = []
        
        # Initialize agents
        self._initialize_agents()
        
        logger.info(f"Multi-agent environment initialized: {num_agents} agents, {world_size} world")
        
    def _initialize_agents(self):
        """Initialize agent positions and states."""
        for i in range(self.num_agents):
            agent_id = f"agent_{i}"
            
            # Random spawn positions (avoiding overlap)
            spawn_attempts = 0
            while spawn_attempts < 100:
                position = torch.tensor([
                    np.random.uniform(1, self.world_size[0] - 1),
                    np.random.uniform(1, self.world_size[1] - 1),
                    1.0  # Ground level
                ])
                
                # Check distance from other agents
                too_close = False
                for other_pos in self.agent_positions.values():
                    if torch.norm(position - other_pos) < 3.0:
                        too_close = True
                        break
                        
                if not too_close:
                    break
                    
                spawn_attempts += 1
                
            self.agent_positions[agent_id] = position
            self.agent_last_actions[agent_id] = torch.zeros(3)
            
            self.agents[agent_id] = {
                'spawn_position': position.clone(),
                'total_food_collected': 0,
                'social_interactions': 0,
                'cooperation_score': 0.0,
                'competition_score': 0.0
            }
            
        logger.info(f"Initialized {len(self.agents)} agents at positions: {self.agent_positions}")
        
    def reset(self):
        """Reset the multi-agent environment."""
        # Reset base environment
        self.base_env.reset()
        
        # Reset agent positions
        self._initialize_agents()
        
        # Clear interaction history
        self.social_interaction_history.clear()
        self.resource_competition_events.clear()
        self.cooperation_events.clear()
        
        logger.info("Multi-agent environment reset")
        
    def step(
        self, 
        agents_dict: Dict[str, Any], 
        actions_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, MultiAgentObservation], Dict[str, Dict], bool]:
        """
        Execute one step for all agents simultaneously.
        
        Args:
            agents_dict: Dictionary of agent_id -> agent_object
            actions_dict: Dictionary of agent_id -> action_tensor
            
        Returns:
            observations: Dictionary of agent_id -> MultiAgentObservation
            action_results: Dictionary of agent_id -> action_result
            done: Whether episode is complete
        """
        observations = {}
        action_results = {}
        
        # Update agent positions based on actions
        for agent_id, action in actions_dict.items():
            if agent_id in self.agent_positions:
                # Simple movement model
                movement = action[:3] * 0.5  # Scale movement
                new_position = self.agent_positions[agent_id] + movement
                
                # Clamp to world bounds
                new_position = torch.clamp(
                    new_position,
                    torch.tensor([0.0, 0.0, 0.5]),
                    torch.tensor([self.world_size[0], self.world_size[1], self.world_size[2]])
                )
                
                self.agent_positions[agent_id] = new_position
                self.agent_last_actions[agent_id] = action
                
        # Check for resource interactions
        food_interactions = self._check_food_interactions()
        
        # Check for social interactions
        social_interactions = self._check_social_interactions()
        
        # Generate observations for each agent
        for agent_id, agent in agents_dict.items():
            if agent_id in self.agent_positions:
                # Get base sensory input from survival environment
                agent_state = agent.get_agent_state()
                agent_state.position = self.agent_positions[agent_id]
                
                base_sensory, base_result, base_done = self.base_env.step(agent, actions_dict[agent_id])
                
                # Enhance with multi-agent information
                other_agents = self._get_visible_agents(agent_id)
                shared_resources = self._get_shared_resource_info(agent_id)
                social_signals = self._get_social_signals(agent_id)
                
                observation = MultiAgentObservation(
                    agent_id=agent_id,
                    sensory_input=base_sensory,
                    other_agents_visible=other_agents,
                    shared_resources=shared_resources,
                    social_signals=social_signals
                )
                
                observations[agent_id] = observation
                
                # Update action results with multi-agent events
                result = base_result.copy()
                result.update({
                    'food_competition': food_interactions.get(agent_id, {}),
                    'social_interaction': social_interactions.get(agent_id, {}),
                    'other_agents_nearby': len(other_agents)
                })
                
                action_results[agent_id] = result
                
        # Update environment state
        self.base_env.update_food_sources()
        
        # Check if episode is done (all agents dead or time limit)
        done = self._check_episode_done(agents_dict)
        
        return observations, action_results, done
        
    def _check_food_interactions(self) -> Dict[str, Dict]:
        """Check for food resource competition between agents."""
        food_interactions = {}
        
        for food_pos in self.base_env.food_sources:
            if not self.base_env.food_available[tuple(food_pos.int().tolist())]:
                continue  # Food already consumed
                
            # Find agents near this food source
            nearby_agents = []
            for agent_id, agent_pos in self.agent_positions.items():
                distance = torch.norm(agent_pos[:2] - food_pos[:2])
                if distance < 2.0:  # Within collection range
                    nearby_agents.append((agent_id, distance))
                    
            # If multiple agents compete for same food
            if len(nearby_agents) > 1:
                # Sort by distance - closest gets the food
                nearby_agents.sort(key=lambda x: x[1])
                winner_id = nearby_agents[0][0]
                
                # Record competition event
                competition_event = {
                    'timestamp': time.time(),
                    'food_position': food_pos,
                    'competing_agents': [aid for aid, _ in nearby_agents],
                    'winner': winner_id
                }
                self.resource_competition_events.append(competition_event)
                
                # Update agent scores
                self.agents[winner_id]['total_food_collected'] += 1
                self.agents[winner_id]['competition_score'] += 1.0
                
                for agent_id, _ in nearby_agents:
                    if agent_id != winner_id:
                        self.agents[agent_id]['competition_score'] += 0.1  # Participation
                        
                    food_interactions[agent_id] = {
                        'competed_for_food': True,
                        'won_competition': agent_id == winner_id,
                        'competitors': len(nearby_agents) - 1
                    }
                    
        return food_interactions
        
    def _check_social_interactions(self) -> Dict[str, Dict]:
        """Check for social interactions between agents."""
        social_interactions = {}
        
        agent_ids = list(self.agent_positions.keys())
        
        for i, agent_id_1 in enumerate(agent_ids):
            for j, agent_id_2 in enumerate(agent_ids[i+1:], i+1):
                pos_1 = self.agent_positions[agent_id_1]
                pos_2 = self.agent_positions[agent_id_2]
                distance = torch.norm(pos_1 - pos_2)
                
                if distance < self.social_interaction_range:
                    # Record social interaction
                    interaction = {
                        'timestamp': time.time(),
                        'agents': [agent_id_1, agent_id_2],
                        'distance': float(distance),
                        'interaction_type': self._classify_interaction(agent_id_1, agent_id_2)
                    }
                    
                    self.social_interaction_history.append(interaction)
                    
                    # Update agent interaction counts
                    self.agents[agent_id_1]['social_interactions'] += 1
                    self.agents[agent_id_2]['social_interactions'] += 1
                    
                    # Create interaction results
                    for agent_id in [agent_id_1, agent_id_2]:
                        if agent_id not in social_interactions:
                            social_interactions[agent_id] = {
                                'interacted_with': [],
                                'interaction_types': []
                            }
                            
                        other_agent = agent_id_2 if agent_id == agent_id_1 else agent_id_1
                        social_interactions[agent_id]['interacted_with'].append(other_agent)
                        social_interactions[agent_id]['interaction_types'].append(interaction['interaction_type'])
                        
        return social_interactions
        
    def _classify_interaction(self, agent_id_1: str, agent_id_2: str) -> str:
        """Classify the type of social interaction between two agents."""
        # Simple heuristic based on recent actions and positions
        action_1 = self.agent_last_actions[agent_id_1]
        action_2 = self.agent_last_actions[agent_id_2]
        
        # Check if agents are moving towards or away from each other
        pos_1 = self.agent_positions[agent_id_1]
        pos_2 = self.agent_positions[agent_id_2]
        
        # Direction vectors
        dir_1_to_2 = (pos_2 - pos_1)[:2]  # Only x, y
        dir_2_to_1 = (pos_1 - pos_2)[:2]
        
        # Movement directions
        move_1 = action_1[:2]
        move_2 = action_2[:2]
        
        # Dot products to see if moving towards each other
        approach_1 = torch.dot(move_1, dir_1_to_2)
        approach_2 = torch.dot(move_2, dir_2_to_1)
        
        if approach_1 > 0 and approach_2 > 0:
            return "mutual_approach"
        elif approach_1 < 0 and approach_2 < 0:
            return "mutual_avoidance"
        elif abs(approach_1) < 0.1 and abs(approach_2) < 0.1:
            return "neutral_proximity"
        else:
            return "mixed_behavior"
            
    def _get_visible_agents(self, observer_id: str) -> List[Dict[str, Any]]:
        """Get information about other agents visible to the observer."""
        visible_agents = []
        observer_pos = self.agent_positions[observer_id]
        
        for agent_id, agent_pos in self.agent_positions.items():
            if agent_id == observer_id:
                continue
                
            distance = torch.norm(observer_pos - agent_pos)
            if distance <= self.agent_visibility_range:
                # Calculate relative position and movement
                relative_pos = agent_pos - observer_pos
                last_action = self.agent_last_actions[agent_id]
                
                visible_agent_info = {
                    'agent_id': agent_id,
                    'relative_position': relative_pos.tolist(),
                    'distance': float(distance),
                    'last_movement': last_action[:3].tolist(),
                    'estimated_energy': 'unknown',  # Could be inferred from behavior
                    'interaction_history': self._get_interaction_history(observer_id, agent_id)
                }
                
                visible_agents.append(visible_agent_info)
                
        return visible_agents
        
    def _get_shared_resource_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about shared resources visible to the agent."""
        agent_pos = self.agent_positions[agent_id]
        
        visible_food = []
        for food_pos in self.base_env.food_sources:
            distance = torch.norm(agent_pos[:2] - food_pos[:2])
            if distance <= self.agent_visibility_range:
                is_available = self.base_env.food_available.get(tuple(food_pos.int().tolist()), False)
                
                visible_food.append({
                    'position': food_pos.tolist(),
                    'distance': float(distance),
                    'available': is_available,
                    'estimated_competition': self._estimate_food_competition(food_pos)
                })
                
        return {
            'visible_food_sources': visible_food,
            'total_food_in_environment': len(self.base_env.food_sources),
            'food_scarcity_level': self._calculate_food_scarcity()
        }
        
    def _get_social_signals(self, agent_id: str) -> Dict[str, float]:
        """Get social signals and reputation information for the agent."""
        agent_info = self.agents[agent_id]
        
        # Calculate social metrics
        total_interactions = agent_info['social_interactions']
        cooperation_ratio = agent_info['cooperation_score'] / max(total_interactions, 1)
        competition_ratio = agent_info['competition_score'] / max(total_interactions, 1)
        
        return {
            'cooperation_reputation': cooperation_ratio,
            'competition_reputation': competition_ratio,
            'social_activity_level': min(total_interactions / 100.0, 1.0),
            'resource_success_rate': agent_info['total_food_collected'] / max(self.step_count, 1) if hasattr(self, 'step_count') else 0.0
        }
        
    def _get_interaction_history(self, agent_1: str, agent_2: str) -> Dict[str, Any]:
        """Get interaction history between two specific agents."""
        interactions = []
        for interaction in self.social_interaction_history[-10:]:  # Last 10 interactions
            if agent_1 in interaction['agents'] and agent_2 in interaction['agents']:
                interactions.append({
                    'type': interaction['interaction_type'],
                    'distance': interaction['distance'],
                    'recency': time.time() - interaction['timestamp']
                })
                
        return {
            'recent_interactions': interactions,
            'total_interactions': len(interactions),
            'dominant_interaction_type': self._get_dominant_interaction_type(interactions)
        }
        
    def _get_dominant_interaction_type(self, interactions: List[Dict]) -> str:
        """Get the most common interaction type from a list of interactions."""
        if not interactions:
            return "none"
            
        type_counts = {}
        for interaction in interactions:
            interaction_type = interaction['type']
            type_counts[interaction_type] = type_counts.get(interaction_type, 0) + 1
            
        return max(type_counts, key=type_counts.get)
        
    def _estimate_food_competition(self, food_pos: torch.Tensor) -> float:
        """Estimate competition level for a specific food source."""
        nearby_agents = 0
        for agent_pos in self.agent_positions.values():
            distance = torch.norm(agent_pos[:2] - food_pos[:2])
            if distance <= self.agent_visibility_range:
                nearby_agents += 1
                
        return min(nearby_agents / 3.0, 1.0)  # Normalize to 0-1
        
    def _calculate_food_scarcity(self) -> float:
        """Calculate overall food scarcity in the environment."""
        # Check if base_env has food_available attribute
        if hasattr(self.base_env, 'food_available'):
            available_food = sum(1 for available in self.base_env.food_available.values() if available)
        else:
            # Fallback: assume all food sources are available
            available_food = len(self.base_env.food_sources)
            
        total_food_sources = len(self.base_env.food_sources)
        
        if total_food_sources == 0:
            return 1.0
            
        scarcity = 1.0 - (available_food / total_food_sources)
        return scarcity
        
    def _check_episode_done(self, agents_dict: Dict[str, Any]) -> bool:
        """Check if the multi-agent episode should end."""
        # Episode ends if all agents are dead or after maximum steps
        living_agents = 0
        for agent in agents_dict.values():
            if agent.get_agent_state().energy > 0:
                living_agents += 1
                
        return living_agents == 0
        
    def get_environment_metrics(self) -> Dict[str, Any]:
        """Get comprehensive environment metrics."""
        return {
            'total_agents': self.num_agents,
            'resource_competition_events': len(self.resource_competition_events),
            'cooperation_events': len(self.cooperation_events),
            'social_interactions': len(self.social_interaction_history),
            'food_scarcity': self._calculate_food_scarcity(),
            'agent_metrics': {
                agent_id: {
                    'food_collected': info['total_food_collected'],
                    'social_interactions': info['social_interactions'],
                    'cooperation_score': info['cooperation_score'],
                    'competition_score': info['competition_score']
                }
                for agent_id, info in self.agents.items()
            }
        }
        
    def increase_complexity(self):
        """Increase environment complexity for more challenging scenarios."""
        self.complexity_level += 1
        
        # Reduce food sources to increase competition
        if self.num_food_sources > 1:
            self.num_food_sources -= 1
            
        # Increase food respawn time
        self.food_respawn_time *= 1.5
        
        # Update base environment
        self.base_env.complexity_level = self.complexity_level
        self.base_env.increase_complexity()
        
        logger.info(f"Multi-agent environment complexity increased to level {self.complexity_level}")
