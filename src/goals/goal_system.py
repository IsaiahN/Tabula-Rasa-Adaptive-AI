"""
Goal Invention System - Autonomous goal generation and management.

This module implements the phased approach to goal generation:
- Phase 1: Fixed survival goals
- Phase 2: Template-based goals  
- Phase 3: Emergent goal discovery (future)
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from enum import Enum
import logging
from dataclasses import dataclass
import time

from core.data_models import Goal, AgentState

logger = logging.getLogger(__name__)


class GoalPhase(Enum):
    """Goal system development phases."""
    SURVIVAL = "survival"
    TEMPLATE = "template"
    EMERGENT = "emergent"


@dataclass
class GoalTemplate:
    """Template for generating parameterized goals."""
    name: str
    description: str
    parameter_ranges: Dict[str, Tuple[float, float]]
    achievement_criteria: str
    difficulty_level: float


class SurvivalGoals:
    """Phase 1: Hard-coded survival goals."""
    
    def __init__(self):
        self.goal_definitions = {
            "maintain_energy": {
                "description": "Keep energy above 50%",
                "threshold": 50.0,
                "priority": 1.0
            },
            "find_food": {
                "description": "Locate and consume food sources",
                "success_reward": 1.0,
                "priority": 0.8
            },
            "avoid_death": {
                "description": "Prevent energy from reaching zero",
                "threshold": 10.0,
                "priority": 1.0
            }
        }
        
    def get_active_goals(self, agent_state: AgentState) -> List[Goal]:
        """Generate survival goals based on current agent state."""
        goals = []
        
        # Energy maintenance goal
        if agent_state.energy < 70.0:  # Activate when energy gets low
            goal = Goal(
                target_state_cluster=torch.tensor([agent_state.energy + 20.0]),
                achievement_radius=10.0,
                success_rate=0.0,
                learning_progress_history=[],
                creation_timestamp=int(time.time()),
                goal_id="maintain_energy",
                goal_type="survival"
            )
            goals.append(goal)
            
        # Food seeking goal
        goal = Goal(
            target_state_cluster=torch.tensor([1.0]),  # Binary: found food
            achievement_radius=0.1,
            success_rate=0.0,
            learning_progress_history=[],
            creation_timestamp=int(time.time()),
            goal_id="find_food",
            goal_type="survival"
        )
        goals.append(goal)
        
        # Death avoidance goal (always active)
        goal = Goal(
            target_state_cluster=torch.tensor([agent_state.energy]),
            achievement_radius=5.0,
            success_rate=0.0,
            learning_progress_history=[],
            creation_timestamp=int(time.time()),
            goal_id="avoid_death",
            goal_type="survival"
        )
        goals.append(goal)
        
        return goals
        
    def evaluate_achievement(self, goal: Goal, agent_state: AgentState, action_result: Dict) -> bool:
        """Evaluate if a survival goal has been achieved."""
        if goal.goal_id == "maintain_energy":
            return agent_state.energy >= 50.0
        elif goal.goal_id == "find_food":
            return action_result.get("food_collected", False)
        elif goal.goal_id == "avoid_death":
            return agent_state.energy > 10.0
        else:
            return False


class TemplateGoals:
    """Phase 2: Template-based goal generation."""
    
    def __init__(self, environment_bounds: Tuple[float, float, float, float] = (-10, 10, -10, 10)):
        self.environment_bounds = environment_bounds  # (min_x, max_x, min_y, max_y)
        
        self.templates = {
            "reach_location": GoalTemplate(
                name="reach_location",
                description="Navigate to specific coordinates",
                parameter_ranges={
                    "target_x": (environment_bounds[0], environment_bounds[1]),
                    "target_y": (environment_bounds[2], environment_bounds[3])
                },
                achievement_criteria="distance < radius",
                difficulty_level=0.5
            ),
            "explore_area": GoalTemplate(
                name="explore_area",
                description="Visit a specific region of the environment",
                parameter_ranges={
                    "center_x": (environment_bounds[0], environment_bounds[1]),
                    "center_y": (environment_bounds[2], environment_bounds[3]),
                    "radius": (2.0, 8.0)
                },
                achievement_criteria="time_in_area > threshold",
                difficulty_level=0.3
            ),
            "maintain_distance": GoalTemplate(
                name="maintain_distance",
                description="Stay within certain distance of a point",
                parameter_ranges={
                    "anchor_x": (environment_bounds[0], environment_bounds[1]),
                    "anchor_y": (environment_bounds[2], environment_bounds[3]),
                    "max_distance": (3.0, 15.0)
                },
                achievement_criteria="distance < max_distance for duration",
                difficulty_level=0.4
            )
        }
        
        # Goal generation state
        self.active_template_goals = []
        self.goal_success_history = {}
        
    def generate_goal_from_template(self, template_name: str) -> Goal:
        """Generate a specific goal instance from a template."""
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
            
        template = self.templates[template_name]
        
        # Sample parameters from ranges
        parameters = {}
        for param_name, (min_val, max_val) in template.parameter_ranges.items():
            parameters[param_name] = np.random.uniform(min_val, max_val)
            
        # Create goal with sampled parameters
        if template_name == "reach_location":
            target_state = torch.tensor([parameters["target_x"], parameters["target_y"]])
            achievement_radius = 2.0
        elif template_name == "explore_area":
            target_state = torch.tensor([
                parameters["center_x"], 
                parameters["center_y"], 
                parameters["radius"]
            ])
            achievement_radius = parameters["radius"]
        elif template_name == "maintain_distance":
            target_state = torch.tensor([
                parameters["anchor_x"], 
                parameters["anchor_y"], 
                parameters["max_distance"]
            ])
            achievement_radius = parameters["max_distance"]
        else:
            target_state = torch.zeros(2)
            achievement_radius = 1.0
            
        goal = Goal(
            target_state_cluster=target_state,
            achievement_radius=achievement_radius,
            success_rate=0.0,
            learning_progress_history=[],
            creation_timestamp=int(time.time()),
            goal_id=f"{template_name}_{int(time.time())}",
            goal_type="template"
        )
        
        # Store parameters in goal for evaluation
        goal.parameters = parameters
        goal.template_name = template_name
        
        return goal
        
    def get_active_goals(self, agent_state: AgentState, max_goals: int = 3) -> List[Goal]:
        """Get active template-based goals."""
        # Remove completed or failed goals
        self._cleanup_goals()
        
        # Generate new goals if needed
        while len(self.active_template_goals) < max_goals:
            template_name = np.random.choice(list(self.templates.keys()))
            new_goal = self.generate_goal_from_template(template_name)
            self.active_template_goals.append(new_goal)
            logger.info(f"Generated new goal: {new_goal.goal_id}")
            
        return self.active_template_goals.copy()
        
    def evaluate_achievement(self, goal: Goal, agent_state: AgentState, action_result: Dict) -> bool:
        """Evaluate if a template goal has been achieved."""
        if not hasattr(goal, 'template_name'):
            return False
            
        current_pos = agent_state.position[:2]  # x, y coordinates
        
        if goal.template_name == "reach_location":
            target_pos = goal.target_state_cluster[:2]
            distance = torch.norm(current_pos - target_pos)
            return distance < goal.achievement_radius
            
        elif goal.template_name == "explore_area":
            center_pos = goal.target_state_cluster[:2]
            radius = goal.target_state_cluster[2]
            distance = torch.norm(current_pos - center_pos)
            
            # Track time in area (simplified - would need proper state tracking)
            if distance < radius:
                if not hasattr(goal, 'time_in_area'):
                    goal.time_in_area = 0
                goal.time_in_area += 1
                return goal.time_in_area > 50  # 50 steps in area
            return False
            
        elif goal.template_name == "maintain_distance":
            anchor_pos = goal.target_state_cluster[:2]
            max_distance = goal.target_state_cluster[2]
            distance = torch.norm(current_pos - anchor_pos)
            
            # Track time within distance
            if distance < max_distance:
                if not hasattr(goal, 'time_in_range'):
                    goal.time_in_range = 0
                goal.time_in_range += 1
                return goal.time_in_range > 100  # 100 steps in range
            else:
                goal.time_in_range = 0  # Reset if out of range
            return False
            
        return False
        
    def _cleanup_goals(self):
        """Remove old or completed goals."""
        current_time = int(time.time())
        
        # Remove goals older than 5 minutes or with very low success rate
        self.active_template_goals = [
            goal for goal in self.active_template_goals
            if (current_time - goal.creation_timestamp < 300 and 
                goal.success_rate < 0.9)  # Retire if 90% success rate
        ]


class EmergentGoals:
    """Phase 3: Emergent goal discovery (future implementation)."""
    
    def __init__(self):
        self.experience_clusters = []
        self.goal_candidates = []
        
    def get_active_goals(self, agent_state: AgentState) -> List[Goal]:
        """Placeholder for emergent goal discovery."""
        # Future implementation: cluster high-LP experiences
        return []
        
    def evaluate_achievement(self, goal: Goal, agent_state: AgentState, action_result: Dict) -> bool:
        """Placeholder for emergent goal evaluation."""
        return False


class GoalInventionSystem:
    """
    Main goal invention and management system.
    
    Implements phased approach starting with survival goals.
    """
    
    def __init__(
        self, 
        phase: GoalPhase = GoalPhase.SURVIVAL,
        environment_bounds: Tuple[float, float, float, float] = (-10, 10, -10, 10)
    ):
        self.current_phase = phase
        self.environment_bounds = environment_bounds
        
        # Initialize phase-specific goal systems
        self.survival_goals = SurvivalGoals()
        self.template_goals = TemplateGoals(environment_bounds)
        self.emergent_goals = EmergentGoals()
        
        # Goal tracking
        self.goal_history = []
        self.achievement_stats = {}
        
        # Phase transition criteria
        self.phase_transition_criteria = {
            GoalPhase.SURVIVAL: {
                "min_survival_rate": 0.8,
                "min_episodes": 10
            },
            GoalPhase.TEMPLATE: {
                "min_goal_achievement_rate": 0.7,
                "min_goals_completed": 20
            }
        }
        
        logger.info(f"Goal system initialized in {phase.value} phase")
        
    def get_active_goals(self, agent_state: AgentState) -> List[Goal]:
        """Get active goals based on current phase."""
        if self.current_phase == GoalPhase.SURVIVAL:
            return self.survival_goals.get_active_goals(agent_state)
        elif self.current_phase == GoalPhase.TEMPLATE:
            # Combine survival and template goals
            survival = self.survival_goals.get_active_goals(agent_state)
            template = self.template_goals.get_active_goals(agent_state, max_goals=2)
            return survival + template
        elif self.current_phase == GoalPhase.EMERGENT:
            # All goal types active
            survival = self.survival_goals.get_active_goals(agent_state)
            template = self.template_goals.get_active_goals(agent_state, max_goals=1)
            emergent = self.emergent_goals.get_active_goals(agent_state)
            return survival + template + emergent
        else:
            return []
            
    def evaluate_goal_achievement(
        self, 
        goal: Goal, 
        agent_state: AgentState, 
        action_result: Dict
    ) -> bool:
        """Evaluate if a goal has been achieved."""
        if goal.goal_type == "survival":
            achieved = self.survival_goals.evaluate_achievement(goal, agent_state, action_result)
        elif goal.goal_type == "template":
            achieved = self.template_goals.evaluate_achievement(goal, agent_state, action_result)
        elif goal.goal_type == "emergent":
            achieved = self.emergent_goals.evaluate_achievement(goal, agent_state, action_result)
        else:
            achieved = False
            
        # Update goal statistics
        if achieved:
            self._update_goal_success(goal)
            logger.debug(f"Goal achieved: {goal.goal_id}")
            
        return achieved
        
    def update_goal_progress(self, goal: Goal, learning_progress: float):
        """Update goal with learning progress information."""
        goal.learning_progress_history.append(learning_progress)
        
        # Keep history bounded
        if len(goal.learning_progress_history) > 100:
            goal.learning_progress_history = goal.learning_progress_history[-50:]
            
    def check_phase_transition(self) -> bool:
        """Check if conditions are met to advance to next phase."""
        if self.current_phase == GoalPhase.SURVIVAL:
            # Check if ready for template phase
            criteria = self.phase_transition_criteria[GoalPhase.SURVIVAL]
            survival_rate = self._compute_survival_rate()
            episode_count = len(self.goal_history)
            
            if (survival_rate >= criteria["min_survival_rate"] and 
                episode_count >= criteria["min_episodes"]):
                self._transition_to_phase(GoalPhase.TEMPLATE)
                return True
                
        elif self.current_phase == GoalPhase.TEMPLATE:
            # Check if ready for emergent phase
            criteria = self.phase_transition_criteria[GoalPhase.TEMPLATE]
            achievement_rate = self._compute_goal_achievement_rate()
            goals_completed = sum(self.achievement_stats.values())
            
            if (achievement_rate >= criteria["min_goal_achievement_rate"] and
                goals_completed >= criteria["min_goals_completed"]):
                self._transition_to_phase(GoalPhase.EMERGENT)
                return True
                
        return False
        
    def _transition_to_phase(self, new_phase: GoalPhase):
        """Transition to a new goal system phase."""
        logger.info(f"Transitioning from {self.current_phase.value} to {new_phase.value} phase")
        self.current_phase = new_phase
        
    def _update_goal_success(self, goal: Goal):
        """Update success statistics for a goal."""
        if goal.goal_id not in self.achievement_stats:
            self.achievement_stats[goal.goal_id] = 0
        self.achievement_stats[goal.goal_id] += 1
        
        # Update goal success rate
        total_attempts = len([g for g in self.goal_history if g.goal_id == goal.goal_id])
        if total_attempts > 0:
            goal.success_rate = self.achievement_stats[goal.goal_id] / total_attempts
            
    def _compute_survival_rate(self) -> float:
        """Compute survival rate from recent episodes."""
        if not self.goal_history:
            return 0.0
            
        # Count episodes where agent didn't die
        recent_episodes = self.goal_history[-20:]  # Last 20 episodes
        survival_count = sum(1 for episode in recent_episodes 
                           if not episode.get('died', False))
        
        return survival_count / len(recent_episodes)
        
    def _compute_goal_achievement_rate(self) -> float:
        """Compute overall goal achievement rate."""
        if not self.achievement_stats:
            return 0.0
            
        total_achievements = sum(self.achievement_stats.values())
        total_attempts = len(self.goal_history)
        
        return total_achievements / max(total_attempts, 1)
        
    def get_goal_metrics(self) -> Dict[str, float]:
        """Get goal system performance metrics."""
        return {
            'current_phase': self.current_phase.value,
            'survival_rate': self._compute_survival_rate(),
            'goal_achievement_rate': self._compute_goal_achievement_rate(),
            'total_goals_attempted': len(self.goal_history),
            'total_goals_achieved': sum(self.achievement_stats.values()),
            'active_goal_types': len(set(g.get('goal_type', 'unknown') if isinstance(g, dict) else g.goal_type for g in self.goal_history[-10:]))
        }
        
    def reset_episode(self, episode_data: Dict):
        """Reset for new episode and record episode data."""
        self.goal_history.append(episode_data)
        
        # Keep history bounded
        if len(self.goal_history) > 1000:
            self.goal_history = self.goal_history[-500:]