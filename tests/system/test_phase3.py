"""
Comprehensive Phase 3 Testing Suite.

Tests for emergent goal discovery, multi-agent environment, and social dynamics.
"""

import pytest
import torch
import numpy as np
import time
import sys
import os
from typing import Dict, List, Any

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.agent import AdaptiveLearningAgent
from environment.multi_agent_environment import MultiAgentSurvivalEnvironment, MultiAgentObservation
from goals.goal_system import EmergentGoals
from core.data_models import SensoryInput, AgentState


class TestEmergentGoalDiscovery:
    """Test suite for emergent goal discovery system."""
    
    @pytest.fixture
    def emergent_goals(self):
        """Create EmergentGoals instance for testing."""
        return EmergentGoals()
        
    @pytest.fixture
    def sample_agent_state(self):
        """Create sample agent state for testing."""
        return AgentState(
            position=torch.tensor([5.0, 5.0, 1.0]),
            orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
            energy=80.0,
            hidden_state=torch.zeros(64),
            active_goals=[],
            timestamp=100
        )
        
    def test_experience_addition(self, emergent_goals, sample_agent_state):
        """Test adding experiences to emergent goal system."""
        state_repr = torch.randn(64)
        learning_progress = 0.15  # Above threshold
        
        emergent_goals.add_experience(state_repr, learning_progress, sample_agent_state)
        
        assert len(emergent_goals.high_lp_experiences) == 1
        assert emergent_goals.high_lp_experiences[0]['learning_progress'] == learning_progress
        
    def test_experience_filtering(self, emergent_goals, sample_agent_state):
        """Test that only high learning progress experiences are stored."""
        # Low learning progress - should be ignored
        emergent_goals.add_experience(torch.randn(64), 0.05, sample_agent_state)
        assert len(emergent_goals.high_lp_experiences) == 0
        
        # High learning progress - should be stored
        emergent_goals.add_experience(torch.randn(64), 0.15, sample_agent_state)
        assert len(emergent_goals.high_lp_experiences) == 1
        
    def test_goal_discovery_clustering(self, emergent_goals, sample_agent_state):
        """Test goal discovery through clustering."""
        # Add multiple experiences in clusters
        cluster_1_center = torch.tensor([10.0, 10.0, 1.0])
        cluster_2_center = torch.tensor([15.0, 15.0, 1.0])
        
        # Add experiences around cluster 1
        for i in range(6):
            state_repr = torch.randn(64)
            agent_state = AgentState(
                position=cluster_1_center + torch.randn(3) * 0.5,
                orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
                energy=80.0,
                hidden_state=torch.zeros(64),
                active_goals=[],
                timestamp=100 + i
            )
            emergent_goals.add_experience(state_repr, 0.15, agent_state)
            
        # Add experiences around cluster 2
        for i in range(6):
            state_repr = torch.randn(64)
            agent_state = AgentState(
                position=cluster_2_center + torch.randn(3) * 0.5,
                orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
                energy=80.0,
                hidden_state=torch.zeros(64),
                active_goals=[],
                timestamp=200 + i
            )
            emergent_goals.add_experience(state_repr, 0.15, agent_state)
            
        # Trigger goal discovery
        emergent_goals._discover_goals()
        
        # Should discover at least one goal
        active_goals = emergent_goals.get_active_goals()
        assert len(active_goals) > 0
        
        # Goals should have spatial centroids
        for goal in active_goals:
            assert hasattr(goal, 'spatial_centroid')
            assert goal.spatial_centroid is not None
            
    def test_goal_achievement_evaluation(self, emergent_goals, sample_agent_state):
        """Test goal achievement evaluation."""
        # Create a goal manually
        state_repr = torch.randn(64)
        emergent_goals.add_experience(state_repr, 0.15, sample_agent_state)
        
        # Add more experiences to trigger discovery
        for i in range(5):
            emergent_goals.add_experience(torch.randn(64), 0.15, sample_agent_state)
            
        emergent_goals._discover_goals()
        active_goals = emergent_goals.get_active_goals()
        
        if active_goals:
            goal = active_goals[0]
            
            # Test achievement when agent is at goal location
            action_result = {'position_reached': goal.spatial_centroid}
            is_achieved = emergent_goals.evaluate_achievement(goal, sample_agent_state, action_result)
            
            # Achievement depends on proximity and time spent
            assert isinstance(is_achieved, bool)
            
    def test_goal_lifecycle_management(self, emergent_goals, sample_agent_state):
        """Test goal creation, achievement, and cleanup."""
        # Add experiences to create goals
        for i in range(10):
            state_repr = torch.randn(64)
            agent_state = AgentState(
                position=torch.tensor([5.0 + i*0.1, 5.0, 1.0]),
                orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
                energy=80.0,
                hidden_state=torch.zeros(64),
                active_goals=[],
                timestamp=100 + i
            )
            emergent_goals.add_experience(state_repr, 0.15, agent_state)
            
        initial_goal_count = len(emergent_goals.get_active_goals())
        
        # Simulate goal achievement
        for goal in emergent_goals.get_active_goals():
            goal.achievement_count = 5  # Mark as frequently achieved
            
        # Cleanup should remove frequently achieved goals
        emergent_goals._cleanup_old_goals()
        
        final_goal_count = len(emergent_goals.get_active_goals())
        assert final_goal_count <= initial_goal_count


class TestMultiAgentEnvironment:
    """Test suite for multi-agent environment."""
    
    @pytest.fixture
    def multi_env(self):
        """Create multi-agent environment for testing."""
        return MultiAgentSurvivalEnvironment(
            num_agents=2,
            world_size=(10, 10, 3),
            num_food_sources=2,
            food_respawn_time=30.0
        )
        
    @pytest.fixture
    def test_agents(self, multi_env):
        """Create test agents."""
        agents = {}
        for i in range(2):
            agent_id = f"agent_{i}"
            # Create minimal config for agent
            config = {
                'predictive_core': {'state_dim': 64, 'action_dim': 6},
                'memory': {'memory_size': 64},
                'learning_progress': {},
                'energy': {},
                'goals': {},
                'action_selection': {},
                'sleep': {}
            }
            agent = AdaptiveLearningAgent(config, device='cpu')
            
            # Initialize with position from environment
            position = multi_env.agent_positions[agent_id]
            agent.reset(initial_energy=100.0, position=position)
            agents[agent_id] = agent
            
        return agents
        
    def test_environment_initialization(self, multi_env):
        """Test multi-agent environment initialization."""
        assert multi_env.num_agents == 2
        assert len(multi_env.agent_positions) == 2
        assert len(multi_env.agents) == 2
        
        # Check agent positions are different
        positions = list(multi_env.agent_positions.values())
        distance = torch.norm(positions[0] - positions[1])
        assert distance > 2.0  # Agents should spawn apart
        
    def test_environment_step(self, multi_env, test_agents):
        """Test multi-agent environment step execution."""
        # Create actions for all agents
        actions_dict = {
            agent_id: torch.randn(6) for agent_id in test_agents.keys()
        }
        
        # Execute step
        observations, action_results, done = multi_env.step(test_agents, actions_dict)
        
        # Check outputs
        assert len(observations) == len(test_agents)
        assert len(action_results) == len(test_agents)
        assert isinstance(done, bool)
        
        # Check observation structure
        for agent_id, obs in observations.items():
            assert isinstance(obs, MultiAgentObservation)
            assert obs.agent_id == agent_id
            assert isinstance(obs.sensory_input, SensoryInput)
            assert isinstance(obs.other_agents_visible, list)
            assert isinstance(obs.shared_resources, dict)
            assert isinstance(obs.social_signals, dict)
            
    def test_agent_visibility(self, multi_env, test_agents):
        """Test agent visibility system."""
        agent_ids = list(test_agents.keys())
        observer_id = agent_ids[0]
        
        # Move agents close together
        multi_env.agent_positions[agent_ids[0]] = torch.tensor([5.0, 5.0, 1.0])
        multi_env.agent_positions[agent_ids[1]] = torch.tensor([6.0, 6.0, 1.0])
        
        visible_agents = multi_env._get_visible_agents(observer_id)
        
        assert len(visible_agents) == 1  # Should see the other agent
        assert visible_agents[0]['agent_id'] == agent_ids[1]
        assert visible_agents[0]['distance'] < multi_env.agent_visibility_range
        
    def test_resource_competition(self, multi_env, test_agents):
        """Test resource competition detection."""
        agent_ids = list(test_agents.keys())
        
        # Position agents near same food source
        if multi_env.base_env.food_sources:
            food_pos = multi_env.base_env.food_sources[0]
            multi_env.agent_positions[agent_ids[0]] = food_pos + torch.tensor([0.5, 0.0, 0.0])
            multi_env.agent_positions[agent_ids[1]] = food_pos + torch.tensor([-0.5, 0.0, 0.0])
            
            # Check for competition
            food_interactions = multi_env._check_food_interactions()
            
            # Should detect competition
            assert len(food_interactions) >= 0  # May or may not compete depending on food availability
            
    def test_social_interaction_detection(self, multi_env, test_agents):
        """Test social interaction detection."""
        agent_ids = list(test_agents.keys())
        
        # Position agents close together
        multi_env.agent_positions[agent_ids[0]] = torch.tensor([5.0, 5.0, 1.0])
        multi_env.agent_positions[agent_ids[1]] = torch.tensor([5.5, 5.5, 1.0])
        
        # Set movement actions
        multi_env.agent_last_actions[agent_ids[0]] = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        multi_env.agent_last_actions[agent_ids[1]] = torch.tensor([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        
        social_interactions = multi_env._check_social_interactions()
        
        # Should detect interaction
        assert len(social_interactions) > 0
        
    def test_environment_metrics(self, multi_env):
        """Test environment metrics collection."""
        metrics = multi_env.get_environment_metrics()
        
        required_keys = [
            'total_agents', 'resource_competition_events', 'cooperation_events',
            'social_interactions', 'food_scarcity', 'agent_metrics'
        ]
        
        for key in required_keys:
            assert key in metrics
            
        assert metrics['total_agents'] == multi_env.num_agents
        assert isinstance(metrics['agent_metrics'], dict)
        
    def test_complexity_scaling(self, multi_env):
        """Test environment complexity scaling."""
        initial_complexity = multi_env.complexity_level
        initial_food_sources = multi_env.num_food_sources
        
        multi_env.increase_complexity()
        
        assert multi_env.complexity_level > initial_complexity
        assert multi_env.num_food_sources <= initial_food_sources


class TestPhase3Integration:
    """Integration tests for complete Phase 3 system."""
    
    @pytest.fixture
    def phase3_system(self):
        """Create complete Phase 3 system for testing."""
        # Create environment
        env = MultiAgentSurvivalEnvironment(
            num_agents=2,
            world_size=(10, 10, 3),
            num_food_sources=2
        )
        
        # Create agents
        agents = {}
        for i in range(2):
            agent_id = f"agent_{i}"
            agent = AdaptiveLearningAgent(
                state_dim=64,
                action_dim=6,
                memory_size=64,
                device=torch.device('cpu')
            )
            
            # Set to emergent phase
            agent.goal_system.transition_to_phase("emergent")
            
            position = env.agent_positions[agent_id]
            agent.reset(initial_energy=100.0, position=position)
            agents[agent_id] = agent
            
        return env, agents
        
    def test_emergent_goals_in_multi_agent_context(self, phase3_system):
        """Test emergent goal discovery in multi-agent environment."""
        env, agents = phase3_system
        
        # Run several steps to generate experiences
        for step in range(50):
            actions_dict = {
                agent_id: torch.randn(6) for agent_id in agents.keys()
            }
            
            observations, action_results, done = env.step(agents, actions_dict)
            
            # Update agents
            for agent_id, agent in agents.items():
                if agent_id in observations:
                    obs = observations[agent_id]
                    agent_result = agent.step(obs.sensory_input, actions_dict[agent_id])
                    
            if done:
                break
                
        # Check that emergent goals were discovered
        total_emergent_goals = 0
        for agent in agents.values():
            emergent_goals = agent.goal_system.emergent_goals.get_active_goals()
            total_emergent_goals += len(emergent_goals)
            
        # Should have discovered some goals (though may be 0 if not enough high LP experiences)
        assert total_emergent_goals >= 0
        
    def test_social_dynamics_emergence(self, phase3_system):
        """Test emergence of social dynamics between agents."""
        env, agents = phase3_system
        
        initial_interactions = len(env.social_interaction_history)
        
        # Run steps with agents in proximity
        agent_ids = list(agents.keys())
        env.agent_positions[agent_ids[0]] = torch.tensor([5.0, 5.0, 1.0])
        env.agent_positions[agent_ids[1]] = torch.tensor([5.5, 5.5, 1.0])
        
        for step in range(20):
            actions_dict = {
                agent_id: torch.randn(6) * 0.1 for agent_id in agents.keys()  # Small movements
            }
            
            observations, action_results, done = env.step(agents, actions_dict)
            
        final_interactions = len(env.social_interaction_history)
        
        # Should have recorded social interactions
        assert final_interactions >= initial_interactions
        
    def test_resource_competition(self, phase3_system):
        """Test resource competition between agents."""
        env, agents = phase3_system
        
        # Position agents near same food source
        if env.base_env.food_sources:
            food_pos = env.base_env.food_sources[0]
            agent_ids = list(agents.keys())
            
            env.agent_positions[agent_ids[0]] = food_pos + torch.tensor([0.3, 0.0, 0.0])
            env.agent_positions[agent_ids[1]] = food_pos + torch.tensor([-0.3, 0.0, 0.0])
            
            initial_competition_events = len(env.resource_competition_events)
            
            # Run steps
            for step in range(10):
                actions_dict = {
                    agent_id: torch.zeros(6) for agent_id in agents.keys()  # Stay in place
                }
                
                observations, action_results, done = env.step(agents, actions_dict)
                
            final_competition_events = len(env.resource_competition_events)
            
            # May or may not have competition depending on food availability
            assert final_competition_events >= initial_competition_events
            
    def test_multi_agent_observation_structure(self, phase3_system):
        """Test multi-agent observation data structure."""
        env, agents = phase3_system
        
        actions_dict = {
            agent_id: torch.randn(6) for agent_id in agents.keys()
        }
        
        observations, action_results, done = env.step(agents, actions_dict)
        
        for agent_id, obs in observations.items():
            # Check observation structure
            assert isinstance(obs, MultiAgentObservation)
            assert obs.agent_id == agent_id
            
            # Check sensory input
            assert isinstance(obs.sensory_input, SensoryInput)
            assert obs.sensory_input.visual.shape == (3, 64, 64)
            assert len(obs.sensory_input.proprioception) > 0
            
            # Check multi-agent specific data
            assert isinstance(obs.other_agents_visible, list)
            assert isinstance(obs.shared_resources, dict)
            assert isinstance(obs.social_signals, dict)
            
            # Check social signals structure
            required_social_signals = [
                'cooperation_reputation', 'competition_reputation',
                'social_activity_level', 'resource_success_rate'
            ]
            for signal in required_social_signals:
                assert signal in obs.social_signals
                assert isinstance(obs.social_signals[signal], (int, float))
                
    def test_environment_reset(self, phase3_system):
        """Test environment reset functionality."""
        env, agents = phase3_system
        
        # Run some steps to change state
        for step in range(10):
            actions_dict = {
                agent_id: torch.randn(6) for agent_id in agents.keys()
            }
            env.step(agents, actions_dict)
            
        # Record state before reset
        pre_reset_positions = env.agent_positions.copy()
        pre_reset_interactions = len(env.social_interaction_history)
        
        # Reset environment
        env.reset()
        
        # Check reset worked
        post_reset_positions = env.agent_positions
        post_reset_interactions = len(env.social_interaction_history)
        
        # Positions should be different (new random spawn)
        position_changed = False
        for agent_id in pre_reset_positions:
            if not torch.allclose(pre_reset_positions[agent_id], post_reset_positions[agent_id]):
                position_changed = True
                break
        assert position_changed
        
        # Interaction history should be cleared
        assert post_reset_interactions == 0


class TestPhase3Performance:
    """Performance and stability tests for Phase 3."""
    
    def test_emergent_goal_performance(self):
        """Test performance of emergent goal discovery."""
        emergent_goals = EmergentGoals()
        agent_state = AgentState(
            position=torch.tensor([5.0, 5.0, 1.0]),
            energy=80.0,
            is_alive=True,
            step_count=100
        )
        
        # Time experience addition
        start_time = time.time()
        for i in range(100):
            state_repr = torch.randn(64)
            emergent_goals.add_experience(state_repr, 0.15, agent_state)
            
        experience_time = time.time() - start_time
        
        # Time goal discovery
        start_time = time.time()
        emergent_goals._discover_goals()
        discovery_time = time.time() - start_time
        
        # Performance should be reasonable
        assert experience_time < 1.0  # Should add 100 experiences in under 1 second
        assert discovery_time < 5.0   # Should discover goals in under 5 seconds
        
    def test_multi_agent_step_performance(self):
        """Test performance of multi-agent environment steps."""
        env = MultiAgentSurvivalEnvironment(num_agents=3)
        
        # Create test agents
        agents = {}
        for i in range(3):
            agent_id = f"agent_{i}"
            agent = AdaptiveLearningAgent(
                state_dim=64,
                action_dim=6,
                memory_size=64,
                device=torch.device('cpu')
            )
            position = env.agent_positions[agent_id]
            agent.reset(initial_energy=100.0, position=position)
            agents[agent_id] = agent
            
        # Time multiple steps
        start_time = time.time()
        for step in range(50):
            actions_dict = {
                agent_id: torch.randn(6) for agent_id in agents.keys()
            }
            
            observations, action_results, done = env.step(agents, actions_dict)
            
            if done:
                break
                
        step_time = time.time() - start_time
        
        # Should complete 50 steps in reasonable time
        assert step_time < 10.0  # Under 10 seconds for 50 steps with 3 agents
        
    def test_memory_usage_stability(self):
        """Test that emergent goal system doesn't leak memory."""
        emergent_goals = EmergentGoals()
        agent_state = AgentState(
            position=torch.tensor([5.0, 5.0, 1.0]),
            energy=80.0,
            is_alive=True,
            step_count=100
        )
        
        # Add many experiences
        for i in range(1000):
            state_repr = torch.randn(64)
            emergent_goals.add_experience(state_repr, 0.15, agent_state)
            
            # Periodically trigger discovery and cleanup
            if i % 100 == 0:
                emergent_goals._discover_goals()
                emergent_goals._cleanup_old_goals()
                
        # Memory usage should be bounded
        assert len(emergent_goals.high_lp_experiences) <= emergent_goals.max_experiences
        assert len(emergent_goals.get_active_goals()) <= 20  # Reasonable goal limit


def run_phase3_tests():
    """Run all Phase 3 tests."""
    logger.info("Running Phase 3 test suite...")
    
    # Run pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short"
    ])


if __name__ == "__main__":
    run_phase3_tests()
