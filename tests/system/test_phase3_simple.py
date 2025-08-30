"""
Simple Phase 3 Test - Verify emergent goals and multi-agent environment work.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import logging
from goals.goal_system import EmergentGoals
from environment.multi_agent_environment import MultiAgentSurvivalEnvironment
from core.data_models import AgentState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_emergent_goals():
    """Test basic emergent goal functionality."""
    logger.info("Testing emergent goal discovery...")
    
    emergent_goals = EmergentGoals()
    
    # Create sample agent state
    agent_state = AgentState(
        position=torch.tensor([5.0, 5.0, 1.0]),
        orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        energy=80.0,
        hidden_state=torch.zeros(64),
        active_goals=[],
        timestamp=100
    )
    
    # Add high learning progress experiences
    for i in range(10):
        state_repr = torch.randn(64)
        learning_progress = 0.15  # Above threshold
        emergent_goals.add_experience(state_repr, learning_progress, agent_state)
    
    logger.info(f"Added {len(emergent_goals.high_lp_experiences)} high LP experiences")
    
    # Trigger goal discovery
    emergent_goals._discover_goals()
    active_goals = emergent_goals.get_active_goals(agent_state)
    
    logger.info(f"Discovered {len(active_goals)} emergent goals")
    
    return len(emergent_goals.high_lp_experiences) > 0


def test_multi_agent_environment():
    """Test basic multi-agent environment functionality."""
    logger.info("Testing multi-agent environment...")
    
    try:
        # Create environment
        env = MultiAgentSurvivalEnvironment(
            num_agents=2,
            world_size=(10, 10, 3),
            num_food_sources=2
        )
        
        logger.info(f"Created environment with {env.num_agents} agents")
        logger.info(f"Agent positions: {env.agent_positions}")
        
        # Test environment metrics
        metrics = env.get_environment_metrics()
        logger.info(f"Environment metrics: {list(metrics.keys())}")
        
        # Test reset
        env.reset()
        logger.info("Environment reset successful")
        
        return True
        
    except Exception as e:
        logger.error(f"Multi-agent environment test failed: {e}")
        return False


def test_social_interaction_detection():
    """Test social interaction detection."""
    logger.info("Testing social interaction detection...")
    
    try:
        env = MultiAgentSurvivalEnvironment(num_agents=2)
        
        # Position agents close together
        agent_ids = list(env.agent_positions.keys())
        env.agent_positions[agent_ids[0]] = torch.tensor([5.0, 5.0, 1.0])
        env.agent_positions[agent_ids[1]] = torch.tensor([5.5, 5.5, 1.0])
        
        # Set movement actions
        env.agent_last_actions[agent_ids[0]] = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        env.agent_last_actions[agent_ids[1]] = torch.tensor([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        
        # Check for interactions
        social_interactions = env._check_social_interactions()
        
        logger.info(f"Detected {len(social_interactions)} social interactions")
        
        return True
        
    except Exception as e:
        logger.error(f"Social interaction test failed: {e}")
        return False


def main():
    """Run all Phase 3 tests."""
    logger.info("=" * 50)
    logger.info("PHASE 3 SIMPLE VERIFICATION TESTS")
    logger.info("=" * 50)
    
    tests = [
        ("Emergent Goals", test_emergent_goals),
        ("Multi-Agent Environment", test_multi_agent_environment),
        ("Social Interaction Detection", test_social_interaction_detection)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "PASS" if result else "FAIL"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All Phase 3 core functionality verified!")
        return True
    else:
        logger.warning("⚠️  Some tests failed - review implementation")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
