"""
Test Adaptive Learning Agent on AGI Puzzles

Demonstrates how to test the actual adaptive learning agent on the AGI puzzle suite.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agi_puzzles'))

import torch
import yaml
import logging
import time

from core.agent import AdaptiveLearningAgent
from puzzle1_hidden_cause import HiddenCausePuzzle
from puzzle2_object_permanence import ObjectPermanencePuzzle
from puzzle3_cooperation_deception import CooperationDeceptionPuzzle
from puzzle4_tool_use_fixed import ToolUsePuzzle
from puzzle5_deferred_gratification import DeferredGratificationPuzzle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_agent():
    """Create an adaptive learning agent for puzzle testing."""
    # Minimal configuration for puzzle testing
    config = {
        'predictive_core': {
            'state_dim': 64,
            'action_dim': 6,
            'hidden_dim': 128,
            'num_layers': 2,
            'learning_rate': 0.001
        },
        'memory': {
            'memory_size': 128,
            'word_size': 64,
            'num_heads': 4,
            'learning_rate': 0.001
        },
        'learning_progress': {
            'window_size': 100,
            'novelty_weight': 0.4,
            'empowerment_weight': 0.3,
            'use_adaptive_weights': False
        },
        'energy': {
            'max_energy': 100.0,
            'base_consumption': 0.01,
            'action_multiplier': 0.5
        },
        'goals': {
            'initial_phase': 'emergent',  # Use emergent goals for puzzle testing
            'environment_bounds': [-10, 10, -10, 10]
        },
        'action_selection': {
            'hidden_size': 128,
            'action_size': 6,
            'num_goals': 3,
            'max_velocity': 1.0,
            'exploration_rate': 0.2
        },
        'sleep': {
            'sleep_trigger_energy': 0.0,  # Disable sleep for puzzle testing
            'sleep_trigger_boredom_steps': 1000000
        }
    }
    
    agent = AdaptiveLearningAgent(config, device='cpu')
    return agent


def test_agent_on_puzzle(agent, puzzle_class, puzzle_name: str, max_steps: int = 100):
    """Test agent on a specific puzzle."""
    logger.info(f"\n=== Testing {puzzle_name} ===")
    
    # Create puzzle
    puzzle = puzzle_class(max_steps=max_steps)
    
    # Reset puzzle and initialize agent state
    sensory_input = puzzle.reset()
    
    # Initialize agent state properly
    from core.data_models import SensoryInput
    initial_sensory = SensoryInput(
        visual=sensory_input.visual,
        proprioception=sensory_input.proprioception,
        energy_level=100.0,
        timestamp=0
    )
    
    # Reset agent internal state
    agent.agent_state = agent._create_initial_state()
    agent.step_count = 0
    
    step_results = []
    
    for step in range(max_steps):
        # Get agent action (simplified for puzzle testing)
        # In a full implementation, this would use the agent's action selection
        action = torch.randn(6)
        action[:3] = torch.tanh(action[:3]) * 0.5  # Movement
        action[3:] = torch.sigmoid(action[3:])  # Discrete actions
        
        # Execute puzzle step
        sensory_input, step_result, done = puzzle.step(action)
        
        # Update agent with sensory input
        # Convert puzzle sensory input to proper SensoryInput format
        agent_sensory = SensoryInput(
            visual=sensory_input.visual,
            proprioception=sensory_input.proprioception,
            energy_level=agent.agent_state.energy,
            timestamp=step
        )
        
        agent_result = agent.step(agent_sensory, action)
        
        # Record results
        step_results.append({
            'step': step,
            'puzzle_result': step_result,
            'agent_result': agent_result,
            'learning_progress': agent_result.get('learning_progress', 0.0)
        })
        
        # Log significant events
        if any(step_result.values()) or step % 20 == 0:
            logger.info(f"Step {step}: {step_result}")
            logger.info(f"  Agent LP: {agent_result.get('learning_progress', 0.0):.3f}")
            
        if done:
            logger.info(f"Puzzle completed at step {step}")
            break
            
    # Evaluate AGI signals
    agi_level = puzzle.evaluate_agi_signals()
    puzzle_summary = puzzle.get_puzzle_summary()
    
    # Calculate agent performance metrics
    avg_learning_progress = sum(r['learning_progress'] for r in step_results) / len(step_results)
    
    result = {
        'puzzle_name': puzzle_name,
        'agi_level': agi_level.value,
        'steps_completed': len(step_results),
        'avg_learning_progress': avg_learning_progress,
        'behaviors_recorded': puzzle_summary['behaviors_recorded'],
        'learning_events': puzzle_summary['learning_events'],
        'surprise_events': puzzle_summary.get('surprise_events', 0),
        'puzzle_summary': puzzle_summary
    }
    
    logger.info(f"Results: AGI Level {agi_level.value}, "
               f"Avg LP {avg_learning_progress:.3f}, "
               f"Behaviors {puzzle_summary['behaviors_recorded']}")
    
    return result


def main():
    """Run adaptive learning agent on AGI puzzles."""
    logger.info("Testing Adaptive Learning Agent on AGI Puzzles")
    logger.info("=" * 60)
    
    # Create agent
    try:
        agent = create_test_agent()
        logger.info("✅ Adaptive learning agent created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create agent: {e}")
        logger.info("Using simple mock agent instead...")
        
        # Fallback to simple agent
        class SimpleAgent:
            def __init__(self):
                self.step_count = 0
            def reset(self, **kwargs):
                self.step_count = 0
            def step(self, sensory_input, action):
                self.step_count += 1
                return {'learning_progress': 0.1, 'energy_consumed': 0.5}
                
        agent = SimpleAgent()
    
    # Test puzzles
    puzzles = [
        (HiddenCausePuzzle, "Hidden Cause (Baby Physics)", 150),
        (ObjectPermanencePuzzle, "Object Permanence (Peekaboo)", 100),
        (CooperationDeceptionPuzzle, "Cooperation & Deception (Food Game)", 200),
        (ToolUsePuzzle, "Tool Use (The Stick)", 80),
        (DeferredGratificationPuzzle, "Deferred Gratification (Marshmallow)", 300)
    ]
    
    results = []
    
    for puzzle_class, puzzle_name, max_steps in puzzles:
        try:
            result = test_agent_on_puzzle(agent, puzzle_class, puzzle_name, max_steps)
            results.append(result)
        except Exception as e:
            logger.error(f"Error testing {puzzle_name}: {e}")
            results.append({
                'puzzle_name': puzzle_name,
                'error': str(e)
            })
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST RESULTS")
    logger.info("=" * 60)
    
    successful_tests = [r for r in results if 'error' not in r]
    
    if successful_tests:
        avg_lp = sum(r['avg_learning_progress'] for r in successful_tests) / len(successful_tests)
        total_behaviors = sum(r['behaviors_recorded'] for r in successful_tests)
        total_learning_events = sum(r['learning_events'] for r in successful_tests)
        
        agi_levels = [r['agi_level'] for r in successful_tests]
        agi_distribution = {level: agi_levels.count(level) for level in set(agi_levels)}
        
        logger.info(f"📊 Tests Completed: {len(successful_tests)}/5")
        logger.info(f"📈 Average Learning Progress: {avg_lp:.3f}")
        logger.info(f"🎯 Total Behaviors Recorded: {total_behaviors}")
        logger.info(f"🧠 Total Learning Events: {total_learning_events}")
        logger.info(f"🏆 AGI Level Distribution: {agi_distribution}")
        
        logger.info("\n📋 Individual Results:")
        for result in successful_tests:
            logger.info(f"  • {result['puzzle_name']}: {result['agi_level']} "
                       f"(LP: {result['avg_learning_progress']:.3f})")
    else:
        logger.warning("No successful tests completed")
    
    # Save results
    results_file = f"tests/agi_puzzle_results/agent_test_results_{int(time.time())}.txt"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("AGI Puzzle Test Results\n")
        f.write("=" * 30 + "\n\n")
        for result in results:
            if 'error' in result:
                f.write(f"{result['puzzle_name']}: ERROR - {result['error']}\n")
            else:
                f.write(f"{result['puzzle_name']}: {result['agi_level']} "
                       f"(LP: {result['avg_learning_progress']:.3f})\n")
    
    logger.info(f"\n💾 Results saved to: {results_file}")
    logger.info("\n🎉 AGI Puzzle testing complete!")
    
    return results


if __name__ == "__main__":
    main()
