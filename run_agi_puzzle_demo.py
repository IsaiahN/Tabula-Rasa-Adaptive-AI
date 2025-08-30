"""
Simple AGI Puzzle Demo

Demonstrates the AGI puzzle functionality with basic testing.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agi_puzzles'))

import torch
import logging
import time

from puzzle1_hidden_cause import HiddenCausePuzzle
from puzzle2_object_permanence import ObjectPermanencePuzzle
from puzzle3_cooperation_deception import CooperationDeceptionPuzzle
from puzzle4_tool_use import ToolUsePuzzle
from puzzle5_deferred_gratification import DeferredGratificationPuzzle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleAgent:
    """Simple agent for puzzle demonstration."""
    
    def __init__(self, agent_id: str = "demo_agent"):
        self.agent_id = agent_id
        self.step_count = 0
        
    def reset(self):
        """Reset agent state."""
        self.step_count = 0
        
    def step(self, sensory_input, action):
        """Process one step."""
        self.step_count += 1
        return {
            'learning_progress': 0.1,
            'energy_consumed': 0.5,
            'prediction_error': 0.2
        }


def demo_single_puzzle(puzzle_class, puzzle_name: str):
    """Demonstrate a single puzzle."""
    logger.info(f"\n=== {puzzle_name} Demo ===")
    
    # Create puzzle and agent
    puzzle = puzzle_class()
    agent = SimpleAgent()
    
    # Reset puzzle
    sensory_input = puzzle.reset()
    agent.reset()
    
    # Run for a few steps
    for step in range(20):
        # Generate random action
        action = torch.randn(6)
        action[:3] = torch.tanh(action[:3])  # Movement
        action[3:] = torch.sigmoid(action[3:])  # Discrete actions
        
        # Execute step
        sensory_input, step_result, done = puzzle.step(action)
        
        # Update agent
        agent_result = agent.step(sensory_input, action)
        
        # Log interesting events
        if any(step_result.values()):
            logger.info(f"Step {step}: {step_result}")
            
        if done:
            logger.info(f"Puzzle completed at step {step}")
            break
            
    # Evaluate AGI signals
    agi_level = puzzle.evaluate_agi_signals()
    logger.info(f"AGI Signal Level: {agi_level.value}")
    
    # Get puzzle summary
    summary = puzzle.get_puzzle_summary()
    logger.info(f"Behaviors recorded: {summary['behaviors_recorded']}")
    logger.info(f"Learning events: {summary['learning_events']}")
    
    return {
        'puzzle_name': puzzle_name,
        'agi_level': agi_level.value,
        'summary': summary
    }


def main():
    """Run AGI puzzle demonstrations."""
    logger.info("AGI Puzzle Suite Demonstration")
    logger.info("=" * 50)
    
    puzzles = [
        (HiddenCausePuzzle, "Hidden Cause (Baby Physics)"),
        (ObjectPermanencePuzzle, "Object Permanence (Peekaboo)"),
        (CooperationDeceptionPuzzle, "Cooperation & Deception (Food Game)"),
        (ToolUsePuzzle, "Tool Use (The Stick)"),
        (DeferredGratificationPuzzle, "Deferred Gratification (Marshmallow)")
    ]
    
    results = []
    
    for puzzle_class, puzzle_name in puzzles:
        try:
            result = demo_single_puzzle(puzzle_class, puzzle_name)
            results.append(result)
        except Exception as e:
            logger.error(f"Error in {puzzle_name}: {e}")
            results.append({
                'puzzle_name': puzzle_name,
                'error': str(e)
            })
            
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("DEMONSTRATION SUMMARY")
    logger.info("=" * 50)
    
    for result in results:
        if 'error' in result:
            logger.info(f"❌ {result['puzzle_name']}: ERROR - {result['error']}")
        else:
            logger.info(f"✅ {result['puzzle_name']}: AGI Level {result['agi_level']}")
            
    logger.info(f"\nCompleted demonstration of {len(puzzles)} AGI puzzles")
    logger.info("Puzzle implementations are ready for testing with the adaptive learning agent!")
    
    return results


if __name__ == "__main__":
    main()
