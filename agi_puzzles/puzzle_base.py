"""
Base classes for AGI puzzle environments.

Provides common functionality for puzzle setup, agent interaction,
and AGI signal detection.
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from core.data_models import SensoryInput, AgentState

logger = logging.getLogger(__name__)


class AGISignalLevel(Enum):
    """Levels of AGI capability demonstration."""
    NONE = "none"
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class PuzzleResult:
    """Result of a puzzle test."""
    puzzle_name: str
    agent_id: str
    success_rate: float
    agi_signal_level: AGISignalLevel
    key_behaviors: List[str]
    learning_trajectory: List[Dict[str, Any]]
    total_steps: int
    completion_time: float
    detailed_metrics: Dict[str, Any]


class BasePuzzleEnvironment(ABC):
    """Base class for all AGI puzzle environments."""
    
    def __init__(self, puzzle_name: str, max_steps: int = 1000):
        self.puzzle_name = puzzle_name
        self.max_steps = max_steps
        self.current_step = 0
        self.start_time = None
        
        # Tracking for AGI signals
        self.behavior_history = []
        self.learning_events = []
        self.surprise_events = []
        self.hypothesis_tests = []
        
        # Puzzle-specific state
        self.puzzle_state = {}
        
    @abstractmethod
    def reset(self) -> SensoryInput:
        """Reset puzzle to initial state."""
        pass
        
    @abstractmethod
    def step(self, action: torch.Tensor) -> Tuple[SensoryInput, Dict[str, Any], bool]:
        """Execute one step in the puzzle environment."""
        pass
        
    @abstractmethod
    def evaluate_agi_signals(self) -> AGISignalLevel:
        """Evaluate the level of AGI capability demonstrated."""
        pass
        
    def record_behavior(self, behavior_type: str, details: Dict[str, Any]):
        """Record a significant behavior for AGI analysis."""
        self.behavior_history.append({
            'step': self.current_step,
            'type': behavior_type,
            'details': details,
            'timestamp': time.time()
        })
        
    def record_learning_event(self, event_type: str, details: Dict[str, Any]):
        """Record a learning event."""
        self.learning_events.append({
            'step': self.current_step,
            'type': event_type,
            'details': details,
            'timestamp': time.time()
        })
        
    def record_surprise(self, expected: Any, observed: Any, surprise_level: float):
        """Record a surprise event (prediction error)."""
        self.surprise_events.append({
            'step': self.current_step,
            'expected': expected,
            'observed': observed,
            'surprise_level': surprise_level,
            'timestamp': time.time()
        })
        
    def record_hypothesis_test(self, hypothesis: str, test_action: str, result: bool):
        """Record hypothesis testing behavior."""
        self.hypothesis_tests.append({
            'step': self.current_step,
            'hypothesis': hypothesis,
            'test_action': test_action,
            'result': result,
            'timestamp': time.time()
        })
        
    def get_puzzle_summary(self) -> Dict[str, Any]:
        """Get comprehensive puzzle performance summary."""
        return {
            'puzzle_name': self.puzzle_name,
            'total_steps': self.current_step,
            'duration': time.time() - self.start_time if self.start_time else 0,
            'behaviors_recorded': len(self.behavior_history),
            'learning_events': len(self.learning_events),
            'surprise_events': len(self.surprise_events),
            'hypothesis_tests': len(self.hypothesis_tests),
            'agi_signal_level': self.evaluate_agi_signals().value,
            'puzzle_state': self.puzzle_state.copy()
        }


class PuzzleTester:
    """Manages testing of agents on AGI puzzles."""
    
    def __init__(self, results_dir: str = "tests/agi_puzzle_results"):
        self.results_dir = results_dir
        self.test_results = []
        
        # Ensure results directory exists
        import os
        os.makedirs(results_dir, exist_ok=True)
        
    def run_puzzle(
        self, 
        agent: Any, 
        puzzle_env: BasePuzzleEnvironment, 
        num_episodes: int = 5
    ) -> PuzzleResult:
        """Run an agent on a puzzle environment."""
        logger.info(f"Running {puzzle_env.puzzle_name} with {num_episodes} episodes")
        
        episode_results = []
        total_success = 0
        all_behaviors = []
        
        for episode in range(num_episodes):
            logger.info(f"Episode {episode + 1}/{num_episodes}")
            
            # Reset puzzle and agent
            initial_sensory = puzzle_env.reset()
            agent.reset()
            
            episode_success = False
            episode_behaviors = []
            
            for step in range(puzzle_env.max_steps):
                # Get agent action
                action = self._get_agent_action(agent, initial_sensory)
                
                # Execute step
                sensory_input, step_result, done = puzzle_env.step(action)
                
                # Update agent
                agent_result = agent.step(sensory_input, action)
                
                # Track behaviors
                episode_behaviors.extend(puzzle_env.behavior_history[-1:])
                
                if done:
                    episode_success = step_result.get('success', False)
                    break
                    
            if episode_success:
                total_success += 1
                
            episode_results.append({
                'episode': episode,
                'success': episode_success,
                'steps': step + 1,
                'behaviors': episode_behaviors
            })
            
            all_behaviors.extend(episode_behaviors)
            
        # Calculate overall results
        success_rate = total_success / num_episodes
        agi_signal_level = puzzle_env.evaluate_agi_signals()
        
        # Extract key behaviors
        key_behaviors = self._extract_key_behaviors(all_behaviors)
        
        result = PuzzleResult(
            puzzle_name=puzzle_env.puzzle_name,
            agent_id=getattr(agent, 'agent_id', 'unknown'),
            success_rate=success_rate,
            agi_signal_level=agi_signal_level,
            key_behaviors=key_behaviors,
            learning_trajectory=episode_results,
            total_steps=sum(ep['steps'] for ep in episode_results),
            completion_time=sum(ep.get('duration', 0) for ep in episode_results),
            detailed_metrics=puzzle_env.get_puzzle_summary()
        )
        
        self.test_results.append(result)
        self._save_result(result)
        
        return result
        
    def _get_agent_action(self, agent: Any, sensory_input: SensoryInput) -> torch.Tensor:
        """Get action from agent (simplified for puzzle testing)."""
        # For puzzle testing, use random actions with some structure
        action = torch.randn(6)
        action[:3] = torch.tanh(action[:3])  # Movement
        action[3:] = torch.sigmoid(action[3:])  # Discrete actions
        return action
        
    def _extract_key_behaviors(self, behaviors: List[Dict]) -> List[str]:
        """Extract key behavior patterns from behavior history."""
        behavior_types = {}
        for behavior in behaviors:
            behavior_type = behavior.get('type', 'unknown')
            behavior_types[behavior_type] = behavior_types.get(behavior_type, 0) + 1
            
        # Return most common behaviors
        sorted_behaviors = sorted(behavior_types.items(), key=lambda x: x[1], reverse=True)
        return [behavior_type for behavior_type, count in sorted_behaviors[:5]]
        
    def _save_result(self, result: PuzzleResult):
        """Save puzzle result to file."""
        import json
        import os
        
        filename = f"{result.puzzle_name}_{result.agent_id}_{int(time.time())}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Convert result to serializable format
        result_dict = {
            'puzzle_name': result.puzzle_name,
            'agent_id': result.agent_id,
            'success_rate': result.success_rate,
            'agi_signal_level': result.agi_signal_level.value,
            'key_behaviors': result.key_behaviors,
            'learning_trajectory': result.learning_trajectory,
            'total_steps': result.total_steps,
            'completion_time': result.completion_time,
            'detailed_metrics': result.detailed_metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
            
        logger.info(f"Saved puzzle result: {filepath}")
        
    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        if not self.test_results:
            return "No test results available."
            
        report = ["# AGI Puzzle Test Report\n"]
        
        # Overall summary
        total_puzzles = len(self.test_results)
        avg_success_rate = np.mean([r.success_rate for r in self.test_results])
        
        agi_signal_counts = {}
        for result in self.test_results:
            level = result.agi_signal_level.value
            agi_signal_counts[level] = agi_signal_counts.get(level, 0) + 1
            
        report.append(f"## Overall Summary")
        report.append(f"- **Total Puzzles Tested**: {total_puzzles}")
        report.append(f"- **Average Success Rate**: {avg_success_rate:.2%}")
        report.append(f"- **AGI Signal Distribution**: {agi_signal_counts}")
        report.append("")
        
        # Individual puzzle results
        report.append("## Individual Puzzle Results")
        for result in self.test_results:
            report.append(f"### {result.puzzle_name}")
            report.append(f"- **Success Rate**: {result.success_rate:.2%}")
            report.append(f"- **AGI Signal Level**: {result.agi_signal_level.value}")
            report.append(f"- **Key Behaviors**: {', '.join(result.key_behaviors)}")
            report.append(f"- **Total Steps**: {result.total_steps}")
            report.append("")
            
        return "\n".join(report)
