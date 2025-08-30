"""
AGI Puzzle Test Runner

Comprehensive test suite for evaluating AGI capabilities through specialized puzzles.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agi_puzzles'))

import torch
import logging
import time
from typing import Dict, List, Any
import json

from puzzle_base import PuzzleTester, AGISignalLevel
from puzzle1_hidden_cause import HiddenCausePuzzle
from puzzle2_object_permanence import ObjectPermanencePuzzle
from puzzle3_cooperation_deception import CooperationDeceptionPuzzle
from puzzle4_tool_use import ToolUsePuzzle
from puzzle5_deferred_gratification import DeferredGratificationPuzzle

# Mock agent for testing (replace with actual agent)
class MockAgent:
    """Simple mock agent for puzzle testing."""
    
    def __init__(self, agent_id: str = "test_agent"):
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AGIPuzzleTestSuite:
    """Complete test suite for AGI puzzle evaluation."""
    
    def __init__(self, results_dir: str = "tests/agi_puzzle_results"):
        self.results_dir = results_dir
        self.tester = PuzzleTester(results_dir)
        
        # Initialize all puzzles
        self.puzzles = {
            'hidden_cause': HiddenCausePuzzle(),
            'object_permanence': ObjectPermanencePuzzle(),
            'cooperation_deception': CooperationDeceptionPuzzle(),
            'tool_use': ToolUsePuzzle(),
            'deferred_gratification': DeferredGratificationPuzzle()
        }
        
        logger.info(f"Initialized {len(self.puzzles)} AGI puzzles")
        
    def run_single_puzzle(self, puzzle_name: str, agent: Any, num_episodes: int = 3) -> Dict[str, Any]:
        """Run a single puzzle test."""
        if puzzle_name not in self.puzzles:
            raise ValueError(f"Unknown puzzle: {puzzle_name}")
            
        puzzle = self.puzzles[puzzle_name]
        logger.info(f"Running {puzzle_name} puzzle...")
        
        start_time = time.time()
        result = self.tester.run_puzzle(agent, puzzle, num_episodes)
        duration = time.time() - start_time
        
        logger.info(f"Completed {puzzle_name}: Success rate {result.success_rate:.2%}, "
                   f"AGI level {result.agi_signal_level.value}, Duration {duration:.1f}s")
        
        return {
            'puzzle_name': puzzle_name,
            'result': result,
            'duration': duration
        }
        
    def run_full_suite(self, agent: Any, num_episodes_per_puzzle: int = 3) -> Dict[str, Any]:
        """Run the complete AGI puzzle test suite."""
        logger.info("Starting full AGI puzzle test suite...")
        suite_start_time = time.time()
        
        results = {}
        puzzle_summaries = []
        
        for puzzle_name in self.puzzles.keys():
            try:
                puzzle_result = self.run_single_puzzle(puzzle_name, agent, num_episodes_per_puzzle)
                results[puzzle_name] = puzzle_result
                
                # Create summary
                result = puzzle_result['result']
                puzzle_summaries.append({
                    'puzzle': puzzle_name,
                    'success_rate': result.success_rate,
                    'agi_level': result.agi_signal_level.value,
                    'key_behaviors': result.key_behaviors,
                    'duration': puzzle_result['duration']
                })
                
            except Exception as e:
                logger.error(f"Error running {puzzle_name}: {e}")
                results[puzzle_name] = {'error': str(e)}
                
        total_duration = time.time() - suite_start_time
        
        # Generate comprehensive analysis
        analysis = self._analyze_results(puzzle_summaries)
        
        suite_results = {
            'agent_id': agent.agent_id,
            'total_duration': total_duration,
            'puzzles_completed': len([r for r in results.values() if 'error' not in r]),
            'puzzle_results': results,
            'analysis': analysis,
            'timestamp': time.time()
        }
        
        # Save suite results
        self._save_suite_results(suite_results)
        
        logger.info(f"Completed full test suite in {total_duration:.1f}s")
        return suite_results
        
    def _analyze_results(self, puzzle_summaries: List[Dict]) -> Dict[str, Any]:
        """Analyze overall performance across puzzles."""
        if not puzzle_summaries:
            return {'error': 'No puzzle results to analyze'}
            
        # Calculate aggregate metrics
        success_rates = [p['success_rate'] for p in puzzle_summaries]
        avg_success_rate = sum(success_rates) / len(success_rates)
        
        # AGI signal distribution
        agi_levels = [p['agi_level'] for p in puzzle_summaries]
        agi_distribution = {}
        for level in agi_levels:
            agi_distribution[level] = agi_distribution.get(level, 0) + 1
            
        # Identify strengths and weaknesses
        best_puzzle = max(puzzle_summaries, key=lambda p: p['success_rate'])
        worst_puzzle = min(puzzle_summaries, key=lambda p: p['success_rate'])
        
        # Advanced capabilities check
        advanced_puzzles = [p for p in puzzle_summaries if p['agi_level'] == 'advanced']
        intermediate_puzzles = [p for p in puzzle_summaries if p['agi_level'] == 'intermediate']
        
        return {
            'overall_success_rate': avg_success_rate,
            'agi_signal_distribution': agi_distribution,
            'best_performance': {
                'puzzle': best_puzzle['puzzle'],
                'success_rate': best_puzzle['success_rate'],
                'agi_level': best_puzzle['agi_level']
            },
            'worst_performance': {
                'puzzle': worst_puzzle['puzzle'],
                'success_rate': worst_puzzle['success_rate'],
                'agi_level': worst_puzzle['agi_level']
            },
            'advanced_capabilities': len(advanced_puzzles),
            'intermediate_capabilities': len(intermediate_puzzles),
            'capability_areas': {
                'causality_learning': next((p for p in puzzle_summaries if p['puzzle'] == 'hidden_cause'), {}).get('agi_level', 'none'),
                'object_permanence': next((p for p in puzzle_summaries if p['puzzle'] == 'object_permanence'), {}).get('agi_level', 'none'),
                'theory_of_mind': next((p for p in puzzle_summaries if p['puzzle'] == 'cooperation_deception'), {}).get('agi_level', 'none'),
                'tool_use': next((p for p in puzzle_summaries if p['puzzle'] == 'tool_use'), {}).get('agi_level', 'none'),
                'self_control': next((p for p in puzzle_summaries if p['puzzle'] == 'deferred_gratification'), {}).get('agi_level', 'none')
            }
        }
        
    def _save_suite_results(self, suite_results: Dict):
        """Save complete suite results."""
        filename = f"agi_suite_{suite_results['agent_id']}_{int(suite_results['timestamp'])}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Convert to serializable format
        serializable_results = self._make_serializable(suite_results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
            
        logger.info(f"Saved suite results: {filepath}")
        
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if hasattr(obj, '__dict__'):
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif hasattr(obj, 'value'):  # Enum
            return obj.value
        else:
            return obj
            
    def generate_report(self, suite_results: Dict) -> str:
        """Generate human-readable test report."""
        report = ["# AGI Puzzle Test Suite Report\n"]
        
        # Header information
        report.append(f"**Agent ID**: {suite_results['agent_id']}")
        report.append(f"**Test Duration**: {suite_results['total_duration']:.1f} seconds")
        report.append(f"**Puzzles Completed**: {suite_results['puzzles_completed']}/5")
        report.append("")
        
        # Overall analysis
        analysis = suite_results['analysis']
        report.append("## Overall Performance")
        report.append(f"- **Average Success Rate**: {analysis['overall_success_rate']:.2%}")
        report.append(f"- **Advanced Capabilities**: {analysis['advanced_capabilities']}/5 puzzles")
        report.append(f"- **Intermediate Capabilities**: {analysis['intermediate_capabilities']}/5 puzzles")
        report.append("")
        
        # Best and worst performance
        report.append("### Performance Highlights")
        report.append(f"- **Best Performance**: {analysis['best_performance']['puzzle']} "
                     f"({analysis['best_performance']['success_rate']:.2%}, "
                     f"{analysis['best_performance']['agi_level']})")
        report.append(f"- **Needs Improvement**: {analysis['worst_performance']['puzzle']} "
                     f"({analysis['worst_performance']['success_rate']:.2%}, "
                     f"{analysis['worst_performance']['agi_level']})")
        report.append("")
        
        # Capability breakdown
        report.append("## Capability Assessment")
        capabilities = analysis['capability_areas']
        for capability, level in capabilities.items():
            report.append(f"- **{capability.replace('_', ' ').title()}**: {level}")
        report.append("")
        
        # Individual puzzle results
        report.append("## Individual Puzzle Results")
        for puzzle_name, puzzle_data in suite_results['puzzle_results'].items():
            if 'error' in puzzle_data:
                report.append(f"### {puzzle_name.replace('_', ' ').title()}")
                report.append(f"**Error**: {puzzle_data['error']}")
                report.append("")
                continue
                
            result = puzzle_data['result']
            report.append(f"### {puzzle_name.replace('_', ' ').title()}")
            report.append(f"- **Success Rate**: {result.success_rate:.2%}")
            report.append(f"- **AGI Signal Level**: {result.agi_signal_level.value}")
            report.append(f"- **Key Behaviors**: {', '.join(result.key_behaviors)}")
            report.append(f"- **Total Steps**: {result.total_steps}")
            report.append(f"- **Duration**: {puzzle_data['duration']:.1f}s")
            report.append("")
            
        return "\n".join(report)


def run_puzzle_tests():
    """Main function to run AGI puzzle tests."""
    logger.info("Initializing AGI Puzzle Test Suite...")
    
    # Create test suite
    test_suite = AGIPuzzleTestSuite()
    
    # Create mock agent for testing
    agent = MockAgent("adaptive_learning_agent_v1")
    
    # Run full test suite
    results = test_suite.run_full_suite(agent, num_episodes_per_puzzle=2)
    
    # Generate and display report
    report = test_suite.generate_report(results)
    print("\n" + "="*60)
    print(report)
    print("="*60)
    
    return results


if __name__ == "__main__":
    run_puzzle_tests()
