"""
Test runner specifically for CORE_GAME_MECHANICS AI-enhanced system.
Provides comprehensive testing with detailed reporting.
"""

import pytest
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Add CORE_GAME_MECHANICS to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CORE_GAME_MECHANICS'))


class CoreMechanicsTestRunner:
    """Specialized test runner for CORE_GAME_MECHANICS system."""

    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None

    def run_component_tests(self) -> Dict[str, Any]:
        """Run all component tests with detailed reporting."""
        self.start_time = time.time()

        print("🚀 Starting CORE_GAME_MECHANICS AI System Tests")
        print("=" * 60)

        # Define test modules
        test_modules = [
            {
                'name': 'AI Integration Tests',
                'module': 'test_core_game_mechanics_ai.py',
                'description': 'Tests for AI orchestration, vision, patterns, and knowledge integration'
            },
            {
                'name': 'Performance Validation',
                'module': 'test_ai_performance_validation.py',
                'description': 'Validates AI performance improvements and benchmarks'
            }
        ]

        all_results = {}

        for test_module in test_modules:
            print(f"\n🧪 Running {test_module['name']}")
            print(f"   {test_module['description']}")
            print("-" * 40)

            # Run pytest for this module
            module_path = os.path.join(os.path.dirname(__file__), test_module['module'])

            if os.path.exists(module_path):
                exit_code = pytest.main([
                    module_path,
                    "-v",
                    "--tb=short",
                    f"--json-report",
                    f"--json-report-file=test_results_{test_module['name'].lower().replace(' ', '_')}.json"
                ])

                all_results[test_module['name']] = {
                    'exit_code': exit_code,
                    'module': test_module['module'],
                    'status': 'PASSED' if exit_code == 0 else 'FAILED'
                }

                print(f"   Result: {'✅ PASSED' if exit_code == 0 else '❌ FAILED'}")
            else:
                print(f"   ⚠️  Module not found: {module_path}")
                all_results[test_module['name']] = {
                    'exit_code': -1,
                    'module': test_module['module'],
                    'status': 'SKIPPED'
                }

        self.end_time = time.time()
        self.test_results = all_results

        return self._generate_summary_report()

    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests specifically."""
        print("\n🔗 Running Integration Tests")
        print("=" * 40)

        # Integration test scenarios
        integration_tests = [
            'TestIntegrationScenarios::test_full_game_simulation',
            'TestIntegrationScenarios::test_performance_monitoring_integration',
            'TestAIPerformanceValidation::test_system_integration_performance'
        ]

        results = {}
        for test in integration_tests:
            print(f"   Running {test}")
            exit_code = pytest.main([
                f"tests/test_core_game_mechanics_ai.py::{test}",
                "-v"
            ])
            results[test] = 'PASSED' if exit_code == 0 else 'FAILED'

        return results

    def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmark tests."""
        print("\n⚡ Running Performance Benchmarks")
        print("=" * 40)

        benchmark_tests = [
            'TestAISystemBenchmarks::test_decision_speed_benchmark',
            'TestAIPerformanceValidation::test_ai_vs_random_action_selection',
            'TestAIPerformanceValidation::test_vision_coordinate_optimization'
        ]

        results = {}
        for test in benchmark_tests:
            print(f"   Running {test}")
            exit_code = pytest.main([
                f"tests/test_ai_performance_validation.py::{test}",
                "-v",
                "-s"  # Show print statements for benchmarks
            ])
            results[test] = 'PASSED' if exit_code == 0 else 'FAILED'

        return results

    def _generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        total_time = self.end_time - self.start_time if self.end_time and self.start_time else 0

        summary = {
            'test_run_info': {
                'start_time': self.start_time,
                'end_time': self.end_time,
                'total_duration': total_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'test_results': self.test_results,
            'summary_stats': {
                'total_modules': len(self.test_results),
                'passed_modules': len([r for r in self.test_results.values() if r['status'] == 'PASSED']),
                'failed_modules': len([r for r in self.test_results.values() if r['status'] == 'FAILED']),
                'skipped_modules': len([r for r in self.test_results.values() if r['status'] == 'SKIPPED'])
            }
        }

        return summary

    def print_summary_report(self, summary: Dict[str, Any]):
        """Print formatted summary report."""
        print("\n" + "=" * 60)
        print("📊 CORE_GAME_MECHANICS AI Test Summary")
        print("=" * 60)

        stats = summary['summary_stats']
        print(f"Total Test Modules: {stats['total_modules']}")
        print(f"✅ Passed: {stats['passed_modules']}")
        print(f"❌ Failed: {stats['failed_modules']}")
        print(f"⚠️  Skipped: {stats['skipped_modules']}")
        print(f"⏱️  Duration: {summary['test_run_info']['total_duration']:.2f}s")

        print(f"\nDetailed Results:")
        for module_name, result in summary['test_results'].items():
            status_emoji = {
                'PASSED': '✅',
                'FAILED': '❌',
                'SKIPPED': '⚠️'
            }.get(result['status'], '❓')

            print(f"  {status_emoji} {module_name}: {result['status']}")

        # Overall status
        overall_status = 'PASSED' if stats['failed_modules'] == 0 else 'FAILED'
        print(f"\n🎯 Overall Status: {overall_status}")

        return overall_status


def main():
    """Main test runner entry point."""
    runner = CoreMechanicsTestRunner()

    try:
        # Run all component tests
        summary = runner.run_component_tests()

        # Run integration tests
        integration_results = runner.run_integration_tests()
        summary['integration_results'] = integration_results

        # Run performance benchmarks
        benchmark_results = runner.run_performance_benchmarks()
        summary['benchmark_results'] = benchmark_results

        # Print summary
        overall_status = runner.print_summary_report(summary)

        # Save detailed results
        results_file = f"core_mechanics_test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n📄 Detailed results saved to: {results_file}")

        # Return appropriate exit code
        return 0 if overall_status == 'PASSED' else 1

    except Exception as e:
        print(f"\n❌ Test runner error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)