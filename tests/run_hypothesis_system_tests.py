"""
Test Runner for Game-Specific Hypothesis Generation and Testing System

This script runs all tests for the hypothesis system components:
- GamePatternAnalyzer tests
- HypothesisGenerator tests
- HypothesisTester tests
- HypothesisIntegrationSystem tests
- Complete system integration tests

Usage:
    python tests/run_hypothesis_system_tests.py
    python tests/run_hypothesis_system_tests.py --verbose
    python tests/run_hypothesis_system_tests.py --component pattern_analyzer
"""

import sys
import os
import subprocess
import argparse
from datetime import datetime

# Disable pycache
sys.dont_write_bytecode = True

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_tests(component=None, verbose=False):
    """Run hypothesis system tests."""

    test_files = {
        'pattern_analyzer': 'test_game_pattern_analyzer.py',
        'hypothesis_generator': 'test_hypothesis_generator.py',
        'hypothesis_tester': 'test_hypothesis_tester.py',
        'integration': 'test_hypothesis_integration.py',
        'system_integration': 'test_hypothesis_system_integration.py'
    }

    # Determine which tests to run
    if component:
        if component not in test_files:
            print(f"Unknown component: {component}")
            print(f"Available components: {', '.join(test_files.keys())}")
            return False
        tests_to_run = [test_files[component]]
    else:
        tests_to_run = list(test_files.values())

    print("=" * 60)
    print("GAME-SPECIFIC HYPOTHESIS GENERATION AND TESTING SYSTEM")
    print("Test Suite Runner")
    print("=" * 60)
    print(f"Running tests: {', '.join(tests_to_run)}")
    print(f"Verbose mode: {verbose}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    overall_success = True
    test_results = {}

    for test_file in tests_to_run:
        print(f"Running {test_file}...")
        print("-" * 40)

        # Build pytest command
        cmd = ['python', '-m', 'pytest', f'tests/{test_file}']

        if verbose:
            cmd.extend(['-v', '-s'])
        else:
            cmd.append('-q')

        # Add coverage if available
        try:
            import pytest_cov
            cmd.extend(['--cov=intelligence', '--cov-report=term-missing'])
        except ImportError:
            pass

        try:
            # Run the test
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(__file__))
            )

            # Display results
            if result.returncode == 0:
                print(f"✅ {test_file} - PASSED")
                test_results[test_file] = "PASSED"
            else:
                print(f"❌ {test_file} - FAILED")
                test_results[test_file] = "FAILED"
                overall_success = False

            # Show output if verbose or if failed
            if verbose or result.returncode != 0:
                if result.stdout:
                    print("STDOUT:")
                    print(result.stdout)
                if result.stderr:
                    print("STDERR:")
                    print(result.stderr)

        except Exception as e:
            print(f"❌ {test_file} - ERROR: {e}")
            test_results[test_file] = f"ERROR: {e}"
            overall_success = False

        print()

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_file, result in test_results.items():
        status_emoji = "✅" if result == "PASSED" else "❌"
        print(f"{status_emoji} {test_file}: {result}")

    print()
    print(f"Overall result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return overall_success

def check_dependencies():
    """Check if required dependencies are available."""
    print("Checking dependencies...")

    required_packages = [
        'pytest',
        'numpy',
        'opencv-python'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n❌ Missing dependencies: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False

    print("✅ All dependencies available")
    return True

def show_test_info():
    """Show information about the test suite."""
    print("=" * 60)
    print("HYPOTHESIS SYSTEM TEST SUITE INFORMATION")
    print("=" * 60)
    print()
    print("Components tested:")
    print("  • GamePatternAnalyzer - Visual pattern detection and game mechanics analysis")
    print("  • HypothesisGenerator - Automatic hypothesis generation with database integration")
    print("  • HypothesisTester - Experimental framework with action reasoning")
    print("  • HypothesisIntegrationSystem - Integration with existing gameplay systems")
    print("  • Complete System Integration - End-to-end workflow testing")
    print()
    print("Test categories:")
    print("  • Unit tests - Individual component functionality")
    print("  • Integration tests - Component interaction")
    print("  • System tests - Complete workflow testing")
    print("  • Error handling tests - Resilience and edge cases")
    print("  • Performance tests - System performance validation")
    print()
    print("Test coverage includes:")
    print("  • Pattern recognition and analysis")
    print("  • Hypothesis generation from multiple sources")
    print("  • Systematic hypothesis testing with reasoning")
    print("  • Database integration and persistence")
    print("  • Multi-level learning insights")
    print("  • Action6Coordinator integration")
    print("  • Enhanced Gameplay coordination")
    print("  • Error handling and resilience")
    print()

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Run tests for Game-Specific Hypothesis Generation and Testing System"
    )
    parser.add_argument(
        '--component',
        choices=['pattern_analyzer', 'hypothesis_generator', 'hypothesis_tester', 'integration', 'system_integration'],
        help='Run tests for specific component only'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--info', '-i',
        action='store_true',
        help='Show test suite information'
    )
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check dependencies only'
    )

    args = parser.parse_args()

    if args.info:
        show_test_info()
        return

    if args.check_deps:
        success = check_dependencies()
        sys.exit(0 if success else 1)

    # Check dependencies before running tests
    if not check_dependencies():
        print("\n❌ Cannot run tests due to missing dependencies")
        sys.exit(1)

    print()

    # Run tests
    success = run_tests(args.component, args.verbose)

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()