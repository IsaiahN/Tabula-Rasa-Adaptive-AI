#!/usr/bin/env python3
"""
Simple test execution script for CORE_GAME_MECHANICS AI system.
Run this script to execute all tests with proper configuration.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_tests():
    """Execute all CORE_GAME_MECHANICS tests."""

    # Get the tests directory
    tests_dir = Path(__file__).parent

    print("🚀 CORE_GAME_MECHANICS AI System Test Suite")
    print("=" * 50)

    # Check if pytest is available
    try:
        import pytest
        print("✅ pytest is available")
    except ImportError:
        print("❌ pytest not found. Please install: pip install pytest")
        return 1

    # Set environment variables
    os.environ['PYTHONPATH'] = str(tests_dir.parent / 'CORE_GAME_MECHANICS')

    # Test files to run
    test_files = [
        'test_core_game_mechanics_ai.py',
        'test_ai_performance_validation.py'
    ]

    # Check which test files exist
    available_tests = []
    for test_file in test_files:
        test_path = tests_dir / test_file
        if test_path.exists():
            available_tests.append(str(test_path))
            print(f"✅ Found: {test_file}")
        else:
            print(f"⚠️  Missing: {test_file}")

    if not available_tests:
        print("❌ No test files found!")
        return 1

    print(f"\n🧪 Running {len(available_tests)} test modules...")

    # Run tests with appropriate configuration
    cmd = [
        sys.executable, '-m', 'pytest',
        *available_tests,
        '-v',
        '--tb=short',
        '--color=yes',
        '--disable-warnings'
    ]

    try:
        result = subprocess.run(cmd, cwd=tests_dir, timeout=300)
        return result.returncode
    except subprocess.TimeoutExpired:
        print("❌ Tests timed out after 5 minutes")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_tests()
    print(f"\n🎯 Test execution completed with exit code: {exit_code}")
    sys.exit(exit_code)