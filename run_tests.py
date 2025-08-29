#!/usr/bin/env python3
"""
Test runner for the Adaptive Learning Agent project.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_unit_tests():
    """Run unit tests with pytest."""
    print("Running unit tests...")
    result = subprocess.run([
        sys.executable, '-m', 'pytest', 
        'tests/', 
        '-v', 
        '--tb=short'
    ], capture_output=False)
    
    return result.returncode == 0


def run_phase0_experiments():
    """Run Phase 0 validation experiments."""
    print("\nRunning Phase 0 experiments...")
    
    experiments = [
        'experiments/phase0_lp_validation.py',
        'experiments/phase0_memory_test.py', 
        'experiments/phase0_survival_test.py'
    ]
    
    results = {}
    
    for experiment in experiments:
        print(f"\n{'='*60}")
        print(f"Running {experiment}")
        print('='*60)
        
        result = subprocess.run([
            sys.executable, experiment,
            '--config', 'configs/phase0_config.yaml'
        ], capture_output=False)
        
        results[experiment] = result.returncode == 0
        
        if results[experiment]:
            print(f"✓ {experiment} PASSED")
        else:
            print(f"✗ {experiment} FAILED")
            
    return all(results.values())


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description='Adaptive Learning Agent Test Runner')
    parser.add_argument('--unit-tests', action='store_true', 
                       help='Run unit tests only')
    parser.add_argument('--experiments', action='store_true',
                       help='Run Phase 0 experiments only')
    parser.add_argument('--all', action='store_true', default=True,
                       help='Run all tests (default)')
    
    args = parser.parse_args()
    
    # If specific flags are set, disable default all
    if args.unit_tests or args.experiments:
        args.all = False
        
    success = True
    
    if args.all or args.unit_tests:
        success &= run_unit_tests()
        
    if args.all or args.experiments:
        success &= run_phase0_experiments()
        
    print("\n" + "="*60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED!")
        sys.exit(1)


if __name__ == '__main__':
    main()