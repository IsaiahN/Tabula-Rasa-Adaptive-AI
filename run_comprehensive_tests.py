#!/usr/bin/env python3
"""
Comprehensive Test Suite Runner for ARC Training System
Provides unified test execution with filtering, reporting, and validation.
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


class ARCTestRunner:
    """Comprehensive test runner for the ARC training system."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_dir = self.project_root / "tests"
        self.results = {}
        
    def discover_tests(self) -> Dict[str, List[str]]:
        """Discover all available tests organized by category."""
        test_categories = {
            'unit': [],
            'integration': [],
            'system': []
        }
        
        for category in test_categories.keys():
            category_dir = self.tests_dir / category
            if category_dir.exists():
                for test_file in category_dir.glob('test_*.py'):
                    test_categories[category].append(str(test_file))
        
        return test_categories
    
    def run_test_file(self, test_file: str, verbose: bool = True) -> Dict[str, Any]:
        """Run a single test file and return results."""
        print(f"\n{'='*60}")
        print(f"🧪 Running: {os.path.basename(test_file)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Run with pytest for better async support and output
            cmd = [
                sys.executable, '-m', 'pytest', 
                test_file, 
                '-v' if verbose else '-q',
                '--tb=short',
                '--disable-warnings'
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                cwd=self.project_root
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            success = result.returncode == 0
            stdout_lines = result.stdout.split('\n')
            stderr_lines = result.stderr.split('\n')
            
            # Count tests
            test_count = 0
            passed_count = 0
            failed_count = 0
            
            for line in stdout_lines:
                if '::' in line and ('PASSED' in line or 'FAILED' in line):
                    test_count += 1
                    if 'PASSED' in line:
                        passed_count += 1
                    elif 'FAILED' in line:
                        failed_count += 1
            
            test_result = {
                'file': test_file,
                'success': success,
                'execution_time': execution_time,
                'test_count': test_count,
                'passed': passed_count,
                'failed': failed_count,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
            # Print summary
            status_emoji = "✅" if success else "❌"
            print(f"{status_emoji} {'PASSED' if success else 'FAILED'} - {test_count} tests in {execution_time:.2f}s")
            if not success:
                print(f"   💔 {failed_count} failed, {passed_count} passed")
                if result.stderr:
                    print("   🔍 Errors:")
                    for line in stderr_lines[:5]:  # Show first 5 error lines
                        if line.strip():
                            print(f"      {line}")
            else:
                print(f"   🎉 All {passed_count} tests passed!")
            
            return test_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ EXCEPTION: {str(e)}")
            
            return {
                'file': test_file,
                'success': False,
                'execution_time': execution_time,
                'test_count': 0,
                'passed': 0,
                'failed': 0,
                'stdout': '',
                'stderr': f"Exception: {str(e)}",
                'return_code': -1,
                'exception': str(e)
            }
    
    def run_category(self, category: str, verbose: bool = True) -> List[Dict[str, Any]]:
        """Run all tests in a specific category."""
        test_categories = self.discover_tests()
        if category not in test_categories:
            print(f"❌ Category '{category}' not found. Available: {list(test_categories.keys())}")
            return []
        
        test_files = test_categories[category]
        if not test_files:
            print(f"ℹ️  No tests found in category '{category}'")
            return []
        
        print(f"\n🚀 Running {category.upper()} tests ({len(test_files)} files)")
        print(f"{'='*80}")
        
        results = []
        for test_file in test_files:
            result = self.run_test_file(test_file, verbose)
            results.append(result)
        
        return results
    
    def run_specific_tests(self, test_patterns: List[str], verbose: bool = True) -> List[Dict[str, Any]]:
        """Run tests matching specific patterns."""
        all_tests = self.discover_tests()
        matching_tests = []
        
        for category, test_files in all_tests.items():
            for test_file in test_files:
                for pattern in test_patterns:
                    if pattern in os.path.basename(test_file):
                        matching_tests.append(test_file)
                        break
        
        if not matching_tests:
            print(f"❌ No tests found matching patterns: {test_patterns}")
            return []
        
        print(f"\n🎯 Running specific tests ({len(matching_tests)} files)")
        print(f"{'='*80}")
        
        results = []
        for test_file in matching_tests:
            result = self.run_test_file(test_file, verbose)
            results.append(result)
        
        return results
    
    def run_all_tests(self, verbose: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """Run all tests in all categories."""
        print(f"\n🏃‍♂️ Running ALL tests in the ARC training system")
        print(f"{'='*80}")
        
        all_results = {}
        test_categories = self.discover_tests()
        
        for category in ['unit', 'integration', 'system']:
            if category in test_categories and test_categories[category]:
                category_results = self.run_category(category, verbose)
                all_results[category] = category_results
        
        return all_results
    
    def print_summary(self, all_results: Dict[str, List[Dict[str, Any]]]):
        """Print comprehensive test summary."""
        print(f"\n{'='*80}")
        print(f"📊 COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*80}")
        
        total_files = 0
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_time = 0.0
        category_summaries = {}
        
        for category, results in all_results.items():
            if not results:
                continue
                
            category_files = len(results)
            category_tests = sum(r['test_count'] for r in results)
            category_passed = sum(r['passed'] for r in results)
            category_failed = sum(r['failed'] for r in results)
            category_time = sum(r['execution_time'] for r in results)
            category_success = all(r['success'] for r in results)
            
            total_files += category_files
            total_tests += category_tests
            total_passed += category_passed
            total_failed += category_failed
            total_time += category_time
            
            status_emoji = "✅" if category_success else "❌"
            print(f"\n{status_emoji} {category.upper()} TESTS:")
            print(f"   📁 Files: {category_files}")
            print(f"   🧪 Tests: {category_tests}")
            print(f"   ✅ Passed: {category_passed}")
            print(f"   ❌ Failed: {category_failed}")
            print(f"   ⏱️  Time: {category_time:.2f}s")
            
            category_summaries[category] = {
                'files': category_files,
                'tests': category_tests,
                'passed': category_passed,
                'failed': category_failed,
                'time': category_time,
                'success': category_success
            }
            
            # Show failed tests
            if category_failed > 0:
                failed_tests = [r for r in results if not r['success']]
                print(f"   💔 Failed test files:")
                for failed in failed_tests:
                    print(f"      • {os.path.basename(failed['file'])}")
        
        print(f"\n{'='*40}")
        print(f"🎯 OVERALL SUMMARY:")
        print(f"{'='*40}")
        print(f"📁 Total Files: {total_files}")
        print(f"🧪 Total Tests: {total_tests}")
        print(f"✅ Total Passed: {total_passed}")
        print(f"❌ Total Failed: {total_failed}")
        print(f"⏱️  Total Time: {total_time:.2f}s")
        
        success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
        overall_success = total_failed == 0
        
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if overall_success:
            print(f"\n🎉 ALL TESTS PASSED! 🎉")
            print(f"🚀 The ARC training system is ready for deployment!")
        else:
            print(f"\n⚠️  SOME TESTS FAILED")
            print(f"🔧 Please review and fix failing tests before deployment.")
        
        return {
            'overall_success': overall_success,
            'total_files': total_files,
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'success_rate': success_rate,
            'total_time': total_time,
            'category_summaries': category_summaries
        }
    
    def validate_test_environment(self) -> bool:
        """Validate that the test environment is properly set up."""
        print("🔍 Validating test environment...")
        
        # Check directory structure
        required_dirs = [
            self.tests_dir,
            self.tests_dir / "unit",
            self.tests_dir / "integration",
            self.tests_dir / "system",
            self.project_root / "src"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not dir_path.exists():
                missing_dirs.append(str(dir_path))
        
        if missing_dirs:
            print(f"❌ Missing directories: {missing_dirs}")
            return False
        
        # Check for pytest
        try:
            subprocess.run([sys.executable, '-m', 'pytest', '--version'], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ pytest not available. Install with: pip install pytest")
            return False
        
        # Check for key test files
        key_test_files = [
            'test_continuous_learning_loop.py',
            'test_train_arc_agent.py',
            'test_arc_training_pipeline.py'
        ]
        
        missing_tests = []
        for test_file in key_test_files:
            found = False
            for category in ['unit', 'integration']:
                if (self.tests_dir / category / test_file).exists():
                    found = True
                    break
            if not found:
                missing_tests.append(test_file)
        
        if missing_tests:
            print(f"❌ Missing key test files: {missing_tests}")
            return False
        
        print("✅ Test environment validation passed!")
        return True


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Test Suite Runner for ARC Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_comprehensive_tests.py --all                    # Run all tests
  python run_comprehensive_tests.py --category unit          # Run unit tests only
  python run_comprehensive_tests.py --category integration   # Run integration tests
  python run_comprehensive_tests.py --tests continuous energy # Run tests matching patterns
  python run_comprehensive_tests.py --validate              # Validate test environment
  python run_comprehensive_tests.py --list                  # List all available tests
        """
    )
    
    parser.add_argument('--all', action='store_true', 
                       help='Run all tests in all categories')
    parser.add_argument('--category', choices=['unit', 'integration', 'system'],
                       help='Run tests in a specific category')
    parser.add_argument('--tests', nargs='+', metavar='PATTERN',
                       help='Run tests matching specific patterns')
    parser.add_argument('--validate', action='store_true',
                       help='Validate test environment setup')
    parser.add_argument('--list', action='store_true',
                       help='List all available tests')
    parser.add_argument('--quiet', action='store_true',
                       help='Run tests in quiet mode')
    parser.add_argument('--summary-only', action='store_true',
                       help='Show only the final summary')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = ARCTestRunner()
    
    # Handle validation
    if args.validate:
        success = runner.validate_test_environment()
        sys.exit(0 if success else 1)
    
    # Handle test discovery
    if args.list:
        test_categories = runner.discover_tests()
        print("\n📋 Available Tests:")
        print("="*50)
        for category, tests in test_categories.items():
            print(f"\n{category.upper()} ({len(tests)} files):")
            for test in tests:
                print(f"  • {os.path.basename(test)}")
        return
    
    # Validate environment before running tests
    if not runner.validate_test_environment():
        print("❌ Test environment validation failed!")
        sys.exit(1)
    
    verbose = not args.quiet
    
    # Run tests based on arguments
    if args.all:
        all_results = runner.run_all_tests(verbose)
        summary = runner.print_summary(all_results)
        sys.exit(0 if summary['overall_success'] else 1)
    
    elif args.category:
        results = runner.run_category(args.category, verbose)
        if results:
            all_results = {args.category: results}
            summary = runner.print_summary(all_results)
            sys.exit(0 if summary['overall_success'] else 1)
        else:
            sys.exit(1)
    
    elif args.tests:
        results = runner.run_specific_tests(args.tests, verbose)
        if results:
            all_results = {'specific': results}
            summary = runner.print_summary(all_results)
            sys.exit(0 if summary['overall_success'] else 1)
        else:
            sys.exit(1)
    
    else:
        # Default: show help and run basic validation
        parser.print_help()
        print("\n" + "="*60)
        print("💡 Quick Start:")
        print("  python run_comprehensive_tests.py --all          # Run everything")
        print("  python run_comprehensive_tests.py --category unit # Run unit tests")
        print("  python run_comprehensive_tests.py --validate     # Check setup")
        

if __name__ == '__main__':
    main()
