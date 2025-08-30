#!/usr/bin/env python3
"""
Unified Test Runner for Tabula Rasa

Clearly separates different types of testing:
- Regular unit/integration tests
- ARC-3 competition testing
- Performance benchmarks
- System validation

Usage:
    # Regular testing
    python run_tests.py --type unit                    # Unit tests only
    python run_tests.py --type integration            # Integration tests
    python run_tests.py --type system                 # Full system tests
    python run_tests.py --type all                    # All regular tests
    
    # ARC-3 Competition testing
    python run_tests.py --type arc3 --mode demo       # Quick ARC-3 demo
    python run_tests.py --type arc3 --mode full       # Full ARC-3 training
    python run_tests.py --type arc3 --mode comparison # Compare strategies
    
    # Performance testing
    python run_tests.py --type performance            # Performance benchmarks
    python run_tests.py --type agi-puzzles           # AGI puzzle evaluation
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_unit_tests():
    """Run unit tests using pytest."""
    import subprocess
    print("🧪 Running Unit Tests")
    print("=" * 50)
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/unit/", 
        "-v", 
        "--tb=short"
    ], capture_output=False)
    
    return result.returncode == 0

def run_integration_tests():
    """Run integration tests."""
    import subprocess
    print("🔗 Running Integration Tests")
    print("=" * 50)
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/integration/", 
        "-v", 
        "--tb=short"
    ], capture_output=False)
    
    return result.returncode == 0

def run_system_tests():
    """Run system-level tests."""
    import subprocess
    print("🖥️ Running System Tests")
    print("=" * 50)
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/system/", 
        "-v", 
        "--tb=short"
    ], capture_output=False)
    
    return result.returncode == 0

def run_performance_tests():
    """Run performance benchmarks."""
    print("⚡ Running Performance Tests")
    print("=" * 50)
    
    try:
        from experiments.phase0_lp_validation import main as lp_validation
        from experiments.phase0_memory_test import main as memory_test
        from experiments.phase0_survival_test import main as survival_test
        
        print("📊 Learning Progress Validation...")
        lp_validation()
        
        print("🧠 Memory System Test...")
        memory_test()
        
        print("💀 Survival System Test...")
        survival_test()
        
        return True
    except Exception as e:
        logger.error(f"Performance tests failed: {e}")
        return False

def run_agi_puzzles():
    """Run AGI puzzle evaluation."""
    print("🧩 Running AGI Puzzle Evaluation")
    print("=" * 50)
    
    try:
        # Import and run AGI puzzle tests
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/integration/test_agent_on_puzzles.py", 
            "-v", 
            "--tb=short"
        ], capture_output=False)
        
        return result.returncode == 0
    except Exception as e:
        logger.error(f"AGI puzzle tests failed: {e}")
        return False

async def run_arc3_tests(mode: str = "demo"):
    """Run ARC-3 competition tests."""
    print(f"🏆 Running ARC-3 Competition Tests (Mode: {mode})")
    print("=" * 50)
    print("🌐 REAL ARC-3 API CONNECTION")
    print("📊 Results will be recorded on official leaderboard")
    print("=" * 50)
    
    try:
        from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop
        
        # Find ARC-AGI-3-Agents path
        arc_agents_path = os.getenv('ARC_AGENTS_PATH')
        if not arc_agents_path:
            possible_paths = [
                Path.cwd().parent / "ARC-AGI-3-Agents",
                Path.cwd() / "ARC-AGI-3-Agents",
                Path("C:/Users/Admin/Documents/GitHub/ARC-AGI-3-Agents")
            ]
            
            for path in possible_paths:
                if path.exists() and (path / "main.py").exists():
                    arc_agents_path = str(path)
                    break
                    
        if not arc_agents_path:
            print("❌ ARC-AGI-3-Agents repository not found")
            return False
            
        # Check API key
        api_key = os.getenv('ARC_API_KEY')
        if not api_key:
            print("❌ ARC_API_KEY not found in environment")
            print("💡 Set your API key from https://three.arcprize.org")
            return False
            
        print(f"✅ ARC-AGI-3-Agents found at: {arc_agents_path}")
        print(f"✅ API Key: {api_key[:8]}...{api_key[-4:]}")
        
        # Create learning loop
        learning_loop = ContinuousLearningLoop(
            arc_agents_path=arc_agents_path,
            tabula_rasa_path=str(Path.cwd()),
            api_key=api_key
        )
        
        # Run appropriate mode
        if mode == "demo":
            results = await learning_loop.run_demo_mode()
        elif mode == "full":
            results = await learning_loop.run_full_training_mode()
        elif mode == "comparison":
            results = await learning_loop.run_comparison_mode()
        else:
            raise ValueError(f"Unknown ARC-3 mode: {mode}")
            
        print(f"🎉 ARC-3 testing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"ARC-3 tests failed: {e}")
        return False

def main():
    """Main test runner with clear separation of test types."""
    parser = argparse.ArgumentParser(description='Unified Test Runner for Tabula Rasa')
    parser.add_argument('--type', 
                        choices=['unit', 'integration', 'system', 'all', 'arc3', 'performance', 'agi-puzzles'], 
                        default='all',
                        help='Type of tests to run')
    parser.add_argument('--mode', 
                        choices=['demo', 'full', 'comparison'], 
                        default='demo',
                        help='ARC-3 test mode (only used with --type arc3)')
    
    args = parser.parse_args()
    
    print("🚀 Tabula Rasa Test Runner")
    print("=" * 50)
    print(f"Test Type: {args.type}")
    if args.type == 'arc3':
        print(f"ARC-3 Mode: {args.mode}")
        print("🌐 CONNECTING TO REAL ARC-3 SERVERS")
    print("=" * 50)
    
    success = True
    
    if args.type == 'unit':
        success &= run_unit_tests()
    elif args.type == 'integration':
        success &= run_integration_tests()
    elif args.type == 'system':
        success &= run_system_tests()
    elif args.type == 'all':
        success &= run_unit_tests()
        success &= run_integration_tests()
        success &= run_system_tests()
    elif args.type == 'arc3':
        success &= asyncio.run(run_arc3_tests(args.mode))
    elif args.type == 'performance':
        success &= run_performance_tests()
    elif args.type == 'agi-puzzles':
        success &= run_agi_puzzles()
    else:
        print(f"❌ Unknown test type: {args.type}")
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()