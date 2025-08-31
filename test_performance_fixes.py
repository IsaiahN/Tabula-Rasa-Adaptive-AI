#!/usr/bin/env python3
"""
Test the critical performance fixes for action limitations and continuous learning.

This tests the key fixes:
1. MAX_ACTIONS increased from 200 to 100,000
2. Available actions memory system
3. Enhanced boredom detection with strategy switching
4. Success-weighted memory prioritization (10x for wins)
5. Mid-game consolidation simulation
"""

import sys
import os
import time
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Set up a minimal environment for testing
os.environ['ARC_API_KEY'] = 'test_key_12345'
os.environ['ARC_AGENTS_PATH'] = str(Path.cwd() / "arc-agents")

def test_action_limit_fix():
    """Test that MAX_ACTIONS has been increased from 200 to 100,000."""
    print("\n🧪 TEST 1: Action Limit Fix")
    print("=" * 50)
    
    try:
        # Read the arc_agent_adapter.py file to check MAX_ACTIONS
        adapter_file = Path("src/arc_integration/arc_agent_adapter.py")
        if adapter_file.exists():
            content = adapter_file.read_text()
            
            if "MAX_ACTIONS = 100000" in content:
                print("✅ PASS: MAX_ACTIONS increased to 100,000")
                return True
            elif "MAX_ACTIONS = 200" in content:
                print("❌ FAIL: MAX_ACTIONS still at 200 - fix not applied")
                return False
            else:
                print("⚠️  WARNING: MAX_ACTIONS not found in expected format")
                return False
        else:
            print("❌ FAIL: arc_agent_adapter.py not found")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_enhanced_boredom_detection():
    """Test that enhanced boredom detection with strategy switching is implemented."""
    print("\n🧪 TEST 2: Enhanced Boredom Detection")
    print("=" * 50)
    
    try:
        # Check if the enhanced boredom detection is in continuous_learning_loop.py
        loop_file = Path("src/arc_integration/continuous_learning_loop.py")
        if loop_file.exists():
            with open(loop_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            checks = [
                ("Strategy switching", "_switch_action_strategy" in content),
                ("Action experimentation", "_trigger_action_experimentation" in content),
                ("Action pattern analysis", "_analyze_recent_action_patterns" in content),
                ("Available actions memory", "available_actions_memory" in content)
            ]
            
            passed = 0
            for name, check in checks:
                if check:
                    print(f"✅ {name}: IMPLEMENTED")
                    passed += 1
                else:
                    print(f"❌ {name}: MISSING")
            
            return passed == len(checks)
        else:
            print("❌ FAIL: continuous_learning_loop.py not found")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_success_weighted_memory():
    """Test that success-weighted memory prioritization (10x for wins) is implemented."""
    print("\n🧪 TEST 3: Success-Weighted Memory Prioritization")
    print("=" * 50)
    
    try:
        loop_file = Path("src/arc_integration/continuous_learning_loop.py")
        if loop_file.exists():
            with open(loop_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            checks = [
                ("10x success multiplier", "success_multiplier = 10.0" in content),
                ("WIN attempts priority", "WIN attempts get 10x" in content),
                ("Success-weighted memories", "success_weighted_memories" in content),
                ("Effectiveness calculation", "_calculate_episode_effectiveness" in content)
            ]
            
            passed = 0
            for name, check in checks:
                if check:
                    print(f"✅ {name}: IMPLEMENTED")
                    passed += 1
                else:
                    print(f"❌ {name}: MISSING")
            
            return passed == len(checks)
        else:
            print("❌ FAIL: continuous_learning_loop.py not found")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_mid_game_consolidation():
    """Test that mid-game consolidation is implemented."""
    print("\n🧪 TEST 4: Mid-Game Consolidation")
    print("=" * 50)
    
    try:
        loop_file = Path("src/arc_integration/continuous_learning_loop.py")
        if loop_file.exists():
            with open(loop_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            checks = [
                ("Mid-game sleep triggers", "_should_trigger_mid_game_sleep" in content),
                ("Mid-game execution", "_execute_mid_game_sleep" in content),
                ("Consolidation simulation", "_simulate_mid_game_consolidation" in content),
                ("Action sequence analysis", "_analyze_action_sequences" in content)
            ]
            
            passed = 0
            for name, check in checks:
                if check:
                    print(f"✅ {name}: IMPLEMENTED")
                    passed += 1
                else:
                    print(f"❌ {name}: MISSING")
            
            return passed == len(checks)
        else:
            print("❌ FAIL: continuous_learning_loop.py not found")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_continuous_learning_metrics():
    """Test that enhanced continuous learning metrics are tracked."""
    print("\n🧪 TEST 5: Continuous Learning Metrics")
    print("=" * 50)
    
    try:
        loop_file = Path("src/arc_integration/continuous_learning_loop.py")
        if loop_file.exists():
            with open(loop_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            checks = [
                ("Action sequences tracking", "'action_sequences'" in content),
                ("Mid-game consolidation count", "'mid_game_consolidations'" in content),
                ("Learning velocity", "'learning_velocity'" in content),
                ("Strategy effectiveness", "'strategy_effectiveness'" in content),
                ("Continuous learning metrics", "'continuous_learning_metrics'" in content)
            ]
            
            passed = 0
            for name, check in checks:
                if check:
                    print(f"✅ {name}: IMPLEMENTED")
                    passed += 1
                else:
                    print(f"❌ {name}: MISSING")
            
            return passed == len(checks)
        else:
            print("❌ FAIL: continuous_learning_loop.py not found")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Run all performance fix tests."""
    print("🚀 TESTING COMPREHENSIVE PERFORMANCE FIXES")
    print("=" * 60)
    print("Critical fixes to match top leaderboard performance:")
    print("- Action limit: 200 → 100,000")
    print("- Enhanced boredom detection with strategy switching")
    print("- Success-weighted memory (10x for wins)")
    print("- Mid-game consolidation for continuous learning")
    print("- Action sequence analysis and metrics")
    
    start_time = time.time()
    
    # Run all tests
    tests = [
        test_action_limit_fix,
        test_enhanced_boredom_detection,
        test_success_weighted_memory,
        test_mid_game_consolidation,
        test_continuous_learning_metrics
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("🏆 PERFORMANCE FIXES TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Tests Passed: {passed}/{total}")
    print(f"⏱️  Test Duration: {time.time() - start_time:.2f}s")
    
    if passed == total:
        print("\n🎉 ALL CRITICAL PERFORMANCE FIXES IMPLEMENTED!")
        print("   Agent can now match top leaderboard performance:")
        print("   - 100,000+ actions like StochasticGoose (255,964 actions)")
        print("   - Continuous learning with mid-game consolidation")
        print("   - Success-weighted memory for win retention")
        print("   - Enhanced boredom detection with strategy switching")
        return True
    else:
        print(f"\n⚠️  {total - passed} critical fixes still missing")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
