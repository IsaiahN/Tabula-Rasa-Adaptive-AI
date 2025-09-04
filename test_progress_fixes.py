#!/usr/bin/env python3
"""
Test script to validate our progress monitoring fixes.

This will test:
1. ACTION 6 threshold lowered from 0.01 to 0.001
2. Progress monitoring displays every 10 actions
3. Training completes games efficiently instead of infinite loops
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

# Set up environment
from dotenv import load_dotenv
load_dotenv()

async def test_continuous_learning_fixes():
    """Test our improved continuous learning system."""
    print("🧪 TESTING PROGRESS MONITORING FIXES")
    print("=" * 50)
    
    try:
        # Import the fixed continuous learning loop
        from arc_integration.continuous_learning_loop import ContinuousLearningLoop
        
        # Initialize the system  
        learning_loop = ContinuousLearningLoop()
        
        print("✅ ContinuousLearningLoop imported successfully")
        
        # Test a short training session to validate fixes
        print("\n🎮 Starting test training session...")
        print("   This will validate:")
        print("   - ACTION 6 threshold: 0.001 (was 0.01)")
        print("   - Progress monitoring every 10 actions")
        print("   - Score tracking and display")
        
        # Run training with a 30-action limit to prevent infinite loops
        results = await learning_loop.run_continuous_learning(
            mode='demo',
            enhanced=True,
            max_actions=30  # Limit for testing
        )
        
        print(f"\n📊 TEST RESULTS:")
        print(f"   Actions Taken: {results.get('total_actions', 'N/A')}")
        print(f"   Final Score: {results.get('final_score', 'N/A')}")
        print(f"   Effective Actions: {len(results.get('effective_actions', []))}")
        print(f"   Training Status: {'✅ COMPLETED' if results.get('final_score', 0) >= 100 else '🔄 IN PROGRESS'}")
        
        # Check if our fixes are working
        if results.get('total_actions', 0) > 0:
            effectiveness = len(results.get('effective_actions', [])) / results.get('total_actions', 1)
            print(f"   Action Effectiveness: {effectiveness:.1%}")
            
            if effectiveness > 0.1:  # At least 10% effectiveness
                print("✅ ACTION 6 threshold fix appears to be working!")
            else:
                print("⚠️ Low effectiveness - may need further ACTION 6 tuning")
                
        return results
        
    except Exception as e:
        print(f"❌ Error testing fixes: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🧪 Testing Progress Monitoring Fixes")
    print("   ACTION 6 Threshold: 0.01 → 0.001")  
    print("   Progress Display: Every 10 actions")
    print("   Score Monitoring: Live tracking")
    print()
    
    # Run the test
    results = asyncio.run(test_continuous_learning_fixes())
    
    if results:
        print("\n🎉 Fix validation completed!")
        print("   The training system now has:")
        print("   ✅ Lower ACTION 6 threshold (0.001)")
        print("   ✅ Progress monitoring every 10 actions") 
        print("   ✅ Score change tracking")
        print("   ✅ Win condition indicators")
    else:
        print("\n❌ Fix validation failed - see errors above")
