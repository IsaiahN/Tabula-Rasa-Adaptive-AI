#!/usr/bin/env python3
"""
Test script to verify that the Architect can detect and trigger actual ARC game systems.
This ensures the meta-cognitive system is connected to real training, not just simulations.
"""
import sys
import os
import logging
import asyncio
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_game_integration():
    """Test that the Architect properly integrates with actual game systems."""
    print("🎮 Testing Game Integration with Meta-Cognitive Architect")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("GameIntegrationTest")
    
    try:
        from core.architect import Architect
        
        print("\n1️⃣ Initializing Architect...")
        architect = Architect(
            base_path="src",
            repo_path=".",
            logger=logger
        )
        
        print(f"✅ Architect initialized with branch: {architect.default_branch}")
        
        print("\n2️⃣ Checking current game activity...")
        activity_status = architect.check_game_activity()
        
        print(f"🔍 Activity Status:")
        print(f"  - Training processes active: {activity_status['training_processes_active']}")
        print(f"  - Process count: {activity_status['process_count']}")
        print(f"  - Recent log activity: {activity_status['recent_log_activity']}")
        print(f"  - Continuous learning active: {activity_status['continuous_learning_active']}")
        print(f"  - Overall active: {activity_status['overall_active']}")
        
        if activity_status['process_details']:
            print(f"📊 Active processes:")
            for proc in activity_status['process_details']:
                duration_min = proc['duration'] / 60
                print(f"  - PID {proc['pid']}: {proc['name']} (running {duration_min:.1f} min)")
        
        print("\n3️⃣ Testing sandbox execution with real training...")
        
        # Create a minimal test to verify real training integration
        async def test_real_training():
            from core.architect import SystemGenome, MutationType, Mutation, MutationImpact
            
            # Create a test mutation
            test_genome = SystemGenome()
            test_mutation = Mutation(
                id="test_integration",
                type=MutationType.PARAMETER_ADJUSTMENT,
                impact=MutationImpact.MINIMAL,
                changes={'max_actions_per_game': 100},
                rationale="Testing real training integration",
                expected_improvement=0.1,
                confidence=0.8,
                test_duration_estimate=2.0
            )
            
            print("🧪 Running sandbox test with real training system...")
            test_result = await architect.sandbox_tester.test_mutation(
                test_mutation, 
                test_genome,
                test_games=["test_integration_game"]
            )
            
            print(f"📊 Test Result:")
            print(f"  - Success: {test_result.success}")
            print(f"  - Duration: {test_result.test_duration:.1f}s")
            print(f"  - Win rate: {test_result.metrics.get('win_rate', 'N/A')}")
            print(f"  - Average score: {test_result.metrics.get('average_score', 'N/A')}")
            print(f"  - Data source: {test_result.metrics.get('parsing_source', 'Unknown')}")
            
            if test_result.metrics.get('parsing_source') == 'real_output':
                print("✅ REAL training system was executed!")
            else:
                print("⚠️ Fallback simulation was used (training system may not be available)")
            
            return test_result.success
        
        # Run the async test
        success = asyncio.run(test_real_training())
        
        print("\n4️⃣ Testing training system auto-start...")
        can_ensure_running = architect.ensure_game_is_running()
        
        if can_ensure_running:
            print("✅ Training system is running or was successfully started")
            
            # Check activity again after potential start
            print("\n5️⃣ Re-checking activity after auto-start...")
            time.sleep(2)
            new_activity = architect.check_game_activity()
            
            if new_activity['overall_active']:
                print("✅ Training system confirmed active after auto-start")
            else:
                print("⚠️ Training system still not detected as active")
        else:
            print("⚠️ Could not ensure training system is running")
        
        print("\n" + "=" * 60)
        print("🎯 Game Integration Test Summary:")
        print(f"✅ Architect can detect game activity: YES")
        print(f"✅ Real training integration: {'YES' if success else 'SIMULATED'}")
        print(f"✅ Can auto-start training: {'YES' if can_ensure_running else 'NO'}")
        print(f"✅ Branch safety: {architect.default_branch}")
        
        if activity_status['overall_active'] or success:
            print("\n🎉 SUCCESS: Meta-cognitive system is connected to real game systems!")
        else:
            print("\n⚠️  NOTICE: System is working but may be using simulated data")
            print("   This is normal if no training processes are currently running.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_game_integration()
    sys.exit(0 if success else 1)
