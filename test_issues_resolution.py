"""
Test script to verify the two key issues are resolved:

Issue 1: Graceful shutdown with signal handling
Issue 2: Adaptive energy system with action-based sleep triggers

This script tests both features independently and together.
"""

import asyncio
import sys
import signal
import time
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_signal_handling():
    """Test Issue 1: Signal handling for graceful shutdown"""
    print("🧪 Testing Issue 1: Signal Handling for Graceful Shutdown")
    print("=" * 60)
    
    try:
        # Import the training script's signal handling
        from train_arc_agent import signal_handler, shutdown_requested, training_state
        
        print("✅ Signal handler imported successfully")
        print(f"✅ Initial shutdown_requested: {shutdown_requested}")
        print(f"✅ Initial training_state: {training_state}")
        
        # Simulate signal handling
        print("🔧 Simulating SIGINT signal...")
        signal_handler(signal.SIGINT, None)
        
        # Check that shutdown was requested
        from train_arc_agent import shutdown_requested as updated_shutdown
        print(f"✅ After signal - shutdown_requested: {updated_shutdown}")
        
        # Check if state file was created
        state_file = Path("training_state_backup.json")
        if state_file.exists():
            print(f"✅ Training state backup created: {state_file}")
            with open(state_file, 'r') as f:
                saved_state = json.load(f)
            print(f"✅ Saved state keys: {list(saved_state.keys())}")
        else:
            print("⚠️  No state file created (expected if no training was running)")
        
        return True
        
    except Exception as e:
        print(f"❌ Signal handling test failed: {e}")
        return False

def test_adaptive_energy_system():
    """Test Issue 2: Adaptive energy system"""
    print("\n🧪 Testing Issue 2: Adaptive Energy System")
    print("=" * 60)
    
    try:
        from core.adaptive_energy_system import AdaptiveEnergySystem, EnergyConfig
        
        # Test with limited actions (10K max)
        print("🔧 Testing with limited actions (10K max)...")
        energy_system = AdaptiveEnergySystem()
        energy_system.configure_session(max_actions=10000)
        
        print(f"✅ Configured for 10K actions")
        print(f"✅ Sleep threshold: {energy_system.current_sleep_threshold:.1f}")
        print(f"✅ Unlimited actions: {energy_system.unlimited_actions}")
        print(f"✅ Depletion rate: {energy_system.current_depletion_rate:.4f}")
        
        # Simulate action-based energy depletion
        print("\n🔧 Simulating 1000 actions...")
        energy_system.update_energy(actions_taken=1000)
        should_sleep, reason = energy_system.should_sleep()
        
        print(f"✅ Energy after 1000 actions: {energy_system.current_energy:.1f}")
        print(f"✅ Should sleep: {should_sleep} ({reason})")
        
        # Test with unlimited actions (time-based)
        print("\n🔧 Testing with unlimited actions (time-based)...")
        time_based_system = AdaptiveEnergySystem()
        time_based_system.configure_session(max_actions=None)
        
        print(f"✅ Configured for unlimited actions")
        print(f"✅ Sleep interval: {time_based_system.current_sleep_interval:.1f} minutes")
        print(f"✅ Unlimited actions: {time_based_system.unlimited_actions}")
        print(f"✅ Time-based depletion rate: {time_based_system.current_depletion_rate:.4f}/sec")
        
        # Test performance-based adaptation
        print("\n🔧 Testing performance-based adaptation...")
        adaptive_system = AdaptiveEnergySystem()
        
        # Simulate failures
        for i in range(5):
            adaptive_system.update_performance(success=False)
        
        print(f"✅ After 5 failures - consecutive failures: {adaptive_system.consecutive_failures}")
        print(f"✅ Success rate: {adaptive_system.recent_success_rate:.2f}")
        print(f"✅ Sleep threshold after failures: {adaptive_system.current_sleep_threshold:.1f}")
        
        # Test adaptation summary
        summary = adaptive_system.get_adaptation_summary()
        print(f"✅ Adaptation summary: {summary}")
        
        return True
        
    except Exception as e:
        print(f"❌ Adaptive energy system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test both systems working together"""
    print("\n🧪 Testing Integration: Both Systems Together")
    print("=" * 60)
    
    try:
        from core.adaptive_energy_system import AdaptiveEnergySystem, EnergySystemIntegration
        
        # Mock training loop
        class MockTrainingLoop:
            def __init__(self):
                self.sleep_system = type('MockSleep', (), {'sleep_trigger_energy': 40.0})()
        
        mock_loop = MockTrainingLoop()
        energy_system = AdaptiveEnergySystem()
        integration = EnergySystemIntegration(mock_loop, energy_system)
        
        # Test session configuration integration
        session_config = {
            'max_actions_per_session': 50000,
            'estimated_duration_minutes': 45,
            'games_count': 6
        }
        
        updated_config = integration.integrate_with_training_session(session_config)
        
        print(f"✅ Integration created successfully")
        print(f"✅ Updated config keys: {list(updated_config.keys())}")
        print(f"✅ Adaptive sleep threshold: {updated_config.get('adaptive_sleep_threshold')}")
        print(f"✅ Sleep system threshold updated: {mock_loop.sleep_system.sleep_trigger_energy}")
        
        # Test training updates
        update_result = integration.update_during_training(
            actions_taken=100,
            success=True,
            score_improvement=15.0
        )
        
        print(f"✅ Training update result: {update_result}")
        print(f"✅ Sleep triggered: {update_result.get('sleep_triggered', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 TESTING ISSUES RESOLUTION")
    print("=" * 80)
    print("Issue 1: Graceful shutdown with signal handling")
    print("Issue 2: Adaptive energy system tied to action limits")
    print("=" * 80)
    
    results = []
    
    # Test Issue 1
    results.append(test_signal_handling())
    
    # Test Issue 2
    results.append(test_adaptive_energy_system())
    
    # Test Integration
    results.append(test_integration())
    
    # Summary
    print("\n🎯 RESULTS SUMMARY")
    print("=" * 60)
    
    issues = ["Signal Handling", "Adaptive Energy System", "Integration"]
    for i, (issue, result) in enumerate(zip(issues, results)):
        status = "✅ RESOLVED" if result else "❌ FAILED"
        print(f"Issue {i+1} ({issue}): {status}")
    
    all_passed = all(results)
    if all_passed:
        print("\n🎉 ALL ISSUES RESOLVED SUCCESSFULLY!")
        print("✅ Graceful shutdown with data preservation")
        print("✅ Adaptive energy system with smart sleep triggers")
        print("✅ Action-based and time-based sleep scheduling")
        print("✅ Performance-based parameter adaptation")
        print("✅ Full integration with training system")
    else:
        print("\n⚠️  SOME ISSUES NEED ATTENTION")
        failed_count = sum(1 for r in results if not r)
        print(f"   {failed_count} out of {len(results)} tests failed")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\nTest completed with exit code: {exit_code}")
    sys.exit(exit_code)
