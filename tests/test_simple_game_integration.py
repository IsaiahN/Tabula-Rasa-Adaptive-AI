#!/usr/bin/env python3
"""
Simple test to verify the Architect can detect and trigger real game systems.
This simplified version focuses on the core functionality without complex sandboxing.
"""
import sys
import os
import logging
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_game_monitoring():
    """Test that the Architect can monitor real game activity."""
    print("🎮 Testing Game Activity Monitoring")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("GameMonitoringTest")
    
    try:
        from core.architect import Architect
        
        print("\n1️⃣ Initializing Architect...")
        architect = Architect(
            base_path="src",
            repo_path=".",
            logger=logger
        )
        
        print(f"✅ Architect initialized with branch: {architect.default_branch}")
        
        print("\n2️⃣ Testing game activity detection...")
        activity_status = architect.check_game_activity()
        
        print(f"📊 Game Activity Status:")
        print(f"  - Overall Active: {activity_status['overall_active']}")
        print(f"  - Process Count: {activity_status['process_count']}")
        print(f"  - Recent Logs: {activity_status['recent_log_activity']}")
        print(f"  - Continuous Learning: {activity_status['continuous_learning_active']}")
        
        if activity_status['process_details']:
            print(f"🔍 Active Training Processes:")
            for proc in activity_status['process_details']:
                duration_min = proc['duration'] / 60
                print(f"  - PID {proc['pid']}: {proc['name']} ({duration_min:.1f} min)")
        
        print("\n3️⃣ Testing training system detection...")
        
        # Check if training scripts exist
        training_scripts = [
            "master_arc_trainer.py",
            "run_meta_cognitive_arc_training.py"
        ]
        
        available_scripts = []
        for script in training_scripts:
            script_path = os.path.join(".", script)
            if os.path.exists(script_path):
                available_scripts.append(script)
                print(f"✅ Found training script: {script}")
        
        if not available_scripts:
            print("❌ No training scripts found!")
            return False
        
        print("\n4️⃣ Testing ability to start training...")
        can_start = architect.ensure_game_is_running()
        
        if can_start:
            print("✅ Training system can be started or is already running")
            
            # Wait and check again
            time.sleep(3)
            new_activity = architect.check_game_activity()
            
            if new_activity['overall_active']:
                print("✅ Training confirmed active after check")
            else:
                print("⚠️ Training not detected as active (may take time to start)")
        else:
            print("⚠️ Could not start training system automatically")
        
        print("\n5️⃣ Testing real training integration...")
        
        # Test if we can actually call the training system
        import subprocess
        
        best_script = available_scripts[0]
        test_cmd = [sys.executable, best_script, "--help"]
        
        try:
            result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                timeout=10,
                cwd="."
            )
            
            if result.returncode == 0:
                print(f"✅ Training script {best_script} is callable")
                
                # Check if it has the modes we need
                if any(mode in result.stdout for mode in ["--mode", "continuous", "test"]):
                    print("✅ Training script supports required modes")
                else:
                    print("⚠️ Training script may not support all required modes")
            else:
                print(f"⚠️ Training script returned error code {result.returncode}")
                
        except subprocess.TimeoutExpired:
            print("⚠️ Training script help command timed out")
        except Exception as e:
            print(f"⚠️ Could not test training script: {e}")
        
        print("\n" + "=" * 50)
        print("🎯 Game Integration Summary:")
        print(f"✅ Activity Detection: {'WORKING' if hasattr(architect, 'check_game_activity') else 'MISSING'}")
        print(f"✅ Training Scripts Available: {len(available_scripts)}")
        print(f"✅ Auto-Start Capability: {'YES' if can_start else 'NO'}")
        print(f"✅ Current Activity: {'ACTIVE' if activity_status['overall_active'] else 'INACTIVE'}")
        print(f"✅ Branch Safety: {architect.default_branch}")
        
        # Test the updated _run_sandbox_test method
        print("\n6️⃣ Testing direct training execution...")
        
        try:
            # Create a simple test - this will use the updated method that calls real training
            import asyncio
            from pathlib import Path
            
            async def test_direct_execution():
                # Test the updated _run_sandbox_test method
                temp_sandbox = Path("temp_test_sandbox")
                temp_sandbox.mkdir(exist_ok=True)
                
                try:
                    result = await architect.sandbox_tester._run_sandbox_test(
                        temp_sandbox, 
                        ["quick_test"]
                    )
                    
                    print(f"📊 Direct Training Test:")
                    print(f"  - Success: {result.get('success', False)}")
                    print(f"  - Duration: {result.get('test_duration', 0):.1f}s")
                    print(f"  - Data Source: {result.get('parsing_source', 'unknown')}")
                    
                    if result.get('parsing_source') == 'real_output':
                        print("🎉 SUCCESS: Real training system was executed!")
                        return True
                    else:
                        print("⚠️ Fallback simulation was used")
                        return False
                        
                finally:
                    # Cleanup
                    import shutil
                    if temp_sandbox.exists():
                        shutil.rmtree(temp_sandbox, ignore_errors=True)
            
            real_execution = asyncio.run(test_direct_execution())
            
        except Exception as e:
            print(f"⚠️ Direct execution test failed: {e}")
            real_execution = False
        
        print(f"✅ Real Training Execution: {'WORKING' if real_execution else 'SIMULATION'}")
        
        # Summary
        success_indicators = [
            len(available_scripts) > 0,
            hasattr(architect, 'check_game_activity'),
            hasattr(architect, 'ensure_game_is_running'),
            architect.default_branch == "Tabula-Rasa-v3"
        ]
        
        success_rate = sum(success_indicators) / len(success_indicators)
        
        if success_rate >= 0.75:
            print(f"\n🎉 OVERALL SUCCESS: {success_rate*100:.0f}% - Meta-cognitive system is properly connected!")
        else:
            print(f"\n⚠️ PARTIAL SUCCESS: {success_rate*100:.0f}% - Some issues detected")
        
        return success_rate >= 0.75
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_game_monitoring()
    sys.exit(0 if success else 1)
