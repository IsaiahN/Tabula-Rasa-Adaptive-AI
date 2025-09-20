#!/usr/bin/env python3

"""
Test all critical fixes with a short training session.
"""

import sys
import os
import json
import asyncio
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training import MasterARCTrainer, TrainingConfig

def test_critical_fixes_integration():
    """Run the async critical fixes integration test via asyncio.run so pytest
    doesn't require an async plugin."""
    async def _run():
        print("🧪 TESTING ALL CRITICAL FIXES WITH SHORT TRAINING SESSION")
        print("=" * 70)
        
        try:
            # Create a minimal test configuration
            config = TrainingConfig(
                mode="test-fixes",
                api_key=os.getenv('ARC_API_KEY', 'test_key'),
                arc_agents_path="C:\\Users\\Admin\\Documents\\GitHub\\ARC-AGI-3-Agents",
                max_actions_per_game=25,  # Limited actions for quick test
                max_learning_cycles=3,    # Only 3 games for validation
                target_score=50.0,        # Lower threshold for testing
                
                # Enable critical systems for testing
                enable_frame_analysis=True,      # Fixed: frame analysis integration
                enable_action_intelligence=True, # Fixed: action effectiveness tracking
                enable_meta_learning=True,       # Fixed: learning feedback loop
                enable_energy_system=True,      # Keep energy management
                enable_salience_system=True,    # Keep salience for memory management
                
                # Disable heavy systems for quick test
                enable_swarm=False,
                enable_coordinates=False,  # Skip coordinates for now
                enable_sleep_cycles=False,
                enable_dnc_memory=False,
                enable_contrarian_strategy=False,
                enable_boundary_detection=False,
                enable_memory_consolidation=False,
                enable_goal_invention=False,
                enable_learning_progress_drive=False,
                enable_death_manager=False,
                enable_exploration_strategies=False,
                enable_pattern_recognition=False
            )
            
            print(f"📋 Test Configuration:")
            print(f"   • Max actions per game: {config.max_actions_per_game}")
            print(f"   • Max learning cycles: {config.max_learning_cycles}")
            print(f"   • Frame analysis: {config.enable_frame_analysis}")
            print(f"   • Action intelligence: {config.enable_action_intelligence}")
            print(f"   • Meta learning: {config.enable_meta_learning}")
            
            # Create trainer
            trainer = MasterARCTrainer(config)
            await trainer.initialize()
            
            print(f"\n🚀 Starting test training session...")
            results = await trainer.run()
            
            print(f"\n📊 TEST RESULTS:")
            print(f"   • Episodes completed: {results.get('episodes_completed', 0)}")
            print(f"   • Average score: {results.get('average_score', 0):.2f}")
            print(f"   • Win rate: {results.get('win_rate', 0)*100:.1f}%")
            print(f"   • Max score achieved: {results.get('max_score', 0)}")
            print(f"   • Actions per game: {results.get('average_actions_per_game', 0):.1f}")
            
            # Check if we improved from 0% performance
            win_rate = results.get('win_rate', 0)
            average_score = results.get('average_score', 0)
            episodes = results.get('episodes_completed', 0)
            
            if episodes > 0:
                if win_rate > 0 or average_score > 0:
                    print(f"\n✅ SUCCESS: System performance improved from 0%!")
                    print(f"   🎯 Win rate: {win_rate*100:.1f}% (was 0%)")
                    print(f"   🎯 Average score: {average_score:.2f} (was 0)")
                    return True
                else:
                    print(f"\n⚠️ PARTIAL: System ran but still at 0% performance")
                    print(f"   May need more games or additional fixes")
                    return False
            else:
                print(f"\n❌ FAILURE: No episodes completed")
                return False
                
        except Exception as e:
            print(f"❌ Critical fixes test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Run the async test body synchronously for pytest
    return __import__('asyncio').run(_run())

if __name__ == "__main__":
    success = asyncio.run(test_critical_fixes_integration())
    
    print("\n" + "=" * 70)
    if success:
        print("🎯 CRITICAL FIXES VALIDATION: SUCCESS")
        print("   System is ready for full training sessions!")
    else:
        print("⚠️ CRITICAL FIXES VALIDATION: NEEDS MORE WORK")
        print("   Additional debugging may be required.")
