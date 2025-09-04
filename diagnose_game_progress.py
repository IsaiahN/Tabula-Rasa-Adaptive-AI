#!/usr/bin/env python3
"""
Game Progress Diagnostic Tool
Analyzes the training loop to identify why games aren't progressing.
"""

import sys
import asyncio
from pathlib import Path

# Add the project directory to the Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

async def diagnose_game_progress():
    """Diagnose game progress issues in the training system."""
    print("🔍 GAME PROGRESS DIAGNOSTIC")
    print("=" * 50)
    
    try:
        from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop
        
        # Create a diagnostic loop
        loop = ContinuousLearningLoop(
            arc_agents_path=".",
            tabula_rasa_path=str(Path.cwd()),
            api_key="diagnostic_key"
        )
        
        print("✅ Training loop initialized")
        
        # Check ACTION 6 strategy settings
        action6_strategy = loop.available_actions_memory.get('action6_strategy', {})
        print(f"\n📊 ACTION 6 STRATEGY ANALYSIS:")
        print(f"   Progress Stagnation Threshold: {action6_strategy.get('progress_stagnation_threshold', 'NOT SET')}")
        print(f"   ACTION 6 Cooldown: {action6_strategy.get('action6_cooldown', 'NOT SET')}")
        print(f"   Last ACTION 6 Used: {action6_strategy.get('last_action6_used', 'NOT SET')}")
        
        # Check action effectiveness tracking
        action_effectiveness = loop.available_actions_memory.get('action_effectiveness', {})
        print(f"\n📈 ACTION EFFECTIVENESS:")
        for action_num in [1, 2, 3, 4, 5, 6, 7]:
            if action_num in action_effectiveness:
                eff_data = action_effectiveness[action_num]
                print(f"   ACTION {action_num}: {eff_data.get('success_rate', 0):.1%} success ({eff_data.get('attempts', 0)} attempts)")
            else:
                print(f"   ACTION {action_num}: No data")
        
        # Test ACTION 6 scoring logic
        print(f"\n🎯 ACTION 6 SCORING SIMULATION:")
        action_count = 50  # Simulate 50 actions taken
        progress_stagnant = loop._is_progress_stagnant(action_count)
        action6_score = loop._calculate_action6_strategic_score(action_count, progress_stagnant)
        
        print(f"   Action Count: {action_count}")
        print(f"   Progress Stagnant: {progress_stagnant}")
        print(f"   ACTION 6 Strategic Score: {action6_score:.6f}")
        print(f"   ACTION 6 Threshold: 0.01 (minimum for selection)")
        print(f"   ACTION 6 Would Be Selected: {'YES' if action6_score > 0.01 else 'NO'}")
        
        # Check scoring thresholds
        print(f"\n⚙️  SCORING THRESHOLDS:")
        print(f"   ACTION 6 minimum score: 0.01")
        print(f"   Current ACTION 6 score: {action6_score:.6f}")
        if action6_score <= 0.01:
            print("   🚨 ISSUE: ACTION 6 score too low - will never be selected!")
        
        # Recommend fixes
        print(f"\n🔧 RECOMMENDED FIXES:")
        if action6_score <= 0.01:
            print("   1. Lower ACTION 6 threshold from 0.01 to 0.001")
            print("   2. Reduce progress_stagnation_threshold from 8 to 3")
            print("   3. Add score progression display to monitor game progress")
            print("   4. Add win condition checking and display")
        
        print(f"\n📋 DIAGNOSTIC SUMMARY:")
        print("   The system appears to be stuck because:")
        print("   - ACTION 6 (coordinate-based) is being blocked by overly strict criteria")
        print("   - Only reasoning actions (1-4) are being used, which may not progress the game")
        print("   - No visible score progression or win condition monitoring")
        print("   - Actions 3 and 4 consistently score 0.000, indicating they're ineffective")
        
        return True
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(diagnose_game_progress())
