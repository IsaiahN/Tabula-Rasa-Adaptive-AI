#!/usr/bin/env python3
"""
Full ARC-AGI-3 Training Session with Enhanced Continuous Learning

This script runs a complete training session with:
- Sleep state integration
- Memory consolidation tracking
- Game reset decision making
- SWARM mode for multiple games
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from arc_integration.continuous_learning_loop import ContinuousLearningLoop
    from core.salience_system import SalienceMode
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the tabula-rasa directory")
    sys.exit(1)

async def run_full_training():
    """Run a complete ARC-AGI-3 training session."""
    
    print("🚀 STARTING FULL ARC-AGI-3 TRAINING SESSION")
    print("="*60)
    
    # Initialize the continuous learning loop
    try:
        continuous_loop = ContinuousLearningLoop(
            arc_agents_path="C:/Users/Admin/Documents/GitHub/ARC-AGI-3-Agents",
            tabula_rasa_path="C:/Users/Admin/Documents/GitHub/tabula-rasa"
        )
        print("✅ Continuous Learning Loop initialized")
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Configure training parameters for real ARC games
    training_games = [
        "00d62c1b",  # Simple pattern recognition
        "007bbfb7",  # Color transformation  
        "017c7c7b",  # Shape completion
        "025d127b",  # Grid manipulation
        "045e512c"   # Pattern extension
    ]
    
    print(f"\n🎯 TRAINING CONFIGURATION:")
    print(f"Games: {len(training_games)}")
    print(f"SWARM Mode: {'ENABLED' if len(training_games) > 2 else 'DISABLED'}")
    print(f"Salience Mode: DECAY_COMPRESSION")
    print(f"Max Episodes per Game: 20")
    print(f"Target Win Rate: 30%")
    
    # Start training session
    try:
        session_id = continuous_loop.start_training_session(
            games=training_games,
            max_episodes_per_game=20,
            target_win_rate=0.3,
            target_avg_score=50.0,
            salience_mode=SalienceMode.DECAY_COMPRESSION,
            enable_salience_comparison=False
        )
        
        print(f"✅ Training session started: {session_id}")
        
        # Show initial system status
        print(f"\n🧠 INITIAL SYSTEM STATUS:")
        initial_status = continuous_loop.get_system_status_flags()
        for key, value in initial_status.items():
            if value:  # Only show active flags
                print(f"  ✅ {key}: {value}")
        
        # Run the actual training
        print(f"\n🏃 EXECUTING TRAINING SESSION...")
        print("This will:")
        print("  • Run games concurrently (SWARM mode)")  
        print("  • Execute sleep cycles with memory consolidation")
        print("  • Make intelligent game reset decisions")
        print("  • Track all memory operations")
        
        session_results = await continuous_loop.run_continuous_learning(session_id)
        
        # Show final comprehensive results
        print(f"\n🎉 TRAINING SESSION COMPLETE!")
        print("="*60)
        
        # Show final system status
        final_status = continuous_loop.get_system_status_flags()
        final_detailed = continuous_loop.get_sleep_and_memory_status()
        
        print(f"\n📊 FINAL SYSTEM STATUS:")
        print(f"Sleep Cycles: {final_detailed['sleep_status']['sleep_cycles_this_session']}")
        print(f"Memory Consolidations: {final_detailed['memory_consolidation_status']['consolidation_operations_completed']}")
        print(f"High-Salience Strengthened: {final_detailed['memory_consolidation_status']['high_salience_memories_strengthened']}")
        print(f"Low-Salience Decayed: {final_detailed['memory_consolidation_status']['low_salience_memories_decayed']}")
        print(f"Game Reset Decisions: {final_detailed['game_reset_status']['total_reset_decisions']}")
        
        if final_detailed['game_reset_status']['has_made_reset_decisions']:
            print(f"Reset Success Rate: {final_detailed['game_reset_status']['reset_success_rate']:.1%}")
            print(f"Last Reset Reason: {final_detailed['game_reset_status']['last_reset_reason']}")
        
        # Performance summary
        overall_perf = session_results.get('overall_performance', {})
        print(f"\n🏆 PERFORMANCE RESULTS:")
        print(f"Games Trained: {overall_perf.get('games_trained', 0)}")
        print(f"Total Episodes: {overall_perf.get('total_episodes', 0)}")
        print(f"Overall Win Rate: {overall_perf.get('overall_win_rate', 0):.1%}")
        print(f"Average Score: {overall_perf.get('overall_average_score', 0):.1f}")
        print(f"Learning Efficiency: {overall_perf.get('learning_efficiency', 0):.2f}")
        
        # Grid size analysis
        grid_summary = session_results.get('grid_sizes_summary', {})
        if grid_summary.get('dynamic_sizing_verified'):
            print(f"\n🎯 DYNAMIC GRID ANALYSIS:")
            print(f"Unique Grid Sizes: {grid_summary.get('unique_sizes', 0)}")
            print(f"Sizes Encountered: {', '.join(grid_summary.get('sizes_encountered', []))}")
            print("✅ Dynamic grid sizing system working correctly!")
        
        print(f"\nARC-3 Scoreboard: https://arcprize.org/leaderboard")
        print("="*60)
        
        return session_results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("This might happen if ARC-AGI-3-Agents is not properly set up")
        
        # Still show system status even if training failed
        status = continuous_loop.get_system_status_flags()
        active_systems = [k for k, v in status.items() if v]
        if active_systems:
            print(f"\n🔧 ACTIVE SYSTEMS: {', '.join(active_systems)}")
        
        return None

def main():
    """Main entry point for full training."""
    try:
        results = asyncio.run(run_full_training())
        if results:
            print(f"\n✅ Training completed successfully!")
        else:
            print(f"\n⚠️  Training completed with issues")
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
