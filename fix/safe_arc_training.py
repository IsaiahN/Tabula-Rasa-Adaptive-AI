#!/usr/bin/env python3
"""
Safe ARC training script with enhanced error handling and NoneType prevention.
This version includes comprehensive null safety and better error recovery.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the Python path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir / "src"))

from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop
from src.core.salience_system import SalienceMode

async def safe_arc_training():
    """Run ARC training with enhanced safety measures."""
    print("🛡️ STARTING SAFE ARC TRAINING")
    print("=" * 50)
    
    try:
        # Initialize with safety checks
        print("🔧 Initializing continuous learning system...")
        
        continuous_loop = ContinuousLearningLoop(
            arc_agents_path=Path.home() / "Documents" / "GitHub" / "ARC-AGI-3-Agents",
            tabula_rasa_path=script_dir,
            save_directory="continuous_learning_data"
        )
        
        print("✅ System initialized")
        
        # Validate API connection first
        if hasattr(continuous_loop, '_validate_api_connection'):
            print("🌐 Validating API connection...")
            api_valid = await continuous_loop._validate_api_connection()
            if not api_valid:
                print("❌ API validation failed. Please run diagnose_api_connection.py")
                return False
            print("✅ API connection validated")
        
        # Start training session with safety
        print("🚀 Starting training session...")
        
        games = [
            "vc33-58ec4396715d",
            "ft09-f340c8e5138e", 
            "as66-821a4dcad9c2"  # Start with fewer games for testing
        ]
        
        session_id = continuous_loop.start_training_session(
            games=games,
            max_mastery_sessions_per_game=10,  # Reduced for safety (corrected parameter name)
            target_win_rate=0.1,
            target_avg_score=10.0,
            salience_mode=SalienceMode.LOSSLESS,
            enable_salience_comparison=False,
            swarm_enabled=False  # Disable swarm mode for safer testing
        )
        
        print(f"📋 Session started: {session_id}")
        
        # Run the training with enhanced monitoring
        print("🎯 Running continuous learning...")
        session_results = await continuous_loop.run_continuous_learning(session_id)
        
        print("\n🏆 TRAINING COMPLETED")
        print("=" * 50)
        print(f"Session ID: {session_results.get('session_id', 'Unknown')}")
        print(f"Games Played: {len(session_results.get('games_played', {}))}")
        
        # Display results safely
        overall_perf = session_results.get('overall_performance', {})
        win_rate = overall_perf.get('overall_win_rate', 0.0)
        avg_score = overall_perf.get('overall_average_score', 0.0)
        
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Average Score: {avg_score:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("💡 This error has been caught safely")
        return False

def main():
    """Main entry point."""
    try:
        result = asyncio.run(safe_arc_training())
        if result:
            print("\n🎉 Safe training completed successfully!")
        else:
            print("\n⚠️ Training completed with issues")
    except KeyboardInterrupt:
        print("\n⏹️ Training stopped by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")

if __name__ == "__main__":
    main()
