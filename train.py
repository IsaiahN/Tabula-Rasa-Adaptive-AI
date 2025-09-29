#!/usr/bin/env python3
"""
9-Hour Continuous Training Script

Runs continuous learning for a full 9-hour period, playing sequential games
until the time limit is reached instead of stopping after a fixed number of games.
"""

# Disable Python bytecode caching
import sys
import os
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import asyncio
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.core.continuous_learning_loop import ContinuousLearningLoop

async def main():
    """Run 9-hour continuous training session."""

    print("=" * 80)
    print("STARTING 9-HOUR CONTINUOUS TRAINING SESSION")
    print("=" * 80)
    print()
    print("This will run continuous learning for 9 hours, playing sequential games")
    print("until the time limit is reached. The system will:")
    print("- Learn from wins and failures")
    print("- Detect and break losing streaks")
    print("- Apply escalating interventions when stuck")
    print("- Track coordinate-specific Action 6 patterns")
    print("- Provide progress updates every 10 games")
    print()

    # Initialize the continuous learning loop
    try:
        # Get API key from environment
        api_key = os.getenv('ARC_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("ERROR: ARC_API_KEY environment variable not set")
            print("Please set your API key: export ARC_API_KEY='your-key-here'")
            return

        # Use temporary directory for session data - database stores persistent data
        import tempfile
        temp_session_dir = tempfile.mkdtemp(prefix="training_session_")

        learning_loop = ContinuousLearningLoop(
            api_key=api_key,
            save_directory=Path(temp_session_dir)
        )

        # Run for 9 hours (no game limit)
        print("Starting 9-hour training session...")
        print("Press Ctrl+C to stop early if needed")
        print()

        results = await learning_loop.run_continuous_learning(
            max_games=None,  # No game limit
            max_hours=9.0    # 9-hour time limit
        )

        print()
        print("=" * 80)
        print("FINAL TRAINING STATISTICS")
        print("=" * 80)
        print(f"Total games played: {results.get('games_completed', 0)}")
        print(f"Games won: {results.get('games_won', 0)}")
        print(f"Success rate: {results.get('success_rate', 0.0):.2%}")
        print(f"Total time: {results.get('total_time_hours', 0.0):.2f} hours")
        print(f"Games per hour: {results.get('games_per_hour', 0.0):.1f}")
        print(f"Total actions: {results.get('total_actions', 0)}")
        print(f"Total score: {results.get('total_score', 0.0):.2f}")
        print()

        if results.get('learning_insights'):
            print("Learning Insights:")
            for insight in results['learning_insights'][:5]:  # Show top 5
                print(f"  - {insight}")
            print()

        print("Training session completed successfully!")

    except KeyboardInterrupt:
        print("\nTraining stopped by user (Ctrl+C)")
        print("Partial results may be available in the database.")

    except Exception as e:
        print(f"\nERROR: Training failed with exception: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print(f"\nSession ended. Session data saved to: {temp_session_dir}")
        print("Persistent training data is stored in the database.")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())