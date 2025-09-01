"""Test the cleaned up output showing only available_actions changes."""

import os
import asyncio
from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop
from src.core.data_models import SalienceMode

async def test_clean_output():
    """Test the continuous learning loop with clean output."""
    
    # Initialize the continuous learning loop
    continuous_loop = ContinuousLearningLoop()
    
    # Validate API connection first
    if not await continuous_loop._validate_api_connection():
        print("❌ Cannot connect to API. Check your API key and internet connection.")
        return
    
    print("🧪 Testing Clean Output - Available Actions Tracking Only")
    print("=" * 60)
    
    # Select a couple games for testing
    games = await continuous_loop.select_training_games(count=2)
    print(f"📋 Selected {len(games)} games for testing")
    
    # Start training session with minimal actions for testing
    session_id = continuous_loop.start_training_session(
        games=games,
        max_mastery_sessions_per_game=2,  # Minimal for testing
        max_actions_per_session=20,      # Just a few actions to see the output
        enable_contrarian_mode=False,    # Keep it simple
        target_win_rate=0.8,
        salience_mode=SalienceMode.DECAY_COMPRESSION,
        swarm_enabled=False             # Single game for cleaner testing
    )
    
    print(f"🚀 Starting clean output test session: {session_id}")
    print("👀 Watch for 'Available Actions Changed' messages only...")
    print("=" * 60)
    
    # Run the training
    result = await continuous_loop.run_continuous_learning(session_id)
    
    print("=" * 60)
    print(f"✅ Test completed. Result: {result.get('status', 'unknown')}")

if __name__ == "__main__":
    asyncio.run(test_clean_output())
