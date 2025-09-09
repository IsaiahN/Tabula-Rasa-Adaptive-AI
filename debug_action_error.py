#!/usr/bin/env python3
"""
Debug script to identify the 'str' object has no attribute 'items' error
"""
import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from arc_integration.continuous_learning_loop import ContinuousLearningLoop

async def debug_action_error():
    """Debug the action sending error."""
    try:
        # Create dummy paths for initialization
        dummy_arc_agents_path = Path("C:/Users/Admin/Documents/GitHub/tabula-rasa/arc_agents")
        dummy_tabula_rasa_path = Path("C:/Users/Admin/Documents/GitHub/tabula-rasa")

        # Set the actual API key for the test BEFORE creating the loop
        os.environ['ARC_API_KEY'] = "9fd18a2c-8b5e-4f3a-9c2d-1e8f7a6b5c4d"
        
        # Ensure dummy paths exist for the test to run without FileNotFoundError
        dummy_arc_agents_path.mkdir(parents=True, exist_ok=True)
        dummy_tabula_rasa_path.mkdir(parents=True, exist_ok=True)

        # Initialize ContinuousLearningLoop with dummy paths
        loop = ContinuousLearningLoop(
            arc_agents_path=dummy_arc_agents_path,
            tabula_rasa_path=dummy_tabula_rasa_path
        )

        print("🔍 Testing action sending with detailed error tracking...")
        
        # Test the direct control method with error tracking
        try:
            result = await loop.start_training_with_direct_control(
                game_id="test-game", 
                max_actions=1, 
                session_count=1
            )
            print(f"✅ Result type: {type(result)}")
            print(f"✅ Result: {result}")
        except Exception as e:
            print(f"❌ Exception caught: {e}")
            print(f"❌ Exception type: {type(e)}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            
    except Exception as e:
        print(f"❌ Setup error: {e}")
        import traceback
        print(f"❌ Setup traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(debug_action_error())
