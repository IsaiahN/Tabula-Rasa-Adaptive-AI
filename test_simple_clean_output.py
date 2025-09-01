"""Simple test of the direct control system with clean output."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import aiohttp
import json
import time

# Test API key
API_KEY = os.getenv('ARC3_API_KEY', 'YOUR_ARC3_API_KEY')
ARC3_BASE_URL = "https://three.arcprize.org"

class SimpleAvailableActionsTracker:
    def __init__(self):
        self._last_available_actions = {}
    
    def track_actions(self, game_id, current_actions):
        """Track and display available_actions only if they changed."""
        last_actions = self._last_available_actions.get(game_id, [])
        if current_actions != last_actions:
            print(f"🎮 Available Actions Changed for {game_id}: {current_actions}")
            self._last_available_actions[game_id] = current_actions
            return True
        return False

async def test_clean_api_output():
    """Test clean output showing only available_actions changes."""
    print("🧪 Testing Clean API Output - Available Actions Only")
    print("=" * 60)
    
    tracker = SimpleAvailableActionsTracker()
    
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "X-API-Key": API_KEY,
                "Content-Type": "application/json"
            }
            
            # 1. Open scorecard
            print("🔄 Opening scorecard...")
            async with session.post(f"{ARC3_BASE_URL}/api/cmd/OPEN", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    scorecard_id = data.get('card_id')
                    print(f"✅ Scorecard opened: {scorecard_id}")
                else:
                    print("❌ Failed to open scorecard")
                    return
            
            # 2. Get games and select one
            async with session.get(f"{ARC3_BASE_URL}/api/games", headers=headers) as response:
                if response.status == 200:
                    games = await response.json()
                    test_game = games[0]['game_id']  # Use first game
                    print(f"🎯 Testing with game: {test_game}")
                else:
                    print("❌ Failed to get games")
                    return
            
            # 3. Start game session (RESET)
            print(f"🔄 NEW GAME RESET for {test_game}")
            reset_payload = {"game_id": test_game, "card_id": scorecard_id}
            async with session.post(f"{ARC3_BASE_URL}/api/cmd/RESET", headers=headers, json=reset_payload) as response:
                if response.status == 200:
                    data = await response.json()
                    guid = data.get('guid')
                    available_actions = data.get('actions', [])
                    print(f"✅ NEW GAME successful: {test_game}")
                    
                    # Track initial available actions
                    tracker.track_actions(test_game, available_actions)
                else:
                    print("❌ RESET failed")
                    return
            
            # 4. Execute a few actions and track available_actions changes
            print(f"🎯 Executing actions - watching for available_actions changes...")
            for i in range(5):  # Try 5 actions
                # Try different actions to see changes
                action_num = (i % 7) + 1  # Cycle through actions 1-7
                
                url = f"{ARC3_BASE_URL}/api/cmd/ACTION{action_num}"
                payload = {"game_id": test_game, "guid": guid}
                
                if action_num == 6:
                    # ACTION6 needs coordinates
                    payload["x"] = 10 + i
                    payload["y"] = 10 + i
                else:
                    # Other actions use reasoning
                    payload["reasoning"] = {"action": f"test_action_{action_num}"}
                
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        current_actions = data.get('actions', [])
                        
                        # Only show if actions changed
                        changed = tracker.track_actions(test_game, current_actions)
                        if not changed:
                            print(f"   ACTION {action_num}: No change in available actions")
                        
                    else:
                        print(f"❌ ACTION {action_num} FAILED: {response.status}")
                        break
                
                # Small delay between actions
                await asyncio.sleep(0.5)
            
            # 5. Close scorecard
            print("🔄 Closing scorecard...")
            async with session.post(f"{ARC3_BASE_URL}/api/cmd/CLOSE", headers=headers, json={"card_id": scorecard_id}) as response:
                if response.status == 200:
                    print("✅ Scorecard closed")
                else:
                    print("❌ Failed to close scorecard")
    
    except Exception as e:
        print(f"❌ Test error: {e}")
    
    print("=" * 60)
    print("✅ Clean output test completed")

if __name__ == "__main__":
    asyncio.run(test_clean_api_output())
