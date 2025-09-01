"""Test the ultra-clean output - only available_actions when they change."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import aiohttp
import json

# Test API key
API_KEY = os.getenv('ARC3_API_KEY', 'YOUR_ARC3_API_KEY')
ARC3_BASE_URL = "https://three.arcprize.org"

class UltraCleanTracker:
    def __init__(self):
        self._last_available_actions = {}
    
    def track_actions(self, game_id, current_actions):
        """Track and display available_actions only if they changed."""
        last_actions = self._last_available_actions.get(game_id, [])
        if current_actions != last_actions:
            print(f"🎮 Available Actions for {game_id}: {current_actions}")
            self._last_available_actions[game_id] = current_actions
            return True
        return False

async def test_ultra_clean_output():
    """Test ultra-clean output - ONLY available_actions changes."""
    print("🧪 ULTRA CLEAN TEST - Only Available Actions Changes")
    print("=" * 60)
    
    tracker = UltraCleanTracker()
    
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "X-API-Key": API_KEY,
                "Content-Type": "application/json"
            }
            
            # 1. Get a test game
            async with session.get(f"{ARC3_BASE_URL}/api/games", headers=headers) as response:
                if response.status != 200:
                    print("❌ Cannot connect to API")
                    return
                games = await response.json()
                test_game = games[0]['game_id']
            
            # 2. Open scorecard silently
            async with session.post(f"{ARC3_BASE_URL}/api/cmd/OPEN", headers=headers) as response:
                if response.status != 200:
                    print("❌ Failed to open scorecard")
                    return
                data = await response.json()
                scorecard_id = data.get('card_id')
            
            # 3. Start game session (RESET)
            reset_payload = {"game_id": test_game, "card_id": scorecard_id}
            async with session.post(f"{ARC3_BASE_URL}/api/cmd/RESET", headers=headers, json=reset_payload) as response:
                if response.status != 200:
                    print("❌ RESET failed")
                    return
                data = await response.json()
                guid = data.get('guid')
                available_actions = data.get('actions', [])
                
                # Track initial available actions
                tracker.track_actions(test_game, available_actions)
            
            print(f"🚀 Testing with game: {test_game}")
            print("👀 Only showing when available_actions change...")
            print("=" * 60)
            
            # 4. Execute actions and show ONLY when available_actions change
            for i in range(10):  # Try 10 actions
                action_num = (i % 6) + 1  # Cycle through actions 1-6
                
                url = f"{ARC3_BASE_URL}/api/cmd/ACTION{action_num}"
                payload = {"game_id": test_game, "guid": guid}
                
                if action_num == 6:
                    payload["x"] = 10 + i
                    payload["y"] = 10 + i
                else:
                    payload["reasoning"] = {"action": f"test_action_{action_num}"}
                
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        current_actions = data.get('actions', [])
                        
                        # ONLY show if actions changed
                        changed = tracker.track_actions(test_game, current_actions)
                        
                        # If no change, completely silent
                        
                    else:
                        print(f"❌ ACTION {action_num} failed")
                        break
                
                await asyncio.sleep(0.3)
            
            # 5. Close scorecard silently
            await session.post(f"{ARC3_BASE_URL}/api/cmd/CLOSE", headers=headers, json={"card_id": scorecard_id})
    
    except Exception as e:
        print(f"❌ Test error: {e}")
    
    print("=" * 60)
    print("✅ Ultra-clean test completed - only saw changes!")

if __name__ == "__main__":
    print("Expected output: Only 'Available Actions for game_xxx: [...]' when they actually change")
    print("No verbose action scores, no selection details, no status updates")
    print()
    asyncio.run(test_ultra_clean_output())
