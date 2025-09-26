#!/usr/bin/env python3
"""
DEBUG TRAINING SCRIPT - Simplified approach to diagnose scoring issues

This script bypasses complex subsystems to test basic functionality:
- Simple action selection (no complex scoring)
- Direct API calls without heavy processing
- Clear logging of what's happening
- Minimal energy/sleep systems
"""

import os
import sys
import asyncio
import aiohttp
import time
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Basic constants
ARC3_BASE_URL = "https://three.arcprize.org"
ARC3_RATE_LIMIT = {'request_timeout': 30}

class SimpleTrainingDebugger:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session_data = {}

    async def debug_single_game(self, game_id: str = "testing") -> Dict[str, Any]:
        """Debug a single game with minimal complexity."""
        print(f"\n[DEBUG] DEBUGGING GAME: {game_id}")
        print("=" * 60)

        try:
            # Step 1: Get available games
            games = await self._get_available_games()
            if not games:
                return {"error": "No games available"}

            actual_game_id = games[0] if isinstance(games[0], str) else games[0].get('game_id', 'testing')
            print(f"[OK] Using game: {actual_game_id}")

            # Step 2: Start session
            session = await self._start_session(actual_game_id)
            if not session:
                return {"error": "Failed to start session"}

            guid = session.get('guid')
            initial_score = session.get('score', 0)
            initial_state = session.get('state', 'NOT_STARTED')

            print(f"[OK] Session started: GUID={guid}, Score={initial_score}, State={initial_state}")

            # Step 3: Try simple actions
            actions_tried = []
            max_actions = 20  # Keep it simple

            for action_num in range(1, max_actions + 1):
                print(f"\n--- Action {action_num} ---")

                # Simple action selection: cycle through 1-7
                selected_action = ((action_num - 1) % 7) + 1

                print(f"Trying ACTION{selected_action}")

                # Execute action
                result = await self._execute_simple_action(actual_game_id, guid, selected_action)

                if result:
                    new_score = result.get('score', initial_score)
                    new_state = result.get('state', initial_state)
                    score_change = new_score - initial_score

                    print(f"[OK] Result: Score {initial_score} -> {new_score} ({score_change:+}), State: {new_state}")

                    actions_tried.append({
                        'action': selected_action,
                        'score_before': initial_score,
                        'score_after': new_score,
                        'score_change': score_change,
                        'state': new_state,
                        'success': score_change > 0 or new_state == 'WIN'
                    })

                    # Update for next iteration
                    initial_score = new_score
                    initial_state = new_state

                    # Stop if we win or reach high score
                    if new_state == 'WIN' or new_score >= 100:
                        print(f"[SUCCESS] Final state: {new_state}, Final score: {new_score}")
                        break

                else:
                    print(f"[FAIL] ACTION{selected_action} failed")
                    actions_tried.append({
                        'action': selected_action,
                        'score_before': initial_score,
                        'score_after': initial_score,
                        'score_change': 0,
                        'state': initial_state,
                        'success': False,
                        'failed': True
                    })

                # Small delay between actions
                await asyncio.sleep(0.2)

            # Summary
            successful_actions = [a for a in actions_tried if a.get('success', False)]
            total_score_gained = sum(a['score_change'] for a in actions_tried)

            print(f"\n[SUMMARY]:")
            print(f"   Actions tried: {len(actions_tried)}")
            print(f"   Successful actions: {len(successful_actions)}")
            print(f"   Total score gained: {total_score_gained}")
            print(f"   Final score: {initial_score}")
            print(f"   Final state: {initial_state}")

            return {
                'final_score': initial_score,
                'final_state': initial_state,
                'actions_tried': actions_tried,
                'successful_actions': successful_actions,
                'total_score_gained': total_score_gained,
                'success': len(successful_actions) > 0 or total_score_gained > 0
            }

        except Exception as e:
            print(f"[ERROR] DEBUG ERROR: {e}")
            return {"error": str(e)}

    async def _get_available_games(self) -> List[Any]:
        """Get available games from API."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"X-API-Key": self.api_key}
                url = f"{ARC3_BASE_URL}/api/games"

                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        games = await response.json()
                        print(f"[INFO] Found {len(games)} available games")
                        return games
                    else:
                        error_text = await response.text()
                        print(f"[ERROR] Failed to get games: {response.status} - {error_text}")
                        return []
        except Exception as e:
            print(f"[ERROR] Error getting games: {e}")
            return []

    async def _start_session(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Start a game session."""
        try:
            # First open a scorecard
            scorecard_id = await self._open_scorecard()
            if not scorecard_id:
                print(f"[ERROR] Failed to open scorecard")
                return None

            async with aiohttp.ClientSession() as session:
                headers = {
                    "X-API-Key": self.api_key,
                    "Content-Type": "application/json"
                }
                url = f"{ARC3_BASE_URL}/api/cmd/RESET"
                payload = {
                    "game_id": game_id,
                    "card_id": scorecard_id
                }

                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"[INFO] Session started for {game_id}")
                        return data
                    else:
                        error_text = await response.text()
                        print(f"[ERROR] Failed to start session: {response.status} - {error_text}")
                        return None
        except Exception as e:
            print(f"[ERROR] Error starting session: {e}")
            return None

    async def _open_scorecard(self) -> Optional[str]:
        """Open a scorecard for tracking results."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "X-API-Key": self.api_key,
                    "Content-Type": "application/json"
                }
                url = f"{ARC3_BASE_URL}/api/scorecard/open"

                async with session.post(url, headers=headers, json={}) as response:
                    if response.status == 200:
                        data = await response.json()
                        scorecard_id = data.get('card_id')
                        print(f"[INFO] Opened scorecard: {scorecard_id}")
                        return scorecard_id
                    else:
                        error_text = await response.text()
                        print(f"[ERROR] Failed to open scorecard: {response.status} - {error_text}")
                        return None
        except Exception as e:
            print(f"[ERROR] Error opening scorecard: {e}")
            return None

    async def _execute_simple_action(self, game_id: str, guid: str, action_number: int) -> Optional[Dict[str, Any]]:
        """Execute a simple action without complex processing."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "X-API-Key": self.api_key,
                    "Content-Type": "application/json"
                }
                url = f"{ARC3_BASE_URL}/api/cmd/ACTION{action_number}"

                payload = {
                    "game_id": game_id,
                    "guid": guid
                }

                # For ACTION6, add simple coordinates
                if action_number == 6:
                    payload["x"] = 5  # Simple fixed coordinate
                    payload["y"] = 5

                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        error_text = await response.text()
                        print(f"[ERROR] ACTION{action_number} failed: {response.status} - {error_text}")
                        return None
        except Exception as e:
            print(f"[ERROR] Error executing ACTION{action_number}: {e}")
            return None

async def main():
    """Run debug training."""
    print("DEBUG TABULA RASA - DEBUG TRAINING")
    print("=" * 60)
    print("Purpose: Test basic functionality without complex subsystems")
    print()

    # Check for API key
    api_key = os.environ.get('ARC_API_KEY')
    if not api_key:
        print("[ERROR] ARC_API_KEY not found in environment variables")
        return 1

    print(f"[OK] API Key found: {api_key[:10]}...")

    # Create debugger
    debugger = SimpleTrainingDebugger(api_key)

    # Run debug session
    result = await debugger.debug_single_game()

    print(f"\n[FINAL RESULT]:")
    if result.get('success'):
        print(f"[SUCCESS] System can score points!")
        print(f"   Final score: {result.get('final_score', 0)}")
        print(f"   Successful actions: {len(result.get('successful_actions', []))}")
    else:
        print(f"[FAILURE] System cannot score points")
        if 'error' in result:
            print(f"   Error: {result['error']}")

    return 0 if result.get('success') else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))