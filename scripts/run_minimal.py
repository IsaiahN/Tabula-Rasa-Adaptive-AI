"""Simple CLI to run TB-Seed minimal mode using GameRunner."""
import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file (like train.py does)
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] Environment variables loaded from .env file")
except ImportError:
    print("[INFO] python-dotenv not installed. Using system environment variables.")

from src.training.game_runner import GameRunner
from src.vision.schema import validate_detections

# Try to import APIManager from existing codebase
try:
    from src.api import APIManager
except Exception:
    APIManager = None

try:
    from src.api.stub_api_manager import StubAPIManager
except Exception:
    StubAPIManager = None

async def get_random_game_id(api_manager):
    """Get a random game ID from available ARC 3 games."""
    try:
        if hasattr(api_manager, 'get_available_games'):
            print("[AUTO] Fetching available games from ARC 3...")
            games = await api_manager.get_available_games()
            if games and len(games) > 0:
                import random
                selected_game = random.choice(games)
                game_id = selected_game.get('id') or selected_game.get('game_id', 'unknown')
                print(f"[AUTO] Selected random game: {game_id} (from {len(games)} available)")
                return game_id
            else:
                print("[AUTO] No games available from API, using fallback")
                return "fallback_game_auto"
        else:
            print("[AUTO] API doesn't support get_available_games, using fallback")
            return "fallback_game_auto"
    except Exception as e:
        print(f"[AUTO] Error fetching games: {e}, using fallback")
        return "fallback_game_auto"

async def main(args):
    """Main function - defaults to REAL ARC 3 API unless stub explicitly requested."""
    api = None

    # Check if we should use stub API (explicit opt-in only)
    if args.use_stub_api:
        print("[STUB] Using stub API for CI/testing")
        if not StubAPIManager:
            raise RuntimeError("StubAPIManager not available")
        api = StubAPIManager()
        await api.initialize()
    else:
        # Default to REAL ARC 3 API (like train.py does)
        print("[REAL] Using real ARC 3 API")
        if not APIManager:
            raise RuntimeError("APIManager not available - check imports")

        # Get API key from environment variable
        api_key = os.getenv('ARC_API_KEY') or os.getenv('ARC_AGI_3_API_KEY')
        if not api_key:
            print("ERROR: No ARC_API_KEY found in environment variables")
            print("Options:")
            print("1. Check your .env file has: ARC_API_KEY=your_key_here")
            print("2. Set environment variable: set ARC_API_KEY=your_key_here")
            print("3. Use stub API for testing: --use-stub-api-for-ci")
            raise RuntimeError("ARC_API_KEY environment variable required for real API calls")

        print(f"[OK] Found ARC_API_KEY: {api_key[:8]}...")
        api = APIManager(api_key=api_key)
        if hasattr(api, 'initialize'):
            await api.initialize()
            print("[OK] Real ARC 3 API initialized successfully")

    # Auto-select game ID if not provided
    game_id = args.game_id
    if not game_id or game_id.lower() in ['auto', 'random']:
        game_id = await get_random_game_id(api)

    runner = GameRunner(api_manager=api)
    result = await runner.run_game(game_id, max_actions=args.max_actions)
    print(f"Result: {result}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run minimal ARC 3 games with automatic game selection')
    parser.add_argument('--game-id', default='auto', help='Game ID to play (default: auto-select from available games)')
    parser.add_argument('--max-actions', type=int, default=50, help='Maximum number of actions to take')
    parser.add_argument('--use-stub-api-for-ci', dest='use_stub_api', action='store_true', help='Opt-in stub ARC3 API for CI/testing only')
    args = parser.parse_args()
    asyncio.run(main(args))