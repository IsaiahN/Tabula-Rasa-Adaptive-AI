"""Simple CLI to run TB-Seed minimal mode using GameRunner."""
import asyncio
import argparse

from src.training.game_runner import GameRunner

# Try to import APIManager from existing codebase
try:
    from src.api import APIManager
except Exception:
    APIManager = None

try:
    from src.api.stub_api_manager import StubAPIManager
except Exception:
    StubAPIManager = None

async def main(args):
    api = None
    if args.use_stub_api:
        if not StubAPIManager:
            raise RuntimeError("StubAPIManager not available")
        api = StubAPIManager()
        await api.initialize()
    else:
        if APIManager:
            api = APIManager(api_key=None)
            if hasattr(api, 'initialize'):
                await api.initialize()

    runner = GameRunner(api_manager=api)
    result = await runner.run_game(args.game_id, max_actions=args.max_actions)
    print(f"Result: {result}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game-id', default='test_game')
    parser.add_argument('--max-actions', type=int, default=50)
    parser.add_argument('--use-stub-api-for-ci', dest='use_stub_api', action='store_true', help='Opt-in stub ARC3 API for CI/testing only')
    args = parser.parse_args()
    asyncio.run(main(args))
