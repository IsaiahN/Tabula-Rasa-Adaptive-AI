import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when running script directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.training.analysis.action_selector import ActionSelector

logging.basicConfig(level=logging.INFO)

async def run_test():
    selector = ActionSelector(None)
    # simulate a game state with repeated action6 availability
    game_state = {'frame': [[[0]]], 'score': 0, 'game_id': 'test_game', 'state': 'RUNNING'}
    available_actions = [1,2,3,4,5,6,7]

    for i in range(10):
        action = await selector.select_action(game_state, available_actions)
        print(f"Selected action: {action}")
        # Simulate a failed attempt for action6 at (32,32) to trigger penalty
        if action.get('id') == 6 or action.get('action') == 'ACTION6':
            x = action.get('x', 32)
            y = action.get('y', 32)
            # record a failed attempt
            await selector.frame_analyzer.record_coordinate_attempt(x, y, False, 'test_game')
        await asyncio.sleep(0.1)

if __name__ == '__main__':
    asyncio.run(run_test())
