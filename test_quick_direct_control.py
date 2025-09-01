import asyncio
import os
import sys
sys.path.append('src')
from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop

async def quick_test():
    try:
        api_key = os.environ.get('ARC_API_KEY')
        if not api_key:
            print('❌ No API key found')
            return
        
        print('🎮 Quick Direct Control Test')
        print('=' * 40)
        
        loop = ContinuousLearningLoop(
            arc_agents_path='c:/Users/Admin/Documents/GitHub/ARC-AGI-3-Agents',
            tabula_rasa_path='c:/Users/Admin/Documents/GitHub/tabula-rasa',
            api_key=api_key
        )
        
        # Get first available game
        games = await loop.get_available_games()
        if games:
            game_id = games[0]['game_id']
            game_title = games[0]['title']
            print(f'🎮 Testing with: {game_title} ({game_id})')
            
            # Test direct control with just 5 actions to validate the fix
            result = await loop.start_training_with_direct_control(game_id, max_actions=5, session_count=1)
            
            print(f'\n🎯 QUICK TEST RESULTS:')
            if 'error' in result:
                print(f'❌ Error: {result["error"]}')
            else:
                print(f'✅ SUCCESS!')
                print(f'   Actions Taken: {result.get("total_actions", 0)}')
                print(f'   Final Score: {result.get("final_score", 0)}')
                print(f'   Final State: {result.get("final_state", "UNKNOWN")}')
                print(f'   Effective Actions: {len(result.get("effective_actions", []))}')
        else:
            print('❌ No games available')
            
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(quick_test())
