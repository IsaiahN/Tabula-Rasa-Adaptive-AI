import asyncio
import sys
import os
sys.path.append('src')
from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop

async def test_direct_control():
    try:
        # Set API key (replace with actual key)
        os.environ['ARC_API_KEY'] = os.environ.get('ARC_API_KEY', 'your-key-here')
        
        loop = ContinuousLearningLoop(
            arc_agents_path='c:/Users/Admin/Documents/GitHub/ARC-AGI-3-Agents',
            tabula_rasa_path='c:/Users/Admin/Documents/GitHub/tabula-rasa',
            api_key=os.environ.get('ARC_API_KEY')
        )
        
        # First, get available games from the API
        print('🔍 Getting available games from ARC-3 API...')
        available_games = await loop.get_available_games()
        
        if not available_games:
            print('❌ No games available from API')
            return {'error': 'No games available'}
        
        print(f'✅ Found {len(available_games)} available games:')
        for i, game in enumerate(available_games[:5]):  # Show first 5
            print(f'   {i+1}. {game["title"]} ({game["game_id"]})')
        
        # Use the first available game for testing
        game_id = available_games[0]['game_id']
        game_title = available_games[0]['title']
        
        print(f'\n🎮 Testing direct API control with: {game_title} ({game_id})')
        
        # Test the direct control with just 10 actions to verify it works
        result = await loop.start_training_with_direct_control(game_id, max_actions=10, session_count=1)
        
        print(f'\n✅ Direct Control Test Results:')
        if 'error' in result:
            print(f'❌ Error: {result["error"]}')
        else:
            print(f'   Total Actions: {result.get("total_actions", 0)}')
            print(f'   Final Score: {result.get("final_score", 0)}')
            print(f'   Final State: {result.get("final_state", "UNKNOWN")}')
            print(f'   Effective Actions: {len(result.get("effective_actions", []))}')
            
        return result
        
    except Exception as e:
        print(f'❌ Test failed: {str(e)}')
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

if __name__ == "__main__":
    # Run the test
    result = asyncio.run(test_direct_control())
