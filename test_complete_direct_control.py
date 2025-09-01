import asyncio
import sys
import os
sys.path.append('src')
from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop

async def test_complete_direct_control():
    """Test the complete direct control system with real ARC-3 API integration."""
    try:
        # Set API key from environment
        api_key = os.environ.get('ARC_API_KEY')
        if not api_key:
            print('❌ ARC_API_KEY environment variable not set')
            return {'error': 'No API key'}
        
        print('🎮 TESTING COMPLETE DIRECT CONTROL SYSTEM')
        print('='*60)
        
        loop = ContinuousLearningLoop(
            arc_agents_path='c:/Users/Admin/Documents/GitHub/ARC-AGI-3-Agents',
            tabula_rasa_path='c:/Users/Admin/Documents/GitHub/tabula-rasa',
            api_key=api_key
        )
        
        # Step 1: Get real available games
        print('\n🔍 Step 1: Getting available games from ARC-3 API...')
        available_games = await loop.get_available_games()
        
        if not available_games:
            print('❌ No games available from API')
            return {'error': 'No games available'}
        
        print(f'✅ Found {len(available_games)} available games')
        for i, game in enumerate(available_games[:3]):  # Show first 3
            print(f'   {i+1}. {game["title"]} ({game["game_id"]})')
        
        # Step 2: Test scorecard operations
        print('\n🏁 Step 2: Testing scorecard operations...')
        scorecard_id = await loop._open_scorecard()
        if scorecard_id:
            print(f'✅ Scorecard opened: {scorecard_id}')
        else:
            print('❌ Failed to open scorecard')
            return {'error': 'Scorecard creation failed'}
        
        # Step 3: Test game session with first available game
        game_id = available_games[0]['game_id']
        game_title = available_games[0]['title']
        print(f'\n🎯 Step 3: Starting game session with: {game_title} ({game_id})')
        
        session_data = await loop._start_game_session(game_id)
        if not session_data:
            print('❌ Failed to start game session')
            return {'error': 'Game session failed'}
        
        guid = session_data['guid']
        print(f'✅ Game session started with GUID: {guid}')
        print(f'   Initial State: {session_data.get("state", "UNKNOWN")}')
        print(f'   Available Actions: {session_data.get("available_actions", [])}')
        
        # Step 4: Test direct control with intelligent action selection
        print(f'\n🎮 Step 4: Testing direct control with enhanced action selection...')
        print(f'   Running up to 20 actions to demonstrate system')
        
        result = await loop.start_training_with_direct_control(
            game_id, max_actions=20, session_count=1
        )
        
        print(f'\n✅ DIRECT CONTROL TEST COMPLETE:')
        if 'error' in result:
            print(f'❌ Error: {result["error"]}')
        else:
            print(f'   Game: {game_title} ({game_id})')
            print(f'   Total Actions: {result.get("total_actions", 0)}')
            print(f'   Final Score: {result.get("final_score", 0)}')
            print(f'   Final State: {result.get("final_state", "UNKNOWN")}')
            print(f'   Effective Actions: {len(result.get("effective_actions", []))}')
            print(f'   Action Effectiveness: {len(result.get("effective_actions", [])) / max(1, result.get("total_actions", 1)) * 100:.1f}%')
        
        # Step 5: Test level reset capability
        print(f'\n🔄 Step 5: Testing level reset capability...')
        reset_result = await loop._reset_level(game_id)
        if reset_result:
            print(f'✅ Level reset successful')
            print(f'   New State: {reset_result.get("state", "UNKNOWN")}')
        else:
            print('❌ Level reset failed')
        
        # Step 6: Close scorecard
        print(f'\n🏁 Step 6: Closing scorecard...')
        close_success = await loop._close_scorecard(scorecard_id)
        if close_success:
            print(f'✅ Scorecard closed successfully')
        else:
            print('❌ Failed to close scorecard')
        
        # Summary
        print(f'\n🎯 COMPLETE DIRECT CONTROL SYSTEM TEST RESULTS:')
        print(f'   ✅ API Connection: SUCCESS')
        print(f'   ✅ Games Available: {len(available_games)}')
        print(f'   ✅ Scorecard Operations: {"SUCCESS" if scorecard_id and close_success else "PARTIAL"}')
        print(f'   ✅ Game Session: {"SUCCESS" if session_data else "FAILED"}')
        print(f'   ✅ Direct Control: {"SUCCESS" if "error" not in result else "FAILED"}')
        print(f'   ✅ Level Reset: {"SUCCESS" if reset_result else "FAILED"}')
        
        return result
        
    except Exception as e:
        print(f'❌ Test failed: {str(e)}')
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

if __name__ == "__main__":
    print('🚀 Starting Complete Direct Control System Test...')
    result = asyncio.run(test_complete_direct_control())
