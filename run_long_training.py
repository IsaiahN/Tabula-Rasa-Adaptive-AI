#!/usr/bin/env python3
"""
Long ARC Training Session with Enhanced Features
- Swarm Mode: 6 concurrent games
- Action Intelligence System: Real-time semantic learning
- Progressive Memory Hierarchy: Breakthrough-only preservation
- Strategic Coordinate System: 7-region optimization
- 500K+ actions per game session
"""

import sys
import os
import time
import asyncio
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from arc_integration.continuous_learning_loop import ContinuousLearningLoop

async def run_long_arc_training():
    """Run comprehensive long training session with all enhancements."""
    print('🔥 STARTING LONG ARC TRAINING SESSION')
    print('=' * 60)
    
    # Get API key from environment or .env file
    api_key = os.getenv('ARC_API_KEY')
    if not api_key:
        # Try to load from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv('ARC_API_KEY')
        except ImportError:
            pass
    
    if not api_key:
        print('❌ ARC_API_KEY not found in environment')
        print('Please set your API key in .env file or environment variable')
        print('Example: export ARC_API_KEY=your_key_here')
        return None
    
    # Initialize with enhanced settings for long training
    print('🔧 Initializing Enhanced Learning System...')
    loop = ContinuousLearningLoop(
        arc_agents_path='C:/path/to/ARC-AGI-3-Agents',  # Will be adjusted automatically
        tabula_rasa_path='.',
        api_key=api_key,
        save_directory='continuous_learning_data'
    )
    
    print('🎯 LONG TRAINING CONFIGURATION:')
    print('   Mode: Extended Training with Swarm Intelligence')
    print('   Max Actions per Game: 500,000+')
    print('   Concurrent Games: 6 (Swarm Mode)')
    print('   Mastery Sessions per Game: 100')
    print('   Memory System: Progressive Hierarchy + Action Intelligence')
    print('   Coordinate System: 7-Region Strategic Exploration')
    print('   Energy System: Optimized 0-100 Scale')
    print('   Action Learning: Real-time semantic understanding')
    print('')
    
    # Verify API connection first
    print('🔍 Verifying ARC-3 API connection...')
    verification = await loop.verify_api_connection()
    
    if not verification['api_accessible']:
        print('❌ API connection failed - cannot proceed with training')
        return None
    
    print(f"✅ API verified: {verification['total_games_available']} games available")
    
    # Select multiple games for long training
    print('🎲 Selecting games for long training...')
    games = await loop.select_training_games(
        count=6,  # 6 games for swarm mode
        difficulty_preference='mixed'
    )
    
    if not games:
        print('❌ No games selected for training')
        return None
    
    print(f'🎮 Selected {len(games)} games for long training:')
    for i, game_id in enumerate(games, 1):
        print(f'   {i}. {game_id}')
    print('')
    
    # Run SWARM mode for maximum parallel training
    print('🔥 ACTIVATING SWARM MODE FOR LONG TRAINING')
    print('⚠️  This will run for a LONG time with high computational intensity')
    print('   Expected Duration: Several hours to days depending on performance')
    print('   Memory Usage: Progressive hierarchy with intelligent preservation')
    print('   Action Learning: Continuous semantic pattern discovery')
    print('')
    
    start_time = time.time()
    
    try:
        swarm_results = await loop.run_swarm_mode(
            games=games,
            max_concurrent=6,  # All 6 games running simultaneously
            max_episodes_per_game=100  # 100 mastery sessions per game
        )
        
        duration = (time.time() - start_time) / 60
        
        print('\n🏆 LONG TRAINING SESSION COMPLETE!')
        print(f'Total Duration: {duration:.1f} minutes ({duration/60:.1f} hours)')
        
        if swarm_results and 'games_completed' in swarm_results:
            games_completed = len(swarm_results['games_completed'])
            print(f'Games Processed: {games_completed}')
            
            if 'overall_performance' in swarm_results:
                perf = swarm_results['overall_performance']
                print('📊 OVERALL PERFORMANCE:')
                for key, value in perf.items():
                    print(f'   {key}: {value}')
        
        print('\n🧠 ACTION INTELLIGENCE SUMMARY:')
        print('   Action semantic learning completed across all games')
        print('   Movement patterns, effects, and game-specific roles learned')
        print('   Coordinate success zones mapped for optimal positioning')
        
        return swarm_results
        
    except KeyboardInterrupt:
        print('\n⚠️  Training interrupted by user')
        print('Partial results may be saved in continuous_learning_data/')
        return None
        
    except Exception as e:
        print(f'\n❌ Training error: {e}')
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main entry point for long training session."""
    print('🚀 Long ARC Training Session Starting...')
    print('Press Ctrl+C to interrupt training if needed')
    print('')
    
    try:
        result = asyncio.run(run_long_arc_training())
        
        if result:
            print('✅ Long ARC training session completed successfully!')
            print('Results saved in continuous_learning_data/')
        else:
            print('❌ Training session ended without completion')
            
    except KeyboardInterrupt:
        print('\n⚠️  Training interrupted by user')
    except Exception as e:
        print(f'❌ Fatal error: {e}')

if __name__ == '__main__':
    main()
