#!/usr/bin/env python3
"""
TABULA RASA - SIMPLE 9 HOUR CONTINUOUS TRAINING SCRIPT

This script runs a simple 9-hour continuous training session
with multiple sequential games for maximum stability.

Features:
- Sequential game execution (no encoding issues)
- Multiple games per hour
- Enhanced learning across games
- Graceful shutdown
- Simple and stable
- DIRECT API CONTROL (no subprocess parallel execution)

Usage:
    python run_9hour_simple_training.py
"""

import os
import sys
import time
import asyncio
import signal
from datetime import datetime
from typing import Dict, List, Any
import json

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not available, using system environment variables")

# Add src to path for database access
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from database.system_integration import get_system_integration
from database.db_initializer import ensure_database_ready
from training import ContinuousLearningLoop

# Global shutdown flag
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle graceful shutdown signals."""
    global shutdown_requested
    print(f"\n🛑 GRACEFUL SHUTDOWN REQUESTED (Signal: {signum})")
    shutdown_requested = True
    print("🛑 Training will stop after current session completes...")
    print("🛑 Press Ctrl+G again to force immediate exit")

async def run_training_session(session_id: int, duration_minutes: int = 15) -> Dict[str, Any]:
    """Run a single training session with direct API control (no subprocess)."""
    print(f"🚀 Starting DIRECT API training session #{session_id}")
    
    # Ensure API key is available
    if 'ARC_API_KEY' not in os.environ:
        print("❌ ARC_API_KEY not found in environment variables")
        return {
            'session_id': session_id,
            'return_code': -1,
            'duration': 0,
            'success': False,
            'error': 'ARC_API_KEY not found in environment variables'
        }
    else:
        print(f"✅ ARC_API_KEY found: {os.environ['ARC_API_KEY'][:10]}...")
    
    start_time = time.time()
    
    try:
        # Initialize the continuous learning loop directly
        print(f"🔧 Initializing ContinuousLearningLoop for session #{session_id}")
        
        # Get the current directory paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        arc_agents_path = os.path.join(current_dir, "ARC-AGI-3-Agents")
        tabula_rasa_path = current_dir
        
        learning_loop = ContinuousLearningLoop(
            arc_agents_path=arc_agents_path,
            tabula_rasa_path=tabula_rasa_path,
            api_key=os.environ.get('ARC_API_KEY')
        )
        
        # Ensure the system is initialized
        learning_loop._ensure_initialized()
        
        # Get available games
        available_games = await learning_loop.get_available_games()
        if not available_games:
            print("❌ No available games found")
            return {
                'session_id': session_id,
                'return_code': -1,
                'duration': time.time() - start_time,
                'success': False,
                'error': 'No available games found'
            }
        
        # Select a single game for this session
        selected_game = available_games[0]  # Always use the first available game
        game_id = selected_game['game_id']  # Extract the game_id string
        print(f"🎮 Selected game: {selected_game['title']} ({game_id})")
        
        # Run training with direct control for the specified duration
        print(f"🎯 Starting direct API training for {duration_minutes} minutes...")
        
        # Run training for the full duration in a single continuous session
        print(f"🎯 Starting continuous training session for {duration_minutes} minutes...")
        
        # Run a single continuous training session for the full duration
        result = await learning_loop.start_training_with_direct_control(
            game_id=game_id,
            max_actions_per_game=500,  # This will be the total actions for the entire session
            session_count=session_id,
            duration_minutes=duration_minutes  # Pass the duration parameter
        )
        
        duration = time.time() - start_time
        
        # Check if training was successful
        success = result is not None and 'error' not in result
        
        return {
            'session_id': session_id,
            'return_code': 0 if success else -1,
            'duration': duration,
            'success': success,
            'result': result
        }
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Error in training session #{session_id}: {e}")
        return {
            'session_id': session_id,
            'return_code': -1,
            'duration': duration,
            'success': False,
            'error': str(e)
        }

async def main():
    """Main function for simple 9-hour training with direct API control."""
    global shutdown_requested
    
    # Setup signal handlers for graceful shutdown
    if hasattr(signal, 'SIGQUIT'):  # Unix/Linux/Mac - Ctrl+G
        signal.signal(signal.SIGQUIT, signal_handler)
    if hasattr(signal, 'SIGBREAK'):  # Windows - Ctrl+G
        signal.signal(signal.SIGBREAK, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    print("=" * 80)
    print("TABULA RASA - SIMPLE 9 HOUR CONTINUOUS TRAINING")
    print("=" * 80)
    print()
    print("🚀 Starting DIRECT API sequential training session...")
    print("⏱️ Duration: 9 hours (540 minutes)")
    print("🎯 Mode: Sequential with direct API control (NO subprocess parallel execution)")
    print("✨ Features: Enhanced learning, stable execution, graceful shutdown")
    print("💾 Database: Enabled (no more JSON files)")
    print("🔒 GUARANTEED: Single-threaded execution - no parallel _train_on_game calls")
    
    # Ensure database is ready before starting training
    print("🔍 Checking database initialization...")
    if not ensure_database_ready():
        print("❌ Database initialization failed. Training cannot proceed.")
        return 1
    
    # Record start time
    start_time = datetime.now()
    print(f"🕐 Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Press Ctrl+G to stop gracefully")
    print()
    
    # Configuration - SINGLE SESSION FOCUS
    total_duration = 9 * 60 * 60  # 9 hours in seconds
    session_duration = 60  # 60 minutes per session
    sessions_per_hour = 1  # 1 session per hour (focused on single games)
    total_sessions = 9 * sessions_per_hour  # 9 total sessions
    
    print(f"📊 Training Plan:")
    print(f"   • Total sessions: {total_sessions}")
    print(f"   • Session duration: {session_duration} minutes")
    print(f"   • Sessions per hour: {sessions_per_hour}")
    print(f"   • Total games: {total_sessions}")
    print(f"   • Execution: DIRECT API (no subprocess parallel execution)")
    print()
    
    all_results = []
    session_count = 0
    
    try:
        while session_count < total_sessions and not shutdown_requested:
            # Check for shutdown request first
            if shutdown_requested:
                print(f"\n🛑 SHUTDOWN REQUESTED - Stopping training gracefully")
                print(f"Completed {session_count} sessions before shutdown")
                break
                
            # Check if 9 hours have elapsed
            current_time = datetime.now()
            elapsed_seconds = (current_time - start_time).total_seconds()
            remaining_seconds = total_duration - elapsed_seconds
            
            if remaining_seconds <= 0:
                print(f"\n🎉 9 HOUR TRAINING COMPLETE!")
                print(f"Total duration: {elapsed_seconds/3600:.2f} hours")
                print(f"Total sessions: {session_count}")
                print(f"Completed at: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                break
            
            session_count += 1
            remaining_hours = remaining_seconds / 3600
            
            print("=" * 80)
            print(f"TRAINING SESSION #{session_count}")
            print("=" * 80)
            print(f"⏰ Time remaining: {remaining_hours:.2f} hours")
            print(f"🕐 Session started: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🎮 Starting DIRECT API training session...")
            print()
            
            # Run training session with direct API control
            result = await run_training_session(session_count, session_duration)
            all_results.append(result)
            
            # Display result
            if result['success']:
                print(f"✅ Session #{session_count} completed successfully in {result['duration']:.1f}s")
            else:
                print(f"❌ Session #{session_count} failed (code: {result['return_code']}) in {result['duration']:.1f}s")
                if 'error' in result:
                    print(f"   Error: {result['error']}")
            
            # Calculate progress
            successful_sessions = sum(1 for r in all_results if r['success'])
            success_rate = successful_sessions / len(all_results) * 100
            
            print(f"📊 Progress: {len(all_results)}/{total_sessions} sessions")
            print(f"✅ Success rate: {success_rate:.1f}%")
            print()
            
            # Check for shutdown request after each session
            if shutdown_requested:
                print(f"\n🛑 SHUTDOWN REQUESTED - Stopping training gracefully")
                print(f"Completed {session_count} sessions before shutdown")
                break
            
            # Brief pause between sessions
            await asyncio.sleep(2)
            
    except KeyboardInterrupt:
        current_time = datetime.now()
        elapsed_seconds = (current_time - start_time).total_seconds()
        print(f"\n🛑 Training stopped by user (Ctrl+G)")
        print(f"⏱️ Total duration: {elapsed_seconds/3600:.2f} hours")
        print(f"📊 Total sessions completed: {session_count}")
        
    except Exception as e:
        print(f"\n❌ Error running training: {e}")
        return 1
    
    # Final statistics
    print("\n" + "=" * 80)
    print("FINAL TRAINING STATISTICS")
    print("=" * 80)
    
    if all_results:
        successful_sessions = sum(1 for r in all_results if r['success'])
        failed_sessions = len(all_results) - successful_sessions
        total_duration = sum(r['duration'] for r in all_results)
        avg_duration = total_duration / len(all_results)
        
        print(f"📊 Overall Results:")
        print(f"   🎮 Total sessions: {len(all_results)}")
        print(f"   ✅ Successful: {successful_sessions}")
        print(f"   ❌ Failed: {failed_sessions}")
        print(f"   🎯 Success rate: {successful_sessions/len(all_results)*100:.1f}%")
        print(f"   ⏱️ Total training time: {total_duration/3600:.2f} hours")
        print(f"   📈 Average session duration: {avg_duration:.1f}s")
        print(f"   🔒 Execution mode: DIRECT API (single-threaded)")
        
        # Database-only mode: No file saving
        print(f"\n💾 Results saved to database")
    
    print(f"\nTraining session ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
