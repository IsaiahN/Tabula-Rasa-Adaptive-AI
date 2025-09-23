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
    print("[OK] Environment variables loaded from .env file")
except ImportError:
    print("[WARNING] python-dotenv not available, using system environment variables")

# Add src to path for database access
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from database.system_integration import get_system_integration
from database.db_initializer import ensure_database_ready
from training import ContinuousLearningLoop

# Global shutdown flag and cleanup handler
shutdown_requested = False
cleanup_handler = None

def signal_handler(signum, frame):
    """Handle graceful shutdown signals."""
    global shutdown_requested, cleanup_handler
    if shutdown_requested:
    print(f"\n[STOP] FORCE EXIT REQUESTED (Signal: {signum})")
    print("[STOP] Exiting immediately...")
        # Try to cleanup before force exit
        if cleanup_handler:
            try:
                import asyncio
                asyncio.run(cleanup_handler())
            except:
                pass
        sys.exit(0)
    
    print(f"\n[STOP] GRACEFUL SHUTDOWN REQUESTED (Signal: {signum})")
    shutdown_requested = True
    print("[STOP] Training will stop after current session completes...")
    print("[STOP] Press Ctrl+C again to force immediate exit")

async def run_training_session(session_id: int, duration_minutes: int = 15) -> Dict[str, Any]:
    """Run a single training session with direct API control (no subprocess)."""
    print(f"[START] Starting DIRECT API training session #{session_id}")
    
    # Ensure API key is available
    if 'ARC_API_KEY' not in os.environ:
        print("[ERROR] ARC_API_KEY not found in environment variables")
        return {
            'session_id': session_id,
            'return_code': -1,
            'duration': 0,
            'success': False,
            'error': 'ARC_API_KEY not found in environment variables'
        }
    else:
        print(f"[OK] ARC_API_KEY found: {os.environ['ARC_API_KEY'][:10]}...")
    
    start_time = time.time()
    
    try:
        # Initialize the continuous learning loop directly
        print(f"[INIT] Initializing ContinuousLearningLoop for session #{session_id}")
        
        # Get the current directory paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        arc_agents_path = os.path.join(current_dir, "ARC-AGI-3-Agents")
        tabula_rasa_path = current_dir
        
        learning_loop = ContinuousLearningLoop(
            arc_agents_path=arc_agents_path,
            tabula_rasa_path=tabula_rasa_path,
            api_key=os.environ.get('ARC_API_KEY')
        )
        
        # Set up cleanup handler for graceful shutdown
        async def cleanup():
            """Cleanup function for graceful shutdown."""
            try:
                print("[CLEANUP] Cleaning up resources...")
                await learning_loop.close()
                print("[OK] Cleanup completed")
            except Exception as e:
                print(f"[WARN] Error during cleanup: {e}")
        
        cleanup_handler = cleanup
        
        # Ensure the system is initialized
        learning_loop._ensure_initialized()
        
        # Get available games
        available_games = await learning_loop.get_available_games()
        if not available_games:
            print("[ERROR] No available games found")
            return {
                'session_id': session_id,
                'return_code': -1,
                'duration': time.time() - start_time,
                'success': False,
                'error': 'No available games found'
            }
        
        # Select a single game for this session
        selected_game = available_games[0]  # Always use the first available game
        game_id = selected_game.get('game_id', selected_game)  # Handle both dict and string
        print(f"[TARGET] Selected game: {game_id}")
        
        # Run training with direct control for the specified duration
        print(f"[TARGET] Starting direct API training for {duration_minutes} minutes...")
        
        # Run training for the full duration in a single continuous session
        print(f"[TARGET] Starting continuous training session for {duration_minutes} minutes...")
        
        # Coordinate shutdown between main script and learning loop
        if shutdown_requested:
            learning_loop.request_shutdown()
        
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
        print(f"[ERROR] Error in training session #{session_id}: {e}")
        return {
            'session_id': session_id,
            'return_code': -1,
            'duration': duration,
            'success': False,
            'error': str(e)
        }
    
    finally:
        # Cleanup resources for this session
        try:
            if 'learning_loop' in locals():
                await learning_loop.close()
        except Exception as e:
            print(f"[WARN] Error cleaning up session #{session_id}: {e}")

async def main():
    """Main function for simple 9-hour training with direct API control."""
    global shutdown_requested, cleanup_handler
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    print("=" * 80)
    print("TABULA RASA - SIMPLE 9 HOUR CONTINUOUS TRAINING")
    print("=" * 80)
    print()
    print("[START] Starting DIRECT API sequential training session...")
    print("[INFO] Duration: 9 hours (540 minutes)")
    print("[INFO] Mode: Sequential with direct API control (NO subprocess parallel execution)")
    print("[INFO] Features: Enhanced learning, stable execution, graceful shutdown")
    print("[INFO] Database: Enabled (no more JSON files)")
    print("[INFO] GUARANTEED: Single-threaded execution - no parallel _train_on_game calls")
    
    # Ensure database is ready before starting training
    print("[CHECK] Checking database initialization...")
    if not ensure_database_ready():
        print("[ERROR] Database initialization failed. Training cannot proceed.")
        return 1
    
    # Record start time
    start_time = datetime.now()
    print(f"🕐 Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Press Ctrl+C to stop gracefully")
    print()
    
    # Configuration - SINGLE SESSION FOCUS
    total_duration = 9 * 60 * 60  # 9 hours in seconds
    session_duration = 60  # 60 minutes per session
    sessions_per_hour = 1  # 1 session per hour (focused on single games)
    total_sessions = 9 * sessions_per_hour  # 9 total sessions
    
    print(f"[PLAN] Training Plan:")
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
                print(f"\n[STOP] SHUTDOWN REQUESTED - Stopping training gracefully")
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
            print(f"[TIME] Time remaining: {remaining_hours:.2f} hours")
            print(f"[TIME] Session started: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"[START] Starting DIRECT API training session...")
            print()
            
            # Run training session with direct API control
            result = await run_training_session(session_count, session_duration)
            all_results.append(result)
            
            # Display result
            if result['success']:
                print(f"[OK] Session #{session_count} completed successfully in {result['duration']:.1f}s")
            else:
                print(f"[ERROR] Session #{session_count} failed (code: {result['return_code']}) in {result['duration']:.1f}s")
                if 'error' in result:
                    print(f"   Error: {result['error']}")
            
            # Calculate progress
            successful_sessions = sum(1 for r in all_results if r['success'])
            success_rate = successful_sessions / len(all_results) * 100
            
            print(f"[PROGRESS] {len(all_results)}/{total_sessions} sessions")
            print(f"[OK] Success rate: {success_rate:.1f}%")
            print()
            
            # Check for shutdown request after each session
            if shutdown_requested:
                print(f"\n[STOP] SHUTDOWN REQUESTED - Stopping training gracefully")
                print(f"Completed {session_count} sessions before shutdown")
                break
            
            # Brief pause between sessions
            await asyncio.sleep(2)
            
    except KeyboardInterrupt:
        current_time = datetime.now()
        elapsed_seconds = (current_time - start_time).total_seconds()
        print(f"\n[STOP] Training stopped by user (Ctrl+C)")
        print(f"[TIME] Total duration: {elapsed_seconds/3600:.2f} hours")
        print(f"[PROGRESS] Total sessions completed: {session_count}")
        
    except Exception as e:
        print(f"\n[ERROR] Error running training: {e}")
        return 1
    
    finally:
        # Cleanup resources
        try:
            if cleanup_handler:
                print("🧹 Cleaning up resources...")
                await cleanup_handler()
        except Exception as e:
            print(f"[WARNING] Error during cleanup: {e}")
    
    # Final statistics
    print("\n" + "=" * 80)
    print("FINAL TRAINING STATISTICS")
    print("=" * 80)
    
    if all_results:
        successful_sessions = sum(1 for r in all_results if r['success'])
        failed_sessions = len(all_results) - successful_sessions
        total_duration = sum(r['duration'] for r in all_results)
        avg_duration = total_duration / len(all_results)
        
        print(f"[SUMMARY] Overall Results:")
        print(f"   Total sessions: {len(all_results)}")
        print(f"   Successful: {successful_sessions}")
        print(f"   Failed: {failed_sessions}")
        print(f"   Success rate: {successful_sessions/len(all_results)*100:.1f}%")
        print(f"   [TIME] Total training time: {total_duration/3600:.2f} hours")
        print(f"   Average session duration: {avg_duration:.1f}s")
        print(f"   Execution mode: DIRECT API (single-threaded)")
        
        # Database-only mode: No file saving
        print(f"\n💾 Results saved to database")
    
    print(f"\nTraining session ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
