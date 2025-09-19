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

Usage:
    python run_9hour_simple_training.py
"""

import os
import sys
import subprocess
import time
import asyncio
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

def run_training_session(session_id: int, duration_minutes: int = 15) -> Dict[str, Any]:
    """Run a single training session with specific parameters."""
    print(f"Starting training session #{session_id}")
    
    # Set environment variables for optimal performance
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['TRAINING_SESSION_ID'] = str(session_id)
    
    # Ensure API key is available
    if 'ARC_API_KEY' not in env:
        print("❌ ARC_API_KEY not found in environment variables")
        return {
            'session_id': session_id,
            'return_code': -1,
            'duration': 0,
            'success': False,
            'error': 'ARC_API_KEY not found in environment variables'
        }
    else:
        print(f"✅ ARC_API_KEY found: {env['ARC_API_KEY'][:10]}...")
    
    # Build the command with OPTIMIZED INTELLIGENCE settings
    cmd = [
        'python', 'master_arc_trainer.py',
        '--mode', 'maximum-intelligence',
        '--session-duration', str(duration_minutes),
        '--max-actions', '500',         # Optimized action limit for better learning
        '--max-cycles', '100',        # Moderate cycles
        '--target-score', '85.0',     # Target score
            # Enhanced Space-Time Governor is now the default (no disable flag needed)
        '--enable-detailed-monitoring',
        '--salience-threshold', '0.4',
        '--salience-decay', '0.95',
        '--memory-size', '512',
        '--memory-word-size', '64',
        '--memory-read-heads', '4',
        '--memory-write-heads', '1',
        '--dashboard', 'console',
        '--verbose'
    ]
    
    start_time = time.time()
    
    try:
        # Run the training without capturing output to avoid encoding issues
        process = subprocess.run(cmd, env=env, check=False)
        
        duration = time.time() - start_time
        
        return {
            'session_id': session_id,
            'return_code': process.returncode,
            'duration': duration,
            'success': process.returncode == 0
        }
        
    except Exception as e:
        duration = time.time() - start_time
        return {
            'session_id': session_id,
            'return_code': -1,
            'duration': duration,
            'success': False,
            'error': str(e)
        }

def main():
    """Main function for simple 9-hour training."""
    print("=" * 80)
    print("TABULA RASA - SIMPLE 9 HOUR CONTINUOUS TRAINING")
    print("=" * 80)
    print()
    print("Starting simple sequential training session...")
    print("Duration: 9 hours (540 minutes)")
    print("Mode: Sequential with multiple games per hour")
    print("Features: Enhanced learning, stable execution, graceful shutdown")
    print("Database: Enabled (no more JSON files)")
    
    # Ensure database is ready before starting training
    print("🔍 Checking database initialization...")
    if not ensure_database_ready():
        print("❌ Database initialization failed. Training cannot proceed.")
        return
    
    # Record start time
    start_time = datetime.now()
    print(f"🕐 Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Press Ctrl+C to stop gracefully")
    print()
    
    # Configuration
    total_duration = 9 * 60 * 60  # 9 hours in seconds
    session_duration = 60  # 60 minutes per session (increased for game completion)
    sessions_per_hour = 1  # 1 session per hour (focused on single games)
    total_sessions = 9 * sessions_per_hour  # 9 total sessions
    
    print(f"📊 Training Plan:")
    print(f"   • Total sessions: {total_sessions}")
    print(f"   • Session duration: {session_duration} minutes")
    print(f"   • Sessions per hour: {sessions_per_hour}")
    print(f"   • Total games: {total_sessions}")
    print()
    
    all_results = []
    session_count = 0
    
    try:
        while session_count < total_sessions:
            # Check for shutdown request first (will be checked in run_training_session)
            # The shutdown check is handled within the training session itself
                
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
            print(f"🎮 Starting training session...")
            print()
            
            # Run training session
            result = run_training_session(session_count, session_duration)
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
            
            # Brief pause between sessions
            time.sleep(2)
            
    except KeyboardInterrupt:
        current_time = datetime.now()
        elapsed_seconds = (current_time - start_time).total_seconds()
        print(f"\n🛑 Training stopped by user (Ctrl+C)")
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
        
        # Database-only mode: No file saving
        print(f"\n💾 Results saved to database")
    
    print(f"\nTraining session ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
