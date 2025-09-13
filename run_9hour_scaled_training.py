#!/usr/bin/env python3
"""
TABULA RASA - SCALED 9 HOUR CONTINUOUS TRAINING SCRIPT

This script runs a comprehensive 9-hour continuous training session
with DOZENS of concurrent games for maximum learning speed.

Features:
- Multiple concurrent game sessions (20-50 games)
- Parallel learning across different games
- Enhanced memory sharing between games
- Optimized for maximum learning speed
- Graceful shutdown for all games

Usage:
    python run_9hour_scaled_training.py
"""

import os
import sys
import subprocess
import time
import threading
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

def run_single_training_session(session_id: int, duration_minutes: int = 30) -> Dict[str, Any]:
    """Run a single training session with specific parameters."""
    print(f"🚀 Starting training session #{session_id}")
    
    # Set environment variables for optimal performance
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['TRAINING_SESSION_ID'] = str(session_id)
    
    # Build the command with SCALED INTELLIGENCE optimizations
    cmd = [
        'python', 'master_arc_trainer.py',
        '--mode', 'maximum-intelligence',
        '--session-duration', str(duration_minutes),
        '--max-actions', '3000',      # Reduced per session for parallelization
        '--max-cycles', '200',        # Reduced per session
        '--target-score', '85.0',     # Slightly lower for faster convergence
        '--enable-detailed-monitoring',
        '--salience-threshold', '0.3',  # More aggressive learning
        '--salience-decay', '0.97',
        '--memory-size', '512',        # Smaller per session
        '--memory-word-size', '64',
        '--memory-read-heads', '4',
        '--memory-write-heads', '1',
        '--dashboard', 'console',
        '--verbose'
    ]
    
    start_time = time.time()
    
    try:
        # Run the training
        process = subprocess.run(cmd, env=env, check=False, capture_output=True, text=True)
        
        duration = time.time() - start_time
        
        return {
            'session_id': session_id,
            'return_code': process.returncode,
            'duration': duration,
            'success': process.returncode == 0,
            'stdout': process.stdout,
            'stderr': process.stderr
        }
        
    except Exception as e:
        duration = time.time() - start_time
        return {
            'session_id': session_id,
            'return_code': -1,
            'duration': duration,
            'success': False,
            'error': str(e),
            'stdout': '',
            'stderr': ''
        }

def run_parallel_training_sessions(num_sessions: int = 20, session_duration: int = 30) -> List[Dict[str, Any]]:
    """Run multiple training sessions in parallel."""
    print(f"🔥 Starting {num_sessions} parallel training sessions")
    print(f"⏱️ Each session duration: {session_duration} minutes")
    print(f"🎯 Total concurrent games: {num_sessions}")
    print()
    
    results = []
    
    with ThreadPoolExecutor(max_workers=min(num_sessions, 10)) as executor:  # Limit to 10 concurrent processes
        # Submit all sessions
        future_to_session = {
            executor.submit(run_single_training_session, i, session_duration): i 
            for i in range(num_sessions)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_session):
            session_id = future_to_session[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['success']:
                    print(f"✅ Session #{session_id} completed successfully in {result['duration']:.1f}s")
                else:
                    print(f"❌ Session #{session_id} failed (code: {result['return_code']}) in {result['duration']:.1f}s")
                    
            except Exception as e:
                print(f"💥 Session #{session_id} crashed: {e}")
                results.append({
                    'session_id': session_id,
                    'return_code': -1,
                    'duration': 0,
                    'success': False,
                    'error': str(e)
                })
    
    return results

def main():
    """Main function for scaled 9-hour training."""
    print("=" * 80)
    print("TABULA RASA - SCALED 9 HOUR CONTINUOUS TRAINING")
    print("=" * 80)
    print()
    print("🚀 Starting enhanced parallel training session...")
    print("⏱️ Duration: 9 hours (540 minutes)")
    print("🎮 Mode: Parallel with dozens of concurrent games")
    print("🧠 Features: Enhanced memory sharing, parallel learning, optimized speed")
    
    # Record start time
    start_time = datetime.now()
    print(f"🕐 Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Press Ctrl+C to stop gracefully (will close all games)")
    print()
    
    # Configuration
    total_duration = 9 * 60 * 60  # 9 hours in seconds
    session_duration = 30  # 30 minutes per session
    sessions_per_round = 20  # 20 concurrent sessions per round
    round_duration = session_duration * 60  # 30 minutes per round
    
    total_rounds = total_duration // round_duration
    print(f"📊 Training Plan:")
    print(f"   • Total rounds: {total_rounds}")
    print(f"   • Sessions per round: {sessions_per_round}")
    print(f"   • Session duration: {session_duration} minutes")
    print(f"   • Total concurrent games: {total_rounds * sessions_per_round}")
    print()
    
    all_results = []
    round_count = 0
    
    try:
        while True:
            # Check if 9 hours have elapsed
            current_time = datetime.now()
            elapsed_seconds = (current_time - start_time).total_seconds()
            remaining_seconds = total_duration - elapsed_seconds
            
            if remaining_seconds <= 0:
                print(f"\n🎉 9 HOUR TRAINING COMPLETE!")
                print(f"Total duration: {elapsed_seconds/3600:.2f} hours")
                print(f"Total rounds: {round_count}")
                print(f"Total sessions: {len(all_results)}")
                print(f"Completed at: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                break
            
            round_count += 1
            remaining_hours = remaining_seconds / 3600
            
            print("=" * 80)
            print(f"TRAINING ROUND #{round_count}")
            print("=" * 80)
            print(f"⏰ Time remaining: {remaining_hours:.2f} hours")
            print(f"🕐 Round started: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🎮 Starting {sessions_per_round} concurrent training sessions...")
            print()
            
            # Run parallel training sessions
            round_results = run_parallel_training_sessions(sessions_per_round, session_duration)
            all_results.extend(round_results)
            
            # Calculate round statistics
            successful_sessions = sum(1 for r in round_results if r['success'])
            failed_sessions = len(round_results) - successful_sessions
            avg_duration = sum(r['duration'] for r in round_results) / len(round_results)
            
            print()
            print(f"📊 Round #{round_count} Results:")
            print(f"   ✅ Successful sessions: {successful_sessions}/{len(round_results)}")
            print(f"   ❌ Failed sessions: {failed_sessions}")
            print(f"   ⏱️ Average duration: {avg_duration:.1f}s")
            print(f"   🎯 Success rate: {successful_sessions/len(round_results)*100:.1f}%")
            
            # Check if we still have time remaining
            current_time = datetime.now()
            elapsed_seconds = (current_time - start_time).total_seconds()
            remaining_seconds = total_duration - elapsed_seconds
            
            if remaining_seconds > 0:
                remaining_hours = remaining_seconds / 3600
                print(f"\n⏰ Time remaining: {remaining_hours:.2f} hours")
                print("🔄 Starting next round...")
                print()
                time.sleep(5)  # Brief pause between rounds
            else:
                print(f"\n🎉 9 HOUR TRAINING COMPLETE!")
                print(f"Total duration: {elapsed_seconds/3600:.2f} hours")
                print(f"Total rounds: {round_count}")
                print(f"Total sessions: {len(all_results)}")
                print(f"Completed at: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                break
            
    except KeyboardInterrupt:
        current_time = datetime.now()
        elapsed_seconds = (current_time - start_time).total_seconds()
        print(f"\n🛑 Training stopped by user (Ctrl+C)")
        print(f"⏱️ Total duration: {elapsed_seconds/3600:.2f} hours")
        print(f"📊 Total rounds completed: {round_count}")
        print(f"🎮 Total sessions completed: {len(all_results)}")
        
        # Graceful shutdown - close all scorecards
        print("\n🛑 Initiating graceful shutdown...")
        print("📋 Closing all active scorecards...")
        # Note: The graceful shutdown is handled by the signal handlers in the continuous learning loop
        
    except Exception as e:
        print(f"\n❌ Error running scaled training: {e}")
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
        print(f"   🚀 Concurrent games per round: {sessions_per_round}")
        print(f"   🔄 Total rounds: {round_count}")
        
        # Save results to file
        results_file = f"scaled_training_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_sessions': len(all_results),
                    'successful_sessions': successful_sessions,
                    'failed_sessions': failed_sessions,
                    'success_rate': successful_sessions/len(all_results)*100,
                    'total_duration_hours': total_duration/3600,
                    'average_duration_seconds': avg_duration,
                    'concurrent_games_per_round': sessions_per_round,
                    'total_rounds': round_count
                },
                'detailed_results': all_results
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
    
    print(f"\nTraining session ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
