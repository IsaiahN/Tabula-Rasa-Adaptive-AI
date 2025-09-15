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
import psutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
import json

def analyze_system_resources() -> Dict[str, Any]:
    """Analyze system resources and determine optimal concurrent session count."""
    print("🔍 Analyzing system resources...")
    
    # Get system information
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Calculate available resources
    total_ram_gb = memory.total / (1024**3)
    available_ram_gb = memory.available / (1024**3)
    used_ram_gb = memory.used / (1024**3)
    ram_usage_percent = memory.percent
    
    print(f"💾 Memory Analysis:")
    print(f"   • Total RAM: {total_ram_gb:.1f} GB")
    print(f"   • Available RAM: {available_ram_gb:.1f} GB")
    print(f"   • Used RAM: {used_ram_gb:.1f} GB ({ram_usage_percent:.1f}%)")
    print(f"   • CPU Cores: {cpu_count}")
    print(f"   • Current CPU Usage: {cpu_percent:.1f}%")
    
    # Estimate memory per training session (conservative estimate)
    estimated_memory_per_session = 0.5  # 500MB per session (conservative)
    estimated_cpu_per_session = 25  # 25% CPU per session (conservative)
    
    # Calculate maximum concurrent sessions based on available resources
    max_sessions_by_ram = int(available_ram_gb * 0.8 / estimated_memory_per_session)  # Use 80% of available RAM
    max_sessions_by_cpu = int(cpu_count * 4)  # 4 sessions per CPU core max
    max_sessions_by_ram_usage = int((100 - ram_usage_percent) / 10)  # 1 session per 10% available RAM
    
    # Take the most conservative estimate
    max_concurrent_sessions = min(max_sessions_by_ram, max_sessions_by_cpu, max_sessions_by_cpu)
    
    # Apply safety limits
    max_concurrent_sessions = max(1, min(max_concurrent_sessions, 20))  # Between 1 and 20 sessions
    
    # Determine session duration based on resources
    if total_ram_gb >= 16:
        session_duration = 30  # 30 minutes for high-end systems
        memory_size = 512
    elif total_ram_gb >= 8:
        session_duration = 25  # 25 minutes for mid-range systems
        memory_size = 256
    else:
        session_duration = 20  # 20 minutes for lower-end systems
        memory_size = 128
    
    # Adjust based on current CPU usage
    if cpu_percent > 80:
        max_concurrent_sessions = max(1, max_concurrent_sessions // 2)
        print(f"⚠️ High CPU usage detected, reducing concurrent sessions")
    elif cpu_percent > 60:
        max_concurrent_sessions = max(1, int(max_concurrent_sessions * 0.8))
        print(f"⚠️ Moderate CPU usage, slightly reducing concurrent sessions")
    
    print(f"\n🧠 Intelligent Resource Analysis:")
    print(f"   • Max sessions by RAM: {max_sessions_by_ram}")
    print(f"   • Max sessions by CPU: {max_sessions_by_cpu}")
    print(f"   • Max sessions by RAM usage: {max_sessions_by_ram_usage}")
    print(f"   • Recommended concurrent sessions: {max_concurrent_sessions}")
    print(f"   • Recommended session duration: {session_duration} minutes")
    print(f"   • Recommended memory size: {memory_size} MB")
    
    return {
        'max_concurrent_sessions': max_concurrent_sessions,
        'session_duration': session_duration,
        'memory_size': memory_size,
        'total_ram_gb': total_ram_gb,
        'available_ram_gb': available_ram_gb,
        'cpu_count': cpu_count,
        'cpu_percent': cpu_percent,
        'ram_usage_percent': ram_usage_percent
    }

def run_single_training_session(session_id: int, duration_minutes: int = 30) -> Dict[str, Any]:
    """Run a single training session with specific parameters."""
    print(f"🚀 Starting training session #{session_id}")
    
    # Set environment variables for optimal performance
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['TRAINING_SESSION_ID'] = str(session_id)
    
    # Build the command with OPTIMIZED INTELLIGENCE settings
    cmd = [
        'python', 'master_arc_trainer.py',
        '--mode', 'maximum-intelligence',
        '--session-duration', str(duration_minutes),
        '--max-actions', '5',         # Optimized action limit for better learning
        '--max-cycles', '100',        # Reduced per session
        '--target-score', '85.0',     # Target score
        '--enable-detailed-monitoring',
        '--enable-action-intelligence',
        '--enable-knowledge-transfer',
        '--enable-pattern-recognition',
        '--enable-coordinates',
        '--enable-predictive-coordinates',
        '--enable-action-experimentation',
        '--enable-exploration-strategies',
        '--enable-stagnation-detection',
        '--salience-threshold', '0.4',  # Balanced learning
        '--salience-decay', '0.95',
        '--memory-size', '512',        # Smaller per session
        '--memory-word-size', '64',
        '--memory-read-heads', '4',
        '--memory-write-heads', '1',
        '--dashboard', 'console',
        '--verbose'
    ]
    
    start_time = time.time()
    
    try:
        # Run the training with proper encoding handling
        process = subprocess.run(
            cmd, 
            env=env, 
            check=False, 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            errors='replace'  # Replace problematic characters instead of failing
        )
        
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
    
    # Use intelligent resource detection with reasonable safety limits
    max_workers = min(num_sessions, 20)  # Cap at 20 for safety, but respect intelligent detection
    
    if num_sessions > max_workers:
        print(f"⚠️ Limiting to {max_workers} concurrent workers for system stability")
        print(f"   (Requested: {num_sessions}, Using: {max_workers})")
        print()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
    print("TABULA RASA - INTELLIGENT SCALED 9 HOUR CONTINUOUS TRAINING")
    print("=" * 80)
    print()
    print("🚀 Starting intelligent parallel training session...")
    print("⏱️ Duration: 9 hours (540 minutes)")
    print("🎮 Mode: Intelligent parallel with dynamic resource optimization")
    print("🧠 Features: RAM-aware scaling, CPU optimization, adaptive learning")
    
    # Analyze system resources and determine optimal configuration
    resource_analysis = analyze_system_resources()
    
    # Record start time
    start_time = datetime.now()
    print(f"🕐 Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Press Ctrl+C to stop gracefully (will close all games)")
    print()
    
    # Use intelligent configuration based on system analysis
    total_duration = 9 * 60 * 60  # 9 hours in seconds
    session_duration = resource_analysis['session_duration']  # Dynamic based on RAM
    sessions_per_round = resource_analysis['max_concurrent_sessions']  # Dynamic based on resources
    round_duration = session_duration * 60  # Dynamic round duration
    
    total_rounds = total_duration // round_duration
    print(f"📊 Intelligent Training Plan:")
    print(f"   • Total rounds: {total_rounds}")
    print(f"   • Sessions per round: {sessions_per_round} (auto-detected based on {resource_analysis['total_ram_gb']:.1f}GB RAM)")
    print(f"   • Session duration: {session_duration} minutes (optimized for {resource_analysis['total_ram_gb']:.1f}GB RAM)")
    print(f"   • Total concurrent games: {total_rounds * sessions_per_round}")
    print(f"   • Memory per session: {resource_analysis['memory_size']}MB")
    print(f"   • System: {resource_analysis['cpu_count']} cores, {resource_analysis['total_ram_gb']:.1f}GB RAM")
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
            
            # Check current resource usage and adjust if needed
            current_memory = psutil.virtual_memory()
            current_cpu = psutil.cpu_percent(interval=0.1)
            
            # Dynamic adjustment based on current resource usage
            if current_memory.percent > 90:
                sessions_per_round = max(1, sessions_per_round // 2)
                print(f"⚠️ High memory usage ({current_memory.percent:.1f}%), reducing to {sessions_per_round} concurrent sessions")
            elif current_memory.percent > 80:
                sessions_per_round = max(1, int(sessions_per_round * 0.8))
                print(f"⚠️ Moderate memory usage ({current_memory.percent:.1f}%), reducing to {sessions_per_round} concurrent sessions")
            
            if current_cpu > 90:
                sessions_per_round = max(1, sessions_per_round // 2)
                print(f"⚠️ High CPU usage ({current_cpu:.1f}%), reducing to {sessions_per_round} concurrent sessions")
            elif current_cpu > 80:
                sessions_per_round = max(1, int(sessions_per_round * 0.8))
                print(f"⚠️ Moderate CPU usage ({current_cpu:.1f}%), reducing to {sessions_per_round} concurrent sessions")
            
            print(f"🎮 Starting {sessions_per_round} concurrent training sessions...")
            print(f"💾 Current memory usage: {current_memory.percent:.1f}%")
            print(f"🖥️ Current CPU usage: {current_cpu:.1f}%")
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
