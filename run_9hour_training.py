#!/usr/bin/env python3
"""
TABULA RASA - 9 HOUR CONTINUOUS TRAINING SCRIPT

This script runs a comprehensive 9-hour continuous training session
with all advanced meta-cognitive features enabled.

Usage:
    python run_9hour_training.py

Features enabled:
- Meta-cognitive Governor (Third Brain)
- Architect Evolution (Zeroth Brain) 
- Enhanced visual targeting
- Improved energy system
- Reduced avoidance sensitivity
- Comprehensive data utilization
- Detailed monitoring and logging
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def main():
    print("=" * 60)
    print("TABULA RASA - 9 HOUR CONTINUOUS TRAINING")
    print("=" * 60)
    print()
    print("Starting enhanced meta-cognitive training session...")
    print(f"Duration: 9 hours (540 minutes)")
    print(f"Mode: Continuous with all advanced features enabled")
    
    # Record start time
    start_time = datetime.now()
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Press Ctrl+C to stop gracefully")
    print()
    
    # Set environment variables for optimal performance
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    
    # Build the command with MAXIMUM INTELLIGENCE optimizations
    cmd = [
        'python', 'master_arc_trainer.py',
        '--mode', 'maximum-intelligence',  # MAXIMUM INTELLIGENCE MODE
        '--session-duration', '540',  # 9 hours in minutes
        '--max-actions', '5000',      # More actions per game
        '--max-cycles', '500',        # More learning cycles
        '--target-score', '90.0',     # Higher target score
        '--enable-detailed-monitoring',
        '--salience-threshold', '0.4',  # Reduced from 0.6
        '--salience-decay', '0.95',
        '--memory-size', '1024',        # Larger memory
        '--memory-word-size', '128',    # Larger word size
        '--memory-read-heads', '8',     # More read heads
        '--memory-write-heads', '2',    # More write heads
        '--dashboard', 'console',
        '--verbose'
    ]
    
    session_count = 0
    total_duration = 9 * 60 * 60  # 9 hours in seconds
    
    try:
        while True:
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
            
            print("=" * 60)
            print(f"TRAINING SESSION #{session_count}")
            print("=" * 60)
            print(f"Time remaining: {remaining_hours:.2f} hours")
            print(f"Session started: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            print("🚀 Launching continuous training...")
            print(f"Command: {' '.join(cmd)}")
            print()
            
            # Run the training
            process = subprocess.run(cmd, env=env, check=False)
            
            if process.returncode == 0:
                print(f"\n✅ Training session #{session_count} completed successfully!")
            else:
                print(f"\n⚠️ Training session #{session_count} ended with return code: {process.returncode}")
            
            # Check if we still have time remaining
            current_time = datetime.now()
            elapsed_seconds = (current_time - start_time).total_seconds()
            remaining_seconds = total_duration - elapsed_seconds
            
            if remaining_seconds > 0:
                remaining_hours = remaining_seconds / 3600
                print(f"\n⏰ Time remaining: {remaining_hours:.2f} hours")
                print("🔄 Restarting training session...")
                print()
                time.sleep(2)  # Brief pause before restart
            else:
                print(f"\n🎉 9 HOUR TRAINING COMPLETE!")
                print(f"Total duration: {elapsed_seconds/3600:.2f} hours")
                print(f"Total sessions: {session_count}")
                print(f"Completed at: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                break
            
    except KeyboardInterrupt:
        current_time = datetime.now()
        elapsed_seconds = (current_time - start_time).total_seconds()
        print(f"\n🛑 Training stopped by user (Ctrl+C)")
        print(f"Total duration: {elapsed_seconds/3600:.2f} hours")
        print(f"Total sessions completed: {session_count}")
    except Exception as e:
        print(f"\n❌ Error running training: {e}")
        return 1
    
    print(f"\nTraining session ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
