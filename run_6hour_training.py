#!/usr/bin/env python3
"""
TABULA RASA - 6 HOUR CONTINUOUS TRAINING SCRIPT

This script runs a comprehensive 6-hour continuous training session
with all advanced meta-cognitive features enabled.

Usage:
    python run_6hour_training.py

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
    print("TABULA RASA - 6 HOUR CONTINUOUS TRAINING")
    print("=" * 60)
    print()
    print("Starting enhanced meta-cognitive training session...")
    print(f"Duration: 6 hours (360 minutes)")
    print(f"Mode: Continuous with all advanced features enabled")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
        '--session-duration', '360',  # 6 hours in minutes
        '--max-actions', '1000',      # More actions per game
        '--max-cycles', '100',        # More learning cycles
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
    
    try:
        print("🚀 Launching continuous training...")
        print(f"Command: {' '.join(cmd)}")
        print()
        
        # Run the training
        process = subprocess.run(cmd, env=env, check=False)
        
        if process.returncode == 0:
            print("\n✅ Training session completed successfully!")
        else:
            print(f"\n⚠️ Training session ended with return code: {process.returncode}")
            
    except KeyboardInterrupt:
        print("\n🛑 Training stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error running training: {e}")
        return 1
    
    print(f"\nTraining session ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
