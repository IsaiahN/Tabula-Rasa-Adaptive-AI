#!/usr/bin/env python3
"""
DEPRECATED: Demo functionality moved to train_arc_agent.py

This script now redirects to the unified training system.
Please use train_arc_agent.py instead for all functionality.

Usage:
    # Old way (deprecated):
    python enhanced_performance_demo.py
    
    # New way (recommended):
    python train_arc_agent.py --run-mode demo --demo-type enhanced
    
See train_arc_agent.py --help for full options.
"""

import sys
import subprocess

def main():
    print("⚠️  DEPRECATED: enhanced_performance_demo.py has been integrated into train_arc_agent.py")
    print("🔄 Redirecting to unified system...")
    print()
    
    cmd = [sys.executable, "train_arc_agent.py", "--run-mode", "demo", "--demo-type", "comparison"]
    
    print(f"Running: {' '.join(cmd)}")
    print("="*50)
    
    # Execute the new unified command
    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
