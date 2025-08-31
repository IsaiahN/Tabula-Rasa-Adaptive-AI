#!/usr/bin/env python3
"""
DEPRECATED: Test functionality moved to train_arc_agent.py

This script now redirects to the unified training system.
Please use train_arc_agent.py instead for all functionality.

Usage:
    # Old way (deprecated):
    python run_tests.py --type unit
    
    # New way (recommended):
    python train_arc_agent.py --run-mode test --test-type unit
    
See train_arc_agent.py --help for full options.
"""

import sys
import subprocess

def main():
    print("⚠️  DEPRECATED: run_tests.py has been integrated into train_arc_agent.py")
    print("� Redirecting to unified system...")
    print()
    
    # Map old arguments to new system
    if len(sys.argv) == 1:
        # No arguments, show help
        cmd = [sys.executable, "train_arc_agent.py", "--run-mode", "test", "--help"]
    else:
        cmd = [sys.executable, "train_arc_agent.py", "--run-mode", "test"]
        
        # Map old arguments to new ones
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == "--type":
                i += 1
                if i < len(sys.argv):
                    cmd.extend(["--test-type", sys.argv[i]])
            elif sys.argv[i] == "--mode":
                i += 1
                if i < len(sys.argv):
                    cmd.extend(["--arc3-mode", sys.argv[i]])
            else:
                cmd.append(sys.argv[i])
            i += 1
    
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