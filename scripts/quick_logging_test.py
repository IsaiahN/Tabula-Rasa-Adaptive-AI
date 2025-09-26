#!/usr/bin/env python3
"""
Quick test to verify logging behavior without running full training.
"""

import sys
import os
import logging
import subprocess
import time

def test_logging_setup():
    """Test the logging setup by running the training script briefly."""
    print("Testing logging setup with a very brief run...")

    try:
        # Run the training script for just 5 seconds
        proc = subprocess.Popen(
            [sys.executable, 'run_9hour_simple_training.py'],
            cwd='C:\\Users\\Admin\\Documents\\GitHub\\tabula-rasa',
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stdout and stderr
            text=True
        )

        # Wait 5 seconds then kill it
        time.sleep(5)
        proc.terminate()

        # Get output
        try:
            output, _ = proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            output, _ = proc.communicate()

        print("Process output (first 20 lines):")
        lines = output.split('\n')[:20]

        info_count = 0
        error_count = 0

        for i, line in enumerate(lines):
            if line.strip():
                print(f"  {i+1}: {line}")
                if "INFO:" in line:
                    info_count += 1
                if "ERROR:" in line:
                    error_count += 1

        print(f"\nSummary:")
        print(f"  INFO messages: {info_count}")
        print(f"  ERROR messages: {error_count}")

        if info_count == 0:
            print("SUCCESS: No INFO messages in console output")
        else:
            print("ISSUE: INFO messages still appearing")

        return info_count == 0

    except Exception as e:
        print(f"Error during test: {e}")
        return False

if __name__ == "__main__":
    success = test_logging_setup()
    print(f"Test result: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)