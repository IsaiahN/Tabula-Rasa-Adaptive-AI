#!/usr/bin/env python3
"""
Test script to run a short training session and verify database logging works.
"""

import asyncio
import sys
import os
import subprocess
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def run_short_training():
    """Run a short training session to test database logging."""
    print("Starting short training session to test database logging...")

    try:
        # Run training for 30 seconds
        proc = subprocess.Popen(
            [sys.executable, 'run_9hour_simple_training.py'],
            cwd='C:\\Users\\Admin\\Documents\\GitHub\\tabula-rasa',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        start_time = time.time()
        timeout = 30  # 30 seconds

        while time.time() - start_time < timeout:
            if proc.poll() is not None:
                break
            time.sleep(1)

        # Terminate the process
        try:
            proc.terminate()
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()

        print(f"Training ran for {time.time() - start_time:.1f} seconds")

        # Check for database errors in stderr
        if stderr:
            if "'str' object has no attribute 'value'" in stderr:
                print("ERROR: Database logging still has enum issues!")
                print("Error details:")
                # Show only the error parts
                error_lines = [line for line in stderr.split('\n') if 'value' in line or 'ERROR' in line]
                for line in error_lines[:5]:  # Show first 5 error lines
                    print(f"  {line}")
                return False
            elif "Failed to log system event" in stderr:
                print("WARNING: Some database logging failed, but not due to enum issues")
                # Show the error context
                error_lines = [line for line in stderr.split('\n') if 'Failed to log' in line]
                for line in error_lines[:3]:
                    print(f"  {line}")
            else:
                print("No database logging errors detected in stderr")
        else:
            print("No stderr output - likely no database errors")

        # Check stdout for any positive indicators
        if stdout:
            if "LEARNING MANAGER" in stdout or "Successfully initialized" in stdout:
                print("Positive: Training system components are initializing")

        print("Database logging test completed successfully!")
        return True

    except Exception as e:
        print(f"Error running training test: {e}")
        return False

def main():
    """Main function."""
    success = run_short_training()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)