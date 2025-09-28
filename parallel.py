#!/usr/bin/env python3
"""
Simple parallel training wrapper - runs multiple train.py instances

Usage:
    python parallel.py        # Run 2 instances (default)
    python parallel.py 4      # Run 4 instances
    python parallel.py 8      # Run 8 instances
"""

import subprocess
import sys
import os
from concurrent.futures import ProcessPoolExecutor
import time
from datetime import datetime

def run_training_instance(instance_id):
    """Run a single train.py instance"""
    print(f"[START] Training instance #{instance_id}")

    env = os.environ.copy()
    env['TRAINING_INSTANCE_ID'] = str(instance_id)
    env['PYTHONUNBUFFERED'] = '1'

    start_time = time.time()

    try:
        result = subprocess.run([
            sys.executable, 'train.py'
        ], env=env, capture_output=True, text=True)

        duration = time.time() - start_time
        success = result.returncode == 0

        if success:
            print(f"[OK] Instance #{instance_id} completed successfully in {duration:.1f}s")
        else:
            print(f"[ERROR] Instance #{instance_id} failed (code: {result.returncode}) in {duration:.1f}s")

        return {
            'instance_id': instance_id,
            'success': success,
            'returncode': result.returncode,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr
        }

    except Exception as e:
        duration = time.time() - start_time
        print(f"[CRASH] Instance #{instance_id} crashed: {e}")
        return {
            'instance_id': instance_id,
            'success': False,
            'returncode': -1,
            'duration': duration,
            'error': str(e)
        }

def main():
    """Simple parallel training execution"""

    # Simple configuration
    num_instances = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    print("=" * 60)
    print("PARALLEL TRAINING WRAPPER")
    print("=" * 60)
    print(f"Starting {num_instances} parallel train.py instances")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Each instance runs train.py independently")
    print()

    start_time = time.time()

    try:
        # Simple parallel execution
        with ProcessPoolExecutor(max_workers=num_instances) as executor:
            results = list(executor.map(run_training_instance, range(1, num_instances + 1)))

        # Simple results summary
        total_duration = time.time() - start_time
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful

        print()
        print("=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Successful instances: {successful}/{num_instances}")
        print(f"Failed instances: {failed}/{num_instances}")
        print(f"Success rate: {successful/num_instances*100:.1f}%")
        print(f"Total time: {total_duration/60:.1f} minutes")
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return 0 if successful > 0 else 1

    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user (Ctrl+C)")
        print("Some instances may still be running...")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Error running parallel training: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())