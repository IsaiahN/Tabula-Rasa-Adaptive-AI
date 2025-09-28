#!/usr/bin/env python3
"""
Run test cycles with 5-minute timeout to verify database updates.
"""

import asyncio
import sys
import os
import time
import signal
import sqlite3
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def check_database_changes():
    """Check for recent database changes."""
    try:
        db = sqlite3.connect('tabula_rasa.db')
        cursor = db.cursor()

        # Check recent system_logs entries (last 10 minutes)
        cursor.execute('''
            SELECT component, COUNT(*) as count
            FROM system_logs
            WHERE timestamp > datetime("now", "-10 minutes")
            GROUP BY component
            ORDER BY count DESC
        ''')

        recent_logs = cursor.fetchall()

        # Check key tables for recent activity
        tables_to_check = [
            'training_sessions', 'game_results', 'action_effectiveness',
            'coordinate_intelligence', 'system_logs'
        ]

        print(f"Database activity in last 10 minutes:")

        if recent_logs:
            print("Recent system logs by component:")
            for component, count in recent_logs[:10]:
                print(f"  {component}: {count} entries")
        else:
            print("  No recent system logs found")

        # Check table row counts
        print("\nCurrent table counts:")
        for table in tables_to_check:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  {table}: {count} rows")

        db.close()
        return len(recent_logs) > 0

    except Exception as e:
        print(f"Error checking database: {e}")
        return False

async def run_training_cycle(cycle_num: int, timeout_seconds: int = 300):
    """Run a single training cycle with timeout."""
    print(f"\n{'='*50}")
    print(f"CYCLE {cycle_num}: Starting {timeout_seconds//60}-minute training cycle")
    print(f"{'='*50}")

    start_time = time.time()

    try:
        # Use subprocess to run training with timeout
        proc = await asyncio.create_subprocess_exec(
            sys.executable, 'train.py',
            cwd='C:\\Users\\Admin\\Documents\\GitHub\\tabula-rasa',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_seconds
            )

            return_code = proc.returncode

        except asyncio.TimeoutError:
            print(f"Cycle {cycle_num}: Timeout reached, terminating process...")
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
            return_code = -1
            stdout, stderr = b"", b"Timeout"

        elapsed = time.time() - start_time

        print(f"Cycle {cycle_num}: Completed in {elapsed:.1f}s (return code: {return_code})")

        # Check for errors
        if stderr:
            stderr_str = stderr.decode('utf-8', errors='ignore')
            if "UnicodeEncodeError" not in stderr_str:  # Ignore unicode errors
                print(f"Cycle {cycle_num}: stderr output (last 500 chars):")
                print(stderr_str[-500:])

        # Check database activity
        print(f"\nCycle {cycle_num}: Checking database activity...")
        had_activity = check_database_changes()

        if had_activity:
            print(f"Cycle {cycle_num}:  Database activity detected")
        else:
            print(f"Cycle {cycle_num}:  No recent database activity")

        return {
            'cycle': cycle_num,
            'duration': elapsed,
            'return_code': return_code,
            'database_activity': had_activity,
            'timeout': elapsed >= timeout_seconds - 5  # Consider timeout if within 5 seconds
        }

    except Exception as e:
        print(f"Cycle {cycle_num}: Error - {e}")
        return {
            'cycle': cycle_num,
            'duration': time.time() - start_time,
            'return_code': -2,
            'database_activity': False,
            'error': str(e)
        }

async def main():
    """Main function to run test cycles."""
    print("TABULA RASA - DATABASE UPDATE VERIFICATION")
    print("Running 5-minute training cycles until database issues are fixed")
    print()

    cycles_run = 0
    max_cycles = 12  # Maximum 1 hour of testing
    timeout_seconds = 300  # 5 minutes

    results = []

    while cycles_run < max_cycles:
        cycles_run += 1

        # Check initial database state
        print(f"Pre-cycle {cycles_run} database check:")
        check_database_changes()

        # Run training cycle
        result = await run_training_cycle(cycles_run, timeout_seconds)
        results.append(result)

        # Analyze results
        if result.get('database_activity', False):
            print(f" Cycle {cycles_run}: Database is being updated!")
        else:
            print(f" Cycle {cycles_run}: Database updates may be missing")

        # Check if we should continue
        recent_activity = sum(1 for r in results[-3:] if r.get('database_activity', False))

        if recent_activity >= 2:  # At least 2 of last 3 cycles had activity
            print(f"\n SUCCESS: Database updates are working consistently!")
            print("Recent cycles show consistent database activity.")
            break

        if cycles_run >= 5 and recent_activity == 0:
            print(f"\n ISSUE: No database activity in last 5 cycles")
            print("Database updates may not be working properly.")
            break

        # Short pause between cycles
        if cycles_run < max_cycles:
            print(f"\nPausing 30 seconds before next cycle...")
            await asyncio.sleep(30)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Cycles run: {len(results)}")

    active_cycles = sum(1 for r in results if r.get('database_activity', False))
    print(f"Cycles with database activity: {active_cycles}/{len(results)}")

    if active_cycles >= len(results) * 0.6:  # 60% success rate
        print(" RESULT: Database updates are working adequately")
        return 0
    else:
        print(" RESULT: Database update issues detected")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)