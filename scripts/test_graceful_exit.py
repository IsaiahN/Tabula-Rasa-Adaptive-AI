#!/usr/bin/env python3
"""
Test script to verify graceful exit and comprehensive data saving.
"""

import subprocess
import signal
import time
import sys
import sqlite3
import os
from datetime import datetime

def test_graceful_exit():
    """Test graceful exit with proper data saving."""
    print("Testing graceful exit and data saving...")

    # Get initial database state
    initial_state = get_database_state()
    print(f"Initial database state:")
    print(f"  Sessions: {initial_state['sessions']}")
    print(f"  Games: {initial_state['games']}")
    print(f"  Running sessions: {initial_state['running_sessions']}")

    try:
        # Start training process
        print("\nStarting training process...")
        proc = subprocess.Popen(
            [sys.executable, 'train.py'],
            cwd='C:\\Users\\Admin\\Documents\\GitHub\\tabula-rasa',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Let it run for 30 seconds to generate some data
        print("Letting process run for 30 seconds...")
        time.sleep(30)

        # Send graceful shutdown signal (SIGTERM)
        print("Sending graceful shutdown signal (SIGTERM)...")
        proc.terminate()

        # Wait for graceful shutdown (up to 60 seconds)
        print("Waiting for graceful shutdown...")
        start_wait = time.time()
        try:
            stdout, stderr = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            print("Graceful shutdown timed out, forcing kill...")
            proc.kill()
            stdout, stderr = proc.communicate()

        wait_time = time.time() - start_wait
        print(f"Process exited after {wait_time:.1f} seconds")
        print(f"Exit code: {proc.returncode}")

        # Check for cleanup messages in output
        combined_output = stdout + stderr
        cleanup_messages = [
            'CLEANUP] Starting comprehensive cleanup',
            'CLEANUP] Finishing current game',
            'CLEANUP] Closing current session',
            'CLEANUP] Saving scorecard data',
            'CLEANUP] Flushing database writes',
            'OK] Comprehensive cleanup completed'
        ]

        print("\\nChecking for cleanup messages:")
        cleanup_found = 0
        for msg in cleanup_messages:
            if msg in combined_output:
                print(f"  Found: {msg}")
                cleanup_found += 1
            else:
                print(f"  Missing: {msg}")

        print(f"Cleanup messages found: {cleanup_found}/{len(cleanup_messages)}")

        # Show some output for debugging
        if combined_output:
            print(f"\\nProcess output (first 500 chars):")
            print(combined_output[:500])
        else:
            print("\\nNo process output captured")

        # Get final database state
        time.sleep(2)  # Give database time to complete writes
        final_state = get_database_state()

        print(f"\nFinal database state:")
        print(f"  Sessions: {final_state['sessions']}")
        print(f"  Games: {final_state['games']}")
        print(f"  Running sessions: {final_state['running_sessions']}")

        # Analyze changes
        sessions_added = final_state['sessions'] - initial_state['sessions']
        games_added = final_state['games'] - initial_state['games']
        running_sessions_change = final_state['running_sessions'] - initial_state['running_sessions']

        print(f"\nDatabase changes:")
        print(f"  Sessions added: {sessions_added}")
        print(f"  Games added: {games_added}")
        print(f"  Running sessions change: {running_sessions_change}")

        # Check if sessions were properly closed
        if running_sessions_change <= 0:
            print("  Sessions were properly closed")
        else:
            print("  Sessions may not have been properly closed")

        # Success criteria
        success_criteria = [
            cleanup_found >= 4,  # At least 4 cleanup messages
            sessions_added >= 0,  # At least no sessions lost
            running_sessions_change <= 0,  # Running sessions should not increase
            proc.returncode == 0 or proc.returncode == -15  # Normal or terminated exit
        ]

        success = all(success_criteria)
        print(f"\n{'SUCCESS' if success else 'FAILURE'}: Graceful exit test")

        if not success:
            print("Issues found:")
            if cleanup_found < 4:
                print("  - Insufficient cleanup messages")
            if sessions_added < 0:
                print("  - Sessions were lost")
            if running_sessions_change > 0:
                print("  - Sessions not properly closed")

        return success

    except Exception as e:
        print(f"Error during graceful exit test: {e}")
        return False

def get_database_state():
    """Get current database state for comparison."""
    try:
        db = sqlite3.connect('C:\\Users\\Admin\\Documents\\GitHub\\tabula-rasa\\tabula_rasa.db')
        cursor = db.cursor()

        # Get session count
        cursor.execute('SELECT COUNT(*) FROM training_sessions')
        sessions = cursor.fetchone()[0]

        # Get game count
        cursor.execute('SELECT COUNT(*) FROM game_results')
        games = cursor.fetchone()[0]

        # Get running sessions count
        cursor.execute('SELECT COUNT(*) FROM training_sessions WHERE status = "running"')
        running_sessions = cursor.fetchone()[0]

        db.close()

        return {
            'sessions': sessions,
            'games': games,
            'running_sessions': running_sessions
        }

    except Exception as e:
        print(f"Error getting database state: {e}")
        return {'sessions': 0, 'games': 0, 'running_sessions': 0}

if __name__ == "__main__":
    success = test_graceful_exit()
    sys.exit(0 if success else 1)