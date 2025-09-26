#!/usr/bin/env python3
"""
Test script to verify the logging modifications work correctly.
"""

import subprocess
import sys
import os
import time
import sqlite3

def test_logging_behavior():
    """Test that INFO messages are hidden from console but ERROR messages are shown and logged."""
    print("Testing logging behavior modifications...")

    # Run the training script for a short time and capture output
    try:
        proc = subprocess.Popen(
            [sys.executable, 'run_9hour_simple_training.py'],
            cwd='C:\\Users\\Admin\\Documents\\GitHub\\tabula-rasa',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Let it run for 15 seconds
        time.sleep(15)

        # Terminate and capture output
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()

        print(f"Process ran for ~15 seconds")
        print(f"Return code: {proc.returncode}")

        # Check console output for INFO messages
        combined_output = stdout + stderr
        info_count = combined_output.count("INFO:")
        error_count = combined_output.count("ERROR:")

        print(f"Console output analysis:")
        print(f"  INFO messages in console: {info_count}")
        print(f"  ERROR messages in console: {error_count}")

        if info_count == 0:
            print("✓ SUCCESS: No INFO messages shown in console (as intended)")
        else:
            print("✗ ISSUE: INFO messages still appearing in console")

        if error_count > 0:
            print(f"✓ ERROR messages displayed: {error_count} (this is expected)")
            print("Sample error messages:")
            error_lines = [line for line in combined_output.split('\n') if 'ERROR:' in line]
            for line in error_lines[:3]:
                print(f"  {line}")

        # Check database for recent ERROR log entries
        print("\nChecking database for ERROR log entries...")
        check_database_error_logs()

        return info_count == 0  # Success if no INFO messages in console

    except Exception as e:
        print(f"Error during test: {e}")
        return False

def check_database_error_logs():
    """Check if ERROR messages are being logged to the database."""
    try:
        db = sqlite3.connect('C:\\Users\\Admin\\Documents\\GitHub\\tabula-rasa\\tabula_rasa.db')
        cursor = db.cursor()

        # Check for recent ERROR logs
        cursor.execute('''
            SELECT COUNT(*)
            FROM system_logs
            WHERE log_level = 'ERROR'
            AND timestamp > datetime('now', '-1 hour')
        ''')

        error_count = cursor.fetchone()[0]

        print(f"ERROR entries in database (last hour): {error_count}")

        if error_count > 0:
            print("✓ SUCCESS: ERROR messages are being logged to database")

            # Show sample ERROR messages
            cursor.execute('''
                SELECT logger_name, message, timestamp
                FROM system_logs
                WHERE log_level = 'ERROR'
                AND timestamp > datetime('now', '-1 hour')
                ORDER BY timestamp DESC
                LIMIT 3
            ''')

            recent_errors = cursor.fetchall()
            print("Recent ERROR log entries:")
            for logger, message, timestamp in recent_errors:
                print(f"  [{timestamp}] {logger}: {message}")
        else:
            print("ℹ INFO: No recent ERROR entries (this is fine if no errors occurred)")

        db.close()

    except Exception as e:
        print(f"Error checking database: {e}")

def main():
    """Main function."""
    success = test_logging_behavior()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)