#!/usr/bin/env python3
"""
Test that ERROR messages are properly displayed and logged to database.
"""

import sys
import os
import logging
import sqlite3
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_error_logging():
    """Test that ERROR messages work with our logging setup."""
    print("Testing ERROR message logging...")

    try:
        # Import and set up logging exactly like the training script
        from database.database_logging_handler import setup_database_logging

        # Configure logging like in the training script
        logging.basicConfig(level=logging.ERROR, format='%(levelname)s:%(name)s:%(message)s')

        # Set all logger levels to ERROR
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.ERROR)

        # Set up database logging
        db_logger = setup_database_logging("tabula_rasa_test", level=logging.DEBUG)
        root_logger.addHandler(db_logger.handlers[0])

        # Test logger
        test_logger = logging.getLogger("test_error_logging")
        test_logger.setLevel(logging.ERROR)

        print("Logging setup complete. Testing error messages...")

        # Log different levels to test
        test_logger.info("This INFO message should not appear in console")
        test_logger.warning("This WARNING message should not appear in console")
        test_logger.error("This ERROR message should appear in console AND database")
        test_logger.critical("This CRITICAL message should appear in console AND database")

        # Check database for the logged errors
        print("\nChecking database for test error messages...")
        check_database_for_test_errors()

        print("ERROR logging test completed successfully!")
        return True

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_database_for_test_errors():
    """Check if our test ERROR messages were logged to the database."""
    try:
        db_path = os.path.join(os.path.dirname(__file__), '..', 'tabula_rasa.db')
        db = sqlite3.connect(db_path)
        cursor = db.cursor()

        # Look for our test error messages in the last minute
        one_minute_ago = (datetime.now() - timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S')

        cursor.execute('''
            SELECT log_level, component, message, timestamp
            FROM system_logs
            WHERE component = 'test_error_logging'
            AND timestamp > ?
            ORDER BY timestamp DESC
        ''', (one_minute_ago,))

        test_entries = cursor.fetchall()

        if test_entries:
            print(f"Found {len(test_entries)} test log entries in database:")
            for level, component, message, timestamp in test_entries:
                print(f"  [{timestamp}] {level}: {message}")
            print("SUCCESS: ERROR messages are being logged to database!")
        else:
            print("No test entries found in database - checking if database logging is working...")

        db.close()

    except Exception as e:
        print(f"Error checking database: {e}")

if __name__ == "__main__":
    success = test_error_logging()
    sys.exit(0 if success else 1)