#!/usr/bin/env python3
"""
Quick test to verify database logging is working for the components we've added.
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.database.system_integration import get_system_integration

async def test_database_logging():
    """Test database logging integration."""
    print("Testing database logging...")

    try:
        # Test system integration
        integration = get_system_integration()

        # Test logging a sample event
        await integration.log_system_event(
            level="INFO",
            component="test_component",
            message="Database logging test",
            data={
                'test_type': 'database_logging_verification',
                'timestamp': '2025-09-25',
                'components_tested': ['enhanced_space_time_governor', 'master_trainer', 'stagnation_intervention_system']
            },
            session_id='test_session'
        )

        print("Database logging test successful")
        return True

    except Exception as e:
        print(f"Database logging test failed: {e}")
        return False

async def check_table_counts_after_test():
    """Check table counts to see if new logs were added."""
    try:
        from src.database.api import DatabaseAPI

        db_api = DatabaseAPI()

        # Check system_logs table
        result = await db_api.execute_query(
            "SELECT COUNT(*) as count FROM system_logs WHERE component = 'test_component'"
        )

        count = result[0]['count'] if result else 0
        print(f"Test entries in system_logs: {count}")

        return count > 0

    except Exception as e:
        print(f"Failed to check table counts: {e}")
        return False

async def main():
    """Main test function."""
    print("Starting database logging verification test...")

    # Test basic logging
    logging_success = await test_database_logging()

    # Check if logs were persisted
    persistence_success = await check_table_counts_after_test()

    if logging_success and persistence_success:
        print("All database logging tests passed!")
        return 0
    else:
        print("Some database logging tests failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)