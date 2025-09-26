#!/usr/bin/env python3
"""
Test script to verify the database logging fix works.
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.database.system_integration import get_system_integration

async def test_database_logging():
    """Test database logging with both string and enum values."""
    print("Testing database logging fix...")

    try:
        integration = get_system_integration()

        # Test with string values (this was causing the error)
        success1 = await integration.log_system_event(
            level="INFO",
            component="test_component",
            message="Test with string values",
            data={"test": "string_values"},
            session_id="test_session"
        )

        print(f"String values test: {'SUCCESS' if success1 else 'FAILED'}")

        # Test with enum values (this should also work)
        from src.database.api import LogLevel, Component

        success2 = await integration.log_system_event(
            level=LogLevel.INFO,
            component=Component.GOVERNOR,
            message="Test with enum values",
            data={"test": "enum_values"},
            session_id="test_session"
        )

        print(f"Enum values test: {'SUCCESS' if success2 else 'FAILED'}")

        if success1 and success2:
            print("All database logging tests PASSED!")
            return True
        else:
            print("Some database logging tests FAILED")
            return False

    except Exception as e:
        print(f"Database logging test failed with error: {e}")
        return False

async def main():
    """Main test function."""
    success = await test_database_logging()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)