#!/usr/bin/env python3
"""
Database Issues Fix Script

This script identifies and fixes common database issues like column mismatches,
parameter binding errors, and schema inconsistencies.
"""

import sys
import os
import sqlite3
import json
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.database_validator import validate_and_fix_database
from database.error_handler import DatabaseErrorHandler

def main():
    """Main function to fix database issues."""
    print("🔧 TABULA RASA DATABASE FIX SCRIPT")
    print("=" * 50)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Check if database exists
    db_path = "./tabula_rasa.db"
    if not os.path.exists(db_path):
        print(f"❌ Database not found at {db_path}")
        print("Please run the training system first to create the database.")
        return 1
    
    print(f"📊 Analyzing database: {db_path}")
    
    # Run comprehensive validation
    print("\n🔍 Running database validation...")
    report = validate_and_fix_database(db_path)
    print(report)
    
    # Test the error handler
    print("\n🧪 Testing error handler...")
    error_handler = DatabaseErrorHandler()
    
    # Test with a problematic query
    test_query = "INSERT INTO test_table (col1, col2) VALUES (?, ?, ?)"  # 3 values for 2 columns
    test_params = ("value1", "value2", "extra_value")
    
    success, fix_msg = error_handler.handle_database_error(
        Exception("9 values for 8 columns"), 
        test_query, 
        test_params, 
        "test_table"
    )
    
    if success:
        print(f"✅ Error handler test passed: {fix_msg}")
    else:
        print(f"❌ Error handler test failed: {fix_msg}")
    
    print("\n🎯 RECOMMENDATIONS:")
    print("1. Use the DatabaseErrorHandler in all database operations")
    print("2. Run this script regularly to catch issues early")
    print("3. Consider implementing database schema versioning")
    print("4. Add unit tests for database operations")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
