#!/usr/bin/env python3
"""
Migration script to move persistence_helpers_debug.log data to database table.

This script:
1. Reads the existing persistence_helpers_debug.log file
2. Parses each log entry to extract structured data
3. Inserts the data into the persistence_debug_logs table
4. Handles any parsing errors gracefully
"""

import re
import json
import sqlite3
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Add src to path to import database modules
sys.path.append(str(Path(__file__).parent / "src"))

def parse_log_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single log line and extract structured data."""
    # Log format: [2025-09-28 16:17:24,372][INFO][tabula_rasa.persistence_helpers] [persist_winning_sequence] CALLED with game_id=real_game_test, sequence=[42, 99, 7]

    # Regex pattern to extract log components
    pattern = r'\[([^\]]+)\]\[([^\]]+)\]\[([^\]]+)\] \[([^\]]+)\] (.+)'
    match = re.match(pattern, line.strip())

    if not match:
        return None

    timestamp_str, log_level, logger_name, function_name, message = match.groups()

    # Parse timestamp
    try:
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
    except ValueError:
        # Try alternative format without microseconds
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return None

    # Determine operation type
    operation_type = "UNKNOWN"
    if "CALLED with" in message:
        operation_type = "CALLED"
    elif "Success for" in message:
        operation_type = "Success"
    elif "Commit attempted for" in message:
        operation_type = "Commit attempted"
    elif "Exception:" in message:
        operation_type = "Exception"
    elif "DB:" in message and "SQL:" in message:
        operation_type = "DB_DEBUG"
    elif "Rollback attempted for" in message:
        operation_type = "Rollback attempted"

    # Extract parameters and other info based on operation type
    parameters = None
    db_info = None
    sql_query = None
    result_info = None
    session_id = None
    game_id = None

    if operation_type == "CALLED":
        # Extract parameters from CALLED messages
        if "game_id=" in message:
            game_id_match = re.search(r'game_id=([^,\s]+)', message)
            if game_id_match:
                game_id = game_id_match.group(1)

        if "session_id=" in message:
            session_id_match = re.search(r'session_id=([^,\s]+)', message)
            if session_id_match:
                session_id = session_id_match.group(1)

    elif operation_type == "DB_DEBUG":
        # Extract DB info and SQL from debug messages
        if "DB:" in message:
            db_match = re.search(r'DB: ([^|]+)', message)
            if db_match:
                db_info = db_match.group(1).strip()

        if "SQL:" in message:
            sql_match = re.search(r'SQL: ([^|]+)', message)
            if sql_match:
                sql_query = sql_match.group(1).strip()

        if "Params:" in message:
            params_match = re.search(r'Params: (.+)', message)
            if params_match:
                parameters = params_match.group(1).strip()

    elif operation_type == "Success":
        # Extract result info from success messages
        result_match = re.search(r'Result: (.+)', message)
        if result_match:
            result_info = result_match.group(1).strip()

        # Also extract game_id and session_id from success messages
        if "game_id=" in message:
            game_id_match = re.search(r'game_id=([^,\s]+)', message)
            if game_id_match:
                game_id = game_id_match.group(1)

        if "session_id=" in message:
            session_id_match = re.search(r'session_id=([^,\s]+)', message)
            if session_id_match:
                session_id = session_id_match.group(1)

    return {
        'timestamp': timestamp,
        'log_level': log_level,
        'logger_name': logger_name,
        'function_name': function_name,
        'operation_type': operation_type,
        'message': message,
        'parameters': parameters,
        'db_info': db_info,
        'sql_query': sql_query,
        'result_info': result_info,
        'session_id': session_id,
        'game_id': game_id
    }

def create_table_if_not_exists(conn: sqlite3.Connection):
    """Create the persistence_debug_logs table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS persistence_debug_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL,
            log_level TEXT NOT NULL,
            logger_name TEXT NOT NULL,
            function_name TEXT NOT NULL,
            operation_type TEXT NOT NULL,
            message TEXT NOT NULL,
            parameters TEXT,
            db_info TEXT,
            sql_query TEXT,
            result_info TEXT,
            session_id TEXT,
            game_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

def migrate_logs():
    """Main migration function."""
    log_file_path = Path("persistence_helpers_debug.log")

    if not log_file_path.exists():
        print(f"ERROR: Log file {log_file_path} does not exist")
        return False

    # Connect to database
    db_path = Path("data/tabula_rasa.db")
    db_path.parent.mkdir(exist_ok=True)

    try:
        with sqlite3.connect(str(db_path)) as conn:
            print(f"Connected to database: {db_path}")

            # Create table if needed
            create_table_if_not_exists(conn)
            print("Created/verified persistence_debug_logs table")

            # Read and process log file
            processed_count = 0
            error_count = 0

            print(f"Reading log file: {log_file_path}")

            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():  # Skip empty lines
                        parsed_data = parse_log_line(line)

                        if parsed_data:
                            try:
                                # Insert into database
                                conn.execute("""
                                    INSERT INTO persistence_debug_logs
                                    (timestamp, log_level, logger_name, function_name, operation_type,
                                     message, parameters, db_info, sql_query, result_info, session_id, game_id)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    parsed_data['timestamp'],
                                    parsed_data['log_level'],
                                    parsed_data['logger_name'],
                                    parsed_data['function_name'],
                                    parsed_data['operation_type'],
                                    parsed_data['message'],
                                    parsed_data['parameters'],
                                    parsed_data['db_info'],
                                    parsed_data['sql_query'],
                                    parsed_data['result_info'],
                                    parsed_data['session_id'],
                                    parsed_data['game_id']
                                ))
                                processed_count += 1

                                if processed_count % 100 == 0:
                                    print(f"Processed {processed_count} log entries...")

                            except Exception as e:
                                print(f"ERROR inserting line {line_num}: {e}")
                                error_count += 1
                        else:
                            print(f"WARNING: Could not parse line {line_num}: {line.strip()[:100]}...")
                            error_count += 1

            # Commit all changes
            conn.commit()

            print(f"\nMigration completed:")
            print(f"  Processed entries: {processed_count}")
            print(f"  Errors/unparsed: {error_count}")
            print(f"  Database: {db_path}")

            # Verify data was inserted
            cursor = conn.execute("SELECT COUNT(*) FROM persistence_debug_logs")
            total_rows = cursor.fetchone()[0]
            print(f"  Total rows in database: {total_rows}")

            return error_count == 0

    except Exception as e:
        print(f"ERROR during migration: {e}")
        return False

if __name__ == "__main__":
    print("=== Persistence Debug Logs Migration ===")
    success = migrate_logs()
    sys.exit(0 if success else 1)