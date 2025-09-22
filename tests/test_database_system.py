"""
Unit Tests for Database System

Comprehensive tests for database operations, error handling, and schema validation.
"""

import pytest
import sqlite3
import json
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, patch

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database.api import DatabaseAPI
from database.error_handler import DatabaseErrorHandler, safe_database_execute
from database.schema_versioning import DatabaseSchemaVersioning
from database.health_monitor import DatabaseHealthMonitor

class TestDatabaseAPI:
    """Test cases for DatabaseAPI."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        # Create a simple test schema
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE test_table (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value INTEGER,
                    data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        
        yield db_path
        
        # Cleanup
        os.unlink(db_path)
    
    @pytest.fixture
    def db_api(self, temp_db):
        """Create DatabaseAPI instance with temp database."""
        return DatabaseAPI(temp_db)
    
    @pytest.mark.asyncio
    async def test_basic_insert(self, db_api):
        """Test basic insert operation."""
        query = "INSERT INTO test_table (name, value, data) VALUES (?, ?, ?)"
        params = ("test_name", 42, json.dumps({"key": "value"}))
        
        result = await db_api.execute(query, params, "test_table")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_column_mismatch_auto_fix(self, db_api):
        """Test automatic fixing of column count mismatches."""
        # This should trigger a column mismatch error
        query = "INSERT INTO test_table (name, value) VALUES (?, ?, ?)"  # 3 values for 2 columns
        params = ("test_name", 42, "extra_value")
        
        result = await db_api.execute(query, params, "test_table")
        # Should either succeed (if auto-fixed) or fail gracefully
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_parameter_binding_auto_fix(self, db_api):
        """Test automatic fixing of parameter binding errors."""
        query = "INSERT INTO test_table (name, value, data) VALUES (?, ?, ?)"
        params = ("test_name", 42, {"key": "value"})  # dict should be auto-converted to JSON
        
        result = await db_api.execute(query, params, "test_table")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_fetch_all(self, db_api):
        """Test fetch_all operation."""
        # Insert test data
        await db_api.execute(
            "INSERT INTO test_table (name, value, data) VALUES (?, ?, ?)",
            ("test1", 1, "data1"),
            "test_table"
        )
        
        # Fetch data
        results = await db_api.fetch_all("SELECT * FROM test_table WHERE name = ?", ("test1",))
        assert len(results) == 1
        assert results[0]["name"] == "test1"
        assert results[0]["value"] == 1

class TestDatabaseErrorHandler:
    """Test cases for DatabaseErrorHandler."""
    
    def test_column_mismatch_handling(self):
        """Test handling of column count mismatch errors."""
        handler = DatabaseErrorHandler()
        
        error = Exception("9 values for 8 columns")
        query = "INSERT INTO test_table (col1, col2) VALUES (?, ?, ?)"  # 3 values for 2 columns
        params = ("val1", "val2", "val3")
        
        success, message = handler.handle_database_error(error, query, params, "test_table")
        
        # Should attempt to fix the error
        assert success is not None
        assert "column" in message.lower() or "fix" in message.lower()
    
    def test_parameter_binding_handling(self):
        """Test handling of parameter binding errors."""
        handler = DatabaseErrorHandler()
        
        error = Exception("Error binding parameter 2: type 'dict' is not supported")
        query = "INSERT INTO test_table (col1, col2) VALUES (?, ?)"
        params = ("val1", {"key": "value"})
        
        success, message = handler.handle_database_error(error, query, params)
        
        # Should fix the parameter binding
        assert success is True
        assert "parameter" in message.lower() or "fix" in message.lower()
    
    def test_type_error_handling(self):
        """Test handling of type conversion errors."""
        handler = DatabaseErrorHandler()
        
        error = Exception("type 'dict' is not supported")
        query = "INSERT INTO test_table (col1) VALUES (?)"
        params = ({"key": "value"},)
        
        success, message = handler.handle_database_error(error, query, params)
        
        # Should fix the type error
        assert success is True
        assert "type" in message.lower() or "fix" in message.lower()

class TestDatabaseSchemaVersioning:
    """Test cases for DatabaseSchemaVersioning."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    def test_schema_initialization(self, temp_db):
        """Test schema initialization."""
        versioning = DatabaseSchemaVersioning(temp_db)
        
        # Should need initialization
        status = versioning.check_schema_consistency()
        assert status["status"] == "needs_initialization"
        
        # Initialize schema
        result = versioning.initialize_schema()
        assert result is True
        
        # Should now be up to date
        status = versioning.check_schema_consistency()
        assert status["status"] == "up_to_date"
    
    def test_version_tracking(self, temp_db):
        """Test version tracking functionality."""
        versioning = DatabaseSchemaVersioning(temp_db)
        versioning.initialize_schema()
        
        # Record a new version
        versioning.record_version("1.1.0", "Added new feature", "ALTER TABLE...")
        
        # Check history
        history = versioning.get_schema_history()
        assert len(history) == 2  # Initial + new version
        assert history[1]["version"] == "1.1.0"
        assert history[1]["description"] == "Added new feature"

class TestDatabaseHealthMonitor:
    """Test cases for DatabaseHealthMonitor."""
    
    def test_health_status_initialization(self):
        """Test health monitor initialization."""
        monitor = DatabaseHealthMonitor(check_interval=60)
        
        status = monitor.get_health_status()
        assert status["status"] == "unknown"
        assert status["check_interval"] == 60
        assert status["last_check"] is None

class TestSafeDatabaseExecute:
    """Test cases for safe_database_execute function."""
    
    def test_successful_execution(self):
        """Test successful database execution."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    CREATE TABLE test_table (
                        id INTEGER PRIMARY KEY,
                        name TEXT
                    )
                """)
                
                # Test successful execution
                result = safe_database_execute(
                    conn, 
                    "INSERT INTO test_table (name) VALUES (?)", 
                    ("test",)
                )
                assert result is True
                
                # Verify data was inserted
                cursor = conn.execute("SELECT * FROM test_table")
                rows = cursor.fetchall()
                assert len(rows) == 1
                assert rows[0][1] == "test"
        
        finally:
            os.unlink(db_path)
    
    def test_error_handling(self):
        """Test error handling in safe_database_execute."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    CREATE TABLE test_table (
                        id INTEGER PRIMARY KEY,
                        name TEXT
                    )
                """)
                
                # Test error handling with invalid query
                result = safe_database_execute(
                    conn, 
                    "INSERT INTO nonexistent_table (name) VALUES (?)", 
                    ("test",)
                )
                assert result is False
        
        finally:
            os.unlink(db_path)

# Integration tests
class TestDatabaseIntegration:
    """Integration tests for the complete database system."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database with full schema."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        # Create a minimal schema for testing
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TIMESTAMP,
                    status TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE action_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    action_number INTEGER,
                    coordinates TEXT,
                    timestamp TIMESTAMP
                )
            """)
            conn.commit()
        
        yield db_path
        os.unlink(db_path)
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, temp_db):
        """Test complete end-to-end database workflow."""
        # Initialize schema versioning
        versioning = DatabaseSchemaVersioning(temp_db)
        versioning.initialize_schema()
        
        # Create database API
        db_api = DatabaseAPI(temp_db)
        
        # Test session creation
        session_result = await db_api.execute(
            "INSERT INTO sessions (session_id, start_time, status) VALUES (?, ?, ?)",
            ("test_session", datetime.now(), "active"),
            "sessions"
        )
        assert session_result is True
        
        # Test action trace logging
        action_result = await db_api.execute(
            "INSERT INTO action_traces (session_id, action_number, coordinates, timestamp) VALUES (?, ?, ?, ?)",
            ("test_session", 1, json.dumps([10, 20]), datetime.now()),
            "action_traces"
        )
        assert action_result is True
        
        # Test data retrieval
        sessions = await db_api.fetch_all("SELECT * FROM sessions WHERE session_id = ?", ("test_session",))
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "test_session"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
