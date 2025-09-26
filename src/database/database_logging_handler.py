"""
Database Logging Handler
Replaces file-based logging with database storage.
"""

import logging
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
import sqlite3
from pathlib import Path

class DatabaseLoggingHandler(logging.Handler):
    """
    Custom logging handler that stores logs in the database.
    Replaces file-based logging.
    """
    
    def __init__(self, db_path: str = "tabula_rasa.db", level=logging.NOTSET):
        super().__init__(level)
        self.db_path = db_path
        self._ensure_tables_exist()
    
    def _ensure_tables_exist(self):
        """Ensure the system_logs table exists with correct schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Use the existing schema that matches the actual table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    log_level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    data TEXT,
                    session_id TEXT,
                    game_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def emit(self, record):
        """Emit a log record to the database."""
        try:
            # Extract information from the log record
            component_name = record.name if record.name else 'unknown'

            # Create metadata with additional info
            metadata = {
                'module': getattr(record, 'module', 'unknown'),
                'function': getattr(record, 'funcName', 'unknown'),
                'line_number': getattr(record, 'lineno', 0)
            }

            # Add exception info if present
            if record.exc_info:
                metadata['exception'] = {
                    'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                    'message': str(record.exc_info[1]) if record.exc_info[1] else None
                }

            # Insert into database using the existing schema
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO system_logs
                    (log_level, component, message, data, session_id, game_id, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.levelname,
                    component_name,
                    record.getMessage(),
                    json.dumps(metadata),
                    getattr(record, 'session_id', None),
                    getattr(record, 'game_id', None),
                    datetime.fromtimestamp(record.created).isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            # Fallback to console if database logging fails
            print(f"Database logging failed: {e}")
            print(f"Original log: {record.getMessage()}")
    
    def _extract_metadata(self, record) -> str:
        """Extract metadata from log record."""
        metadata = {}
        
        # Add exception info if present
        if record.exc_info:
            metadata['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add any custom attributes
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info', 'session_id', 'game_id']:
                try:
                    # Only include JSON-serializable values
                    json.dumps(value)
                    metadata[key] = value
                except (TypeError, ValueError):
                    metadata[key] = str(value)
        
        return json.dumps(metadata) if metadata else None

class DatabaseLoggingManager:
    """
    Manages database logging configuration and utilities.
    """
    
    def __init__(self, db_path: str = "tabula_rasa.db"):
        self.db_path = db_path
        self.handler = DatabaseLoggingHandler(db_path)
    
    def setup_logging(self, 
                     logger_name: str = "tabula_rasa",
                     level: int = logging.INFO,
                     session_id: str = None,
                     game_id: str = None) -> logging.Logger:
        """
        Set up a logger with database logging.
        
        Args:
            logger_name: Name of the logger
            level: Logging level
            session_id: Current session ID
            game_id: Current game ID
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add database handler
        logger.addHandler(self.handler)
        
        # Add session and game context to all log records
        if session_id or game_id:
            old_factory = logging.getLogRecordFactory()
            
            def record_factory(*args, **kwargs):
                record = old_factory(*args, **kwargs)
                if session_id:
                    record.session_id = session_id
                if game_id:
                    record.game_id = game_id
                return record
            
            logging.setLogRecordFactory(record_factory)
        
        return logger
    
    def get_logs(self,
                level: str = None,
                component: str = None,
                session_id: str = None,
                game_id: str = None,
                limit: int = 1000,
                offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get logs from the database with optional filtering.

        Args:
            level: Filter by log level
            component: Filter by component name
            session_id: Filter by session ID
            game_id: Filter by game ID
            limit: Maximum number of logs to return
            offset: Number of logs to skip

        Returns:
            List of log dictionaries
        """
        query = "SELECT * FROM system_logs WHERE 1=1"
        params = []

        if level:
            query += " AND log_level = ?"
            params.append(level)

        if component:
            query += " AND component = ?"
            params.append(component)

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        if game_id:
            query += " AND game_id = ?"
            params.append(game_id)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)

            logs = []
            for row in cursor.fetchall():
                logs.append({
                    'id': row[0],
                    'log_level': row[1],
                    'component': row[2],
                    'message': row[3],
                    'data': json.loads(row[4]) if row[4] else {},
                    'session_id': row[5],
                    'game_id': row[6],
                    'timestamp': row[7],
                    'created_at': row[8] if len(row) > 8 else None
                })

            return logs
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get statistics about logged messages."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_logs,
                    COUNT(DISTINCT log_level) as log_levels,
                    COUNT(DISTINCT component) as components,
                    COUNT(DISTINCT session_id) as sessions,
                    COUNT(DISTINCT game_id) as games
                FROM system_logs
            """)
            
            stats = cursor.fetchone()
            
            # Get log level distribution
            cursor = conn.execute("""
                SELECT log_level, COUNT(*) as count
                FROM system_logs
                GROUP BY log_level
                ORDER BY count DESC
            """)
            
            level_distribution = [
                {'level': row[0], 'count': row[1]}
                for row in cursor.fetchall()
            ]
            
            return {
                'total_logs': stats[0] or 0,
                'log_levels': stats[1] or 0,
                'components': stats[2] or 0,
                'sessions': stats[3] or 0,
                'games': stats[4] or 0,
                'level_distribution': level_distribution
            }
    
    def cleanup_old_logs(self, days_old: int = 30) -> int:
        """Remove old log entries."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM system_logs 
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days_old))
            
            return cursor.rowcount

# Global instance
_database_logging_manager = None

def get_database_logging_manager() -> DatabaseLoggingManager:
    """Get the global database logging manager instance."""
    global _database_logging_manager
    if _database_logging_manager is None:
        _database_logging_manager = DatabaseLoggingManager()
    return _database_logging_manager

def setup_database_logging(logger_name: str = "tabula_rasa", **kwargs) -> logging.Logger:
    """Convenience function to set up database logging."""
    return get_database_logging_manager().setup_logging(logger_name, **kwargs)
