"""
Error Logging Manager with Deduplication
Replaces file-based logging with database storage that prevents duplicate errors.
"""

import hashlib
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
import sqlite3
from pathlib import Path

class ErrorLoggingManager:
    """
    Manages error logging in the database with deduplication.
    Prevents the same error from being logged multiple times.
    """
    
    def __init__(self, db_path: str = "tabula_rasa.db"):
        self.db_path = db_path
        self._ensure_tables_exist()
    
    def _ensure_tables_exist(self):
        """Ensure the error_logs table exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS error_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    error_hash TEXT NOT NULL UNIQUE,
                    stack_trace TEXT,
                    context TEXT,
                    occurrence_count INTEGER DEFAULT 1,
                    first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def _generate_error_hash(self, error_type: str, error_message: str, stack_trace: str = "") -> str:
        """Generate a hash for error deduplication."""
        content = f"{error_type}:{error_message}:{stack_trace}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def log_error(self, 
                  error_type: str, 
                  error_message: str, 
                  stack_trace: str = "", 
                  context: Dict[str, Any] = None,
                  session_id: str = None,
                  game_id: str = None) -> Dict[str, Any]:
        """
        Log an error with deduplication.
        
        Args:
            error_type: Type of error (e.g., 'ValueError', 'KeyError')
            error_message: Error message
            stack_trace: Stack trace string
            context: Additional context data
            session_id: Current session ID
            game_id: Current game ID
            
        Returns:
            Dict with logging result and error info
        """
        error_hash = self._generate_error_hash(error_type, error_message, stack_trace)
        context_json = json.dumps(context or {})
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if error already exists
            cursor = conn.execute("""
                SELECT id, occurrence_count, first_seen, last_seen 
                FROM error_logs 
                WHERE error_hash = ?
            """, (error_hash,))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing error
                error_id, count, first_seen, last_seen = existing
                conn.execute("""
                    UPDATE error_logs 
                    SET occurrence_count = occurrence_count + 1,
                        last_seen = CURRENT_TIMESTAMP,
                        context = ?
                    WHERE id = ?
                """, (context_json, error_id))
                
                return {
                    'is_new': False,
                    'error_id': error_id,
                    'occurrence_count': count + 1,
                    'first_seen': first_seen,
                    'last_seen': datetime.now().isoformat(),
                    'message': f"Error logged (occurrence #{count + 1})"
                }
            else:
                # Insert new error
                cursor = conn.execute("""
                    INSERT INTO error_logs 
                    (error_type, error_message, error_hash, stack_trace, context, occurrence_count)
                    VALUES (?, ?, ?, ?, ?, 1)
                """, (error_type, error_message, error_hash, stack_trace, context_json))
                
                error_id = cursor.lastrowid
                conn.commit()
                
                return {
                    'is_new': True,
                    'error_id': error_id,
                    'occurrence_count': 1,
                    'first_seen': datetime.now().isoformat(),
                    'last_seen': datetime.now().isoformat(),
                    'message': "New error logged"
                }
    
    def log_exception(self, 
                     exception: Exception, 
                     context: Dict[str, Any] = None,
                     session_id: str = None,
                     game_id: str = None) -> Dict[str, Any]:
        """
        Log an exception with automatic type detection and stack trace.
        
        Args:
            exception: The exception object
            context: Additional context data
            session_id: Current session ID
            game_id: Current game ID
            
        Returns:
            Dict with logging result and error info
        """
        error_type = type(exception).__name__
        error_message = str(exception)
        stack_trace = traceback.format_exc()
        
        if context is None:
            context = {}
        
        if session_id:
            context['session_id'] = session_id
        if game_id:
            context['game_id'] = game_id
        
        return self.log_error(error_type, error_message, stack_trace, context, session_id, game_id)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get statistics about logged errors."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_errors,
                    COUNT(DISTINCT error_hash) as unique_errors,
                    SUM(occurrence_count) as total_occurrences,
                    AVG(occurrence_count) as avg_occurrences,
                    COUNT(CASE WHEN resolved = 1 THEN 1 END) as resolved_errors
                FROM error_logs
            """)
            
            stats = cursor.fetchone()
            
            # Get most frequent errors
            cursor = conn.execute("""
                SELECT error_type, error_message, occurrence_count, last_seen
                FROM error_logs
                ORDER BY occurrence_count DESC
                LIMIT 10
            """)
            
            frequent_errors = cursor.fetchall()
            
            return {
                'total_errors': stats[0] or 0,
                'unique_errors': stats[1] or 0,
                'total_occurrences': stats[2] or 0,
                'avg_occurrences': stats[3] or 0,
                'resolved_errors': stats[4] or 0,
                'frequent_errors': [
                    {
                        'error_type': row[0],
                        'error_message': row[1],
                        'occurrence_count': row[2],
                        'last_seen': row[3]
                    }
                    for row in frequent_errors
                ]
            }
    
    def get_recent_errors(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent errors."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, error_type, error_message, occurrence_count, 
                       first_seen, last_seen, resolved, context
                FROM error_logs
                ORDER BY last_seen DESC
                LIMIT ?
            """, (limit,))
            
            return [
                {
                    'id': row[0],
                    'error_type': row[1],
                    'error_message': row[2],
                    'occurrence_count': row[3],
                    'first_seen': row[4],
                    'last_seen': row[5],
                    'resolved': bool(row[6]),
                    'context': json.loads(row[7]) if row[7] else {}
                }
                for row in cursor.fetchall()
            ]
    
    def mark_error_resolved(self, error_id: int) -> bool:
        """Mark an error as resolved."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE error_logs 
                SET resolved = 1 
                WHERE id = ?
            """, (error_id,))
            
            return cursor.rowcount > 0
    
    def cleanup_old_errors(self, days_old: int = 30) -> int:
        """Remove old resolved errors."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM error_logs 
                WHERE resolved = 1 
                AND last_seen < datetime('now', '-{} days')
            """.format(days_old))
            
            return cursor.rowcount

# Global instance
_error_logging_manager = None

def get_error_logging_manager() -> ErrorLoggingManager:
    """Get the global error logging manager instance."""
    global _error_logging_manager
    if _error_logging_manager is None:
        _error_logging_manager = ErrorLoggingManager()
    return _error_logging_manager

def log_error(error_type: str, error_message: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to log an error."""
    return get_error_logging_manager().log_error(error_type, error_message, **kwargs)

def log_exception(exception: Exception, **kwargs) -> Dict[str, Any]:
    """Convenience function to log an exception."""
    return get_error_logging_manager().log_exception(exception, **kwargs)
