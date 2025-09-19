"""
Performance Data Manager for storing and retrieving performance data in the database.

This module replaces the growing in-memory data structures with database storage
to prevent memory leaks and provide persistent storage.
"""

import asyncio
import json
import sqlite3
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class PerformanceDataManager:
    """Manages performance data storage in the database."""
    
    def __init__(self, db_path: str = "tabula_rasa.db"):
        self.db_path = db_path
        self._ensure_schema()
    
    def _convert_sets_to_lists(self, obj):
        """Convert sets to lists recursively for JSON serialization."""
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_sets_to_lists(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_sets_to_lists(item) for item in obj]
        else:
            return obj
    
    def _ensure_schema(self):
        """Ensure the performance schema is created."""
        try:
            schema_path = Path(__file__).parent / "performance_schema.sql"
            if schema_path.exists():
                with open(schema_path, 'r', encoding='utf-8') as f:
                    schema_sql = f.read()
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.executescript(schema_sql)
                logger.info("Performance schema ensured")
        except Exception as e:
            logger.error(f"Failed to ensure performance schema: {e}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    async def store_performance_data(self, session_id: str, performance_data: Dict[str, Any]) -> bool:
        """Store performance data in the database."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO performance_history 
                    (session_id, game_id, score, win_rate, learning_efficiency, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    performance_data.get('game_id'),
                    performance_data.get('score'),
                    performance_data.get('win_rate'),
                    performance_data.get('learning_efficiency'),
                    json.dumps(performance_data.get('metadata', {}))
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store performance data: {e}")
            return False
    
    async def get_performance_history(self, session_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get performance history from the database."""
        try:
            with self._get_connection() as conn:
                if session_id:
                    cursor = conn.execute("""
                        SELECT * FROM performance_history 
                        WHERE session_id = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (session_id, limit))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM performance_history 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (limit,))
                
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    if result['metadata']:
                        result['metadata'] = json.loads(result['metadata'])
                    results.append(result)
                return results
        except Exception as e:
            logger.error(f"Failed to get performance history: {e}")
            return []
    
    async def store_session_data(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """Store session data in the database."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO session_history 
                    (session_id, game_id, status, duration_seconds, actions_taken, score, win, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    session_data.get('game_id'),
                    session_data.get('status'),
                    session_data.get('duration_seconds'),
                    session_data.get('actions_taken'),
                    session_data.get('score'),
                    session_data.get('win'),
                    json.dumps(session_data.get('metadata', {}))
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store session data: {e}")
            return False
    
    async def get_session_history(self, session_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get session history from the database."""
        try:
            with self._get_connection() as conn:
                if session_id:
                    cursor = conn.execute("""
                        SELECT * FROM session_history 
                        WHERE session_id = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (session_id, limit))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM session_history 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (limit,))
                
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    if result['metadata']:
                        result['metadata'] = json.loads(result['metadata'])
                    results.append(result)
                return results
        except Exception as e:
            logger.error(f"Failed to get session history: {e}")
            return []
    
    async def store_action_tracking(self, game_id: str, action_data: Dict[str, Any]) -> bool:
        """Store action tracking data in the database."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO action_tracking 
                    (game_id, action_type, action_sequence, effectiveness, context)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    game_id,
                    action_data.get('action_type'),
                    json.dumps(action_data.get('action_sequence', [])),
                    action_data.get('effectiveness'),
                    json.dumps(action_data.get('context', {}))
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store action tracking: {e}")
            return False
    
    async def get_action_tracking(self, game_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get action tracking data from the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM action_tracking 
                    WHERE game_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (game_id, limit))
                
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    if result['action_sequence']:
                        result['action_sequence'] = json.loads(result['action_sequence'])
                    if result['context']:
                        result['context'] = json.loads(result['context'])
                    results.append(result)
                return results
        except Exception as e:
            logger.error(f"Failed to get action tracking: {e}")
            return []
    
    async def store_score_data(self, game_id: str, score_data: Dict[str, Any]) -> bool:
        """Store score data in the database."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO score_history 
                    (game_id, session_id, score, score_type)
                    VALUES (?, ?, ?, ?)
                """, (
                    game_id,
                    score_data.get('session_id'),
                    score_data.get('score'),
                    score_data.get('score_type', 'current')
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store score data: {e}")
            return False
    
    async def get_score_history(self, game_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get score history from the database (replaces the growing score_history lists)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM score_history 
                    WHERE game_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (game_id, limit))
                
                results = []
                for row in cursor.fetchall():
                    results.append(dict(row))
                return results
        except Exception as e:
            logger.error(f"Failed to get score history: {e}")
            return []
    
    async def store_coordinate_tracking(self, game_id: str, coordinate_data: Dict[str, Any]) -> bool:
        """Store coordinate tracking data in the database."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO coordinate_tracking 
                    (game_id, coordinate_x, coordinate_y, action_type, success, context)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    game_id,
                    coordinate_data.get('coordinate_x'),
                    coordinate_data.get('coordinate_y'),
                    coordinate_data.get('action_type'),
                    coordinate_data.get('success'),
                    json.dumps(self._convert_sets_to_lists(coordinate_data.get('context', {})))
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store coordinate tracking: {e}")
            return False
    
    async def get_coordinate_tracking(self, game_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Get coordinate tracking data from the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM coordinate_tracking 
                    WHERE game_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (game_id, limit))
                
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    if result['context']:
                        result['context'] = json.loads(result['context'])
                    results.append(result)
                return results
        except Exception as e:
            logger.error(f"Failed to get coordinate tracking: {e}")
            return []
    
    async def store_frame_tracking(self, game_id: str, frame_data: Dict[str, Any]) -> bool:
        """Store frame tracking data in the database."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO frame_tracking 
                    (game_id, frame_hash, frame_analysis, stagnation_detected)
                    VALUES (?, ?, ?, ?)
                """, (
                    game_id,
                    frame_data.get('frame_hash'),
                    json.dumps(frame_data.get('frame_analysis', {})),
                    frame_data.get('stagnation_detected', False)
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store frame tracking: {e}")
            return False
    
    async def get_frame_tracking(self, game_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get frame tracking data from the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM frame_tracking 
                    WHERE game_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (game_id, limit))
                
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    if result['frame_analysis']:
                        result['frame_analysis'] = json.loads(result['frame_analysis'])
                    results.append(result)
                return results
        except Exception as e:
            logger.error(f"Failed to get frame tracking: {e}")
            return []
    
    async def cleanup_old_data(self, days: int = 30) -> bool:
        """Clean up old data from the database."""
        try:
            with self._get_connection() as conn:
                cutoff_date = datetime.now() - timedelta(days=days)
                cutoff_str = cutoff_date.isoformat()
                
                # Clean up old data from all tables
                tables = [
                    'performance_history',
                    'session_history', 
                    'action_tracking',
                    'score_history',
                    'coordinate_tracking',
                    'frame_tracking'
                ]
                
                for table in tables:
                    conn.execute(f"DELETE FROM {table} WHERE created_at < ?", (cutoff_str,))
                
                conn.commit()
                logger.info(f"Cleaned up data older than {days} days")
                return True
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored data."""
        try:
            with self._get_connection() as conn:
                stats = {}
                
                tables = [
                    'performance_history',
                    'session_history',
                    'action_tracking', 
                    'score_history',
                    'coordinate_tracking',
                    'frame_tracking'
                ]
                
                for table in tables:
                    cursor = conn.execute(f"SELECT COUNT(*) as count FROM {table}")
                    count = cursor.fetchone()['count']
                    stats[table] = count
                
                return stats
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}


# Global instance for easy access
_performance_manager = None

def get_performance_manager() -> PerformanceDataManager:
    """Get the global performance data manager instance."""
    global _performance_manager
    if _performance_manager is None:
        _performance_manager = PerformanceDataManager()
    return _performance_manager
