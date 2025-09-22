"""
TABULA RASA DATABASE API
Comprehensive Python API for Director/LLM, Architect, and Governor integration
"""

import sqlite3
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager
import threading
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# DATA MODELS
# ============================================================================

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class Component(Enum):
    DIRECTOR = "director"
    GOVERNOR = "governor"
    ARCHITECT = "architect"
    LEARNING_LOOP = "learning_loop"
    COORDINATE_SYSTEM = "coordinate_system"
    MEMORY_SYSTEM = "memory_system"
    EXPERIMENT = "experiment"
    TASK_PERFORMANCE = "task_performance"
    SUBSYSTEM_MONITOR = "subsystem_monitor"
    FRAME_ANALYSIS = "frame_analysis"
    GAME_RESULT = "game_result"

@dataclass
class TrainingSession:
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    mode: str = "maximum-intelligence"
    status: str = "running"
    total_actions: int = 0
    total_wins: int = 0
    total_games: int = 0
    win_rate: float = 0.0
    avg_score: float = 0.0
    energy_level: float = 100.0
    memory_operations: int = 0
    sleep_cycles: int = 0

@dataclass
class GameResult:
    game_id: str
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    final_score: float = 0.0
    total_actions: int = 0
    actions_taken: List[int] = None
    win_detected: bool = False
    level_completions: int = 0
    frame_changes: int = 0
    coordinate_attempts: int = 0
    coordinate_successes: int = 0

@dataclass
class ActionEffectiveness:
    game_id: str
    action_number: int
    attempts: int = 0
    successes: int = 0
    success_rate: float = 0.0
    avg_score_impact: float = 0.0
    last_used: Optional[datetime] = None

@dataclass
class CoordinateIntelligence:
    game_id: str
    x: int
    y: int
    attempts: int = 0
    successes: int = 0
    success_rate: float = 0.0
    frame_changes: int = 0
    last_used: Optional[datetime] = None

# ============================================================================
# DATABASE API CLASS
# ============================================================================

class TabulaRasaDatabase:
    """
    Main database API for Tabula Rasa system.
    Provides high-level interface for all system components.
    """
    
    def __init__(self, db_path: str = "tabula_rasa.db"):
        # Ensure database is always created in project root, not relative to current working directory
        if not os.path.isabs(db_path):
            # Find project root by looking for common project files
            current_dir = Path(__file__).parent
            project_root = current_dir
            while project_root.parent != project_root:
                if (project_root / "README.md").exists() or (project_root / "requirements.txt").exists():
                    break
                project_root = project_root.parent
            
            # Use absolute path to project root
            self.db_path = project_root / db_path
        else:
            self.db_path = Path(db_path)
        
        # Only create parent directory if it's not the current directory
        if self.db_path.parent.name:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.RLock()
        self._initialize_database()
        self.logger = logging.getLogger(__name__)
    
    def _initialize_database(self):
        """Initialize database with schema."""
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(schema_sql)
            conn.commit()
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    # ============================================================================
    # SESSION MANAGEMENT
    # ============================================================================
    
    async def create_session(self, session: TrainingSession) -> bool:
        """Create a new training session."""
        async with self.get_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO training_sessions 
                    (session_id, start_time, end_time, mode, status, total_actions, 
                     total_wins, total_games, win_rate, avg_score, energy_level, 
                     memory_operations, sleep_cycles)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id, session.start_time, session.end_time,
                    session.mode, session.status, session.total_actions,
                    session.total_wins, session.total_games, session.win_rate,
                    session.avg_score, session.energy_level, session.memory_operations,
                    session.sleep_cycles
                ))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update training session data."""
        if not updates:
            return True
        
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        updates["updated_at"] = datetime.now()
        updates["session_id"] = session_id
        
        async with self.get_connection() as conn:
            try:
                conn.execute(f"""
                    UPDATE training_sessions 
                    SET {set_clause}, updated_at = ?
                    WHERE session_id = ?
                """, list(updates.values()) + [session_id])
                conn.commit()
                return True
            except Exception as e:
                self.logger.error(f"Failed to update session {session_id}: {e}")
                return False
    
    async def get_session(self, session_id: str) -> Optional[TrainingSession]:
        """Get training session by ID."""
        async with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM training_sessions WHERE session_id = ?
            """, (session_id,))
            row = cursor.fetchone()
            if row:
                return TrainingSession(**{k: v for k, v in dict(row).items() if k not in ['game_id', 'created_at', 'updated_at']})
            return None
    
    async def get_active_sessions(self) -> List[TrainingSession]:
        """Get all active training sessions."""
        async with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM training_sessions 
                WHERE status = 'running'
                ORDER BY start_time DESC
            """)
            return [TrainingSession(**{k: v for k, v in dict(row).items() if k not in ['game_id', 'created_at', 'updated_at']}) for row in cursor.fetchall()]
    
    # ============================================================================
    # GAME RESULTS MANAGEMENT
    # ============================================================================
    
    async def create_game_result(self, game_result: GameResult) -> bool:
        """Create a new game result."""
        async with self.get_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO game_results 
                    (game_id, session_id, start_time, end_time, status, final_score,
                     total_actions, actions_taken, win_detected, level_completions,
                     frame_changes, coordinate_attempts, coordinate_successes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    game_result.game_id, game_result.session_id, game_result.start_time,
                    game_result.end_time, game_result.status, game_result.final_score,
                    game_result.total_actions, json.dumps(game_result.actions_taken or []),
                    game_result.win_detected, game_result.level_completions,
                    game_result.frame_changes, game_result.coordinate_attempts,
                    game_result.coordinate_successes
                ))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False
    
    async def update_game_result(self, game_id: str, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update game result data."""
        if not updates:
            return True
        
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        updates["game_id"] = game_id
        updates["session_id"] = session_id
        
        async with self.get_connection() as conn:
            try:
                conn.execute(f"""
                    UPDATE game_results 
                    SET {set_clause}
                    WHERE game_id = ? AND session_id = ?
                """, list(updates.values()) + [game_id, session_id])
                conn.commit()
                return True
            except Exception as e:
                self.logger.error(f"Failed to update game result {game_id}: {e}")
                return False
    
    async def get_game_results(self, session_id: str = None, game_id: str = None) -> List[GameResult]:
        """Get game results with optional filtering."""
        query = """SELECT game_id, session_id, start_time, end_time, status, final_score, 
                   total_actions, actions_taken, win_detected, level_completions, 
                   frame_changes, coordinate_attempts, coordinate_successes 
                   FROM game_results WHERE 1=1"""
        params = []
        
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        
        if game_id:
            query += " AND game_id = ?"
            params.append(game_id)
        
        query += " ORDER BY start_time DESC"
        
        async with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            results = []
            for row in cursor.fetchall():
                data = dict(row)
                data["actions_taken"] = json.loads(data["actions_taken"] or "[]")
                results.append(GameResult(**data))
            return results
    
    # ============================================================================
    # ACTION INTELLIGENCE MANAGEMENT
    # ============================================================================
    
    async def update_action_effectiveness(self, effectiveness: ActionEffectiveness) -> bool:
        """Update action effectiveness data."""
        async with self.get_connection() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO action_effectiveness
                    (game_id, action_number, attempts, successes, success_rate,
                     avg_score_impact, last_used, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    effectiveness.game_id, effectiveness.action_number,
                    effectiveness.attempts, effectiveness.successes,
                    effectiveness.success_rate, effectiveness.avg_score_impact,
                    effectiveness.last_used
                ))
                conn.commit()
                return True
            except Exception as e:
                self.logger.error(f"Failed to update action effectiveness: {e}")
                return False
    
    async def get_action_effectiveness(self, game_id: str = None, action_number: int = None) -> List[ActionEffectiveness]:
        """Get action effectiveness data."""
        query = "SELECT * FROM action_effectiveness WHERE 1=1"
        params = []
        
        if game_id:
            query += " AND game_id = ?"
            params.append(game_id)
        
        if action_number:
            query += " AND action_number = ?"
            params.append(action_number)
        
        query += " ORDER BY success_rate DESC"
        
        async with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                # Remove 'id' field that's not in the dataclass
                row_dict.pop('id', None)
                row_dict.pop('created_at', None)
                row_dict.pop('updated_at', None)
                results.append(ActionEffectiveness(**row_dict))
            return results
    
    async def update_coordinate_intelligence(self, intelligence: CoordinateIntelligence) -> bool:
        """Update coordinate intelligence data."""
        async with self.get_connection() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO coordinate_intelligence
                    (game_id, x, y, attempts, successes, success_rate,
                     frame_changes, last_used, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    intelligence.game_id, intelligence.x, intelligence.y,
                    intelligence.attempts, intelligence.successes,
                    intelligence.success_rate, intelligence.frame_changes,
                    intelligence.last_used
                ))
                conn.commit()
                return True
            except Exception as e:
                self.logger.error(f"Failed to update coordinate intelligence: {e}")
                return False
    
    async def get_coordinate_intelligence(self, game_id: str = None, min_success_rate: float = 0.0) -> List[CoordinateIntelligence]:
        """Get coordinate intelligence data."""
        query = "SELECT * FROM coordinate_intelligence WHERE success_rate >= ?"
        params = [min_success_rate]
        
        if game_id:
            query += " AND game_id = ?"
            params.append(game_id)
        
        query += " ORDER BY success_rate DESC, attempts DESC"
        
        async with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                # Remove 'id' field that's not in the dataclass
                row_dict.pop('id', None)
                row_dict.pop('created_at', None)
                row_dict.pop('updated_at', None)
                results.append(CoordinateIntelligence(**row_dict))
            return results
    
    # ============================================================================
    # LOGGING AND TRACING
    # ============================================================================
    
    async def log_system_event(self, level: LogLevel, component: Component, message: str, 
                              data: Dict[str, Any] = None, session_id: str = None, 
                              game_id: str = None) -> bool:
        """Log system event."""
        async with self.get_connection() as conn:
            try:
                # Custom JSON encoder to handle datetime objects
                def json_serializer(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
                
                conn.execute("""
                    INSERT INTO system_logs
                    (log_level, component, message, data, session_id, game_id, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    level.value, component.value, message,
                    json.dumps(data, default=json_serializer) if data else None,
                    session_id, game_id, datetime.now()
                ))
                conn.commit()
                return True
            except Exception as e:
                self.logger.error(f"Failed to log system event: {e}")
                return False
    
    async def log_action_trace(self, session_id: str, game_id: str, action_number: int,
                              coordinates: Tuple[int, int] = None, frame_before: List = None,
                              frame_after: List = None, frame_changed: bool = False,
                              score_before: float = 0.0, score_after: float = 0.0,
                              response_data: Dict[str, Any] = None) -> bool:
        """Log detailed action trace."""
        async with self.get_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO action_traces
                    (session_id, game_id, action_number, coordinates, timestamp,
                     frame_before, frame_after, frame_changed, score_before,
                     score_after, score_change, response_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, game_id, action_number,
                    json.dumps(coordinates) if coordinates else None,
                    datetime.now(),
                    json.dumps(frame_before) if frame_before else None,
                    json.dumps(frame_after) if frame_after else None,
                    frame_changed, score_before, score_after,
                    score_after - score_before,
                    json.dumps(response_data) if response_data else None
                ))
                conn.commit()
                return True
            except Exception as e:
                self.logger.error(f"Failed to log action trace: {e}")
                return False
    
    # ============================================================================
    # REAL-TIME QUERIES FOR DIRECTOR
    # ============================================================================
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get real-time system status."""
        async with self.get_connection() as conn:
            # Get active sessions
            cursor = conn.execute("SELECT * FROM system_status")
            active_sessions = [dict(row) for row in cursor.fetchall()]
            
            # Get recent performance
            cursor = conn.execute("SELECT * FROM recent_performance LIMIT 7")
            recent_performance = [dict(row) for row in cursor.fetchall()]
            
            # Get action effectiveness summary
            cursor = conn.execute("SELECT * FROM action_effectiveness_summary")
            action_summary = [dict(row) for row in cursor.fetchall()]
            
            # Get global counters
            cursor = conn.execute("SELECT * FROM global_counters")
            global_counters = {row["counter_name"]: row["counter_value"] for row in cursor.fetchall()}
            
            return {
                "active_sessions": active_sessions,
                "recent_performance": recent_performance,
                "action_effectiveness": action_summary,
                "global_counters": global_counters,
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_learning_insights(self, game_id: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get learning insights for Director analysis."""
        async with self.get_connection() as conn:
            # Get coordinate intelligence
            cursor = conn.execute("""
                SELECT * FROM coordinate_intelligence_summary
                WHERE game_id = ? OR ? IS NULL
                ORDER BY avg_success_rate DESC
                LIMIT 20
            """, (game_id, game_id))
            coordinate_insights = [dict(row) for row in cursor.fetchall()]
            
            # Get winning sequences from learned_patterns (replacement for deleted winning_sequences table)
            cursor = conn.execute("""
                SELECT pattern_data, success_rate, frequency
                FROM learned_patterns
                WHERE pattern_type = 'winning_sequence' AND (game_context = ? OR ? IS NULL)
                ORDER BY success_rate DESC, frequency DESC
                LIMIT 10
            """, (game_id, game_id))
            winning_sequences = [dict(row) for row in cursor.fetchall()]
            
            # Get recent patterns
            cursor = conn.execute("""
                SELECT * FROM learned_patterns
                WHERE created_at >= datetime('now', '-{} hours')
                ORDER BY confidence DESC, success_rate DESC
                LIMIT 20
            """.format(hours))
            recent_patterns = [dict(row) for row in cursor.fetchall()]
            
            return {
                "coordinate_insights": coordinate_insights,
                "winning_sequences": winning_sequences,
                "recent_patterns": recent_patterns,
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    async def get_training_sessions(self, session_id: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get performance metrics for analysis."""
        async with self.get_connection() as conn:
            # Get session performance
            if session_id:
                cursor = conn.execute("""
                    SELECT * FROM training_sessions WHERE session_id = ?
                """, (session_id,))
                session_data = dict(cursor.fetchone()) if cursor.fetchone() else None
            else:
                cursor = conn.execute("""
                    SELECT * FROM training_sessions
                    WHERE start_time >= datetime('now', '-{} hours')
                    ORDER BY start_time DESC
                """.format(hours))
                session_data = [dict(row) for row in cursor.fetchall()]
            
            # Get game results
            cursor = conn.execute("""
                SELECT * FROM game_results
                WHERE start_time >= datetime('now', '-{} hours')
                ORDER BY start_time DESC
            """.format(hours))
            game_results = [dict(row) for row in cursor.fetchall()]
            
            # Calculate metrics
            total_sessions = len(session_data) if isinstance(session_data, list) else 1
            total_games = len(game_results)
            total_wins = sum(gr["win_detected"] for gr in game_results)
            win_rate = total_wins / total_games if total_games > 0 else 0.0
            
            return {
                "session_data": session_data,
                "game_results": game_results,
                "metrics": {
                    "total_sessions": total_sessions,
                    "total_games": total_games,
                    "total_wins": total_wins,
                    "win_rate": win_rate,
                    "analysis_timestamp": datetime.now().isoformat()
                }
            }

    async def execute(self, query: str, params: tuple = ()) -> bool:
        """Execute a SQL query."""
        try:
            async with self.get_connection() as conn:
                conn.execute(query, params)
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"Database execute error: {e}")
            return False
    
    async def fetch_all(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Fetch all rows from a query."""
        try:
            async with self.get_connection() as conn:
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Database fetch_all error: {e}")
            return []
    
    async def fetch_one(self, query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """Fetch one row from a query."""
        try:
            async with self.get_connection() as conn:
                cursor = conn.execute(query, params)
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            self.logger.error(f"Database fetch_one error: {e}")
            return None

# ============================================================================
# GLOBAL DATABASE INSTANCE

# Global database instance for easy access
_db_instance = None

def get_database() -> TabulaRasaDatabase:
    """Get global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = TabulaRasaDatabase()
    return _db_instance

# ============================================================================
# CONVENIENCE FUNCTIONS FOR DIRECTOR
# ============================================================================

async def log_director_decision(decision: str, rationale: str, confidence: float, 
                               session_id: str = None, data: Dict[str, Any] = None):
    """Log Director decision."""
    db = get_database()
    data_dict = {"rationale": rationale, "confidence": confidence}
    if data:
        data_dict.update(data)
    
    await db.log_system_event(
        LogLevel.INFO, Component.DIRECTOR, f"Director Decision: {decision}",
        data_dict, session_id
    )

async def get_director_status() -> Dict[str, Any]:
    """Get status for Director analysis."""
    db = get_database()
    return await db.get_system_status()

async def get_director_insights(game_id: str = None) -> Dict[str, Any]:
    """Get learning insights for Director."""
    db = get_database()
    return await db.get_learning_insights(game_id)

async def update_global_counter(counter_name: str, value: int, description: str = None):
    """Update global counter."""
    db = get_database()
    async with db.get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO global_counters
            (counter_name, counter_value, description, last_updated)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (counter_name, value, description))
        conn.commit()
