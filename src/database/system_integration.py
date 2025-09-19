"""
SYSTEM INTEGRATION LAYER
Replaces file I/O operations with database API calls
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from .api import get_database, TrainingSession, GameResult, ActionEffectiveness, CoordinateIntelligence
from dataclasses import asdict
from .director_commands import get_director_commands

class SystemIntegration:
    """
    Integration layer that replaces file-based operations with database calls.
    Provides backward-compatible interfaces for existing systems.
    """
    
    def __init__(self):
        self.db = get_database()
        self.director_commands = get_director_commands()
        self.logger = logging.getLogger(__name__)
    
    # ============================================================================
    # SESSION MANAGEMENT INTEGRATION
    # ============================================================================
    
    async def create_training_session(self, session_id: str, mode: str = "maximum-intelligence") -> bool:
        """Create a new training session."""
        session = TrainingSession(
            session_id=session_id,
            start_time=datetime.now(),
            mode=mode,
            status="running"
        )
        
        success = await self.db.create_session(session)
        if success:
            await self.db.log_system_event(
                "INFO", "learning_loop", f"Created training session {session_id}",
                {"mode": mode}, session_id
            )
        
        return success
    
    async def update_session_metrics(self, session_id: str, metrics: Dict[str, Any]) -> bool:
        """Update session metrics."""
        # First try to create a new session, then update it
        from datetime import datetime
        
        # Create a TrainingSession object from the metrics
        session = TrainingSession(
            session_id=session_id,
            start_time=metrics.get('start_time', datetime.now()),
            mode=metrics.get('mode', 'unknown'),
            status=metrics.get('status', 'completed'),
            total_actions=metrics.get('total_actions', 0),
            total_wins=metrics.get('total_wins', 0),
            total_games=metrics.get('total_games', 1),
            win_rate=metrics.get('win_rate', 0.0),
            avg_score=metrics.get('avg_score', 0.0),
            energy_level=metrics.get('energy_level', 100.0),
            memory_operations=metrics.get('memory_operations', 0),
            sleep_cycles=metrics.get('sleep_cycles', 0)
        )
        
        # Try to create the session (this will insert or replace)
        return await self.db.create_session(session)
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session status."""
        session = await self.db.get_session(session_id)
        return asdict(session) if session else None
    
    # ============================================================================
    # GAME RESULTS INTEGRATION
    # ============================================================================
    
    async def log_game_result(self, game_id: str, session_id: str, 
                             result_data: Dict[str, Any]) -> bool:
        """Log game result."""
        game_result = GameResult(
            game_id=game_id,
            session_id=session_id,
            start_time=result_data.get("start_time", datetime.now()),
            end_time=result_data.get("end_time"),
            status=result_data.get("status", "completed"),
            final_score=result_data.get("final_score", 0.0),
            total_actions=result_data.get("total_actions", 0),
            actions_taken=result_data.get("actions_taken", []),
            win_detected=result_data.get("win_detected", False),
            level_completions=result_data.get("level_completions", 0),
            frame_changes=result_data.get("frame_changes", 0),
            coordinate_attempts=result_data.get("coordinate_attempts", 0),
            coordinate_successes=result_data.get("coordinate_successes", 0)
        )
        
        success = await self.db.create_game_result(game_result)
        if success:
            await self.db.log_system_event(
                "INFO", "learning_loop", f"Logged game result for {game_id}",
                result_data, session_id, game_id
            )
        
        return success
    
    async def get_game_results(self, session_id: str = None, game_id: str = None) -> List[Dict[str, Any]]:
        """Get game results."""
        results = await self.db.get_game_results(session_id, game_id)
        return results  # Return GameResult objects directly
    
    # ============================================================================
    # ACTION INTELLIGENCE INTEGRATION
    # ============================================================================
    
    async def update_action_effectiveness(self, game_id: str, action_number: int,
                                        attempts: int, successes: int,
                                        success_rate: float = None,
                                        avg_score_impact: float = 0.0) -> bool:
        """Update action effectiveness."""
        if success_rate is None:
            success_rate = successes / attempts if attempts > 0 else 0.0
        
        effectiveness = ActionEffectiveness(
            game_id=game_id,
            action_number=action_number,
            attempts=attempts,
            successes=successes,
            success_rate=success_rate,
            avg_score_impact=avg_score_impact,
            last_used=datetime.now()
        )
        
        success = await self.db.update_action_effectiveness(effectiveness)
        if success:
            await self.db.log_system_event(
                "DEBUG", "learning_loop", f"Updated action {action_number} effectiveness",
                {"game_id": game_id, "success_rate": success_rate}, None, game_id
            )
        
        return success
    
    async def get_action_effectiveness(self, game_id: str = None, action_number: int = None) -> List[Dict[str, Any]]:
        """Get action effectiveness data."""
        effectiveness = await self.db.get_action_effectiveness(game_id, action_number)
        return [asdict(e) for e in effectiveness]
    
    async def get_best_actions(self, game_id: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Get best performing actions."""
        effectiveness = await self.db.get_action_effectiveness(game_id)
        sorted_actions = sorted(effectiveness, key=lambda x: x.success_rate, reverse=True)
        return [asdict(action) for action in sorted_actions[:limit]]
    
    def get_action_effectiveness_by_prefix(self, game_prefix: str) -> List[Dict[str, Any]]:
        """Get action effectiveness data for all games with the given prefix."""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.execute("""
                SELECT game_id, action_number, attempts, successes, success_rate, 
                       avg_score_impact, last_used, created_at, updated_at
                FROM action_effectiveness 
                WHERE game_id LIKE ?
                ORDER BY success_rate DESC
            """, (f"{game_prefix}-%",))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'game_id': row[0],
                    'action_number': row[1],
                    'attempts': row[2],
                    'successes': row[3],
                    'success_rate': row[4],
                    'avg_score_impact': row[5],
                    'last_used': row[6],
                    'created_at': row[7],
                    'updated_at': row[8]
                })
            conn.close()
            return results
        except Exception as e:
            self.logger.error(f"Failed to get action effectiveness by prefix {game_prefix}: {e}")
            return []
    
    # ============================================================================
    # COORDINATE INTELLIGENCE INTEGRATION
    # ============================================================================
    
    async def update_coordinate_intelligence(self, game_id: str, x: int, y: int,
                                           attempts: int, successes: int,
                                           success_rate: float = None,
                                           frame_changes: int = 0) -> bool:
        """Update coordinate intelligence."""
        if success_rate is None:
            success_rate = successes / attempts if attempts > 0 else 0.0
        
        intelligence = CoordinateIntelligence(
            game_id=game_id,
            x=x,
            y=y,
            attempts=attempts,
            successes=successes,
            success_rate=success_rate,
            frame_changes=frame_changes,
            last_used=datetime.now()
        )
        
        success = await self.db.update_coordinate_intelligence(intelligence)
        if success:
            await self.db.log_system_event(
                "DEBUG", "coordinate_system", f"Updated coordinate ({x},{y}) intelligence",
                {"game_id": game_id, "success_rate": success_rate}, None, game_id
            )
        
        return success
    
    async def get_coordinate_intelligence(self, game_id: str = None, 
                                        min_success_rate: float = 0.0) -> List[Dict[str, Any]]:
        """Get coordinate intelligence data."""
        intelligence = await self.db.get_coordinate_intelligence(game_id, min_success_rate)
        return [asdict(c) for c in intelligence]
    
    async def get_best_coordinates(self, game_id: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get best performing coordinates."""
        intelligence = await self.db.get_coordinate_intelligence(game_id, 0.1)
        sorted_coords = sorted(intelligence, key=lambda x: x.success_rate, reverse=True)
        return [asdict(coord) for coord in sorted_coords[:limit]]
    
    # ============================================================================
    # LOGGING INTEGRATION
    # ============================================================================
    
    async def log_system_event(self, level: str, component: str, message: str,
                              data: Dict[str, Any] = None, session_id: str = None,
                              game_id: str = None) -> bool:
        """Log system event."""
        from .api import LogLevel, Component
        
        level_map = {
            "DEBUG": LogLevel.DEBUG,
            "INFO": LogLevel.INFO,
            "WARNING": LogLevel.WARNING,
            "ERROR": LogLevel.ERROR,
            "CRITICAL": LogLevel.CRITICAL
        }
        
        component_map = {
            "governor": Component.GOVERNOR,
            "architect": Component.ARCHITECT,
            "director": Component.DIRECTOR,
            "learning_loop": Component.LEARNING_LOOP,
            "coordinate_system": Component.COORDINATE_SYSTEM,
            "memory_system": Component.MEMORY_SYSTEM,
            "system": Component.LEARNING_LOOP,  # Map to existing component
            "meta_learning": Component.LEARNING_LOOP,  # Map to existing component
            "game_result": Component.LEARNING_LOOP  # Map to existing component
        }
        
        log_level = level_map.get(level, LogLevel.INFO)
        log_component = component_map.get(component, Component.LEARNING_LOOP)
        
        return await self.db.log_system_event(
            log_level, log_component, message, data, session_id, game_id
        )
    
    async def log_action_trace(self, session_id: str, game_id: str, action_number: int,
                              coordinates: Tuple[int, int] = None, frame_before: List = None,
                              frame_after: List = None, frame_changed: bool = False,
                              score_before: float = 0.0, score_after: float = 0.0,
                              response_data: Dict[str, Any] = None) -> bool:
        """Log action trace."""
        return await self.db.log_action_trace(
            session_id, game_id, action_number, coordinates,
            frame_before, frame_after, frame_changed,
            score_before, score_after, response_data
        )
    
    # ============================================================================
    # PATTERN LEARNING INTEGRATION
    # ============================================================================
    
    async def save_learned_pattern(self, pattern_type: str, pattern_data: Dict[str, Any],
                                  confidence: float = 0.5, frequency: int = 1,
                                  success_rate: float = 0.0, game_context: str = None) -> bool:
        """Save learned pattern."""
        async with self.db.get_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO learned_patterns
                    (pattern_type, pattern_data, confidence, frequency, success_rate, game_context, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern_type, json.dumps(pattern_data), confidence,
                    frequency, success_rate, game_context, datetime.now()
                ))
                conn.commit()
                return True
            except Exception as e:
                self.logger.error(f"Failed to save learned pattern: {e}")
                return False
    
    async def get_learned_patterns(self, pattern_type: str = None, 
                                  min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """Get learned patterns."""
        async with self.db.get_connection() as conn:
            query = "SELECT * FROM learned_patterns WHERE confidence >= ?"
            params = [min_confidence]
            
            if pattern_type:
                query += " AND pattern_type = ?"
                params.append(pattern_type)
            
            query += " ORDER BY confidence DESC, success_rate DESC"
            
            cursor = conn.execute(query, params)
            patterns = []
            for row in cursor.fetchall():
                pattern = dict(row)
                pattern["pattern_data"] = json.loads(pattern["pattern_data"])
                patterns.append(pattern)
            
            return patterns
    
    # ============================================================================
    # DIRECTOR INTEGRATION
    # ============================================================================
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status for Director."""
        return await self.director_commands.get_system_overview()
    
    async def get_learning_analysis(self, game_id: str = None) -> Dict[str, Any]:
        """Get learning analysis for Director."""
        return await self.director_commands.get_learning_analysis(game_id)
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get system health analysis."""
        return await self.director_commands.analyze_system_health()
    
    async def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary."""
        return await self.director_commands.get_performance_summary(hours)
    
    async def get_action_effectiveness_data(self, game_id: str = None) -> List[Dict[str, Any]]:
        """Get action effectiveness data (alias for compatibility)."""
        return await self.director_commands.get_action_effectiveness(game_id)
    
    async def get_coordinate_intelligence_data(self, game_id: str = None) -> List[Dict[str, Any]]:
        """Get coordinate intelligence data (alias for compatibility)."""
        return await self.director_commands.get_coordinate_intelligence(game_id)
    
    async def get_self_model_entries(self, limit: int = 100, type: str = None) -> List[Dict[str, Any]]:
        """Retrieve Director self-model entries."""
        async with self.db.get_connection() as conn:
            try:
                if type:
                    query = "SELECT * FROM director_self_model WHERE type = ? ORDER BY created_at DESC LIMIT ?"
                    cursor = conn.execute(query, (type, limit))
                else:
                    query = "SELECT * FROM director_self_model ORDER BY created_at DESC LIMIT ?"
                    cursor = conn.execute(query, (limit,))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            except Exception as e:
                self.logger.error(f"Database error retrieving self-model entries: {e}")
                return []
    
    async def add_self_model_entry(self, type: str, content: str, session_id: str = None, 
                                 importance: int = 1, metadata: dict = None) -> bool:
        """Add a new entry to the Director's self-model."""
        async with self.db.get_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO director_self_model (type, content, session_id, importance, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    type,
                    content,
                    session_id,
                    importance,
                    json.dumps(metadata) if metadata else '{}'
                ))
                conn.commit()
                return True
            except Exception as e:
                self.logger.error(f"Database error adding self-model entry: {e}")
                return False
    
    # ============================================================================
    # GLOBAL COUNTERS INTEGRATION
    # ============================================================================
    
    async def update_global_counter(self, counter_name: str, value: int, 
                                   description: str = None) -> bool:
        """Update global counter."""
        async with self.db.get_connection() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO global_counters
                    (counter_name, counter_value, description, last_updated)
                    VALUES (?, ?, ?, ?)
                """, (counter_name, value, description, datetime.now()))
                conn.commit()
                return True
            except Exception as e:
                self.logger.error(f"Failed to update global counter {counter_name}: {e}")
                return False
    
    async def get_global_counters(self) -> Dict[str, int]:
        """Get all global counters."""
        async with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT counter_name, counter_value FROM global_counters")
            return {row["counter_name"]: row["counter_value"] for row in cursor.fetchall()}
    
    # ============================================================================
    # BACKWARD COMPATIBILITY METHODS
    # ============================================================================
    
    async def save_session_data(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Save session data (backward compatibility)."""
        return await self.update_session_metrics(session_id, data)
    
    async def load_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session data (backward compatibility)."""
        return await self.get_session_status(session_id)
    
    async def save_action_intelligence(self, game_id: str, intelligence_data: Dict[str, Any]) -> bool:
        """Save action intelligence (backward compatibility)."""
        effective_actions = intelligence_data.get("effective_actions", {})
        
        for action_num_str, action_data in effective_actions.items():
            try:
                action_number = int(action_num_str)
                await self.update_action_effectiveness(
                    game_id, action_number,
                    action_data.get("attempts", 0),
                    action_data.get("successes", 0),
                    action_data.get("success_rate", 0.0),
                    action_data.get("avg_score_impact", 0.0)
                )
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Invalid action data: {e}")
                continue
        
        return True
    
    async def load_action_intelligence(self, game_id: str) -> Dict[str, Any]:
        """Load action intelligence (backward compatibility)."""
        effectiveness = await self.get_action_effectiveness(game_id)
        
        effective_actions = {}
        for action in effectiveness:
            effective_actions[str(action["action_number"])] = {
                "attempts": action["attempts"],
                "successes": action["successes"],
                "success_rate": action["success_rate"],
                "avg_score_impact": action["avg_score_impact"],
                "last_used": action["last_used"].isoformat() if action["last_used"] else None
            }
        
        return {
            "game_id": game_id,
            "effective_actions": effective_actions,
            "last_updated": datetime.now().isoformat()
        }
    
    async def log_experiment(self, experiment_name: str, experiment_type: str, parameters: Dict[str, Any],
                           results: Dict[str, Any], success: bool, duration: float = 0.0,
                           notes: str = None) -> bool:
        """Log experiment data to system_logs (experiments table was deleted)."""
        try:
            # Log experiment as system event instead of using deleted experiments table
            await self.log_system_event(
                level="INFO",
                component="experiment",
                message=f"Experiment: {experiment_name}",
                data={
                    "experiment_name": experiment_name,
                    "experiment_type": experiment_type,
                    "parameters": parameters,
                    "results": results,
                    "success": success,
                    "duration": duration,
                    "notes": notes
                },
                session_id=f"experiment_{experiment_name}"
            )
            return True
        except Exception as e:
            print(f"Database error: {e}")
            return False
    
    async def update_task_performance(self, task_id: str, performance_metrics: Dict[str, Any],
                                    learning_progress: Dict[str, Any], success_rate: float,
                                    notes: str = None) -> bool:
        """Update task performance data to system_logs (task_performance table was deleted)."""
        try:
            # Log task performance as system event instead of using deleted task_performance table
            await self.log_system_event(
                level="INFO",
                component="task_performance",
                message=f"Task Performance: {task_id}",
                data={
                    "task_id": task_id,
                    "performance_metrics": performance_metrics,
                    "learning_progress": learning_progress,
                    "success_rate": success_rate,
                    "notes": notes
                },
                session_id=f"task_{task_id}"
            )
            return True
        except Exception as e:
            print(f"Database error: {e}")
            return False
    
    async def save_game_result(self, game_id: str, session_id: str, final_score: int, 
                             total_actions: int, win_detected: bool, final_state: str = "UNKNOWN",
                             termination_reason: str = "COMPLETED") -> bool:
        """Save game result to database."""
        try:
            game_result = GameResult(
                game_id=game_id,
                session_id=session_id,
                start_time=datetime.now(),
                final_score=final_score,
                total_actions=total_actions,
                win_detected=win_detected,
                status=final_state,
                end_time=datetime.now()
            )
            
            success = await self.db.create_game_result(game_result)
            if success:
                await self.db.log_system_event(
                    "INFO", "game_result", f"Saved game result for {game_id}",
                    {"session_id": session_id, "score": final_score, "win_detected": win_detected}, session_id
                )
            
            return success
        except Exception as e:
            print(f"Error saving game result: {e}")
            return False

# ============================================================================
# GLOBAL INTEGRATION INSTANCE
# ============================================================================

# Global instance for easy access
_integration_instance = None

def get_system_integration() -> SystemIntegration:
    """Get global system integration instance."""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = SystemIntegration()
    return _integration_instance

# ============================================================================
# DIRECTOR SELF-MODEL PERSISTENCE
# ============================================================================

async def add_self_model_entry(type: str, content: str, session_id: int = None, 
                             importance: int = 1, metadata: dict = None) -> bool:
    """Add a new entry to the Director's self-model."""
    integration = get_system_integration()
    async with integration.db.get_connection() as conn:
        try:
            conn.execute("""
                INSERT INTO director_self_model (type, content, session_id, importance, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                type,
                content,
                session_id,
                importance,
                json.dumps(metadata) if metadata else '{}'
            ))
            conn.commit()  # Ensure the transaction is committed
            return True
        except Exception as e:
            print(f"Database error: {e}")
            return False

async def get_self_model_entries(limit: int = 100, type: str = None) -> List[Dict[str, Any]]:
    """Retrieve Director self-model entries."""
    integration = get_system_integration()
    async with integration.db.get_connection() as conn:
        try:
            if type:
                query = "SELECT * FROM director_self_model WHERE type = ? ORDER BY created_at DESC LIMIT ?"
                cursor = conn.execute(query, (type, limit))
            else:
                query = "SELECT * FROM director_self_model ORDER BY created_at DESC LIMIT ?"
                cursor = conn.execute(query, (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"Database error: {e}")
            return []

# ============================================================================       
# CONVENIENCE FUNCTIONS
# ============================================================================       

async def log_event(level: str, component: str, message: str, **kwargs):
    """Quick log event function."""
    integration = get_system_integration()
    return await integration.log_system_event(level, component, message, **kwargs)

async def get_status() -> Dict[str, Any]:
    """Quick status function."""
    integration = get_system_integration()
    return await integration.get_system_status()

async def get_health() -> Dict[str, Any]:
    """Quick health function."""
    integration = get_system_integration()
    return await integration.get_system_health()
