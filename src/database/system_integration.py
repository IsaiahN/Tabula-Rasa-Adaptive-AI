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

from .api import get_database, TrainingSession, GameResult, ActionEffectiveness, CoordinateIntelligence, LogLevel, Component
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
                LogLevel.INFO, Component.LEARNING_LOOP, f"Created training session {session_id}",
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
    
    async def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all training sessions."""
        sessions = await self.db.fetch_all("""
            SELECT * FROM training_sessions
            ORDER BY start_time DESC
        """)
        return [dict(session) for session in sessions]
    
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
                LogLevel.INFO, Component.LEARNING_LOOP, f"Logged game result for {game_id}",
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
                LogLevel.DEBUG, Component.LEARNING_LOOP, f"Updated action {action_number} effectiveness",
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
                LogLevel.DEBUG, Component.COORDINATE_SYSTEM, f"Updated coordinate ({x},{y}) intelligence",
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
    
    async def log_system_event(self, level: LogLevel, component: Component, message: str,
                              data: Dict[str, Any] = None, session_id: str = None,
                              game_id: str = None) -> bool:
        """Log system event."""
        return await self.db.log_system_event(
            level, component, message, data, session_id, game_id
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
                # Defensive sanitization: sqlite3 does not accept dict/list types directly.
                # Ensure pattern_data and any dict/list/game_context are JSON-serialized.
                safe_pattern_data = json.dumps(pattern_data)

                # success_rate should normally be a number; if it's a dict/list, serialize it.
                if isinstance(success_rate, (dict, list)):
                    safe_success_rate = json.dumps(success_rate)
                else:
                    # coerce None -> 0.0, otherwise keep numeric as-is
                    safe_success_rate = 0.0 if success_rate is None else float(success_rate)

                # game_context may be a dict in some callsites; serialize if needed
                if isinstance(game_context, (dict, list)):
                    safe_game_context = json.dumps(game_context)
                else:
                    safe_game_context = game_context

                # created_at: store an ISO formatted string to keep storage consistent
                created_at_str = datetime.now().isoformat()

                conn.execute("""
                    INSERT INTO learned_patterns
                    (pattern_type, pattern_data, confidence, frequency, success_rate, game_context, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern_type, safe_pattern_data, float(confidence), int(frequency),
                    safe_success_rate, safe_game_context, created_at_str
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
        """Log experiment data to experiments table."""
        try:
            # Save to experiments table
            await self.db.execute("""
                INSERT INTO experiments 
                (experiment_name, experiment_type, parameters, results, success, duration, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_name,
                experiment_type,
                json.dumps(parameters),
                json.dumps(results),
                success,
                duration,
                notes
            ))
            
            # Also log as system event for tracking
            await self.log_system_event(
                level=LogLevel.INFO,
                component=Component.EXPERIMENT,
                message=f"Experiment: {experiment_name}",
                data={
                    "experiment_name": experiment_name,
                    "experiment_type": experiment_type,
                    "success": success,
                    "duration": duration
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
        """Update task performance data to task_performance table."""
        try:
            # Save to task_performance table
            await self.db.execute("""
                INSERT OR REPLACE INTO task_performance 
                (task_id, training_sessions, learning_progress, success_rate, notes, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                task_id,
                json.dumps(performance_metrics),
                json.dumps(learning_progress),
                success_rate,
                notes
            ))
            
            # Also log as system event for tracking
            await self.log_system_event(
                level=LogLevel.INFO,
                component=Component.TASK_PERFORMANCE,
                message=f"Task Performance: {task_id}",
                data={
                    "task_id": task_id,
                    "success_rate": success_rate,
                    "performance_metrics": performance_metrics
                },
                session_id=f"task_{task_id}"
            )
            return True
        except Exception as e:
            print(f"Database error: {e}")
            return False
    
    async def log_error(self, error_type: str, error_message: str, stack_trace: str = None,
                       context: str = None, session_id: str = None, game_id: str = None) -> bool:
        """Log error to error_logs table."""
        try:
            import hashlib
            error_hash = hashlib.md5(f"{error_type}:{error_message}".encode()).hexdigest()
            
            # Check if error already exists
            existing = await self.db.fetch_all(
                "SELECT occurrence_count FROM error_logs WHERE error_hash = ?",
                (error_hash,)
            )
            
            if existing:
                # Update occurrence count
                await self.db.execute("""
                    UPDATE error_logs 
                    SET occurrence_count = occurrence_count + 1, last_seen = CURRENT_TIMESTAMP
                    WHERE error_hash = ?
                """, (error_hash,))
            else:
                # Insert new error
                await self.db.execute("""
                    INSERT INTO error_logs 
                    (error_type, error_message, error_hash, stack_trace, context, occurrence_count)
                    VALUES (?, ?, ?, ?, ?, 1)
                """, (error_type, error_message, error_hash, stack_trace, context))
            return True
        except Exception as e:
            print(f"Database error logging failed: {e}")
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
                    LogLevel.INFO, Component.GAME_RESULT, f"Saved game result for {game_id}",
                    {"session_id": session_id, "score": final_score, "win_detected": win_detected}, session_id
                )
            
            return success
        except Exception as e:
            print(f"Error saving game result: {e}")
            return False

    # ============================================================================
    # ADVANCED ACTION SYSTEM INTEGRATION
    # ============================================================================
    
    async def store_visual_target(self, 
                                game_id: str,
                                target_x: int,
                                target_y: int,
                                target_type: str,
                                confidence: float,
                                detection_method: str,
                                interaction_successful: bool = False,
                                frame_changes_detected: bool = False,
                                score_impact: float = 0.0) -> bool:
        """Store visual target detection result."""
        try:
            import time
            await self.db.execute("""
                INSERT INTO visual_targets
                (game_id, target_x, target_y, target_type, confidence, detection_method,
                 interaction_successful, frame_changes_detected, score_impact, detection_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (game_id, target_x, target_y, target_type, confidence, detection_method,
                  interaction_successful, frame_changes_detected, score_impact, time.time()))
            return True
        except Exception as e:
            self.logger.error(f"Error storing visual target: {e}")
            return False
    
    async def store_stagnation_event(self, 
                                   game_id: str,
                                   session_id: str,
                                   stagnation_type: str,
                                   severity: float,
                                   consecutive_count: int,
                                   context_data: Dict[str, Any],
                                   recovery_action: Optional[str] = None,
                                   recovery_successful: bool = False) -> bool:
        """Store stagnation event."""
        try:
            import time
            await self.db.execute("""
                INSERT INTO stagnation_events
                (game_id, session_id, stagnation_type, severity, consecutive_count,
                 stagnation_context, recovery_action, recovery_successful, detection_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (game_id, session_id, stagnation_type, severity, consecutive_count,
                  json.dumps(context_data), recovery_action, recovery_successful, time.time()))
            return True
        except Exception as e:
            self.logger.error(f"Error storing stagnation event: {e}")
            return False
    
    async def store_winning_strategy(self, 
                                   strategy_id: str,
                                   game_type: str,
                                   game_id: str,
                                   action_sequence: List[int],
                                   score_progression: List[float],
                                   total_score_increase: float,
                                   efficiency: float) -> bool:
        """Store winning strategy."""
        try:
            import time
            await self.db.execute("""
                INSERT INTO winning_strategies
                (strategy_id, game_type, game_id, action_sequence, score_progression,
                 total_score_increase, efficiency, discovery_timestamp, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (strategy_id, game_type, game_id, json.dumps(action_sequence),
                  json.dumps(score_progression), total_score_increase, efficiency,
                  time.time(), True))
            return True
        except Exception as e:
            self.logger.error(f"Error storing winning strategy: {e}")
            return False
    
    async def store_frame_change_analysis(self, 
                                        game_id: str,
                                        action_number: int,
                                        coordinates: Optional[Tuple[int, int]],
                                        change_type: str,
                                        num_pixels_changed: int,
                                        change_percentage: float,
                                        movement_detected: bool,
                                        change_locations: List[Tuple[int, int]],
                                        classification_confidence: float) -> bool:
        """Store frame change analysis."""
        try:
            import time
            await self.db.execute("""
                INSERT INTO frame_change_analysis
                (game_id, action_number, coordinates_x, coordinates_y, change_type,
                 num_pixels_changed, change_percentage, movement_detected,
                 change_locations, classification_confidence, analysis_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (game_id, action_number,
                  coordinates[0] if coordinates else None,
                  coordinates[1] if coordinates else None,
                  change_type, num_pixels_changed, change_percentage, movement_detected,
                  json.dumps(change_locations), classification_confidence, time.time()))
            return True
        except Exception as e:
            self.logger.error(f"Error storing frame change analysis: {e}")
            return False
    
    async def store_exploration_phase(self, 
                                    game_id: str,
                                    session_id: str,
                                    phase_name: str,
                                    phase_attempts: int,
                                    successful_attempts: int,
                                    coordinates_tried: List[Tuple[int, int]],
                                    phase_success_rate: float) -> bool:
        """Store exploration phase data."""
        try:
            import time
            await self.db.execute("""
                INSERT INTO exploration_phases
                (game_id, session_id, phase_name, phase_attempts, successful_attempts,
                 coordinates_tried, phase_start_time, phase_success_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (game_id, session_id, phase_name, phase_attempts, successful_attempts,
                  json.dumps(coordinates_tried), time.time(), phase_success_rate))
            return True
        except Exception as e:
            self.logger.error(f"Error storing exploration phase: {e}")
            return False
    
    async def store_emergency_override(self, 
                                     game_id: str,
                                     session_id: str,
                                     override_type: str,
                                     trigger_reason: str,
                                     actions_before_override: int,
                                     override_action: int,
                                     override_successful: bool = False) -> bool:
        """Store emergency override event."""
        try:
            import time
            await self.db.execute("""
                INSERT INTO emergency_overrides
                (game_id, session_id, override_type, trigger_reason, actions_before_override,
                 override_action, override_successful, override_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (game_id, session_id, override_type, trigger_reason, actions_before_override,
                  override_action, override_successful, time.time()))
            return True
        except Exception as e:
            self.logger.error(f"Error storing emergency override: {e}")
            return False
    
    async def store_governor_decision(self, 
                                    session_id: str,
                                    decision_type: str,
                                    context_data: Dict[str, Any],
                                    governor_confidence: float,
                                    decision_outcome: Dict[str, Any]) -> bool:
        """Store governor decision."""
        try:
            import time
            await self.db.execute("""
                INSERT INTO governor_decisions
                (session_id, decision_type, context_data, governor_confidence, decision_outcome, decision_timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, decision_type, json.dumps(context_data), governor_confidence,
                  json.dumps(decision_outcome), time.time()))
            return True
        except Exception as e:
            self.logger.error(f"Error storing governor decision: {e}")
            return False
    
    async def get_visual_targets_for_game(self, game_id: str) -> List[Dict[str, Any]]:
        """Get visual targets for a game."""
        try:
            results = await self.db.fetch_all("""
                SELECT target_x, target_y, target_type, confidence, detection_method,
                       interaction_successful, frame_changes_detected, score_impact
                FROM visual_targets
                WHERE game_id = ?
                ORDER BY detection_timestamp DESC
            """, (game_id,))
            return [dict(row) for row in results]
        except Exception as e:
            self.logger.error(f"Error getting visual targets: {e}")
            return []
    
    async def get_stagnation_events_for_game(self, game_id: str) -> List[Dict[str, Any]]:
        """Get stagnation events for a game."""
        try:
            results = await self.db.fetch_all("""
                SELECT stagnation_type, severity, consecutive_count, stagnation_context,
                       recovery_action, recovery_successful, detection_timestamp
                FROM stagnation_events
                WHERE game_id = ?
                ORDER BY detection_timestamp DESC
            """, (game_id,))
            return [dict(row) for row in results]
        except Exception as e:
            self.logger.error(f"Error getting stagnation events: {e}")
            return []
    
    async def get_winning_strategies_for_game_type(self, game_type: str) -> List[Dict[str, Any]]:
        """Get winning strategies for a game type."""
        try:
            results = await self.db.fetch_all("""
                SELECT strategy_id, game_id, action_sequence, score_progression,
                       total_score_increase, efficiency, replication_attempts,
                       successful_replications, refinement_level, is_active
                FROM winning_strategies
                WHERE game_type = ? AND is_active = 1
                ORDER BY efficiency DESC
            """, (game_type,))
            return [dict(row) for row in results]
        except Exception as e:
            self.logger.error(f"Error getting winning strategies: {e}")
            return []
    
    async def get_frame_change_analysis_for_game(self, game_id: str) -> List[Dict[str, Any]]:
        """Get frame change analysis for a game."""
        try:
            results = await self.db.fetch_all("""
                SELECT action_number, coordinates_x, coordinates_y, change_type,
                       num_pixels_changed, change_percentage, movement_detected,
                       change_locations, classification_confidence, analysis_timestamp
                FROM frame_change_analysis
                WHERE game_id = ?
                ORDER BY analysis_timestamp DESC
            """, (game_id,))
            return [dict(row) for row in results]
        except Exception as e:
            self.logger.error(f"Error getting frame change analysis: {e}")
            return []
    
    async def get_exploration_phases_for_game(self, game_id: str) -> List[Dict[str, Any]]:
        """Get exploration phases for a game."""
        try:
            results = await self.db.fetch_all("""
                SELECT phase_name, phase_attempts, successful_attempts, coordinates_tried,
                       phase_start_time, phase_end_time, phase_success_rate
                FROM exploration_phases
                WHERE game_id = ?
                ORDER BY phase_start_time ASC
            """, (game_id,))
            return [dict(row) for row in results]
        except Exception as e:
            self.logger.error(f"Error getting exploration phases: {e}")
            return []
    
    async def get_emergency_overrides_for_game(self, game_id: str) -> List[Dict[str, Any]]:
        """Get emergency overrides for a game."""
        try:
            results = await self.db.fetch_all("""
                SELECT override_type, trigger_reason, actions_before_override,
                       override_action, override_successful, override_timestamp
                FROM emergency_overrides
                WHERE game_id = ?
                ORDER BY override_timestamp DESC
            """, (game_id,))
            return [dict(row) for row in results]
        except Exception as e:
            self.logger.error(f"Error getting emergency overrides: {e}")
            return []
    
    async def get_governor_decisions_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get governor decisions for a session."""
        try:
            results = await self.db.fetch_all("""
                SELECT decision_type, context_data, governor_confidence,
                       decision_outcome, decision_timestamp
                FROM governor_decisions
                WHERE session_id = ?
                ORDER BY decision_timestamp DESC
            """, (session_id,))
            return [dict(row) for row in results]
        except Exception as e:
            self.logger.error(f"Error getting governor decisions: {e}")
            return []

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
# SUBSYSTEM MONITORING INTEGRATION
# ============================================================================

async def store_subsystem_metrics(metrics_data: Dict[str, Any]) -> bool:
    """Store subsystem metrics in the database."""
    integration = get_system_integration()
    try:
        # Store in a dedicated subsystem_metrics table
        # This would be implemented in the database API
        await integration.db.log_system_event(
            LogLevel.INFO, Component.SUBSYSTEM_MONITOR, 
            f"Stored metrics for subsystem {metrics_data.get('subsystem_id', 'unknown')}",
            metrics_data, metrics_data.get('subsystem_id', 'unknown')
        )
        return True
    except Exception as e:
        print(f"Failed to store subsystem metrics: {e}")
        return False

async def store_subsystem_config(subsystem_id: str, config_data: Dict[str, Any]) -> bool:
    """Store subsystem configuration in the database."""
    integration = get_system_integration()
    try:
        await integration.db.log_system_event(
            LogLevel.INFO, Component.SUBSYSTEM_MONITOR,
            f"Updated configuration for subsystem {subsystem_id}",
            config_data, subsystem_id
        )
        return True
    except Exception as e:
        print(f"Failed to store subsystem config: {e}")
        return False

async def store_subsystem_alerts(subsystem_id: str, alerts: List[Dict[str, Any]]) -> bool:
    """Store subsystem alerts in the database."""
    integration = get_system_integration()
    try:
        for alert in alerts:
            await integration.db.log_system_event(
                LogLevel.WARNING if alert.get('severity') == 'warning' else LogLevel.ERROR,
                Component.SUBSYSTEM_MONITOR,
                f"Subsystem {subsystem_id} alert: {alert.get('message', 'Unknown alert')}",
                alert, subsystem_id
            )
        return True
    except Exception as e:
        print(f"Failed to store subsystem alerts: {e}")
        return False

async def get_subsystem_config(subsystem_id: str) -> Optional[Dict[str, Any]]:
    """Get subsystem configuration from the database."""
    integration = get_system_integration()
    try:
        # This would query the database for stored configuration
        # For now, return None to use default configuration
        return None
    except Exception as e:
        print(f"Failed to get subsystem config: {e}")
        return None

async def get_subsystem_metrics_history(subsystem_id: str, hours: int = 24) -> List[Dict[str, Any]]:
    """Get subsystem metrics history from the database."""
    integration = get_system_integration()
    try:
        # This would query the database for historical metrics
        # For now, return empty list
        return []
    except Exception as e:
        print(f"Failed to get subsystem metrics history: {e}")
        return []

async def get_subsystem_alerts(subsystem_id: str, hours: int = 24) -> List[Dict[str, Any]]:
    """Get subsystem alerts from the database."""
    integration = get_system_integration()
    try:
        # This would query the database for alerts
        # For now, return empty list
        return []
    except Exception as e:
        print(f"Failed to get subsystem alerts: {e}")
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
