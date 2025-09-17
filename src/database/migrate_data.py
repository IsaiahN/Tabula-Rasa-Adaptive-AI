"""
DATA MIGRATION SCRIPT
Migrate existing file-based data to SQLite database
"""

import asyncio
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import sqlite3
import glob
import os

from .api import get_database, TrainingSession, GameResult, ActionEffectiveness, CoordinateIntelligence
from .director_commands import get_director_commands

class DataMigrator:
    """
    Migrates existing file-based data to SQLite database.
    Handles all data types from the file storage paradigm.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.db = get_database()
        self.logger = logging.getLogger(__name__)
        self.migration_stats = {
            "sessions_migrated": 0,
            "games_migrated": 0,
            "action_intelligence_migrated": 0,
            "coordinate_intelligence_migrated": 0,
            "logs_migrated": 0,
            "patterns_migrated": 0,
            "errors": []
        }
    
    async def migrate_all_data(self) -> Dict[str, Any]:
        """
        Migrate all existing data to database.
        
        Returns:
            Migration statistics and results
        """
        self.logger.info("Starting data migration from file storage to SQLite database")
        
        try:
            # Migrate in order of dependencies
            await self._migrate_training_sessions()
            await self._migrate_game_results()
            await self._migrate_action_intelligence()
            await self._migrate_coordinate_intelligence()
            await self._migrate_learned_patterns()
            await self._migrate_system_logs()
            await self._migrate_system_logs()
            await self._migrate_system_logs()
            await self._migrate_experiments()
            await self._migrate_global_counters()
            await self._migrate_training_sessions()
            await self._migrate_system_logs()
            
            self.logger.info("Data migration completed successfully")
            return self.migration_stats
            
        except Exception as e:
            self.logger.error(f"Data migration failed: {e}")
            self.migration_stats["errors"].append(str(e))
            return self.migration_stats
    
    async def _migrate_training_sessions(self):
        """Migrate training session data."""
        self.logger.info("Migrating training sessions...")
        
        # Look for session files
        session_files = list(self.data_dir.glob("sessions/*.json"))
        
        for session_file in session_files:
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                # Extract session information
                session_id = session_data.get("session_id", session_file.stem)
                start_time = datetime.fromisoformat(session_data.get("start_time", datetime.now().isoformat()))
                end_time = datetime.fromisoformat(session_data.get("end_time", datetime.now().isoformat())) if session_data.get("end_time") else None
                
                # Create training session
                session = TrainingSession(
                    session_id=session_id,
                    start_time=start_time,
                    end_time=end_time,
                    mode=session_data.get("mode", "unknown"),
                    status=session_data.get("status", "completed"),
                    total_actions=session_data.get("total_actions", 0),
                    total_wins=session_data.get("total_wins", 0),
                    total_games=session_data.get("total_games", 0),
                    win_rate=session_data.get("win_rate", 0.0),
                    avg_score=session_data.get("avg_score", 0.0),
                    energy_level=session_data.get("energy_level", 100.0),
                    memory_operations=session_data.get("memory_operations", 0),
                    sleep_cycles=session_data.get("sleep_cycles", 0)
                )
                
                await self.db.create_session(session)
                self.migration_stats["sessions_migrated"] += 1
                
            except Exception as e:
                self.logger.error(f"Failed to migrate session {session_file}: {e}")
                self.migration_stats["errors"].append(f"Session {session_file}: {e}")
    
    async def _migrate_game_results(self):
        """Migrate game result data."""
        self.logger.info("Migrating game results...")
        
        # Look for game result files
        game_files = list(self.data_dir.glob("sessions/*.json"))
        
        for game_file in game_files:
            try:
                with open(game_file, 'r') as f:
                    game_data = json.load(f)
                
                # Extract game information
                game_id = game_data.get("game_id", "unknown")
                session_id = game_data.get("session_id", "unknown")
                start_time = datetime.fromisoformat(game_data.get("start_time", datetime.now().isoformat()))
                end_time = datetime.fromisoformat(game_data.get("end_time", datetime.now().isoformat())) if game_data.get("end_time") else None
                
                # Create game result
                game_result = GameResult(
                    game_id=game_id,
                    session_id=session_id,
                    start_time=start_time,
                    end_time=end_time,
                    status=game_data.get("status", "completed"),
                    final_score=game_data.get("final_score", 0.0),
                    total_actions=game_data.get("total_actions", 0),
                    actions_taken=game_data.get("actions_taken", []),
                    win_detected=game_data.get("win_detected", False),
                    level_completions=game_data.get("level_completions", 0),
                    frame_changes=game_data.get("frame_changes", 0),
                    coordinate_attempts=game_data.get("coordinate_attempts", 0),
                    coordinate_successes=game_data.get("coordinate_successes", 0)
                )
                
                await self.db.create_game_result(game_result)
                self.migration_stats["games_migrated"] += 1
                
            except Exception as e:
                self.logger.error(f"Failed to migrate game {game_file}: {e}")
                self.migration_stats["errors"].append(f"Game {game_file}: {e}")
    
    async def _migrate_action_intelligence(self):
        """Migrate action intelligence data."""
        self.logger.info("Migrating action intelligence...")
        
        # Look for action intelligence files
        intelligence_files = list(self.data_dir.glob("action_intelligence_*.json"))
        
        for intel_file in intelligence_files:
            try:
                with open(intel_file, 'r') as f:
                    intel_data = json.load(f)
                
                game_id = intel_data.get("game_id", "unknown")
                effective_actions = intel_data.get("effective_actions", {})
                
                # Migrate each action
                for action_num_str, action_data in effective_actions.items():
                    try:
                        action_number = int(action_num_str)
                        
                        effectiveness = ActionEffectiveness(
                            game_id=game_id,
                            action_number=action_number,
                            attempts=action_data.get("attempts", 0),
                            successes=action_data.get("successes", 0),
                            success_rate=action_data.get("success_rate", 0.0),
                            avg_score_impact=action_data.get("avg_score_impact", 0.0),
                            last_used=datetime.fromtimestamp(action_data.get("last_used", 0)) if action_data.get("last_used") else None
                        )
                        
                        await self.db.update_action_effectiveness(effectiveness)
                        self.migration_stats["action_intelligence_migrated"] += 1
                        
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Invalid action data in {intel_file}: {e}")
                        continue
                
            except Exception as e:
                self.logger.error(f"Failed to migrate action intelligence {intel_file}: {e}")
                self.migration_stats["errors"].append(f"Action intelligence {intel_file}: {e}")
    
    async def _migrate_coordinate_intelligence(self):
        """Migrate coordinate intelligence data."""
        self.logger.info("Migrating coordinate intelligence...")
        
        # Look for action intelligence files with coordinate data
        intelligence_files = list(self.data_dir.glob("action_intelligence_*.json"))
        
        for intel_file in intelligence_files:
            try:
                with open(intel_file, 'r') as f:
                    intel_data = json.load(f)
                
                game_id = intel_data.get("game_id", "unknown")
                coordinate_patterns = intel_data.get("coordinate_patterns", {})
                
                # Migrate coordinate patterns
                for coord_str, coord_data in coordinate_patterns.items():
                    try:
                        # Parse coordinate string like "(32,32)"
                        coord_str = coord_str.strip('()')
                        x, y = map(int, coord_str.split(','))
                        
                        intelligence = CoordinateIntelligence(
                            game_id=game_id,
                            x=x,
                            y=y,
                            attempts=coord_data.get("attempts", 0),
                            successes=coord_data.get("successes", 0),
                            success_rate=coord_data.get("success_rate", 0.0),
                            frame_changes=coord_data.get("frame_changes", 0),
                            last_used=datetime.fromtimestamp(coord_data.get("last_used", 0)) if coord_data.get("last_used") else None
                        )
                        
                        await self.db.update_coordinate_intelligence(intelligence)
                        self.migration_stats["coordinate_intelligence_migrated"] += 1
                        
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Invalid coordinate data in {intel_file}: {e}")
                        continue
                
            except Exception as e:
                self.logger.error(f"Failed to migrate coordinate intelligence {intel_file}: {e}")
                self.migration_stats["errors"].append(f"Coordinate intelligence {intel_file}: {e}")
    
    async def _migrate_learned_patterns(self):
        """Migrate learned patterns data."""
        self.logger.info("Migrating learned patterns...")
        
        # Look for learned patterns pickle file
        patterns_file = self.data_dir / "learned_patterns.pkl"
        
        if patterns_file.exists():
            try:
                with open(patterns_file, 'rb') as f:
                    patterns_data = pickle.load(f)
                
                # Migrate patterns to database
                async with self.db.get_connection() as conn:
                    for pattern_type, pattern_data in patterns_data.items():
                        try:
                            conn.execute("""
                                INSERT INTO learned_patterns
                                (pattern_type, pattern_data, confidence, frequency, success_rate, created_at)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (
                                pattern_type,
                                json.dumps(pattern_data),
                                0.5,  # Default confidence
                                1,    # Default frequency
                                0.0,  # Default success rate
                                datetime.now()
                            ))
                            
                            self.migration_stats["patterns_migrated"] += 1
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to migrate pattern {pattern_type}: {e}")
                            continue
                    
                    conn.commit()
                
            except Exception as e:
                self.logger.error(f"Failed to migrate learned patterns {patterns_file}: {e}")
                self.migration_stats["errors"].append(f"Learned patterns {patterns_file}: {e}")
    
    async def _migrate_system_logs(self):
        """Migrate system log data."""
        self.logger.info("Migrating system logs...")
        
        # Look for log files
        log_files = list(self.data_dir.glob("logs/*.log"))
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    log_lines = f.readlines()
                
                # Parse log lines and migrate
                async with self.db.get_connection() as conn:
                    for line in log_lines:
                        try:
                            # Parse log line format: timestamp - component - level - message
                            parts = line.strip().split(' - ', 3)
                            if len(parts) >= 4:
                                timestamp_str, component, level, message = parts
                                
                                # Parse timestamp
                                timestamp = datetime.fromisoformat(timestamp_str.replace(' ', 'T'))
                                
                                # Map log levels
                                level_map = {
                                    'DEBUG': 'DEBUG',
                                    'INFO': 'INFO',
                                    'WARNING': 'WARNING',
                                    'ERROR': 'ERROR',
                                    'CRITICAL': 'CRITICAL'
                                }
                                
                                log_level = level_map.get(level, 'INFO')
                                
                                # Insert log entry
                                conn.execute("""
                                    INSERT INTO system_logs
                                    (log_level, component, message, timestamp, created_at)
                                    VALUES (?, ?, ?, ?, ?)
                                """, (
                                    log_level,
                                    component,
                                    message,
                                    timestamp,
                                    datetime.now()
                                ))
                                
                                self.migration_stats["logs_migrated"] += 1
                                
                        except Exception as e:
                            self.logger.warning(f"Failed to parse log line: {e}")
                            continue
                    
                    conn.commit()
                
            except Exception as e:
                self.logger.error(f"Failed to migrate log file {log_file}: {e}")
                self.migration_stats["errors"].append(f"Log file {log_file}: {e}")
    
    async def _migrate_system_logs(self):
        """Migrate governor decision data."""
        self.logger.info("Migrating governor decisions...")
        
        # Look for governor decision files
        gov_files = list(self.data_dir.glob("logs/system_logs_*.log"))
        
        for gov_file in gov_files:
            try:
                with open(gov_file, 'r', encoding='utf-8', errors='ignore') as f:
                    gov_lines = f.readlines()
                
                # Parse governor decision lines
                async with self.db.get_connection() as conn:
                    for line in gov_lines:
                        try:
                            # Parse governor decision format
                            if "GOVERNOR DECISION" in line:
                                # Extract decision information
                                decision_data = {
                                    "decision_type": "unknown",
                                    "rationale": "Migrated from log",
                                    "confidence": 0.5
                                }
                                
                                conn.execute("""
                                    INSERT INTO system_logs
                                    (session_id, decision_type, decision_data, rationale, confidence, timestamp, created_at)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    "migrated_session",
                                    decision_data["decision_type"],
                                    json.dumps(decision_data),
                                    decision_data["rationale"],
                                    decision_data["confidence"],
                                    datetime.now(),
                                    datetime.now()
                                ))
                                
                        except Exception as e:
                            self.logger.warning(f"Failed to parse governor decision line: {e}")
                            continue
                    
                    conn.commit()
                
            except Exception as e:
                self.logger.error(f"Failed to migrate governor decisions {gov_file}: {e}")
                self.migration_stats["errors"].append(f"Governor decisions {gov_file}: {e}")
    
    async def _migrate_system_logs(self):
        """Migrate architecture evolution data."""
        self.logger.info("Migrating architecture evolution...")
        
        # Look for architecture evolution files
        arch_files = list(self.data_dir.glob("architecture/evolution/*.json"))
        
        for arch_file in arch_files:
            try:
                with open(arch_file, 'r') as f:
                    arch_data = json.load(f)
                
                # Migrate architecture evolution data
                async with self.db.get_connection() as conn:
                    conn.execute("""
                        INSERT INTO system_logs
                        (generation, evolution_type, changes, performance_impact, success, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        arch_data.get("generation", 1),
                        arch_data.get("evolution_type", "unknown"),
                        json.dumps(arch_data.get("changes", {})),
                        arch_data.get("performance_impact", 0.0),
                        arch_data.get("success", False),
                        datetime.now()
                    ))
                    
                    conn.commit()
                
            except Exception as e:
                self.logger.error(f"Failed to migrate architecture evolution {arch_file}: {e}")
                self.migration_stats["errors"].append(f"Architecture evolution {arch_file}: {e}")
    
    async def _migrate_experiments(self):
        """Migrate experiment data."""
        self.logger.info("Migrating experiments...")
        
        # Look for experiment files
        exp_files = list(self.data_dir.glob("experiments/*.json"))
        
        for exp_file in exp_files:
            try:
                with open(exp_file, 'r') as f:
                    exp_data = json.load(f)
                
                # Migrate experiment data
                async with self.db.get_connection() as conn:
                    conn.execute("""
                        -- INSERT INTO experiments (table deleted, use system_logs)
                        (experiment_name, experiment_type, parameters, results, success, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        exp_data.get("experiment_name", exp_file.stem),
                        exp_data.get("experiment_type", "unknown"),
                        json.dumps(exp_data.get("parameters", {})),
                        json.dumps(exp_data.get("results", {})),
                        exp_data.get("success", False),
                        datetime.now()
                    ))
                    
                    conn.commit()
                
            except Exception as e:
                self.logger.error(f"Failed to migrate experiment {exp_file}: {e}")
                self.migration_stats["errors"].append(f"Experiment {exp_file}: {e}")
    
    async def _migrate_global_counters(self):
        """Migrate global counters data."""
        self.logger.info("Migrating global counters...")
        
        # Look for global counters file
        counters_file = self.data_dir / "global_counters.json"
        
        if counters_file.exists():
            try:
                with open(counters_file, 'r') as f:
                    counters_data = json.load(f)
                
                # Migrate counters to database
                async with self.db.get_connection() as conn:
                    for counter_name, counter_value in counters_data.items():
                        conn.execute("""
                            INSERT OR REPLACE INTO global_counters
                            (counter_name, counter_value, last_updated)
                            VALUES (?, ?, ?)
                        """, (
                            counter_name,
                            counter_value,
                            datetime.now()
                        ))
                    
                    conn.commit()
                
            except Exception as e:
                self.logger.error(f"Failed to migrate global counters {counters_file}: {e}")
                self.migration_stats["errors"].append(f"Global counters {counters_file}: {e}")
    
    async def _migrate_training_sessions(self):
        """Migrate performance metrics data."""
        self.logger.info("Migrating performance metrics...")
        
        # Look for performance metrics files
        perf_files = list(self.data_dir.glob("task_performance.json"))
        
        for perf_file in perf_files:
            try:
                with open(perf_file, 'r') as f:
                    perf_data = json.load(f)
                
                # Migrate performance metrics
                async with self.db.get_connection() as conn:
                    for metric_name, metric_value in perf_data.items():
                        conn.execute("""
                            INSERT INTO training_sessions
                            (metric_name, metric_value, metric_type, timestamp, created_at)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            metric_name,
                            metric_value,
                            "counter",
                            datetime.now(),
                            datetime.now()
                        ))
                    
                    conn.commit()
                
            except Exception as e:
                self.logger.error(f"Failed to migrate performance metrics {perf_file}: {e}")
                self.migration_stats["errors"].append(f"Performance metrics {perf_file}: {e}")
    
    async def _migrate_system_logs(self):
        """Migrate reset debug logs."""
        self.logger.info("Migrating reset debug logs...")
        
        # Look for reset debug log files
        reset_files = list(self.data_dir.glob("system_logs/*.ndjson"))
        
        for reset_file in reset_files:
            try:
                with open(reset_file, 'r') as f:
                    reset_lines = f.readlines()
                
                # Parse reset debug logs
                async with self.db.get_connection() as conn:
                    for line in reset_lines:
                        try:
                            reset_data = json.loads(line.strip())
                            
                            conn.execute("""
                                INSERT INTO system_logs
                                (session_id, game_id, reset_reason, debug_data, timestamp, created_at)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (
                                reset_data.get("session_id", "unknown"),
                                reset_data.get("game_id", "unknown"),
                                reset_data.get("reset_reason", "unknown"),
                                json.dumps(reset_data),
                                datetime.fromtimestamp(reset_data.get("ts", 0)),
                                datetime.now()
                            ))
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to parse reset debug line: {e}")
                            continue
                    
                    conn.commit()
                
            except Exception as e:
                self.logger.error(f"Failed to migrate reset debug logs {reset_file}: {e}")
                self.migration_stats["errors"].append(f"Reset debug logs {reset_file}: {e}")

# ============================================================================
# MIGRATION EXECUTION
# ============================================================================

async def run_migration(data_dir: str = "data") -> Dict[str, Any]:
    """
    Run the complete data migration.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Migration statistics and results
    """
    migrator = DataMigrator(data_dir)
    return await migrator.migrate_all_data()

if __name__ == "__main__":
    # Run migration when script is executed directly
    import asyncio
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    
    async def main():
        print("Starting data migration...")
        results = await run_migration(data_dir)
        
        print("\nMigration Results:")
        print(f"Sessions migrated: {results['sessions_migrated']}")
        print(f"Games migrated: {results['games_migrated']}")
        print(f"Action intelligence migrated: {results['action_intelligence_migrated']}")
        print(f"Coordinate intelligence migrated: {results['coordinate_intelligence_migrated']}")
        print(f"Logs migrated: {results['logs_migrated']}")
        print(f"Patterns migrated: {results['patterns_migrated']}")
        
        if results['errors']:
            print(f"\nErrors encountered: {len(results['errors'])}")
            for error in results['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
        
        print("\nMigration completed!")
    
    asyncio.run(main())
