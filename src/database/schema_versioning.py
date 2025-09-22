"""
Database Schema Versioning System

Manages database schema versions and migrations.
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseSchemaVersioning:
    """Manages database schema versions and migrations."""
    
    def __init__(self, db_path: str = "./tabula_rasa.db"):
        self.db_path = db_path
        self.current_version = None
        self.target_version = "1.0.0"
        
    def get_current_version(self) -> Optional[str]:
        """Get the current database schema version."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT version FROM schema_version ORDER BY created_at DESC LIMIT 1")
                result = cursor.fetchone()
                if result:
                    self.current_version = result[0]
                    return self.current_version
                else:
                    # No version table exists - this is a new database
                    return None
        except Exception as e:
            logger.error(f"Error getting current version: {e}")
            return None
    
    def create_version_table(self):
        """Create the schema version tracking table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version TEXT NOT NULL,
                        description TEXT,
                        migration_script TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                logger.info("Schema version table created")
        except Exception as e:
            logger.error(f"Error creating version table: {e}")
    
    def record_version(self, version: str, description: str = "", migration_script: str = ""):
        """Record a new schema version."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO schema_version (version, description, migration_script)
                    VALUES (?, ?, ?)
                """, (version, description, migration_script))
                conn.commit()
                self.current_version = version
                logger.info(f"Recorded schema version: {version}")
        except Exception as e:
            logger.error(f"Error recording version: {e}")
    
    def check_schema_consistency(self) -> Dict[str, Any]:
        """Check if the database schema is consistent with expected version."""
        try:
            current_version = self.get_current_version()
            
            if current_version is None:
                # New database - needs initialization
                return {
                    "status": "needs_initialization",
                    "current_version": None,
                    "target_version": self.target_version,
                    "message": "Database needs schema initialization"
                }
            elif current_version == self.target_version:
                # Schema is up to date
                return {
                    "status": "up_to_date",
                    "current_version": current_version,
                    "target_version": self.target_version,
                    "message": "Schema is up to date"
                }
            else:
                # Schema needs migration
                return {
                    "status": "needs_migration",
                    "current_version": current_version,
                    "target_version": self.target_version,
                    "message": f"Schema needs migration from {current_version} to {self.target_version}"
                }
                
        except Exception as e:
            logger.error(f"Error checking schema consistency: {e}")
            return {
                "status": "error",
                "current_version": None,
                "target_version": self.target_version,
                "message": f"Error checking schema: {e}"
            }
    
    def initialize_schema(self):
        """Initialize the database schema for a new database."""
        try:
            # Create version table first
            self.create_version_table()
            
            # Record initial version
            self.record_version(
                version=self.target_version,
                description="Initial schema version",
                migration_script="Initial database creation"
            )
            
            logger.info(f"Schema initialized with version {self.target_version}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing schema: {e}")
            return False
    
    def get_schema_history(self) -> List[Dict[str, Any]]:
        """Get the complete schema version history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT version, description, migration_script, created_at
                    FROM schema_version
                    ORDER BY created_at ASC
                """)
                
                history = []
                for row in cursor.fetchall():
                    history.append({
                        "version": row[0],
                        "description": row[1],
                        "migration_script": row[2],
                        "created_at": row[3]
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"Error getting schema history: {e}")
            return []
    
    def validate_schema_integrity(self) -> Dict[str, Any]:
        """Validate that all expected tables exist and have correct structure."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get all tables
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                # Expected tables (from schema.sql)
                expected_tables = [
                    "sessions", "game_results", "action_traces", "coordinate_intelligence",
                    "learned_patterns", "system_logs", "error_logs", "performance_metrics",
                    "coordinate_penalties", "penalty_decay_config", "failure_learning",
                    "visual_targets", "stagnation_events", "winning_strategies",
                    "exploration_phases", "emergency_overrides", "frame_change_analysis",
                    "action_effectiveness_detailed", "governor_decisions", "score_history",
                    "frame_tracking", "performance_history", "session_history",
                    "action_tracking", "gan_model_checkpoints", "gan_pattern_learning",
                    "gan_reverse_engineering", "gan_validation_results", "gan_performance_metrics",
                    "button_priorities", "strategy_refinements", "strategy_replications",
                    "schema_version"  # This should exist if versioning is working
                ]
                
                missing_tables = [table for table in expected_tables if table not in tables]
                extra_tables = [table for table in tables if table not in expected_tables]
                
                return {
                    "status": "valid" if not missing_tables else "invalid",
                    "total_tables": len(tables),
                    "expected_tables": len(expected_tables),
                    "missing_tables": missing_tables,
                    "extra_tables": extra_tables,
                    "integrity_score": (len(expected_tables) - len(missing_tables)) / len(expected_tables)
                }
                
        except Exception as e:
            logger.error(f"Error validating schema integrity: {e}")
            return {
                "status": "error",
                "message": f"Error validating schema: {e}"
            }

# Global schema versioning instance
schema_versioning = DatabaseSchemaVersioning()

def get_schema_status() -> Dict[str, Any]:
    """Get current schema status."""
    return schema_versioning.check_schema_consistency()

def initialize_database_schema() -> bool:
    """Initialize database schema versioning."""
    return schema_versioning.initialize_schema()

def validate_database_schema() -> Dict[str, Any]:
    """Validate database schema integrity."""
    return schema_versioning.validate_schema_integrity()

if __name__ == "__main__":
    # Test schema versioning
    versioning = DatabaseSchemaVersioning()
    
    print("🔍 Checking schema status...")
    status = versioning.check_schema_consistency()
    print(f"Schema Status: {status}")
    
    if status["status"] == "needs_initialization":
        print("🔧 Initializing schema...")
        versioning.initialize_schema()
    
    print("✅ Schema validation complete")
