"""
Database Cleanup Script

This script cleans the database by:
1. Identifying and keeping only successful level completion data
2. Removing stale or irrelevant data
3. Maintaining referential integrity
4. Keeping only the most valuable training data
"""

import sqlite3
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def configure_logging():
    """Configure logging for database cleanup."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

class DatabaseError(Exception):
    """Database operation error."""
    pass

class DatabaseConnectionError(DatabaseError):
    """Database connection error."""
    pass

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseCleaner:
    """Handles intelligent database cleanup operations."""

    def __init__(self, db_path: str):
        """Initialize the database cleaner.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._cursor: Optional[sqlite3.Cursor] = None
        self.stats = {
            'rows_deleted': 0,
            'rows_kept': 0,
            'tables_cleaned': 0,
            'space_freed': 0
        }
        self.connect()
        
    @property
    def conn(self) -> sqlite3.Connection:
        """Database connection property."""
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn
        
    @property
    def cursor(self) -> sqlite3.Cursor:
        """Database cursor property."""
        if self._cursor is None:
            self.connect()
        assert self._cursor is not None
        return self._cursor
    
    def connect(self):
        """Establish database connection."""
        if not os.path.exists(self.db_path):
            raise DatabaseConnectionError(f"Database file not found: {self.db_path}")
            
        try:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._cursor = self._conn.cursor()
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            self._conn = None
            self._cursor = None
            raise DatabaseConnectionError(f"Could not connect to database: {e}")

    def run_cleanup(self):
        """Execute the main cleanup operation."""
        try:
            if self._conn is None or self._cursor is None:
                self.connect()
            
            # 1. First identify successful level completions
            successful_game_ids = self._get_successful_game_ids()
            if not successful_game_ids:
                logger.warning("No successful game completions found")
                return
                
            logger.info(f"Found {len(successful_game_ids)} successful game completions")
            
            # Start transaction for main cleanup
            self.cursor.execute("BEGIN TRANSACTION")
            
            # 2. Clean up various tables while maintaining references
            self._cleanup_training_data(successful_game_ids)
            self._cleanup_action_data(successful_game_ids)
            self._cleanup_coordinate_data(successful_game_ids)
            
            # 3. Remove orphaned data
            self._cleanup_orphaned_data()
            
            # 4. Commit transaction
            self.conn.commit()
            
            # 5. Close connection before vacuum
            if self._conn is not None:
                self._conn.close()
                self._conn = None
                self._cursor = None
            
            # 6. Reconnect and vacuum
            self.connect()
            self.cursor.execute("VACUUM")
            
            logger.info("Database cleanup completed successfully")
            self._log_cleanup_stats()
            
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")
            if self._conn:
                self._conn.rollback()
            raise
        finally:
            if self._conn:
                try:
                    self._conn.close()
                    self._conn = None
                    self._cursor = None
                except Exception as e:
                    logger.error(f"Error closing database connection: {e}")

    def _ensure_connected(self):
        """Ensure database connection is active."""
        if self._conn is None or self._cursor is None:
            self.connect()

    def _get_successful_game_ids(self) -> List[str]:
        """Get IDs of games that were successfully completed."""
        self._ensure_connected()
        query = """
        SELECT DISTINCT game_id
        FROM winning_sequences 
        WHERE avg_score > 0 
           OR success_rate > 0.5
           OR frequency > 1
        """
        
        try:
            self.cursor.execute(query)
            return [row['game_id'] for row in self.cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting successful game IDs: {e}")
            return []

    def _cleanup_training_data(self, successful_game_ids: List[str]):
        """Clean up training-related tables."""
        self._ensure_connected()
        tables = [
            ('strategy_refinements', 'refinement_timestamp'),
            ('strategy_replications', 'replication_timestamp')
        ]
        
        for table, timestamp_field in tables:
            try:
                before_count = self._get_table_count(table)
                
                query = f"""
                DELETE FROM {table}
                WHERE game_id NOT IN ({','.join('?' for _ in successful_game_ids)})
                AND datetime({timestamp_field}) < datetime('now', '-30 days')
                """
                
                self.cursor.execute(query, successful_game_ids)
                after_count = self._get_table_count(table)
                
                deleted = before_count - after_count
                self.stats['rows_deleted'] += deleted
                self.stats['rows_kept'] += after_count
                self.stats['tables_cleaned'] += 1
                
                logger.info(f"Cleaned {table}: removed {deleted} rows, kept {after_count} rows")
                
            except Exception as e:
                logger.error(f"Error cleaning {table}: {e}")

    def _cleanup_action_data(self, successful_game_ids: List[str]):
        """Clean up action-related tables."""
        self._ensure_connected()
        
        # Clean action_effectiveness
        try:
            before_count = self._get_table_count('action_effectiveness')
            query = f"""
            DELETE FROM action_effectiveness
            WHERE game_id NOT IN ({','.join('?' for _ in successful_game_ids)})
            AND (
                success_rate < 0.5
                OR datetime(last_used) < datetime('now', '-15 days')
            )
            """
            self.cursor.execute(query, successful_game_ids)
            after_count = self._get_table_count('action_effectiveness')
            deleted = before_count - after_count
            self.stats['rows_deleted'] += deleted
            self.stats['rows_kept'] += after_count
            self.stats['tables_cleaned'] += 1
            logger.info(f"Cleaned action_effectiveness: removed {deleted} rows, kept {after_count} rows")
        except Exception as e:
            logger.error(f"Error cleaning action_effectiveness: {e}")
            
        # Clean action_effectiveness_detailed
        try:
            before_count = self._get_table_count('action_effectiveness_detailed')
            query = f"""
            DELETE FROM action_effectiveness_detailed
            WHERE game_id NOT IN ({','.join('?' for _ in successful_game_ids)})
            AND (
                success_rate < 0.5
                OR datetime(last_updated) < datetime('now', '-15 days')
            )
            """
            self.cursor.execute(query, successful_game_ids)
            after_count = self._get_table_count('action_effectiveness_detailed')
            deleted = before_count - after_count
            self.stats['rows_deleted'] += deleted
            self.stats['rows_kept'] += after_count
            self.stats['tables_cleaned'] += 1
            logger.info(f"Cleaned action_effectiveness_detailed: removed {deleted} rows, kept {after_count} rows")
        except Exception as e:
            logger.error(f"Error cleaning action_effectiveness_detailed: {e}")
            
        # Clean winning_sequences
        try:
            before_count = self._get_table_count('winning_sequences')
            query = f"""
            DELETE FROM winning_sequences
            WHERE game_id NOT IN ({','.join('?' for _ in successful_game_ids)})
            AND (
                (success_rate < 0.5 AND avg_score = 0)
                OR datetime(last_used) < datetime('now', '-15 days')
            )
            """
            self.cursor.execute(query, successful_game_ids)
            after_count = self._get_table_count('winning_sequences')
            deleted = before_count - after_count
            self.stats['rows_deleted'] += deleted
            self.stats['rows_kept'] += after_count
            self.stats['tables_cleaned'] += 1
            logger.info(f"Cleaned winning_sequences: removed {deleted} rows, kept {after_count} rows")
        except Exception as e:
            logger.error(f"Error cleaning winning_sequences: {e}")
            
        # Clean button_priorities
        try:
            before_count = self._get_table_count('button_priorities')
            query = f"""
            DELETE FROM button_priorities 
            WHERE game_type NOT IN ({','.join('?' for _ in successful_game_ids)})
            AND (
                confidence < 0.5
                OR datetime(last_used) < datetime('now', '-15 days')
            )
            """
            self.cursor.execute(query, successful_game_ids)
            after_count = self._get_table_count('button_priorities')
            deleted = before_count - after_count
            self.stats['rows_deleted'] += deleted
            self.stats['rows_kept'] += after_count
            self.stats['tables_cleaned'] += 1
            logger.info(f"Cleaned button_priorities: removed {deleted} rows, kept {after_count} rows")
        except Exception as e:
            logger.error(f"Error cleaning button_priorities: {e}")

    def _cleanup_coordinate_data(self, successful_game_ids: List[str]):
        """Clean up coordinate intelligence and related data."""
        self._ensure_connected()
        try:
            before_count = self._get_table_count('coordinate_intelligence')
            
            # Keep successful data and high success rates
            query = f"""
            DELETE FROM coordinate_intelligence
            WHERE game_id NOT IN ({','.join('?' for _ in successful_game_ids)})
            AND (
                success_rate < 0.4
                OR datetime(last_used) < datetime('now', '-15 days')
            )
            """
            
            self.cursor.execute(query, successful_game_ids)
            after_count = self._get_table_count('coordinate_intelligence')
            
            deleted = before_count - after_count
            self.stats['rows_deleted'] += deleted
            self.stats['rows_kept'] += after_count
            self.stats['tables_cleaned'] += 1
            
            logger.info(f"Cleaned coordinate_intelligence: removed {deleted} rows, kept {after_count} rows")
            
        except Exception as e:
            logger.error(f"Error cleaning coordinate_intelligence: {e}")

    def cleanup(self):
        """Execute the main cleanup operation."""
        try:
            if self._conn is None or self._cursor is None:
                self.connect()
            
            self._cursor.execute("BEGIN TRANSACTION")
            
            # 1. First identify successful level completions
            successful_game_ids = self._get_successful_game_ids()
            if not successful_game_ids:
                logger.warning("No successful game completions found")
                return
                
            logger.info(f"Found {len(successful_game_ids)} successful game completions")
            
            # 2. Clean up various tables while maintaining references
            self._cleanup_training_data(successful_game_ids)
            self._cleanup_action_data(successful_game_ids)
            self._cleanup_coordinate_data(successful_game_ids)
            
            # 3. Remove orphaned data
            self._cleanup_orphaned_data()
            
            # 4. Vacuum database to reclaim space
            self._vacuum_database()
            
            # Commit transaction
            self._conn.commit()
            
            logger.info("Database cleanup completed successfully")
            self._log_cleanup_stats()
            
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")
            if self._conn:
                self._conn.rollback()
            raise
        finally:
            if self._conn:
                try:
                    self._conn.close()
                    self._conn = None
                    self._cursor = None
                except Exception as e:
                    logger.error(f"Error closing database connection: {e}")
        
        for table in tables:
            try:
                before_count = self._get_table_count(table)
                
                # Keep successful game analysis and recent useful data
                query = f"""
                DELETE FROM {table}
                WHERE game_id NOT IN ({','.join('?' for _ in successful_game_ids)})
                AND (
                    datetime(timestamp) < datetime('now', '-30 days')
                    OR confidence < 0.4
                    OR effectiveness < 0.3
                )
                """
                
                self._cursor.execute(query, successful_game_ids)
                after_count = self._get_table_count(table)
                
                deleted = before_count - after_count
                self.stats['rows_deleted'] += deleted
                self.stats['rows_kept'] += after_count
                self.stats['tables_cleaned'] += 1
                
                logger.info(f"Cleaned {table}: removed {deleted} rows, kept {after_count} rows")
                
            except Exception as e:
                logger.error(f"Error cleaning {table}: {e}")

    def _cleanup_orphaned_data(self):
        """Remove orphaned data that has no references."""
        self._ensure_connected()
        # Get all tables with game_id column
        tables_with_game_id = [
            'action_effectiveness',
            'action_effectiveness_detailed', 
            'winning_sequences',
            'coordinate_intelligence',
            'strategy_replications'
        ]
        
        # Get successful games for reference
        successful_game_ids = self._get_successful_game_ids()
        if not successful_game_ids:
            logger.warning("No reference successful games found for orphan cleanup")
            return
            
        for table in tables_with_game_id:
            try:
                before_count = self._get_table_count(table)
                
                # Remove rows where game_id doesn't exist in winning games
                query = f"""
                DELETE FROM {table}
                WHERE game_id NOT IN (
                    SELECT DISTINCT game_id 
                    FROM winning_sequences
                    WHERE avg_score > 0 OR success_rate > 0.5
                )
                """
                
                self.cursor.execute(query)
                after_count = self._get_table_count(table)
                
                deleted = before_count - after_count
                self.stats['rows_deleted'] += deleted
                
                if deleted > 0:
                    logger.info(f"Removed {deleted} orphaned rows from {table}")
                
            except Exception as e:
                logger.error(f"Error cleaning orphaned data from {table}: {e}")

    def cleanup(self):
        """Execute the main cleanup operation."""
        try:
            if self._conn is None or self._cursor is None:
                self.connect()
            
            # 1. First identify successful level completions
            successful_game_ids = self._get_successful_game_ids()
            if not successful_game_ids:
                logger.warning("No successful game completions found")
                return
                
            logger.info(f"Found {len(successful_game_ids)} successful game completions")
            
            # Start transaction for main cleanup
            self._cursor.execute("BEGIN TRANSACTION")
            
            # 2. Clean up various tables while maintaining references
            self._cleanup_training_data(successful_game_ids)
            self._cleanup_action_data(successful_game_ids)
            self._cleanup_coordinate_data(successful_game_ids)
            
            # 3. Remove orphaned data
            self._cleanup_orphaned_data()
            
            # 4. Commit transaction
            self._conn.commit()
            
            # 5. Vacuum database to reclaim space (must be outside transaction)
            self._vacuum_database()
            
            logger.info("Database cleanup completed successfully")
            self._log_cleanup_stats()
            
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")
            if self._conn:
                self._conn.rollback()
            raise
        finally:
            if self._conn:
                try:
                    self._conn.close()
                    self._conn = None
                    self._cursor = None
                except Exception as e:
                    logger.error(f"Error closing database connection: {e}")

    def _vacuum_database(self):
        """Vacuum the database to reclaim space."""
        self._ensure_connected()
        try:
            # Get size before vacuum
            size_before = os.path.getsize(self.db_path)
            
            # Vacuum database
            self._cursor.execute("VACUUM")
            
            # Get size after vacuum
            size_after = os.path.getsize(self.db_path)
            
            space_freed = size_before - size_after
            self.stats['space_freed'] = space_freed
            
            logger.info(f"Database vacuum complete: freed {space_freed / 1024:.2f} KB")
            
        except Exception as e:
            logger.error(f"Error during database vacuum: {e}")

    def _get_table_count(self, table: str) -> int:
        """Get the number of rows in a table."""
        self._ensure_connected()
        try:
            self.cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            return self.cursor.fetchone()['count']
        except Exception as e:
            logger.error(f"Error getting count for table {table}: {e}")
            return 0

    def _log_cleanup_stats(self):
        """Log final cleanup statistics."""
        logger.info("=== Database Cleanup Statistics ===")
        logger.info(f"Total rows deleted: {self.stats['rows_deleted']}")
        logger.info(f"Total rows kept: {self.stats['rows_kept']}")
        logger.info(f"Tables cleaned: {self.stats['tables_cleaned']}")
        logger.info(f"Space freed: {self.stats['space_freed'] / 1024:.2f} KB")
        logger.info("================================")


def main():
    """Main execution function."""
    configure_logging()
    logger.info("Starting database cleanup")
    
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          'tabula_rasa.db')
    
    try:
        cleaner = DatabaseCleaner(db_path)
        cleaner.run_cleanup()
        logger.info("Database cleanup completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Database cleanup failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())