"""
Migration Script: File Data to Database
Migrates all file-based data to database storage and removes files.
"""

import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, Any, List
import sqlite3

from .error_logging_manager import get_error_logging_manager
from .reward_cap_manager import get_reward_cap_manager
from .learned_patterns_manager import get_learned_patterns_manager
from .database_logging_handler import get_database_logging_manager

class FileDataMigrator:
    """
    Migrates file-based data to database storage.
    """
    
    def __init__(self, db_path: str = "tabula_rasa.db"):
        self.db_path = db_path
        self.error_manager = get_error_logging_manager()
        self.reward_cap_manager = get_reward_cap_manager()
        self.patterns_manager = get_learned_patterns_manager()
        self.logging_manager = get_database_logging_manager()
        
        # Files to migrate and remove
        self.files_to_migrate = {
            'data/reward_cap_config.json': self._migrate_reward_cap_config,
            'data/learned_patterns.pkl': self._migrate_learned_patterns,
            'data/logs/master_arc_trainer.log': self._migrate_log_file,
            'data/logs/master_arc_trainer_output.log': self._migrate_log_file,
        }
        
        # Directories to clean up
        self.directories_to_clean = [
            'data/logs',
            'data/patterns',
            'data/sessions',
            'data/training',
            'data/optimization',
            'data/performance',
            'data/architect',
            'data/governor',
            'data/memory',
            'data/learning',
            'data/curiosity',
            'data/energy',
            'data/action_intelligence',
            'data/coordinate_intelligence',
            'data/scorecard',
            'data/visual_analysis',
            'data/tree_evaluation',
            'data/space_time',
            'data/conscious_architecture',
            'data/behavioral_systems',
            'data/high_priority_enhancements',
        ]
    
    def migrate_all(self) -> Dict[str, Any]:
        """
        Migrate all file-based data to database.
        
        Returns:
            Dict with migration results
        """
        results = {
            'migrated_files': [],
            'removed_files': [],
            'cleaned_directories': [],
            'errors': [],
            'success': True
        }
        
        print("🔄 Starting file data migration to database...")
        
        # Migrate individual files
        for file_path, migrate_func in self.files_to_migrate.items():
            try:
                if Path(file_path).exists():
                    print(f"  📁 Migrating {file_path}...")
                    success = migrate_func(file_path)
                    if success:
                        results['migrated_files'].append(file_path)
                        # Remove the file after successful migration
                        Path(file_path).unlink()
                        results['removed_files'].append(file_path)
                        print(f"    ✅ Migrated and removed {file_path}")
                    else:
                        results['errors'].append(f"Failed to migrate {file_path}")
                        print(f"    ❌ Failed to migrate {file_path}")
                else:
                    print(f"  ⏭️  Skipping {file_path} (not found)")
            except Exception as e:
                error_msg = f"Error migrating {file_path}: {e}"
                results['errors'].append(error_msg)
                print(f"    ❌ {error_msg}")
        
        # Clean up empty directories
        for dir_path in self.directories_to_clean:
            try:
                if Path(dir_path).exists():
                    # Check if directory is empty
                    if not any(Path(dir_path).iterdir()):
                        Path(dir_path).rmdir()
                        results['cleaned_directories'].append(dir_path)
                        print(f"  🗑️  Removed empty directory {dir_path}")
                    else:
                        print(f"  ⚠️  Directory {dir_path} not empty, keeping it")
            except Exception as e:
                error_msg = f"Error cleaning directory {dir_path}: {e}"
                results['errors'].append(error_msg)
                print(f"    ❌ {error_msg}")
        
        # Update database schema
        self._update_database_schema()
        
        if results['errors']:
            results['success'] = False
            print(f"\n⚠️  Migration completed with {len(results['errors'])} errors")
        else:
            print(f"\n✅ Migration completed successfully!")
            print(f"   Migrated: {len(results['migrated_files'])} files")
            print(f"   Removed: {len(results['removed_files'])} files")
            print(f"   Cleaned: {len(results['cleaned_directories'])} directories")
        
        return results
    
    def _migrate_reward_cap_config(self, file_path: str) -> bool:
        """Migrate reward cap configuration."""
        return self.reward_cap_manager.migrate_from_file(file_path)
    
    def _migrate_learned_patterns(self, file_path: str) -> bool:
        """Migrate learned patterns."""
        return self.patterns_manager.migrate_from_pickle(file_path)
    
    def _migrate_log_file(self, file_path: str) -> bool:
        """Migrate log file to database."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Parse log lines and insert into database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        log_level TEXT NOT NULL,
                        logger_name TEXT,
                        message TEXT NOT NULL,
                        module TEXT,
                        function TEXT,
                        line_number INTEGER,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        session_id TEXT,
                        game_id TEXT,
                        metadata TEXT
                    )
                """)
                
                for line in lines:
                    if line.strip():
                        # Simple log line parsing (adjust as needed)
                        conn.execute("""
                            INSERT INTO system_logs 
                            (log_level, logger_name, message, timestamp)
                            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                        """, ('INFO', 'migrated', line.strip(),))
                
                conn.commit()
            
            return True
        except Exception as e:
            print(f"Error migrating log file {file_path}: {e}")
            return False
    
    def _update_database_schema(self):
        """Update database schema with new tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Read and execute the schema file
                schema_file = Path(__file__).parent / "performance_schema.sql"
                if schema_file.exists():
                    with open(schema_file, 'r') as f:
                        schema_sql = f.read()
                    
                    # Execute schema updates
                    conn.executescript(schema_sql)
                    conn.commit()
                    print("  📊 Database schema updated")
        except Exception as e:
            print(f"Error updating database schema: {e}")
    
    def verify_migration(self) -> Dict[str, Any]:
        """Verify that migration was successful."""
        verification = {
            'database_exists': Path(self.db_path).exists(),
            'tables_created': [],
            'data_migrated': {},
            'files_removed': [],
            'success': True
        }
        
        if not verification['database_exists']:
            verification['success'] = False
            return verification
        
        # Check if tables exist
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' 
                ORDER BY name
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            verification['tables_created'] = tables
            
            # Check data counts
            table_counts = {}
            for table in tables:
                try:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    table_counts[table] = count
                except:
                    table_counts[table] = 0
            
            verification['data_migrated'] = table_counts
        
        # Check if files were removed
        for file_path in self.files_to_migrate.keys():
            if not Path(file_path).exists():
                verification['files_removed'].append(file_path)
        
        return verification

def migrate_file_data_to_database():
    """Main function to migrate file data to database."""
    migrator = FileDataMigrator()
    
    print("🚀 Tabula Rasa File Data Migration")
    print("=" * 50)
    
    # Perform migration
    results = migrator.migrate_all()
    
    # Verify migration
    print("\n🔍 Verifying migration...")
    verification = migrator.verify_migration()
    
    # Print summary
    print("\n📊 Migration Summary:")
    print(f"   Database exists: {verification['database_exists']}")
    print(f"   Tables created: {len(verification['tables_created'])}")
    print(f"   Files migrated: {len(results['migrated_files'])}")
    print(f"   Files removed: {len(results['removed_files'])}")
    print(f"   Directories cleaned: {len(results['cleaned_directories'])}")
    
    if verification['data_migrated']:
        print("\n📈 Data in database:")
        for table, count in verification['data_migrated'].items():
            print(f"   {table}: {count} records")
    
    if results['errors']:
        print(f"\n⚠️  Errors encountered:")
        for error in results['errors']:
            print(f"   - {error}")
    
    return results, verification

if __name__ == "__main__":
    migrate_file_data_to_database()
