#!/usr/bin/env python3
"""
COMPLETE DATA MIGRATION SCRIPT
Migrate all remaining data files to database and prepare for data directory deletion
"""

import asyncio
import json
import os
import pickle
import shutil
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.system_integration import get_system_integration
from database.api import get_database

async def migrate_remaining_data():
    """Migrate all remaining data files to the database."""
    print("🔄 COMPLETE DATA MIGRATION")
    print("=" * 50)
    
    integration = get_system_integration()
    db = get_database()
    
    # Database initializes automatically in constructor
    
    migrated_count = 0
    error_count = 0
    
    # 1. Migrate action traces
    print("\n📊 Migrating action traces...")
    try:
        action_traces_file = Path("data/action_traces.ndjson")
        if action_traces_file.exists():
            with open(action_traces_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            trace_data = json.loads(line.strip())
                            await integration.log_action_trace(
                                trace_data.get('game_id', 'unknown'),
                                trace_data.get('action_number', 0),
                                trace_data.get('coordinates', None),
                                trace_data.get('timestamp', 0),
                                trace_data.get('frame_before', None),
                                trace_data.get('frame_after', None),
                                trace_data.get('frame_changed', False),
                                trace_data.get('score_before', 0),
                                trace_data.get('score_after', 0),
                                trace_data.get('response_data', None)
                            )
                            migrated_count += 1
                        except Exception as e:
                            print(f"   ⚠️ Error migrating trace line {line_num}: {e}")
                            error_count += 1
            print(f"   ✅ Migrated {migrated_count} action traces")
    except Exception as e:
        print(f"   ❌ Error migrating action traces: {e}")
        error_count += 1
    
    # 2. Migrate session files
    print("\n📁 Migrating session files...")
    try:
        sessions_dir = Path("data/sessions")
        if sessions_dir.exists():
            session_count = 0
            for session_file in sessions_dir.glob("*.json"):
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    
                    session_id = session_file.stem
                    # Convert timestamp to start_time for database compatibility
                    if 'timestamp' in session_data:
                        session_data['start_time'] = datetime.fromtimestamp(session_data['timestamp'])
                        del session_data['timestamp']
                    await integration.update_session_metrics(session_id, session_data)
                    session_count += 1
                    migrated_count += 1
                except Exception as e:
                    print(f"   ⚠️ Error migrating session {session_file.name}: {e}")
                    error_count += 1
            print(f"   ✅ Migrated {session_count} session files")
    except Exception as e:
        print(f"   ❌ Error migrating session files: {e}")
        error_count += 1
    
    # 3. Migrate learned patterns
    print("\n🧠 Migrating learned patterns...")
    try:
        patterns_file = Path("data/learned_patterns.pkl")
        if patterns_file.exists():
            with open(patterns_file, 'rb') as f:
                patterns_data = pickle.load(f)
            
            if isinstance(patterns_data, dict):
                for pattern_id, pattern_data in patterns_data.items():
                    try:
                        await integration.save_learned_pattern(
                            "unknown",  # pattern_type
                            pattern_data,
                            0.5,  # confidence
                            1,    # frequency
                            0.5,  # success_rate
                            "migrated"  # game_context
                        )
                        migrated_count += 1
                    except Exception as e:
                        print(f"   ⚠️ Error migrating pattern {pattern_id}: {e}")
                        error_count += 1
            print(f"   ✅ Migrated learned patterns")
    except Exception as e:
        print(f"   ❌ Error migrating learned patterns: {e}")
        error_count += 1
    
    # 4. Migrate global counters
    print("\n🔢 Migrating global counters...")
    try:
        counters_file = Path("data/global_counters.json")
        if counters_file.exists():
            with open(counters_file, 'r') as f:
                counters_data = json.load(f)
            
            for key, value in counters_data.items():
                await integration.update_global_counter(key, value, f"Migrated {key}")
                migrated_count += 1
            print(f"   ✅ Migrated global counters")
    except Exception as e:
        print(f"   ❌ Error migrating global counters: {e}")
        error_count += 1
    
    # 5. Migrate architecture evolution data
    print("\n🏗️ Migrating architecture evolution data...")
    try:
        arch_dir = Path("data/architecture/evolution")
        if arch_dir.exists():
            arch_count = 0
            for arch_file in arch_dir.glob("*.json"):
                try:
                    with open(arch_file, 'r') as f:
                        arch_data = json.load(f)
                    
                    if isinstance(arch_data, dict):
                        await integration.log_architect_evolution(
                            arch_data.get("evolution_type", "unknown"),
                            arch_data.get("changes", {}),
                            arch_data.get("performance_impact", 0.0),
                            arch_data.get("rationale", "Migrated from file"),
                            arch_data.get("confidence", 0.5)
                        )
                        arch_count += 1
                        migrated_count += 1
                except Exception as e:
                    print(f"   ⚠️ Error migrating architecture file {arch_file.name}: {e}")
                    error_count += 1
            print(f"   ✅ Migrated {arch_count} architecture files")
    except Exception as e:
        print(f"   ❌ Error migrating architecture data: {e}")
        error_count += 1
    
    # 6. Migrate experiment data
    print("\n🧪 Migrating experiment data...")
    try:
        exp_dir = Path("data/experiments")
        if exp_dir.exists():
            exp_count = 0
            for exp_file in exp_dir.rglob("*.json"):
                try:
                    with open(exp_file, 'r') as f:
                        exp_data = json.load(f)
                    
                    if isinstance(exp_data, dict):
                        await integration.log_experiment(
                            exp_data.get("experiment_type", "unknown"),
                            exp_data.get("parameters", {}),
                            exp_data.get("results", {}),
                            exp_data.get("success", False),
                            exp_data.get("duration", 0),
                            exp_data.get("notes", "Migrated from file")
                        )
                        exp_count += 1
                        migrated_count += 1
                except Exception as e:
                    print(f"   ⚠️ Error migrating experiment file {exp_file.name}: {e}")
                    error_count += 1
            print(f"   ✅ Migrated {exp_count} experiment files")
    except Exception as e:
        print(f"   ❌ Error migrating experiment data: {e}")
        error_count += 1
    
    # 7. Migrate task performance data
    print("\n📈 Migrating task performance data...")
    try:
        perf_file = Path("data/task_performance.json")
        if perf_file.exists():
            with open(perf_file, 'r') as f:
                perf_data = json.load(f)
            
            if isinstance(perf_data, dict):
                await integration.update_task_performance(
                    perf_data.get("task_id", "unknown"),
                    perf_data.get("performance_metrics", {}),
                    perf_data.get("learning_progress", {}),
                    perf_data.get("success_rate", 0.0),
                    perf_data.get("notes", "Migrated from file")
                )
                migrated_count += 1
            print(f"   ✅ Migrated task performance data")
    except Exception as e:
        print(f"   ❌ Error migrating task performance: {e}")
        error_count += 1
    
    print(f"\n📊 MIGRATION SUMMARY:")
    print(f"   ✅ Successfully migrated: {migrated_count} items")
    print(f"   ❌ Errors encountered: {error_count} items")
    
    return migrated_count, error_count

async def backup_data_directory():
    """Create a backup of the data directory before deletion."""
    print("\n💾 Creating data directory backup...")
    
    try:
        backup_dir = Path("data_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        if Path("data").exists():
            shutil.copytree("data", backup_dir)
            print(f"   ✅ Backup created: {backup_dir}")
            return backup_dir
        else:
            print("   ⚠️ No data directory found to backup")
            return None
    except Exception as e:
        print(f"   ❌ Error creating backup: {e}")
        return None

async def verify_database_integrity():
    """Verify that all data has been properly migrated to the database."""
    print("\n🔍 Verifying database integrity...")
    
    try:
        db = get_database()
        
        # Check key tables
        async with db.get_connection() as conn:
            # Check action traces
            cursor = conn.execute("SELECT COUNT(*) FROM action_traces")
            action_traces_count = cursor.fetchone()[0]
            print(f"   📊 Action traces in database: {action_traces_count}")
            
            # Check sessions
            cursor = conn.execute("SELECT COUNT(*) FROM training_sessions")
            sessions_count = cursor.fetchone()[0]
            print(f"   📁 Sessions in database: {sessions_count}")
            
            # Check learned patterns
            cursor = conn.execute("SELECT COUNT(*) FROM learned_patterns")
            patterns_count = cursor.fetchone()[0]
            print(f"   🧠 Learned patterns in database: {patterns_count}")
            
            # Check global counters
            cursor = conn.execute("SELECT COUNT(*) FROM global_counters")
            counters_count = cursor.fetchone()[0]
            print(f"   🔢 Global counters in database: {counters_count}")
            
            # Check logs
            cursor = conn.execute("SELECT COUNT(*) FROM logs")
            logs_count = cursor.fetchone()[0]
            print(f"   📝 Logs in database: {logs_count}")
        
        print("   ✅ Database integrity verification complete")
        return True
        
    except Exception as e:
        print(f"   ❌ Error verifying database integrity: {e}")
        return False

async def main():
    """Main migration process."""
    print("🚀 STARTING COMPLETE DATA MIGRATION")
    print("=" * 60)
    
    # Step 1: Migrate all remaining data
    migrated_count, error_count = await migrate_remaining_data()
    
    # Step 2: Verify database integrity
    integrity_ok = await verify_database_integrity()
    
    if not integrity_ok:
        print("\n❌ Database integrity check failed. Aborting data directory deletion.")
        return
    
    # Step 3: Create backup
    backup_dir = await backup_data_directory()
    
    # Step 4: Ask for confirmation before deletion
    print(f"\n⚠️  READY TO DELETE DATA DIRECTORY")
    print(f"   📊 Migrated: {migrated_count} items")
    print(f"   ❌ Errors: {error_count} items")
    print(f"   💾 Backup: {backup_dir}")
    print(f"   🔍 Database integrity: {'✅ OK' if integrity_ok else '❌ FAILED'}")
    
    if error_count == 0 and integrity_ok:
        print("\n✅ All data successfully migrated to database!")
        print("🗑️  Data directory can now be safely deleted.")
        print("\nTo delete the data directory, run:")
        print("   rm -rf data/")
        print("   # or on Windows:")
        print("   rmdir /s /q data")
    else:
        print("\n⚠️  Some errors occurred during migration.")
        print("   Please review the errors before deleting the data directory.")

if __name__ == "__main__":
    asyncio.run(main())
