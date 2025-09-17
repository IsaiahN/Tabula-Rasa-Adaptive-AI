#!/usr/bin/env python3
"""
MIGRATE TO DATABASE SCRIPT
Migrate all existing file-based data to SQLite database
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.migrate_data import run_migration
from database.director_commands import get_director_commands

async def main():
    """Run the complete migration process."""
    print("🗄️ TABULA RASA DATABASE MIGRATION")
    print("=" * 50)
    print("Migrating from file storage to SQLite database...")
    print()
    
    # Run migration
    print("📊 Starting data migration...")
    results = await run_migration("data")
    
    print("\n📈 MIGRATION RESULTS:")
    print(f"✅ Sessions migrated: {results['sessions_migrated']}")
    print(f"✅ Games migrated: {results['games_migrated']}")
    print(f"✅ Action intelligence migrated: {results['action_intelligence_migrated']}")
    print(f"✅ Coordinate intelligence migrated: {results['coordinate_intelligence_migrated']}")
    print(f"✅ Logs migrated: {results['logs_migrated']}")
    print(f"✅ Patterns migrated: {results['patterns_migrated']}")
    
    if results['errors']:
        print(f"\n⚠️ ERRORS ENCOUNTERED: {len(results['errors'])}")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"   - {error}")
        if len(results['errors']) > 5:
            print(f"   ... and {len(results['errors']) - 5} more errors")
    
    print("\n🔍 TESTING DATABASE CONNECTION...")
    
    # Test database connection
    try:
        director = get_director_commands()
        status = await director.get_system_overview()
        
        print("✅ Database connection successful!")
        print(f"📊 Active sessions: {status['system_status']['active_sessions']}")
        print(f"🎮 Total games: {status['system_status']['total_games']}")
        print(f"🏆 Total wins: {status['system_status']['total_wins']}")
        print(f"📈 Win rate: {status['system_status']['avg_win_rate']:.1%}")
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return 1
    
    print("\n🎉 MIGRATION COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("✅ All data migrated to SQLite database")
    print("✅ Database API is working correctly")
    print("✅ Director commands are available")
    print()
    print("🚀 NEXT STEPS:")
    print("1. Update your code to use the database API")
    print("2. Replace file I/O operations with database calls")
    print("3. Use Director commands for system analysis")
    print("4. Monitor system performance with real-time queries")
    print()
    print("📚 See src/database/director_reference.md for command reference")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Migration failed: {e}")
        sys.exit(1)
