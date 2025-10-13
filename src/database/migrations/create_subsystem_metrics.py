"""
Database migration for subsystem metrics tables
"""

async def create_subsystem_metrics_tables(db):
    """Create required tables for subsystem metrics if they don't exist."""
    
    # Table for storing subsystem metrics
    await db.execute("""
        CREATE TABLE IF NOT EXISTS subsystem_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subsystem_id TEXT NOT NULL,
            metrics_data TEXT NOT NULL,
            collection_timestamp TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Table for storing subsystem configurations 
    await db.execute("""
        CREATE TABLE IF NOT EXISTS subsystem_configs (
            subsystem_id TEXT PRIMARY KEY,
            config_data TEXT NOT NULL,
            last_updated TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create index on subsystem_id for faster lookups
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_subsystem_metrics_id 
        ON subsystem_metrics(subsystem_id)
    """)
    
    # Create index on collection timestamp for faster historical queries
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_subsystem_metrics_timestamp
        ON subsystem_metrics(collection_timestamp)
    """)