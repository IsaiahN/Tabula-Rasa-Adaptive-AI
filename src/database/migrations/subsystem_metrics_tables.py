"""Create subsystem metrics tables migration"""

def create_subsystem_metrics_tables(conn):
    """Create subsystem metrics tables."""
    
    # Table for storing subsystem metrics
    conn.execute("""
        CREATE TABLE IF NOT EXISTS subsystem_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subsystem_id TEXT NOT NULL,
            metrics_data TEXT NOT NULL,
            collection_timestamp TEXT NOT NULL,
            health_score REAL NOT NULL,
            performance_score REAL NOT NULL,
            efficiency_score REAL NOT NULL,
            error_count INTEGER NOT NULL DEFAULT 0,
            warning_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Table for storing subsystem configurations
    conn.execute("""
        CREATE TABLE IF NOT EXISTS subsystem_configs (
            subsystem_id TEXT PRIMARY KEY,
            config_data TEXT NOT NULL,
            last_updated TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indices for faster lookups
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_subsystem_metrics_id 
        ON subsystem_metrics(subsystem_id)
    """)
    
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_subsystem_metrics_timestamp
        ON subsystem_metrics(collection_timestamp)
    """)