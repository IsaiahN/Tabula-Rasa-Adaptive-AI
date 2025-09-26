"""
Advanced Hot/Warm/Cold Data Partitioning System
Optimizes query performance by segregating data based on access patterns
"""

import sqlite3
import asyncio
import logging
import time
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class PartitionConfig:
    """Configuration for data partitioning"""
    table_name: str
    timestamp_column: str
    hot_days: int       # Data accessed in last N days
    warm_days: int      # Data accessed in last N days (but not hot)
    partition_size_mb: int  # Max size before creating new partition
    enabled: bool = True

@dataclass
class PartitionStats:
    """Statistics for partition operations"""
    hot_records: int
    warm_records: int
    cold_records: int
    hot_size_mb: float
    warm_size_mb: float
    cold_size_mb: float
    partitioning_time_ms: float
    query_performance_improvement: float

class DataPartitioningSystem:
    """Advanced data partitioning system for optimal query performance"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.hot_db_path = db_path.replace('.db', '_hot.db')
        self.warm_db_path = db_path.replace('.db', '_warm.db')
        self.cold_db_path = db_path.replace('.db', '_cold.db')

        self.partition_configs = self._define_partition_configs()
        self.stats_history = []

    def _define_partition_configs(self) -> List[PartitionConfig]:
        """Define partitioning strategies for different tables"""
        return [
            # High-frequency operational data - aggressive hot partitioning
            PartitionConfig("action_traces", "timestamp", hot_days=1, warm_days=7, partition_size_mb=50),
            PartitionConfig("system_logs", "timestamp", hot_days=1, warm_days=3, partition_size_mb=20),
            PartitionConfig("frame_tracking", "timestamp", hot_days=2, warm_days=14, partition_size_mb=30),

            # Learning data - moderate partitioning
            PartitionConfig("coordinate_intelligence", "last_used", hot_days=7, warm_days=30, partition_size_mb=40),
            PartitionConfig("coordinate_penalties", "last_penalty_applied", hot_days=3, warm_days=14, partition_size_mb=25),
            PartitionConfig("performance_history", "timestamp", hot_days=7, warm_days=30, partition_size_mb=35),

            # Training session data - balanced partitioning
            PartitionConfig("training_sessions", "start_time", hot_days=3, warm_days=21, partition_size_mb=50),
            PartitionConfig("game_results", "start_time", hot_days=3, warm_days=21, partition_size_mb=45),

            # GAN system data - size-based partitioning
            PartitionConfig("gan_training_data", "timestamp", hot_days=1, warm_days=7, partition_size_mb=100),
            PartitionConfig("gan_generated_states", "created_at", hot_days=2, warm_days=14, partition_size_mb=75),

            # Analytics data - cold-optimized partitioning
            PartitionConfig("error_logs", "last_seen", hot_days=1, warm_days=7, partition_size_mb=30),
            PartitionConfig("stagnation_events", "detection_timestamp", hot_days=7, warm_days=30, partition_size_mb=25),
        ]

    async def initialize_partition_databases(self):
        """Initialize separate databases for hot, warm, and cold data"""
        logger.info("🔄 Initializing data partition databases...")

        # Create database files if they don't exist
        for db_path in [self.hot_db_path, self.warm_db_path, self.cold_db_path]:
            if not Path(db_path).exists():
                # Copy schema from main database
                await self._copy_database_schema(self.db_path, db_path)

        logger.info(f"✅ Partition databases initialized:")
        logger.info(f"   🔥 Hot:  {self.hot_db_path}")
        logger.info(f"   🌡️  Warm: {self.warm_db_path}")
        logger.info(f"   ❄️  Cold: {self.cold_db_path}")

    async def _copy_database_schema(self, source_db: str, target_db: str):
        """Copy database schema to new partition database"""
        with sqlite3.connect(source_db) as source_conn:
            # Get schema
            cursor = source_conn.execute("""
                SELECT sql FROM sqlite_master
                WHERE type IN ('table', 'index', 'view', 'trigger')
                AND name NOT LIKE 'sqlite_%'
                ORDER BY type DESC
            """)
            schema_statements = [row[0] for row in cursor.fetchall() if row[0]]

        # Create target database with schema
        with sqlite3.connect(target_db) as target_conn:
            for statement in schema_statements:
                try:
                    target_conn.execute(statement)
                except Exception as e:
                    logger.debug(f"Schema copy warning: {e}")
            target_conn.commit()

    async def partition_all_tables(self) -> Dict[str, PartitionStats]:
        """Partition all configured tables into hot/warm/cold storage"""
        logger.info("🔄 Starting comprehensive data partitioning...")

        start_time = time.time()
        results = {}
        total_stats = PartitionStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

        for config in self.partition_configs:
            if not config.enabled:
                continue

            try:
                logger.info(f"📊 Partitioning {config.table_name}...")
                stats = await self._partition_table(config)
                results[config.table_name] = stats

                # Aggregate statistics
                total_stats.hot_records += stats.hot_records
                total_stats.warm_records += stats.warm_records
                total_stats.cold_records += stats.cold_records
                total_stats.hot_size_mb += stats.hot_size_mb
                total_stats.warm_size_mb += stats.warm_size_mb
                total_stats.cold_size_mb += stats.cold_size_mb

            except Exception as e:
                logger.error(f"Failed to partition {config.table_name}: {e}")
                results[config.table_name] = PartitionStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

        total_stats.partitioning_time_ms = (time.time() - start_time) * 1000

        logger.info("🎉 Data partitioning completed:")
        logger.info(f"   🔥 Hot: {total_stats.hot_records:,} records ({total_stats.hot_size_mb:.1f} MB)")
        logger.info(f"   🌡️  Warm: {total_stats.warm_records:,} records ({total_stats.warm_size_mb:.1f} MB)")
        logger.info(f"   ❄️  Cold: {total_stats.cold_records:,} records ({total_stats.cold_size_mb:.1f} MB)")
        logger.info(f"   ⏱️  Time: {total_stats.partitioning_time_ms:.0f}ms")

        results['_TOTAL'] = total_stats
        self.stats_history.append(total_stats)

        return results

    async def _partition_table(self, config: PartitionConfig) -> PartitionStats:
        """Partition a single table based on its configuration"""
        start_time = time.time()

        # Calculate date boundaries
        now = datetime.now()
        hot_cutoff = now - timedelta(days=config.hot_days)
        warm_cutoff = now - timedelta(days=config.warm_days)

        hot_cutoff_str = hot_cutoff.strftime('%Y-%m-%d %H:%M:%S')
        warm_cutoff_str = warm_cutoff.strftime('%Y-%m-%d %H:%M:%S')

        # Check if table exists
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name=?
            """, (config.table_name,))
            if not cursor.fetchone():
                logger.warning(f"Table {config.table_name} not found")
                return PartitionStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Get table structure
        table_columns = await self._get_table_columns(config.table_name)
        if not table_columns:
            return PartitionStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Move hot data
        hot_stats = await self._move_data_to_partition(
            config.table_name, config.timestamp_column,
            f"{config.timestamp_column} >= '{hot_cutoff_str}'",
            self.hot_db_path, table_columns
        )

        # Move warm data
        warm_stats = await self._move_data_to_partition(
            config.table_name, config.timestamp_column,
            f"{config.timestamp_column} >= '{warm_cutoff_str}' AND {config.timestamp_column} < '{hot_cutoff_str}'",
            self.warm_db_path, table_columns
        )

        # Move cold data
        cold_stats = await self._move_data_to_partition(
            config.table_name, config.timestamp_column,
            f"{config.timestamp_column} < '{warm_cutoff_str}'",
            self.cold_db_path, table_columns
        )

        execution_time = (time.time() - start_time) * 1000

        return PartitionStats(
            hot_records=hot_stats['records'],
            warm_records=warm_stats['records'],
            cold_records=cold_stats['records'],
            hot_size_mb=hot_stats['size_mb'],
            warm_size_mb=warm_stats['size_mb'],
            cold_size_mb=cold_stats['size_mb'],
            partitioning_time_ms=execution_time,
            query_performance_improvement=0.0  # Will be calculated later
        )

    async def _get_table_columns(self, table_name: str) -> List[str]:
        """Get column names for a table"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"PRAGMA table_info({table_name})")
            return [row[1] for row in cursor.fetchall()]

    async def _move_data_to_partition(self, table_name: str, timestamp_column: str,
                                     condition: str, target_db: str, columns: List[str]) -> Dict[str, Any]:
        """Move data matching condition to target partition database"""

        # First, copy data to target partition
        with sqlite3.connect(self.db_path) as source_conn:
            # Get data to move
            select_sql = f"SELECT * FROM {table_name} WHERE {condition}"
            cursor = source_conn.execute(select_sql)
            rows_to_move = cursor.fetchall()

            if not rows_to_move:
                return {'records': 0, 'size_mb': 0.0}

        # Insert into target partition
        with sqlite3.connect(target_db) as target_conn:
            # Prepare INSERT statement
            placeholders = ', '.join(['?' for _ in columns])
            insert_sql = f"INSERT OR REPLACE INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

            # Insert data in batches
            batch_size = 1000
            for i in range(0, len(rows_to_move), batch_size):
                batch = rows_to_move[i:i + batch_size]
                target_conn.executemany(insert_sql, batch)

            target_conn.commit()

        # Delete from source after successful copy
        with sqlite3.connect(self.db_path) as source_conn:
            delete_sql = f"DELETE FROM {table_name} WHERE {condition}"
            cursor = source_conn.execute(delete_sql)
            deleted_count = cursor.rowcount
            source_conn.commit()

        # Estimate size (rough calculation)
        estimated_size_mb = (len(rows_to_move) * len(columns) * 50) / (1024 * 1024)  # Rough estimate

        return {
            'records': deleted_count,
            'size_mb': estimated_size_mb
        }

    async def create_unified_view(self, table_name: str):
        """Create a unified view that queries across all partitions"""
        logger.info(f"🔗 Creating unified view for {table_name}...")

        # Get table columns
        columns = await self._get_table_columns(table_name)
        if not columns:
            return

        columns_sql = ', '.join(columns)

        # Create view that unions all partitions
        view_sql = f"""
        CREATE VIEW IF NOT EXISTS {table_name}_unified AS
        SELECT {columns_sql}, 'hot' as partition_type FROM hot_{table_name}
        UNION ALL
        SELECT {columns_sql}, 'warm' as partition_type FROM warm_{table_name}
        UNION ALL
        SELECT {columns_sql}, 'cold' as partition_type FROM cold_{table_name}
        """

        # Create the view in main database
        with sqlite3.connect(self.db_path) as conn:
            # First attach partition databases
            conn.execute(f"ATTACH DATABASE '{self.hot_db_path}' AS hot")
            conn.execute(f"ATTACH DATABASE '{self.warm_db_path}' AS warm")
            conn.execute(f"ATTACH DATABASE '{self.cold_db_path}' AS cold")

            # Create the unified view
            conn.execute(view_sql)
            conn.commit()

            # Detach databases
            conn.execute("DETACH DATABASE hot")
            conn.execute("DETACH DATABASE warm")
            conn.execute("DETACH DATABASE cold")

        logger.info(f"✅ Unified view {table_name}_unified created")

    class PartitionedQueryExecutor:
        """Smart query executor that routes queries to appropriate partitions"""

        def __init__(self, partitioning_system):
            self.system = partitioning_system

        async def execute_query(self, query: str, params: tuple = ()) -> List[Tuple]:
            """Execute query with automatic partition routing"""
            # Analyze query to determine optimal partition strategy
            query_upper = query.upper()

            # Check if query has date/time filters
            if self._has_recent_date_filter(query):
                # Query recent data - start with hot partition
                return await self._execute_with_partition_priority(
                    query, params, ['hot', 'warm', 'cold']
                )
            elif self._has_old_date_filter(query):
                # Query old data - start with cold partition
                return await self._execute_with_partition_priority(
                    query, params, ['cold', 'warm', 'hot']
                )
            else:
                # No date filter - query all partitions
                return await self._execute_unified_query(query, params)

        def _has_recent_date_filter(self, query: str) -> bool:
            """Check if query filters for recent data"""
            query_lower = query.lower()
            recent_keywords = ['last', 'recent', 'current', 'today', 'yesterday']
            return any(keyword in query_lower for keyword in recent_keywords)

        def _has_old_date_filter(self, query: str) -> bool:
            """Check if query filters for old data"""
            query_lower = query.lower()
            old_keywords = ['archive', 'historical', 'old', 'before']
            return any(keyword in query_lower for keyword in old_keywords)

        async def _execute_with_partition_priority(self, query: str, params: tuple,
                                                  partition_order: List[str]) -> List[Tuple]:
            """Execute query with partition priority order"""
            db_map = {
                'hot': self.system.hot_db_path,
                'warm': self.system.warm_db_path,
                'cold': self.system.cold_db_path
            }

            results = []
            for partition in partition_order:
                db_path = db_map[partition]
                try:
                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.execute(query, params)
                        partition_results = cursor.fetchall()
                        results.extend(partition_results)

                        # If we got results and query has LIMIT, we might be done
                        if partition_results and 'LIMIT' in query.upper():
                            break

                except Exception as e:
                    logger.debug(f"Query failed on {partition} partition: {e}")

            return results

        async def _execute_unified_query(self, query: str, params: tuple) -> List[Tuple]:
            """Execute query across all partitions using unified view"""
            # This would require modifying the query to use unified views
            # For now, execute on all partitions and combine results
            all_results = []

            for db_path in [self.system.hot_db_path, self.system.warm_db_path, self.system.cold_db_path]:
                try:
                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.execute(query, params)
                        results = cursor.fetchall()
                        all_results.extend(results)
                except Exception as e:
                    logger.debug(f"Query failed on partition {db_path}: {e}")

            return all_results

    async def get_partition_statistics(self) -> Dict[str, Any]:
        """Get comprehensive partition statistics"""
        stats = {}

        for db_name, db_path in [('hot', self.hot_db_path), ('warm', self.warm_db_path), ('cold', self.cold_db_path)]:
            try:
                with sqlite3.connect(db_path) as conn:
                    # Get database size
                    db_size = Path(db_path).stat().st_size / (1024 * 1024) if Path(db_path).exists() else 0

                    # Get table statistics
                    table_stats = {}
                    for config in self.partition_configs:
                        try:
                            cursor = conn.execute(f"SELECT COUNT(*) FROM {config.table_name}")
                            row_count = cursor.fetchone()[0]
                            table_stats[config.table_name] = row_count
                        except Exception:
                            table_stats[config.table_name] = 0

                    stats[db_name] = {
                        'database_size_mb': db_size,
                        'table_counts': table_stats,
                        'total_records': sum(table_stats.values())
                    }

            except Exception as e:
                logger.error(f"Failed to get stats for {db_name}: {e}")
                stats[db_name] = {'database_size_mb': 0, 'table_counts': {}, 'total_records': 0}

        return stats

    async def optimize_partition_performance(self):
        """Optimize partition databases for better performance"""
        logger.info("⚡ Optimizing partition databases...")

        optimization_commands = [
            "VACUUM",
            "ANALYZE",
            "PRAGMA optimize"
        ]

        for db_name, db_path in [('hot', self.hot_db_path), ('warm', self.warm_db_path), ('cold', self.cold_db_path)]:
            try:
                with sqlite3.connect(db_path) as conn:
                    for command in optimization_commands:
                        conn.execute(command)
                        conn.commit()

                logger.info(f"✅ Optimized {db_name} partition")

            except Exception as e:
                logger.error(f"Failed to optimize {db_name} partition: {e}")

    def get_query_executor(self):
        """Get a query executor that handles partition routing"""
        return self.PartitionedQueryExecutor(self)


# Factory function
def create_partitioning_system(db_path: str = "tabula_rasa.db") -> DataPartitioningSystem:
    """Create and configure data partitioning system"""
    return DataPartitioningSystem(db_path)

# Utility function for one-time partitioning
async def partition_database(db_path: str = "tabula_rasa.db") -> Dict[str, PartitionStats]:
    """One-time database partitioning operation"""
    system = create_partitioning_system(db_path)
    await system.initialize_partition_databases()
    return await system.partition_all_tables()