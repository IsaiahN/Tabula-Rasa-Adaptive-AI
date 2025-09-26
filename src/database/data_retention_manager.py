"""
Advanced Data Retention Management System
Automatic cleanup and lifecycle management for optimal database performance
"""

import sqlite3
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class RetentionPolicy:
    """Defines retention policy for a table"""
    table_name: str
    retention_days: int
    timestamp_column: str
    cleanup_batch_size: int = 1000
    enabled: bool = True

@dataclass
class CleanupResult:
    """Results from cleanup operation"""
    table_name: str
    rows_deleted: int
    space_freed_mb: float
    execution_time_ms: float

class DataRetentionManager:
    """Advanced data retention and cleanup system"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.retention_policies = self._define_retention_policies()
        self.cleanup_history = []

    def _define_retention_policies(self) -> List[RetentionPolicy]:
        """Define retention policies for all tables"""
        return [
            # High-volume transactional data - aggressive cleanup
            RetentionPolicy("action_traces", 7, "timestamp", 2000),
            RetentionPolicy("system_logs", 3, "timestamp", 5000),
            RetentionPolicy("frame_tracking", 14, "timestamp", 1000),
            RetentionPolicy("gan_training_data", 30, "timestamp", 500),

            # Coordinate and learning data - moderate retention
            RetentionPolicy("coordinate_penalties", 60, "last_penalty_applied", 1000),
            RetentionPolicy("coordinate_diversity", 45, "last_used", 1000),
            RetentionPolicy("failure_learning", 90, "last_failure", 500),

            # Performance and analytics - longer retention
            RetentionPolicy("performance_history", 180, "timestamp", 500),
            RetentionPolicy("session_history", 90, "timestamp", 1000),
            RetentionPolicy("score_history", 120, "timestamp", 1000),

            # Error logs - moderate retention with deduplication priority
            RetentionPolicy("error_logs", 30, "last_seen", 500),

            # GAN system data - aggressive cleanup for large objects
            RetentionPolicy("gan_generated_states", 14, "created_at", 200),
            RetentionPolicy("gan_validation_results", 21, "created_at", 500),
            RetentionPolicy("gan_performance_metrics", 60, "timestamp", 1000),

            # Advanced action system data
            RetentionPolicy("stagnation_events", 45, "detection_timestamp", 500),
            RetentionPolicy("frame_change_analysis", 30, "analysis_timestamp", 1000),
            RetentionPolicy("emergency_overrides", 60, "override_timestamp", 500),
            RetentionPolicy("visual_targets", 30, "detection_timestamp", 1000),
        ]

    async def execute_full_cleanup(self) -> List[CleanupResult]:
        """Execute comprehensive cleanup across all tables"""
        logger.info("🧹 Starting comprehensive database cleanup...")

        start_time = time.time()
        results = []
        total_space_freed = 0.0
        total_rows_deleted = 0

        # Get initial database size
        initial_size = await self._get_database_size_mb()
        logger.info(f"📊 Initial database size: {initial_size:.2f} MB")

        for policy in self.retention_policies:
            if not policy.enabled:
                continue

            try:
                result = await self._cleanup_table(policy)
                results.append(result)
                total_space_freed += result.space_freed_mb
                total_rows_deleted += result.rows_deleted

                logger.info(f"✅ {policy.table_name}: {result.rows_deleted:,} rows deleted, "
                           f"{result.space_freed_mb:.2f} MB freed in {result.execution_time_ms:.0f}ms")

            except Exception as e:
                logger.error(f"❌ Failed to cleanup {policy.table_name}: {e}")

        # VACUUM to reclaim space
        await self._vacuum_database()

        # Get final database size
        final_size = await self._get_database_size_mb()
        actual_space_freed = initial_size - final_size

        total_time = (time.time() - start_time) * 1000

        logger.info(f"🎉 Cleanup completed in {total_time:.0f}ms:")
        logger.info(f"   📉 {total_rows_deleted:,} total rows deleted")
        logger.info(f"   💾 {actual_space_freed:.2f} MB actual space freed")
        logger.info(f"   📊 Database size: {initial_size:.2f} MB → {final_size:.2f} MB")
        logger.info(f"   📈 Space reduction: {(actual_space_freed/initial_size)*100:.1f}%")

        # Save cleanup history
        self.cleanup_history.append({
            'timestamp': datetime.now(),
            'total_rows_deleted': total_rows_deleted,
            'space_freed_mb': actual_space_freed,
            'execution_time_ms': total_time,
            'table_results': results
        })

        return results

    async def _cleanup_table(self, policy: RetentionPolicy) -> CleanupResult:
        """Cleanup a specific table according to its retention policy"""
        start_time = time.time()

        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=policy.retention_days)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')

        with sqlite3.connect(self.db_path) as conn:
            # Count rows to be deleted
            count_query = f"""
                SELECT COUNT(*) FROM {policy.table_name}
                WHERE {policy.timestamp_column} < ?
            """
            cursor = conn.execute(count_query, (cutoff_str,))
            rows_to_delete = cursor.fetchone()[0]

            if rows_to_delete == 0:
                return CleanupResult(policy.table_name, 0, 0.0, 0.0)

            # Get size before deletion
            size_before = await self._get_table_size_mb(policy.table_name)

            # Delete in batches to avoid locking
            total_deleted = 0
            while total_deleted < rows_to_delete:
                delete_query = f"""
                    DELETE FROM {policy.table_name}
                    WHERE {policy.timestamp_column} < ?
                    LIMIT ?
                """
                cursor = conn.execute(delete_query, (cutoff_str, policy.cleanup_batch_size))
                batch_deleted = cursor.rowcount
                total_deleted += batch_deleted

                if batch_deleted == 0:
                    break

                # Commit batch
                conn.commit()

                # Small delay to prevent database locking
                await asyncio.sleep(0.01)

            # Get size after deletion
            size_after = await self._get_table_size_mb(policy.table_name)
            space_freed = size_before - size_after

        execution_time = (time.time() - start_time) * 1000
        return CleanupResult(policy.table_name, total_deleted, space_freed, execution_time)

    async def _get_database_size_mb(self) -> float:
        """Get total database size in MB"""
        import os
        try:
            size_bytes = os.path.getsize(self.db_path)
            return size_bytes / (1024 * 1024)
        except Exception:
            return 0.0

    async def _get_table_size_mb(self, table_name: str) -> float:
        """Estimate table size in MB"""
        with sqlite3.connect(self.db_path) as conn:
            # Get approximate table size using PRAGMA
            try:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]

                if row_count == 0:
                    return 0.0

                # Rough estimate: average 1KB per row for most tables
                estimated_size_mb = (row_count * 1024) / (1024 * 1024)
                return estimated_size_mb

            except Exception:
                return 0.0

    async def _vacuum_database(self):
        """VACUUM database to reclaim space"""
        logger.info("🔄 VACUUMing database to reclaim space...")
        start_time = time.time()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")

        vacuum_time = (time.time() - start_time) * 1000
        logger.info(f"✅ VACUUM completed in {vacuum_time:.0f}ms")

    async def get_cleanup_schedule_recommendation(self) -> Dict:
        """Recommend optimal cleanup schedule based on data growth"""
        with sqlite3.connect(self.db_path) as conn:
            # Analyze data growth patterns
            growth_analysis = {}

            for policy in self.retention_policies:
                try:
                    # Count recent data (last 24h)
                    recent_query = f"""
                        SELECT COUNT(*) FROM {policy.table_name}
                        WHERE {policy.timestamp_column} >= datetime('now', '-1 day')
                    """
                    cursor = conn.execute(recent_query)
                    recent_count = cursor.fetchone()[0]

                    # Daily growth rate
                    daily_growth = recent_count

                    # Estimate cleanup frequency needed
                    if daily_growth > 1000:
                        recommended_frequency = "daily"
                    elif daily_growth > 100:
                        recommended_frequency = "weekly"
                    else:
                        recommended_frequency = "monthly"

                    growth_analysis[policy.table_name] = {
                        'daily_growth': daily_growth,
                        'recommended_frequency': recommended_frequency,
                        'retention_days': policy.retention_days
                    }

                except Exception as e:
                    logger.debug(f"Could not analyze growth for {policy.table_name}: {e}")

        return {
            'recommended_schedule': self._calculate_optimal_schedule(growth_analysis),
            'table_analysis': growth_analysis,
            'next_cleanup_estimate': datetime.now() + timedelta(days=1)
        }

    def _calculate_optimal_schedule(self, growth_analysis: Dict) -> str:
        """Calculate optimal cleanup schedule"""
        high_growth_tables = sum(1 for table_data in growth_analysis.values()
                                if table_data['daily_growth'] > 500)

        if high_growth_tables > 5:
            return "daily"
        elif high_growth_tables > 2:
            return "weekly"
        else:
            return "monthly"

    async def emergency_cleanup(self, space_target_mb: float = 50.0) -> CleanupResult:
        """Emergency cleanup to reach target database size"""
        logger.warning(f"🚨 Emergency cleanup triggered - target: {space_target_mb} MB")

        current_size = await self._get_database_size_mb()
        if current_size <= space_target_mb:
            logger.info(f"✅ Database size ({current_size:.2f} MB) already under target")
            return CleanupResult("emergency", 0, 0.0, 0.0)

        # Aggressive cleanup - reduce retention periods by 50%
        aggressive_policies = []
        for policy in self.retention_policies:
            aggressive_policy = RetentionPolicy(
                policy.table_name,
                max(1, policy.retention_days // 2),  # Half retention time
                policy.timestamp_column,
                policy.cleanup_batch_size * 2,  # Larger batches
                policy.enabled
            )
            aggressive_policies.append(aggressive_policy)

        # Execute aggressive cleanup
        original_policies = self.retention_policies
        self.retention_policies = aggressive_policies

        try:
            results = await self.execute_full_cleanup()
            total_space_freed = sum(r.space_freed_mb for r in results)
            total_rows_deleted = sum(r.rows_deleted for r in results)

            final_size = await self._get_database_size_mb()

            logger.warning(f"🚨 Emergency cleanup completed:")
            logger.warning(f"   📊 {current_size:.2f} MB → {final_size:.2f} MB")
            logger.warning(f"   🗑️ {total_rows_deleted:,} rows deleted")

            return CleanupResult("emergency", total_rows_deleted, total_space_freed, 0.0)

        finally:
            # Restore original policies
            self.retention_policies = original_policies


# Factory function for easy integration
def create_retention_manager(db_path: str = "tabula_rasa.db") -> DataRetentionManager:
    """Create and configure data retention manager"""
    return DataRetentionManager(db_path)


# Async cleanup job for integration with training systems
async def run_scheduled_cleanup(db_path: str = "tabula_rasa.db"):
    """Run scheduled cleanup job"""
    manager = create_retention_manager(db_path)
    results = await manager.execute_full_cleanup()
    return results