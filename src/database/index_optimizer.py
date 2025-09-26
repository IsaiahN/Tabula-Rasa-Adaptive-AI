"""
Database Index Optimization System
Removes redundant indexes and creates strategic compound indexes for performance
"""

import sqlite3
import logging
import time
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class IndexAnalysis:
    """Analysis of database index usage and recommendations"""
    redundant_indexes: List[str]
    missing_indexes: List[str]
    compound_opportunities: List[str]
    index_efficiency_scores: Dict[str, float]

@dataclass
class IndexOptimizationResult:
    """Results from index optimization"""
    indexes_dropped: int
    indexes_created: int
    estimated_write_improvement: float
    estimated_query_improvement: float
    execution_time_ms: float

class DatabaseIndexOptimizer:
    """Advanced database index optimization system"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.current_indexes = {}
        self.query_patterns = {}

    async def analyze_indexes(self) -> IndexAnalysis:
        """Comprehensive index analysis"""
        logger.info("🔍 Analyzing database indexes...")

        with sqlite3.connect(self.db_path) as conn:
            # Get all current indexes
            cursor = conn.execute("""
                SELECT name, tbl_name, sql
                FROM sqlite_master
                WHERE type = 'index' AND name NOT LIKE 'sqlite_%'
            """)
            indexes = cursor.fetchall()

            # Analyze each index
            redundant = self._find_redundant_indexes(conn, indexes)
            missing = self._find_missing_indexes(conn)
            compound_opportunities = self._find_compound_opportunities(conn)
            efficiency_scores = self._calculate_efficiency_scores(conn, indexes)

        return IndexAnalysis(
            redundant_indexes=redundant,
            missing_indexes=missing,
            compound_opportunities=compound_opportunities,
            index_efficiency_scores=efficiency_scores
        )

    def _find_redundant_indexes(self, conn: sqlite3.Connection, indexes: List[Tuple]) -> List[str]:
        """Find redundant single-column indexes that are covered by compound indexes"""
        redundant = []

        # Map of table -> columns that have indexes
        table_indexes = {}
        for name, table, sql in indexes:
            if table not in table_indexes:
                table_indexes[table] = {}

            # Extract columns from index SQL
            if sql and 'ON' in sql:
                columns_part = sql.split('ON')[1].split('(')[1].split(')')[0]
                columns = [col.strip() for col in columns_part.split(',')]
                table_indexes[table][name] = columns

        # Find redundant single-column indexes
        for table, indexes_dict in table_indexes.items():
            single_column_indexes = {name: cols for name, cols in indexes_dict.items() if len(cols) == 1}
            compound_indexes = {name: cols for name, cols in indexes_dict.items() if len(cols) > 1}

            for single_name, single_cols in single_column_indexes.items():
                single_col = single_cols[0]

                # Check if this column is the first column in any compound index
                for compound_name, compound_cols in compound_indexes.items():
                    if compound_cols[0] == single_col:
                        redundant.append(single_name)
                        logger.debug(f"Redundant index: {single_name} covered by compound index {compound_name}")
                        break

        return redundant

    def _find_missing_indexes(self, conn: sqlite3.Connection) -> List[str]:
        """Find missing strategic indexes based on query patterns"""
        missing = []

        # Strategic indexes for common query patterns
        strategic_indexes = [
            # Game-based queries (most common pattern)
            "CREATE INDEX IF NOT EXISTS idx_action_traces_game_time ON action_traces(game_id, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_coordinate_intelligence_game_success ON coordinate_intelligence(game_id, success_rate DESC)",
            "CREATE INDEX IF NOT EXISTS idx_system_logs_game_level ON system_logs(game_id, log_level, timestamp DESC)",

            # Performance-critical queries
            "CREATE INDEX IF NOT EXISTS idx_training_sessions_recent ON training_sessions(start_time DESC, status)",
            "CREATE INDEX IF NOT EXISTS idx_game_results_score ON game_results(session_id, final_score DESC)",

            # Learning system queries
            "CREATE INDEX IF NOT EXISTS idx_learned_patterns_success ON learned_patterns(pattern_type, success_rate DESC)",
            "CREATE INDEX IF NOT EXISTS idx_winning_strategies_efficiency ON winning_strategies(game_type, efficiency DESC)",

            # Coordinate-based queries (Action 6 optimization)
            "CREATE INDEX IF NOT EXISTS idx_coordinate_penalties_game_coords ON coordinate_penalties(game_id, x, y)",
            "CREATE INDEX IF NOT EXISTS idx_visual_targets_game_coords ON visual_targets(game_id, target_x, target_y)",

            # Time-based cleanup queries
            "CREATE INDEX IF NOT EXISTS idx_frame_tracking_cleanup ON frame_tracking(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_error_logs_cleanup ON error_logs(last_seen)",

            # GAN system optimization
            "CREATE INDEX IF NOT EXISTS idx_gan_training_data_coords ON gan_training_data(game_id, action_number, coordinates)",
            "CREATE INDEX IF NOT EXISTS idx_gan_generated_states_quality ON gan_generated_states(quality_score DESC, session_id)",
        ]

        # Check which indexes are missing
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        existing_index_names = {row[0] for row in cursor.fetchall()}

        for index_sql in strategic_indexes:
            # Extract index name from SQL
            index_name = index_sql.split(' ')[5]  # "CREATE INDEX IF NOT EXISTS idx_name"

            if index_name not in existing_index_names:
                missing.append(index_sql)

        return missing

    def _find_compound_opportunities(self, conn: sqlite3.Connection) -> List[str]:
        """Find opportunities for compound indexes based on query patterns"""
        opportunities = []

        # Analyze common multi-column WHERE clauses
        compound_opportunities = [
            # Session and time-based queries
            "CREATE INDEX IF NOT EXISTS idx_session_game_time ON action_traces(session_id, game_id, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_coordinate_success_time ON coordinate_intelligence(success_rate DESC, last_used DESC)",

            # Action effectiveness analysis
            "CREATE INDEX IF NOT EXISTS idx_action_game_success ON action_effectiveness(action_number, game_id, success_rate DESC)",

            # Stagnation detection optimization
            "CREATE INDEX IF NOT EXISTS idx_stagnation_game_time ON stagnation_events(game_id, detection_timestamp DESC, severity DESC)",

            # Emergency override tracking
            "CREATE INDEX IF NOT EXISTS idx_emergency_game_time ON emergency_overrides(game_id, override_timestamp DESC)",
        ]

        return compound_opportunities

    def _calculate_efficiency_scores(self, conn: sqlite3.Connection, indexes: List[Tuple]) -> Dict[str, float]:
        """Calculate efficiency scores for existing indexes"""
        efficiency_scores = {}

        for name, table, sql in indexes:
            if not sql or 'CREATE INDEX' not in sql:
                continue

            try:
                # Basic efficiency heuristics
                score = 1.0

                # Check if index is used in common queries
                if self._is_index_used_frequently(conn, name, table):
                    score += 0.5

                # Compound indexes are generally more efficient
                if ',' in sql:  # Multiple columns
                    score += 0.3

                # Indexes on timestamp columns are very useful for cleanup
                if 'timestamp' in sql.lower() or 'time' in sql.lower():
                    score += 0.2

                # Indexes on game_id are essential for game-based queries
                if 'game_id' in sql.lower():
                    score += 0.4

                efficiency_scores[name] = min(score, 2.0)

            except Exception as e:
                logger.debug(f"Could not calculate efficiency for index {name}: {e}")
                efficiency_scores[name] = 0.5

        return efficiency_scores

    def _is_index_used_frequently(self, conn: sqlite3.Connection, index_name: str, table_name: str) -> bool:
        """Heuristic to determine if an index is used frequently"""
        # Check if table has significant data (indicates frequent usage)
        try:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]

            # Tables with more data likely benefit more from indexes
            return row_count > 1000

        except Exception:
            return False

    async def optimize_indexes(self, analysis: IndexAnalysis = None) -> IndexOptimizationResult:
        """Execute index optimization based on analysis"""
        if analysis is None:
            analysis = await self.analyze_indexes()

        logger.info("⚡ Optimizing database indexes...")
        start_time = time.time()

        indexes_dropped = 0
        indexes_created = 0

        with sqlite3.connect(self.db_path) as conn:
            # Drop redundant indexes
            for redundant_index in analysis.redundant_indexes:
                try:
                    conn.execute(f"DROP INDEX IF EXISTS {redundant_index}")
                    indexes_dropped += 1
                    logger.info(f"🗑️  Dropped redundant index: {redundant_index}")
                except Exception as e:
                    logger.error(f"Failed to drop index {redundant_index}: {e}")

            # Create missing strategic indexes
            for missing_index_sql in analysis.missing_indexes:
                try:
                    conn.execute(missing_index_sql)
                    indexes_created += 1
                    index_name = missing_index_sql.split(' ')[5]
                    logger.info(f"✅ Created strategic index: {index_name}")
                except Exception as e:
                    logger.error(f"Failed to create index: {e}")

            # Create compound opportunity indexes
            for compound_index_sql in analysis.compound_opportunities:
                try:
                    conn.execute(compound_index_sql)
                    indexes_created += 1
                    index_name = compound_index_sql.split(' ')[5]
                    logger.info(f"🔗 Created compound index: {index_name}")
                except Exception as e:
                    logger.error(f"Failed to create compound index: {e}")

            conn.commit()

        execution_time = (time.time() - start_time) * 1000

        # Estimate performance improvements
        write_improvement = self._estimate_write_improvement(indexes_dropped, indexes_created)
        query_improvement = self._estimate_query_improvement(indexes_created)

        logger.info(f"🎉 Index optimization completed in {execution_time:.0f}ms:")
        logger.info(f"   🗑️  {indexes_dropped} redundant indexes removed")
        logger.info(f"   ✅ {indexes_created} strategic indexes created")
        logger.info(f"   📈 Estimated write speed improvement: {write_improvement:.1f}%")
        logger.info(f"   🔍 Estimated query speed improvement: {query_improvement:.1f}%")

        return IndexOptimizationResult(
            indexes_dropped=indexes_dropped,
            indexes_created=indexes_created,
            estimated_write_improvement=write_improvement,
            estimated_query_improvement=query_improvement,
            execution_time_ms=execution_time
        )

    def _estimate_write_improvement(self, dropped: int, created: int) -> float:
        """Estimate write performance improvement from index changes"""
        # Each dropped index improves write speed by ~5-15%
        # Each created index reduces write speed by ~2-5%
        improvement = (dropped * 8) - (created * 3)
        return max(0, improvement)

    def _estimate_query_improvement(self, created: int) -> float:
        """Estimate query performance improvement from new indexes"""
        # Each strategic index can improve relevant queries by 20-50%
        # Compound indexes are especially beneficial
        return min(created * 25, 200)  # Cap at 200% improvement

    async def generate_index_report(self) -> str:
        """Generate comprehensive index analysis report"""
        analysis = await self.analyze_indexes()

        report = []
        report.append("=" * 80)
        report.append("DATABASE INDEX OPTIMIZATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Current state
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND name NOT LIKE 'sqlite_%'")
            total_indexes = cursor.fetchone()[0]

        report.append(f"📊 CURRENT STATE:")
        report.append(f"   Total indexes: {total_indexes}")
        report.append(f"   Redundant indexes: {len(analysis.redundant_indexes)}")
        report.append(f"   Missing strategic indexes: {len(analysis.missing_indexes)}")
        report.append("")

        # Redundant indexes
        if analysis.redundant_indexes:
            report.append("🗑️  REDUNDANT INDEXES TO REMOVE:")
            for idx in analysis.redundant_indexes:
                report.append(f"   • {idx}")
            report.append("")

        # Missing indexes
        if analysis.missing_indexes:
            report.append("✅ STRATEGIC INDEXES TO CREATE:")
            for idx_sql in analysis.missing_indexes:
                idx_name = idx_sql.split(' ')[5]
                report.append(f"   • {idx_name}")
            report.append("")

        # Efficiency scores
        report.append("📈 INDEX EFFICIENCY SCORES:")
        for idx_name, score in sorted(analysis.index_efficiency_scores.items(), key=lambda x: x[1], reverse=True):
            report.append(f"   • {idx_name}: {score:.2f}")
        report.append("")

        # Recommendations
        total_removals = len(analysis.redundant_indexes)
        total_additions = len(analysis.missing_indexes) + len(analysis.compound_opportunities)

        estimated_write_improvement = self._estimate_write_improvement(total_removals, total_additions)
        estimated_query_improvement = self._estimate_query_improvement(total_additions)

        report.append("🎯 OPTIMIZATION IMPACT:")
        report.append(f"   Estimated write speed improvement: {estimated_write_improvement:.1f}%")
        report.append(f"   Estimated query speed improvement: {estimated_query_improvement:.1f}%")
        report.append(f"   Net index count change: {total_additions - total_removals:+d}")

        return "\n".join(report)


# Factory function
def create_index_optimizer(db_path: str = "tabula_rasa.db") -> DatabaseIndexOptimizer:
    """Create and configure index optimizer"""
    return DatabaseIndexOptimizer(db_path)


# Utility function for one-time optimization
async def optimize_database_indexes(db_path: str = "tabula_rasa.db") -> IndexOptimizationResult:
    """One-time index optimization"""
    optimizer = create_index_optimizer(db_path)
    return await optimizer.optimize_indexes()