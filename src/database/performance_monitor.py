"""
Advanced Database Performance Monitoring System
Tracks query performance, identifies bottlenecks, and provides optimization insights
"""

import sqlite3
import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import psutil
import os

logger = logging.getLogger(__name__)

@dataclass
class QueryMetric:
    """Metrics for a single query execution"""
    query_hash: str
    query_sql: str
    execution_time_ms: float
    rows_affected: int
    timestamp: datetime
    memory_usage_mb: float
    database_size_mb: float

@dataclass
class PerformanceAlert:
    """Performance alert information"""
    alert_type: str
    severity: str  # 'info', 'warning', 'critical'
    message: str
    metric_value: float
    threshold: float
    timestamp: datetime
    suggestions: List[str]

@dataclass
class DatabaseHealth:
    """Overall database health metrics"""
    avg_query_time_ms: float
    slow_query_count: int
    database_size_mb: float
    index_efficiency_score: float
    write_throughput_per_sec: float
    read_throughput_per_sec: float
    cache_hit_ratio: float
    health_score: float  # 0-100
    recommendations: List[str]

class DatabasePerformanceMonitor:
    """Advanced database performance monitoring and analytics"""

    def __init__(self, db_path: str, slow_query_threshold_ms: float = 100.0):
        self.db_path = db_path
        self.slow_query_threshold_ms = slow_query_threshold_ms

        # Performance tracking
        self.query_metrics = deque(maxlen=10000)  # Keep last 10k queries
        self.slow_queries = deque(maxlen=1000)    # Keep last 1k slow queries
        self.query_frequency = defaultdict(int)   # Query hash -> frequency
        self.query_avg_times = defaultdict(list)  # Query hash -> execution times

        # Real-time monitoring
        self.monitoring_enabled = False
        self.monitor_thread = None
        self.alerts = deque(maxlen=100)

        # Performance thresholds
        self.thresholds = {
            'slow_query_ms': slow_query_threshold_ms,
            'database_size_mb': 500.0,  # Alert if over 500MB
            'avg_query_time_ms': 50.0,   # Alert if avg over 50ms
            'slow_query_ratio': 0.1,     # Alert if >10% queries are slow
            'memory_usage_mb': 1000.0,   # Alert if over 1GB memory
        }

        # Thread-safe locks
        self._metrics_lock = threading.Lock()
        self._alerts_lock = threading.Lock()

    async def initialize_monitoring_tables(self):
        """Create tables for performance monitoring"""
        logger.info("📊 Initializing performance monitoring tables...")

        with sqlite3.connect(self.db_path) as conn:
            # Query performance history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT NOT NULL,
                    query_sql TEXT NOT NULL,
                    execution_time_ms REAL NOT NULL,
                    rows_affected INTEGER DEFAULT 0,
                    memory_usage_mb REAL DEFAULT 0,
                    database_size_mb REAL DEFAULT 0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Performance alerts
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    threshold_value REAL NOT NULL,
                    suggestions TEXT,  -- JSON array
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Database health snapshots
            conn.execute("""
                CREATE TABLE IF NOT EXISTS database_health_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    avg_query_time_ms REAL NOT NULL,
                    slow_query_count INTEGER NOT NULL,
                    database_size_mb REAL NOT NULL,
                    index_efficiency_score REAL NOT NULL,
                    write_throughput_per_sec REAL NOT NULL,
                    read_throughput_per_sec REAL NOT NULL,
                    cache_hit_ratio REAL NOT NULL,
                    health_score REAL NOT NULL,
                    recommendations TEXT,  -- JSON array
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_perf_timestamp ON query_performance_history(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_perf_hash ON query_performance_history(query_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON performance_alerts(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_health_timestamp ON database_health_snapshots(timestamp DESC)")

            conn.commit()

        logger.info("✅ Performance monitoring tables initialized")

    def start_monitoring(self):
        """Start real-time performance monitoring"""
        if self.monitoring_enabled:
            return

        self.monitoring_enabled = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("🔍 Database performance monitoring started")

    def stop_monitoring(self):
        """Stop real-time performance monitoring"""
        self.monitoring_enabled = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        logger.info("🛑 Database performance monitoring stopped")

    def track_query(self, query_sql: str, execution_time_ms: float, rows_affected: int = 0):
        """Track a query execution for performance analysis"""
        # Generate query hash for grouping similar queries
        query_hash = self._generate_query_hash(query_sql)

        # Get current memory and database size
        memory_usage = self._get_memory_usage_mb()
        db_size = self._get_database_size_mb()

        # Create metric
        metric = QueryMetric(
            query_hash=query_hash,
            query_sql=query_sql,
            execution_time_ms=execution_time_ms,
            rows_affected=rows_affected,
            timestamp=datetime.now(),
            memory_usage_mb=memory_usage,
            database_size_mb=db_size
        )

        with self._metrics_lock:
            # Add to metrics history
            self.query_metrics.append(metric)

            # Track frequency and average times
            self.query_frequency[query_hash] += 1
            self.query_avg_times[query_hash].append(execution_time_ms)

            # Keep only recent times for averages (last 100 executions)
            if len(self.query_avg_times[query_hash]) > 100:
                self.query_avg_times[query_hash] = self.query_avg_times[query_hash][-100:]

            # Track slow queries
            if execution_time_ms > self.slow_query_threshold_ms:
                self.slow_queries.append(metric)
                self._check_slow_query_alert(metric)

        # Log slow queries immediately
        if execution_time_ms > self.slow_query_threshold_ms:
            logger.warning(f"🐌 Slow query detected ({execution_time_ms:.1f}ms): {query_sql[:100]}...")

    def _generate_query_hash(self, query_sql: str) -> str:
        """Generate a hash for query grouping (normalize similar queries)"""
        import hashlib

        # Normalize query for grouping
        normalized = query_sql.upper().strip()

        # Replace parameter placeholders to group similar queries
        import re
        normalized = re.sub(r"'[^']*'", "'?'", normalized)  # String literals
        normalized = re.sub(r'\b\d+\b', '?', normalized)    # Numbers
        normalized = re.sub(r'\s+', ' ', normalized)        # Multiple spaces

        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def _get_database_size_mb(self) -> float:
        """Get current database size in MB"""
        try:
            return os.path.getsize(self.db_path) / (1024 * 1024)
        except Exception:
            return 0.0

    def _check_slow_query_alert(self, metric: QueryMetric):
        """Check if slow query warrants an alert"""
        # Count recent slow queries (last 5 minutes)
        recent_cutoff = datetime.now() - timedelta(minutes=5)
        recent_slow_count = sum(1 for m in self.slow_queries if m.timestamp > recent_cutoff)

        if recent_slow_count > 10:  # More than 10 slow queries in 5 minutes
            suggestions = [
                "Consider adding indexes for frequently queried columns",
                "Review query patterns for optimization opportunities",
                "Check if database needs VACUUM or ANALYZE",
                "Consider implementing query caching for repeated queries"
            ]

            alert = PerformanceAlert(
                alert_type="high_slow_query_rate",
                severity="warning",
                message=f"{recent_slow_count} slow queries detected in last 5 minutes",
                metric_value=recent_slow_count,
                threshold=10,
                timestamp=datetime.now(),
                suggestions=suggestions
            )

            self._add_alert(alert)

    def _add_alert(self, alert: PerformanceAlert):
        """Add a performance alert"""
        with self._alerts_lock:
            self.alerts.append(alert)

        # Log alert
        log_level = logger.error if alert.severity == 'critical' else logger.warning
        log_level(f"🚨 Performance Alert [{alert.severity.upper()}]: {alert.message}")

    def _monitoring_loop(self):
        """Background monitoring loop"""
        logger.info("🔄 Performance monitoring loop started")

        while self.monitoring_enabled:
            try:
                # Check database health every 60 seconds
                health = self.get_database_health()
                self._check_health_alerts(health)

                # Save health snapshot
                self._save_health_snapshot(health)

                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Wait before retrying

    def _check_health_alerts(self, health: DatabaseHealth):
        """Check database health for alert conditions"""
        alerts = []

        # Database size alert
        if health.database_size_mb > self.thresholds['database_size_mb']:
            alerts.append(PerformanceAlert(
                alert_type="database_size",
                severity="warning" if health.database_size_mb < self.thresholds['database_size_mb'] * 1.5 else "critical",
                message=f"Database size ({health.database_size_mb:.1f} MB) exceeds threshold",
                metric_value=health.database_size_mb,
                threshold=self.thresholds['database_size_mb'],
                timestamp=datetime.now(),
                suggestions=[
                    "Run data retention cleanup to remove old data",
                    "Implement frame compression to reduce storage",
                    "Archive historical data to separate storage"
                ]
            ))

        # Average query time alert
        if health.avg_query_time_ms > self.thresholds['avg_query_time_ms']:
            alerts.append(PerformanceAlert(
                alert_type="avg_query_time",
                severity="warning",
                message=f"Average query time ({health.avg_query_time_ms:.1f} ms) is high",
                metric_value=health.avg_query_time_ms,
                threshold=self.thresholds['avg_query_time_ms'],
                timestamp=datetime.now(),
                suggestions=[
                    "Optimize database indexes",
                    "Review and optimize slow queries",
                    "Consider database VACUUM and ANALYZE"
                ]
            ))

        # Health score alert
        if health.health_score < 70:
            severity = "critical" if health.health_score < 50 else "warning"
            alerts.append(PerformanceAlert(
                alert_type="health_score",
                severity=severity,
                message=f"Database health score is low ({health.health_score:.1f}/100)",
                metric_value=health.health_score,
                threshold=70,
                timestamp=datetime.now(),
                suggestions=health.recommendations
            ))

        # Add all alerts
        for alert in alerts:
            self._add_alert(alert)

    def _save_health_snapshot(self, health: DatabaseHealth):
        """Save health snapshot to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                import json
                conn.execute("""
                    INSERT INTO database_health_snapshots
                    (avg_query_time_ms, slow_query_count, database_size_mb, index_efficiency_score,
                     write_throughput_per_sec, read_throughput_per_sec, cache_hit_ratio, health_score, recommendations)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    health.avg_query_time_ms,
                    health.slow_query_count,
                    health.database_size_mb,
                    health.index_efficiency_score,
                    health.write_throughput_per_sec,
                    health.read_throughput_per_sec,
                    health.cache_hit_ratio,
                    health.health_score,
                    json.dumps(health.recommendations)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save health snapshot: {e}")

    def get_database_health(self) -> DatabaseHealth:
        """Calculate comprehensive database health metrics"""
        with self._metrics_lock:
            metrics_list = list(self.query_metrics)
            slow_queries_list = list(self.slow_queries)

        if not metrics_list:
            return DatabaseHealth(0, 0, 0, 100, 0, 0, 100, 100, [])

        # Recent metrics (last hour)
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_metrics = [m for m in metrics_list if m.timestamp > recent_cutoff]

        # Calculate metrics
        avg_query_time = statistics.mean([m.execution_time_ms for m in recent_metrics]) if recent_metrics else 0

        # Recent slow queries
        recent_slow = [m for m in slow_queries_list if m.timestamp > recent_cutoff]
        slow_query_count = len(recent_slow)

        database_size = self._get_database_size_mb()

        # Calculate throughput (queries per second in last hour)
        if recent_metrics:
            time_span_hours = (max(m.timestamp for m in recent_metrics) - min(m.timestamp for m in recent_metrics)).total_seconds() / 3600
            if time_span_hours > 0:
                total_throughput = len(recent_metrics) / time_span_hours / 3600  # per second
            else:
                total_throughput = 0
        else:
            total_throughput = 0

        # Estimate read/write split (rough heuristic)
        write_queries = sum(1 for m in recent_metrics if any(keyword in m.query_sql.upper() for keyword in ['INSERT', 'UPDATE', 'DELETE']))
        write_throughput = (write_queries / len(recent_metrics) * total_throughput) if recent_metrics else 0
        read_throughput = total_throughput - write_throughput

        # Index efficiency (heuristic based on query times)
        index_efficiency = max(0, 100 - avg_query_time * 2)  # Rough estimate

        # Cache hit ratio (simplified estimate)
        cache_hit_ratio = max(0, 100 - slow_query_count * 5)  # Rough estimate

        # Calculate overall health score
        health_score = self._calculate_health_score(
            avg_query_time, slow_query_count, database_size, index_efficiency
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            avg_query_time, slow_query_count, database_size, health_score
        )

        return DatabaseHealth(
            avg_query_time_ms=avg_query_time,
            slow_query_count=slow_query_count,
            database_size_mb=database_size,
            index_efficiency_score=index_efficiency,
            write_throughput_per_sec=write_throughput,
            read_throughput_per_sec=read_throughput,
            cache_hit_ratio=cache_hit_ratio,
            health_score=health_score,
            recommendations=recommendations
        )

    def _calculate_health_score(self, avg_query_time: float, slow_query_count: int,
                               database_size: float, index_efficiency: float) -> float:
        """Calculate overall health score (0-100)"""
        score = 100

        # Penalize slow average query time
        if avg_query_time > 50:
            score -= min(30, (avg_query_time - 50) / 2)

        # Penalize slow queries
        score -= min(20, slow_query_count)

        # Penalize large database size
        if database_size > 500:
            score -= min(20, (database_size - 500) / 100)

        # Factor in index efficiency
        score = score * (index_efficiency / 100)

        return max(0, score)

    def _generate_recommendations(self, avg_query_time: float, slow_query_count: int,
                                 database_size: float, health_score: float) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []

        if avg_query_time > 50:
            recommendations.append("Optimize database indexes for better query performance")
            recommendations.append("Review and optimize frequently executed queries")

        if slow_query_count > 5:
            recommendations.append("Investigate and optimize slow queries")
            recommendations.append("Consider query caching for repeated operations")

        if database_size > 500:
            recommendations.append("Implement data retention policies to control database size")
            recommendations.append("Enable frame compression to reduce storage requirements")

        if health_score < 70:
            recommendations.append("Run database VACUUM to optimize storage")
            recommendations.append("Analyze query patterns and add strategic indexes")

        return recommendations

    def get_slow_queries_report(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get report of slowest queries"""
        with self._metrics_lock:
            # Group by query hash and calculate averages
            query_stats = defaultdict(lambda: {'total_time': 0, 'count': 0, 'max_time': 0, 'example_sql': ''})

            for metric in self.query_metrics:
                stats = query_stats[metric.query_hash]
                stats['total_time'] += metric.execution_time_ms
                stats['count'] += 1
                stats['max_time'] = max(stats['max_time'], metric.execution_time_ms)
                if not stats['example_sql']:
                    stats['example_sql'] = metric.query_sql

            # Sort by average execution time
            sorted_queries = sorted(
                query_stats.items(),
                key=lambda x: x[1]['total_time'] / x[1]['count'],
                reverse=True
            )

        report = []
        for query_hash, stats in sorted_queries[:limit]:
            avg_time = stats['total_time'] / stats['count']
            report.append({
                'query_hash': query_hash,
                'average_time_ms': avg_time,
                'max_time_ms': stats['max_time'],
                'execution_count': stats['count'],
                'total_time_ms': stats['total_time'],
                'example_sql': stats['example_sql'][:200]  # Truncate for readability
            })

        return report

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        health = self.get_database_health()
        slow_queries = self.get_slow_queries_report(5)

        with self._alerts_lock:
            recent_alerts = [alert for alert in self.alerts if alert.timestamp > datetime.now() - timedelta(hours=24)]

        return {
            'health': asdict(health),
            'slow_queries': slow_queries,
            'recent_alerts': [asdict(alert) for alert in recent_alerts],
            'monitoring_enabled': self.monitoring_enabled,
            'total_queries_tracked': len(self.query_metrics),
            'database_size_mb': self._get_database_size_mb(),
            'memory_usage_mb': self._get_memory_usage_mb()
        }


# Factory function
def create_performance_monitor(db_path: str = "tabula_rasa.db", slow_query_threshold_ms: float = 100.0) -> DatabasePerformanceMonitor:
    """Create and configure performance monitor"""
    return DatabasePerformanceMonitor(db_path, slow_query_threshold_ms)

# Context manager for query tracking
class QueryTracker:
    """Context manager for automatic query performance tracking"""

    def __init__(self, monitor: DatabasePerformanceMonitor, query_sql: str):
        self.monitor = monitor
        self.query_sql = query_sql
        self.start_time = None
        self.rows_affected = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            execution_time_ms = (time.time() - self.start_time) * 1000
            self.monitor.track_query(self.query_sql, execution_time_ms, self.rows_affected)

    def set_rows_affected(self, count: int):
        """Set the number of rows affected by the query"""
        self.rows_affected = count