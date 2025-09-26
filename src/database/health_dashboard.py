"""
Comprehensive Database Health Dashboard
Real-time monitoring and management of all database optimizations
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import threading

# Import our optimization systems
from .data_retention_manager import create_retention_manager
from .index_optimizer import create_index_optimizer
from .async_batch_writer import get_global_batcher
from .frame_compression_system import create_frame_compression_system
from .performance_monitor import create_performance_monitor
from .data_partitioning_system import create_partitioning_system

logger = logging.getLogger(__name__)

@dataclass
class SystemHealth:
    """Overall system health status"""
    overall_score: float  # 0-100
    database_size_mb: float
    query_performance_score: float
    storage_efficiency_score: float
    write_performance_score: float
    last_updated: datetime
    status: str  # 'excellent', 'good', 'warning', 'critical'
    recommendations: List[str]

@dataclass
class OptimizationMetrics:
    """Metrics for all optimization systems"""
    retention_cleanup: Dict[str, Any]
    index_optimization: Dict[str, Any]
    async_batching: Dict[str, Any]
    frame_compression: Dict[str, Any]
    performance_monitoring: Dict[str, Any]
    data_partitioning: Dict[str, Any]

class DatabaseHealthDashboard:
    """Comprehensive dashboard for database health monitoring"""

    def __init__(self, db_path: str = "tabula_rasa.db"):
        self.db_path = db_path

        # Initialize all optimization systems
        self.retention_manager = create_retention_manager(db_path)
        self.index_optimizer = create_index_optimizer(db_path)
        self.frame_compression = create_frame_compression_system(db_path)
        self.performance_monitor = create_performance_monitor(db_path)
        self.partitioning_system = create_partitioning_system(db_path)

        # Dashboard state
        self.dashboard_enabled = False
        self.dashboard_thread = None
        self.health_history = []
        self.auto_optimization_enabled = True

        # Update intervals (seconds)
        self.health_check_interval = 300  # 5 minutes
        self.auto_cleanup_interval = 3600  # 1 hour
        self.auto_optimization_interval = 86400  # 24 hours

        # Last optimization times
        self.last_cleanup = None
        self.last_optimization = None
        self.last_health_check = None

    async def initialize_dashboard(self):
        """Initialize the dashboard and all subsystems"""
        logger.info("🚀 Initializing Database Health Dashboard...")

        try:
            # Initialize all subsystem tables
            await self.retention_manager.execute_full_cleanup()  # Initial cleanup
            await self.performance_monitor.initialize_monitoring_tables()
            await self.frame_compression.initialize_compression_tables()
            await self.partitioning_system.initialize_partition_databases()

            # Start performance monitoring
            self.performance_monitor.start_monitoring()

            # Start async batching
            await get_global_batcher(self.db_path)

            logger.info("✅ Database Health Dashboard initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize dashboard: {e}")
            raise

    def start_dashboard(self):
        """Start the dashboard monitoring loop"""
        if self.dashboard_enabled:
            return

        self.dashboard_enabled = True
        self.dashboard_thread = threading.Thread(target=self._dashboard_loop, daemon=True)
        self.dashboard_thread.start()

        logger.info("📊 Database Health Dashboard started")

    def stop_dashboard(self):
        """Stop the dashboard monitoring"""
        self.dashboard_enabled = False
        if self.dashboard_thread:
            self.dashboard_thread.join(timeout=10)

        # Stop subsystems
        self.performance_monitor.stop_monitoring()

        logger.info("🛑 Database Health Dashboard stopped")

    def _dashboard_loop(self):
        """Main dashboard monitoring loop"""
        logger.info("🔄 Dashboard monitoring loop started")

        while self.dashboard_enabled:
            try:
                # Create event loop for async operations
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Run health check
                loop.run_until_complete(self._perform_health_check())

                # Run auto-optimizations if enabled
                if self.auto_optimization_enabled:
                    loop.run_until_complete(self._run_auto_optimizations())

                loop.close()

                # Sleep until next check
                time.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Error in dashboard loop: {e}")
                time.sleep(60)  # Wait before retrying

    async def _perform_health_check(self):
        """Perform comprehensive health check"""
        try:
            # Get health metrics from all systems
            health = await self.get_comprehensive_health()

            # Store in history
            self.health_history.append(health)
            if len(self.health_history) > 288:  # Keep 24 hours (5-min intervals)
                self.health_history = self.health_history[-288:]

            self.last_health_check = datetime.now()

            # Log health status
            status_emoji = {
                'excellent': '🟢',
                'good': '🟡',
                'warning': '🟠',
                'critical': '🔴'
            }
            emoji = status_emoji.get(health.status, '❓')

            logger.info(f"{emoji} Database Health: {health.overall_score:.1f}/100 ({health.status})")

            # Auto-trigger optimizations for critical issues
            if health.status == 'critical':
                await self._emergency_optimization()

        except Exception as e:
            logger.error(f"Health check failed: {e}")

    async def _run_auto_optimizations(self):
        """Run automatic optimizations based on schedule"""
        now = datetime.now()

        # Auto cleanup (hourly)
        if (self.last_cleanup is None or
            (now - self.last_cleanup).total_seconds() > self.auto_cleanup_interval):

            logger.info("🧹 Running scheduled cleanup...")
            await self.retention_manager.execute_full_cleanup()
            self.last_cleanup = now

        # Auto optimization (daily)
        if (self.last_optimization is None or
            (now - self.last_optimization).total_seconds() > self.auto_optimization_interval):

            logger.info("⚡ Running scheduled optimization...")
            await self._run_full_optimization()
            self.last_optimization = now

    async def _emergency_optimization(self):
        """Emergency optimization for critical health issues"""
        logger.warning("🚨 Emergency optimization triggered due to critical health status")

        try:
            # Emergency cleanup
            await self.retention_manager.emergency_cleanup(space_target_mb=100.0)

            # Compress frames immediately
            await self.frame_compression.migrate_all_frame_tables()

            # Optimize indexes
            await self.index_optimizer.optimize_indexes()

            logger.warning("✅ Emergency optimization completed")

        except Exception as e:
            logger.error(f"Emergency optimization failed: {e}")

    async def get_comprehensive_health(self) -> SystemHealth:
        """Get comprehensive health assessment across all systems"""
        try:
            # Collect metrics from all systems
            metrics = await self.get_optimization_metrics()

            # Calculate component scores
            query_score = self._calculate_query_performance_score(metrics.performance_monitoring)
            storage_score = self._calculate_storage_efficiency_score(metrics)
            write_score = self._calculate_write_performance_score(metrics.async_batching)

            # Calculate overall score
            overall_score = (query_score + storage_score + write_score) / 3

            # Determine status
            if overall_score >= 85:
                status = 'excellent'
            elif overall_score >= 70:
                status = 'good'
            elif overall_score >= 50:
                status = 'warning'
            else:
                status = 'critical'

            # Generate recommendations
            recommendations = self._generate_health_recommendations(metrics, overall_score)

            # Get database size
            import os
            db_size = os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0

            return SystemHealth(
                overall_score=overall_score,
                database_size_mb=db_size,
                query_performance_score=query_score,
                storage_efficiency_score=storage_score,
                write_performance_score=write_score,
                last_updated=datetime.now(),
                status=status,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Failed to calculate comprehensive health: {e}")
            return SystemHealth(0, 0, 0, 0, 0, datetime.now(), 'critical', ["Health check failed"])

    def _calculate_query_performance_score(self, perf_metrics: Dict[str, Any]) -> float:
        """Calculate query performance score (0-100)"""
        try:
            health = perf_metrics.get('health', {})
            avg_query_time = health.get('avg_query_time_ms', 100)
            slow_query_count = health.get('slow_query_count', 10)

            # Score based on query performance
            score = 100

            # Penalize slow average query time
            if avg_query_time > 50:
                score -= min(40, (avg_query_time - 50) / 2)

            # Penalize slow queries
            score -= min(30, slow_query_count * 2)

            return max(0, score)

        except Exception:
            return 50.0

    def _calculate_storage_efficiency_score(self, metrics: OptimizationMetrics) -> float:
        """Calculate storage efficiency score (0-100)"""
        try:
            score = 100

            # Database size penalty
            import os
            db_size = os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0

            if db_size > 200:  # Penalize large databases
                score -= min(30, (db_size - 200) / 20)

            # Compression effectiveness bonus
            compression_stats = metrics.frame_compression
            if compression_stats.get('compression_ratio', 1.0) < 0.5:
                score += 10  # Bonus for good compression

            # Partitioning effectiveness bonus
            partition_stats = metrics.data_partitioning
            if partition_stats.get('hot_size_mb', 0) < db_size * 0.3:
                score += 10  # Bonus for effective partitioning

            return max(0, min(100, score))

        except Exception:
            return 50.0

    def _calculate_write_performance_score(self, batch_metrics: Dict[str, Any]) -> float:
        """Calculate write performance score (0-100)"""
        try:
            if not batch_metrics.get('running', False):
                return 30  # Low score if batching not running

            stats = batch_metrics.get('statistics', {})
            throughput = stats.get('write_throughput_per_second', 0)
            avg_batch_size = stats.get('average_batch_size', 1)

            score = 50  # Base score for running batching

            # Bonus for high throughput
            if throughput > 10:
                score += min(30, throughput * 2)

            # Bonus for good batch efficiency
            if avg_batch_size > 20:
                score += min(20, avg_batch_size / 5)

            return min(100, score)

        except Exception:
            return 30.0

    def _generate_health_recommendations(self, metrics: OptimizationMetrics, overall_score: float) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []

        if overall_score < 50:
            recommendations.append("🚨 Critical: Run emergency database optimization")

        # Database size recommendations
        import os
        db_size = os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0

        if db_size > 300:
            recommendations.append("💾 Large database detected - run data cleanup and compression")

        # Performance recommendations
        perf_health = metrics.performance_monitoring.get('health', {})
        if perf_health.get('avg_query_time_ms', 0) > 100:
            recommendations.append("🐌 Slow queries detected - optimize indexes and query patterns")

        # Batching recommendations
        if not metrics.async_batching.get('running', False):
            recommendations.append("⚡ Enable async write batching for better performance")

        # Compression recommendations
        compression_ratio = metrics.frame_compression.get('compression_ratio', 1.0)
        if compression_ratio > 0.7:
            recommendations.append("📦 Frame compression can be improved - check compression settings")

        if not recommendations:
            recommendations.append("✅ Database health is good - no immediate actions needed")

        return recommendations

    async def get_optimization_metrics(self) -> OptimizationMetrics:
        """Collect metrics from all optimization systems"""
        try:
            # Get metrics from all systems
            retention_stats = {'last_cleanup': self.last_cleanup}  # Simplified for now
            index_stats = {}  # Would need to implement metrics collection

            # Async batching metrics
            try:
                batcher = await get_global_batcher(self.db_path)
                batch_metrics = await batcher.health_check()
            except Exception:
                batch_metrics = {'running': False}

            # Frame compression metrics
            try:
                compression_metrics = await self.frame_compression.get_compression_statistics()
            except Exception:
                compression_metrics = {}

            # Performance metrics
            try:
                perf_metrics = self.performance_monitor.get_performance_summary()
            except Exception:
                perf_metrics = {}

            # Partitioning metrics
            try:
                partition_metrics = await self.partitioning_system.get_partition_statistics()
            except Exception:
                partition_metrics = {}

            return OptimizationMetrics(
                retention_cleanup=retention_stats,
                index_optimization=index_stats,
                async_batching=batch_metrics,
                frame_compression=compression_metrics,
                performance_monitoring=perf_metrics,
                data_partitioning=partition_metrics
            )

        except Exception as e:
            logger.error(f"Failed to collect optimization metrics: {e}")
            return OptimizationMetrics({}, {}, {}, {}, {}, {})

    async def _run_full_optimization(self):
        """Run complete optimization across all systems"""
        logger.info("🔧 Running full database optimization...")

        try:
            # 1. Data retention cleanup
            await self.retention_manager.execute_full_cleanup()

            # 2. Index optimization
            await self.index_optimizer.optimize_indexes()

            # 3. Frame compression
            await self.frame_compression.migrate_all_frame_tables()

            # 4. Data partitioning
            await self.partitioning_system.partition_all_tables()

            # 5. Performance optimization
            await self.partitioning_system.optimize_partition_performance()

            logger.info("✅ Full optimization completed successfully")

        except Exception as e:
            logger.error(f"Full optimization failed: {e}")

    async def generate_health_report(self) -> str:
        """Generate comprehensive health report"""
        health = await self.get_comprehensive_health()
        metrics = await self.get_optimization_metrics()

        report = []
        report.append("=" * 80)
        report.append("DATABASE HEALTH DASHBOARD REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Database: {self.db_path}")
        report.append("")

        # Overall health
        status_emoji = {'excellent': '🟢', 'good': '🟡', 'warning': '🟠', 'critical': '🔴'}
        emoji = status_emoji.get(health.status, '❓')
        report.append(f"{emoji} OVERALL HEALTH: {health.overall_score:.1f}/100 ({health.status.upper()})")
        report.append("")

        # Component scores
        report.append("📊 COMPONENT SCORES:")
        report.append(f"   Query Performance: {health.query_performance_score:.1f}/100")
        report.append(f"   Storage Efficiency: {health.storage_efficiency_score:.1f}/100")
        report.append(f"   Write Performance: {health.write_performance_score:.1f}/100")
        report.append("")

        # Database metrics
        report.append("💾 DATABASE METRICS:")
        report.append(f"   Database Size: {health.database_size_mb:.1f} MB")

        # System status
        report.append("🔧 OPTIMIZATION SYSTEMS:")
        report.append(f"   Async Batching: {'✅ Running' if metrics.async_batching.get('running') else '❌ Stopped'}")
        report.append(f"   Performance Monitor: {'✅ Active' if self.performance_monitor.monitoring_enabled else '❌ Inactive'}")
        report.append(f"   Auto Optimization: {'✅ Enabled' if self.auto_optimization_enabled else '❌ Disabled'}")
        report.append("")

        # Recommendations
        if health.recommendations:
            report.append("💡 RECOMMENDATIONS:")
            for rec in health.recommendations:
                report.append(f"   • {rec}")
            report.append("")

        # Health history trend
        if len(self.health_history) > 1:
            recent_scores = [h.overall_score for h in self.health_history[-12:]]  # Last hour
            trend = "📈 Improving" if recent_scores[-1] > recent_scores[0] else "📉 Declining"
            report.append(f"📈 HEALTH TREND (last hour): {trend}")
            report.append("")

        return "\n".join(report)

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get dashboard summary for API/web interface"""
        try:
            # Get latest health from history
            latest_health = self.health_history[-1] if self.health_history else None

            return {
                'dashboard_enabled': self.dashboard_enabled,
                'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
                'last_cleanup': self.last_cleanup.isoformat() if self.last_cleanup else None,
                'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None,
                'auto_optimization_enabled': self.auto_optimization_enabled,
                'current_health': asdict(latest_health) if latest_health else None,
                'health_history_count': len(self.health_history),
                'performance_monitoring': self.performance_monitor.monitoring_enabled
            }

        except Exception as e:
            logger.error(f"Failed to get dashboard summary: {e}")
            return {'error': str(e)}


# Global dashboard instance
_global_dashboard: Optional[DatabaseHealthDashboard] = None

async def get_global_dashboard(db_path: str = "tabula_rasa.db") -> DatabaseHealthDashboard:
    """Get or create global dashboard instance"""
    global _global_dashboard

    if _global_dashboard is None:
        _global_dashboard = DatabaseHealthDashboard(db_path)
        await _global_dashboard.initialize_dashboard()
        _global_dashboard.start_dashboard()

    return _global_dashboard

async def shutdown_global_dashboard():
    """Shutdown global dashboard"""
    global _global_dashboard
    if _global_dashboard:
        _global_dashboard.stop_dashboard()
        _global_dashboard = None

# Convenience functions
async def get_database_health(db_path: str = "tabula_rasa.db") -> SystemHealth:
    """Get current database health"""
    dashboard = await get_global_dashboard(db_path)
    return await dashboard.get_comprehensive_health()

async def run_database_optimization(db_path: str = "tabula_rasa.db"):
    """Run complete database optimization"""
    dashboard = await get_global_dashboard(db_path)
    await dashboard._run_full_optimization()

async def generate_health_report(db_path: str = "tabula_rasa.db") -> str:
    """Generate health report"""
    dashboard = await get_global_dashboard(db_path)
    return await dashboard.generate_health_report()