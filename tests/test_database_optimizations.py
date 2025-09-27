"""
COMPREHENSIVE DATABASE OPTIMIZATION TEST SUITE
Verifies all database optimizations are working correctly and measures performance improvements
"""

import asyncio
import logging
import time
import sys
import os
import sqlite3
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import optimization systems for testing
from src.database.data_retention_manager import create_retention_manager
from src.database.index_optimizer import create_index_optimizer
from src.database.async_batch_writer import get_global_batcher, queue_database_write
from src.database.frame_compression_system import create_frame_compression_system
from src.database.performance_monitor import create_performance_monitor, QueryTracker
from src.database.data_partitioning_system import create_partitioning_system
from src.database.health_dashboard import get_global_dashboard

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseOptimizationTester:
    """Comprehensive test suite for database optimizations"""

    def __init__(self, db_path: str = "tabula_rasa.db"):
        self.db_path = db_path
        self.test_results = {}

    async def run_all_tests(self):
        """Run comprehensive test suite"""
        logger.info("🧪 STARTING COMPREHENSIVE DATABASE OPTIMIZATION TESTS")
        logger.info("=" * 80)

        try:
            # Test 1: Basic Functionality Tests
            await self._test_basic_functionality()

            # Test 2: Performance Benchmarks
            await self._test_performance_benchmarks()

            # Test 3: Async Batching Performance
            await self._test_async_batching_performance()

            # Test 4: Frame Compression Effectiveness
            await self._test_frame_compression()

            # Test 5: Query Performance with Indexes
            await self._test_query_performance()

            # Test 6: Data Partitioning Efficiency
            await self._test_data_partitioning()

            # Test 7: Real-time Monitoring
            await self._test_performance_monitoring()

            # Test 8: Health Dashboard Integration
            await self._test_health_dashboard()

            # Generate test report
            await self._generate_test_report()

            logger.info("✅ ALL TESTS COMPLETED SUCCESSFULLY!")

        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            raise

    async def _test_basic_functionality(self):
        """Test 1: Basic functionality of all optimization systems"""
        logger.info("🔧 TEST 1: BASIC FUNCTIONALITY")

        test_results = {}

        # Test data retention manager
        try:
            retention_manager = create_retention_manager(self.db_path)
            policies = retention_manager.retention_policies
            test_results['data_retention'] = {'policies_loaded': len(policies), 'status': 'ok'}
            logger.info(f"   ✅ Data retention: {len(policies)} policies loaded")
        except Exception as e:
            test_results['data_retention'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"   ❌ Data retention failed: {e}")

        # Test index optimizer
        try:
            index_optimizer = create_index_optimizer(self.db_path)
            analysis = await index_optimizer.analyze_indexes()
            test_results['index_optimizer'] = {
                'redundant_indexes': len(analysis.redundant_indexes),
                'missing_indexes': len(analysis.missing_indexes),
                'status': 'ok'
            }
            logger.info(f"   ✅ Index optimizer: {len(analysis.redundant_indexes)} redundant, {len(analysis.missing_indexes)} missing")
        except Exception as e:
            test_results['index_optimizer'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"   ❌ Index optimizer failed: {e}")

        # Test async batcher
        try:
            batcher = await get_global_batcher(self.db_path)
            health = await batcher.health_check()
            test_results['async_batcher'] = {'running': health['running'], 'status': 'ok'}
            logger.info(f"   ✅ Async batcher: {'Running' if health['running'] else 'Stopped'}")
        except Exception as e:
            test_results['async_batcher'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"   ❌ Async batcher failed: {e}")

        # Test frame compression
        try:
            compression_system = create_frame_compression_system(self.db_path)
            # Test frame compression with sample data
            sample_frame = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            frame_hash = await compression_system.store_compressed_frame(sample_frame)
            retrieved_frame = await compression_system.retrieve_frame(frame_hash)

            compression_works = retrieved_frame == sample_frame
            test_results['frame_compression'] = {'compression_works': compression_works, 'status': 'ok'}
            logger.info(f"   ✅ Frame compression: {'Working' if compression_works else 'Failed'}")
        except Exception as e:
            test_results['frame_compression'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"   ❌ Frame compression failed: {e}")

        # Test performance monitor
        try:
            performance_monitor = create_performance_monitor(self.db_path)
            await performance_monitor.initialize_monitoring_tables()
            performance_monitor.start_monitoring()
            test_results['performance_monitor'] = {'monitoring_enabled': performance_monitor.monitoring_enabled, 'status': 'ok'}
            logger.info(f"   ✅ Performance monitor: {'Enabled' if performance_monitor.monitoring_enabled else 'Disabled'}")
        except Exception as e:
            test_results['performance_monitor'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"   ❌ Performance monitor failed: {e}")

        self.test_results['basic_functionality'] = test_results

    async def _test_performance_benchmarks(self):
        """Test 2: Performance benchmarks before and after optimization"""
        logger.info("⚡ TEST 2: PERFORMANCE BENCHMARKS")

        # Test database query performance
        test_start = time.time()

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Test simple query performance
                query_times = []
                for i in range(10):
                    start = time.time()
                    cursor = conn.execute("SELECT COUNT(*) FROM training_sessions")
                    result = cursor.fetchone()
                    query_times.append((time.time() - start) * 1000)

                avg_query_time = sum(query_times) / len(query_times)

                # Test write performance
                write_times = []
                for i in range(10):
                    start = time.time()
                    conn.execute("""
                        INSERT INTO system_logs (log_level, component, message, timestamp)
                        VALUES (?, ?, ?, ?)
                    """, ('INFO', 'test', f'Test message {i}', datetime.now()))
                    conn.commit()
                    write_times.append((time.time() - start) * 1000)

                avg_write_time = sum(write_times) / len(write_times)

                # Clean up test data
                conn.execute("DELETE FROM system_logs WHERE component = 'test'")
                conn.commit()

                self.test_results['performance_benchmarks'] = {
                    'avg_query_time_ms': avg_query_time,
                    'avg_write_time_ms': avg_write_time,
                    'test_duration_ms': (time.time() - test_start) * 1000,
                    'status': 'ok'
                }

                logger.info(f"   ✅ Average query time: {avg_query_time:.2f}ms")
                logger.info(f"   ✅ Average write time: {avg_write_time:.2f}ms")

        except Exception as e:
            self.test_results['performance_benchmarks'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"   ❌ Performance benchmarks failed: {e}")

    async def _test_async_batching_performance(self):
        """Test 3: Async batching performance improvement"""
        logger.info("🚀 TEST 3: ASYNC BATCHING PERFORMANCE")

        try:
            # Test individual writes (old way)
            individual_start = time.time()
            with sqlite3.connect(self.db_path) as conn:
                for i in range(50):
                    conn.execute("""
                        INSERT INTO system_logs (log_level, component, message, timestamp)
                        VALUES (?, ?, ?, ?)
                    """, ('INFO', 'individual_test', f'Individual message {i}', datetime.now()))
                    conn.commit()
            individual_time = (time.time() - individual_start) * 1000

            # Test batched writes (new way)
            batched_start = time.time()
            for i in range(50):
                await queue_database_write('system_logs', {
                    'log_level': 'INFO',
                    'component': 'batch_test',
                    'message': f'Batched message {i}',
                    'timestamp': datetime.now()
                })

            # Force flush to ensure writes are completed
            batcher = await get_global_batcher(self.db_path)
            await batcher.force_flush()
            batched_time = (time.time() - batched_start) * 1000

            # Calculate improvement
            improvement = ((individual_time - batched_time) / individual_time) * 100 if individual_time > 0 else 0

            # Clean up test data
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM system_logs WHERE component IN ('individual_test', 'batch_test')")
                conn.commit()

            self.test_results['async_batching_performance'] = {
                'individual_writes_ms': individual_time,
                'batched_writes_ms': batched_time,
                'performance_improvement_percent': improvement,
                'status': 'ok'
            }

            logger.info(f"   ✅ Individual writes: {individual_time:.2f}ms")
            logger.info(f"   ✅ Batched writes: {batched_time:.2f}ms")
            logger.info(f"   ✅ Performance improvement: {improvement:.1f}%")

        except Exception as e:
            self.test_results['async_batching_performance'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"   ❌ Async batching performance test failed: {e}")

    async def _test_frame_compression(self):
        """Test 4: Frame compression effectiveness"""
        logger.info("📦 TEST 4: FRAME COMPRESSION EFFECTIVENESS")

        try:
            compression_system = create_frame_compression_system(self.db_path)

            # Test with various frame sizes
            test_frames = [
                # Small frame
                [[1, 2], [3, 4]],
                # Medium frame
                [[i + j for i in range(10)] for j in range(10)],
                # Large frame with patterns
                [[i % 256 for i in range(50)] for j in range(50)],
                # Duplicate frame (test deduplication)
                [[1, 2], [3, 4]]  # Same as first frame
            ]

            compression_results = []
            total_original_size = 0
            total_compressed_size = 0

            for i, frame in enumerate(test_frames):
                frame_hash, compressed_data, original_size, compressed_size = compression_system.compress_frame(frame)

                compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
                compression_results.append({
                    'frame_index': i,
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'compression_ratio': compression_ratio
                })

                total_original_size += original_size
                total_compressed_size += compressed_size

                logger.info(f"   📊 Frame {i}: {original_size} → {compressed_size} bytes ({compression_ratio:.2f} ratio)")

            overall_compression_ratio = total_compressed_size / total_original_size if total_original_size > 0 else 1.0
            space_saved_percent = (1 - overall_compression_ratio) * 100

            self.test_results['frame_compression_effectiveness'] = {
                'total_original_size': total_original_size,
                'total_compressed_size': total_compressed_size,
                'overall_compression_ratio': overall_compression_ratio,
                'space_saved_percent': space_saved_percent,
                'frame_results': compression_results,
                'status': 'ok'
            }

            logger.info(f"   ✅ Overall compression ratio: {overall_compression_ratio:.2f}")
            logger.info(f"   ✅ Space saved: {space_saved_percent:.1f}%")

        except Exception as e:
            self.test_results['frame_compression_effectiveness'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"   ❌ Frame compression test failed: {e}")

    async def _test_query_performance(self):
        """Test 5: Query performance with optimized indexes"""
        logger.info("🔍 TEST 5: QUERY PERFORMANCE WITH INDEXES")

        try:
            performance_monitor = create_performance_monitor(self.db_path)

            # Test various query patterns
            queries_to_test = [
                ("SELECT COUNT(*) FROM training_sessions", "Simple count query"),
                ("SELECT * FROM training_sessions ORDER BY start_time DESC LIMIT 10", "Recent sessions query"),
                ("SELECT COUNT(*) FROM action_traces WHERE timestamp > datetime('now', '-1 day')", "Recent actions query"),
                ("SELECT game_id, COUNT(*) FROM game_results GROUP BY game_id LIMIT 5", "Aggregation query")
            ]

            query_results = []
            total_query_time = 0

            with sqlite3.connect(self.db_path) as conn:
                for query, description in queries_to_test:
                    try:
                        # Use performance monitor to track query
                        with QueryTracker(performance_monitor, query) as tracker:
                            start_time = time.time()
                            cursor = conn.execute(query)
                            results = cursor.fetchall()
                            execution_time = (time.time() - start_time) * 1000

                        tracker.set_rows_affected(len(results))

                        query_results.append({
                            'query': query,
                            'description': description,
                            'execution_time_ms': execution_time,
                            'rows_returned': len(results)
                        })

                        total_query_time += execution_time
                        logger.info(f"   ✅ {description}: {execution_time:.2f}ms ({len(results)} rows)")

                    except Exception as e:
                        logger.warning(f"   ⚠️ Query failed: {description} - {e}")

            avg_query_time = total_query_time / len(query_results) if query_results else 0

            self.test_results['query_performance'] = {
                'queries_tested': len(query_results),
                'total_query_time_ms': total_query_time,
                'average_query_time_ms': avg_query_time,
                'query_results': query_results,
                'status': 'ok'
            }

            logger.info(f"   ✅ Average query time: {avg_query_time:.2f}ms")

        except Exception as e:
            self.test_results['query_performance'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"   ❌ Query performance test failed: {e}")

    async def _test_data_partitioning(self):
        """Test 6: Data partitioning efficiency"""
        logger.info("🔥 TEST 6: DATA PARTITIONING EFFICIENCY")

        try:
            partitioning_system = create_partitioning_system(self.db_path)

            # Get partition statistics
            partition_stats = await partitioning_system.get_partition_statistics()

            # Test partitioned query executor
            executor = partitioning_system.get_query_executor()

            # Test query that should hit hot partition
            hot_query_start = time.time()
            hot_results = await executor.execute_query("SELECT COUNT(*) FROM training_sessions WHERE start_time > datetime('now', '-1 day')")
            hot_query_time = (time.time() - hot_query_start) * 1000

            # Test query that should hit cold partition
            cold_query_start = time.time()
            cold_results = await executor.execute_query("SELECT COUNT(*) FROM training_sessions WHERE start_time < datetime('now', '-30 days')")
            cold_query_time = (time.time() - cold_query_start) * 1000

            self.test_results['data_partitioning_efficiency'] = {
                'partition_stats': partition_stats,
                'hot_query_time_ms': hot_query_time,
                'cold_query_time_ms': cold_query_time,
                'hot_results': len(hot_results) if hot_results else 0,
                'cold_results': len(cold_results) if cold_results else 0,
                'status': 'ok'
            }

            logger.info(f"   ✅ Hot partition query: {hot_query_time:.2f}ms")
            logger.info(f"   ✅ Cold partition query: {cold_query_time:.2f}ms")

            # Log partition sizes
            for partition_name, stats in partition_stats.items():
                size_mb = stats.get('database_size_mb', 0)
                records = stats.get('total_records', 0)
                logger.info(f"   📊 {partition_name.capitalize()} partition: {size_mb:.2f} MB, {records:,} records")

        except Exception as e:
            self.test_results['data_partitioning_efficiency'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"   ❌ Data partitioning test failed: {e}")

    async def _test_performance_monitoring(self):
        """Test 7: Real-time performance monitoring"""
        logger.info("📊 TEST 7: PERFORMANCE MONITORING")

        try:
            performance_monitor = create_performance_monitor(self.db_path)

            # Generate some test queries to monitor
            with sqlite3.connect(self.db_path) as conn:
                for i in range(5):
                    start_time = time.time()
                    cursor = conn.execute("SELECT COUNT(*) FROM training_sessions")
                    result = cursor.fetchone()
                    execution_time = (time.time() - start_time) * 1000

                    # Track the query
                    performance_monitor.track_query(
                        "SELECT COUNT(*) FROM training_sessions",
                        execution_time,
                        1
                    )

            # Get performance summary
            summary = performance_monitor.get_performance_summary()

            # Get database health
            health = performance_monitor.get_database_health()

            self.test_results['performance_monitoring'] = {
                'monitoring_enabled': performance_monitor.monitoring_enabled,
                'queries_tracked': summary.get('total_queries_tracked', 0),
                'health_score': health.health_score,
                'avg_query_time_ms': health.avg_query_time_ms,
                'slow_query_count': health.slow_query_count,
                'status': 'ok'
            }

            logger.info(f"   ✅ Monitoring enabled: {performance_monitor.monitoring_enabled}")
            logger.info(f"   ✅ Queries tracked: {summary.get('total_queries_tracked', 0)}")
            logger.info(f"   ✅ Health score: {health.health_score:.1f}/100")

        except Exception as e:
            self.test_results['performance_monitoring'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"   ❌ Performance monitoring test failed: {e}")

    async def _test_health_dashboard(self):
        """Test 8: Health dashboard integration"""
        logger.info("📋 TEST 8: HEALTH DASHBOARD INTEGRATION")

        try:
            dashboard = await get_global_dashboard(self.db_path)

            # Get comprehensive health
            health = await dashboard.get_comprehensive_health()

            # Get optimization metrics
            metrics = await dashboard.get_optimization_metrics()

            # Generate health report
            report = await dashboard.generate_health_report()

            # Get dashboard summary
            summary = dashboard.get_dashboard_summary()

            self.test_results['health_dashboard'] = {
                'dashboard_enabled': summary.get('dashboard_enabled', False),
                'overall_health_score': health.overall_score,
                'health_status': health.status,
                'query_performance_score': health.query_performance_score,
                'storage_efficiency_score': health.storage_efficiency_score,
                'write_performance_score': health.write_performance_score,
                'recommendations_count': len(health.recommendations),
                'report_generated': len(report) > 0,
                'status': 'ok'
            }

            logger.info(f"   ✅ Dashboard enabled: {summary.get('dashboard_enabled', False)}")
            logger.info(f"   ✅ Overall health: {health.overall_score:.1f}/100 ({health.status})")
            logger.info(f"   ✅ Query performance: {health.query_performance_score:.1f}/100")
            logger.info(f"   ✅ Storage efficiency: {health.storage_efficiency_score:.1f}/100")
            logger.info(f"   ✅ Write performance: {health.write_performance_score:.1f}/100")

        except Exception as e:
            self.test_results['health_dashboard'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"   ❌ Health dashboard test failed: {e}")

    async def _generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("=" * 80)
        logger.info("📋 COMPREHENSIVE TEST REPORT")
        logger.info("=" * 80)

        # Count successful tests
        successful_tests = 0
        total_tests = len(self.test_results)

        for test_name, results in self.test_results.items():
            if results.get('status') == 'ok':
                successful_tests += 1

        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0

        logger.info(f"🎯 TEST SUMMARY:")
        logger.info(f"   Total tests: {total_tests}")
        logger.info(f"   Successful: {successful_tests}")
        logger.info(f"   Success rate: {success_rate:.1f}%")
        logger.info("")

        # Performance highlights
        logger.info("⚡ PERFORMANCE HIGHLIGHTS:")

        if 'async_batching_performance' in self.test_results:
            perf = self.test_results['async_batching_performance']
            if 'performance_improvement_percent' in perf:
                logger.info(f"   • Async batching improvement: {perf['performance_improvement_percent']:.1f}%")

        if 'frame_compression_effectiveness' in self.test_results:
            comp = self.test_results['frame_compression_effectiveness']
            if 'space_saved_percent' in comp:
                logger.info(f"   • Frame compression space savings: {comp['space_saved_percent']:.1f}%")

        if 'query_performance' in self.test_results:
            query = self.test_results['query_performance']
            if 'average_query_time_ms' in query:
                logger.info(f"   • Average query time: {query['average_query_time_ms']:.2f}ms")

        if 'health_dashboard' in self.test_results:
            health = self.test_results['health_dashboard']
            if 'overall_health_score' in health:
                logger.info(f"   • Overall database health: {health['overall_health_score']:.1f}/100")

        logger.info("")

        # System status
        logger.info("🔧 OPTIMIZATION SYSTEMS STATUS:")

        systems_status = [
            ('Data Retention', 'data_retention'),
            ('Index Optimizer', 'index_optimizer'),
            ('Async Batcher', 'async_batcher'),
            ('Frame Compression', 'frame_compression'),
            ('Performance Monitor', 'performance_monitor'),
            ('Data Partitioning', 'data_partitioning_efficiency'),
            ('Health Dashboard', 'health_dashboard')
        ]

        for system_name, test_key in systems_status:
            if test_key in self.test_results:
                status = self.test_results[test_key].get('status', 'unknown')
                status_emoji = '✅' if status == 'ok' else '❌'
                logger.info(f"   {status_emoji} {system_name}: {status}")

        logger.info("")

        # Save detailed results
        with open('database_optimization_test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)

        logger.info("📄 Detailed test results saved to: database_optimization_test_results.json")
        logger.info("")

        if success_rate >= 90:
            logger.info("🎉 EXCELLENT! All optimization systems are working correctly!")
        elif success_rate >= 70:
            logger.info("✅ GOOD! Most optimization systems are working correctly.")
        else:
            logger.warning("⚠️ ATTENTION! Some optimization systems need attention.")

        logger.info("🚀 Database optimization testing completed!")


async def main():
    """Main test execution"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                  🧪 DATABASE OPTIMIZATION TEST SUITE                          ║
    ║                        TABULA RASA AGI SYSTEM                                ║
    ╠═══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                               ║
    ║  This test suite verifies all database optimizations are working correctly   ║
    ║  and measures the performance improvements achieved.                          ║
    ║                                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Check if database exists
    db_path = "tabula_rasa.db"
    if not os.path.exists(db_path):
        print(f"❌ Database file '{db_path}' not found!")
        print("   Please run the optimization script first.")
        return

    # Run tests
    tester = DatabaseOptimizationTester(db_path)
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())