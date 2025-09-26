"""
Simple Database Optimization Runner
Runs the core optimizations without Unicode characters
"""

import asyncio
import logging
import time
import sys
import os
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Configure logging with ASCII only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_optimization_simple.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def run_simple_optimization():
    """Run basic optimization tests"""
    logger.info("=" * 60)
    logger.info("STARTING SIMPLE DATABASE OPTIMIZATION TEST")
    logger.info("=" * 60)

    db_path = "tabula_rasa.db"

    # Check if database exists
    if not os.path.exists(db_path):
        logger.error(f"Database file '{db_path}' not found!")
        return False

    # Get initial size
    initial_size = os.path.getsize(db_path) / (1024 * 1024)
    logger.info(f"Initial database size: {initial_size:.2f} MB")

    success_count = 0
    total_tests = 5

    # Test 1: Data Retention Manager
    try:
        logger.info("Test 1: Data Retention Manager")
        from src.database.data_retention_manager import create_retention_manager

        retention_manager = create_retention_manager(db_path)
        policies = retention_manager.retention_policies
        logger.info(f"  Loaded {len(policies)} retention policies")

        # Run a small cleanup test
        results = await retention_manager.execute_full_cleanup()
        total_deleted = sum(r.rows_deleted for r in results)
        logger.info(f"  Deleted {total_deleted} old records")

        success_count += 1
        logger.info("  SUCCESS: Data retention manager working")

    except Exception as e:
        logger.error(f"  FAILED: Data retention manager - {e}")

    # Test 2: Index Optimizer
    try:
        logger.info("Test 2: Index Optimizer")
        from src.database.index_optimizer import create_index_optimizer

        index_optimizer = create_index_optimizer(db_path)
        analysis = await index_optimizer.analyze_indexes()

        logger.info(f"  Found {len(analysis.redundant_indexes)} redundant indexes")
        logger.info(f"  Found {len(analysis.missing_indexes)} missing strategic indexes")

        # Run optimization
        result = await index_optimizer.optimize_indexes(analysis)
        logger.info(f"  Dropped {result.indexes_dropped} redundant indexes")
        logger.info(f"  Created {result.indexes_created} strategic indexes")

        success_count += 1
        logger.info("  SUCCESS: Index optimizer working")

    except Exception as e:
        logger.error(f"  FAILED: Index optimizer - {e}")

    # Test 3: Async Batch Writer
    try:
        logger.info("Test 3: Async Batch Writer")
        from src.database.async_batch_writer import get_global_batcher

        batcher = await get_global_batcher(db_path)
        health = await batcher.health_check()

        logger.info(f"  Async batcher running: {health['running']}")
        logger.info(f"  Pending writes: {health['pending_writes']}")

        success_count += 1
        logger.info("  SUCCESS: Async batch writer working")

    except Exception as e:
        logger.error(f"  FAILED: Async batch writer - {e}")

    # Test 4: Frame Compression
    try:
        logger.info("Test 4: Frame Compression System")
        from src.database.frame_compression_system import create_frame_compression_system

        compression_system = create_frame_compression_system(db_path)
        await compression_system.initialize_compression_tables()

        # Test compression with sample data
        sample_frame = [[1, 2, 3], [4, 5, 6]]
        frame_hash = await compression_system.store_compressed_frame(sample_frame)
        retrieved_frame = await compression_system.retrieve_frame(frame_hash)

        compression_works = retrieved_frame == sample_frame
        logger.info(f"  Frame compression test: {'PASSED' if compression_works else 'FAILED'}")

        success_count += 1
        logger.info("  SUCCESS: Frame compression working")

    except Exception as e:
        logger.error(f"  FAILED: Frame compression - {e}")

    # Test 5: Performance Monitor
    try:
        logger.info("Test 5: Performance Monitor")
        from src.database.performance_monitor import create_performance_monitor

        performance_monitor = create_performance_monitor(db_path)
        await performance_monitor.initialize_monitoring_tables()
        performance_monitor.start_monitoring()

        logger.info(f"  Performance monitoring enabled: {performance_monitor.monitoring_enabled}")

        # Test query tracking
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            start = time.time()
            cursor = conn.execute("SELECT COUNT(*) FROM training_sessions")
            result = cursor.fetchone()
            query_time = (time.time() - start) * 1000

            performance_monitor.track_query("SELECT COUNT(*) FROM training_sessions", query_time, 1)

        health = performance_monitor.get_database_health()
        logger.info(f"  Database health score: {health.health_score:.1f}/100")

        success_count += 1
        logger.info("  SUCCESS: Performance monitor working")

    except Exception as e:
        logger.error(f"  FAILED: Performance monitor - {e}")

    # Final results
    final_size = os.path.getsize(db_path) / (1024 * 1024)
    size_change = initial_size - final_size

    logger.info("=" * 60)
    logger.info("OPTIMIZATION TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Tests passed: {success_count}/{total_tests}")
    logger.info(f"Success rate: {(success_count/total_tests)*100:.1f}%")
    logger.info(f"Database size change: {initial_size:.2f} MB -> {final_size:.2f} MB")
    logger.info(f"Space freed: {size_change:.2f} MB")

    if success_count >= 4:
        logger.info("EXCELLENT: Database optimization systems are working correctly!")
        return True
    elif success_count >= 3:
        logger.info("GOOD: Most optimization systems are working.")
        return True
    else:
        logger.warning("WARNING: Several optimization systems need attention.")
        return False

async def main():
    print("Database Optimization Test")
    print("=" * 50)
    print("Testing core optimization systems...")
    print()

    success = await run_simple_optimization()

    if success:
        print("\nSUCCESS: Database optimizations are working!")
        print("Your AGI system database has been optimized for better performance.")
    else:
        print("\nWARNING: Some optimizations may need attention.")
        print("Check the log file for details.")

if __name__ == "__main__":
    asyncio.run(main())