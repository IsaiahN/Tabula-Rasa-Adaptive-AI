"""
Advanced Frame Deduplication and Compression System
Dramatically reduces database size by deduplicating and compressing frame data
"""

import sqlite3
import asyncio
import logging
import hashlib
import time
import json
import zlib
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class FrameCompressionStats:
    """Statistics for frame compression operations"""
    frames_processed: int
    duplicate_frames_found: int
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    processing_time_ms: float
    deduplication_savings_mb: float

@dataclass
class CompressedFrame:
    """Represents a compressed frame with metadata"""
    frame_hash: str
    compressed_data: bytes
    original_size: int
    compressed_size: int
    compression_method: str
    width: int
    height: int
    created_at: datetime

class FrameCompressionSystem:
    """Advanced frame compression and deduplication system"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.frame_cache = {}  # hash -> CompressedFrame
        self.hash_to_id = {}   # hash -> frame_id in database
        self.compression_stats = FrameCompressionStats(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    async def initialize_compression_tables(self):
        """Create tables for compressed frame storage"""
        logger.info("🗃️  Initializing frame compression tables...")

        with sqlite3.connect(self.db_path) as conn:
            # Create compressed frames table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compressed_frames (
                    frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame_hash TEXT UNIQUE NOT NULL,
                    compressed_data BLOB NOT NULL,
                    original_size INTEGER NOT NULL,
                    compressed_size INTEGER NOT NULL,
                    compression_method TEXT DEFAULT 'zlib',
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    reference_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create frame references table for tracking usage
            conn.execute("""
                CREATE TABLE IF NOT EXISTS frame_references (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame_id INTEGER NOT NULL,
                    table_name TEXT NOT NULL,
                    row_id INTEGER NOT NULL,
                    column_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (frame_id) REFERENCES compressed_frames(frame_id),
                    UNIQUE(table_name, row_id, column_name)
                )
            """)

            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_compressed_frames_hash ON compressed_frames(frame_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_frame_references_frame_id ON frame_references(frame_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_frame_references_table ON frame_references(table_name, row_id)")

            conn.commit()

        logger.info("✅ Frame compression tables initialized")

    def compress_frame(self, frame_data: Any, compression_method: str = "zlib") -> Tuple[str, bytes, int, int]:
        """Compress a single frame and return hash, compressed data, and sizes"""

        # Normalize frame data to consistent format
        if isinstance(frame_data, list):
            # Convert list to numpy array for consistent hashing
            frame_array = np.array(frame_data, dtype=np.uint8)
        elif isinstance(frame_data, np.ndarray):
            frame_array = frame_data.astype(np.uint8)
        else:
            # Convert other formats to string then to bytes
            frame_str = str(frame_data)
            frame_array = np.frombuffer(frame_str.encode(), dtype=np.uint8)

        # Calculate hash for deduplication
        frame_hash = hashlib.md5(frame_array.tobytes()).hexdigest()

        # Get dimensions
        if len(frame_array.shape) >= 2:
            height, width = frame_array.shape[:2]
        else:
            height, width = 1, len(frame_array)

        # Serialize frame data
        if compression_method == "pickle":
            serialized_data = pickle.dumps(frame_data)
        else:
            # Default to JSON serialization
            serialized_data = json.dumps(frame_data).encode('utf-8')

        original_size = len(serialized_data)

        # Compress the serialized data
        if compression_method == "zlib":
            compressed_data = zlib.compress(serialized_data, level=9)  # Maximum compression
        elif compression_method == "gzip":
            import gzip
            compressed_data = gzip.compress(serialized_data, compresslevel=9)
        else:
            # No compression
            compressed_data = serialized_data

        compressed_size = len(compressed_data)

        return frame_hash, compressed_data, original_size, compressed_size

    def decompress_frame(self, compressed_data: bytes, compression_method: str = "zlib") -> Any:
        """Decompress frame data back to original format"""

        # Decompress the data
        if compression_method == "zlib":
            decompressed_data = zlib.decompress(compressed_data)
        elif compression_method == "gzip":
            import gzip
            decompressed_data = gzip.decompress(compressed_data)
        else:
            decompressed_data = compressed_data

        # Deserialize based on method used during compression
        if compression_method == "pickle":
            return pickle.loads(decompressed_data)
        else:
            # Default JSON deserialization
            return json.loads(decompressed_data.decode('utf-8'))

    async def store_compressed_frame(self, frame_data: Any, compression_method: str = "zlib") -> str:
        """Store a frame with compression and deduplication, return frame hash"""

        frame_hash, compressed_data, original_size, compressed_size = self.compress_frame(frame_data, compression_method)

        # Check if frame already exists (deduplication)
        if frame_hash in self.hash_to_id:
            # Frame already exists, just increment reference count
            await self._increment_reference_count(frame_hash)
            self.compression_stats.duplicate_frames_found += 1
            return frame_hash

        # Store new compressed frame
        with sqlite3.connect(self.db_path) as conn:
            # Get dimensions
            if isinstance(frame_data, (list, np.ndarray)) and len(np.array(frame_data).shape) >= 2:
                height, width = np.array(frame_data).shape[:2]
            else:
                height, width = 1, len(str(frame_data))

            cursor = conn.execute("""
                INSERT INTO compressed_frames
                (frame_hash, compressed_data, original_size, compressed_size, compression_method, width, height)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (frame_hash, compressed_data, original_size, compressed_size, compression_method, width, height))

            frame_id = cursor.lastrowid
            self.hash_to_id[frame_hash] = frame_id

            # Update statistics
            self.compression_stats.frames_processed += 1
            self.compression_stats.original_size_mb += original_size / (1024 * 1024)
            self.compression_stats.compressed_size_mb += compressed_size / (1024 * 1024)

            conn.commit()

        logger.debug(f"Stored compressed frame {frame_hash}: {original_size} -> {compressed_size} bytes "
                    f"({(1 - compressed_size/original_size)*100:.1f}% compression)")

        return frame_hash

    async def retrieve_frame(self, frame_hash: str) -> Any:
        """Retrieve and decompress a frame by hash"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT compressed_data, compression_method
                FROM compressed_frames
                WHERE frame_hash = ?
            """, (frame_hash,))

            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Frame with hash {frame_hash} not found")

            compressed_data, compression_method = row

            # Update last accessed timestamp
            conn.execute("""
                UPDATE compressed_frames
                SET last_accessed = CURRENT_TIMESTAMP
                WHERE frame_hash = ?
            """, (frame_hash,))

            conn.commit()

        # Decompress and return
        return self.decompress_frame(compressed_data, compression_method)

    async def migrate_existing_frames(self, table_name: str, frame_columns: List[str]) -> FrameCompressionStats:
        """Migrate existing frame data to compressed format"""
        logger.info(f"🔄 Migrating existing frames from {table_name}...")

        start_time = time.time()
        migration_stats = FrameCompressionStats(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

        with sqlite3.connect(self.db_path) as conn:
            # Get all rows with frame data
            columns_sql = ", ".join(["id"] + frame_columns)
            cursor = conn.execute(f"SELECT {columns_sql} FROM {table_name}")
            rows = cursor.fetchall()

            logger.info(f"Found {len(rows)} rows to migrate in {table_name}")

            for row in rows:
                row_id = row[0]

                for i, column_name in enumerate(frame_columns):
                    frame_data_raw = row[i + 1]  # +1 because first column is id

                    if frame_data_raw:
                        try:
                            # Parse JSON frame data
                            if isinstance(frame_data_raw, str):
                                frame_data = json.loads(frame_data_raw)
                            else:
                                frame_data = frame_data_raw

                            # Store compressed frame
                            frame_hash = await self.store_compressed_frame(frame_data)

                            # Create reference
                            await self._create_frame_reference(frame_hash, table_name, row_id, column_name)

                            # Update original table to store hash instead of full data
                            conn.execute(f"""
                                UPDATE {table_name}
                                SET {column_name} = ?
                                WHERE id = ?
                            """, (f"frame_hash:{frame_hash}", row_id))

                            migration_stats.frames_processed += 1

                        except Exception as e:
                            logger.error(f"Failed to migrate frame in {table_name}.{column_name}[{row_id}]: {e}")

                # Commit every 100 rows
                if migration_stats.frames_processed % 100 == 0:
                    conn.commit()
                    logger.info(f"Migrated {migration_stats.frames_processed} frames...")

            conn.commit()

        # Calculate final statistics
        migration_stats.processing_time_ms = (time.time() - start_time) * 1000
        migration_stats.compression_ratio = (
            migration_stats.compressed_size_mb / migration_stats.original_size_mb
            if migration_stats.original_size_mb > 0 else 1.0
        )
        migration_stats.deduplication_savings_mb = (
            migration_stats.duplicate_frames_found *
            (migration_stats.original_size_mb / max(1, migration_stats.frames_processed))
        )

        logger.info(f"✅ Migration completed in {migration_stats.processing_time_ms:.0f}ms:")
        logger.info(f"   📊 {migration_stats.frames_processed} frames processed")
        logger.info(f"   🔄 {migration_stats.duplicate_frames_found} duplicates found")
        logger.info(f"   📦 Compression ratio: {migration_stats.compression_ratio:.2f}")
        logger.info(f"   💾 Deduplication savings: {migration_stats.deduplication_savings_mb:.2f} MB")

        return migration_stats

    async def _increment_reference_count(self, frame_hash: str):
        """Increment reference count for existing frame"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE compressed_frames
                SET reference_count = reference_count + 1,
                    last_accessed = CURRENT_TIMESTAMP
                WHERE frame_hash = ?
            """, (frame_hash,))
            conn.commit()

    async def _create_frame_reference(self, frame_hash: str, table_name: str, row_id: int, column_name: str):
        """Create a reference tracking entry"""
        frame_id = self.hash_to_id.get(frame_hash)
        if not frame_id:
            # Look up frame_id from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT frame_id FROM compressed_frames WHERE frame_hash = ?", (frame_hash,))
                row = cursor.fetchone()
                if row:
                    frame_id = row[0]
                    self.hash_to_id[frame_hash] = frame_id

        if frame_id:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO frame_references
                    (frame_id, table_name, row_id, column_name)
                    VALUES (?, ?, ?, ?)
                """, (frame_id, table_name, row_id, column_name))
                conn.commit()

    async def cleanup_unused_frames(self) -> int:
        """Remove compressed frames that are no longer referenced"""
        logger.info("🧹 Cleaning up unused compressed frames...")

        with sqlite3.connect(self.db_path) as conn:
            # Find frames with no references
            cursor = conn.execute("""
                SELECT cf.frame_id, cf.frame_hash
                FROM compressed_frames cf
                LEFT JOIN frame_references fr ON cf.frame_id = fr.frame_id
                WHERE fr.frame_id IS NULL
            """)
            unused_frames = cursor.fetchall()

            if not unused_frames:
                logger.info("✅ No unused frames found")
                return 0

            # Delete unused frames
            unused_ids = [frame_id for frame_id, _ in unused_frames]
            placeholders = ", ".join("?" * len(unused_ids))

            conn.execute(f"DELETE FROM compressed_frames WHERE frame_id IN ({placeholders})", unused_ids)
            deleted_count = conn.total_changes

            conn.commit()

            logger.info(f"🗑️  Deleted {deleted_count} unused compressed frames")
            return deleted_count

    async def get_compression_statistics(self) -> Dict[str, Any]:
        """Get comprehensive compression statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Get compressed frames statistics
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_frames,
                    SUM(original_size) as total_original_size,
                    SUM(compressed_size) as total_compressed_size,
                    AVG(reference_count) as avg_reference_count,
                    MAX(reference_count) as max_reference_count
                FROM compressed_frames
            """)

            stats_row = cursor.fetchone()
            if stats_row:
                total_frames, total_original, total_compressed, avg_refs, max_refs = stats_row
            else:
                total_frames = total_original = total_compressed = avg_refs = max_refs = 0

            # Get reference statistics
            cursor = conn.execute("""
                SELECT table_name, COUNT(*) as reference_count
                FROM frame_references
                GROUP BY table_name
                ORDER BY reference_count DESC
            """)
            table_references = dict(cursor.fetchall())

        # Calculate metrics
        compression_ratio = total_compressed / max(1, total_original)
        space_saved_mb = (total_original - total_compressed) / (1024 * 1024)

        return {
            'total_frames': total_frames or 0,
            'total_original_size_mb': (total_original or 0) / (1024 * 1024),
            'total_compressed_size_mb': (total_compressed or 0) / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'space_saved_mb': space_saved_mb,
            'average_reference_count': avg_refs or 0,
            'max_reference_count': max_refs or 0,
            'table_references': table_references,
            'deduplication_effectiveness': (avg_refs or 0) - 1  # How many duplicates per frame on average
        }

    async def migrate_all_frame_tables(self) -> Dict[str, FrameCompressionStats]:
        """Migrate all tables containing frame data"""
        logger.info("🚀 Starting comprehensive frame migration...")

        # Define tables and their frame columns
        frame_tables = {
            'action_traces': ['frame_before', 'frame_after'],
            'gan_training_data': ['previous_frame', 'current_frame'],
            'frame_tracking': ['frame_analysis'],  # This contains JSON analysis data
            'gan_generated_states': ['state_data']  # This contains game state data
        }

        results = {}
        total_stats = FrameCompressionStats(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

        for table_name, frame_columns in frame_tables.items():
            try:
                logger.info(f"📋 Migrating {table_name}...")
                stats = await self.migrate_existing_frames(table_name, frame_columns)
                results[table_name] = stats

                # Aggregate statistics
                total_stats.frames_processed += stats.frames_processed
                total_stats.duplicate_frames_found += stats.duplicate_frames_found
                total_stats.original_size_mb += stats.original_size_mb
                total_stats.compressed_size_mb += stats.compressed_size_mb

            except Exception as e:
                logger.error(f"Failed to migrate {table_name}: {e}")
                results[table_name] = FrameCompressionStats(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Calculate totals
        total_stats.compression_ratio = (
            total_stats.compressed_size_mb / max(1, total_stats.original_size_mb)
        )

        logger.info("🎉 Comprehensive frame migration completed:")
        logger.info(f"   📊 Total frames: {total_stats.frames_processed}")
        logger.info(f"   🔄 Total duplicates: {total_stats.duplicate_frames_found}")
        logger.info(f"   💾 Size reduction: {total_stats.original_size_mb:.2f} -> {total_stats.compressed_size_mb:.2f} MB")
        logger.info(f"   📈 Compression ratio: {total_stats.compression_ratio:.2f}")

        results['_TOTAL'] = total_stats
        return results

# Factory function
def create_frame_compression_system(db_path: str = "tabula_rasa.db") -> FrameCompressionSystem:
    """Create and configure frame compression system"""
    return FrameCompressionSystem(db_path)

# Integration helper for storing frames in training loops
async def store_frame_compressed(frame_data: Any, db_path: str = "tabula_rasa.db") -> str:
    """Convenience function for storing compressed frames"""
    system = create_frame_compression_system(db_path)
    return await system.store_compressed_frame(frame_data)

# Integration helper for retrieving frames
async def retrieve_frame_decompressed(frame_hash: str, db_path: str = "tabula_rasa.db") -> Any:
    """Convenience function for retrieving decompressed frames"""
    system = create_frame_compression_system(db_path)
    return await system.retrieve_frame(frame_hash)