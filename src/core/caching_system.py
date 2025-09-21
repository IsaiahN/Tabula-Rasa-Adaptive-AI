#!/usr/bin/env python3
"""
Comprehensive Caching System

This module provides a unified caching system for all components with:
- Multi-level caching (memory, disk, database)
- Cache invalidation strategies
- Performance monitoring and metrics
- Memory management and cleanup
- Cache persistence and recovery
"""

import time
import json
import hashlib
import logging
import threading
from typing import Dict, Any, Optional, Callable, Union, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import OrderedDict, defaultdict
from pathlib import Path
import pickle
import numpy as np
from datetime import datetime, timedelta
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache storage levels."""
    MEMORY = "memory"
    DISK = "disk"
    DATABASE = "database"


class CacheStrategy(Enum):
    """Cache invalidation strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    SIZE = "size"  # Size-based eviction


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    # Memory cache settings
    max_memory_size: int = 1000  # Maximum number of items in memory
    max_memory_mb: float = 512.0  # Maximum memory usage in MB
    
    # Disk cache settings
    disk_cache_dir: str = "cache"
    max_disk_size_mb: float = 2048.0  # Maximum disk cache size in MB
    
    # TTL settings
    default_ttl_seconds: int = 3600  # 1 hour default TTL
    max_ttl_seconds: int = 86400  # 24 hours maximum TTL
    
    # Performance monitoring
    enable_metrics: bool = True
    metrics_retention_days: int = 7
    
    # Cache strategies
    memory_strategy: CacheStrategy = CacheStrategy.LRU
    disk_strategy: CacheStrategy = CacheStrategy.SIZE


@dataclass
class CacheEntry:
    """A single cache entry."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    level: CacheLevel = CacheLevel.MEMORY
    
    def is_expired(self) -> bool:
        """Check if the cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    def get_age_seconds(self) -> float:
        """Get the age of the cache entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    def get_idle_seconds(self) -> float:
        """Get the idle time of the cache entry in seconds."""
        return (datetime.now() - self.last_accessed).total_seconds()


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    total_size_bytes: int = 0
    avg_response_time_ms: float = 0.0
    hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0
    
    def update_hit(self, response_time_ms: float):
        """Update metrics for a cache hit."""
        self.hits += 1
        self.total_requests += 1
        self._update_avg_response_time(response_time_ms)
        self.hit_rate = self.hits / self.total_requests if self.total_requests > 0 else 0.0
    
    def update_miss(self, response_time_ms: float):
        """Update metrics for a cache miss."""
        self.misses += 1
        self.total_requests += 1
        self._update_avg_response_time(response_time_ms)
        self.hit_rate = self.hits / self.total_requests if self.total_requests > 0 else 0.0
    
    def update_eviction(self, size_bytes: int):
        """Update metrics for a cache eviction."""
        self.evictions += 1
        self.total_size_bytes -= size_bytes
    
    def _update_avg_response_time(self, response_time_ms: float):
        """Update average response time."""
        if self.total_requests == 1:
            self.avg_response_time_ms = response_time_ms
        else:
            self.avg_response_time_ms = (
                (self.avg_response_time_ms * (self.total_requests - 1) + response_time_ms) 
                / self.total_requests
            )


class MemoryCache:
    """In-memory cache implementation."""
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_size = max_size
        self.strategy = strategy
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.total_size_bytes = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check if expired
            if entry.is_expired():
                del self.cache[key]
                self.total_size_bytes -= entry.size_bytes
                return None
            
            # Update access info
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            
            # Move to end for LRU
            if self.strategy == CacheStrategy.LRU:
                self.cache.move_to_end(key)
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set a value in the cache."""
        with self.lock:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds,
                level=CacheLevel.MEMORY
            )
            
            # Remove old entry if exists
            if key in self.cache:
                old_entry = self.cache[key]
                self.total_size_bytes -= old_entry.size_bytes
            
            # Add new entry
            self.cache[key] = entry
            self.total_size_bytes += size_bytes
            
            # Evict if necessary
            self._evict_if_necessary()
    
    def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        with self.lock:
            if key not in self.cache:
                return False
            
            entry = self.cache[key]
            del self.cache[key]
            self.total_size_bytes -= entry.size_bytes
            return True
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.total_size_bytes = 0
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate the size of a value in bytes."""
        try:
            if isinstance(value, (str, int, float, bool)):
                return len(str(value).encode('utf-8'))
            elif isinstance(value, (list, dict, tuple)):
                return len(pickle.dumps(value))
            elif isinstance(value, np.ndarray):
                return value.nbytes
            else:
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default size estimate
    
    def _evict_if_necessary(self) -> None:
        """Evict entries if cache is full."""
        while len(self.cache) > self.max_size:
            if self.strategy == CacheStrategy.LRU:
                # Remove least recently used
                key, _ = self.cache.popitem(last=False)
            elif self.strategy == CacheStrategy.LFU:
                # Remove least frequently used
                key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
                del self.cache[key]
            else:
                # Default to LRU
                key, _ = self.cache.popitem(last=False)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'total_size_bytes': self.total_size_bytes,
                'strategy': self.strategy.value
            }


class DiskCache:
    """Disk-based cache implementation."""
    
    def __init__(self, cache_dir: str = "cache", max_size_mb: float = 2048.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.lock = threading.RLock()
        self.total_size_bytes = 0
        self._load_metadata()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the disk cache."""
        with self.lock:
            cache_file = self.cache_dir / f"{key}.cache"
            if not cache_file.exists():
                return None
            
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                
                # Check if expired
                if entry.is_expired():
                    cache_file.unlink()
                    self.total_size_bytes -= entry.size_bytes
                    return None
                
                # Update access info
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
                # Save updated entry
                with open(cache_file, 'wb') as f:
                    pickle.dump(entry, f)
                
                return entry.value
            except Exception as e:
                logger.error(f"Failed to read cache file {cache_file}: {e}")
                return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set a value in the disk cache."""
        with self.lock:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds,
                level=CacheLevel.DISK
            )
            
            # Remove old entry if exists
            cache_file = self.cache_dir / f"{key}.cache"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        old_entry = pickle.load(f)
                    self.total_size_bytes -= old_entry.size_bytes
                except Exception:
                    pass
            
            # Save new entry
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(entry, f)
                self.total_size_bytes += size_bytes
                
                # Evict if necessary
                self._evict_if_necessary()
            except Exception as e:
                logger.error(f"Failed to write cache file {cache_file}: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete a value from the disk cache."""
        with self.lock:
            cache_file = self.cache_dir / f"{key}.cache"
            if not cache_file.exists():
                return False
            
            try:
                # Get size before deletion
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                self.total_size_bytes -= entry.size_bytes
                
                cache_file.unlink()
                return True
            except Exception as e:
                logger.error(f"Failed to delete cache file {cache_file}: {e}")
                return False
    
    def clear(self) -> None:
        """Clear all disk cache entries."""
        with self.lock:
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete cache file {cache_file}: {e}")
            self.total_size_bytes = 0
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate the size of a value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default size estimate
    
    def _evict_if_necessary(self) -> None:
        """Evict entries if cache is too large."""
        if self.total_size_bytes <= self.max_size_bytes:
            return
        
        # Get all cache files with their access times
        cache_files = []
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                cache_files.append((cache_file, entry.last_accessed, entry.size_bytes))
            except Exception:
                continue
        
        # Sort by last accessed time (oldest first)
        cache_files.sort(key=lambda x: x[1])
        
        # Remove oldest files until under limit
        for cache_file, _, size_bytes in cache_files:
            if self.total_size_bytes <= self.max_size_bytes:
                break
            
            try:
                cache_file.unlink()
                self.total_size_bytes -= size_bytes
            except Exception as e:
                logger.error(f"Failed to delete cache file {cache_file}: {e}")
    
    def _load_metadata(self) -> None:
        """Load cache metadata from disk."""
        self.total_size_bytes = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                self.total_size_bytes += entry.size_bytes
            except Exception:
                continue
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'cache_dir': str(self.cache_dir),
                'total_size_bytes': self.total_size_bytes,
                'max_size_bytes': self.max_size_bytes,
                'usage_percent': (self.total_size_bytes / self.max_size_bytes) * 100
            }


class UnifiedCachingSystem:
    """Unified caching system with multiple levels."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.memory_cache = MemoryCache(
            max_size=self.config.max_memory_size,
            strategy=self.config.memory_strategy
        )
        self.disk_cache = DiskCache(
            cache_dir=self.config.disk_cache_dir,
            max_size_mb=self.config.max_disk_size_mb
        )
        self.metrics = CacheMetrics()
        self.lock = threading.RLock()
        
        logger.info("Unified Caching System initialized")
    
    def get(self, key: str, level: Optional[CacheLevel] = None) -> Optional[Any]:
        """Get a value from the cache."""
        start_time = time.time()
        
        with self.lock:
            # Try memory cache first
            if level is None or level == CacheLevel.MEMORY:
                value = self.memory_cache.get(key)
                if value is not None:
                    response_time = (time.time() - start_time) * 1000
                    self.metrics.update_hit(response_time)
                    return value
            
            # Try disk cache
            if level is None or level == CacheLevel.DISK:
                value = self.disk_cache.get(key)
                if value is not None:
                    # Promote to memory cache
                    self.memory_cache.set(key, value, self.config.default_ttl_seconds)
                    response_time = (time.time() - start_time) * 1000
                    self.metrics.update_hit(response_time)
                    return value
            
            # Cache miss
            response_time = (time.time() - start_time) * 1000
            self.metrics.update_miss(response_time)
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None, 
            level: CacheLevel = CacheLevel.MEMORY) -> None:
        """Set a value in the cache."""
        if ttl_seconds is None:
            ttl_seconds = self.config.default_ttl_seconds
        
        with self.lock:
            if level == CacheLevel.MEMORY:
                self.memory_cache.set(key, value, ttl_seconds)
            elif level == CacheLevel.DISK:
                self.disk_cache.set(key, value, ttl_seconds)
            elif level == CacheLevel.DATABASE:
                # Database caching would be implemented here
                logger.warning("Database caching not implemented yet")
    
    def delete(self, key: str, level: Optional[CacheLevel] = None) -> bool:
        """Delete a value from the cache."""
        with self.lock:
            deleted = False
            if level is None or level == CacheLevel.MEMORY:
                deleted |= self.memory_cache.delete(key)
            if level is None or level == CacheLevel.DISK:
                deleted |= self.disk_cache.delete(key)
            return deleted
    
    def clear(self, level: Optional[CacheLevel] = None) -> None:
        """Clear the cache."""
        with self.lock:
            if level is None or level == CacheLevel.MEMORY:
                self.memory_cache.clear()
            if level is None or level == CacheLevel.DISK:
                self.disk_cache.clear()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache metrics."""
        with self.lock:
            memory_stats = self.memory_cache.get_stats()
            disk_stats = self.disk_cache.get_stats()
            
            return {
                'metrics': asdict(self.metrics),
                'memory_cache': memory_stats,
                'disk_cache': disk_stats,
                'total_size_mb': (memory_stats['total_size_bytes'] + disk_stats['total_size_bytes']) / (1024 * 1024)
            }
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries from all caches."""
        with self.lock:
            cleaned = 0
            
            # Clean memory cache
            expired_keys = []
            for key, entry in self.memory_cache.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                if self.memory_cache.delete(key):
                    cleaned += 1
            
            # Clean disk cache
            for cache_file in self.disk_cache.cache_dir.glob("*.cache"):
                try:
                    with open(cache_file, 'rb') as f:
                        entry = pickle.load(f)
                    if entry.is_expired():
                        cache_file.unlink()
                        self.disk_cache.total_size_bytes -= entry.size_bytes
                        cleaned += 1
                except Exception:
                    continue
            
            return cleaned


# Global cache instance
_global_cache: Optional[UnifiedCachingSystem] = None


def get_global_cache() -> UnifiedCachingSystem:
    """Get the global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = UnifiedCachingSystem()
    return _global_cache


def cache_result(ttl_seconds: int = 3600, level: CacheLevel = CacheLevel.MEMORY):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = {
                'func': func.__name__,
                'args': args,
                'kwargs': kwargs
            }
            key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
            
            # Try to get from cache
            cache = get_global_cache()
            result = cache.get(key, level)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result, ttl_seconds, level)
            return result
        
        return wrapper
    return decorator


def cache_async_result(ttl_seconds: int = 3600, level: CacheLevel = CacheLevel.MEMORY):
    """Decorator to cache async function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = {
                'func': func.__name__,
                'args': args,
                'kwargs': kwargs
            }
            key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
            
            # Try to get from cache
            cache = get_global_cache()
            result = cache.get(key, level)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache.set(key, result, ttl_seconds, level)
            return result
        
        return wrapper
    return decorator


# Factory function
def create_caching_system(config: Optional[CacheConfig] = None) -> UnifiedCachingSystem:
    """Create a new caching system instance."""
    return UnifiedCachingSystem(config)
