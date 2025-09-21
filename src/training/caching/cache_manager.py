"""
Cache Manager

Centralized cache management system with multiple backend support.
"""

import time
import hashlib
import json
from typing import Any, Dict, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from .memory_cache import MemoryCache
from .disk_cache import DiskCache
from .redis_cache import RedisCache


class CacheBackend(Enum):
    """Available cache backends."""
    MEMORY = "memory"
    DISK = "disk"
    REDIS = "redis"


@dataclass
class CacheConfig:
    """Configuration for cache manager."""
    backend: CacheBackend = CacheBackend.MEMORY
    max_size: int = 1000
    ttl_seconds: int = 3600
    cleanup_interval: int = 300
    compression: bool = True
    redis_url: Optional[str] = None
    disk_path: str = "./cache"


class CacheManager:
    """
    Centralized cache manager supporting multiple backends.
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache = self._create_backend()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0
        }
    
    def _create_backend(self):
        """Create the appropriate cache backend."""
        if self.config.backend == CacheBackend.MEMORY:
            return MemoryCache(max_size=self.config.max_size, ttl=self.config.ttl_seconds)
        elif self.config.backend == CacheBackend.DISK:
            return DiskCache(path=self.config.disk_path, ttl=self.config.ttl_seconds)
        elif self.config.backend == CacheBackend.REDIS:
            return RedisCache(url=self.config.redis_url, ttl=self.config.ttl_seconds)
        else:
            raise ValueError(f"Unsupported cache backend: {self.config.backend}")
    
    def _generate_key(self, key: Union[str, Dict[str, Any]]) -> str:
        """Generate a cache key from string or dictionary."""
        if isinstance(key, dict):
            # Sort keys for consistent hashing
            sorted_key = json.dumps(key, sort_keys=True)
            return hashlib.md5(sorted_key.encode()).hexdigest()
        return str(key)
    
    def get(self, key: Union[str, Dict[str, Any]]) -> Optional[Any]:
        """Get value from cache."""
        cache_key = self._generate_key(key)
        value = self._cache.get(cache_key)
        
        if value is not None:
            self._stats['hits'] += 1
        else:
            self._stats['misses'] += 1
        
        return value
    
    def set(self, key: Union[str, Dict[str, Any]], value: Any, 
            ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        cache_key = self._generate_key(key)
        ttl = ttl or self.config.ttl_seconds
        
        success = self._cache.set(cache_key, value, ttl)
        if success:
            self._stats['sets'] += 1
        
        return success
    
    def delete(self, key: Union[str, Dict[str, Any]]) -> bool:
        """Delete value from cache."""
        cache_key = self._generate_key(key)
        success = self._cache.delete(cache_key)
        if success:
            self._stats['deletes'] += 1
        return success
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._stats = {k: 0 for k in self._stats}
    
    def exists(self, key: Union[str, Dict[str, Any]]) -> bool:
        """Check if key exists in cache."""
        cache_key = self._generate_key(key)
        return self._cache.exists(cache_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self._stats['hits'] / max(1, self._stats['hits'] + self._stats['misses'])
        return {
            **self._stats,
            'hit_rate': hit_rate,
            'backend': self.config.backend.value,
            'size': self._cache.size()
        }
    
    def cleanup(self) -> int:
        """Clean up expired entries."""
        return self._cache.cleanup()


def cached(ttl: int = 3600, key_func: Optional[Callable] = None):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        key_func: Function to generate cache key from arguments
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cache_manager = getattr(wrapper, '_cache_manager', None)
            if cache_manager:
                cached_result = cache_manager.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            if cache_manager:
                cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def cache_result(cache_manager: CacheManager, ttl: int = 3600):
    """
    Decorator that uses a specific cache manager.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            return result
        
        # Store cache manager reference
        wrapper._cache_manager = cache_manager
        return wrapper
    return decorator


def invalidate_cache(cache_manager: CacheManager, pattern: Optional[str] = None):
    """
    Decorator to invalidate cache entries after function execution.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if pattern:
                # Invalidate entries matching pattern
                cache_manager._cache.invalidate_pattern(pattern)
            else:
                # Invalidate all entries
                cache_manager.clear()
            
            return result
        return wrapper
    return decorator
