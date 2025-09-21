"""
Memory Cache Implementation

In-memory cache with TTL support and LRU eviction.
"""

import time
import threading
from typing import Any, Dict, Optional
from collections import OrderedDict


class MemoryCache:
    """
    Thread-safe in-memory cache with TTL and LRU eviction.
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        return time.time() > entry['expires_at']
    
    def _cleanup_expired(self) -> int:
        """Remove expired entries and return count."""
        expired_keys = []
        current_time = time.time()
        
        with self._lock:
            for key, entry in self._cache.items():
                if current_time > entry['expires_at']:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
        
        return len(expired_keys)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            if self._is_expired(entry):
                del self._cache[key]
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        ttl = ttl or self.ttl
        expires_at = time.time() + ttl
        
        with self._lock:
            # Remove if already exists
            if key in self._cache:
                del self._cache[key]
            
            # Add new entry
            self._cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': time.time()
            }
            
            # Evict if over capacity
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)  # Remove oldest
        
        return True
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            if key not in self._cache:
                return False
            
            if self._is_expired(self._cache[key]):
                del self._cache[key]
                return False
            
            return True
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            # Clean up expired entries first
            self._cleanup_expired()
            return len(self._cache)
    
    def cleanup(self) -> int:
        """Clean up expired entries."""
        return self._cleanup_expired()
    
    def get_info(self) -> Dict[str, Any]:
        """Get cache information."""
        with self._lock:
            self._cleanup_expired()
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'ttl': self.ttl,
                'keys': list(self._cache.keys())
            }
