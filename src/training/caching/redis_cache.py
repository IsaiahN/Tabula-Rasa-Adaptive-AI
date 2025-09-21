"""
Redis Cache Implementation

Redis-based cache with TTL support.
"""

import json
import time
from typing import Any, Dict, Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisCache:
    """
    Redis-based cache with TTL support.
    """
    
    def __init__(self, url: str = "redis://localhost:6379/0", ttl: int = 3600):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis package not available. Install with: pip install redis")
        
        self.url = url
        self.ttl = ttl
        self._client = None
        self._connect()
    
    def _connect(self) -> None:
        """Connect to Redis."""
        try:
            self._client = redis.from_url(self.url, decode_responses=True)
            # Test connection
            self._client.ping()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")
    
    def _serialize(self, value: Any) -> str:
        """Serialize value for storage."""
        return json.dumps(value, default=str)
    
    def _deserialize(self, data: str) -> Any:
        """Deserialize value from storage."""
        return json.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            data = self._client.get(key)
            if data is None:
                return None
            return self._deserialize(data)
        except Exception:
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            ttl = ttl or self.ttl
            data = self._serialize(value)
            return self._client.setex(key, ttl, data)
        except Exception:
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            return bool(self._client.delete(key))
        except Exception:
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return bool(self._client.exists(key))
        except Exception:
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            self._client.flushdb()
        except Exception:
            pass
    
    def size(self) -> int:
        """Get current cache size."""
        try:
            return self._client.dbsize()
        except Exception:
            return 0
    
    def cleanup(self) -> int:
        """Redis handles TTL automatically, so no cleanup needed."""
        return 0
    
    def get_info(self) -> Dict[str, Any]:
        """Get cache information."""
        try:
            info = self._client.info()
            return {
                'size': self._client.dbsize(),
                'url': self.url,
                'ttl': self.ttl,
                'redis_version': info.get('redis_version', 'unknown'),
                'used_memory': info.get('used_memory_human', 'unknown')
            }
        except Exception:
            return {
                'size': 0,
                'url': self.url,
                'ttl': self.ttl,
                'error': 'Unable to get Redis info'
            }
