"""
Disk Cache Implementation

File-based cache with compression and TTL support.
"""

import os
import time
import json
import pickle
import gzip
from pathlib import Path
from typing import Any, Dict, Optional


class DiskCache:
    """
    File-based cache with compression and TTL support.
    """
    
    def __init__(self, path: str = "./cache", ttl: int = 3600, compress: bool = True):
        self.path = Path(path)
        self.ttl = ttl
        self.compress = compress
        self.path.mkdir(parents=True, exist_ok=True)
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use first 2 characters as subdirectory to avoid too many files in one dir
        subdir = self.path / key[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key}.cache"
    
    def _is_expired(self, metadata: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        return time.time() > metadata.get('expires_at', 0)
    
    def _load_entry(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load cache entry from file."""
        if not file_path.exists():
            return None
        
        try:
            if self.compress:
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            return data
        except (EOFError, pickle.PickleError, OSError):
            # File corrupted or doesn't exist
            file_path.unlink(missing_ok=True)
            return None
    
    def _save_entry(self, file_path: Path, data: Dict[str, Any]) -> bool:
        """Save cache entry to file."""
        try:
            if self.compress:
                with gzip.open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            return True
        except (OSError, pickle.PickleError):
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        file_path = self._get_file_path(key)
        entry = self._load_entry(file_path)
        
        if entry is None:
            return None
        
        if self._is_expired(entry):
            file_path.unlink(missing_ok=True)
            return None
        
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        ttl = ttl or self.ttl
        expires_at = time.time() + ttl
        
        entry = {
            'value': value,
            'expires_at': expires_at,
            'created_at': time.time()
        }
        
        file_path = self._get_file_path(key)
        return self._save_entry(file_path, entry)
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        file_path = self._get_file_path(key)
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        file_path = self._get_file_path(key)
        entry = self._load_entry(file_path)
        
        if entry is None:
            return False
        
        if self._is_expired(entry):
            file_path.unlink(missing_ok=True)
            return False
        
        return True
    
    def clear(self) -> None:
        """Clear all cache entries."""
        if self.path.exists():
            import shutil
            shutil.rmtree(self.path)
            self.path.mkdir(parents=True, exist_ok=True)
    
    def size(self) -> int:
        """Get current cache size (number of files)."""
        count = 0
        if self.path.exists():
            for subdir in self.path.iterdir():
                if subdir.is_dir():
                    count += len(list(subdir.glob("*.cache")))
        return count
    
    def cleanup(self) -> int:
        """Clean up expired entries."""
        cleaned = 0
        if self.path.exists():
            for subdir in self.path.iterdir():
                if subdir.is_dir():
                    for cache_file in subdir.glob("*.cache"):
                        entry = self._load_entry(cache_file)
                        if entry is None or self._is_expired(entry):
                            cache_file.unlink(missing_ok=True)
                            cleaned += 1
        return cleaned
    
    def get_info(self) -> Dict[str, Any]:
        """Get cache information."""
        size = self.size()
        return {
            'size': size,
            'path': str(self.path),
            'ttl': self.ttl,
            'compress': self.compress
        }
