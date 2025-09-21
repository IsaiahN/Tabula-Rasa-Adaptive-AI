"""
Cache Decorators

Convenient decorators for caching function results.
"""

import functools
import inspect
from typing import Any, Callable, Optional, Union, Dict
from .cache_manager import CacheManager, CacheConfig, CacheBackend


def cached(ttl: int = 3600, 
          key_func: Optional[Callable] = None,
          cache_manager: Optional[CacheManager] = None,
          exclude_args: Optional[list] = None):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        key_func: Function to generate cache key from arguments
        cache_manager: Specific cache manager to use
        exclude_args: List of argument names to exclude from cache key
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache manager
            manager = cache_manager or getattr(wrapper, '_cache_manager', None)
            if not manager:
                # Create default cache manager
                config = CacheConfig(backend=CacheBackend.MEMORY, ttl_seconds=ttl)
                manager = CacheManager(config)
                wrapper._cache_manager = manager
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _generate_cache_key(func, args, kwargs, exclude_args)
            
            # Try to get from cache
            cached_result = manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            manager.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator


def cache_result(cache_manager: CacheManager, ttl: int = 3600):
    """
    Decorator that uses a specific cache manager.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = _generate_cache_key(func, args, kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator


def invalidate_cache(cache_manager: Optional[CacheManager] = None, 
                   pattern: Optional[str] = None):
    """
    Decorator to invalidate cache entries after function execution.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            manager = cache_manager or getattr(wrapper, '_cache_manager', None)
            if manager:
                if pattern:
                    # Invalidate entries matching pattern
                    manager._cache.invalidate_pattern(pattern)
                else:
                    # Invalidate all entries
                    manager.clear()
            
            return result
        
        return wrapper
    return decorator


def cache_key_from_args(*arg_names: str):
    """
    Decorator factory for creating cache key from specific arguments.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Create key from specified arguments
            key_parts = []
            for arg_name in arg_names:
                if arg_name in bound_args.arguments:
                    value = bound_args.arguments[arg_name]
                    key_parts.append(f"{arg_name}={value}")
            
            cache_key = f"{func.__name__}:{':'.join(key_parts)}"
            
            # Get cache manager
            manager = getattr(wrapper, '_cache_manager', None)
            if not manager:
                config = CacheConfig(backend=CacheBackend.MEMORY)
                manager = CacheManager(config)
                wrapper._cache_manager = manager
            
            # Try to get from cache
            cached_result = manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            manager.set(cache_key, result)
            return result
        
        return wrapper
    return decorator


def _generate_cache_key(func: Callable, args: tuple, kwargs: dict, 
                       exclude_args: Optional[list] = None) -> str:
    """Generate cache key from function arguments."""
    exclude_args = exclude_args or []
    
    # Get function signature
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    
    # Create key parts
    key_parts = [func.__name__]
    
    for param_name, param_value in bound_args.arguments.items():
        if param_name not in exclude_args:
            # Convert to string representation
            if isinstance(param_value, (dict, list, tuple)):
                sorted_value = (sorted(param_value) 
                               if isinstance(param_value, dict) 
                               else param_value)
                hash_value = hash(str(sorted_value))
                key_parts.append(f"{param_name}={hash_value}")
            else:
                key_parts.append(f"{param_name}={param_value}")
    
    return ":".join(key_parts)


class Cacheable:
    """
    Mixin class for objects that support caching.
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self._cache_manager = cache_manager or CacheManager(CacheConfig())
    
    def cache_method(self, method_name: str, ttl: int = 3600):
        """Cache a method result."""
        method = getattr(self, method_name)
        return cached(ttl=ttl, cache_manager=self._cache_manager)(method)
    
    def invalidate_method_cache(self, pattern: Optional[str] = None):
        """Invalidate cache for a method."""
        if pattern:
            self._cache_manager._cache.invalidate_pattern(pattern)
        else:
            self._cache_manager.clear()
