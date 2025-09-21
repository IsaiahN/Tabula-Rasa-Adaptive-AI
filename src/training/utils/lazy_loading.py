"""
Lazy Loading Utilities

Provides lazy loading capabilities for heavy components to improve startup performance.
"""

import weakref
from typing import Any, Callable, Optional, Dict
from functools import lru_cache


class LazyLoader:
    """
    Generic lazy loader for heavy components.
    """
    
    def __init__(self, factory: Callable, *args, **kwargs):
        self.factory = factory
        self.args = args
        self.kwargs = kwargs
        self._instance = None
        self._weak_ref = None
    
    def __call__(self) -> Any:
        if self._instance is None or self._weak_ref is None:
            self._instance = self.factory(*self.args, **self.kwargs)
            self._weak_ref = weakref.ref(self._instance)
        return self._instance
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self(), name)
    
    def clear(self):
        """Clear the cached instance."""
        self._instance = None
        self._weak_ref = None


class LazyComponentManager:
    """
    Manages multiple lazy-loaded components.
    """
    
    def __init__(self):
        self._components: Dict[str, LazyLoader] = {}
    
    def register(self, name: str, factory: Callable, *args, **kwargs):
        """Register a component for lazy loading."""
        self._components[name] = LazyLoader(factory, *args, **kwargs)
    
    def get(self, name: str) -> Any:
        """Get a lazy-loaded component."""
        if name not in self._components:
            raise KeyError(f"Component '{name}' not registered")
        return self._components[name]()
    
    def clear(self, name: Optional[str] = None):
        """Clear component cache."""
        if name is None:
            for component in self._components.values():
                component.clear()
        elif name in self._components:
            self._components[name].clear()
    
    def __getattr__(self, name: str) -> Any:
        return self.get(name)


class CachedFunction:
    """
    Cached function wrapper with memory management.
    """
    
    def __init__(self, func: Callable, max_size: int = 128):
        self.func = func
        self.max_size = max_size
        self._cache = {}
        self._access_order = []
    
    def __call__(self, *args, **kwargs):
        # Create cache key
        key = (args, tuple(sorted(kwargs.items())))
        
        if key in self._cache:
            # Move to end of access order
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        
        # Compute result
        result = self.func(*args, **kwargs)
        
        # Add to cache
        self._cache[key] = result
        self._access_order.append(key)
        
        # Manage cache size
        if len(self._cache) > self.max_size:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
        
        return result
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()


def lazy_property(factory: Callable) -> property:
    """
    Create a lazy property that loads a component only when accessed.
    """
    def getter(self):
        if not hasattr(self, '_lazy_cache'):
            self._lazy_cache = {}
        
        prop_name = factory.__name__
        if prop_name not in self._lazy_cache:
            self._lazy_cache[prop_name] = factory()
        
        return self._lazy_cache[prop_name]
    
    return property(getter)


# Example usage for common components
def create_lazy_vision_system():
    """Create a lazy-loaded vision system."""
    manager = LazyComponentManager()
    
    # Register vision components
    manager.register('analyzer', lambda: __import__('src.vision', fromlist=['FrameAnalyzer']).FrameAnalyzer())
    manager.register('extractor', lambda: __import__('src.vision', fromlist=['FeatureExtractor']).FeatureExtractor())
    manager.register('detector', lambda: __import__('src.vision', fromlist=['ObjectDetector']).ObjectDetector())
    
    return manager


def create_lazy_training_system():
    """Create a lazy-loaded training system."""
    manager = LazyComponentManager()
    
    # Register training components
    manager.register('loop', lambda: __import__('src.training', fromlist=['ContinuousLearningLoop']).ContinuousLearningLoop())
    manager.register('trainer', lambda: __import__('src.training', fromlist=['MasterARCTrainer', 'MasterTrainingConfig']).MasterARCTrainer(
        __import__('src.training', fromlist=['MasterTrainingConfig']).MasterTrainingConfig()
    ))
    
    return manager


def create_lazy_learning_system():
    """Create a lazy-loaded learning system."""
    manager = LazyComponentManager()
    
    # Register learning components
    manager.register('meta_learning', lambda: __import__('src.learning', fromlist=['ARCMetaLearningSystem']).ARCMetaLearningSystem())
    manager.register('pattern_recognizer', lambda: __import__('src.learning', fromlist=['ARCPatternRecognizer']).ARCPatternRecognizer())
    manager.register('knowledge_transfer', lambda: __import__('src.learning', fromlist=['KnowledgeTransfer']).KnowledgeTransfer())
    
    return manager


# Global lazy managers
vision_system = create_lazy_vision_system()
training_system = create_lazy_training_system()
learning_system = create_lazy_learning_system()


# Example usage:
# from src.training.utils.lazy_loading import vision_system, training_system
# analyzer = vision_system.get('analyzer')  # Loads only when needed
# loop = training_system.get('loop')  # Loads only when needed
