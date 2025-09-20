"""
Performance Monitor

Monitors system performance, memory usage, and optimization metrics.
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitors system performance and optimization metrics."""
    
    def __init__(self):
        self.startup_time = time.time()
        self.performance_metrics = {
            'startup_time': self.startup_time,
            'memory_usage': 0,
            'query_cache_hits': 0,
            'query_cache_misses': 0,
            'optimization_savings': 0,
            'lazy_imports_used': 0,
            'total_actions_taken': 0,
            'total_games_played': 0
        }
        self.memory_check_interval = 100  # Check memory every 100 actions
        self.last_memory_check = 0
        self.query_optimizer = None
        self._initialize_query_optimizer()
    
    def _initialize_query_optimizer(self) -> None:
        """Initialize query optimizer for database performance."""
        try:
            from src.database.query_optimizer import get_query_optimizer
            self.query_optimizer = get_query_optimizer()
            logger.debug("Query optimizer initialized")
        except ImportError:
            self.query_optimizer = None
            logger.warning("Query optimizer not available")
    
    def update_metric(self, metric: str, value: float = 1.0) -> None:
        """Update a performance metric."""
        if metric in self.performance_metrics:
            self.performance_metrics[metric] += value
        else:
            self.performance_metrics[metric] = value
    
    def set_metric(self, metric: str, value: float) -> None:
        """Set a performance metric to a specific value."""
        self.performance_metrics[metric] = value
    
    def get_metric(self, metric: str, default: float = 0.0) -> float:
        """Get a performance metric value."""
        return self.performance_metrics.get(metric, default)
    
    def check_memory_usage(self) -> float:
        """Check and log memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.performance_metrics['memory_usage'] = memory_mb
            
            if memory_mb > 1000:  # Alert if using more than 1GB
                logger.warning(f"High memory usage: {memory_mb:.1f}MB")
            
            return memory_mb
        except ImportError:
            # psutil not available, skip memory monitoring
            logger.debug("psutil not available for memory monitoring")
            return 0.0
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
            return 0.0
    
    def should_check_memory(self, action_count: int) -> bool:
        """Check if memory should be checked based on action count."""
        if action_count - self.last_memory_check >= self.memory_check_interval:
            self.last_memory_check = action_count
            return True
        return False
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        runtime = time.time() - self.startup_time
        memory_usage = self.get_metric('memory_usage', 0)
        
        # Calculate cache hit ratio
        cache_hits = self.get_metric('query_cache_hits', 0)
        cache_misses = self.get_metric('query_cache_misses', 1)
        cache_hit_ratio = cache_hits / max(cache_misses, 1)
        
        # Calculate rates
        actions_per_second = self.get_metric('total_actions_taken', 0) / max(runtime, 1)
        games_per_hour = (self.get_metric('total_games_played', 0) / max(runtime, 1)) * 3600
        
        return {
            'runtime_seconds': runtime,
            'memory_usage_mb': memory_usage,
            'query_cache_hits': cache_hits,
            'query_cache_misses': cache_misses,
            'lazy_imports_used': self.get_metric('lazy_imports_used', 0),
            'optimization_savings': self.get_metric('optimization_savings', 0),
            'cache_hit_ratio': cache_hit_ratio,
            'actions_per_second': actions_per_second,
            'games_per_hour': games_per_hour,
            'total_actions_taken': self.get_metric('total_actions_taken', 0),
            'total_games_played': self.get_metric('total_games_played', 0)
        }
    
    def log_performance_summary(self) -> None:
        """Log a performance summary."""
        report = self.get_performance_report()
        logger.info(f"Performance Summary - Runtime: {report['runtime_seconds']:.1f}s, "
                   f"Memory: {report['memory_usage_mb']:.1f}MB, "
                   f"Actions/sec: {report['actions_per_second']:.2f}, "
                   f"Games/hour: {report['games_per_hour']:.1f}")
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.startup_time = time.time()
        self.performance_metrics = {
            'startup_time': self.startup_time,
            'memory_usage': 0,
            'query_cache_hits': 0,
            'query_cache_misses': 0,
            'optimization_savings': 0,
            'lazy_imports_used': 0,
            'total_actions_taken': 0,
            'total_games_played': 0
        }
        self.last_memory_check = 0
        logger.info("Performance metrics reset")
    
    def is_performance_healthy(self) -> bool:
        """Check if system performance is healthy."""
        try:
            # Check memory usage
            memory_usage = self.get_metric('memory_usage', 0)
            if memory_usage > 2000:  # More than 2GB
                logger.warning(f"High memory usage: {memory_usage:.1f}MB")
                return False
            
            # Check cache hit ratio
            cache_hit_ratio = self.get_performance_report()['cache_hit_ratio']
            if cache_hit_ratio < 0.5:  # Less than 50% cache hit rate
                logger.warning(f"Low cache hit ratio: {cache_hit_ratio:.2f}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking performance health: {e}")
            return False
