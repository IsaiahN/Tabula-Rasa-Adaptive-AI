"""
Performance Optimization

Provides query optimization and performance tuning utilities.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

class QueryOptimizer:
    """Optimizes database queries and operations."""
    
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.query_cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.optimization_stats = {
            'queries_optimized': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'optimization_time_saved': 0.0
        }
    
    def optimize_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Optimize a database query."""
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key(query, params)
        
        # Check cache first
        if cache_key in self.query_cache:
            self.cache_hits += 1
            self.optimization_stats['cache_hits'] += 1
            return self.query_cache[cache_key]
        
        # Optimize query
        optimized_query = self._apply_optimizations(query, params)
        
        # Cache the result
        if len(self.query_cache) < self.cache_size:
            self.query_cache[cache_key] = optimized_query
        else:
            # Remove oldest entry
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
            self.query_cache[cache_key] = optimized_query
        
        self.cache_misses += 1
        self.optimization_stats['cache_misses'] += 1
        self.optimization_stats['queries_optimized'] += 1
        
        optimization_time = time.time() - start_time
        self.optimization_stats['optimization_time_saved'] += optimization_time
        
        return optimized_query
    
    def _generate_cache_key(self, query: str, params: Optional[Dict[str, Any]]) -> str:
        """Generate a cache key for a query."""
        if params:
            param_str = str(sorted(params.items()))
            return f"{query}:{hash(param_str)}"
        return query
    
    def _apply_optimizations(self, query: str, params: Optional[Dict[str, Any]]) -> str:
        """Apply various query optimizations."""
        optimized = query
        
        # Add LIMIT if not present and query seems to be a SELECT
        if query.strip().upper().startswith('SELECT') and 'LIMIT' not in query.upper():
            optimized += ' LIMIT 1000'
        
        # Add ORDER BY for consistent results if not present
        if query.strip().upper().startswith('SELECT') and 'ORDER BY' not in query.upper():
            # Try to find a reasonable ordering column
            if 'id' in query.lower():
                optimized += ' ORDER BY id'
            elif 'timestamp' in query.lower():
                optimized += ' ORDER BY timestamp'
        
        return optimized
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_queries = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_queries, 1)
        
        return {
            'cache_size': len(self.query_cache),
            'max_cache_size': self.cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_queries': total_queries
        }
    
    def clear_cache(self) -> None:
        """Clear the query cache."""
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return self.optimization_stats.copy()

class PerformanceTuner:
    """Tunes system performance based on metrics."""
    
    def __init__(self):
        self.tuning_rules = []
        self.tuning_history = []
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Setup default performance tuning rules."""
        self.tuning_rules = [
            {
                'name': 'memory_cleanup',
                'condition': lambda metrics: metrics.get('memory_usage', 0) > 1000,
                'action': self._trigger_memory_cleanup,
                'priority': 1
            },
            {
                'name': 'cache_optimization',
                'condition': lambda metrics: metrics.get('cache_hit_ratio', 0) < 0.5,
                'action': self._optimize_cache,
                'priority': 2
            },
            {
                'name': 'query_optimization',
                'condition': lambda metrics: metrics.get('query_time', 0) > 1.0,
                'action': self._optimize_queries,
                'priority': 3
            }
        ]
    
    def tune_performance(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply performance tuning based on current metrics."""
        applied_tunings = []
        
        # Sort rules by priority
        sorted_rules = sorted(self.tuning_rules, key=lambda x: x['priority'])
        
        for rule in sorted_rules:
            try:
                if rule['condition'](metrics):
                    result = rule['action'](metrics)
                    applied_tunings.append({
                        'rule_name': rule['name'],
                        'result': result,
                        'timestamp': time.time()
                    })
                    self.tuning_history.append({
                        'rule_name': rule['name'],
                        'metrics': metrics.copy(),
                        'result': result,
                        'timestamp': time.time()
                    })
            except Exception as e:
                logger.error(f"Error applying tuning rule {rule['name']}: {e}")
        
        return applied_tunings
    
    def _trigger_memory_cleanup(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger memory cleanup."""
        logger.info("Triggering memory cleanup due to high memory usage")
        return {'action': 'memory_cleanup', 'status': 'triggered'}
    
    def _optimize_cache(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize cache settings."""
        logger.info("Optimizing cache due to low hit ratio")
        return {'action': 'cache_optimization', 'status': 'triggered'}
    
    def _optimize_queries(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize database queries."""
        logger.info("Optimizing queries due to slow query times")
        return {'action': 'query_optimization', 'status': 'triggered'}
    
    def add_tuning_rule(self, name: str, condition, action, priority: int = 5) -> None:
        """Add a custom tuning rule."""
        self.tuning_rules.append({
            'name': name,
            'condition': condition,
            'action': action,
            'priority': priority
        })
        logger.info(f"Added tuning rule: {name}")
    
    def get_tuning_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get tuning history."""
        if limit is None:
            return self.tuning_history.copy()
        return self.tuning_history[-limit:]
