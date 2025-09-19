#!/usr/bin/env python3
"""
Database Query Optimizer for Tabula Rasa

This module provides optimized database queries and caching mechanisms
to improve system performance and reduce database load.
"""

import sqlite3
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class QueryOptimizer:
    """
    Database query optimizer with caching and performance improvements.
    """
    
    def __init__(self, db_path: str = "tabula_rasa.db"):
        self.db_path = db_path
        self.query_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.max_cache_size = 1000
        
    def _get_connection(self):
        """Get database connection with optimized settings."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        # Optimize SQLite settings for performance
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB
        
        return conn
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached query result is still valid."""
        if cache_key not in self.query_cache:
            return False
        
        cache_entry = self.query_cache[cache_key]
        age = datetime.now() - cache_entry['timestamp']
        return age.total_seconds() < self.cache_ttl
    
    def _cache_result(self, cache_key: str, result: Any):
        """Cache query result with TTL."""
        if len(self.query_cache) >= self.max_cache_size:
            # Remove oldest entries
            oldest_key = min(self.query_cache.keys(), 
                           key=lambda k: self.query_cache[k]['timestamp'])
            del self.query_cache[oldest_key]
        
        self.query_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now()
        }
    
    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if valid."""
        if self._is_cache_valid(cache_key):
            return self.query_cache[cache_key]['result']
        return None
    
    # ============================================================================
    # OPTIMIZED QUERY METHODS
    # ============================================================================
    
    def get_recent_errors_optimized(self, hours: int = 2, limit: int = 20) -> List[Dict[str, Any]]:
        """Optimized query for recent errors with caching."""
        cache_key = f"recent_errors_{hours}_{limit}"
        cached = self.get_cached_result(cache_key)
        if cached is not None:
            return cached
        
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT log_level, component, message, timestamp, session_id, game_id
                FROM system_logs 
                WHERE log_level IN ('ERROR', 'WARNING') 
                AND timestamp > datetime('now', '-{} hours')
                ORDER BY timestamp DESC 
                LIMIT ?
            """.format(hours), (limit,))
            
            result = [dict(row) for row in cursor.fetchall()]
            self._cache_result(cache_key, result)
            return result
    
    def get_system_health_optimized(self, hours: int = 2) -> Dict[str, Any]:
        """Optimized system health query with caching."""
        cache_key = f"system_health_{hours}"
        cached = self.get_cached_result(cache_key)
        if cached is not None:
            return cached
        
        with self._get_connection() as conn:
            # Get error counts by type
            cursor = conn.execute("""
                SELECT log_level, COUNT(*) as count
                FROM system_logs 
                WHERE timestamp > datetime('now', '-{} hours')
                GROUP BY log_level
                ORDER BY count DESC
            """.format(hours))
            error_counts = {row['log_level']: row['count'] for row in cursor.fetchall()}
            
            # Get recent game performance
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_games,
                    SUM(win_detected) as total_wins,
                    AVG(final_score) as avg_score,
                    AVG(total_actions) as avg_actions
                FROM game_results
                WHERE start_time > datetime('now', '-{} hours')
            """.format(hours))
            game_stats = dict(cursor.fetchone())
            
            # Get active sessions
            cursor = conn.execute("""
                SELECT COUNT(*) as active_sessions
                FROM training_sessions
                WHERE status = 'active'
            """)
            active_sessions = cursor.fetchone()['active_sessions']
            
            result = {
                'error_counts': error_counts,
                'game_stats': game_stats,
                'active_sessions': active_sessions,
                'timestamp': datetime.now().isoformat()
            }
            
            self._cache_result(cache_key, result)
            return result
    
    def get_performance_trends_optimized(self, days: int = 7) -> Dict[str, Any]:
        """Optimized performance trends query with caching."""
        cache_key = f"performance_trends_{days}"
        cached = self.get_cached_result(cache_key)
        if cached is not None:
            return cached
        
        with self._get_connection() as conn:
            # Get daily performance metrics
            cursor = conn.execute("""
                SELECT 
                    DATE(start_time) as date,
                    COUNT(*) as games_played,
                    SUM(win_detected) as wins,
                    AVG(final_score) as avg_score,
                    AVG(total_actions) as avg_actions
                FROM game_results
                WHERE start_time >= datetime('now', '-{} days')
                GROUP BY DATE(start_time)
                ORDER BY date DESC
            """.format(days))
            
            daily_trends = [dict(row) for row in cursor.fetchall()]
            
            # Get action effectiveness trends
            cursor = conn.execute("""
                SELECT 
                    action_type,
                    AVG(success_rate) as avg_success_rate,
                    COUNT(*) as usage_count
                FROM action_effectiveness
                WHERE created_at >= datetime('now', '-{} days')
                GROUP BY action_type
                ORDER BY avg_success_rate DESC
            """.format(days))
            
            action_trends = [dict(row) for row in cursor.fetchall()]
            
            result = {
                'daily_trends': daily_trends,
                'action_trends': action_trends,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            self._cache_result(cache_key, result)
            return result
    
    def get_coordinate_intelligence_optimized(self, game_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Optimized coordinate intelligence query with caching."""
        cache_key = f"coordinate_intelligence_{game_id}_{limit}"
        cached = self.get_cached_result(cache_key)
        if cached is not None:
            return cached
        
        with self._get_connection() as conn:
            if game_id:
                cursor = conn.execute("""
                    SELECT * FROM coordinate_intelligence
                    WHERE game_id = ?
                    ORDER BY success_rate DESC, usage_count DESC
                    LIMIT ?
                """, (game_id, limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM coordinate_intelligence
                    ORDER BY success_rate DESC, usage_count DESC
                    LIMIT ?
                """, (limit,))
            
            result = [dict(row) for row in cursor.fetchall()]
            self._cache_result(cache_key, result)
            return result
    
    def get_learned_patterns_optimized(self, pattern_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Optimized learned patterns query with caching."""
        cache_key = f"learned_patterns_{pattern_type}_{limit}"
        cached = self.get_cached_result(cache_key)
        if cached is not None:
            return cached
        
        with self._get_connection() as conn:
            if pattern_type:
                cursor = conn.execute("""
                    SELECT * FROM learned_patterns
                    WHERE pattern_type = ?
                    ORDER BY confidence DESC, success_rate DESC
                    LIMIT ?
                """, (pattern_type, limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM learned_patterns
                    ORDER BY confidence DESC, success_rate DESC
                    LIMIT ?
                """, (limit,))
            
            result = [dict(row) for row in cursor.fetchall()]
            self._cache_result(cache_key, result)
            return result
    
    def clear_cache(self):
        """Clear all cached results."""
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.query_cache),
            'max_cache_size': self.max_cache_size,
            'cache_ttl': self.cache_ttl,
            'cache_hit_ratio': getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1)
        }

# Global optimizer instance
_optimizer = None

def get_query_optimizer() -> QueryOptimizer:
    """Get global query optimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = QueryOptimizer()
    return _optimizer

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_recent_errors(hours: int = 2, limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent errors with optimization."""
    return get_query_optimizer().get_recent_errors_optimized(hours, limit)

def get_system_health(hours: int = 2) -> Dict[str, Any]:
    """Get system health with optimization."""
    return get_query_optimizer().get_system_health_optimized(hours)

def get_performance_trends(days: int = 7) -> Dict[str, Any]:
    """Get performance trends with optimization."""
    return get_query_optimizer().get_performance_trends_optimized(days)

def clear_query_cache():
    """Clear query cache."""
    get_query_optimizer().clear_cache()
