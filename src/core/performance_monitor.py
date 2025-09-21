#!/usr/bin/env python3
"""
Comprehensive Performance Monitoring System

This module provides detailed performance monitoring for all system components with:
- Real-time metrics collection
- Performance profiling and analysis
- Memory usage tracking
- CPU usage monitoring
- Database query performance
- Component-specific metrics
- Performance alerts and thresholds
"""

import time
import psutil
import threading
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import asyncio
from functools import wraps
import numpy as np

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """A single performance metric."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    component: str
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'timestamp': self.timestamp.isoformat(),
            'component': self.component,
            'tags': self.tags
        }


@dataclass
class PerformanceAlert:
    """A performance alert."""
    alert_id: str
    component: str
    metric_name: str
    level: AlertLevel
    message: str
    threshold: float
    current_value: float
    timestamp: datetime
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'component': self.component,
            'metric_name': self.metric_name,
            'level': self.level.value,
            'message': self.message,
            'threshold': self.threshold,
            'current_value': self.current_value,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved
        }


@dataclass
class ComponentStats:
    """Statistics for a specific component."""
    component_name: str
    total_calls: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    error_count: int = 0
    success_rate: float = 1.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_call(self, duration_ms: float, success: bool = True):
        """Update statistics for a function call."""
        self.total_calls += 1
        self.total_time_ms += duration_ms
        self.avg_time_ms = self.total_time_ms / self.total_calls
        self.min_time_ms = min(self.min_time_ms, duration_ms)
        self.max_time_ms = max(self.max_time_ms, duration_ms)
        
        if not success:
            self.error_count += 1
        
        self.success_rate = (self.total_calls - self.error_count) / self.total_calls
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'component_name': self.component_name,
            'total_calls': self.total_calls,
            'total_time_ms': self.total_time_ms,
            'avg_time_ms': self.avg_time_ms,
            'min_time_ms': self.min_time_ms if self.min_time_ms != float('inf') else 0.0,
            'max_time_ms': self.max_time_ms,
            'error_count': self.error_count,
            'success_rate': self.success_rate,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'last_updated': self.last_updated.isoformat()
        }


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: deque = deque(maxlen=10000)
        self.component_stats: Dict[str, ComponentStats] = defaultdict(lambda: ComponentStats(""))
        self.alerts: List[PerformanceAlert] = []
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.lock = threading.RLock()
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # System monitoring
        self.process = psutil.Process()
        self.start_time = time.time()
        
        logger.info("Performance Monitor initialized")
    
    def start_monitoring(self, interval_seconds: int = 5):
        """Start background monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Performance monitoring started with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self, interval_seconds: int):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                self._check_thresholds()
                self._cleanup_old_data()
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # Memory usage
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # CPU usage
            cpu_percent = self.process.cpu_percent()
            
            # Record metrics
            self.record_metric("system.memory_usage_mb", memory_mb, "system")
            self.record_metric("system.cpu_usage_percent", cpu_percent, "system")
            
            # Update component stats
            for component_name, stats in self.component_stats.items():
                stats.memory_usage_mb = memory_mb
                stats.cpu_usage_percent = cpu_percent
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def _check_thresholds(self):
        """Check if any metrics exceed thresholds."""
        for metric in list(self.metrics)[-100:]:  # Check last 100 metrics
            if metric.name in self.thresholds:
                thresholds = self.thresholds[metric.name]
                for level, threshold in thresholds.items():
                    if metric.value > threshold:
                        self._create_alert(metric, level, threshold)
    
    def _create_alert(self, metric: PerformanceMetric, level: str, threshold: float):
        """Create a performance alert."""
        alert_id = f"{metric.component}_{metric.name}_{int(time.time())}"
        
        # Check if alert already exists
        for alert in self.alerts:
            if (alert.component == metric.component and 
                alert.metric_name == metric.name and 
                not alert.resolved):
                return
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            component=metric.component,
            metric_name=metric.name,
            level=AlertLevel(level),
            message=f"{metric.name} exceeded {level} threshold ({threshold})",
            threshold=threshold,
            current_value=metric.value,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        logger.warning(f"Performance alert: {alert.message}")
    
    def _cleanup_old_data(self):
        """Clean up old metrics and alerts."""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        # Remove old metrics
        while self.metrics and self.metrics[0].timestamp < cutoff_time:
            self.metrics.popleft()
        
        # Remove old alerts
        self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]
    
    def record_metric(self, name: str, value: float, component: str, 
                     metric_type: MetricType = MetricType.GAUGE, 
                     tags: Optional[Dict[str, str]] = None):
        """Record a performance metric."""
        with self.lock:
            metric = PerformanceMetric(
                name=name,
                value=value,
                metric_type=metric_type,
                timestamp=datetime.now(),
                component=component,
                tags=tags or {}
            )
            self.metrics.append(metric)
    
    def record_timing(self, component: str, function_name: str, duration_ms: float, 
                     success: bool = True):
        """Record timing information for a function call."""
        with self.lock:
            # Update component stats
            if component not in self.component_stats:
                self.component_stats[component] = ComponentStats(component)
            
            self.component_stats[component].update_call(duration_ms, success)
            
            # Record timing metric
            self.record_metric(
                f"{component}.{function_name}_duration_ms",
                duration_ms,
                component,
                MetricType.TIMER
            )
            
            # Record success/failure
            self.record_metric(
                f"{component}.{function_name}_success",
                1.0 if success else 0.0,
                component,
                MetricType.COUNTER
            )
    
    def set_threshold(self, metric_name: str, level: str, threshold: float):
        """Set a performance threshold."""
        if metric_name not in self.thresholds:
            self.thresholds[metric_name] = {}
        self.thresholds[metric_name][level] = threshold
    
    def get_metrics(self, component: Optional[str] = None, 
                   metric_name: Optional[str] = None,
                   last_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get performance metrics."""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(minutes=last_minutes)
            
            filtered_metrics = []
            for metric in self.metrics:
                if metric.timestamp < cutoff_time:
                    continue
                
                if component and metric.component != component:
                    continue
                
                if metric_name and metric.name != metric_name:
                    continue
                
                filtered_metrics.append(metric.to_dict())
            
            return filtered_metrics
    
    def get_component_stats(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Get component statistics."""
        with self.lock:
            if component:
                if component in self.component_stats:
                    return {component: self.component_stats[component].to_dict()}
                else:
                    return {}
            else:
                return {name: stats.to_dict() for name, stats in self.component_stats.items()}
    
    def get_alerts(self, level: Optional[AlertLevel] = None, 
                  resolved: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get performance alerts."""
        with self.lock:
            filtered_alerts = []
            for alert in self.alerts:
                if level and alert.level != level:
                    continue
                if resolved is not None and alert.resolved != resolved:
                    continue
                filtered_alerts.append(alert.to_dict())
            
            return filtered_alerts
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve a performance alert."""
        with self.lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    return True
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self.lock:
            total_metrics = len(self.metrics)
            total_alerts = len(self.alerts)
            active_alerts = len([a for a in self.alerts if not a.resolved])
            
            # Calculate averages
            if self.metrics:
                avg_values = defaultdict(list)
                for metric in self.metrics:
                    avg_values[metric.name].append(metric.value)
                
                avg_metrics = {}
                for name, values in avg_values.items():
                    avg_metrics[name] = {
                        'avg': np.mean(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
            else:
                avg_metrics = {}
            
            return {
                'total_metrics': total_metrics,
                'total_alerts': total_alerts,
                'active_alerts': active_alerts,
                'component_count': len(self.component_stats),
                'avg_metrics': avg_metrics,
                'uptime_seconds': time.time() - self.start_time
            }


# Global monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_global_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
        _global_monitor.start_monitoring()
    return _global_monitor


def monitor_performance(component: str, function_name: Optional[str] = None):
    """Decorator to monitor function performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_global_monitor()
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                func_name = function_name or func.__name__
                monitor.record_timing(component, func_name, duration_ms, success)
        
        return wrapper
    return decorator


def monitor_async_performance(component: str, function_name: Optional[str] = None):
    """Decorator to monitor async function performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            monitor = get_global_monitor()
            start_time = time.time()
            success = True
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                func_name = function_name or func.__name__
                monitor.record_timing(component, func_name, duration_ms, success)
        
        return wrapper
    return decorator


# Factory function
def create_performance_monitor(retention_hours: int = 24) -> PerformanceMonitor:
    """Create a new performance monitor instance."""
    return PerformanceMonitor(retention_hours)
