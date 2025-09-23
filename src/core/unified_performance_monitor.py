#!/usr/bin/env python3
"""
Unified Performance Monitoring System

This module consolidates all performance monitoring functionality into a single,
comprehensive system that replaces multiple duplicate monitors throughout the codebase.

Consolidates functionality from:
- src/core/performance_monitor.py
- src/training/monitoring/performance_monitor.py
- src/training/monitoring/system_monitor.py
- src/monitoring/performance_monitor.py
- src/training/performance/performance_monitor.py
"""

import time
import psutil
import threading
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import numpy as np
from functools import wraps

logger = logging.getLogger(__name__)

class MonitorLevel(Enum):
    """Performance monitoring levels."""
    BASIC = "basic"           # Basic system metrics
    DETAILED = "detailed"     # Detailed component metrics
    PROFILING = "profiling"   # Full profiling and analysis
    DEBUG = "debug"          # Debug-level monitoring

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_io_bytes: int
    process_count: int
    load_average: List[float]
    
    # Component-specific metrics
    database_queries: int = 0
    database_query_time: float = 0.0
    api_calls: int = 0
    api_response_time: float = 0.0
    training_iterations: int = 0
    training_time: float = 0.0
    game_actions: int = 0
    game_score: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceAlert:
    """Performance alert."""
    alert_id: str
    level: AlertLevel
    component: str
    metric: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: float
    resolved: bool = False

class UnifiedPerformanceMonitor:
    """
    Unified Performance Monitoring System
    
    Consolidates all performance monitoring functionality into a single system.
    Provides comprehensive monitoring for all system components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.monitoring_active = False
        self.monitor_level = MonitorLevel(self.config.get('monitor_level', 'detailed'))
        self.collection_interval = self.config.get('collection_interval', 5.0)
        self.retention_hours = self.config.get('retention_hours', 24)
        
        # Metrics storage
        self.metrics_history = deque(maxlen=int(self.retention_hours * 3600 / self.collection_interval))
        self.component_metrics = defaultdict(lambda: defaultdict(list))
        self.custom_metrics = defaultdict(list)
        
        # Alerting
        self.alerts = []
        self.alert_thresholds = self.config.get('alert_thresholds', {})
        self.alert_callbacks = []
        
        # Performance tracking
        self.performance_timers = {}
        self.query_times = deque(maxlen=1000)
        self.api_times = deque(maxlen=1000)
        
        # Threading
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
        logger.info("🔍 Unified Performance Monitor initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default monitoring configuration."""
        return {
            'monitor_level': 'detailed',
            'collection_interval': 5.0,
            'retention_hours': 24,
            'alert_thresholds': {
                'cpu_percent': 80.0,
                'memory_percent': 85.0,
                'disk_usage_percent': 90.0,
                'database_query_time': 1.0,
                'api_response_time': 2.0
            }
        }
    
    def start_monitoring(self) -> bool:
        """Start performance monitoring."""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return False
        
        try:
            self.monitoring_active = True
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitor_thread.start()
            
            logger.info("🔍 Performance monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start performance monitoring: {e}")
            self.monitoring_active = False
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop performance monitoring."""
        if not self.monitoring_active:
            logger.warning("Performance monitoring not active")
            return False
        
        try:
            self.monitoring_active = False
            self._stop_event.set()
            
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)
            
            logger.info("🔍 Performance monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop performance monitoring: {e}")
            return False
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                # Sleep until next collection
                self._stop_event.wait(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self._stop_event.wait(10.0)  # Wait 10 seconds on error
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive system metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average (Unix only)
            try:
                load_avg = list(psutil.getloadavg())
            except AttributeError:
                load_avg = [0.0, 0.0, 0.0]
            
            # Component metrics
            database_queries = len(self.query_times)
            database_query_time = np.mean(self.query_times) if self.query_times else 0.0
            api_calls = len(self.api_times)
            api_response_time = np.mean(self.api_times) if self.api_times else 0.0
            
            return PerformanceMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                network_io_bytes=network.bytes_sent + network.bytes_recv,
                process_count=process_count,
                load_average=load_avg,
                database_queries=database_queries,
                database_query_time=database_query_time,
                api_calls=api_calls,
                api_response_time=api_response_time,
                custom_metrics={}
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return PerformanceMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                disk_free_gb=0.0,
                network_io_bytes=0,
                process_count=0,
                load_average=[0.0, 0.0, 0.0]
            )
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts."""
        try:
            # CPU alert
            if metrics.cpu_percent > self.alert_thresholds.get('cpu_percent', 80.0):
                self._create_alert(
                    'cpu_high',
                    AlertLevel.WARNING,
                    'system',
                    'cpu_percent',
                    metrics.cpu_percent,
                    self.alert_thresholds['cpu_percent'],
                    f"High CPU usage: {metrics.cpu_percent:.1f}%"
                )
            
            # Memory alert
            if metrics.memory_percent > self.alert_thresholds.get('memory_percent', 85.0):
                self._create_alert(
                    'memory_high',
                    AlertLevel.WARNING,
                    'system',
                    'memory_percent',
                    metrics.memory_percent,
                    self.alert_thresholds['memory_percent'],
                    f"High memory usage: {metrics.memory_percent:.1f}%"
                )
            
            # Disk alert
            if metrics.disk_usage_percent > self.alert_thresholds.get('disk_usage_percent', 90.0):
                self._create_alert(
                    'disk_high',
                    AlertLevel.CRITICAL,
                    'system',
                    'disk_usage_percent',
                    metrics.disk_usage_percent,
                    self.alert_thresholds['disk_usage_percent'],
                    f"High disk usage: {metrics.disk_usage_percent:.1f}%"
                )
            
            # Database query time alert
            if metrics.database_query_time > self.alert_thresholds.get('database_query_time', 1.0):
                self._create_alert(
                    'database_slow',
                    AlertLevel.WARNING,
                    'database',
                    'database_query_time',
                    metrics.database_query_time,
                    self.alert_thresholds['database_query_time'],
                    f"Slow database queries: {metrics.database_query_time:.2f}s"
                )
            
            # API response time alert
            if metrics.api_response_time > self.alert_thresholds.get('api_response_time', 2.0):
                self._create_alert(
                    'api_slow',
                    AlertLevel.WARNING,
                    'api',
                    'api_response_time',
                    metrics.api_response_time,
                    self.alert_thresholds['api_response_time'],
                    f"Slow API responses: {metrics.api_response_time:.2f}s"
                )
                
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def _create_alert(self, alert_id: str, level: AlertLevel, component: str, 
                     metric: str, current_value: float, threshold_value: float, message: str):
        """Create a performance alert."""
        alert = PerformanceAlert(
            alert_id=alert_id,
            level=level,
            component=component,
            metric=metric,
            current_value=current_value,
            threshold_value=threshold_value,
            message=message,
            timestamp=time.time()
        )
        
        self.alerts.append(alert)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"🚨 Performance Alert: {message}")
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def track_database_query(self, query_time: float):
        """Track database query performance."""
        self.query_times.append(query_time)
    
    def track_api_call(self, response_time: float):
        """Track API call performance."""
        self.api_times.append(response_time)
    
    def track_training_iteration(self, iteration_time: float, score: float = 0.0):
        """Track training iteration performance."""
        # This would be called by training systems
        pass
    
    def track_game_action(self, action_time: float, score: float = 0.0):
        """Track game action performance."""
        # This would be called by game systems
        pass
    
    def add_custom_metric(self, name: str, value: Any, component: str = "custom"):
        """Add custom metric."""
        self.custom_metrics[name].append({
            'value': value,
            'component': component,
            'timestamp': time.time()
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        latest = self.metrics_history[-1]
        
        # Calculate averages over last hour
        recent_metrics = list(self.metrics_history)[-int(3600 / self.collection_interval):]
        
        return {
            "current": asdict(latest),
            "averages": {
                "cpu_percent": np.mean([m.cpu_percent for m in recent_metrics]),
                "memory_percent": np.mean([m.memory_percent for m in recent_metrics]),
                "database_query_time": np.mean([m.database_query_time for m in recent_metrics]),
                "api_response_time": np.mean([m.api_response_time for m in recent_metrics])
            },
            "alerts": {
                "total": len(self.alerts),
                "unresolved": len([a for a in self.alerts if not a.resolved]),
                "recent": [asdict(a) for a in self.alerts[-10:]]
            },
            "monitoring_status": {
                "active": self.monitoring_active,
                "level": self.monitor_level.value,
                "collection_interval": self.collection_interval,
                "metrics_count": len(self.metrics_history)
            }
        }
    
    def get_component_metrics(self, component: str) -> Dict[str, Any]:
        """Get metrics for specific component."""
        return dict(self.component_metrics[component])
    
    def clear_old_metrics(self, hours: int = 24):
        """Clear metrics older than specified hours."""
        cutoff_time = time.time() - (hours * 3600)
        self.metrics_history = deque(
            [m for m in self.metrics_history if m.timestamp > cutoff_time],
            maxlen=self.metrics_history.maxlen
        )
        logger.info(f"🧹 Cleared metrics older than {hours} hours")
    
    def performance_timer(self, name: str):
        """Context manager for performance timing."""
        return PerformanceTimer(self, name)

class PerformanceTimer:
    """Context manager for performance timing."""
    
    def __init__(self, monitor: UnifiedPerformanceMonitor, name: str):
        self.monitor = monitor
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitor.add_custom_metric(f"timer_{self.name}", duration, "timing")

def monitor_performance(monitor: UnifiedPerformanceMonitor, component: str = "unknown"):
    """Decorator for monitoring function performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with monitor.performance_timer(f"{component}_{func.__name__}"):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Global monitor instance
_global_monitor = None

def get_performance_monitor() -> UnifiedPerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = UnifiedPerformanceMonitor()
    return _global_monitor

def start_global_monitoring():
    """Start global performance monitoring."""
    monitor = get_performance_monitor()
    return monitor.start_monitoring()

def stop_global_monitoring():
    """Stop global performance monitoring."""
    monitor = get_performance_monitor()
    return monitor.stop_monitoring()
