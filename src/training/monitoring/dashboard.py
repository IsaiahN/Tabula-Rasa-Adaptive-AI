"""
Monitoring Dashboard Module

This module provides a monitoring dashboard for the training system.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass

from .performance_monitor import PerformanceMonitor, PerformanceMetrics
from .metrics_collector import MetricsCollector, MetricType

logger = logging.getLogger(__name__)

@dataclass
class DashboardData:
    """Dashboard data structure."""
    timestamp: datetime
    performance_summary: Dict[str, Any]
    metrics_summary: Dict[str, Any]
    system_health: Dict[str, Any]
    alerts: List[Dict[str, Any]]

class MonitoringDashboard:
    """Monitoring dashboard system."""
    
    def __init__(self, performance_monitor: PerformanceMonitor, 
                 metrics_collector: MetricsCollector):
        self.performance_monitor = performance_monitor
        self.metrics_collector = metrics_collector
        self.logger = logger
        self.alerts: List[Dict[str, Any]] = []
        
    def get_dashboard_data(self) -> DashboardData:
        """Get current dashboard data."""
        timestamp = datetime.now()
        
        # Get performance summary
        performance_summary = self._get_performance_summary()
        
        # Get metrics summary
        metrics_summary = self._get_metrics_summary()
        
        # Get system health
        system_health = self._get_system_health()
        
        return DashboardData(
            timestamp=timestamp,
            performance_summary=performance_summary,
            metrics_summary=metrics_summary,
            system_health=system_health,
            alerts=self.alerts.copy()
        )
        
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary data."""
        metrics = self.performance_monitor.get_metrics()
        
        if not metrics:
            return {
                "total_operations": 0,
                "success_rate": 0.0,
                "average_duration": 0.0,
                "recent_operations": []
            }
            
        recent_metrics = [m for m in metrics if 
                         (datetime.now() - m.timestamp).total_seconds() < 3600]  # Last hour
        
        return {
            "total_operations": len(metrics),
            "success_rate": self.performance_monitor.get_success_rate(),
            "average_duration": self.performance_monitor.get_average_duration(),
            "recent_operations": len(recent_metrics),
            "operations_by_type": self._group_operations_by_type(metrics)
        }
        
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary data."""
        all_metrics = self.metrics_collector.get_all_metrics()
        
        if not all_metrics:
            return {
                "total_metrics": 0,
                "counters": {},
                "gauges": {},
                "histograms": {},
                "timers": {}
            }
            
        return {
            "total_metrics": len(all_metrics),
            "counters": dict(self.metrics_collector.counters),
            "gauges": dict(self.metrics_collector.gauges),
            "histograms": {name: self.metrics_collector.get_histogram_stats(name) 
                          for name in self.metrics_collector.histograms},
            "timers": {name: self.metrics_collector.get_timer_stats(name) 
                      for name in self.metrics_collector.timers}
        }
        
    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health data."""
        # Simplified health check
        recent_metrics = self.performance_monitor.get_metrics()
        recent_count = len([m for m in recent_metrics if 
                           (datetime.now() - m.timestamp).total_seconds() < 300])  # Last 5 minutes
        
        return {
            "status": "healthy" if recent_count > 0 else "warning",
            "recent_activity": recent_count,
            "uptime": "unknown",  # Would calculate actual uptime
            "memory_usage": 0.0,  # Would get actual memory usage
            "cpu_usage": 0.0      # Would get actual CPU usage
        }
        
    def _group_operations_by_type(self, metrics: List[PerformanceMetrics]) -> Dict[str, int]:
        """Group operations by type."""
        operation_counts = {}
        for metric in metrics:
            operation = metric.operation
            operation_counts[operation] = operation_counts.get(operation, 0) + 1
        return operation_counts
        
    def add_alert(self, message: str, severity: str = "warning", 
                  operation: Optional[str] = None):
        """Add an alert to the dashboard."""
        alert = {
            "timestamp": datetime.now(),
            "message": message,
            "severity": severity,
            "operation": operation
        }
        self.alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
            
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts.clear()
        
    def get_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get alerts, optionally filtered by severity."""
        if severity:
            return [alert for alert in self.alerts if alert["severity"] == severity]
        return self.alerts.copy()
