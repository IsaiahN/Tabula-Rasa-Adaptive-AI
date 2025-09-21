"""
Reporting Module

This module provides reporting capabilities for the training system.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .performance_monitor import PerformanceMonitor, PerformanceMetrics
from .metrics_collector import MetricsCollector, MetricType

logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Types of reports that can be generated."""
    SUMMARY = "summary"
    DETAILED = "detailed"
    PERFORMANCE = "performance"
    METRICS = "metrics"
    HEALTH = "health"

@dataclass
class ReportData:
    """Report data structure."""
    report_type: ReportType
    timestamp: datetime
    data: Dict[str, Any]
    summary: str

class ReportGenerator:
    """Report generation system."""
    
    def __init__(self, performance_monitor: PerformanceMonitor, 
                 metrics_collector: MetricsCollector):
        self.performance_monitor = performance_monitor
        self.metrics_collector = metrics_collector
        self.logger = logger
        
    def generate_report(self, report_type: ReportType, 
                       time_range: Optional[timedelta] = None) -> ReportData:
        """Generate a report of the specified type."""
        timestamp = datetime.now()
        
        if report_type == ReportType.SUMMARY:
            data = self._generate_summary_report(time_range)
        elif report_type == ReportType.DETAILED:
            data = self._generate_detailed_report(time_range)
        elif report_type == ReportType.PERFORMANCE:
            data = self._generate_performance_report(time_range)
        elif report_type == ReportType.METRICS:
            data = self._generate_metrics_report(time_range)
        elif report_type == ReportType.HEALTH:
            data = self._generate_health_report(time_range)
        else:
            data = {"error": f"Unknown report type: {report_type}"}
            
        summary = self._generate_summary_text(report_type, data)
        
        return ReportData(
            report_type=report_type,
            timestamp=timestamp,
            data=data,
            summary=summary
        )
        
    def _generate_summary_report(self, time_range: Optional[timedelta]) -> Dict[str, Any]:
        """Generate a summary report."""
        metrics = self._filter_metrics_by_time(self.performance_monitor.get_metrics(), time_range)
        
        return {
            "total_operations": len(metrics),
            "success_rate": self.performance_monitor.get_success_rate(),
            "average_duration": self.performance_monitor.get_average_duration(),
            "time_range": str(time_range) if time_range else "all_time",
            "top_operations": self._get_top_operations(metrics)
        }
        
    def _generate_detailed_report(self, time_range: Optional[timedelta]) -> Dict[str, Any]:
        """Generate a detailed report."""
        metrics = self._filter_metrics_by_time(self.performance_monitor.get_metrics(), time_range)
        all_metrics = self._filter_metrics_by_time(self.metrics_collector.get_all_metrics(), time_range)
        
        return {
            "performance": {
                "total_operations": len(metrics),
                "success_rate": self.performance_monitor.get_success_rate(),
                "average_duration": self.performance_monitor.get_average_duration(),
                "operations_by_type": self._group_operations_by_type(metrics)
            },
            "metrics": {
                "total_metrics": len(all_metrics),
                "counters": dict(self.metrics_collector.counters),
                "gauges": dict(self.metrics_collector.gauges),
                "histograms": {name: self.metrics_collector.get_histogram_stats(name) 
                              for name in self.metrics_collector.histograms},
                "timers": {name: self.metrics_collector.get_timer_stats(name) 
                          for name in self.metrics_collector.timers}
            },
            "time_range": str(time_range) if time_range else "all_time"
        }
        
    def _generate_performance_report(self, time_range: Optional[timedelta]) -> Dict[str, Any]:
        """Generate a performance-focused report."""
        metrics = self._filter_metrics_by_time(self.performance_monitor.get_metrics(), time_range)
        
        return {
            "performance_summary": {
                "total_operations": len(metrics),
                "success_rate": self.performance_monitor.get_success_rate(),
                "average_duration": self.performance_monitor.get_average_duration(),
                "min_duration": min((m.duration for m in metrics), default=0.0),
                "max_duration": max((m.duration for m in metrics), default=0.0)
            },
            "operation_breakdown": self._group_operations_by_type(metrics),
            "recent_operations": [self._format_metric(m) for m in metrics[-10:]]  # Last 10
        }
        
    def _generate_metrics_report(self, time_range: Optional[timedelta]) -> Dict[str, Any]:
        """Generate a metrics-focused report."""
        all_metrics = self._filter_metrics_by_time(self.metrics_collector.get_all_metrics(), time_range)
        
        return {
            "metrics_summary": {
                "total_metrics": len(all_metrics),
                "counters": len(self.metrics_collector.counters),
                "gauges": len(self.metrics_collector.gauges),
                "histograms": len(self.metrics_collector.histograms),
                "timers": len(self.metrics_collector.timers)
            },
            "counter_values": dict(self.metrics_collector.counters),
            "gauge_values": dict(self.metrics_collector.gauges),
            "histogram_stats": {name: self.metrics_collector.get_histogram_stats(name) 
                              for name in self.metrics_collector.histograms},
            "timer_stats": {name: self.metrics_collector.get_timer_stats(name) 
                          for name in self.metrics_collector.timers}
        }
        
    def _generate_health_report(self, time_range: Optional[timedelta]) -> Dict[str, Any]:
        """Generate a health-focused report."""
        metrics = self._filter_metrics_by_time(self.performance_monitor.get_metrics(), time_range)
        
        recent_metrics = [m for m in metrics if 
                         (datetime.now() - m.timestamp).total_seconds() < 300]  # Last 5 minutes
        
        return {
            "system_health": {
                "status": "healthy" if recent_metrics else "warning",
                "recent_activity": len(recent_metrics),
                "success_rate": self.performance_monitor.get_success_rate(),
                "average_response_time": self.performance_monitor.get_average_duration()
            },
            "alerts": [],  # Would integrate with alert system
            "recommendations": self._generate_recommendations(metrics)
        }
        
    def _filter_metrics_by_time(self, metrics: List, time_range: Optional[timedelta]) -> List:
        """Filter metrics by time range."""
        if not time_range:
            return metrics
            
        cutoff_time = datetime.now() - time_range
        return [m for m in metrics if m.timestamp >= cutoff_time]
        
    def _get_top_operations(self, metrics: List[PerformanceMetrics]) -> List[Dict[str, Any]]:
        """Get top operations by count."""
        operation_counts = {}
        for metric in metrics:
            operation = metric.operation
            if operation not in operation_counts:
                operation_counts[operation] = {"count": 0, "total_duration": 0.0, "successes": 0}
            operation_counts[operation]["count"] += 1
            operation_counts[operation]["total_duration"] += metric.duration
            if metric.success:
                operation_counts[operation]["successes"] += 1
                
        # Convert to list and sort by count
        top_operations = []
        for operation, data in operation_counts.items():
            top_operations.append({
                "operation": operation,
                "count": data["count"],
                "average_duration": data["total_duration"] / data["count"],
                "success_rate": data["successes"] / data["count"]
            })
            
        return sorted(top_operations, key=lambda x: x["count"], reverse=True)[:10]
        
    def _group_operations_by_type(self, metrics: List[PerformanceMetrics]) -> Dict[str, int]:
        """Group operations by type."""
        operation_counts = {}
        for metric in metrics:
            operation = metric.operation
            operation_counts[operation] = operation_counts.get(operation, 0) + 1
        return operation_counts
        
    def _format_metric(self, metric: PerformanceMetrics) -> Dict[str, Any]:
        """Format a metric for display."""
        return {
            "operation": metric.operation,
            "duration": metric.duration,
            "success": metric.success,
            "timestamp": metric.timestamp.isoformat(),
            "error": metric.error_message
        }
        
    def _generate_recommendations(self, metrics: List[PerformanceMetrics]) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        if not metrics:
            recommendations.append("No recent activity detected")
            return recommendations
            
        success_rate = self.performance_monitor.get_success_rate()
        if success_rate < 0.8:
            recommendations.append(f"Low success rate ({success_rate:.1%}), investigate failures")
            
        avg_duration = self.performance_monitor.get_average_duration()
        if avg_duration > 10.0:  # 10 seconds
            recommendations.append(f"High average duration ({avg_duration:.2f}s), consider optimization")
            
        return recommendations
        
    def _generate_summary_text(self, report_type: ReportType, data: Dict[str, Any]) -> str:
        """Generate a text summary of the report."""
        if "error" in data:
            return f"Error generating {report_type.value} report: {data['error']}"
            
        if report_type == ReportType.SUMMARY:
            return (f"Summary Report: {data['total_operations']} operations, "
                   f"{data['success_rate']:.1%} success rate, "
                   f"{data['average_duration']:.2f}s average duration")
        elif report_type == ReportType.PERFORMANCE:
            perf = data['performance_summary']
            return (f"Performance Report: {perf['total_operations']} operations, "
                   f"{perf['success_rate']:.1%} success rate, "
                   f"{perf['average_duration']:.2f}s average duration")
        else:
            return f"{report_type.value.title()} report generated at {datetime.now().isoformat()}"
