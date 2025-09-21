"""
Performance Analytics

Advanced performance analytics and reporting for the training system.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from ...training.interfaces import ComponentInterface
from ...training.caching import CacheManager, CacheConfig
from ...training.monitoring import PerformanceMonitor


class MetricType(Enum):
    """Types of performance metrics."""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    MEMORY = "memory"
    CPU = "cpu"
    GPU = "gpu"
    NETWORK = "network"
    CUSTOM = "custom"


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    metric_type: MetricType
    value: float
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class PerformanceReport:
    """Performance analysis report."""
    report_id: str
    generated_at: datetime
    time_range: Tuple[datetime, datetime]
    summary: Dict[str, Any]
    metrics: List[PerformanceMetrics]
    trends: Dict[str, Any]
    recommendations: List[str]


class PerformanceAnalytics(ComponentInterface):
    """
    Advanced performance analytics system for monitoring and analyzing
    system performance metrics.
    """
    
    def __init__(self, cache_config: Optional[CacheConfig] = None):
        """Initialize the performance analytics system."""
        self.cache_config = cache_config or CacheConfig()
        self.cache = CacheManager(self.cache_config)
        self.performance_monitor = PerformanceMonitor()
        
        # Analytics state
        self.metrics_history: List[PerformanceMetrics] = []
        self.reports: List[PerformanceReport] = []
        self.alert_thresholds: Dict[str, float] = {}
        
        # Performance tracking
        self.analysis_times: List[float] = []
        self.report_generation_times: List[float] = []
        
        # Analytics configuration
        self.retention_days = 30
        self.analysis_window_hours = 24
        self.alert_cooldown_minutes = 15
        
        self._initialized = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize the performance analytics system."""
        try:
            self.cache.initialize()
            self.performance_monitor.start_monitoring()
            
            # Set default alert thresholds
            self._set_default_alert_thresholds()
            
            self._initialized = True
            self.logger.info("Performance analytics system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize performance analytics: {e}")
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get current component state."""
        return {
            'name': 'PerformanceAnalytics',
            'status': 'running' if self._initialized else 'stopped',
            'last_updated': datetime.now(),
            'metadata': {
                'metrics_count': len(self.metrics_history),
                'reports_count': len(self.reports),
                'alert_thresholds': len(self.alert_thresholds),
                'average_analysis_time': np.mean(self.analysis_times) if self.analysis_times else 0.0
            }
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.performance_monitor.stop_monitoring()
            self.cache.clear()
            self._initialized = False
            self.logger.info("Performance analytics system cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self._initialized and self.cache.is_healthy()
    
    def add_metric(self, metric_type: MetricType, value: float, 
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a performance metric."""
        try:
            metric = PerformanceMetrics(
                metric_type=metric_type,
                value=value,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            
            self.metrics_history.append(metric)
            
            # Clean up old metrics
            self._cleanup_old_metrics()
            
            # Check for alerts
            self._check_alert_thresholds(metric)
            
            # Cache metric
            cache_key = f"metric_{metric.timestamp.timestamp()}"
            self.cache.set(cache_key, metric, ttl=3600)
            
            self.logger.debug(f"Added {metric_type.value} metric: {value}")
            
        except Exception as e:
            self.logger.error(f"Error adding metric: {e}")
    
    def get_metrics(self, metric_type: Optional[MetricType] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[PerformanceMetrics]:
        """Get performance metrics with optional filtering."""
        try:
            metrics = self.metrics_history.copy()
            
            # Filter by metric type
            if metric_type:
                metrics = [m for m in metrics if m.metric_type == metric_type]
            
            # Filter by time range
            if start_time:
                metrics = [m for m in metrics if m.timestamp >= start_time]
            if end_time:
                metrics = [m for m in metrics if m.timestamp <= end_time]
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}")
            return []
    
    def analyze_performance(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> PerformanceReport:
        """Analyze performance metrics and generate a report."""
        try:
            start_time = datetime.now()
            
            # Determine time range
            if time_range:
                start, end = time_range
            else:
                end = datetime.now()
                start = end - timedelta(hours=self.analysis_window_hours)
            
            # Get metrics for analysis
            metrics = self.get_metrics(start_time=start, end_time=end)
            
            if not metrics:
                return self._create_empty_report(start, end)
            
            # Analyze metrics
            summary = self._analyze_metrics_summary(metrics)
            trends = self._analyze_trends(metrics)
            recommendations = self._generate_recommendations(metrics, trends)
            
            # Create report
            report = PerformanceReport(
                report_id=f"perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                generated_at=datetime.now(),
                time_range=(start, end),
                summary=summary,
                metrics=metrics,
                trends=trends,
                recommendations=recommendations
            )
            
            # Store report
            self.reports.append(report)
            
            # Update performance metrics
            analysis_time = (datetime.now() - start_time).total_seconds()
            self.analysis_times.append(analysis_time)
            
            # Cache report
            cache_key = f"report_{report.report_id}"
            self.cache.set(cache_key, report, ttl=86400)  # 24 hours
            
            self.logger.info(f"Generated performance report in {analysis_time:.3f}s")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance: {e}")
            raise
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of current performance metrics."""
        try:
            if not self.metrics_history:
                return {'error': 'No metrics available'}
            
            # Get recent metrics (last hour)
            recent_cutoff = datetime.now() - timedelta(hours=1)
            recent_metrics = [m for m in self.metrics_history if m.timestamp >= recent_cutoff]
            
            if not recent_metrics:
                return {'error': 'No recent metrics available'}
            
            # Calculate summary statistics
            summary = {}
            
            for metric_type in MetricType:
                type_metrics = [m for m in recent_metrics if m.metric_type == metric_type]
                if type_metrics:
                    values = [m.value for m in type_metrics]
                    summary[metric_type.value] = {
                        'count': len(values),
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'latest': values[-1] if values else None
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}
    
    def set_alert_threshold(self, metric_type: MetricType, threshold: float) -> None:
        """Set alert threshold for a metric type."""
        try:
            self.alert_thresholds[metric_type.value] = threshold
            self.logger.info(f"Set alert threshold for {metric_type.value}: {threshold}")
        except Exception as e:
            self.logger.error(f"Error setting alert threshold: {e}")
    
    def get_alert_thresholds(self) -> Dict[str, float]:
        """Get current alert thresholds."""
        return self.alert_thresholds.copy()
    
    def get_analytics_statistics(self) -> Dict[str, Any]:
        """Get analytics system statistics."""
        try:
            return {
                'total_metrics': len(self.metrics_history),
                'total_reports': len(self.reports),
                'alert_thresholds_count': len(self.alert_thresholds),
                'average_analysis_time': np.mean(self.analysis_times) if self.analysis_times else 0.0,
                'average_report_generation_time': np.mean(self.report_generation_times) if self.report_generation_times else 0.0,
                'retention_days': self.retention_days,
                'analysis_window_hours': self.analysis_window_hours
            }
        except Exception as e:
            self.logger.error(f"Error getting analytics statistics: {e}")
            return {'error': str(e)}
    
    def _set_default_alert_thresholds(self) -> None:
        """Set default alert thresholds."""
        self.alert_thresholds = {
            MetricType.MEMORY.value: 0.9,  # 90% memory usage
            MetricType.CPU.value: 0.8,     # 80% CPU usage
            MetricType.LATENCY.value: 5.0, # 5 seconds latency
            MetricType.THROUGHPUT.value: 0.1  # 10% of expected throughput
        }
    
    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.retention_days)
            original_count = len(self.metrics_history)
            self.metrics_history = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
            removed_count = original_count - len(self.metrics_history)
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old metrics")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old metrics: {e}")
    
    def _check_alert_thresholds(self, metric: PerformanceMetrics) -> None:
        """Check if metric exceeds alert thresholds."""
        try:
            metric_type = metric.metric_type.value
            if metric_type not in self.alert_thresholds:
                return
            
            threshold = self.alert_thresholds[metric_type]
            if metric.value > threshold:
                self.logger.warning(f"Alert: {metric_type} value {metric.value} exceeds threshold {threshold}")
                # In a real system, you would trigger alerts here
                
        except Exception as e:
            self.logger.error(f"Error checking alert thresholds: {e}")
    
    def _create_empty_report(self, start: datetime, end: datetime) -> PerformanceReport:
        """Create an empty report when no metrics are available."""
        return PerformanceReport(
            report_id=f"perf_empty_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(),
            time_range=(start, end),
            summary={'error': 'No metrics available for analysis'},
            metrics=[],
            trends={},
            recommendations=['Collect more performance metrics']
        )
    
    def _analyze_metrics_summary(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Analyze metrics and create summary statistics."""
        try:
            summary = {}
            
            # Group metrics by type
            metrics_by_type = {}
            for metric in metrics:
                metric_type = metric.metric_type.value
                if metric_type not in metrics_by_type:
                    metrics_by_type[metric_type] = []
                metrics_by_type[metric_type].append(metric.value)
            
            # Calculate statistics for each type
            for metric_type, values in metrics_by_type.items():
                summary[metric_type] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'percentile_95': np.percentile(values, 95),
                    'percentile_99': np.percentile(values, 99)
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error analyzing metrics summary: {e}")
            return {}
    
    def _analyze_trends(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Analyze trends in the metrics."""
        try:
            trends = {}
            
            # Group metrics by type and sort by timestamp
            metrics_by_type = {}
            for metric in metrics:
                metric_type = metric.metric_type.value
                if metric_type not in metrics_by_type:
                    metrics_by_type[metric_type] = []
                metrics_by_type[metric_type].append(metric)
            
            # Analyze trends for each type
            for metric_type, type_metrics in metrics_by_type.items():
                type_metrics.sort(key=lambda x: x.timestamp)
                
                if len(type_metrics) < 2:
                    continue
                
                values = [m.value for m in type_metrics]
                timestamps = [m.timestamp for m in type_metrics]
                
                # Calculate trend direction
                trend_direction = self._calculate_trend_direction(values)
                
                # Calculate trend strength
                trend_strength = self._calculate_trend_strength(values)
                
                # Calculate volatility
                volatility = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                
                trends[metric_type] = {
                    'direction': trend_direction,
                    'strength': trend_strength,
                    'volatility': volatility,
                    'data_points': len(values),
                    'time_span_hours': (timestamps[-1] - timestamps[0]).total_seconds() / 3600
                }
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends: {e}")
            return {}
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate the direction of a trend."""
        try:
            if len(values) < 2:
                return 'insufficient_data'
            
            # Simple linear regression slope
            x = np.arange(len(values))
            y = np.array(values)
            
            # Calculate slope
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > 0.01:
                return 'increasing'
            elif slope < -0.01:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception as e:
            self.logger.error(f"Error calculating trend direction: {e}")
            return 'unknown'
    
    def _calculate_trend_strength(self, values: List[float]) -> float:
        """Calculate the strength of a trend."""
        try:
            if len(values) < 2:
                return 0.0
            
            # Calculate R-squared for linear trend
            x = np.arange(len(values))
            y = np.array(values)
            
            # Linear regression
            coeffs = np.polyfit(x, y, 1)
            y_pred = np.polyval(coeffs, x)
            
            # Calculate R-squared
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            
            if ss_tot == 0:
                return 0.0
            
            r_squared = 1 - (ss_res / ss_tot)
            return max(0.0, min(1.0, r_squared))
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return 0.0
    
    def _generate_recommendations(self, metrics: List[PerformanceMetrics], 
                                trends: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on analysis."""
        try:
            recommendations = []
            
            # Check memory usage trends
            if 'memory' in trends:
                memory_trend = trends['memory']
                if memory_trend['direction'] == 'increasing' and memory_trend['strength'] > 0.7:
                    recommendations.append("Memory usage is increasing significantly. Consider optimizing memory usage or increasing available memory.")
            
            # Check CPU usage trends
            if 'cpu' in trends:
                cpu_trend = trends['cpu']
                if cpu_trend['direction'] == 'increasing' and cpu_trend['strength'] > 0.7:
                    recommendations.append("CPU usage is increasing significantly. Consider optimizing CPU-intensive operations or scaling horizontally.")
            
            # Check latency trends
            if 'latency' in trends:
                latency_trend = trends['latency']
                if latency_trend['direction'] == 'increasing' and latency_trend['strength'] > 0.7:
                    recommendations.append("Latency is increasing significantly. Consider optimizing network operations or reducing processing complexity.")
            
            # Check throughput trends
            if 'throughput' in trends:
                throughput_trend = trends['throughput']
                if throughput_trend['direction'] == 'decreasing' and throughput_trend['strength'] > 0.7:
                    recommendations.append("Throughput is decreasing significantly. Consider optimizing data processing or increasing system capacity.")
            
            # Check for high volatility
            for metric_type, trend in trends.items():
                if trend['volatility'] > 0.5:
                    recommendations.append(f"{metric_type} shows high volatility. Consider investigating the root cause of fluctuations.")
            
            # Check for insufficient data
            for metric_type, trend in trends.items():
                if trend['data_points'] < 10:
                    recommendations.append(f"Insufficient data for {metric_type}. Collect more metrics for better analysis.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ['Error generating recommendations']
