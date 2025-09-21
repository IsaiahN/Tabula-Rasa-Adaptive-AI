"""
System Analytics

Advanced analytics for system health, resource usage, and performance monitoring.
"""

import numpy as np
import pandas as pd
import psutil
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from ...training.interfaces import ComponentInterface
from ...training.caching import CacheManager, CacheConfig
from ...training.monitoring import PerformanceMonitor


class SystemMetricType(Enum):
    """Types of system metrics."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_IO = "network_io"
    PROCESS_COUNT = "process_count"
    THREAD_COUNT = "thread_count"
    FILE_DESCRIPTORS = "file_descriptors"
    SYSTEM_LOAD = "system_load"


@dataclass
class SystemMetrics:
    """System metrics data structure."""
    metric_type: SystemMetricType
    value: float
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class SystemReport:
    """System analysis report."""
    report_id: str
    generated_at: datetime
    time_range: Tuple[datetime, datetime]
    summary: Dict[str, Any]
    metrics: List[SystemMetrics]
    health_score: float
    recommendations: List[str]


class SystemAnalytics(ComponentInterface):
    """
    Advanced system analytics for monitoring system health,
    resource usage, and performance.
    """
    
    def __init__(self, cache_config: Optional[CacheConfig] = None):
        """Initialize the system analytics system."""
        self.cache_config = cache_config or CacheConfig()
        self.cache = CacheManager(self.cache_config)
        self.performance_monitor = PerformanceMonitor()
        
        # Analytics state
        self.metrics_history: List[SystemMetrics] = []
        self.reports: List[SystemReport] = []
        self.health_scores: List[float] = []
        
        # Performance tracking
        self.analysis_times: List[float] = []
        self.report_generation_times: List[float] = []
        
        # System monitoring configuration
        self.retention_days = 7  # Shorter retention for system metrics
        self.analysis_window_hours = 1  # Shorter analysis window
        self.health_thresholds = {
            'cpu_usage': 0.8,      # 80% CPU usage
            'memory_usage': 0.9,   # 90% memory usage
            'disk_usage': 0.95,    # 95% disk usage
            'system_load': 4.0     # Load average > 4
        }
        
        self._initialized = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize the system analytics system."""
        try:
            self.cache.initialize()
            self.performance_monitor.start_monitoring()
            
            self._initialized = True
            self.logger.info("System analytics system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize system analytics: {e}")
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get current component state."""
        return {
            'name': 'SystemAnalytics',
            'status': 'running' if self._initialized else 'stopped',
            'last_updated': datetime.now(),
            'metadata': {
                'metrics_count': len(self.metrics_history),
                'reports_count': len(self.reports),
                'health_scores_count': len(self.health_scores),
                'average_analysis_time': np.mean(self.analysis_times) if self.analysis_times else 0.0
            }
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.performance_monitor.stop_monitoring()
            self.cache.clear()
            self._initialized = False
            self.logger.info("System analytics system cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self._initialized and self.cache.is_healthy()
    
    def collect_system_metrics(self) -> List[SystemMetrics]:
        """Collect current system metrics."""
        try:
            metrics = []
            current_time = datetime.now()
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(SystemMetrics(
                metric_type=SystemMetricType.CPU_USAGE,
                value=cpu_percent,
                timestamp=current_time,
                metadata={'cpu_count': psutil.cpu_count()}
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics.append(SystemMetrics(
                metric_type=SystemMetricType.MEMORY_USAGE,
                value=memory.percent,
                timestamp=current_time,
                metadata={
                    'total_memory': memory.total,
                    'available_memory': memory.available,
                    'used_memory': memory.used
                }
            ))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            metrics.append(SystemMetrics(
                metric_type=SystemMetricType.DISK_USAGE,
                value=(disk.used / disk.total) * 100,
                timestamp=current_time,
                metadata={
                    'total_disk': disk.total,
                    'used_disk': disk.used,
                    'free_disk': disk.free
                }
            ))
            
            # Network I/O
            network = psutil.net_io_counters()
            metrics.append(SystemMetrics(
                metric_type=SystemMetricType.NETWORK_IO,
                value=network.bytes_sent + network.bytes_recv,
                timestamp=current_time,
                metadata={
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            ))
            
            # Process count
            process_count = len(psutil.pids())
            metrics.append(SystemMetrics(
                metric_type=SystemMetricType.PROCESS_COUNT,
                value=process_count,
                timestamp=current_time,
                metadata={}
            ))
            
            # Thread count
            thread_count = psutil.Process().num_threads()
            metrics.append(SystemMetrics(
                metric_type=SystemMetricType.THREAD_COUNT,
                value=thread_count,
                timestamp=current_time,
                metadata={}
            ))
            
            # File descriptors
            try:
                fd_count = psutil.Process().num_fds()
                metrics.append(SystemMetrics(
                    metric_type=SystemMetricType.FILE_DESCRIPTORS,
                    value=fd_count,
                    timestamp=current_time,
                    metadata={}
                ))
            except (AttributeError, OSError):
                # File descriptors not available on this system
                pass
            
            # System load
            try:
                load_avg = psutil.getloadavg()
                metrics.append(SystemMetrics(
                    metric_type=SystemMetricType.SYSTEM_LOAD,
                    value=load_avg[0],  # 1-minute load average
                    timestamp=current_time,
                    metadata={
                        'load_1min': load_avg[0],
                        'load_5min': load_avg[1],
                        'load_15min': load_avg[2]
                    }
                ))
            except (AttributeError, OSError):
                # Load average not available on this system
                pass
            
            # Store metrics
            self.metrics_history.extend(metrics)
            
            # Clean up old metrics
            self._cleanup_old_metrics()
            
            # Cache metrics
            for metric in metrics:
                cache_key = f"system_metric_{metric.timestamp.timestamp()}"
                self.cache.set(cache_key, metric, ttl=3600)
            
            self.logger.debug(f"Collected {len(metrics)} system metrics")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return []
    
    def get_metrics(self, metric_type: Optional[SystemMetricType] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[SystemMetrics]:
        """Get system metrics with optional filtering."""
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
            self.logger.error(f"Error getting system metrics: {e}")
            return []
    
    def analyze_system_health(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> SystemReport:
        """Analyze system health and generate a report."""
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
                return self._create_empty_system_report(start, end)
            
            # Analyze system health
            summary = self._analyze_system_summary(metrics)
            health_score = self._calculate_health_score(metrics)
            recommendations = self._generate_system_recommendations(metrics, health_score)
            
            # Create report
            report = SystemReport(
                report_id=f"system_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                generated_at=datetime.now(),
                time_range=(start, end),
                summary=summary,
                metrics=metrics,
                health_score=health_score,
                recommendations=recommendations
            )
            
            # Store report
            self.reports.append(report)
            self.health_scores.append(health_score)
            
            # Update performance metrics
            analysis_time = (datetime.now() - start_time).total_seconds()
            self.analysis_times.append(analysis_time)
            
            # Cache report
            cache_key = f"system_report_{report.report_id}"
            self.cache.set(cache_key, report, ttl=86400)  # 24 hours
            
            self.logger.info(f"Generated system health report in {analysis_time:.3f}s")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error analyzing system health: {e}")
            raise
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get a summary of current system metrics."""
        try:
            # Collect current metrics
            current_metrics = self.collect_system_metrics()
            
            if not current_metrics:
                return {'error': 'No system metrics available'}
            
            # Calculate summary
            summary = {}
            for metric in current_metrics:
                metric_type = metric.metric_type.value
                summary[metric_type] = {
                    'value': metric.value,
                    'timestamp': metric.timestamp,
                    'metadata': metric.metadata
                }
            
            # Calculate overall health score
            health_score = self._calculate_health_score(current_metrics)
            summary['health_score'] = health_score
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting system summary: {e}")
            return {'error': str(e)}
    
    def get_health_trends(self) -> Dict[str, Any]:
        """Get health score trends over time."""
        try:
            if not self.health_scores:
                return {'error': 'No health scores available'}
            
            # Calculate trend statistics
            recent_scores = self.health_scores[-10:] if len(self.health_scores) >= 10 else self.health_scores
            
            return {
                'current_health': self.health_scores[-1] if self.health_scores else 0.0,
                'average_health': np.mean(self.health_scores),
                'min_health': np.min(self.health_scores),
                'max_health': np.max(self.health_scores),
                'recent_average': np.mean(recent_scores),
                'trend_direction': self._calculate_trend_direction(self.health_scores),
                'total_measurements': len(self.health_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting health trends: {e}")
            return {'error': str(e)}
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system analytics statistics."""
        try:
            return {
                'total_metrics': len(self.metrics_history),
                'total_reports': len(self.reports),
                'health_scores_count': len(self.health_scores),
                'average_analysis_time': np.mean(self.analysis_times) if self.analysis_times else 0.0,
                'average_report_generation_time': np.mean(self.report_generation_times) if self.report_generation_times else 0.0,
                'retention_days': self.retention_days,
                'analysis_window_hours': self.analysis_window_hours,
                'health_thresholds': self.health_thresholds
            }
        except Exception as e:
            self.logger.error(f"Error getting system statistics: {e}")
            return {'error': str(e)}
    
    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.retention_days)
            original_count = len(self.metrics_history)
            self.metrics_history = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
            removed_count = original_count - len(self.metrics_history)
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old system metrics")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old system metrics: {e}")
    
    def _create_empty_system_report(self, start: datetime, end: datetime) -> SystemReport:
        """Create an empty report when no metrics are available."""
        return SystemReport(
            report_id=f"system_empty_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(),
            time_range=(start, end),
            summary={'error': 'No system metrics available for analysis'},
            metrics=[],
            health_score=0.0,
            recommendations=['Collect more system metrics']
        )
    
    def _analyze_system_summary(self, metrics: List[SystemMetrics]) -> Dict[str, Any]:
        """Analyze system metrics and create summary statistics."""
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
                    'percentile_99': np.percentile(values, 99),
                    'threshold_exceeded': self._check_threshold_exceeded(metric_type, values)
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error analyzing system summary: {e}")
            return {}
    
    def _calculate_health_score(self, metrics: List[SystemMetrics]) -> float:
        """Calculate overall system health score (0-100)."""
        try:
            if not metrics:
                return 0.0
            
            # Group metrics by type
            metrics_by_type = {}
            for metric in metrics:
                metric_type = metric.metric_type.value
                if metric_type not in metrics_by_type:
                    metrics_by_type[metric_type] = []
                metrics_by_type[metric_type].append(metric.value)
            
            # Calculate health score for each metric type
            health_scores = []
            
            for metric_type, values in metrics_by_type.items():
                if not values:
                    continue
                
                # Get latest value
                latest_value = values[-1]
                
                # Calculate health score based on threshold
                if metric_type in self.health_thresholds:
                    threshold = self.health_thresholds[metric_type]
                    if latest_value <= threshold:
                        # Good health
                        health_score = 100.0 - (latest_value / threshold) * 50.0
                    else:
                        # Poor health
                        health_score = max(0.0, 50.0 - ((latest_value - threshold) / threshold) * 50.0)
                else:
                    # No threshold defined, assume good health
                    health_score = 100.0
                
                health_scores.append(health_score)
            
            # Return average health score
            return np.mean(health_scores) if health_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating health score: {e}")
            return 0.0
    
    def _check_threshold_exceeded(self, metric_type: str, values: List[float]) -> bool:
        """Check if metric values exceed thresholds."""
        try:
            if metric_type not in self.health_thresholds:
                return False
            
            threshold = self.health_thresholds[metric_type]
            return any(value > threshold for value in values)
            
        except Exception as e:
            self.logger.error(f"Error checking threshold exceeded: {e}")
            return False
    
    def _generate_system_recommendations(self, metrics: List[SystemMetrics], 
                                       health_score: float) -> List[str]:
        """Generate system recommendations based on analysis."""
        try:
            recommendations = []
            
            # Check overall health score
            if health_score < 50:
                recommendations.append("System health is poor. Investigate resource usage and consider system optimization.")
            elif health_score < 75:
                recommendations.append("System health is moderate. Monitor resource usage and consider preventive measures.")
            
            # Check individual metric thresholds
            metrics_by_type = {}
            for metric in metrics:
                metric_type = metric.metric_type.value
                if metric_type not in metrics_by_type:
                    metrics_by_type[metric_type] = []
                metrics_by_type[metric_type].append(metric.value)
            
            # CPU usage recommendations
            if 'cpu_usage' in metrics_by_type:
                cpu_values = metrics_by_type['cpu_usage']
                if any(v > self.health_thresholds['cpu_usage'] for v in cpu_values):
                    recommendations.append("CPU usage is high. Consider optimizing CPU-intensive operations or scaling horizontally.")
            
            # Memory usage recommendations
            if 'memory_usage' in metrics_by_type:
                memory_values = metrics_by_type['memory_usage']
                if any(v > self.health_thresholds['memory_usage'] for v in memory_values):
                    recommendations.append("Memory usage is high. Consider optimizing memory usage or increasing available memory.")
            
            # Disk usage recommendations
            if 'disk_usage' in metrics_by_type:
                disk_values = metrics_by_type['disk_usage']
                if any(v > self.health_thresholds['disk_usage'] for v in disk_values):
                    recommendations.append("Disk usage is high. Consider cleaning up disk space or increasing storage capacity.")
            
            # System load recommendations
            if 'system_load' in metrics_by_type:
                load_values = metrics_by_type['system_load']
                if any(v > self.health_thresholds['system_load'] for v in load_values):
                    recommendations.append("System load is high. Consider reducing system load or scaling resources.")
            
            # Process count recommendations
            if 'process_count' in metrics_by_type:
                process_values = metrics_by_type['process_count']
                if len(process_values) > 0:
                    avg_processes = np.mean(process_values)
                    if avg_processes > 1000:  # Arbitrary threshold
                        recommendations.append("High process count detected. Consider investigating for potential process leaks.")
            
            # Thread count recommendations
            if 'thread_count' in metrics_by_type:
                thread_values = metrics_by_type['thread_count']
                if len(thread_values) > 0:
                    avg_threads = np.mean(thread_values)
                    if avg_threads > 100:  # Arbitrary threshold
                        recommendations.append("High thread count detected. Consider investigating for potential thread leaks.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating system recommendations: {e}")
            return ['Error generating system recommendations']
    
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
