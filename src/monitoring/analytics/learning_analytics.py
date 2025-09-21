"""
Learning Analytics

Advanced analytics for learning performance and progress tracking.
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


class LearningMetricType(Enum):
    """Types of learning metrics."""
    ACCURACY = "accuracy"
    LOSS = "loss"
    LEARNING_RATE = "learning_rate"
    CONVERGENCE = "convergence"
    GENERALIZATION = "generalization"
    TRANSFER = "transfer"
    META_LEARNING = "meta_learning"
    ADAPTATION = "adaptation"


@dataclass
class LearningMetrics:
    """Learning metrics data structure."""
    metric_type: LearningMetricType
    value: float
    timestamp: datetime
    session_id: str
    metadata: Dict[str, Any]


@dataclass
class LearningReport:
    """Learning analysis report."""
    report_id: str
    generated_at: datetime
    time_range: Tuple[datetime, datetime]
    session_id: str
    summary: Dict[str, Any]
    metrics: List[LearningMetrics]
    learning_curves: Dict[str, List[float]]
    recommendations: List[str]


class LearningAnalytics(ComponentInterface):
    """
    Advanced learning analytics system for monitoring and analyzing
    learning performance and progress.
    """
    
    def __init__(self, cache_config: Optional[CacheConfig] = None):
        """Initialize the learning analytics system."""
        self.cache_config = cache_config or CacheConfig()
        self.cache = CacheManager(self.cache_config)
        self.performance_monitor = PerformanceMonitor()
        
        # Analytics state
        self.metrics_history: List[LearningMetrics] = []
        self.reports: List[LearningReport] = []
        self.learning_curves: Dict[str, List[float]] = {}
        
        # Performance tracking
        self.analysis_times: List[float] = []
        self.report_generation_times: List[float] = []
        
        # Learning analytics configuration
        self.retention_days = 30
        self.analysis_window_hours = 24
        self.convergence_threshold = 0.001
        self.learning_rate_decay = 0.95
        
        self._initialized = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize the learning analytics system."""
        try:
            self.cache.initialize()
            self.performance_monitor.start_monitoring()
            
            self._initialized = True
            self.logger.info("Learning analytics system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize learning analytics: {e}")
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get current component state."""
        return {
            'name': 'LearningAnalytics',
            'status': 'running' if self._initialized else 'stopped',
            'last_updated': datetime.now(),
            'metadata': {
                'metrics_count': len(self.metrics_history),
                'reports_count': len(self.reports),
                'learning_curves_count': len(self.learning_curves),
                'average_analysis_time': np.mean(self.analysis_times) if self.analysis_times else 0.0
            }
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.performance_monitor.stop_monitoring()
            self.cache.clear()
            self._initialized = False
            self.logger.info("Learning analytics system cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self._initialized and self.cache.is_healthy()
    
    def add_metric(self, metric_type: LearningMetricType, value: float, 
                   session_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a learning metric."""
        try:
            metric = LearningMetrics(
                metric_type=metric_type,
                value=value,
                timestamp=datetime.now(),
                session_id=session_id,
                metadata=metadata or {}
            )
            
            self.metrics_history.append(metric)
            
            # Update learning curves
            self._update_learning_curves(metric)
            
            # Clean up old metrics
            self._cleanup_old_metrics()
            
            # Cache metric
            cache_key = f"learning_metric_{metric.timestamp.timestamp()}"
            self.cache.set(cache_key, metric, ttl=3600)
            
            self.logger.debug(f"Added {metric_type.value} metric for session {session_id}: {value}")
            
        except Exception as e:
            self.logger.error(f"Error adding learning metric: {e}")
    
    def get_metrics(self, session_id: Optional[str] = None,
                   metric_type: Optional[LearningMetricType] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[LearningMetrics]:
        """Get learning metrics with optional filtering."""
        try:
            metrics = self.metrics_history.copy()
            
            # Filter by session ID
            if session_id:
                metrics = [m for m in metrics if m.session_id == session_id]
            
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
            self.logger.error(f"Error getting learning metrics: {e}")
            return []
    
    def analyze_learning(self, session_id: str, 
                        time_range: Optional[Tuple[datetime, datetime]] = None) -> LearningReport:
        """Analyze learning performance and generate a report."""
        try:
            start_time = datetime.now()
            
            # Determine time range
            if time_range:
                start, end = time_range
            else:
                end = datetime.now()
                start = end - timedelta(hours=self.analysis_window_hours)
            
            # Get metrics for analysis
            metrics = self.get_metrics(session_id=session_id, start_time=start, end_time=end)
            
            if not metrics:
                return self._create_empty_learning_report(session_id, start, end)
            
            # Analyze learning performance
            summary = self._analyze_learning_summary(metrics)
            learning_curves = self._extract_learning_curves(metrics)
            recommendations = self._generate_learning_recommendations(metrics, learning_curves)
            
            # Create report
            report = LearningReport(
                report_id=f"learning_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                generated_at=datetime.now(),
                time_range=(start, end),
                session_id=session_id,
                summary=summary,
                metrics=metrics,
                learning_curves=learning_curves,
                recommendations=recommendations
            )
            
            # Store report
            self.reports.append(report)
            
            # Update performance metrics
            analysis_time = (datetime.now() - start_time).total_seconds()
            self.analysis_times.append(analysis_time)
            
            # Cache report
            cache_key = f"learning_report_{report.report_id}"
            self.cache.set(cache_key, report, ttl=86400)  # 24 hours
            
            self.logger.info(f"Generated learning report for session {session_id} in {analysis_time:.3f}s")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error analyzing learning: {e}")
            raise
    
    def get_learning_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of learning metrics for a session."""
        try:
            # Get metrics for the session
            metrics = self.get_metrics(session_id=session_id)
            
            if not metrics:
                return {'error': f'No metrics available for session {session_id}'}
            
            # Calculate summary statistics
            summary = {}
            
            for metric_type in LearningMetricType:
                type_metrics = [m for m in metrics if m.metric_type == metric_type]
                if type_metrics:
                    values = [m.value for m in type_metrics]
                    summary[metric_type.value] = {
                        'count': len(values),
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'latest': values[-1] if values else None,
                        'improvement': self._calculate_improvement(values)
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting learning summary: {e}")
            return {'error': str(e)}
    
    def get_learning_curves(self, session_id: str) -> Dict[str, List[float]]:
        """Get learning curves for a session."""
        try:
            curve_key = f"{session_id}_curves"
            if curve_key in self.learning_curves:
                return self.learning_curves[curve_key]
            
            # Extract learning curves from metrics
            metrics = self.get_metrics(session_id=session_id)
            return self._extract_learning_curves(metrics)
            
        except Exception as e:
            self.logger.error(f"Error getting learning curves: {e}")
            return {}
    
    def detect_convergence(self, session_id: str, metric_type: LearningMetricType) -> bool:
        """Detect if learning has converged for a specific metric."""
        try:
            metrics = self.get_metrics(session_id=session_id, metric_type=metric_type)
            
            if len(metrics) < 10:  # Need at least 10 data points
                return False
            
            values = [m.value for m in metrics]
            
            # Check if the last 10 values are within convergence threshold
            recent_values = values[-10:]
            if len(recent_values) < 10:
                return False
            
            # Calculate variance of recent values
            variance = np.var(recent_values)
            
            # Check if variance is below threshold
            return variance < self.convergence_threshold
            
        except Exception as e:
            self.logger.error(f"Error detecting convergence: {e}")
            return False
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning analytics system statistics."""
        try:
            return {
                'total_metrics': len(self.metrics_history),
                'total_reports': len(self.reports),
                'learning_curves_count': len(self.learning_curves),
                'average_analysis_time': np.mean(self.analysis_times) if self.analysis_times else 0.0,
                'average_report_generation_time': np.mean(self.report_generation_times) if self.report_generation_times else 0.0,
                'retention_days': self.retention_days,
                'analysis_window_hours': self.analysis_window_hours,
                'convergence_threshold': self.convergence_threshold
            }
        except Exception as e:
            self.logger.error(f"Error getting learning statistics: {e}")
            return {'error': str(e)}
    
    def _update_learning_curves(self, metric: LearningMetrics) -> None:
        """Update learning curves with new metric."""
        try:
            curve_key = f"{metric.session_id}_curves"
            if curve_key not in self.learning_curves:
                self.learning_curves[curve_key] = {}
            
            metric_type = metric.metric_type.value
            if metric_type not in self.learning_curves[curve_key]:
                self.learning_curves[curve_key][metric_type] = []
            
            self.learning_curves[curve_key][metric_type].append(metric.value)
            
        except Exception as e:
            self.logger.error(f"Error updating learning curves: {e}")
    
    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.retention_days)
            original_count = len(self.metrics_history)
            self.metrics_history = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
            removed_count = original_count - len(self.metrics_history)
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old learning metrics")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old learning metrics: {e}")
    
    def _create_empty_learning_report(self, session_id: str, start: datetime, end: datetime) -> LearningReport:
        """Create an empty report when no metrics are available."""
        return LearningReport(
            report_id=f"learning_empty_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(),
            time_range=(start, end),
            session_id=session_id,
            summary={'error': f'No learning metrics available for session {session_id}'},
            metrics=[],
            learning_curves={},
            recommendations=['Collect more learning metrics']
        )
    
    def _analyze_learning_summary(self, metrics: List[LearningMetrics]) -> Dict[str, Any]:
        """Analyze learning metrics and create summary statistics."""
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
                    'improvement': self._calculate_improvement(values),
                    'convergence': self._check_convergence(values)
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error analyzing learning summary: {e}")
            return {}
    
    def _extract_learning_curves(self, metrics: List[LearningMetrics]) -> Dict[str, List[float]]:
        """Extract learning curves from metrics."""
        try:
            curves = {}
            
            # Group metrics by type
            metrics_by_type = {}
            for metric in metrics:
                metric_type = metric.metric_type.value
                if metric_type not in metrics_by_type:
                    metrics_by_type[metric_type] = []
                metrics_by_type[metric_type].append(metric)
            
            # Sort by timestamp and extract values
            for metric_type, type_metrics in metrics_by_type.items():
                type_metrics.sort(key=lambda x: x.timestamp)
                values = [m.value for m in type_metrics]
                curves[metric_type] = values
            
            return curves
            
        except Exception as e:
            self.logger.error(f"Error extracting learning curves: {e}")
            return {}
    
    def _calculate_improvement(self, values: List[float]) -> float:
        """Calculate improvement percentage."""
        try:
            if len(values) < 2:
                return 0.0
            
            # Calculate improvement from first to last value
            first_value = values[0]
            last_value = values[-1]
            
            if first_value == 0:
                return 0.0
            
            improvement = ((last_value - first_value) / first_value) * 100
            return improvement
            
        except Exception as e:
            self.logger.error(f"Error calculating improvement: {e}")
            return 0.0
    
    def _check_convergence(self, values: List[float]) -> bool:
        """Check if values have converged."""
        try:
            if len(values) < 10:
                return False
            
            # Check if the last 10 values are within convergence threshold
            recent_values = values[-10:]
            variance = np.var(recent_values)
            
            return variance < self.convergence_threshold
            
        except Exception as e:
            self.logger.error(f"Error checking convergence: {e}")
            return False
    
    def _generate_learning_recommendations(self, metrics: List[LearningMetrics], 
                                         learning_curves: Dict[str, List[float]]) -> List[str]:
        """Generate learning recommendations based on analysis."""
        try:
            recommendations = []
            
            # Check for learning rate issues
            if 'learning_rate' in learning_curves:
                lr_values = learning_curves['learning_rate']
                if len(lr_values) > 1:
                    lr_trend = self._calculate_trend_direction(lr_values)
                    if lr_trend == 'decreasing' and lr_values[-1] < 0.001:
                        recommendations.append("Learning rate is very low. Consider increasing the learning rate or using adaptive learning rate scheduling.")
                    elif lr_trend == 'increasing' and lr_values[-1] > 0.1:
                        recommendations.append("Learning rate is very high. Consider decreasing the learning rate to prevent instability.")
            
            # Check for convergence issues
            if 'loss' in learning_curves:
                loss_values = learning_curves['loss']
                if len(loss_values) > 10:
                    convergence = self._check_convergence(loss_values)
                    if not convergence:
                        recommendations.append("Loss has not converged. Consider adjusting learning parameters or increasing training time.")
            
            # Check for overfitting
            if 'accuracy' in learning_curves and 'loss' in learning_curves:
                acc_values = learning_curves['accuracy']
                loss_values = learning_curves['loss']
                if len(acc_values) > 1 and len(loss_values) > 1:
                    acc_improvement = self._calculate_improvement(acc_values)
                    loss_improvement = self._calculate_improvement(loss_values)
                    
                    if acc_improvement > 0 and loss_improvement > 0:
                        recommendations.append("Both accuracy and loss are increasing. This may indicate overfitting. Consider regularization or early stopping.")
            
            # Check for learning stagnation
            if 'accuracy' in learning_curves:
                acc_values = learning_curves['accuracy']
                if len(acc_values) > 10:
                    recent_improvement = self._calculate_improvement(acc_values[-10:])
                    if abs(recent_improvement) < 1.0:  # Less than 1% improvement
                        recommendations.append("Learning appears to have stagnated. Consider changing the learning strategy or increasing model complexity.")
            
            # Check for high variance
            for metric_type, values in learning_curves.items():
                if len(values) > 5:
                    variance = np.var(values)
                    mean_val = np.mean(values)
                    if mean_val != 0 and variance / abs(mean_val) > 0.5:  # High coefficient of variation
                        recommendations.append(f"{metric_type} shows high variance. Consider stabilizing the learning process.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating learning recommendations: {e}")
            return ['Error generating learning recommendations']
    
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
