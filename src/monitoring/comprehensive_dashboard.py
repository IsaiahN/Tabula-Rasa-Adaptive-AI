#!/usr/bin/env python3
"""
Comprehensive Monitoring Dashboard

Integrates all monitoring and analytics features into a unified dashboard
providing real-time insights, alerts, and system health monitoring.
"""

import logging
import time
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum
import threading

from ..core.caching_system import UnifiedCachingSystem, CacheConfig
from .streaming_analytics import get_streaming_analytics, StreamType, AnalysisType
from .anomaly_detection import get_anomaly_detector, AnomalyType, AnomalySeverity
from .predictive_health import get_predictive_health, HealthMetric, HealthStatus
from .enhanced_alerting import get_alerting_system, AlertLevel, AlertState
from .performance_correlation import get_correlation_system, MetricType


class DashboardView(Enum):
    """Dashboard view types."""
    OVERVIEW = "overview"
    REAL_TIME = "real_time"
    ANALYTICS = "analytics"
    ALERTS = "alerts"
    HEALTH = "health"
    CORRELATIONS = "correlations"
    ANOMALIES = "anomalies"


@dataclass
class DashboardWidget:
    """A dashboard widget configuration."""
    widget_id: str
    widget_type: str
    title: str
    data_source: str
    refresh_interval: int  # seconds
    position: Tuple[int, int]  # x, y coordinates
    size: Tuple[int, int]  # width, height
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardData:
    """Dashboard data container."""
    view: DashboardView
    data: Dict[str, Any]
    timestamp: datetime
    refresh_interval: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealTimeWidget:
    """Real-time data widget."""
    
    def __init__(self, widget_id: str, title: str, data_source: str):
        self.widget_id = widget_id
        self.title = title
        self.data_source = data_source
        self.data = {}
        self.last_update = None
    
    def update_data(self, data: Dict[str, Any]) -> None:
        """Update widget data."""
        self.data = data
        self.last_update = datetime.now()
    
    def get_data(self) -> Dict[str, Any]:
        """Get current widget data."""
        return {
            'widget_id': self.widget_id,
            'title': self.title,
            'data': self.data,
            'last_update': self.last_update,
            'status': 'active' if self.last_update and (datetime.now() - self.last_update).seconds < 60 else 'stale'
        }


class AnalyticsWidget:
    """Analytics data widget."""
    
    def __init__(self, widget_id: str, title: str, analysis_type: str):
        self.widget_id = widget_id
        self.title = title
        self.analysis_type = analysis_type
        self.data = {}
        self.trends = []
    
    def update_data(self, data: Dict[str, Any], trends: List[Dict[str, Any]] = None) -> None:
        """Update widget data."""
        self.data = data
        if trends:
            self.trends = trends
    
    def get_data(self) -> Dict[str, Any]:
        """Get current widget data."""
        return {
            'widget_id': self.widget_id,
            'title': self.title,
            'analysis_type': self.analysis_type,
            'data': self.data,
            'trends': self.trends,
            'timestamp': datetime.now()
        }


class ComprehensiveMonitoringDashboard:
    """Main comprehensive monitoring dashboard."""
    
    def __init__(self, cache_config: Optional[CacheConfig] = None):
        self.cache_config = cache_config or CacheConfig()
        self.cache = UnifiedCachingSystem(self.cache_config)
        
        # Monitoring systems
        self.streaming_analytics = get_streaming_analytics()
        self.anomaly_detector = get_anomaly_detector()
        self.predictive_health = get_predictive_health()
        self.alerting_system = get_alerting_system()
        self.correlation_system = get_correlation_system()
        
        # Dashboard state
        self.widgets: Dict[str, DashboardWidget] = {}
        self.real_time_widgets: Dict[str, RealTimeWidget] = {}
        self.analytics_widgets: Dict[str, AnalyticsWidget] = {}
        self.dashboard_data: Dict[DashboardView, DashboardData] = {}
        
        # Update thread
        self.update_thread = None
        self.running = False
        self.update_interval = 5  # seconds
        
        # Callbacks
        self.dashboard_callbacks: List[Callable[[DashboardData], None]] = []
        
        self._initialized = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize the comprehensive dashboard."""
        try:
            # Set up default widgets
            self._setup_default_widgets()
            
            # Set up monitoring callbacks
            self._setup_monitoring_callbacks()
            
            # Start update thread
            self.running = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
            
            self._initialized = True
            self.logger.info("Comprehensive monitoring dashboard initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize dashboard: {e}")
            raise
    
    def _setup_default_widgets(self) -> None:
        """Set up default dashboard widgets."""
        # Real-time widgets
        self.add_real_time_widget("system_metrics", "System Metrics", "system_performance")
        self.add_real_time_widget("learning_progress", "Learning Progress", "learning_progress")
        self.add_real_time_widget("memory_usage", "Memory Usage", "memory_usage")
        
        # Analytics widgets
        self.add_analytics_widget("performance_trends", "Performance Trends", "trend_analysis")
        self.add_analytics_widget("anomaly_summary", "Anomaly Summary", "anomaly_detection")
        self.add_analytics_widget("health_predictions", "Health Predictions", "predictive_health")
        self.add_analytics_widget("correlation_insights", "Correlation Insights", "correlation_analysis")
    
    def _setup_monitoring_callbacks(self) -> None:
        """Set up callbacks for monitoring systems."""
        # Streaming analytics callback
        def on_stream_analysis(analysis):
            self._handle_stream_analysis(analysis)
        
        self.streaming_analytics.add_analysis_callback(on_stream_analysis)
        
        # Anomaly detection callback
        def on_anomaly(anomaly):
            self._handle_anomaly(anomaly)
        
        self.anomaly_detector.add_anomaly_callback(on_anomaly)
        
        # Predictive health callback
        def on_prediction(prediction):
            self._handle_prediction(prediction)
        
        self.predictive_health.add_prediction_callback(on_prediction)
    
    def add_real_time_widget(self, widget_id: str, title: str, data_source: str) -> None:
        """Add a real-time widget."""
        widget = RealTimeWidget(widget_id, title, data_source)
        self.real_time_widgets[widget_id] = widget
        self.logger.info(f"Added real-time widget: {title}")
    
    def add_analytics_widget(self, widget_id: str, title: str, analysis_type: str) -> None:
        """Add an analytics widget."""
        widget = AnalyticsWidget(widget_id, title, analysis_type)
        self.analytics_widgets[widget_id] = widget
        self.logger.info(f"Added analytics widget: {title}")
    
    def _update_loop(self) -> None:
        """Main update loop for dashboard data."""
        while self.running:
            try:
                # Update all dashboard views
                self._update_overview_view()
                self._update_real_time_view()
                self._update_analytics_view()
                self._update_alerts_view()
                self._update_health_view()
                self._update_correlations_view()
                self._update_anomalies_view()
                
                # Update widgets
                self._update_widgets()
                
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Error in dashboard update loop: {e}")
                time.sleep(self.update_interval)
    
    def _update_overview_view(self) -> None:
        """Update overview dashboard view."""
        # Get system summary
        system_summary = self.streaming_analytics.get_system_summary()
        health_summary = self.predictive_health.get_health_summary()
        alert_summary = self.alerting_system.get_alert_summary()
        correlation_summary = self.correlation_system.get_correlation_summary()
        anomaly_summary = self.anomaly_detector.get_anomaly_summary()
        
        overview_data = {
            'system_status': 'healthy' if health_summary['current_health'] == HealthStatus.GOOD else 'warning',
            'active_alerts': alert_summary['active_alerts'],
            'total_streams': system_summary['total_streams'],
            'recent_anomalies': anomaly_summary['recent_anomalies'],
            'strong_correlations': correlation_summary['strong_correlations'],
            'health_score': self._calculate_overall_health_score(health_summary),
            'last_update': datetime.now()
        }
        
        self.dashboard_data[DashboardView.OVERVIEW] = DashboardData(
            view=DashboardView.OVERVIEW,
            data=overview_data,
            timestamp=datetime.now(),
            refresh_interval=5
        )
    
    def _update_real_time_view(self) -> None:
        """Update real-time dashboard view."""
        # Get real-time data from all streams
        real_time_data = {}
        
        for stream_id in self.streaming_analytics.streams.keys():
            stream_summary = self.streaming_analytics.get_stream_summary(stream_id)
            real_time_data[stream_id] = stream_summary
        
        self.dashboard_data[DashboardView.REAL_TIME] = DashboardData(
            view=DashboardView.REAL_TIME,
            data=real_time_data,
            timestamp=datetime.now(),
            refresh_interval=1
        )
    
    def _update_analytics_view(self) -> None:
        """Update analytics dashboard view."""
        # Get recent analyses
        recent_analyses = self.streaming_analytics.get_recent_analyses(50)
        
        # Group analyses by type
        analyses_by_type = defaultdict(list)
        for analysis in recent_analyses:
            analyses_by_type[analysis.analysis_type.value].append(analysis)
        
        analytics_data = {
            'recent_analyses': len(recent_analyses),
            'analyses_by_type': dict(analyses_by_type),
            'trend_analysis': self._analyze_trends(recent_analyses),
            'performance_metrics': self._get_performance_metrics()
        }
        
        self.dashboard_data[DashboardView.ANALYTICS] = DashboardData(
            view=DashboardView.ANALYTICS,
            data=analytics_data,
            timestamp=datetime.now(),
            refresh_interval=10
        )
    
    def _update_alerts_view(self) -> None:
        """Update alerts dashboard view."""
        # Get recent alerts
        recent_alerts = self.alerting_system.get_recent_alerts(50)
        
        # Group alerts by severity
        alerts_by_severity = defaultdict(list)
        for alert in recent_alerts:
            alerts_by_severity[alert.severity.value].append(alert)
        
        alerts_data = {
            'recent_alerts': len(recent_alerts),
            'alerts_by_severity': dict(alerts_by_severity),
            'alert_summary': self.alerting_system.get_alert_summary(),
            'active_alerts': [a for a in recent_alerts if a.state == AlertState.ACTIVE]
        }
        
        self.dashboard_data[DashboardView.ALERTS] = DashboardData(
            view=DashboardView.ALERTS,
            data=alerts_data,
            timestamp=datetime.now(),
            refresh_interval=5
        )
    
    def _update_health_view(self) -> None:
        """Update health dashboard view."""
        # Get health summary
        health_summary = self.predictive_health.get_health_summary()
        
        # Get recent predictions
        recent_predictions = self.predictive_health.get_recent_predictions(20)
        
        health_data = {
            'current_health': health_summary['current_health'],
            'recent_predictions': len(recent_predictions),
            'critical_predictions': health_summary['critical_predictions'],
            'warning_predictions': health_summary['warning_predictions'],
            'trend_summary': health_summary['trend_summary'],
            'predictors_status': health_summary['predictors_status']
        }
        
        self.dashboard_data[DashboardView.HEALTH] = DashboardData(
            view=DashboardView.HEALTH,
            data=health_data,
            timestamp=datetime.now(),
            refresh_interval=10
        )
    
    def _update_correlations_view(self) -> None:
        """Update correlations dashboard view."""
        # Get correlation summary
        correlation_summary = self.correlation_system.get_correlation_summary()
        
        # Get recent insights
        recent_insights = self.correlation_system.get_recent_insights(20)
        
        correlations_data = {
            'correlation_summary': correlation_summary,
            'recent_insights': recent_insights,
            'strong_correlations': correlation_summary['strong_correlations'],
            'metrics_analyzed': correlation_summary['metrics_analyzed']
        }
        
        self.dashboard_data[DashboardView.CORRELATIONS] = DashboardData(
            view=DashboardView.CORRELATIONS,
            data=correlations_data,
            timestamp=datetime.now(),
            refresh_interval=30
        )
    
    def _update_anomalies_view(self) -> None:
        """Update anomalies dashboard view."""
        # Get anomaly summary
        anomaly_summary = self.anomaly_detector.get_anomaly_summary()
        
        # Get recent anomalies
        recent_anomalies = self.anomaly_detector.get_recent_anomalies(50)
        
        anomalies_data = {
            'anomaly_summary': anomaly_summary,
            'recent_anomalies': recent_anomalies,
            'type_distribution': anomaly_summary['type_distribution'],
            'severity_distribution': anomaly_summary['severity_distribution']
        }
        
        self.dashboard_data[DashboardView.ANOMALIES] = DashboardData(
            view=DashboardView.ANOMALIES,
            data=anomalies_data,
            timestamp=datetime.now(),
            refresh_interval=10
        )
    
    def _update_widgets(self) -> None:
        """Update all dashboard widgets."""
        # Update real-time widgets
        for widget_id, widget in self.real_time_widgets.items():
            if widget.data_source == "system_performance":
                data = self._get_system_performance_data()
                widget.update_data(data)
            elif widget.data_source == "learning_progress":
                data = self._get_learning_progress_data()
                widget.update_data(data)
            elif widget.data_source == "memory_usage":
                data = self._get_memory_usage_data()
                widget.update_data(data)
        
        # Update analytics widgets
        for widget_id, widget in self.analytics_widgets.items():
            if widget.analysis_type == "trend_analysis":
                data, trends = self._get_trend_analysis_data()
                widget.update_data(data, trends)
            elif widget.analysis_type == "anomaly_detection":
                data = self._get_anomaly_detection_data()
                widget.update_data(data)
            elif widget.analysis_type == "predictive_health":
                data = self._get_predictive_health_data()
                widget.update_data(data)
            elif widget.analysis_type == "correlation_analysis":
                data = self._get_correlation_analysis_data()
                widget.update_data(data)
    
    def _get_system_performance_data(self) -> Dict[str, Any]:
        """Get system performance data for widgets."""
        return {
            'cpu_usage': 0.0,  # Would get from actual system metrics
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'network_io': 0.0,
            'timestamp': datetime.now()
        }
    
    def _get_learning_progress_data(self) -> Dict[str, Any]:
        """Get learning progress data for widgets."""
        return {
            'current_score': 0.0,  # Would get from actual learning metrics
            'learning_rate': 0.0,
            'efficiency': 0.0,
            'progress_trend': 'stable',
            'timestamp': datetime.now()
        }
    
    def _get_memory_usage_data(self) -> Dict[str, Any]:
        """Get memory usage data for widgets."""
        return {
            'total_memory': 0,  # Would get from actual system metrics
            'used_memory': 0,
            'free_memory': 0,
            'memory_percentage': 0.0,
            'timestamp': datetime.now()
        }
    
    def _get_trend_analysis_data(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Get trend analysis data for widgets."""
        data = {
            'overall_trend': 'stable',
            'trend_strength': 0.5,
            'confidence': 0.8
        }
        
        trends = [
            {'metric': 'performance', 'trend': 'improving', 'strength': 0.7},
            {'metric': 'memory', 'trend': 'stable', 'strength': 0.3}
        ]
        
        return data, trends
    
    def _get_anomaly_detection_data(self) -> Dict[str, Any]:
        """Get anomaly detection data for widgets."""
        summary = self.anomaly_detector.get_anomaly_summary()
        return {
            'total_anomalies': summary['total_anomalies'],
            'recent_anomalies': summary['recent_anomalies'],
            'type_distribution': summary['type_distribution'],
            'severity_distribution': summary['severity_distribution']
        }
    
    def _get_predictive_health_data(self) -> Dict[str, Any]:
        """Get predictive health data for widgets."""
        summary = self.predictive_health.get_health_summary()
        return {
            'current_health': summary['current_health'],
            'recent_predictions': summary['recent_predictions'],
            'critical_predictions': summary['critical_predictions'],
            'warning_predictions': summary['warning_predictions']
        }
    
    def _get_correlation_analysis_data(self) -> Dict[str, Any]:
        """Get correlation analysis data for widgets."""
        summary = self.correlation_system.get_correlation_summary()
        return {
            'total_correlations': summary['total_correlations'],
            'strong_correlations': summary['strong_correlations'],
            'recent_insights': summary['recent_insights'],
            'metrics_analyzed': summary['metrics_analyzed']
        }
    
    def _analyze_trends(self, analyses: List[Any]) -> Dict[str, Any]:
        """Analyze trends from recent analyses."""
        if not analyses:
            return {}
        
        # Simple trend analysis
        trend_counts = defaultdict(int)
        for analysis in analyses:
            if hasattr(analysis, 'analysis_type'):
                trend_counts[analysis.analysis_type.value] += 1
        
        return {
            'trend_distribution': dict(trend_counts),
            'total_analyses': len(analyses)
        }
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            'response_time': 0.0,  # Would get from actual metrics
            'throughput': 0.0,
            'error_rate': 0.0,
            'availability': 1.0
        }
    
    def _calculate_overall_health_score(self, health_summary: Dict[str, Any]) -> float:
        """Calculate overall health score."""
        # Simple health score calculation
        current_health = health_summary.get('current_health', HealthStatus.GOOD)
        
        health_scores = {
            HealthStatus.EXCELLENT: 1.0,
            HealthStatus.GOOD: 0.8,
            HealthStatus.WARNING: 0.6,
            HealthStatus.CRITICAL: 0.3,
            HealthStatus.FAILED: 0.0
        }
        
        return health_scores.get(current_health, 0.5)
    
    def _handle_stream_analysis(self, analysis) -> None:
        """Handle stream analysis results."""
        # Update relevant widgets
        pass
    
    def _handle_anomaly(self, anomaly) -> None:
        """Handle anomaly detection results."""
        # Update anomaly widgets
        pass
    
    def _handle_prediction(self, prediction) -> None:
        """Handle health prediction results."""
        # Update health widgets
        pass
    
    def get_dashboard_data(self, view: DashboardView) -> Optional[DashboardData]:
        """Get dashboard data for a specific view."""
        return self.dashboard_data.get(view)
    
    def get_widget_data(self, widget_id: str) -> Optional[Dict[str, Any]]:
        """Get data for a specific widget."""
        if widget_id in self.real_time_widgets:
            return self.real_time_widgets[widget_id].get_data()
        elif widget_id in self.analytics_widgets:
            return self.analytics_widgets[widget_id].get_data()
        return None
    
    def add_dashboard_callback(self, callback: Callable[[DashboardData], None]) -> None:
        """Add a callback for dashboard updates."""
        self.dashboard_callbacks.append(callback)
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get overall dashboard summary."""
        return {
            'total_views': len(self.dashboard_data),
            'total_widgets': len(self.real_time_widgets) + len(self.analytics_widgets),
            'last_update': max([data.timestamp for data in self.dashboard_data.values()]) if self.dashboard_data else None,
            'system_status': 'running' if self.running else 'stopped',
            'views': [view.value for view in self.dashboard_data.keys()]
        }
    
    def shutdown(self) -> None:
        """Shutdown the dashboard."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        self.logger.info("Dashboard shutdown complete")


# Global instance
_dashboard = None

def get_dashboard() -> ComprehensiveMonitoringDashboard:
    """Get the global dashboard instance."""
    global _dashboard
    if _dashboard is None:
        _dashboard = ComprehensiveMonitoringDashboard()
        _dashboard.initialize()
    return _dashboard
