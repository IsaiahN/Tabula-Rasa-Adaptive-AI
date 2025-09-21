"""
Analytics Dashboard

Comprehensive analytics dashboard for visualizing system performance,
learning progress, and predictive insights.
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

from .performance_analytics import PerformanceAnalytics, MetricType
from .learning_analytics import LearningAnalytics, LearningMetricType
from .system_analytics import SystemAnalytics, SystemMetricType
from .predictive_analytics import PredictiveAnalytics, PredictionType


class DashboardView(Enum):
    """Available dashboard views."""
    OVERVIEW = "overview"
    PERFORMANCE = "performance"
    LEARNING = "learning"
    SYSTEM = "system"
    PREDICTIONS = "predictions"
    CUSTOM = "custom"


@dataclass
class DashboardConfig:
    """Configuration for the analytics dashboard."""
    refresh_interval_seconds: int = 30
    max_data_points: int = 1000
    enable_real_time: bool = True
    enable_predictions: bool = True
    enable_alerts: bool = True
    cache_ttl: int = 300  # 5 minutes


@dataclass
class DashboardData:
    """Dashboard data structure."""
    view: DashboardView
    data: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any]


class AnalyticsDashboard(ComponentInterface):
    """
    Comprehensive analytics dashboard for visualizing system performance,
    learning progress, and predictive insights.
    """
    
    def __init__(self, config: DashboardConfig, cache_config: Optional[CacheConfig] = None):
        """Initialize the analytics dashboard."""
        self.config = config
        self.cache_config = cache_config or CacheConfig()
        self.cache = CacheManager(self.cache_config)
        self.performance_monitor = PerformanceMonitor()
        
        # Analytics components
        self.performance_analytics = PerformanceAnalytics(cache_config)
        self.learning_analytics = LearningAnalytics(cache_config)
        self.system_analytics = SystemAnalytics(cache_config)
        self.predictive_analytics = PredictiveAnalytics(cache_config)
        
        # Dashboard state
        self.dashboard_data: Dict[DashboardView, DashboardData] = {}
        self.last_refresh: Dict[DashboardView, datetime] = {}
        self.alerts: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.refresh_times: List[float] = []
        self.data_generation_times: List[float] = []
        
        self._initialized = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize the analytics dashboard."""
        try:
            self.cache.initialize()
            self.performance_monitor.start_monitoring()
            
            # Initialize analytics components
            self.performance_analytics.initialize()
            self.learning_analytics.initialize()
            self.system_analytics.initialize()
            self.predictive_analytics.initialize()
            
            self._initialized = True
            self.logger.info("Analytics dashboard initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize analytics dashboard: {e}")
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get current component state."""
        return {
            'name': 'AnalyticsDashboard',
            'status': 'running' if self._initialized else 'stopped',
            'last_updated': datetime.now(),
            'metadata': {
                'dashboard_views': len(self.dashboard_data),
                'alerts_count': len(self.alerts),
                'average_refresh_time': np.mean(self.refresh_times) if self.refresh_times else 0.0,
                'performance_analytics_status': self.performance_analytics.get_state()['status'],
                'learning_analytics_status': self.learning_analytics.get_state()['status'],
                'system_analytics_status': self.system_analytics.get_state()['status'],
                'predictive_analytics_status': self.predictive_analytics.get_state()['status']
            }
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.performance_monitor.stop_monitoring()
            self.cache.clear()
            
            # Cleanup analytics components
            self.performance_analytics.cleanup()
            self.learning_analytics.cleanup()
            self.system_analytics.cleanup()
            self.predictive_analytics.cleanup()
            
            self._initialized = False
            self.logger.info("Analytics dashboard cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return (self._initialized and 
                self.cache.is_healthy() and
                self.performance_analytics.is_healthy() and
                self.learning_analytics.is_healthy() and
                self.system_analytics.is_healthy() and
                self.predictive_analytics.is_healthy())
    
    def refresh_dashboard(self, view: DashboardView) -> DashboardData:
        """Refresh dashboard data for a specific view."""
        try:
            start_time = datetime.now()
            
            # Check if refresh is needed
            if self._should_refresh(view):
                # Generate new data
                data = self._generate_dashboard_data(view)
                
                # Create dashboard data
                dashboard_data = DashboardData(
                    view=view,
                    data=data,
                    timestamp=datetime.now(),
                    metadata={
                        'refresh_time': (datetime.now() - start_time).total_seconds(),
                        'data_points': len(data.get('metrics', [])),
                        'generated_at': datetime.now()
                    }
                )
                
                # Store dashboard data
                self.dashboard_data[view] = dashboard_data
                self.last_refresh[view] = datetime.now()
                
                # Update performance metrics
                refresh_time = (datetime.now() - start_time).total_seconds()
                self.refresh_times.append(refresh_time)
                
                # Cache dashboard data
                cache_key = f"dashboard_{view.value}_{datetime.now().timestamp()}"
                self.cache.set(cache_key, dashboard_data, ttl=self.config.cache_ttl)
                
                self.logger.debug(f"Refreshed dashboard view {view.value} in {refresh_time:.3f}s")
                
                return dashboard_data
            else:
                # Return cached data
                return self.dashboard_data.get(view, self._generate_empty_dashboard_data(view))
                
        except Exception as e:
            self.logger.error(f"Error refreshing dashboard view {view.value}: {e}")
            return self._generate_empty_dashboard_data(view)
    
    def get_dashboard_data(self, view: DashboardView) -> DashboardData:
        """Get dashboard data for a specific view."""
        try:
            # Check if we have cached data
            if view in self.dashboard_data:
                return self.dashboard_data[view]
            
            # Generate new data
            return self.refresh_dashboard(view)
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard data for {view.value}: {e}")
            return self._generate_empty_dashboard_data(view)
    
    def get_overview_data(self) -> Dict[str, Any]:
        """Get overview data for the dashboard."""
        try:
            # Get data from all analytics components
            performance_summary = self.performance_analytics.get_performance_summary()
            learning_summary = self.learning_analytics.get_learning_summary('overview')
            system_summary = self.system_analytics.get_system_summary()
            health_trends = self.system_analytics.get_health_trends()
            
            # Get recent predictions
            recent_predictions = self.predictive_analytics.get_prediction_history(
                start_time=datetime.now() - timedelta(hours=1)
            )
            
            # Get alerts
            alerts = self._get_active_alerts()
            
            return {
                'performance': performance_summary,
                'learning': learning_summary,
                'system': system_summary,
                'health_trends': health_trends,
                'recent_predictions': len(recent_predictions),
                'alerts': alerts,
                'last_updated': datetime.now(),
                'dashboard_health': self.is_healthy()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting overview data: {e}")
            return {'error': str(e)}
    
    def get_performance_data(self) -> Dict[str, Any]:
        """Get performance data for the dashboard."""
        try:
            # Get performance metrics
            performance_metrics = self.performance_analytics.get_metrics()
            
            # Get performance summary
            performance_summary = self.performance_analytics.get_performance_summary()
            
            # Get performance trends
            performance_trends = self._analyze_performance_trends(performance_metrics)
            
            # Get performance recommendations
            performance_report = self.performance_analytics.analyze_performance()
            recommendations = performance_report.recommendations
            
            return {
                'metrics': performance_metrics,
                'summary': performance_summary,
                'trends': performance_trends,
                'recommendations': recommendations,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance data: {e}")
            return {'error': str(e)}
    
    def get_learning_data(self, session_id: str = 'overview') -> Dict[str, Any]:
        """Get learning data for the dashboard."""
        try:
            # Get learning metrics
            learning_metrics = self.learning_analytics.get_metrics(session_id=session_id)
            
            # Get learning summary
            learning_summary = self.learning_analytics.get_learning_summary(session_id)
            
            # Get learning curves
            learning_curves = self.learning_analytics.get_learning_curves(session_id)
            
            # Get learning recommendations
            learning_report = self.learning_analytics.analyze_learning(session_id)
            recommendations = learning_report.recommendations
            
            return {
                'metrics': learning_metrics,
                'summary': learning_summary,
                'curves': learning_curves,
                'recommendations': recommendations,
                'session_id': session_id,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting learning data: {e}")
            return {'error': str(e)}
    
    def get_system_data(self) -> Dict[str, Any]:
        """Get system data for the dashboard."""
        try:
            # Collect current system metrics
            current_metrics = self.system_analytics.collect_system_metrics()
            
            # Get system summary
            system_summary = self.system_analytics.get_system_summary()
            
            # Get health trends
            health_trends = self.system_analytics.get_health_trends()
            
            # Get system recommendations
            system_report = self.system_analytics.analyze_system_health()
            recommendations = system_report.recommendations
            
            return {
                'current_metrics': current_metrics,
                'summary': system_summary,
                'health_trends': health_trends,
                'recommendations': recommendations,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system data: {e}")
            return {'error': str(e)}
    
    def get_predictions_data(self) -> Dict[str, Any]:
        """Get predictions data for the dashboard."""
        try:
            # Get prediction history
            prediction_history = self.predictive_analytics.get_prediction_history()
            
            # Get recent predictions
            recent_predictions = self.predictive_analytics.get_prediction_history(
                start_time=datetime.now() - timedelta(hours=24)
            )
            
            # Get model information
            model_info = {}
            for model_key in self.predictive_analytics.models.keys():
                model_info[model_key] = self.predictive_analytics.get_model_info(
                    *model_key.split('_', 1)
                )
            
            # Get analytics statistics
            analytics_stats = self.predictive_analytics.get_analytics_statistics()
            
            return {
                'prediction_history': prediction_history,
                'recent_predictions': recent_predictions,
                'model_info': model_info,
                'analytics_stats': analytics_stats,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting predictions data: {e}")
            return {'error': str(e)}
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts."""
        return self._get_active_alerts()
    
    def get_dashboard_statistics(self) -> Dict[str, Any]:
        """Get dashboard statistics."""
        try:
            return {
                'total_views': len(self.dashboard_data),
                'total_alerts': len(self.alerts),
                'average_refresh_time': np.mean(self.refresh_times) if self.refresh_times else 0.0,
                'average_data_generation_time': np.mean(self.data_generation_times) if self.data_generation_times else 0.0,
                'refresh_interval_seconds': self.config.refresh_interval_seconds,
                'max_data_points': self.config.max_data_points,
                'real_time_enabled': self.config.enable_real_time,
                'predictions_enabled': self.config.enable_predictions,
                'alerts_enabled': self.config.enable_alerts
            }
        except Exception as e:
            self.logger.error(f"Error getting dashboard statistics: {e}")
            return {'error': str(e)}
    
    def _should_refresh(self, view: DashboardView) -> bool:
        """Check if dashboard view should be refreshed."""
        try:
            if view not in self.last_refresh:
                return True
            
            time_since_refresh = datetime.now() - self.last_refresh[view]
            return time_since_refresh.total_seconds() > self.config.refresh_interval_seconds
            
        except Exception as e:
            self.logger.error(f"Error checking refresh need: {e}")
            return True
    
    def _generate_dashboard_data(self, view: DashboardView) -> Dict[str, Any]:
        """Generate dashboard data for a specific view."""
        try:
            start_time = datetime.now()
            
            if view == DashboardView.OVERVIEW:
                data = self.get_overview_data()
            elif view == DashboardView.PERFORMANCE:
                data = self.get_performance_data()
            elif view == DashboardView.LEARNING:
                data = self.get_learning_data()
            elif view == DashboardView.SYSTEM:
                data = self.get_system_data()
            elif view == DashboardView.PREDICTIONS:
                data = self.get_predictions_data()
            else:
                data = {'error': f'Unknown dashboard view: {view.value}'}
            
            # Update performance metrics
            generation_time = (datetime.now() - start_time).total_seconds()
            self.data_generation_times.append(generation_time)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error generating dashboard data for {view.value}: {e}")
            return {'error': str(e)}
    
    def _generate_empty_dashboard_data(self, view: DashboardView) -> DashboardData:
        """Generate empty dashboard data."""
        return DashboardData(
            view=view,
            data={'error': 'No data available'},
            timestamp=datetime.now(),
            metadata={'empty': True}
        )
    
    def _analyze_performance_trends(self, metrics: List[Any]) -> Dict[str, Any]:
        """Analyze performance trends from metrics."""
        try:
            if not metrics:
                return {}
            
            # Group metrics by type
            metrics_by_type = {}
            for metric in metrics:
                metric_type = metric.metric_type.value
                if metric_type not in metrics_by_type:
                    metrics_by_type[metric_type] = []
                metrics_by_type[metric_type].append(metric.value)
            
            # Calculate trends for each type
            trends = {}
            for metric_type, values in metrics_by_type.items():
                if len(values) > 1:
                    # Calculate trend direction
                    x = np.arange(len(values))
                    y = np.array(values)
                    slope = np.polyfit(x, y, 1)[0]
                    
                    if slope > 0.01:
                        direction = 'increasing'
                    elif slope < -0.01:
                        direction = 'decreasing'
                    else:
                        direction = 'stable'
                    
                    trends[metric_type] = {
                        'direction': direction,
                        'slope': slope,
                        'data_points': len(values),
                        'latest_value': values[-1],
                        'average_value': np.mean(values)
                    }
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance trends: {e}")
            return {}
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts."""
        try:
            # This is a simplified alert system
            # In a real system, you would have more sophisticated alerting
            
            alerts = []
            
            # Check system health
            system_summary = self.system_analytics.get_system_summary()
            if 'health_score' in system_summary:
                health_score = system_summary['health_score']
                if health_score < 50:
                    alerts.append({
                        'type': 'system_health',
                        'severity': 'critical',
                        'message': f'System health is critical: {health_score:.1f}%',
                        'timestamp': datetime.now()
                    })
                elif health_score < 75:
                    alerts.append({
                        'type': 'system_health',
                        'severity': 'warning',
                        'message': f'System health is low: {health_score:.1f}%',
                        'timestamp': datetime.now()
                    })
            
            # Check performance metrics
            performance_summary = self.performance_analytics.get_performance_summary()
            for metric_type, data in performance_summary.items():
                if isinstance(data, dict) and 'latest' in data:
                    latest_value = data['latest']
                    if metric_type == 'cpu' and latest_value > 80:
                        alerts.append({
                            'type': 'performance',
                            'severity': 'warning',
                            'message': f'High CPU usage: {latest_value:.1f}%',
                            'timestamp': datetime.now()
                        })
                    elif metric_type == 'memory' and latest_value > 90:
                        alerts.append({
                            'type': 'performance',
                            'severity': 'critical',
                            'message': f'High memory usage: {latest_value:.1f}%',
                            'timestamp': datetime.now()
                        })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error getting active alerts: {e}")
            return []
