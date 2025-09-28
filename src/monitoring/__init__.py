"""
Enhanced Monitoring Package

Comprehensive monitoring and analytics system with real-time streaming,
anomaly detection, predictive health monitoring, and advanced alerting.
"""

# Note: Legacy performance tracking imports removed - files don't exist
from ..core.unified_performance_monitor import UnifiedPerformanceMonitor, get_performance_monitor, monitor_performance

# New enhanced monitoring features
from .streaming_analytics import get_streaming_analytics, StreamingAnalytics, StreamType, AnalysisType
from .anomaly_detection import get_anomaly_detector, AdvancedAnomalyDetection, AnomalyType, AnomalySeverity
from .predictive_health import get_predictive_health, PredictiveHealthMonitoring, HealthMetric, HealthStatus
from .enhanced_alerting import get_alerting_system, EnhancedAlertingSystem, AlertLevel, AlertState
from .performance_correlation import get_correlation_system, PerformanceCorrelationSystem, MetricType
from .comprehensive_dashboard import get_dashboard, ComprehensiveMonitoringDashboard, DashboardView

__all__ = [
    # Original monitoring components
    'UnifiedPerformanceMonitor',
    'get_performance_monitor',
    'monitor_performance',
    
    # Enhanced monitoring features
    'get_streaming_analytics',
    'StreamingAnalytics',
    'StreamType',
    'AnalysisType',
    
    'get_anomaly_detector',
    'AdvancedAnomalyDetection',
    'AnomalyType',
    'AnomalySeverity',
    
    'get_predictive_health',
    'PredictiveHealthMonitoring',
    'HealthMetric',
    'HealthStatus',
    
    'get_alerting_system',
    'EnhancedAlertingSystem',
    'AlertLevel',
    'AlertState',
    
    'get_correlation_system',
    'PerformanceCorrelationSystem',
    'MetricType',
    
    'get_dashboard',
    'ComprehensiveMonitoringDashboard',
    'DashboardView'
]