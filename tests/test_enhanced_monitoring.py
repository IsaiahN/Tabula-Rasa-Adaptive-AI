#!/usr/bin/env python3
"""
Test Enhanced Monitoring System

Tests all the new monitoring and analytics features including streaming analytics,
anomaly detection, predictive health monitoring, enhanced alerting, and correlation analysis.
"""

import unittest
import time
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.streaming_analytics import get_streaming_analytics, StreamType, AnalysisType
from src.monitoring.anomaly_detection import get_anomaly_detector, AnomalyType, AnomalySeverity
from src.monitoring.predictive_health import get_predictive_health, HealthMetric, HealthStatus
from src.monitoring.enhanced_alerting import get_alerting_system, AlertLevel, AlertState
from src.monitoring.performance_correlation import get_correlation_system, MetricType
from src.monitoring.comprehensive_dashboard import get_dashboard, DashboardView


class TestStreamingAnalytics(unittest.TestCase):
    """Test streaming analytics functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.streaming_analytics = get_streaming_analytics()
    
    def test_create_stream(self):
        """Test creating a new data stream."""
        stream_id = "test_stream"
        self.streaming_analytics.create_stream(
            stream_id=stream_id,
            stream_type=StreamType.PERFORMANCE,
            window_size=100,
            analysis_interval=1.0
        )
        
        self.assertIn(stream_id, self.streaming_analytics.streams)
        self.assertEqual(self.streaming_analytics.streams[stream_id].stream_id, stream_id)
    
    def test_add_data_point(self):
        """Test adding data points to a stream."""
        stream_id = "test_stream"
        self.streaming_analytics.create_stream(stream_id, StreamType.PERFORMANCE)
        
        # Add some data points
        analyses = []
        import time
        for i in range(10):
            analysis = self.streaming_analytics.add_data_point(
                stream_id=stream_id,
                value=50.0 + i * 2.0,
                metadata={'iteration': i}
            )
            analyses.extend(analysis)
        
        # Wait for analysis interval to trigger
        time.sleep(1.5)
        
        # Add one more data point to trigger analysis
        analysis = self.streaming_analytics.add_data_point(
            stream_id=stream_id,
            value=70.0,
            metadata={'iteration': 10}
        )
        analyses.extend(analysis)
        
        # Should have some analyses after enough data
        self.assertGreater(len(analyses), 0)
        
        # Check stream summary
        summary = self.streaming_analytics.get_stream_summary(stream_id)
        self.assertEqual(summary['stream_id'], stream_id)
        self.assertGreater(summary['data_points'], 0)
    
    def test_analysis_callbacks(self):
        """Test analysis callbacks."""
        callback_results = []
        
        def analysis_callback(analysis):
            callback_results.append(analysis)
        
        self.streaming_analytics.add_analysis_callback(analysis_callback)
        
        # Add data to trigger analysis
        stream_id = "callback_test"
        self.streaming_analytics.create_stream(stream_id, StreamType.PERFORMANCE)
        
        for i in range(20):
            self.streaming_analytics.add_data_point(stream_id, 50.0 + i)
        
        # Wait for analysis interval to trigger
        time.sleep(1.5)
        
        # Add one more data point to trigger analysis
        self.streaming_analytics.add_data_point(stream_id, 70.0)

        # Should have triggered callbacks
        self.assertGreater(len(callback_results), 0)


class TestAnomalyDetection(unittest.TestCase):
    """Test anomaly detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.anomaly_detector = get_anomaly_detector()
    
    def test_add_data_point(self):
        """Test adding data points for anomaly detection."""
        # Add normal data
        for i in range(20):
            self.anomaly_detector.add_data_point({
                'cpu_usage': 0.5 + (i % 10) * 0.02,
                'memory_usage': 0.6 + (i % 5) * 0.01
            })
        
        # Add anomalous data
        anomalies = self.anomaly_detector.add_data_point({
            'cpu_usage': 0.95,  # High CPU
            'memory_usage': 0.98  # High memory
        })
        
        # Should detect anomalies
        self.assertGreater(len(anomalies), 0)
        
        for anomaly in anomalies:
            self.assertIn(anomaly.anomaly_type, [AnomalyType.STATISTICAL, AnomalyType.PATTERN_BASED, AnomalyType.TEMPORAL])
            self.assertIn(anomaly.severity, [AnomalySeverity.LOW, AnomalySeverity.MEDIUM, 
                                           AnomalySeverity.HIGH, AnomalySeverity.CRITICAL])
    
    def test_anomaly_callbacks(self):
        """Test anomaly detection callbacks."""
        callback_results = []
        
        def anomaly_callback(anomaly):
            callback_results.append(anomaly)
        
        self.anomaly_detector.add_anomaly_callback(anomaly_callback)
        
        # Add anomalous data
        self.anomaly_detector.add_data_point({
            'cpu_usage': 0.99,  # Very high CPU
            'memory_usage': 0.99  # Very high memory
        })
        
        # Should have triggered callbacks
        self.assertGreater(len(callback_results), 0)
    
    def test_anomaly_summary(self):
        """Test anomaly summary generation."""
        # Add some data with anomalies
        for i in range(50):
            cpu_usage = 0.5 if i < 40 else 0.95  # Anomaly in last 10 points
            self.anomaly_detector.add_data_point({
                'cpu_usage': cpu_usage,
                'memory_usage': 0.6
            })
        
        summary = self.anomaly_detector.get_anomaly_summary()
        
        self.assertIn('total_anomalies', summary)
        self.assertIn('recent_anomalies', summary)
        self.assertIn('type_distribution', summary)
        self.assertIn('severity_distribution', summary)


class TestPredictiveHealthMonitoring(unittest.TestCase):
    """Test predictive health monitoring functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictive_health = get_predictive_health()
    
    def test_add_health_data(self):
        """Test adding health data."""
        # Add some health data
        for i in range(30):
            timestamp = datetime.now() - timedelta(hours=30-i)
            self.predictive_health.add_health_data(
                HealthMetric.SYSTEM_PERFORMANCE,
                0.7 + i * 0.01,  # Improving performance
                timestamp
            )
        
        # Should have data in the predictor
        predictor = self.predictive_health.predictors[HealthMetric.SYSTEM_PERFORMANCE]
        self.assertGreater(len(predictor.data_points), 0)
    
    def test_health_prediction(self):
        """Test health prediction generation."""
        # Add historical data (more data points to ensure model training)
        for i in range(100):
            timestamp = datetime.now() - timedelta(hours=100-i)
            self.predictive_health.add_health_data(
                HealthMetric.MEMORY_USAGE,
                0.5 + i * 0.005,  # Gradually increasing memory usage
                timestamp
            )

        # Wait a bit for model training
        time.sleep(0.5)

        # Get predictions
        predictions = self.predictive_health.get_metric_predictions(
            HealthMetric.MEMORY_USAGE, hours_ahead=24
        )

        # Should have some predictions
        self.assertGreater(len(predictions), 0)
    
    def test_health_summary(self):
        """Test health summary generation."""
        # Add some health data
        for i in range(20):
            self.predictive_health.add_health_data(
                HealthMetric.SYSTEM_PERFORMANCE,
                0.8,
                datetime.now() - timedelta(hours=20-i)
            )
        
        summary = self.predictive_health.get_health_summary()
        
        self.assertIn('current_health', summary)
        self.assertIn('recent_predictions', summary)
        self.assertIn('trend_summary', summary)


class TestEnhancedAlerting(unittest.TestCase):
    """Test enhanced alerting functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.alerting_system = get_alerting_system()
    
    def test_process_data(self):
        """Test processing data for alerts."""
        # Process normal data
        alerts = self.alerting_system.process_data({
            'cpu_usage': 0.5,
            'memory_usage': 0.6,
            'error_rate': 0.01
        }, source="test_system")
        
        # Should not generate alerts for normal data
        self.assertEqual(len(alerts), 0)
        
        # Process high CPU data
        alerts = self.alerting_system.process_data({
            'cpu_usage': 0.95,  # High CPU
            'memory_usage': 0.6,
            'error_rate': 0.01
        }, source="test_system")
        
        # Should generate alerts for high CPU
        self.assertGreater(len(alerts), 0)
        
        for alert in alerts:
            self.assertEqual(alert.source, "test_system")
            self.assertIn(alert.severity, [AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL])
    
    def test_alert_acknowledgment(self):
        """Test alert acknowledgment."""
        # Generate an alert
        alerts = self.alerting_system.process_data({
            'cpu_usage': 0.95,
            'memory_usage': 0.6,
            'error_rate': 0.01
        }, source="test_system")
        
        if alerts:
            alert = alerts[0]
            
            # Acknowledge the alert
            success = self.alerting_system.acknowledge_alert(alert.alert_id, "test_user")
            self.assertTrue(success)
            
            # Check alert state
            self.assertEqual(alert.state, AlertState.ACKNOWLEDGED)
            self.assertEqual(alert.acknowledged_by, "test_user")
    
    def test_alert_resolution(self):
        """Test alert resolution."""
        # Generate an alert
        alerts = self.alerting_system.process_data({
            'cpu_usage': 0.95,
            'memory_usage': 0.6,
            'error_rate': 0.01
        }, source="test_system")
        
        if alerts:
            alert = alerts[0]
            
            # Resolve the alert
            success = self.alerting_system.resolve_alert(alert.alert_id)
            self.assertTrue(success)
            
            # Check alert state
            self.assertEqual(alert.state, AlertState.RESOLVED)
    
    def test_alert_summary(self):
        """Test alert summary generation."""
        # Generate some alerts
        for i in range(5):
            self.alerting_system.process_data({
                'cpu_usage': 0.9 + i * 0.01,
                'memory_usage': 0.6,
                'error_rate': 0.01
            }, source=f"test_system_{i}")
        
        summary = self.alerting_system.get_alert_summary()
        
        self.assertIn('stats', summary)
        self.assertIn('active_alerts', summary)
        self.assertIn('channels', summary)
        self.assertIn('rules', summary)


class TestPerformanceCorrelation(unittest.TestCase):
    """Test performance correlation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.correlation_system = get_correlation_system()
    
    def test_add_metric_data(self):
        """Test adding metric data for correlation analysis."""
        # Add correlated data
        for i in range(50):
            timestamp = datetime.now() - timedelta(hours=50-i)
            self.correlation_system.add_metric_data(
                "cpu_usage", MetricType.SYSTEM, 0.5 + i * 0.01, timestamp
            )
            self.correlation_system.add_metric_data(
                "memory_usage", MetricType.SYSTEM, 0.6 + i * 0.005, timestamp
            )
        
        # Should have data in the analyzer
        self.assertIn("cpu_usage", self.correlation_system.correlation_analyzer.metric_data)
        self.assertIn("memory_usage", self.correlation_system.correlation_analyzer.metric_data)
    
    def test_correlation_calculation(self):
        """Test correlation calculation."""
        # Add correlated data
        for i in range(100):
            timestamp = datetime.now() - timedelta(hours=100-i)
            self.correlation_system.add_metric_data(
                "metric1", MetricType.PERFORMANCE, i * 0.01, timestamp
            )
            self.correlation_system.add_metric_data(
                "metric2", MetricType.PERFORMANCE, i * 0.01 + 0.1, timestamp
            )
        
        # Calculate correlation
        correlation = self.correlation_system.correlation_analyzer.calculate_correlation(
            "metric1", "metric2", window_hours=24
        )
        
        if correlation:
            self.assertGreater(correlation.correlation_coefficient, 0.8)  # Should be highly correlated
            self.assertEqual(correlation.metric1, "metric1")
            self.assertEqual(correlation.metric2, "metric2")
    
    def test_correlation_summary(self):
        """Test correlation summary generation."""
        # Add some data
        for i in range(50):
            timestamp = datetime.now() - timedelta(hours=50-i)
            self.correlation_system.add_metric_data(
                "performance", MetricType.PERFORMANCE, 0.8, timestamp
            )
            self.correlation_system.add_metric_data(
                "efficiency", MetricType.LEARNING, 0.7, timestamp
            )
        
        summary = self.correlation_system.get_correlation_summary()
        
        self.assertIn('total_correlations', summary)
        self.assertIn('strong_correlations', summary)
        self.assertIn('metrics_analyzed', summary)


class TestComprehensiveDashboard(unittest.TestCase):
    """Test comprehensive dashboard functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dashboard = get_dashboard()
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization."""
        self.assertTrue(self.dashboard._initialized)
        self.assertGreater(len(self.dashboard.real_time_widgets), 0)
        self.assertGreater(len(self.dashboard.analytics_widgets), 0)
    
    def test_dashboard_views(self):
        """Test dashboard view data generation."""
        # Test overview view
        overview_data = self.dashboard.get_dashboard_data(DashboardView.OVERVIEW)
        self.assertIsNotNone(overview_data)
        self.assertEqual(overview_data.view, DashboardView.OVERVIEW)
        
        # Test real-time view
        real_time_data = self.dashboard.get_dashboard_data(DashboardView.REAL_TIME)
        self.assertIsNotNone(real_time_data)
        self.assertEqual(real_time_data.view, DashboardView.REAL_TIME)
        
        # Test analytics view
        analytics_data = self.dashboard.get_dashboard_data(DashboardView.ANALYTICS)
        self.assertIsNotNone(analytics_data)
        self.assertEqual(analytics_data.view, DashboardView.ANALYTICS)
    
    def test_widget_data(self):
        """Test widget data retrieval."""
        # Test real-time widget
        widget_data = self.dashboard.get_widget_data("system_metrics")
        self.assertIsNotNone(widget_data)
        self.assertIn('widget_id', widget_data)
        self.assertIn('title', widget_data)
        self.assertIn('data', widget_data)
        
        # Test analytics widget
        widget_data = self.dashboard.get_widget_data("performance_trends")
        self.assertIsNotNone(widget_data)
        self.assertIn('widget_id', widget_data)
        self.assertIn('analysis_type', widget_data)
    
    def test_dashboard_summary(self):
        """Test dashboard summary generation."""
        summary = self.dashboard.get_dashboard_summary()
        
        self.assertIn('total_views', summary)
        self.assertIn('total_widgets', summary)
        self.assertIn('system_status', summary)
        self.assertIn('views', summary)
        
        self.assertGreater(summary['total_widgets'], 0)
        self.assertEqual(summary['system_status'], 'running')


class TestMonitoringIntegration(unittest.TestCase):
    """Test integration between monitoring components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.streaming_analytics = get_streaming_analytics()
        self.anomaly_detector = get_anomaly_detector()
        self.predictive_health = get_predictive_health()
        self.alerting_system = get_alerting_system()
        self.correlation_system = get_correlation_system()
        self.dashboard = get_dashboard()
    
    def test_end_to_end_monitoring(self):
        """Test end-to-end monitoring workflow."""
        # Add data to streaming analytics
        self.streaming_analytics.add_data_point(
            "system_performance", 0.8, metadata={'test': True}
        )
        
        # Add data to anomaly detector
        self.anomaly_detector.add_data_point({
            'cpu_usage': 0.7,
            'memory_usage': 0.6
        })
        
        # Add data to predictive health
        self.predictive_health.add_health_data(
            HealthMetric.SYSTEM_PERFORMANCE, 0.8
        )
        
        # Add data to correlation system
        self.correlation_system.add_metric_data(
            "performance", MetricType.PERFORMANCE, 0.8
        )
        
        # Process data for alerts
        alerts = self.alerting_system.process_data({
            'cpu_usage': 0.7,
            'memory_usage': 0.6,
            'error_rate': 0.01
        })
        
        # Check that all systems are working
        self.assertIsNotNone(self.streaming_analytics.get_system_summary())
        self.assertIsNotNone(self.anomaly_detector.get_anomaly_summary())
        self.assertIsNotNone(self.predictive_health.get_health_summary())
        self.assertIsNotNone(self.correlation_system.get_correlation_summary())
        self.assertIsNotNone(self.alerting_system.get_alert_summary())
        
        # Check dashboard integration
        overview_data = self.dashboard.get_dashboard_data(DashboardView.OVERVIEW)
        self.assertIsNotNone(overview_data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
