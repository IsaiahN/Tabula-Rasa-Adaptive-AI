#!/usr/bin/env python3
"""
Real-time Streaming Analytics System

Provides real-time analysis of system metrics, learning progress, and performance data
with advanced pattern recognition and anomaly detection.
"""

import asyncio
import logging
import time
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from enum import Enum
import json

from ..core.caching_system import UnifiedCachingSystem, CacheConfig, CacheLevel
from ..core.unified_performance_monitor import UnifiedPerformanceMonitor, AlertLevel


class StreamType(Enum):
    """Types of data streams."""
    PERFORMANCE = "performance"
    LEARNING = "learning"
    SYSTEM = "system"
    CUSTOM = "custom"


class AnalysisType(Enum):
    """Types of real-time analysis."""
    TREND = "trend"
    ANOMALY = "anomaly"
    CORRELATION = "correlation"
    PREDICTION = "prediction"
    PATTERN = "pattern"


@dataclass
class StreamDataPoint:
    """A single data point in a stream."""
    stream_id: str
    stream_type: StreamType
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class StreamAnalysis:
    """Result of stream analysis."""
    stream_id: str
    analysis_type: AnalysisType
    result: Dict[str, Any]
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamAlert:
    """Alert generated from stream analysis."""
    alert_id: str
    stream_id: str
    alert_type: str
    severity: AlertLevel
    message: str
    timestamp: datetime
    data_point: Optional[StreamDataPoint] = None
    analysis_result: Optional[StreamAnalysis] = None


class StreamProcessor:
    """Processes individual data streams with various analysis techniques."""
    
    def __init__(self, stream_id: str, window_size: int = 100, 
                 analysis_interval: float = 1.0):
        self.stream_id = stream_id
        self.window_size = window_size
        self.analysis_interval = analysis_interval
        
        # Data storage
        self.data_buffer = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.values = deque(maxlen=window_size)
        
        # Analysis state
        self.last_analysis_time = 0
        self.trend_direction = 0  # -1, 0, 1
        self.volatility = 0.0
        self.mean_value = 0.0
        self.std_value = 0.0
        
        # Anomaly detection
        self.anomaly_threshold = 2.0  # Standard deviations
        self.recent_anomalies = deque(maxlen=10)
        
        # Pattern recognition
        self.patterns = []
        self.pattern_window = 20
        
    def add_data_point(self, data_point: StreamDataPoint) -> List[StreamAnalysis]:
        """Add a data point and return any new analyses."""
        self.data_buffer.append(data_point)
        self.timestamps.append(data_point.timestamp)
        self.values.append(data_point.value)
        
        # Update statistics
        self._update_statistics()
        
        # Perform analysis if interval has passed
        current_time = time.time()
        if current_time - self.last_analysis_time >= self.analysis_interval:
            self.last_analysis_time = current_time
            return self._perform_analysis()
        
        return []
    
    def _update_statistics(self):
        """Update running statistics."""
        if len(self.values) < 2:
            return
        
        self.mean_value = np.mean(self.values)
        self.std_value = np.std(self.values)
        
        # Calculate volatility (rolling standard deviation)
        if len(self.values) >= 10:
            recent_values = list(self.values)[-10:]
            self.volatility = np.std(recent_values)
        
        # Calculate trend direction
        if len(self.values) >= 5:
            recent_trend = np.polyfit(range(5), list(self.values)[-5:], 1)[0]
            self.trend_direction = 1 if recent_trend > 0.01 else (-1 if recent_trend < -0.01 else 0)
    
    def _perform_analysis(self) -> List[StreamAnalysis]:
        """Perform various analyses on the current data."""
        analyses = []
        
        if len(self.values) < 3:
            return analyses
        
        # Trend analysis
        trend_analysis = self._analyze_trend()
        if trend_analysis:
            analyses.append(trend_analysis)
        
        # Anomaly detection
        anomaly_analysis = self._detect_anomalies()
        if anomaly_analysis:
            analyses.append(anomaly_analysis)
        
        # Pattern recognition
        pattern_analysis = self._recognize_patterns()
        if pattern_analysis:
            analyses.append(pattern_analysis)
        
        return analyses
    
    def _analyze_trend(self) -> Optional[StreamAnalysis]:
        """Analyze trend in the data."""
        if len(self.values) < 5:
            return None
        
        # Linear regression for trend
        x = np.arange(len(self.values))
        y = np.array(self.values)
        
        # Calculate slope and R-squared
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        
        # Calculate R-squared
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Determine trend strength
        trend_strength = abs(slope) * r_squared
        
        return StreamAnalysis(
            stream_id=self.stream_id,
            analysis_type=AnalysisType.TREND,
            result={
                'slope': slope,
                'r_squared': r_squared,
                'trend_strength': trend_strength,
                'direction': self.trend_direction,
                'volatility': self.volatility
            },
            confidence=min(1.0, r_squared),
            timestamp=datetime.now()
        )
    
    def _detect_anomalies(self) -> Optional[StreamAnalysis]:
        """Detect anomalies in the data using statistical methods."""
        if len(self.values) < 10:
            return None
        
        current_value = self.values[-1]
        
        # Z-score based anomaly detection
        if self.std_value > 0:
            z_score = abs((current_value - self.mean_value) / self.std_value)
            
            if z_score > self.anomaly_threshold:
                # Record anomaly
                self.recent_anomalies.append({
                    'value': current_value,
                    'z_score': z_score,
                    'timestamp': self.timestamps[-1]
                })
                
                return StreamAnalysis(
                    stream_id=self.stream_id,
                    analysis_type=AnalysisType.ANOMALY,
                    result={
                        'z_score': z_score,
                        'anomaly_value': current_value,
                        'expected_range': (self.mean_value - 2*self.std_value, 
                                         self.mean_value + 2*self.std_value),
                        'severity': 'high' if z_score > 3 else 'medium'
                    },
                    confidence=min(1.0, z_score / 5.0),
                    timestamp=datetime.now()
                )
        
        return None
    
    def _recognize_patterns(self) -> Optional[StreamAnalysis]:
        """Recognize patterns in the data."""
        if len(self.values) < self.pattern_window:
            return None
        
        # Look for common patterns
        recent_values = list(self.values)[-self.pattern_window:]
        
        # Check for cyclical patterns
        cycle_analysis = self._detect_cycles(recent_values)
        
        # Check for step changes
        step_analysis = self._detect_step_changes(recent_values)
        
        if cycle_analysis or step_analysis:
            return StreamAnalysis(
                stream_id=self.stream_id,
                analysis_type=AnalysisType.PATTERN,
                result={
                    'cycles': cycle_analysis,
                    'step_changes': step_analysis,
                    'pattern_confidence': max(
                        cycle_analysis.get('confidence', 0) if cycle_analysis else 0,
                        step_analysis.get('confidence', 0) if step_analysis else 0
                    )
                },
                confidence=max(
                    cycle_analysis.get('confidence', 0) if cycle_analysis else 0,
                    step_analysis.get('confidence', 0) if step_analysis else 0
                ),
                timestamp=datetime.now()
            )
        
        return None
    
    def _detect_cycles(self, values: List[float]) -> Optional[Dict[str, Any]]:
        """Detect cyclical patterns in the data."""
        if len(values) < 10:
            return None
        
        # Simple cycle detection using autocorrelation
        values_array = np.array(values)
        
        # Calculate autocorrelation
        autocorr = np.correlate(values_array, values_array, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Normalize
        autocorr = autocorr / autocorr[0]
        
        # Find peaks (potential cycle periods)
        peaks = []
        for i in range(2, len(autocorr) - 1):
            if (autocorr[i] > autocorr[i-1] and 
                autocorr[i] > autocorr[i+1] and 
                autocorr[i] > 0.3):
                peaks.append(i)
        
        if peaks:
            # Find the strongest cycle
            strongest_cycle = max(peaks, key=lambda p: autocorr[p])
            confidence = autocorr[strongest_cycle]
            
            return {
                'cycle_period': strongest_cycle,
                'confidence': confidence,
                'all_peaks': peaks
            }
        
        return None
    
    def _detect_step_changes(self, values: List[float]) -> Optional[Dict[str, Any]]:
        """Detect step changes in the data."""
        if len(values) < 5:
            return None
        
        # Calculate differences
        diffs = np.diff(values)
        
        # Look for large changes
        threshold = 2 * np.std(diffs)
        large_changes = np.where(np.abs(diffs) > threshold)[0]
        
        if len(large_changes) > 0:
            # Find the most significant change
            max_change_idx = large_changes[np.argmax(np.abs(diffs[large_changes]))]
            change_magnitude = diffs[max_change_idx]
            
            return {
                'change_index': max_change_idx,
                'change_magnitude': change_magnitude,
                'confidence': min(1.0, abs(change_magnitude) / (3 * threshold)),
                'all_changes': large_changes.tolist()
            }
        
        return None


class StreamingAnalytics:
    """Main streaming analytics system."""
    
    def __init__(self, cache_config: Optional[CacheConfig] = None):
        self.cache_config = cache_config or CacheConfig()
        self.cache = UnifiedCachingSystem(self.cache_config)
        self.performance_monitor = PerformanceMonitor()
        
        # Stream management
        self.streams: Dict[str, StreamProcessor] = {}
        self.stream_configs: Dict[str, Dict[str, Any]] = {}
        
        # Analysis results
        self.analyses: List[StreamAnalysis] = []
        self.alerts: List[StreamAlert] = []
        
        # Callbacks
        self.analysis_callbacks: List[Callable[[StreamAnalysis], None]] = []
        self.alert_callbacks: List[Callable[[StreamAlert], None]] = []
        
        # Performance tracking
        self.processing_times: List[float] = []
        self.analysis_counts: Dict[AnalysisType, int] = defaultdict(int)
        
        self._initialized = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize the streaming analytics system."""
        try:
            self.performance_monitor.start_monitoring()
            
            # Set up default streams
            self._setup_default_streams()
            
            self._initialized = True
            self.logger.debug("Streaming analytics system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize streaming analytics: {e}")
            raise
    
    def _setup_default_streams(self):
        """Set up default monitoring streams."""
        default_streams = [
            {
                'id': 'system_performance',
                'type': StreamType.SYSTEM,
                'window_size': 200,
                'analysis_interval': 2.0
            },
            {
                'id': 'learning_progress',
                'type': StreamType.LEARNING,
                'window_size': 100,
                'analysis_interval': 5.0
            },
            {
                'id': 'memory_usage',
                'type': StreamType.SYSTEM,
                'window_size': 150,
                'analysis_interval': 1.0
            }
        ]
        
        for stream_config in default_streams:
            self.create_stream(
                stream_id=stream_config['id'],
                stream_type=stream_config['type'],
                window_size=stream_config['window_size'],
                analysis_interval=stream_config['analysis_interval']
            )
    
    def create_stream(self, stream_id: str, stream_type: StreamType,
                     window_size: int = 100, analysis_interval: float = 1.0,
                     config: Optional[Dict[str, Any]] = None) -> None:
        """Create a new data stream."""
        processor = StreamProcessor(stream_id, window_size, analysis_interval)
        self.streams[stream_id] = processor
        self.stream_configs[stream_id] = {
            'type': stream_type,
            'window_size': window_size,
            'analysis_interval': analysis_interval,
            'config': config or {}
        }
        
        self.logger.info(f"Created stream: {stream_id}")
    
    def add_data_point(self, stream_id: str, value: float, 
                      metadata: Optional[Dict[str, Any]] = None,
                      tags: Optional[Dict[str, str]] = None) -> List[StreamAnalysis]:
        """Add a data point to a stream."""
        if stream_id not in self.streams:
            self.logger.warning(f"Stream {stream_id} not found")
            return []
        
        # Create data point
        data_point = StreamDataPoint(
            stream_id=stream_id,
            stream_type=self.stream_configs[stream_id]['type'],
            value=value,
            timestamp=datetime.now(),
            metadata=metadata or {},
            tags=tags or {}
        )
        
        # Process the data point
        start_time = time.time()
        analyses = self.streams[stream_id].add_data_point(data_point)
        processing_time = time.time() - start_time
        
        self.processing_times.append(processing_time)
        
        # Store analyses
        for analysis in analyses:
            self.analyses.append(analysis)
            self.analysis_counts[analysis.analysis_type] += 1
            
            # Trigger callbacks
            for callback in self.analysis_callbacks:
                try:
                    callback(analysis)
                except Exception as e:
                    self.logger.error(f"Error in analysis callback: {e}")
            
            # Check for alerts
            self._check_analysis_for_alerts(analysis, data_point)
        
        return analyses
    
    def _check_analysis_for_alerts(self, analysis: StreamAnalysis, 
                                 data_point: StreamDataPoint) -> None:
        """Check if analysis results should trigger alerts."""
        alerts = []
        
        if analysis.analysis_type == AnalysisType.ANOMALY:
            # Anomaly alert
            severity = AlertLevel.CRITICAL if analysis.result.get('severity') == 'high' else AlertLevel.WARNING
            
            alert = StreamAlert(
                alert_id=f"anomaly_{analysis.stream_id}_{int(time.time())}",
                stream_id=analysis.stream_id,
                alert_type="anomaly_detected",
                severity=severity,
                message=f"Anomaly detected in {analysis.stream_id}: {analysis.result.get('anomaly_value', 'unknown')}",
                timestamp=datetime.now(),
                data_point=data_point,
                analysis_result=analysis
            )
            alerts.append(alert)
        
        elif analysis.analysis_type == AnalysisType.TREND:
            # Trend alert for significant changes
            trend_strength = analysis.result.get('trend_strength', 0)
            if trend_strength > 0.8:  # Strong trend
                direction = "increasing" if analysis.result.get('direction', 0) > 0 else "decreasing"
                
                alert = StreamAlert(
                    alert_id=f"trend_{analysis.stream_id}_{int(time.time())}",
                    stream_id=analysis.stream_id,
                    alert_type="strong_trend",
                    severity=AlertLevel.INFO,
                    message=f"Strong {direction} trend detected in {analysis.stream_id}",
                    timestamp=datetime.now(),
                    data_point=data_point,
                    analysis_result=analysis
                )
                alerts.append(alert)
        
        # Store and notify alerts
        for alert in alerts:
            self.alerts.append(alert)
            
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
    
    def add_analysis_callback(self, callback: Callable[[StreamAnalysis], None]) -> None:
        """Add a callback for analysis results."""
        self.analysis_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[StreamAlert], None]) -> None:
        """Add a callback for alerts."""
        self.alert_callbacks.append(callback)
    
    def get_stream_summary(self, stream_id: str) -> Dict[str, Any]:
        """Get summary statistics for a stream."""
        if stream_id not in self.streams:
            return {}
        
        processor = self.streams[stream_id]
        
        return {
            'stream_id': stream_id,
            'data_points': len(processor.data_buffer),
            'mean_value': processor.mean_value,
            'std_value': processor.std_value,
            'volatility': processor.volatility,
            'trend_direction': processor.trend_direction,
            'recent_anomalies': len(processor.recent_anomalies),
            'last_update': processor.timestamps[-1] if processor.timestamps else None
        }
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get overall system summary."""
        total_streams = len(self.streams)
        total_analyses = len(self.analyses)
        total_alerts = len(self.alerts)
        
        # Calculate average processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        # Analysis type distribution
        analysis_distribution = dict(self.analysis_counts)
        
        return {
            'total_streams': total_streams,
            'total_analyses': total_analyses,
            'total_alerts': total_alerts,
            'average_processing_time': avg_processing_time,
            'analysis_distribution': analysis_distribution,
            'streams': {stream_id: self.get_stream_summary(stream_id) 
                       for stream_id in self.streams.keys()}
        }
    
    def get_recent_analyses(self, limit: int = 50) -> List[StreamAnalysis]:
        """Get recent analyses."""
        return self.analyses[-limit:] if self.analyses else []
    
    def get_recent_alerts(self, limit: int = 20) -> List[StreamAlert]:
        """Get recent alerts."""
        return self.alerts[-limit:] if self.alerts else []
    
    def cleanup_old_data(self, max_age_hours: int = 24) -> None:
        """Clean up old data to prevent memory issues."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Clean up analyses
        self.analyses = [a for a in self.analyses if a.timestamp > cutoff_time]
        
        # Clean up alerts
        self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        # Clean up processing times (keep last 1000)
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-1000:]
        
        self.logger.info(f"Cleaned up data older than {max_age_hours} hours")


# Global instance
_streaming_analytics = None

def get_streaming_analytics() -> StreamingAnalytics:
    """Get the global streaming analytics instance."""
    global _streaming_analytics
    if _streaming_analytics is None:
        _streaming_analytics = StreamingAnalytics()
        _streaming_analytics.initialize()
    return _streaming_analytics
