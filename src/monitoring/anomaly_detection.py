#!/usr/bin/env python3
"""
Advanced Anomaly Detection System

Provides sophisticated anomaly detection using multiple algorithms including
statistical methods, machine learning, and pattern-based detection.
"""

import logging
import numpy as np
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum
import json

from ..core.caching_system import UnifiedCachingSystem, CacheConfig
from ..core.unified_performance_monitor import AlertLevel


class AnomalyType(Enum):
    """Types of anomalies."""
    STATISTICAL = "statistical"
    PATTERN_BASED = "pattern_based"
    MACHINE_LEARNING = "machine_learning"
    TEMPORAL = "temporal"
    MULTIVARIATE = "multivariate"


class AnomalySeverity(Enum):
    """Anomaly severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection."""
    # Statistical detection
    z_score_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    
    # Pattern-based detection
    pattern_window_size: int = 50
    pattern_similarity_threshold: float = 0.8
    
    # Machine learning
    ml_model_type: str = "isolation_forest"
    ml_contamination: float = 0.1
    ml_training_samples: int = 1000
    
    # Temporal detection
    temporal_window_hours: int = 24
    temporal_seasonality_detection: bool = True
    
    # Multivariate detection
    correlation_threshold: float = 0.7
    multivariate_window_size: int = 100


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    score: float
    confidence: float
    timestamp: datetime
    data_point: Dict[str, Any]
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyPattern:
    """A detected anomaly pattern."""
    pattern_id: str
    pattern_type: str
    frequency: int
    first_detected: datetime
    last_detected: datetime
    examples: List[Dict[str, Any]] = field(default_factory=list)


class StatisticalAnomalyDetector:
    """Statistical methods for anomaly detection."""
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.data_history = deque(maxlen=1000)
        self.statistics = {}
        
    def add_data_point(self, data: Dict[str, float]) -> List[AnomalyResult]:
        """Add a data point and detect anomalies using statistical methods."""
        self.data_history.append({
            'data': data,
            'timestamp': datetime.now()
        })
        
        anomalies = []
        
        # Z-score based detection
        z_score_anomalies = self._detect_z_score_anomalies(data)
        anomalies.extend(z_score_anomalies)
        
        # IQR based detection
        iqr_anomalies = self._detect_iqr_anomalies(data)
        anomalies.extend(iqr_anomalies)
        
        # Modified Z-score detection
        modified_z_anomalies = self._detect_modified_z_score_anomalies(data)
        anomalies.extend(modified_z_anomalies)
        
        return anomalies
    
    def _detect_z_score_anomalies(self, data: Dict[str, float]) -> List[AnomalyResult]:
        """Detect anomalies using Z-score method."""
        anomalies = []
        
        for metric, value in data.items():
            if len(self.data_history) < 10:
                continue
            
            # Calculate statistics for this metric
            metric_values = [h['data'].get(metric, 0) for h in self.data_history 
                           if metric in h['data']]
            
            if len(metric_values) < 5:
                continue
            
            mean_val = np.mean(metric_values)
            std_val = np.std(metric_values)
            
            if std_val == 0:
                continue
            
            z_score = abs((value - mean_val) / std_val)
            
            if z_score > self.config.z_score_threshold:
                severity = self._determine_severity(z_score, self.config.z_score_threshold)
                
                anomaly = AnomalyResult(
                    anomaly_id=f"z_score_{metric}_{int(time.time())}",
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=severity,
                    score=z_score,
                    confidence=min(1.0, z_score / (2 * self.config.z_score_threshold)),
                    timestamp=datetime.now(),
                    data_point={metric: value},
                    explanation=f"Z-score {z_score:.2f} exceeds threshold {self.config.z_score_threshold}"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_iqr_anomalies(self, data: Dict[str, float]) -> List[AnomalyResult]:
        """Detect anomalies using Interquartile Range (IQR) method."""
        anomalies = []
        
        for metric, value in data.items():
            if len(self.data_history) < 10:
                continue
            
            metric_values = [h['data'].get(metric, 0) for h in self.data_history 
                           if metric in h['data']]
            
            if len(metric_values) < 5:
                continue
            
            q1 = np.percentile(metric_values, 25)
            q3 = np.percentile(metric_values, 75)
            iqr = q3 - q1
            
            if iqr == 0:
                continue
            
            lower_bound = q1 - self.config.iqr_multiplier * iqr
            upper_bound = q3 + self.config.iqr_multiplier * iqr
            
            if value < lower_bound or value > upper_bound:
                distance = min(abs(value - lower_bound), abs(value - upper_bound))
                severity = self._determine_severity(distance / iqr, 1.0)
                
                anomaly = AnomalyResult(
                    anomaly_id=f"iqr_{metric}_{int(time.time())}",
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=severity,
                    score=distance / iqr,
                    confidence=min(1.0, distance / (2 * iqr)),
                    timestamp=datetime.now(),
                    data_point={metric: value},
                    explanation=f"Value {value:.2f} outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_modified_z_score_anomalies(self, data: Dict[str, float]) -> List[AnomalyResult]:
        """Detect anomalies using modified Z-score (MAD-based)."""
        anomalies = []
        
        for metric, value in data.items():
            if len(self.data_history) < 10:
                continue
            
            metric_values = [h['data'].get(metric, 0) for h in self.data_history 
                           if metric in h['data']]
            
            if len(metric_values) < 5:
                continue
            
            median_val = np.median(metric_values)
            mad = np.median(np.abs(metric_values - median_val))
            
            if mad == 0:
                continue
            
            modified_z_score = 0.6745 * (value - median_val) / mad
            
            if abs(modified_z_score) > self.config.z_score_threshold:
                severity = self._determine_severity(abs(modified_z_score), self.config.z_score_threshold)
                
                anomaly = AnomalyResult(
                    anomaly_id=f"modified_z_{metric}_{int(time.time())}",
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=severity,
                    score=abs(modified_z_score),
                    confidence=min(1.0, abs(modified_z_score) / (2 * self.config.z_score_threshold)),
                    timestamp=datetime.now(),
                    data_point={metric: value},
                    explanation=f"Modified Z-score {modified_z_score:.2f} exceeds threshold"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _determine_severity(self, score: float, threshold: float) -> AnomalySeverity:
        """Determine anomaly severity based on score and threshold."""
        ratio = score / threshold
        
        if ratio >= 3.0:
            return AnomalySeverity.CRITICAL
        elif ratio >= 2.0:
            return AnomalySeverity.HIGH
        elif ratio >= 1.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW


class PatternBasedAnomalyDetector:
    """Pattern-based anomaly detection."""
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.patterns = {}
        self.pattern_history = deque(maxlen=config.pattern_window_size)
        
    def add_data_point(self, data: Dict[str, float]) -> List[AnomalyResult]:
        """Add a data point and detect pattern-based anomalies."""
        self.pattern_history.append({
            'data': data,
            'timestamp': datetime.now()
        })
        
        anomalies = []
        
        if len(self.pattern_history) < 10:
            return anomalies
        
        # Detect pattern breaks
        pattern_anomalies = self._detect_pattern_breaks(data)
        anomalies.extend(pattern_anomalies)
        
        # Detect unusual sequences
        sequence_anomalies = self._detect_unusual_sequences()
        anomalies.extend(sequence_anomalies)
        
        return anomalies
    
    def _detect_pattern_breaks(self, data: Dict[str, float]) -> List[AnomalyResult]:
        """Detect breaks in established patterns."""
        anomalies = []
        
        for metric, value in data.items():
            metric_values = [h['data'].get(metric, 0) for h in self.pattern_history 
                           if metric in h['data']]
            
            if len(metric_values) < 10:
                continue
            
            # Calculate pattern similarity
            recent_pattern = metric_values[-5:]  # Last 5 values
            historical_patterns = self._get_historical_patterns(metric, len(recent_pattern))
            
            if not historical_patterns:
                continue
            
            # Find most similar historical pattern
            max_similarity = 0
            for hist_pattern in historical_patterns:
                similarity = self._calculate_pattern_similarity(recent_pattern, hist_pattern)
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity < self.config.pattern_similarity_threshold:
                severity = self._determine_pattern_severity(max_similarity)
                
                anomaly = AnomalyResult(
                    anomaly_id=f"pattern_break_{metric}_{int(time.time())}",
                    anomaly_type=AnomalyType.PATTERN_BASED,
                    severity=severity,
                    score=1.0 - max_similarity,
                    confidence=max_similarity,
                    timestamp=datetime.now(),
                    data_point={metric: value},
                    explanation=f"Pattern break detected: similarity {max_similarity:.2f} below threshold"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_unusual_sequences(self) -> List[AnomalyResult]:
        """Detect unusual sequences in the data."""
        anomalies = []
        
        if len(self.pattern_history) < 20:
            return anomalies
        
        # Look for unusual sequences across all metrics
        recent_sequence = [h['data'] for h in list(self.pattern_history)[-10:]]
        
        # Calculate sequence characteristics
        sequence_stats = self._calculate_sequence_stats(recent_sequence)
        
        # Compare with historical sequences
        historical_sequences = self._get_historical_sequences()
        
        if historical_sequences:
            # Find most similar historical sequence
            max_similarity = 0
            for hist_sequence in historical_sequences:
                similarity = self._calculate_sequence_similarity(recent_sequence, hist_sequence)
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity < 0.5:  # Low similarity threshold for sequences
                severity = AnomalySeverity.MEDIUM if max_similarity < 0.3 else AnomalySeverity.LOW
                
                anomaly = AnomalyResult(
                    anomaly_id=f"unusual_sequence_{int(time.time())}",
                    anomaly_type=AnomalyType.PATTERN_BASED,
                    severity=severity,
                    score=1.0 - max_similarity,
                    confidence=max_similarity,
                    timestamp=datetime.now(),
                    data_point=recent_sequence[-1],
                    explanation=f"Unusual sequence detected: similarity {max_similarity:.2f}"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _get_historical_patterns(self, metric: str, pattern_length: int) -> List[List[float]]:
        """Get historical patterns for a metric."""
        patterns = []
        
        for i in range(len(self.pattern_history) - pattern_length):
            pattern = [h['data'].get(metric, 0) for h in 
                      list(self.pattern_history)[i:i+pattern_length] 
                      if metric in h['data']]
            if len(pattern) == pattern_length:
                patterns.append(pattern)
        
        return patterns
    
    def _calculate_pattern_similarity(self, pattern1: List[float], pattern2: List[float]) -> float:
        """Calculate similarity between two patterns."""
        if len(pattern1) != len(pattern2):
            return 0.0
        
        # Normalize patterns
        p1_norm = np.array(pattern1) / (np.linalg.norm(pattern1) + 1e-8)
        p2_norm = np.array(pattern2) / (np.linalg.norm(pattern2) + 1e-8)
        
        # Calculate cosine similarity
        similarity = np.dot(p1_norm, p2_norm)
        return max(0.0, similarity)
    
    def _calculate_sequence_stats(self, sequence: List[Dict[str, float]]) -> Dict[str, Any]:
        """Calculate statistics for a sequence of data points."""
        if not sequence:
            return {}
        
        all_values = []
        for data_point in sequence:
            all_values.extend(data_point.values())
        
        return {
            'mean': np.mean(all_values),
            'std': np.std(all_values),
            'trend': self._calculate_trend(all_values),
            'volatility': np.std(np.diff(all_values)) if len(all_values) > 1 else 0
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        return coeffs[0]
    
    def _get_historical_sequences(self) -> List[List[Dict[str, float]]]:
        """Get historical sequences for comparison."""
        sequences = []
        sequence_length = 10
        
        for i in range(len(self.pattern_history) - sequence_length):
            sequence = [h['data'] for h in 
                       list(self.pattern_history)[i:i+sequence_length]]
            sequences.append(sequence)
        
        return sequences
    
    def _calculate_sequence_similarity(self, seq1: List[Dict[str, float]], 
                                     seq2: List[Dict[str, float]]) -> float:
        """Calculate similarity between two sequences."""
        if len(seq1) != len(seq2):
            return 0.0
        
        similarities = []
        for dp1, dp2 in zip(seq1, seq2):
            # Calculate similarity for each data point
            common_metrics = set(dp1.keys()) & set(dp2.keys())
            if not common_metrics:
                similarities.append(0.0)
                continue
            
            point_similarities = []
            for metric in common_metrics:
                val1, val2 = dp1[metric], dp2[metric]
                if val1 == 0 and val2 == 0:
                    point_similarities.append(1.0)
                else:
                    similarity = 1.0 - abs(val1 - val2) / (abs(val1) + abs(val2) + 1e-8)
                    point_similarities.append(max(0.0, similarity))
            
            similarities.append(np.mean(point_similarities))
        
        return np.mean(similarities)
    
    def _determine_pattern_severity(self, similarity: float) -> AnomalySeverity:
        """Determine severity based on pattern similarity."""
        if similarity < 0.2:
            return AnomalySeverity.HIGH
        elif similarity < 0.4:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW


class TemporalAnomalyDetector:
    """Temporal anomaly detection for time-series data."""
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.temporal_data = defaultdict(list)
        self.seasonal_patterns = {}
        
    def add_data_point(self, data: Dict[str, float], timestamp: datetime) -> List[AnomalyResult]:
        """Add a data point and detect temporal anomalies."""
        anomalies = []
        
        for metric, value in data.items():
            self.temporal_data[metric].append({
                'value': value,
                'timestamp': timestamp
            })
            
            # Detect temporal anomalies
            temporal_anomalies = self._detect_temporal_anomalies(metric, value, timestamp)
            anomalies.extend(temporal_anomalies)
        
        return anomalies
    
    def _detect_temporal_anomalies(self, metric: str, value: float, 
                                 timestamp: datetime) -> List[AnomalyResult]:
        """Detect temporal anomalies for a metric."""
        anomalies = []
        
        if len(self.temporal_data[metric]) < 20:
            return anomalies
        
        # Detect seasonality anomalies
        seasonal_anomalies = self._detect_seasonality_anomalies(metric, value, timestamp)
        anomalies.extend(seasonal_anomalies)
        
        # Detect trend anomalies
        trend_anomalies = self._detect_trend_anomalies(metric, value, timestamp)
        anomalies.extend(trend_anomalies)
        
        return anomalies
    
    def _detect_seasonality_anomalies(self, metric: str, value: float, 
                                    timestamp: datetime) -> List[AnomalyResult]:
        """Detect anomalies based on seasonal patterns."""
        anomalies = []
        
        if not self.config.temporal_seasonality_detection:
            return anomalies
        
        # Extract time components
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Get historical data for similar time periods
        similar_times = [
            dp for dp in self.temporal_data[metric]
            if (dp['timestamp'].hour == hour and 
                dp['timestamp'].weekday() == day_of_week)
        ]
        
        if len(similar_times) < 5:
            return anomalies
        
        similar_values = [dp['value'] for dp in similar_times]
        mean_similar = np.mean(similar_values)
        std_similar = np.std(similar_values)
        
        if std_similar > 0:
            z_score = abs((value - mean_similar) / std_similar)
            
            if z_score > 2.0:  # Lower threshold for seasonal detection
                severity = AnomalySeverity.MEDIUM if z_score > 3.0 else AnomalySeverity.LOW
                
                anomaly = AnomalyResult(
                    anomaly_id=f"seasonal_{metric}_{int(time.time())}",
                    anomaly_type=AnomalyType.TEMPORAL,
                    severity=severity,
                    score=z_score,
                    confidence=min(1.0, z_score / 4.0),
                    timestamp=timestamp,
                    data_point={metric: value},
                    explanation=f"Seasonal anomaly: value {value:.2f} unusual for {hour}:00 on day {day_of_week}"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_trend_anomalies(self, metric: str, value: float, 
                              timestamp: datetime) -> List[AnomalyResult]:
        """Detect anomalies based on trend analysis."""
        anomalies = []
        
        if len(self.temporal_data[metric]) < 10:
            return anomalies
        
        # Get recent data for trend analysis
        recent_data = self.temporal_data[metric][-20:]
        recent_values = [dp['value'] for dp in recent_data]
        
        # Calculate trend
        x = np.arange(len(recent_values))
        y = np.array(recent_values)
        
        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        trend_slope = coeffs[0]
        
        # Calculate expected value based on trend
        expected_value = np.polyval(coeffs, len(recent_values))
        
        # Check if current value deviates significantly from trend
        if len(recent_values) > 1:
            residual_std = np.std(y - np.polyval(coeffs, x))
            
            if residual_std > 0:
                deviation = abs(value - expected_value) / residual_std
                
                if deviation > 2.0:
                    severity = AnomalySeverity.MEDIUM if deviation > 3.0 else AnomalySeverity.LOW
                    
                    anomaly = AnomalyResult(
                        anomaly_id=f"trend_{metric}_{int(time.time())}",
                        anomaly_type=AnomalyType.TEMPORAL,
                        severity=severity,
                        score=deviation,
                        confidence=min(1.0, deviation / 4.0),
                        timestamp=timestamp,
                        data_point={metric: value},
                        explanation=f"Trend anomaly: value {value:.2f} deviates from expected {expected_value:.2f}"
                    )
                    anomalies.append(anomaly)
        
        return anomalies


class AdvancedAnomalyDetection:
    """Main anomaly detection system combining multiple methods."""
    
    def __init__(self, config: Optional[AnomalyDetectionConfig] = None):
        self.config = config or AnomalyDetectionConfig()
        self.cache = UnifiedCachingSystem()
        
        # Initialize detectors
        self.statistical_detector = StatisticalAnomalyDetector(self.config)
        self.pattern_detector = PatternBasedAnomalyDetector(self.config)
        self.temporal_detector = TemporalAnomalyDetector(self.config)
        
        # Results storage
        self.anomalies: List[AnomalyResult] = []
        self.anomaly_patterns: Dict[str, AnomalyPattern] = {}
        
        # Callbacks
        self.anomaly_callbacks: List[Callable[[AnomalyResult], None]] = []
        
        self._initialized = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize the anomaly detection system."""
        try:
            self._initialized = True
            self.logger.info("Advanced anomaly detection system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize anomaly detection: {e}")
            raise
    
    def add_data_point(self, data: Dict[str, float], 
                      timestamp: Optional[datetime] = None) -> List[AnomalyResult]:
        """Add a data point and detect anomalies using all methods."""
        if timestamp is None:
            timestamp = datetime.now()
        
        all_anomalies = []
        
        # Statistical detection
        statistical_anomalies = self.statistical_detector.add_data_point(data)
        all_anomalies.extend(statistical_anomalies)
        
        # Pattern-based detection
        pattern_anomalies = self.pattern_detector.add_data_point(data)
        all_anomalies.extend(pattern_anomalies)
        
        # Temporal detection
        temporal_anomalies = self.temporal_detector.add_data_point(data, timestamp)
        all_anomalies.extend(temporal_anomalies)
        
        # Store and process anomalies
        for anomaly in all_anomalies:
            self.anomalies.append(anomaly)
            self._process_anomaly(anomaly)
            
            # Trigger callbacks
            for callback in self.anomaly_callbacks:
                try:
                    callback(anomaly)
                except Exception as e:
                    self.logger.error(f"Error in anomaly callback: {e}")
        
        return all_anomalies
    
    def _process_anomaly(self, anomaly: AnomalyResult) -> None:
        """Process and analyze an anomaly."""
        # Update anomaly patterns
        self._update_anomaly_patterns(anomaly)
        
        # Cache the anomaly
        cache_key = f"anomaly_{anomaly.anomaly_id}"
        self.cache.set(cache_key, anomaly, ttl_seconds=3600)
    
    def _update_anomaly_patterns(self, anomaly: AnomalyResult) -> None:
        """Update anomaly patterns based on new anomaly."""
        pattern_key = f"{anomaly.anomaly_type.value}_{anomaly.severity.value}"
        
        if pattern_key not in self.anomaly_patterns:
            self.anomaly_patterns[pattern_key] = AnomalyPattern(
                pattern_id=pattern_key,
                pattern_type=anomaly.anomaly_type.value,
                frequency=0,
                first_detected=anomaly.timestamp,
                last_detected=anomaly.timestamp
            )
        
        pattern = self.anomaly_patterns[pattern_key]
        pattern.frequency += 1
        pattern.last_detected = anomaly.timestamp
        pattern.examples.append({
            'anomaly_id': anomaly.anomaly_id,
            'score': anomaly.score,
            'timestamp': anomaly.timestamp
        })
        
        # Keep only recent examples
        if len(pattern.examples) > 10:
            pattern.examples = pattern.examples[-10:]
    
    def add_anomaly_callback(self, callback: Callable[[AnomalyResult], None]) -> None:
        """Add a callback for anomaly detection."""
        self.anomaly_callbacks.append(callback)
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies."""
        total_anomalies = len(self.anomalies)
        
        # Count by type
        type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for anomaly in self.anomalies:
            type_counts[anomaly.anomaly_type.value] += 1
            severity_counts[anomaly.severity.value] += 1
        
        # Recent anomalies (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_anomalies = [a for a in self.anomalies if a.timestamp > recent_cutoff]
        
        return {
            'total_anomalies': total_anomalies,
            'recent_anomalies': len(recent_anomalies),
            'type_distribution': dict(type_counts),
            'severity_distribution': dict(severity_counts),
            'pattern_summary': {
                pattern_id: {
                    'frequency': pattern.frequency,
                    'first_detected': pattern.first_detected,
                    'last_detected': pattern.last_detected
                }
                for pattern_id, pattern in self.anomaly_patterns.items()
            }
        }
    
    def get_recent_anomalies(self, limit: int = 50) -> List[AnomalyResult]:
        """Get recent anomalies."""
        return sorted(self.anomalies, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def cleanup_old_data(self, max_age_hours: int = 168) -> None:  # 1 week default
        """Clean up old anomaly data."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Clean up anomalies
        self.anomalies = [a for a in self.anomalies if a.timestamp > cutoff_time]
        
        # Clean up pattern examples
        for pattern in self.anomaly_patterns.values():
            pattern.examples = [ex for ex in pattern.examples 
                              if ex['timestamp'] > cutoff_time]
        
        self.logger.info(f"Cleaned up anomaly data older than {max_age_hours} hours")


# Global instance
_anomaly_detector = None

def get_anomaly_detector() -> AdvancedAnomalyDetection:
    """Get the global anomaly detection instance."""
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = AdvancedAnomalyDetection()
        _anomaly_detector.initialize()
    return _anomaly_detector
