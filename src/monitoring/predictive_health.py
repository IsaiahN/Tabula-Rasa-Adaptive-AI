#!/usr/bin/env python3
"""
Predictive Health Monitoring System

Provides predictive analysis of system health, performance trends, and
potential issues before they become critical.
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
from ..core.performance_monitor import AlertLevel


class HealthMetric(Enum):
    """Types of health metrics."""
    SYSTEM_PERFORMANCE = "system_performance"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    LEARNING_EFFICIENCY = "learning_efficiency"
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"


class HealthStatus(Enum):
    """System health status levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class PredictionType(Enum):
    """Types of predictions."""
    HEALTH_DECLINE = "health_decline"
    PERFORMANCE_DROP = "performance_drop"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    ERROR_SPIKE = "error_spike"
    LEARNING_STAGNATION = "learning_stagnation"


@dataclass
class HealthDataPoint:
    """A single health data point."""
    metric: HealthMetric
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthPrediction:
    """A health prediction result."""
    prediction_id: str
    prediction_type: PredictionType
    confidence: float
    time_horizon: int  # Hours into the future
    current_value: float
    predicted_value: float
    severity: AlertLevel
    timestamp: datetime
    explanation: str
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthTrend:
    """A health trend analysis."""
    metric: HealthMetric
    trend_direction: str  # "improving", "stable", "declining"
    trend_strength: float  # 0-1
    trend_confidence: float  # 0-1
    time_window_hours: int
    slope: float
    r_squared: float


class HealthPredictor:
    """Predicts future health based on historical data."""
    
    def __init__(self, metric: HealthMetric, window_size: int = 100):
        self.metric = metric
        self.window_size = window_size
        self.data_points = deque(maxlen=window_size)
        self.trends = []
        
        # Prediction models
        self.linear_model = None
        self.exponential_model = None
        self.seasonal_model = None
        
    def add_data_point(self, value: float, timestamp: datetime, 
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a health data point."""
        data_point = HealthDataPoint(
            metric=self.metric,
            value=value,
            timestamp=timestamp,
            metadata=metadata or {}
        )
        self.data_points.append(data_point)
        
        # Update models
        self._update_models()
    
    def _update_models(self) -> None:
        """Update prediction models with new data."""
        if len(self.data_points) < 10:
            return
        
        values = [dp.value for dp in self.data_points]
        timestamps = [dp.timestamp for dp in self.data_points]
        
        # Convert timestamps to numeric values
        time_numeric = [(ts - timestamps[0]).total_seconds() / 3600 for ts in timestamps]
        
        # Linear model
        self.linear_model = self._fit_linear_model(time_numeric, values)
        
        # Exponential model (for growth/decay patterns)
        self.exponential_model = self._fit_exponential_model(time_numeric, values)
        
        # Seasonal model (if enough data)
        if len(values) >= 24:  # At least 24 hours of data
            self.seasonal_model = self._fit_seasonal_model(time_numeric, values)
    
    def _fit_linear_model(self, x: List[float], y: List[float]) -> Dict[str, Any]:
        """Fit a linear regression model."""
        if len(x) < 2:
            return None
        
        x_array = np.array(x)
        y_array = np.array(y)
        
        # Linear regression
        coeffs = np.polyfit(x_array, y_array, 1)
        slope, intercept = coeffs
        
        # Calculate R-squared
        y_pred = np.polyval(coeffs, x_array)
        ss_res = np.sum((y_array - y_pred) ** 2)
        ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'predict': lambda t: slope * t + intercept
        }
    
    def _fit_exponential_model(self, x: List[float], y: List[float]) -> Dict[str, Any]:
        """Fit an exponential model."""
        if len(x) < 3:
            return None
        
        try:
            # Ensure positive values for exponential fitting
            y_positive = [max(y_val, 0.001) for y_val in y]
            
            # Log transform
            log_y = np.log(y_positive)
            
            # Linear regression on log-transformed data
            coeffs = np.polyfit(x, log_y, 1)
            a, b = coeffs
            
            # Convert back to exponential form: y = A * e^(B * x)
            A = np.exp(a)
            B = b
            
            # Calculate R-squared
            y_pred = A * np.exp(B * np.array(x))
            ss_res = np.sum((np.array(y) - y_pred) ** 2)
            ss_tot = np.sum((np.array(y) - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'A': A,
                'B': B,
                'r_squared': r_squared,
                'predict': lambda t: A * np.exp(B * t)
            }
        except:
            return None
    
    def _fit_seasonal_model(self, x: List[float], y: List[float]) -> Dict[str, Any]:
        """Fit a seasonal model with daily patterns."""
        if len(x) < 24:
            return None
        
        try:
            # Extract hour of day for each data point
            hours = [(datetime.fromtimestamp(ts * 3600 + time.mktime(self.data_points[0].timestamp.timetuple()))).hour 
                    for ts in x]
            
            # Calculate average value for each hour
            hourly_averages = defaultdict(list)
            for hour, value in zip(hours, y):
                hourly_averages[hour].append(value)
            
            hourly_means = {hour: np.mean(values) for hour, values in hourly_averages.items()}
            
            # Create seasonal component
            def seasonal_predict(t):
                hour = int(t) % 24
                return hourly_means.get(hour, np.mean(y))
            
            # Calculate R-squared
            y_pred = [seasonal_predict(ts) for ts in x]
            ss_res = np.sum((np.array(y) - y_pred) ** 2)
            ss_tot = np.sum((np.array(y) - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'hourly_means': hourly_means,
                'r_squared': r_squared,
                'predict': seasonal_predict
            }
        except:
            return None
    
    def predict(self, hours_ahead: int) -> Optional[HealthPrediction]:
        """Predict health value hours into the future."""
        if len(self.data_points) < 10:
            return None
        
        current_value = self.data_points[-1].value
        current_time = self.data_points[-1].timestamp
        future_time = current_time + timedelta(hours=hours_ahead)
        
        # Calculate time offset for prediction
        time_offset = (future_time - self.data_points[0].timestamp).total_seconds() / 3600
        
        # Try different models and pick the best one
        predictions = []
        
        if self.linear_model:
            pred_value = self.linear_model['predict'](time_offset)
            predictions.append({
                'value': pred_value,
                'confidence': self.linear_model['r_squared'],
                'model': 'linear'
            })
        
        if self.exponential_model:
            pred_value = self.exponential_model['predict'](time_offset)
            predictions.append({
                'value': pred_value,
                'confidence': self.exponential_model['r_squared'],
                'model': 'exponential'
            })
        
        if self.seasonal_model:
            pred_value = self.seasonal_model['predict'](time_offset)
            predictions.append({
                'value': pred_value,
                'confidence': self.seasonal_model['r_squared'],
                'model': 'seasonal'
            })
        
        if not predictions:
            return None
        
        # Pick the model with highest confidence
        best_prediction = max(predictions, key=lambda p: p['confidence'])
        predicted_value = best_prediction['value']
        confidence = best_prediction['confidence']
        
        # Determine prediction type and severity
        prediction_type, severity, explanation = self._analyze_prediction(
            current_value, predicted_value, hours_ahead
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            prediction_type, current_value, predicted_value
        )
        
        return HealthPrediction(
            prediction_id=f"{self.metric.value}_{int(time.time())}",
            prediction_type=prediction_type,
            confidence=confidence,
            time_horizon=hours_ahead,
            current_value=current_value,
            predicted_value=predicted_value,
            severity=severity,
            timestamp=current_time,
            explanation=explanation,
            recommendations=recommendations,
            metadata={'model_used': best_prediction['model']}
        )
    
    def _analyze_prediction(self, current_value: float, predicted_value: float, 
                          hours_ahead: int) -> Tuple[PredictionType, AlertLevel, str]:
        """Analyze the prediction and determine type and severity."""
        change_percent = (predicted_value - current_value) / current_value if current_value != 0 else 0
        
        # Determine prediction type based on metric
        if self.metric in [HealthMetric.MEMORY_USAGE, HealthMetric.CPU_USAGE]:
            if predicted_value > 0.9:  # 90% threshold
                return (PredictionType.RESOURCE_EXHAUSTION, AlertLevel.CRITICAL,
                       f"Resource usage predicted to reach {predicted_value:.1%} in {hours_ahead} hours")
            elif predicted_value > 0.8:  # 80% threshold
                return (PredictionType.RESOURCE_EXHAUSTION, AlertLevel.WARNING,
                       f"Resource usage predicted to reach {predicted_value:.1%} in {hours_ahead} hours")
        
        elif self.metric == HealthMetric.ERROR_RATE:
            if predicted_value > 0.1:  # 10% error rate
                return (PredictionType.ERROR_SPIKE, AlertLevel.CRITICAL,
                       f"Error rate predicted to reach {predicted_value:.1%} in {hours_ahead} hours")
            elif predicted_value > 0.05:  # 5% error rate
                return (PredictionType.ERROR_SPIKE, AlertLevel.WARNING,
                       f"Error rate predicted to reach {predicted_value:.1%} in {hours_ahead} hours")
        
        elif self.metric == HealthMetric.LEARNING_EFFICIENCY:
            if change_percent < -0.2:  # 20% decline
                return (PredictionType.LEARNING_STAGNATION, AlertLevel.WARNING,
                       f"Learning efficiency predicted to decline by {abs(change_percent):.1%} in {hours_ahead} hours")
        
        # General health decline
        if change_percent < -0.3:  # 30% decline
            return (PredictionType.HEALTH_DECLINE, AlertLevel.CRITICAL,
                   f"Health metric predicted to decline by {abs(change_percent):.1%} in {hours_ahead} hours")
        elif change_percent < -0.1:  # 10% decline
            return (PredictionType.HEALTH_DECLINE, AlertLevel.WARNING,
                   f"Health metric predicted to decline by {abs(change_percent):.1%} in {hours_ahead} hours")
        
        # Performance drop
        if change_percent < -0.15:  # 15% decline
            return (PredictionType.PERFORMANCE_DROP, AlertLevel.WARNING,
                   f"Performance predicted to drop by {abs(change_percent):.1%} in {hours_ahead} hours")
        
        return (PredictionType.HEALTH_DECLINE, AlertLevel.INFO,
               f"Health metric predicted to change by {change_percent:.1%} in {hours_ahead} hours")
    
    def _generate_recommendations(self, prediction_type: PredictionType, 
                                current_value: float, predicted_value: float) -> List[str]:
        """Generate recommendations based on prediction."""
        recommendations = []
        
        if prediction_type == PredictionType.RESOURCE_EXHAUSTION:
            recommendations.extend([
                "Consider scaling up resources",
                "Optimize resource usage",
                "Implement resource monitoring alerts"
            ])
        
        elif prediction_type == PredictionType.ERROR_SPIKE:
            recommendations.extend([
                "Review error logs for patterns",
                "Implement error handling improvements",
                "Consider circuit breaker patterns"
            ])
        
        elif prediction_type == PredictionType.LEARNING_STAGNATION:
            recommendations.extend([
                "Review learning parameters",
                "Consider data augmentation",
                "Implement learning rate scheduling"
            ])
        
        elif prediction_type == PredictionType.HEALTH_DECLINE:
            recommendations.extend([
                "Monitor system health closely",
                "Review recent changes",
                "Consider preventive maintenance"
            ])
        
        elif prediction_type == PredictionType.PERFORMANCE_DROP:
            recommendations.extend([
                "Profile system performance",
                "Optimize critical paths",
                "Consider caching strategies"
            ])
        
        return recommendations
    
    def get_trend_analysis(self, window_hours: int = 24) -> Optional[HealthTrend]:
        """Get trend analysis for the metric."""
        if len(self.data_points) < 5:
            return None
        
        # Get data within the window
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_data = [dp for dp in self.data_points if dp.timestamp > cutoff_time]
        
        if len(recent_data) < 3:
            return None
        
        values = [dp.value for dp in recent_data]
        timestamps = [dp.timestamp for dp in recent_data]
        
        # Calculate trend
        time_numeric = [(ts - timestamps[0]).total_seconds() / 3600 for ts in timestamps]
        
        if len(time_numeric) < 2:
            return None
        
        # Linear regression
        coeffs = np.polyfit(time_numeric, values, 1)
        slope = coeffs[0]
        
        # Calculate R-squared
        y_pred = np.polyval(coeffs, time_numeric)
        ss_res = np.sum((np.array(values) - y_pred) ** 2)
        ss_tot = np.sum((np.array(values) - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Determine trend direction
        if slope > 0.01:
            direction = "improving"
        elif slope < -0.01:
            direction = "declining"
        else:
            direction = "stable"
        
        # Calculate trend strength
        trend_strength = min(1.0, abs(slope) * 10)  # Scale slope to 0-1
        
        return HealthTrend(
            metric=self.metric,
            trend_direction=direction,
            trend_strength=trend_strength,
            trend_confidence=r_squared,
            time_window_hours=window_hours,
            slope=slope,
            r_squared=r_squared
        )


class PredictiveHealthMonitoring:
    """Main predictive health monitoring system."""
    
    def __init__(self, cache_config: Optional[CacheConfig] = None):
        self.cache_config = cache_config or CacheConfig()
        self.cache = UnifiedCachingSystem(self.cache_config)
        
        # Health predictors for each metric
        self.predictors: Dict[HealthMetric, HealthPredictor] = {}
        
        # Predictions and trends
        self.predictions: List[HealthPrediction] = []
        self.trends: List[HealthTrend] = []
        
        # Callbacks
        self.prediction_callbacks: List[Callable[[HealthPrediction], None]] = []
        
        # Configuration
        self.prediction_horizons = [1, 6, 24]  # Hours ahead to predict
        self.trend_window_hours = 24
        
        self._initialized = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize the predictive health monitoring system."""
        try:
            # Initialize predictors for all health metrics
            for metric in HealthMetric:
                self.predictors[metric] = HealthPredictor(metric)
            
            self._initialized = True
            self.logger.info("Predictive health monitoring system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize predictive health monitoring: {e}")
            raise
    
    def add_health_data(self, metric: HealthMetric, value: float, 
                       timestamp: Optional[datetime] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add health data for a specific metric."""
        if timestamp is None:
            timestamp = datetime.now()
        
        if metric not in self.predictors:
            self.logger.warning(f"No predictor for metric {metric}")
            return
        
        # Add data to predictor
        self.predictors[metric].add_data_point(value, timestamp, metadata)
        
        # Generate predictions
        self._generate_predictions(metric)
        
        # Update trends
        self._update_trends(metric)
    
    def _generate_predictions(self, metric: HealthMetric) -> None:
        """Generate predictions for a metric."""
        predictor = self.predictors[metric]
        
        for hours_ahead in self.prediction_horizons:
            prediction = predictor.predict(hours_ahead)
            if prediction:
                self.predictions.append(prediction)
                
                # Trigger callbacks
                for callback in self.prediction_callbacks:
                    try:
                        callback(prediction)
                    except Exception as e:
                        self.logger.error(f"Error in prediction callback: {e}")
    
    def _update_trends(self, metric: HealthMetric) -> None:
        """Update trend analysis for a metric."""
        predictor = self.predictors[metric]
        trend = predictor.get_trend_analysis(self.trend_window_hours)
        
        if trend:
            # Remove old trend for this metric
            self.trends = [t for t in self.trends if t.metric != metric]
            self.trends.append(trend)
    
    def add_prediction_callback(self, callback: Callable[[HealthPrediction], None]) -> None:
        """Add a callback for predictions."""
        self.prediction_callbacks.append(callback)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        # Calculate current health status
        current_health = self._calculate_current_health()
        
        # Get recent predictions
        recent_predictions = self._get_recent_predictions(hours=24)
        
        # Get trend summary
        trend_summary = self._get_trend_summary()
        
        return {
            'current_health': current_health,
            'recent_predictions': len(recent_predictions),
            'critical_predictions': len([p for p in recent_predictions if p.severity == AlertLevel.CRITICAL]),
            'warning_predictions': len([p for p in recent_predictions if p.severity == AlertLevel.WARNING]),
            'trend_summary': trend_summary,
            'predictors_status': {
                metric.value: len(predictor.data_points) 
                for metric, predictor in self.predictors.items()
            }
        }
    
    def _calculate_current_health(self) -> HealthStatus:
        """Calculate current overall health status."""
        if not self.predictors:
            return HealthStatus.GOOD
        
        # Get latest values for each metric
        latest_values = {}
        for metric, predictor in self.predictors.items():
            if predictor.data_points:
                latest_values[metric] = predictor.data_points[-1].value
        
        if not latest_values:
            return HealthStatus.GOOD
        
        # Calculate health score based on metric values
        health_score = 0.0
        total_weight = 0.0
        
        # Weight different metrics
        weights = {
            HealthMetric.SYSTEM_PERFORMANCE: 0.3,
            HealthMetric.MEMORY_USAGE: 0.2,
            HealthMetric.CPU_USAGE: 0.2,
            HealthMetric.LEARNING_EFFICIENCY: 0.15,
            HealthMetric.ERROR_RATE: 0.15
        }
        
        for metric, value in latest_values.items():
            if metric in weights:
                # Normalize value to 0-1 scale
                if metric in [HealthMetric.MEMORY_USAGE, HealthMetric.CPU_USAGE, HealthMetric.ERROR_RATE]:
                    # Lower is better for these metrics
                    normalized_value = max(0, 1 - value)
                else:
                    # Higher is better for these metrics
                    normalized_value = min(1, value)
                
                health_score += normalized_value * weights[metric]
                total_weight += weights[metric]
        
        if total_weight > 0:
            health_score /= total_weight
        
        # Convert to health status
        if health_score >= 0.9:
            return HealthStatus.EXCELLENT
        elif health_score >= 0.7:
            return HealthStatus.GOOD
        elif health_score >= 0.5:
            return HealthStatus.WARNING
        elif health_score >= 0.3:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.FAILED
    
    def _get_recent_predictions(self, hours: int = 24) -> List[HealthPrediction]:
        """Get recent predictions within specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [p for p in self.predictions if p.timestamp > cutoff_time]
    
    def _get_trend_summary(self) -> Dict[str, Any]:
        """Get summary of trends."""
        if not self.trends:
            return {}
        
        trend_directions = defaultdict(int)
        avg_confidence = 0.0
        
        for trend in self.trends:
            trend_directions[trend.trend_direction] += 1
            avg_confidence += trend.trend_confidence
        
        avg_confidence /= len(self.trends) if self.trends else 1
        
        return {
            'trend_distribution': dict(trend_directions),
            'average_confidence': avg_confidence,
            'total_trends': len(self.trends)
        }
    
    def get_metric_predictions(self, metric: HealthMetric, 
                             hours_ahead: int = 24) -> List[HealthPrediction]:
        """Get predictions for a specific metric."""
        cutoff_time = datetime.now() - timedelta(hours=hours_ahead)
        return [p for p in self.predictions 
                if p.prediction_id.startswith(metric.value) 
                and p.timestamp > cutoff_time]
    
    def cleanup_old_data(self, max_age_hours: int = 168) -> None:  # 1 week default
        """Clean up old prediction data."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Clean up predictions
        self.predictions = [p for p in self.predictions if p.timestamp > cutoff_time]
        
        # Clean up trends
        self.trends = [t for t in self.trends if t.timestamp > cutoff_time] if hasattr(self.trends[0], 'timestamp') else self.trends
        
        self.logger.info(f"Cleaned up prediction data older than {max_age_hours} hours")


# Global instance
_predictive_health = None

def get_predictive_health() -> PredictiveHealthMonitoring:
    """Get the global predictive health monitoring instance."""
    global _predictive_health
    if _predictive_health is None:
        _predictive_health = PredictiveHealthMonitoring()
        _predictive_health.initialize()
    return _predictive_health
