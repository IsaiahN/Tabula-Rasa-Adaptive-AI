"""
Predictive Analytics

Advanced predictive analytics for forecasting system behavior,
performance trends, and resource requirements.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from ...training.interfaces import ComponentInterface
from ...training.caching import CacheManager, CacheConfig
from ...training.monitoring import PerformanceMonitor


class PredictionType(Enum):
    """Types of predictions."""
    PERFORMANCE = "performance"
    RESOURCE_USAGE = "resource_usage"
    LEARNING_PROGRESS = "learning_progress"
    SYSTEM_HEALTH = "system_health"
    FAILURE = "failure"
    CAPACITY = "capacity"


@dataclass
class PredictionModel:
    """Prediction model data structure."""
    model_type: str
    model: Any
    scaler: Optional[StandardScaler]
    features: List[str]
    target: str
    accuracy: float
    created_at: datetime


@dataclass
class PredictionResult:
    """Prediction result data structure."""
    prediction_type: PredictionType
    predicted_value: float
    confidence: float
    timestamp: datetime
    model_used: str
    metadata: Dict[str, Any]


class PredictiveAnalytics(ComponentInterface):
    """
    Advanced predictive analytics system for forecasting system behavior,
    performance trends, and resource requirements.
    """
    
    def __init__(self, cache_config: Optional[CacheConfig] = None):
        """Initialize the predictive analytics system."""
        self.cache_config = cache_config or CacheConfig()
        self.cache = CacheManager(self.cache_config)
        self.performance_monitor = PerformanceMonitor()
        
        # Analytics state
        self.models: Dict[str, PredictionModel] = {}
        self.predictions: List[PredictionResult] = []
        self.training_data: Dict[str, List[Dict[str, Any]]] = {}
        
        # Performance tracking
        self.prediction_times: List[float] = []
        self.model_training_times: List[float] = []
        
        # Predictive analytics configuration
        self.retention_days = 30
        self.min_training_samples = 50
        self.prediction_horizon_hours = 24
        self.model_retrain_interval_hours = 168  # 1 week
        
        self._initialized = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize the predictive analytics system."""
        try:
            self.cache.initialize()
            self.performance_monitor.start_monitoring()
            
            self._initialized = True
            self.logger.info("Predictive analytics system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize predictive analytics: {e}")
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get current component state."""
        return {
            'name': 'PredictiveAnalytics',
            'status': 'running' if self._initialized else 'stopped',
            'last_updated': datetime.now(),
            'metadata': {
                'models_count': len(self.models),
                'predictions_count': len(self.predictions),
                'training_data_count': sum(len(data) for data in self.training_data.values()),
                'average_prediction_time': np.mean(self.prediction_times) if self.prediction_times else 0.0
            }
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.performance_monitor.stop_monitoring()
            self.cache.clear()
            self._initialized = False
            self.logger.info("Predictive analytics system cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self._initialized and self.cache.is_healthy()
    
    def add_training_data(self, data_type: str, data: Dict[str, Any]) -> None:
        """Add training data for model training."""
        try:
            if data_type not in self.training_data:
                self.training_data[data_type] = []
            
            # Add timestamp if not present
            if 'timestamp' not in data:
                data['timestamp'] = datetime.now()
            
            self.training_data[data_type].append(data)
            
            # Clean up old training data
            self._cleanup_old_training_data()
            
            # Cache training data
            cache_key = f"training_data_{data_type}_{datetime.now().timestamp()}"
            self.cache.set(cache_key, data, ttl=3600)
            
            self.logger.debug(f"Added training data for {data_type}")
            
        except Exception as e:
            self.logger.error(f"Error adding training data: {e}")
    
    def train_model(self, data_type: str, target_column: str, 
                   feature_columns: Optional[List[str]] = None) -> PredictionModel:
        """Train a prediction model."""
        try:
            start_time = datetime.now()
            
            # Get training data
            if data_type not in self.training_data:
                raise ValueError(f"No training data available for {data_type}")
            
            training_data = self.training_data[data_type]
            
            if len(training_data) < self.min_training_samples:
                raise ValueError(f"Insufficient training data. Need at least {self.min_training_samples} samples.")
            
            # Prepare features and target
            if feature_columns is None:
                # Use all columns except target and timestamp
                feature_columns = [col for col in training_data[0].keys() 
                                 if col not in [target_column, 'timestamp']]
            
            # Extract features and target
            X = []
            y = []
            
            for data_point in training_data:
                try:
                    # Extract features
                    features = [data_point.get(col, 0) for col in feature_columns]
                    X.append(features)
                    
                    # Extract target
                    target_value = data_point.get(target_column, 0)
                    y.append(target_value)
                    
                except Exception as e:
                    self.logger.warning(f"Skipping invalid data point: {e}")
                    continue
            
            if len(X) < self.min_training_samples:
                raise ValueError(f"Insufficient valid training data. Need at least {self.min_training_samples} samples.")
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            
            # Calculate accuracy
            y_pred = model.predict(X_scaled)
            accuracy = r2_score(y, y_pred)
            
            # Create prediction model
            prediction_model = PredictionModel(
                model_type='RandomForest',
                model=model,
                scaler=scaler,
                features=feature_columns,
                target=target_column,
                accuracy=accuracy,
                created_at=datetime.now()
            )
            
            # Store model
            model_key = f"{data_type}_{target_column}"
            self.models[model_key] = prediction_model
            
            # Update performance metrics
            training_time = (datetime.now() - start_time).total_seconds()
            self.model_training_times.append(training_time)
            
            # Cache model
            cache_key = f"model_{model_key}"
            self.cache.set(cache_key, prediction_model, ttl=86400)  # 24 hours
            
            self.logger.info(f"Trained model for {data_type}->{target_column} in {training_time:.3f}s (accuracy: {accuracy:.3f})")
            
            return prediction_model
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise
    
    def predict(self, data_type: str, target_column: str, 
               features: Dict[str, float]) -> PredictionResult:
        """Make a prediction using a trained model."""
        try:
            start_time = datetime.now()
            
            # Get model
            model_key = f"{data_type}_{target_column}"
            if model_key not in self.models:
                raise ValueError(f"No trained model available for {data_type}->{target_column}")
            
            model_info = self.models[model_key]
            
            # Prepare features
            feature_values = []
            for feature in model_info.features:
                feature_values.append(features.get(feature, 0))
            
            # Scale features
            X = np.array([feature_values])
            X_scaled = model_info.scaler.transform(X)
            
            # Make prediction
            predicted_value = model_info.model.predict(X_scaled)[0]
            
            # Calculate confidence based on model accuracy
            confidence = min(1.0, max(0.0, model_info.accuracy))
            
            # Create prediction result
            result = PredictionResult(
                prediction_type=PredictionType.PERFORMANCE,  # Default type
                predicted_value=predicted_value,
                confidence=confidence,
                timestamp=datetime.now(),
                model_used=model_key,
                metadata={
                    'data_type': data_type,
                    'target_column': target_column,
                    'features_used': features,
                    'model_accuracy': model_info.accuracy
                }
            )
            
            # Store prediction
            self.predictions.append(result)
            
            # Update performance metrics
            prediction_time = (datetime.now() - start_time).total_seconds()
            self.prediction_times.append(prediction_time)
            
            # Cache prediction
            cache_key = f"prediction_{datetime.now().timestamp()}"
            self.cache.set(cache_key, result, ttl=3600)
            
            self.logger.debug(f"Made prediction for {data_type}->{target_column}: {predicted_value:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            raise
    
    def predict_performance_trend(self, metric_type: str, 
                                 current_value: float, 
                                 time_horizon_hours: int = 24) -> PredictionResult:
        """Predict performance trend for a specific metric."""
        try:
            # This is a simplified trend prediction
            # In a real system, you would use more sophisticated time series models
            
            # Create features for trend prediction
            features = {
                'current_value': current_value,
                'time_horizon': time_horizon_hours,
                'hour_of_day': datetime.now().hour,
                'day_of_week': datetime.now().weekday()
            }
            
            # Make prediction
            result = self.predict('performance', metric_type, features)
            result.prediction_type = PredictionType.PERFORMANCE
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting performance trend: {e}")
            raise
    
    def predict_resource_usage(self, resource_type: str, 
                              current_usage: float, 
                              time_horizon_hours: int = 24) -> PredictionResult:
        """Predict resource usage for a specific resource type."""
        try:
            # Create features for resource usage prediction
            features = {
                'current_usage': current_usage,
                'time_horizon': time_horizon_hours,
                'hour_of_day': datetime.now().hour,
                'day_of_week': datetime.now().weekday()
            }
            
            # Make prediction
            result = self.predict('resource_usage', resource_type, features)
            result.prediction_type = PredictionType.RESOURCE_USAGE
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting resource usage: {e}")
            raise
    
    def predict_system_health(self, current_health: float, 
                             time_horizon_hours: int = 24) -> PredictionResult:
        """Predict system health trend."""
        try:
            # Create features for system health prediction
            features = {
                'current_health': current_health,
                'time_horizon': time_horizon_hours,
                'hour_of_day': datetime.now().hour,
                'day_of_week': datetime.now().weekday()
            }
            
            # Make prediction
            result = self.predict('system_health', 'health_score', features)
            result.prediction_type = PredictionType.SYSTEM_HEALTH
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting system health: {e}")
            raise
    
    def get_model_info(self, data_type: str, target_column: str) -> Optional[PredictionModel]:
        """Get information about a trained model."""
        try:
            model_key = f"{data_type}_{target_column}"
            return self.models.get(model_key)
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return None
    
    def get_prediction_history(self, prediction_type: Optional[PredictionType] = None,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> List[PredictionResult]:
        """Get prediction history with optional filtering."""
        try:
            predictions = self.predictions.copy()
            
            # Filter by prediction type
            if prediction_type:
                predictions = [p for p in predictions if p.prediction_type == prediction_type]
            
            # Filter by time range
            if start_time:
                predictions = [p for p in predictions if p.timestamp >= start_time]
            if end_time:
                predictions = [p for p in predictions if p.timestamp <= end_time]
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error getting prediction history: {e}")
            return []
    
    def get_analytics_statistics(self) -> Dict[str, Any]:
        """Get predictive analytics statistics."""
        try:
            return {
                'total_models': len(self.models),
                'total_predictions': len(self.predictions),
                'training_data_types': list(self.training_data.keys()),
                'training_data_counts': {k: len(v) for k, v in self.training_data.items()},
                'average_prediction_time': np.mean(self.prediction_times) if self.prediction_times else 0.0,
                'average_training_time': np.mean(self.model_training_times) if self.model_training_times else 0.0,
                'retention_days': self.retention_days,
                'min_training_samples': self.min_training_samples,
                'prediction_horizon_hours': self.prediction_horizon_hours
            }
        except Exception as e:
            self.logger.error(f"Error getting analytics statistics: {e}")
            return {'error': str(e)}
    
    def _cleanup_old_training_data(self) -> None:
        """Remove training data older than retention period."""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.retention_days)
            
            for data_type, data_list in self.training_data.items():
                original_count = len(data_list)
                self.training_data[data_type] = [
                    d for d in data_list 
                    if d.get('timestamp', datetime.now()) >= cutoff_time
                ]
                removed_count = original_count - len(self.training_data[data_type])
                
                if removed_count > 0:
                    self.logger.info(f"Cleaned up {removed_count} old training data points for {data_type}")
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up old training data: {e}")
    
    def _cleanup_old_predictions(self) -> None:
        """Remove predictions older than retention period."""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.retention_days)
            original_count = len(self.predictions)
            self.predictions = [p for p in self.predictions if p.timestamp >= cutoff_time]
            removed_count = original_count - len(self.predictions)
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old predictions")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old predictions: {e}")
    
    def _retrain_models_if_needed(self) -> None:
        """Retrain models if they are older than retrain interval."""
        try:
            current_time = datetime.now()
            
            for model_key, model_info in self.models.items():
                time_since_creation = current_time - model_info.created_at
                
                if time_since_creation.total_seconds() > self.model_retrain_interval_hours * 3600:
                    # Model needs retraining
                    data_type, target_column = model_key.split('_', 1)
                    
                    try:
                        self.train_model(data_type, target_column)
                        self.logger.info(f"Retrained model {model_key}")
                    except Exception as e:
                        self.logger.error(f"Error retraining model {model_key}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error checking model retrain needs: {e}")
