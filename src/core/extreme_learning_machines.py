#!/usr/bin/env python3
"""
Extreme Learning Machines (ELMs) for Tabula Rasa Director
Implements fast learning and decision-making for meta-cognitive processes.
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
import asyncio
from scipy.linalg import pinv

from ..database.system_integration import get_system_integration
from ..database.api import Component, LogLevel
from ..core.cognitive_subsystems import LearningProgressMonitor, MetaLearningMonitor, PatternRecognitionMonitor

logger = logging.getLogger(__name__)

class ExtremeLearningMachine:
    """
    Extreme Learning Machine implementation for the Director's meta-cognitive system.
    Provides fast learning and decision-making capabilities.
    """
    
    def __init__(self, config_path: str = None, enable_monitoring: bool = True, enable_database_storage: bool = True):
        self.config_path = config_path
        self.input_size = 100  # Input feature dimension
        self.hidden_size = 200  # Hidden layer size
        self.output_size = 50   # Output dimension
        self.enable_monitoring = enable_monitoring
        self.enable_database_storage = enable_database_storage
        
        # ELM parameters
        self.hidden_weights = None  # Random hidden layer weights
        self.hidden_bias = None     # Random hidden layer bias
        self.output_weights = None  # Learned output weights
        self.activation_function = 'sigmoid'  # Activation function type
        
        # Learning parameters
        self.regularization = 0.01  # L2 regularization parameter
        self.learning_rate = 0.001  # Learning rate for online updates
        
        # Initialize monitoring systems
        if self.enable_monitoring:
            self.learning_progress_monitor = LearningProgressMonitor()
            self.meta_learning_monitor = MetaLearningMonitor()
            self.pattern_recognition_monitor = PatternRecognitionMonitor()
        
        # Initialize database integration
        if self.enable_database_storage:
            self.integration = get_system_integration()
        
        self._load_config()
        self._initialize_elm()
    
    def _load_config(self):
        """Load ELM configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.input_size = config.get('input_size', 100)
                    self.hidden_size = config.get('hidden_size', 200)
                    self.output_size = config.get('output_size', 50)
                    self.regularization = config.get('regularization', 0.01)
                    self.learning_rate = config.get('learning_rate', 0.001)
                    self.activation_function = config.get('activation_function', 'sigmoid')
        except Exception as e:
            logger.warning(f"Could not load ELM config: {e}")
    
    def _initialize_elm(self):
        """Initialize ELM with random weights and bias."""
        # Random hidden layer weights (fixed after initialization)
        self.hidden_weights = np.random.normal(0, 1, (self.input_size, self.hidden_size))
        
        # Random hidden layer bias (fixed after initialization)
        self.hidden_bias = np.random.normal(0, 1, (1, self.hidden_size))
        
        # Output weights (learned)
        self.output_weights = np.zeros((self.hidden_size, self.output_size))
        
        logger.info(f"ELM initialized: {self.input_size} -> {self.hidden_size} -> {self.output_size}")
    
    def _activation_function(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function to hidden layer output."""
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
        elif self.activation_function == 'tanh':
            return np.tanh(x)
        elif self.activation_function == 'relu':
            return np.maximum(0, x)
        elif self.activation_function == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)
        else:
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Default to sigmoid
    
    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        """
        Perform forward pass through the ELM.
        
        Args:
            input_data: Input data of shape (n_samples, input_size)
            
        Returns:
            Output predictions of shape (n_samples, output_size)
        """
        # Hidden layer computation
        hidden_input = np.dot(input_data, self.hidden_weights) + self.hidden_bias
        hidden_output = self._activation_function(hidden_input)
        
        # Output layer computation
        output = np.dot(hidden_output, self.output_weights)
        
        return output
    
    def train_batch(self, input_data: np.ndarray, target_data: np.ndarray) -> Dict[str, float]:
        """
        Train ELM on a batch of data using Moore-Penrose pseudoinverse.
        
        Args:
            input_data: Input data of shape (n_samples, input_size)
            target_data: Target data of shape (n_samples, output_size)
            
        Returns:
            Training metrics
        """
        # Forward pass to get hidden layer output
        hidden_input = np.dot(input_data, self.hidden_weights) + self.hidden_bias
        hidden_output = self._activation_function(hidden_input)
        
        # Compute output weights using regularized least squares
        # H^T * H + λI is invertible, so we can solve (H^T * H + λI) * W = H^T * T
        H = hidden_output
        T = target_data
        
        # Regularized least squares solution
        regularization_matrix = self.regularization * np.eye(self.hidden_size)
        
        try:
            # Solve for output weights
            self.output_weights = np.dot(
                pinv(np.dot(H.T, H) + regularization_matrix),
                np.dot(H.T, T)
            )
            
            # Compute training metrics
            predictions = self.forward_pass(input_data)
            mse = np.mean((predictions - target_data) ** 2)
            mae = np.mean(np.abs(predictions - target_data))
            
            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(np.sqrt(mse)),
                'training_samples': len(input_data)
            }
            
            logger.info(f"ELM trained on {len(input_data)} samples: MSE={mse:.6f}, MAE={mae:.6f}")
            return metrics
            
        except Exception as e:
            logger.error(f"ELM training failed: {e}")
            return {'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf'), 'training_samples': 0}
    
    def online_update(self, input_data: np.ndarray, target_data: np.ndarray) -> Dict[str, float]:
        """
        Perform online update of ELM weights using gradient descent.
        
        Args:
            input_data: Single input sample of shape (1, input_size)
            target_data: Single target sample of shape (1, output_size)
            
        Returns:
            Update metrics
        """
        # Forward pass
        predictions = self.forward_pass(input_data)
        
        # Compute error
        error = predictions - target_data
        
        # Compute gradients
        hidden_input = np.dot(input_data, self.hidden_weights) + self.hidden_bias
        hidden_output = self._activation_function(hidden_input)
        
        # Gradient w.r.t. output weights
        output_gradients = np.dot(hidden_output.T, error)
        
        # Update output weights
        self.output_weights -= self.learning_rate * output_gradients
        
        # Compute metrics
        mse = np.mean(error ** 2)
        mae = np.mean(np.abs(error))
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(np.sqrt(mse)),
            'update_samples': 1
        }
        
        return metrics
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained ELM.
        
        Args:
            input_data: Input data of shape (n_samples, input_size)
            
        Returns:
            Predictions of shape (n_samples, output_size)
        """
        return self.forward_pass(input_data)
    
    def get_feature_importance(self, input_data: np.ndarray) -> np.ndarray:
        """
        Compute feature importance based on hidden layer activations.
        
        Args:
            input_data: Input data of shape (n_samples, input_size)
            
        Returns:
            Feature importance scores
        """
        # Get hidden layer activations
        hidden_input = np.dot(input_data, self.hidden_weights) + self.hidden_bias
        hidden_output = self._activation_function(hidden_input)
        
        # Compute feature importance as sum of absolute output weights
        # weighted by hidden activations
        feature_importance = np.zeros(self.input_size)
        
        for i in range(self.input_size):
            # Sum of absolute output weights for features connected to input i
            feature_importance[i] = np.sum(
                np.abs(self.hidden_weights[i, :]) * np.mean(hidden_output, axis=0)
            )
        
        # Normalize to [0, 1]
        if np.max(feature_importance) > 0:
            feature_importance = feature_importance / np.max(feature_importance)
        
        return feature_importance
    
    def adapt_architecture(self, performance_metrics: Dict[str, float]) -> None:
        """
        Adapt ELM architecture based on performance metrics.
        
        Args:
            performance_metrics: Current performance metrics
        """
        # Adjust regularization based on overfitting
        if performance_metrics.get('mse', float('inf')) > 1.0:
            # Increase regularization if MSE is high
            self.regularization = min(0.1, self.regularization * 1.1)
            logger.info(f"Increased regularization to {self.regularization:.4f}")
        
        elif performance_metrics.get('mse', 0) < 0.01:
            # Decrease regularization if MSE is very low (possible overfitting)
            self.regularization = max(0.001, self.regularization * 0.9)
            logger.info(f"Decreased regularization to {self.regularization:.4f}")
        
        # Adjust learning rate based on convergence
        if performance_metrics.get('convergence_rate', 0) < 0.1:
            self.learning_rate = min(0.01, self.learning_rate * 1.05)
        elif performance_metrics.get('convergence_rate', 0) > 0.5:
            self.learning_rate = max(0.0001, self.learning_rate * 0.95)
    
    def get_elm_report(self) -> Dict:
        """Generate a report on ELM performance and configuration."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'architecture': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'activation_function': self.activation_function
            },
            'parameters': {
                'regularization': self.regularization,
                'learning_rate': self.learning_rate
            },
            'weights_info': {
                'hidden_weights_shape': self.hidden_weights.shape if self.hidden_weights is not None else None,
                'output_weights_shape': self.output_weights.shape if self.output_weights is not None else None,
                'hidden_weights_mean': float(np.mean(self.hidden_weights)) if self.hidden_weights is not None else None,
                'output_weights_mean': float(np.mean(self.output_weights)) if self.output_weights is not None else None
            }
        }
        
        return report
    
    def save_elm_state(self, filepath: str = None) -> None:
        """Save ELM state to file."""
        if filepath is None:
            # Database-only mode: Skip file-based state saving
            return
        
        try:
            elm_state = {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'hidden_weights': self.hidden_weights.tolist() if self.hidden_weights is not None else None,
                'hidden_bias': self.hidden_bias.tolist() if self.hidden_bias is not None else None,
                'output_weights': self.output_weights.tolist() if self.output_weights is not None else None,
                'activation_function': self.activation_function,
                'regularization': self.regularization,
                'learning_rate': self.learning_rate,
                'timestamp': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(elm_state, f, indent=2)
            
            logger.info(f"ELM state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save ELM state: {e}")
    
    def load_elm_state(self, filepath: str) -> bool:
        """Load ELM state from file."""
        try:
            with open(filepath, 'r') as f:
                elm_state = json.load(f)
            
            self.input_size = elm_state['input_size']
            self.hidden_size = elm_state['hidden_size']
            self.output_size = elm_state['output_size']
            self.hidden_weights = np.array(elm_state['hidden_weights']) if elm_state['hidden_weights'] else None
            self.hidden_bias = np.array(elm_state['hidden_bias']) if elm_state['hidden_bias'] else None
            self.output_weights = np.array(elm_state['output_weights']) if elm_state['output_weights'] else None
            self.activation_function = elm_state['activation_function']
            self.regularization = elm_state['regularization']
            self.learning_rate = elm_state['learning_rate']
            
            logger.info(f"ELM state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ELM state: {e}")
            return False
    
    async def enhanced_train_batch(self, input_data: np.ndarray, target_data: np.ndarray, 
                                 context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Enhanced batch training with monitoring and database integration.
        
        Args:
            input_data: Input data of shape (n_samples, input_size)
            target_data: Target data of shape (n_samples, output_size)
            context: Additional context information
            
        Returns:
            Training metrics
        """
        start_time = datetime.now()
        
        try:
            # Perform standard training
            metrics = self.train_batch(input_data, target_data)
            
            # Update monitoring systems
            if self.enable_monitoring:
                await self._update_training_monitoring(input_data, target_data, metrics, context)
            
            # Store training data
            if self.enable_database_storage:
                await self._store_training_data(input_data, target_data, metrics, context, start_time)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Enhanced ELM training failed: {e}")
            return {'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf'), 'training_samples': 0}
    
    async def enhanced_online_update(self, input_data: np.ndarray, target_data: np.ndarray, 
                                   context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Enhanced online update with monitoring and database integration.
        
        Args:
            input_data: Single input sample of shape (1, input_size)
            target_data: Single target sample of shape (1, output_size)
            context: Additional context information
            
        Returns:
            Update metrics
        """
        start_time = datetime.now()
        
        try:
            # Perform standard online update
            metrics = self.online_update(input_data, target_data)
            
            # Update monitoring systems
            if self.enable_monitoring:
                await self._update_online_monitoring(input_data, target_data, metrics, context)
            
            # Store update data
            if self.enable_database_storage:
                await self._store_online_update_data(input_data, target_data, metrics, context, start_time)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Enhanced ELM online update failed: {e}")
            return {'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf'), 'update_samples': 0}
    
    async def _update_training_monitoring(self, input_data: np.ndarray, target_data: np.ndarray, 
                                        metrics: Dict[str, float], context: Optional[Dict[str, Any]]):
        """Update cognitive monitoring systems for training."""
        try:
            if not self.enable_monitoring:
                return
            
            # Calculate ELM-specific metrics
            elm_metrics = self._calculate_elm_metrics(input_data, target_data, metrics)
            
            # Update learning progress monitor
            if hasattr(self, 'learning_progress_monitor'):
                learning_data = {
                    'mse': metrics['mse'],
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'training_samples': metrics['training_samples'],
                    'elm_performance': elm_metrics['elm_performance'],
                    'learning_efficiency': elm_metrics['learning_efficiency'],
                    'context': context
                }
                self.learning_progress_monitor.update_metrics(learning_data)
            
            # Update meta-learning monitor
            if hasattr(self, 'meta_learning_monitor'):
                meta_learning_data = {
                    'elm_adaptation': elm_metrics['elm_adaptation'],
                    'pattern_learning': elm_metrics['pattern_learning'],
                    'context': context
                }
                self.meta_learning_monitor.update_metrics(meta_learning_data)
            
            # Update pattern recognition monitor
            if hasattr(self, 'pattern_recognition_monitor'):
                pattern_data = {
                    'feature_importance': elm_metrics['feature_importance'],
                    'activation_patterns': elm_metrics['activation_patterns'],
                    'context': context
                }
                self.pattern_recognition_monitor.update_metrics(pattern_data)
            
        except Exception as e:
            logger.error(f"Failed to update training monitoring: {e}")
    
    async def _update_online_monitoring(self, input_data: np.ndarray, target_data: np.ndarray, 
                                      metrics: Dict[str, float], context: Optional[Dict[str, Any]]):
        """Update cognitive monitoring systems for online updates."""
        try:
            if not self.enable_monitoring:
                return
            
            # Calculate online update metrics
            online_metrics = self._calculate_online_metrics(input_data, target_data, metrics)
            
            # Update learning progress monitor
            if hasattr(self, 'learning_progress_monitor'):
                learning_data = {
                    'mse': metrics['mse'],
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'update_samples': metrics['update_samples'],
                    'online_learning_rate': online_metrics['online_learning_rate'],
                    'adaptation_speed': online_metrics['adaptation_speed'],
                    'context': context
                }
                self.learning_progress_monitor.update_metrics(learning_data)
            
        except Exception as e:
            logger.error(f"Failed to update online monitoring: {e}")
    
    async def _store_training_data(self, input_data: np.ndarray, target_data: np.ndarray, 
                                 metrics: Dict[str, float], context: Optional[Dict[str, Any]], 
                                 start_time: datetime):
        """Store training data in database."""
        try:
            if not self.enable_database_storage or not hasattr(self, 'integration'):
                return
            
            # Calculate ELM metrics
            elm_metrics = self._calculate_elm_metrics(input_data, target_data, metrics)
            
            # Create operation record
            operation_data = {
                'operation_type': 'elm_batch_training',
                'input_shape': input_data.shape,
                'target_shape': target_data.shape,
                'training_samples': metrics['training_samples'],
                'mse': metrics['mse'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'elm_performance': elm_metrics['elm_performance'],
                'learning_efficiency': elm_metrics['learning_efficiency'],
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'context': context or {}
            }
            
            await self.integration.log_system_event(
                LogLevel.INFO,
                Component.DIRECTOR,
                f"ELM batch training: {metrics['training_samples']} samples",
                operation_data,
                "elm_training"
            )
            
        except Exception as e:
            logger.error(f"Failed to store training data: {e}")
    
    async def _store_online_update_data(self, input_data: np.ndarray, target_data: np.ndarray, 
                                      metrics: Dict[str, float], context: Optional[Dict[str, Any]], 
                                      start_time: datetime):
        """Store online update data in database."""
        try:
            if not self.enable_database_storage or not hasattr(self, 'integration'):
                return
            
            # Calculate online metrics
            online_metrics = self._calculate_online_metrics(input_data, target_data, metrics)
            
            # Create operation record
            operation_data = {
                'operation_type': 'elm_online_update',
                'input_shape': input_data.shape,
                'target_shape': target_data.shape,
                'update_samples': metrics['update_samples'],
                'mse': metrics['mse'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'online_learning_rate': online_metrics['online_learning_rate'],
                'adaptation_speed': online_metrics['adaptation_speed'],
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'context': context or {}
            }
            
            await self.integration.log_system_event(
                LogLevel.INFO,
                Component.DIRECTOR,
                f"ELM online update: {metrics['update_samples']} samples",
                operation_data,
                "elm_online_update"
            )
            
        except Exception as e:
            logger.error(f"Failed to store online update data: {e}")
    
    def _calculate_elm_metrics(self, input_data: np.ndarray, target_data: np.ndarray, 
                             metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate ELM-specific performance metrics."""
        try:
            # Calculate ELM performance (based on training metrics)
            elm_performance = 1.0 / (1.0 + metrics['mse']) if metrics['mse'] > 0 else 1.0
            
            # Calculate learning efficiency (based on sample count and performance)
            learning_efficiency = elm_performance / max(1.0, metrics['training_samples'] / 100.0)
            
            # Calculate ELM adaptation (based on regularization and learning rate)
            elm_adaptation = min(1.0, (self.regularization + self.learning_rate) * 10)
            
            # Calculate pattern learning (based on feature importance)
            feature_importance = self.get_feature_importance(input_data)
            pattern_learning = float(np.mean(feature_importance))
            
            # Calculate activation patterns (based on hidden layer activations)
            hidden_input = np.dot(input_data, self.hidden_weights) + self.hidden_bias
            hidden_output = self._activation_function(hidden_input)
            activation_patterns = float(np.std(hidden_output))
            
            return {
                'elm_performance': elm_performance,
                'learning_efficiency': learning_efficiency,
                'elm_adaptation': elm_adaptation,
                'pattern_learning': pattern_learning,
                'feature_importance': pattern_learning,  # Use pattern_learning as proxy
                'activation_patterns': activation_patterns
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate ELM metrics: {e}")
            return {
                'elm_performance': 0.0,
                'learning_efficiency': 0.0,
                'elm_adaptation': 0.0,
                'pattern_learning': 0.0,
                'feature_importance': 0.0,
                'activation_patterns': 0.0
            }
    
    def _calculate_online_metrics(self, input_data: np.ndarray, target_data: np.ndarray, 
                                metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate online update performance metrics."""
        try:
            # Calculate online learning rate (scaled learning rate)
            online_learning_rate = min(1.0, self.learning_rate * 1000)
            
            # Calculate adaptation speed (based on error reduction)
            adaptation_speed = 1.0 / (1.0 + metrics['mse']) if metrics['mse'] > 0 else 1.0
            
            return {
                'online_learning_rate': online_learning_rate,
                'adaptation_speed': adaptation_speed
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate online metrics: {e}")
            return {
                'online_learning_rate': 0.0,
                'adaptation_speed': 0.0
            }
    
    async def get_enhanced_report(self) -> Dict[str, Any]:
        """Get enhanced ELM report with monitoring data."""
        try:
            # Get base report
            base_report = self.get_elm_report()
            
            # Add monitoring data
            if self.enable_monitoring:
                base_report['monitoring_data'] = {
                    'learning_progress': getattr(self.learning_progress_monitor, 'metrics', {}),
                    'meta_learning': getattr(self.meta_learning_monitor, 'metrics', {}),
                    'pattern_recognition': getattr(self.pattern_recognition_monitor, 'metrics', {})
                }
            
            # Add database integration status
            base_report['database_integration'] = {
                'enabled': self.enable_database_storage,
                'integration_available': hasattr(self, 'integration')
            }
            
            return base_report
            
        except Exception as e:
            logger.error(f"Failed to get enhanced report: {e}")
            return self.get_elm_report()


class DirectorELMEnsemble:
    """
    Ensemble of ELMs for the Director's meta-cognitive decision-making.
    Each ELM handles different aspects of the decision-making process.
    """
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.elms = {}  # Dictionary of ELMs for different tasks
        self.ensemble_weights = {}  # Weights for combining ELM outputs
        self.task_mapping = {}  # Mapping of tasks to ELMs
        
        self._load_config()
        self._initialize_ensemble()
    
    def _load_config(self):
        """Load ELM ensemble configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.task_mapping = config.get('task_mapping', {})
        except Exception as e:
            logger.warning(f"Could not load ELM ensemble config: {e}")
    
    def _initialize_ensemble(self):
        """Initialize ensemble of ELMs for different Director tasks."""
        # Initialize ELMs for different meta-cognitive tasks
        self.elms = {
            'narrative_engine': ExtremeLearningMachine(),
            'affective_agent': ExtremeLearningMachine(),
            'drive_agent': ExtremeLearningMachine(),
            'social_simulant': ExtremeLearningMachine(),
            'strategy_planner': ExtremeLearningMachine(),
            'resource_optimizer': ExtremeLearningMachine()
        }
        
        # Initialize ensemble weights
        for task_name in self.elms.keys():
            self.ensemble_weights[task_name] = 1.0 / len(self.elms)
        
        logger.info(f"Initialized ELM ensemble with {len(self.elms)} ELMs")
    
    def train_ensemble(self, training_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Dict[str, float]]:
        """
        Train the entire ELM ensemble.
        
        Args:
            training_data: Dictionary mapping task names to (input, target) tuples
            
        Returns:
            Training metrics for each ELM
        """
        metrics = {}
        
        for task_name, (input_data, target_data) in training_data.items():
            if task_name in self.elms:
                elm_metrics = self.elms[task_name].train_batch(input_data, target_data)
                metrics[task_name] = elm_metrics
                logger.info(f"Trained {task_name} ELM: MSE={elm_metrics['mse']:.6f}")
        
        return metrics
    
    def predict_ensemble(self, input_data: np.ndarray, task_name: str = None) -> np.ndarray:
        """
        Make predictions using the ELM ensemble.
        
        Args:
            input_data: Input data
            task_name: Specific task to predict (if None, uses ensemble)
            
        Returns:
            Ensemble predictions
        """
        if task_name and task_name in self.elms:
            return self.elms[task_name].predict(input_data)
        
        # Ensemble prediction
        predictions = []
        weights = []
        
        for elm_name, elm in self.elms.items():
            pred = elm.predict(input_data)
            predictions.append(pred)
            weights.append(self.ensemble_weights[elm_name])
        
        # Weighted average of predictions
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize weights
        
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def update_ensemble_weights(self, performance_metrics: Dict[str, float]) -> None:
        """
        Update ensemble weights based on individual ELM performance.
        
        Args:
            performance_metrics: Performance metrics for each ELM
        """
        # Convert MSE to performance scores (lower MSE = higher performance)
        performance_scores = {}
        for task_name, metrics in performance_metrics.items():
            if 'mse' in metrics:
                # Convert MSE to performance score (inverse relationship)
                performance_scores[task_name] = 1.0 / (1.0 + metrics['mse'])
            else:
                performance_scores[task_name] = 0.5  # Default score
        
        # Update ensemble weights based on performance
        total_performance = sum(performance_scores.values())
        if total_performance > 0:
            for task_name in self.ensemble_weights.keys():
                if task_name in performance_scores:
                    self.ensemble_weights[task_name] = performance_scores[task_name] / total_performance
        
        logger.info(f"Updated ensemble weights: {self.ensemble_weights}")
    
    def get_ensemble_report(self) -> Dict:
        """Generate a comprehensive report on the ELM ensemble."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'ensemble_size': len(self.elms),
            'ensemble_weights': self.ensemble_weights,
            'elm_reports': {}
        }
        
        for task_name, elm in self.elms.items():
            report['elm_reports'][task_name] = elm.get_elm_report()
        
        return report
