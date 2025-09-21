#!/usr/bin/env python3
"""
Residual Learning for Tabula Rasa Governor
Implements residual connections to improve gradient flow and learning efficiency.
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
import asyncio

from ..database.system_integration import get_system_integration
from ..database.api import Component, LogLevel
from ..core.cognitive_subsystems import LearningProgressMonitor, PatternRecognitionMonitor

logger = logging.getLogger(__name__)

class ResidualLearningSystem:
    """
    Residual Learning implementation for the Governor's decision-making system.
    Uses residual connections to improve gradient flow and prevent vanishing gradients.
    """
    
    def __init__(self, config_path: str = None, enable_monitoring: bool = True, enable_database_storage: bool = True):
        self.config_path = config_path
        self.residual_layers = {}  # Residual layer configurations
        self.skip_connections = {}  # Skip connection weights
        self.layer_outputs = {}  # Cached layer outputs for residual computation
        self.learning_rate = 0.001  # Learning rate for residual updates
        self.residual_strength = 0.1  # Strength of residual connections
        self.enable_monitoring = enable_monitoring
        self.enable_database_storage = enable_database_storage
        
        # Initialize monitoring systems
        if self.enable_monitoring:
            self.learning_progress_monitor = LearningProgressMonitor()
            self.pattern_recognition_monitor = PatternRecognitionMonitor()
        
        # Initialize database integration
        if self.enable_database_storage:
            self.integration = get_system_integration()
        
        self._load_config()
        self._initialize_residual_system()
    
    def _load_config(self):
        """Load residual learning configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.learning_rate = config.get('learning_rate', 0.001)
                    self.residual_strength = config.get('residual_strength', 0.1)
        except Exception as e:
            logger.warning(f"Could not load residual config: {e}")
    
    def _initialize_residual_system(self):
        """Initialize residual learning system for Governor components."""
        # Initialize residual layers for different Governor subsystems
        self.residual_layers = {
            'resource_allocator': {
                'input_size': 50,
                'hidden_size': 100,
                'output_size': 25,
                'residual_connections': [0, 2, 4]  # Which layers have skip connections
            },
            'performance_monitor': {
                'input_size': 30,
                'hidden_size': 60,
                'output_size': 15,
                'residual_connections': [0, 1, 3]
            },
            'decision_engine': {
                'input_size': 40,
                'hidden_size': 80,
                'output_size': 20,
                'residual_connections': [0, 2, 3, 5]
            },
            'action_selector': {
                'input_size': 35,
                'hidden_size': 70,
                'output_size': 10,
                'residual_connections': [0, 1, 2]
            }
        }
        
        # Initialize skip connection weights
        for layer_name, config in self.residual_layers.items():
            self.skip_connections[layer_name] = {}
            for conn_idx in config['residual_connections']:
                # Skip connection weight matrix
                self.skip_connections[layer_name][conn_idx] = np.random.normal(
                    0, 0.1, (config['input_size'], config['output_size'])
                )
    
    def apply_residual_connection(self, layer_name: str, layer_idx: int, 
                                input_data: np.ndarray, output_data: np.ndarray) -> np.ndarray:
        """
        Apply residual connection to a layer output.
        
        Args:
            layer_name: Name of the residual layer
            layer_idx: Index of the current layer
            input_data: Input to the layer
            output_data: Output from the layer
            
        Returns:
            Residual-enhanced output
        """
        if (layer_name in self.skip_connections and 
            layer_idx in self.skip_connections[layer_name]):
            
            # Get skip connection weights
            skip_weights = self.skip_connections[layer_name][layer_idx]
            
            # Ensure dimensions match
            if input_data.shape[1] == skip_weights.shape[0] and output_data.shape[1] == skip_weights.shape[1]:
                # Apply skip connection: output = output + (input * skip_weights)
                skip_output = np.dot(input_data, skip_weights)
                residual_output = output_data + self.residual_strength * skip_output
                
                logger.debug(f"Applied residual connection to {layer_name}[{layer_idx}]")
                return residual_output
            else:
                logger.warning(f"Dimension mismatch in residual connection for {layer_name}[{layer_idx}]")
                return output_data
        else:
            return output_data
    
    def compute_residual_gradients(self, layer_name: str, layer_idx: int,
                                 input_data: np.ndarray, output_gradients: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute gradients for residual connections.
        
        Args:
            layer_name: Name of the residual layer
            layer_idx: Index of the current layer
            input_data: Input to the layer
            output_gradients: Gradients from the output
            
        Returns:
            Dictionary of gradients for skip connection weights
        """
        gradients = {}
        
        if (layer_name in self.skip_connections and 
            layer_idx in self.skip_connections[layer_name]):
            
            skip_weights = self.skip_connections[layer_name][layer_idx]
            
            # Gradient w.r.t. skip connection weights
            # dL/dW_skip = dL/doutput * doutput/dW_skip
            # doutput/dW_skip = residual_strength * input
            skip_weight_gradients = self.residual_strength * np.dot(input_data.T, output_gradients)
            gradients[f'skip_weights_{layer_idx}'] = skip_weight_gradients
            
            # Gradient w.r.t. input (for backpropagation)
            input_gradients = self.residual_strength * np.dot(output_gradients, skip_weights.T)
            gradients['input_gradients'] = input_gradients
            
            logger.debug(f"Computed residual gradients for {layer_name}[{layer_idx}]")
        
        return gradients
    
    def update_residual_weights(self, layer_name: str, gradients: Dict[str, np.ndarray]) -> None:
        """
        Update residual connection weights using computed gradients.
        
        Args:
            layer_name: Name of the residual layer
            gradients: Computed gradients for skip connections
        """
        if layer_name not in self.skip_connections:
            return
        
        for grad_key, grad_values in gradients.items():
            if grad_key.startswith('skip_weights_'):
                layer_idx = int(grad_key.split('_')[-1])
                
                if layer_idx in self.skip_connections[layer_name]:
                    # Update skip connection weights
                    self.skip_connections[layer_name][layer_idx] -= self.learning_rate * grad_values
                    
                    logger.debug(f"Updated skip weights for {layer_name}[{layer_idx}]")
    
    def forward_pass(self, layer_name: str, input_data: np.ndarray, 
                    layer_weights: np.ndarray, layer_bias: np.ndarray) -> np.ndarray:
        """
        Perform forward pass with residual connections.
        
        Args:
            layer_name: Name of the residual layer
            input_data: Input data
            layer_weights: Layer weights
            layer_bias: Layer bias
            
        Returns:
            Output with residual connections applied
        """
        # Standard forward pass
        output = np.dot(input_data, layer_weights) + layer_bias
        
        # Apply residual connections
        residual_output = self.apply_residual_connection(
            layer_name, 0, input_data, output  # Assuming layer_idx=0 for simplicity
        )
        
        # Cache output for gradient computation
        self.layer_outputs[f"{layer_name}_output"] = residual_output
        
        return residual_output
    
    def backward_pass(self, layer_name: str, input_data: np.ndarray,
                     output_gradients: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform backward pass with residual gradients.
        
        Args:
            layer_name: Name of the residual layer
            input_data: Input data
            output_gradients: Gradients from the output
            
        Returns:
            Dictionary of gradients including residual terms
        """
        gradients = {}
        
        # Compute residual gradients
        residual_gradients = self.compute_residual_gradients(
            layer_name, 0, input_data, output_gradients
        )
        
        # Add residual gradients to main gradients
        gradients.update(residual_gradients)
        
        return gradients
    
    def optimize_residual_architecture(self, performance_metrics: Dict[str, float]) -> None:
        """
        Optimize residual architecture based on performance metrics.
        
        Args:
            performance_metrics: Current performance metrics
        """
        # Adjust residual strength based on performance
        if performance_metrics.get('gradient_flow', 0) < 0.5:
            # Increase residual strength if gradient flow is poor
            self.residual_strength = min(0.5, self.residual_strength * 1.1)
            logger.info(f"Increased residual strength to {self.residual_strength:.3f}")
        
        elif performance_metrics.get('gradient_flow', 0) > 0.8:
            # Decrease residual strength if gradient flow is too strong
            self.residual_strength = max(0.05, self.residual_strength * 0.9)
            logger.info(f"Decreased residual strength to {self.residual_strength:.3f}")
        
        # Adjust learning rate based on convergence
        if performance_metrics.get('convergence_rate', 0) < 0.1:
            self.learning_rate = min(0.01, self.learning_rate * 1.05)
        elif performance_metrics.get('convergence_rate', 0) > 0.5:
            self.learning_rate = max(0.0001, self.learning_rate * 0.95)
    
    def get_residual_report(self) -> Dict:
        """Generate a report on residual learning performance."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'residual_layers': len(self.residual_layers),
            'total_skip_connections': sum(len(conns) for conns in self.skip_connections.values()),
            'residual_strength': self.residual_strength,
            'learning_rate': self.learning_rate,
            'layer_performance': {}
        }
        
        for layer_name, config in self.residual_layers.items():
            report['layer_performance'][layer_name] = {
                'residual_connections': len(config['residual_connections']),
                'input_size': config['input_size'],
                'output_size': config['output_size']
            }
        
        return report
    
    def save_residual_state(self, filepath: str = None) -> None:
        """Save residual learning state to file."""
        if filepath is None:
            # Database-only mode: Skip file-based state saving
            return
        
        try:
            residual_state = {
                'residual_layers': self.residual_layers,
                'skip_connections': {k: {kk: vv.tolist() for kk, vv in v.items()} 
                                   for k, v in self.skip_connections.items()},
                'residual_strength': self.residual_strength,
                'learning_rate': self.learning_rate,
                'timestamp': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(residual_state, f, indent=2)
            
            logger.info(f"Residual state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save residual state: {e}")
    
    def load_residual_state(self, filepath: str) -> bool:
        """Load residual learning state from file."""
        try:
            with open(filepath, 'r') as f:
                residual_state = json.load(f)
            
            self.residual_layers = residual_state['residual_layers']
            self.skip_connections = {k: {kk: np.array(vv) for kk, vv in v.items()} 
                                   for k, v in residual_state['skip_connections'].items()}
            self.residual_strength = residual_state.get('residual_strength', 0.1)
            self.learning_rate = residual_state.get('learning_rate', 0.001)
            
            logger.info(f"Residual state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load residual state: {e}")
            return False
    
    async def enhanced_forward_pass(self, layer_name: str, input_data: np.ndarray, 
                                  layer_weights: np.ndarray, layer_bias: np.ndarray,
                                  context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Enhanced forward pass with monitoring and database integration.
        
        Args:
            layer_name: Name of the residual layer
            input_data: Input data
            layer_weights: Layer weights
            layer_bias: Layer bias
            context: Additional context information
            
        Returns:
            Output with residual connections applied
        """
        start_time = datetime.now()
        
        try:
            # Perform standard forward pass
            output = self.forward_pass(layer_name, input_data, layer_weights, layer_bias)
            
            # Update monitoring systems
            if self.enable_monitoring:
                await self._update_monitoring_systems(layer_name, input_data, output, context)
            
            # Store operation data
            if self.enable_database_storage:
                await self._store_forward_pass_data(layer_name, input_data, output, context, start_time)
            
            return output
            
        except Exception as e:
            logger.error(f"Enhanced forward pass failed: {e}")
            # Fallback to standard forward pass
            return self.forward_pass(layer_name, input_data, layer_weights, layer_bias)
    
    async def enhanced_backward_pass(self, layer_name: str, input_data: np.ndarray,
                                   output_gradients: np.ndarray,
                                   context: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """
        Enhanced backward pass with monitoring and database integration.
        
        Args:
            layer_name: Name of the residual layer
            input_data: Input data
            output_gradients: Gradients from the output
            context: Additional context information
            
        Returns:
            Dictionary of gradients including residual terms
        """
        start_time = datetime.now()
        
        try:
            # Perform standard backward pass
            gradients = self.backward_pass(layer_name, input_data, output_gradients)
            
            # Update monitoring systems
            if self.enable_monitoring:
                await self._update_gradient_monitoring(layer_name, gradients, context)
            
            # Store operation data
            if self.enable_database_storage:
                await self._store_backward_pass_data(layer_name, gradients, context, start_time)
            
            return gradients
            
        except Exception as e:
            logger.error(f"Enhanced backward pass failed: {e}")
            # Fallback to standard backward pass
            return self.backward_pass(layer_name, input_data, output_gradients)
    
    async def _update_monitoring_systems(self, layer_name: str, input_data: np.ndarray, 
                                       output: np.ndarray, context: Optional[Dict[str, Any]]):
        """Update cognitive monitoring systems."""
        try:
            if not self.enable_monitoring:
                return
            
            # Calculate residual learning metrics
            metrics = self._calculate_residual_metrics(layer_name, input_data, output)
            
            # Update learning progress monitor
            if hasattr(self, 'learning_progress_monitor'):
                learning_data = {
                    'gradient_flow': metrics['gradient_flow'],
                    'residual_effectiveness': metrics['residual_effectiveness'],
                    'layer_performance': metrics['layer_performance'],
                    'context': context
                }
                self.learning_progress_monitor.update_metrics(learning_data)
            
            # Update pattern recognition monitor
            if hasattr(self, 'pattern_recognition_monitor'):
                pattern_data = {
                    'activation_patterns': metrics['activation_patterns'],
                    'residual_patterns': metrics['residual_patterns'],
                    'context': context
                }
                self.pattern_recognition_monitor.update_metrics(pattern_data)
            
        except Exception as e:
            logger.error(f"Failed to update monitoring systems: {e}")
    
    async def _update_gradient_monitoring(self, layer_name: str, gradients: Dict[str, np.ndarray], 
                                        context: Optional[Dict[str, Any]]):
        """Update gradient monitoring systems."""
        try:
            if not self.enable_monitoring:
                return
            
            # Calculate gradient metrics
            gradient_metrics = self._calculate_gradient_metrics(layer_name, gradients)
            
            # Update learning progress monitor
            if hasattr(self, 'learning_progress_monitor'):
                learning_data = {
                    'gradient_flow': gradient_metrics['gradient_flow'],
                    'gradient_stability': gradient_metrics['gradient_stability'],
                    'residual_gradient_effectiveness': gradient_metrics['residual_gradient_effectiveness'],
                    'context': context
                }
                self.learning_progress_monitor.update_metrics(learning_data)
            
        except Exception as e:
            logger.error(f"Failed to update gradient monitoring: {e}")
    
    async def _store_forward_pass_data(self, layer_name: str, input_data: np.ndarray, 
                                     output: np.ndarray, context: Optional[Dict[str, Any]], 
                                     start_time: datetime):
        """Store forward pass data in database."""
        try:
            if not self.enable_database_storage or not hasattr(self, 'integration'):
                return
            
            # Calculate metrics
            metrics = self._calculate_residual_metrics(layer_name, input_data, output)
            
            # Create operation record
            operation_data = {
                'operation_type': 'residual_forward_pass',
                'layer_name': layer_name,
                'input_shape': input_data.shape,
                'output_shape': output.shape,
                'gradient_flow': metrics['gradient_flow'],
                'residual_effectiveness': metrics['residual_effectiveness'],
                'layer_performance': metrics['layer_performance'],
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'context': context or {}
            }
            
            await self.integration.log_system_event(
                LogLevel.INFO,
                Component.GOVERNOR,
                f"Residual forward pass: {layer_name}",
                operation_data,
                "residual_learning"
            )
            
        except Exception as e:
            logger.error(f"Failed to store forward pass data: {e}")
    
    async def _store_backward_pass_data(self, layer_name: str, gradients: Dict[str, np.ndarray], 
                                      context: Optional[Dict[str, Any]], start_time: datetime):
        """Store backward pass data in database."""
        try:
            if not self.enable_database_storage or not hasattr(self, 'integration'):
                return
            
            # Calculate gradient metrics
            gradient_metrics = self._calculate_gradient_metrics(layer_name, gradients)
            
            # Create operation record
            operation_data = {
                'operation_type': 'residual_backward_pass',
                'layer_name': layer_name,
                'gradient_count': len(gradients),
                'gradient_flow': gradient_metrics['gradient_flow'],
                'gradient_stability': gradient_metrics['gradient_stability'],
                'residual_gradient_effectiveness': gradient_metrics['residual_gradient_effectiveness'],
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'context': context or {}
            }
            
            await self.integration.log_system_event(
                LogLevel.INFO,
                Component.GOVERNOR,
                f"Residual backward pass: {layer_name}",
                operation_data,
                "residual_learning"
            )
            
        except Exception as e:
            logger.error(f"Failed to store backward pass data: {e}")
    
    def _calculate_residual_metrics(self, layer_name: str, input_data: np.ndarray, 
                                  output: np.ndarray) -> Dict[str, float]:
        """Calculate residual learning performance metrics."""
        try:
            # Calculate gradient flow (based on output variance)
            gradient_flow = float(np.var(output))
            
            # Calculate residual effectiveness (based on skip connections)
            residual_effectiveness = 0.0
            if layer_name in self.skip_connections:
                skip_weights = self.skip_connections[layer_name]
                if skip_weights:
                    # Calculate effectiveness based on skip connection weights
                    total_skip_strength = sum(np.mean(np.abs(weights)) for weights in skip_weights.values())
                    residual_effectiveness = min(1.0, total_skip_strength / len(skip_weights))
            
            # Calculate layer performance (based on output quality)
            layer_performance = min(1.0, 1.0 / (1.0 + np.mean(np.abs(output))))
            
            # Calculate activation patterns (based on output distribution)
            activation_patterns = float(np.std(output))
            
            # Calculate residual patterns (based on skip connection usage)
            residual_patterns = 0.0
            if layer_name in self.residual_layers:
                residual_connections = self.residual_layers[layer_name]['residual_connections']
                residual_patterns = len(residual_connections) / 10.0  # Normalize by max expected connections
            
            return {
                'gradient_flow': gradient_flow,
                'residual_effectiveness': residual_effectiveness,
                'layer_performance': layer_performance,
                'activation_patterns': activation_patterns,
                'residual_patterns': residual_patterns
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate residual metrics: {e}")
            return {
                'gradient_flow': 0.0,
                'residual_effectiveness': 0.0,
                'layer_performance': 0.0,
                'activation_patterns': 0.0,
                'residual_patterns': 0.0
            }
    
    def _calculate_gradient_metrics(self, layer_name: str, gradients: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate gradient performance metrics."""
        try:
            # Calculate gradient flow (based on gradient variance)
            gradient_flow = 0.0
            if gradients:
                gradient_flow = float(np.mean([np.var(grad) for grad in gradients.values()]))
            
            # Calculate gradient stability (based on gradient consistency)
            gradient_stability = 0.0
            if gradients:
                gradient_means = [np.mean(np.abs(grad)) for grad in gradients.values()]
                gradient_stability = 1.0 - float(np.std(gradient_means)) if gradient_means else 0.0
            
            # Calculate residual gradient effectiveness
            residual_gradient_effectiveness = 0.0
            if 'input_gradients' in gradients:
                input_grad_norm = float(np.linalg.norm(gradients['input_gradients']))
                residual_gradient_effectiveness = min(1.0, input_grad_norm)
            
            return {
                'gradient_flow': gradient_flow,
                'gradient_stability': gradient_stability,
                'residual_gradient_effectiveness': residual_gradient_effectiveness
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate gradient metrics: {e}")
            return {
                'gradient_flow': 0.0,
                'gradient_stability': 0.0,
                'residual_gradient_effectiveness': 0.0
            }
    
    async def get_enhanced_report(self) -> Dict[str, Any]:
        """Get enhanced residual learning report with monitoring data."""
        try:
            # Get base report
            base_report = self.get_residual_report()
            
            # Add monitoring data
            if self.enable_monitoring:
                base_report['monitoring_data'] = {
                    'learning_progress': getattr(self.learning_progress_monitor, 'metrics', {}),
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
            return self.get_residual_report()
