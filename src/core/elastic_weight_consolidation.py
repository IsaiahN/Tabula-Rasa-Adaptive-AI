#!/usr/bin/env python3
"""
Elastic Weight Consolidation (EWC) for Tabula Rasa Architect
Prevents catastrophic forgetting during system evolution by preserving important weights.
"""
import sys
sys.dont_write_bytecode = True

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import logging
import asyncio

from ..database.system_integration import get_system_integration
from ..database.api import Component, LogLevel
from ..core.cognitive_subsystems import LearningProgressMonitor, MetaLearningMonitor

logger = logging.getLogger(__name__)

class ElasticWeightConsolidation:
    """
    EWC implementation for the Architect system to prevent catastrophic forgetting.
    Maintains Fisher Information Matrix and importance weights for critical parameters.
    """
    
    def __init__(self, config_path: str = "data/config/ewc_config.json", enable_monitoring: bool = True, enable_database_storage: bool = True):
        self.config_path = config_path
        self.fisher_info: Dict[str, np.ndarray] = {}  # Fisher Information Matrix
        self.importance_weights: Dict[str, np.ndarray] = {}  # Importance weights for each parameter
        self.parameter_history: Dict[str, np.ndarray] = {}  # Historical parameter values
        self.consolidation_threshold = 0.1  # Threshold for parameter importance
        self.learning_rate = 0.01  # EWC learning rate
        self.enable_monitoring = enable_monitoring
        self.enable_database_storage = enable_database_storage
        
        # Initialize monitoring systems
        if self.enable_monitoring:
            self.learning_progress_monitor = LearningProgressMonitor()
            self.meta_learning_monitor = MetaLearningMonitor()
        
        # Initialize database integration
        if self.enable_database_storage:
            self.integration = get_system_integration()
        
        self._load_config()
        self._initialize_ewc()
    
    def _load_config(self) -> None:
        """Load EWC configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.consolidation_threshold = config.get('consolidation_threshold', 0.1)
                    self.learning_rate = config.get('learning_rate', 0.01)
        except Exception as e:
            logger.warning(f"Could not load EWC config: {e}")
    
    def _initialize_ewc(self) -> None:
        """Initialize EWC data structures."""
        self.fisher_info = {
            'action_selection_weights': np.zeros((100, 100)),  # Example size
            'pattern_recognition_weights': np.zeros((50, 50)),
            'memory_consolidation_weights': np.zeros((30, 30)),
            'meta_learning_weights': np.zeros((40, 40))
        }
        
        self.importance_weights = {
            'action_selection_weights': np.ones((100, 100)) * 0.1,
            'pattern_recognition_weights': np.ones((50, 50)) * 0.1,
            'memory_consolidation_weights': np.ones((30, 30)) * 0.1,
            'meta_learning_weights': np.ones((40, 40)) * 0.1
        }

    def _normalize_parameters(self, parameters: Optional[Union[Dict[str, Any], Any]]) -> Dict[str, np.ndarray]:
        """
        Normalize various parameter/gradient representations into a dict of numpy arrays.

        Accepts:
          - None -> {}
          - dict[str, array-like or tensor] -> {str: np.ndarray}
          - single array-like or tensor -> {'param_0': np.ndarray}

        This helper avoids runtime errors when upstream systems pass torch.Tensors,
        lists, tuples, or numpy arrays.
        """
        if parameters is None:
            return {}

        # If it's already a mapping, convert each value
        if isinstance(parameters, dict):
            out: Dict[str, np.ndarray] = {}
            for k, v in parameters.items():
                try:
                    # Handle torch tensors if available
                    if type(v).__name__ == 'Tensor':
                        try:
                            import torch
                            if isinstance(v, torch.Tensor):
                                out[k] = v.detach().cpu().numpy()
                                continue
                        except Exception:
                            pass

                    out[k] = np.asarray(v)
                except Exception:
                    # Fallback: coerce to numpy via list conversion
                    try:
                        out[k] = np.asarray(list(v))
                    except Exception:
                        out[k] = np.asarray(0)
            return out

        # If it's a single tensor/array-like, return a single-entry dict
        try:
            if type(parameters).__name__ == 'Tensor':
                import torch
                if isinstance(parameters, torch.Tensor):
                    return {'param_0': parameters.detach().cpu().numpy()}
        except Exception:
            pass

        try:
            return {'param_0': np.asarray(parameters)}
        except Exception:
            return {'param_0': np.asarray(0)}
    
    def compute_fisher_information(self, parameters: Dict[str, np.ndarray], 
                                 gradients: Dict[str, np.ndarray]) -> None:
        """
        Compute Fisher Information Matrix for current parameters.
        
        Args:
            parameters: Current parameter values
            gradients: Gradients of the loss function
        """
        params = self._normalize_parameters(parameters)
        grads = self._normalize_parameters(gradients)

        for param_name, param_values in params.items():
            if param_name in self.fisher_info:
                # Fisher Information = E[gradient^2]
                grad = grads.get(param_name, np.zeros_like(self.fisher_info[param_name]))
                grad_arr = np.asarray(grad)
                try:
                    fisher_update = grad_arr ** 2
                except Exception:
                    fisher_update = np.square(np.asarray(grad_arr, dtype=float))

                # Exponential moving average for stability
                alpha = 0.9
                try:
                    self.fisher_info[param_name] = (
                        alpha * self.fisher_info[param_name] + 
                        (1 - alpha) * fisher_update
                    )
                except Exception:
                    # Ensure shapes align: coerce to same shape if possible
                    try:
                        fisher_update = np.resize(fisher_update, self.fisher_info[param_name].shape)
                        self.fisher_info[param_name] = (
                            alpha * self.fisher_info[param_name] + 
                            (1 - alpha) * fisher_update
                        )
                    except Exception:
                        logger.debug(f"Skipping fisher update for {param_name} due to shape mismatch")

                try:
                    logger.debug(f"Updated Fisher info for {param_name}: {np.mean(fisher_update):.6f}")
                except Exception:
                    pass
    
    def update_importance_weights(self, parameters: Dict[str, np.ndarray]) -> None:
        """
        Update importance weights based on Fisher Information.
        
        Args:
            parameters: Current parameter values
        """
        params = self._normalize_parameters(parameters)

        for param_name, param_values in params.items():
            if param_name in self.importance_weights:
                # Importance = Fisher Information * (current - old)^2
                if param_name in self.parameter_history:
                    old_params = self.parameter_history[param_name]
                    try:
                        param_diff = (np.asarray(param_values) - np.asarray(old_params)) ** 2
                    except Exception:
                        param_diff = np.square(np.asarray(param_values) - np.asarray(old_params))

                    importance_update = self.fisher_info.get(param_name, np.zeros_like(np.asarray(param_values))) * param_diff

                    # Update importance weights
                    try:
                        self.importance_weights[param_name] = np.maximum(
                            self.importance_weights[param_name],
                            importance_update
                        )
                    except Exception:
                        # Try to resize importance to match
                        try:
                            importance_update = np.resize(importance_update, self.importance_weights[param_name].shape)
                            self.importance_weights[param_name] = np.maximum(
                                self.importance_weights[param_name],
                                importance_update
                            )
                        except Exception:
                            logger.debug(f"Could not update importance for {param_name}")

                # Store current parameters for next update (copy numpy array)
                self.parameter_history[param_name] = np.asarray(param_values).copy()
    
    def compute_ewc_loss(self, parameters: Dict[str, np.ndarray], 
                        old_parameters: Dict[str, np.ndarray]) -> float:
        """
        Compute EWC regularization loss to prevent catastrophic forgetting.
        
        Args:
            parameters: Current parameter values
            old_parameters: Previous parameter values to preserve
            
        Returns:
            EWC loss value
        """
        ewc_loss = 0.0
        
        params = self._normalize_parameters(parameters)
        old_params_dict = self._normalize_parameters(old_parameters)

        for param_name, current_params in params.items():
            if (param_name in old_params_dict and 
                param_name in self.importance_weights):

                old_params = old_params_dict[param_name]
                importance = self.importance_weights[param_name]

                # EWC loss = sum(importance * (current - old)^2)
                param_diff = np.asarray(current_params) - np.asarray(old_params)
                ewc_loss += np.sum(importance * (param_diff ** 2))
        
        return ewc_loss * self.learning_rate
    
    def consolidate_weights(self, parameters: Dict[str, np.ndarray], 
                          old_parameters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply EWC consolidation to prevent catastrophic forgetting.
        
        Args:
            parameters: Current parameter values
            old_parameters: Previous parameter values to preserve
            
        Returns:
            Consolidated parameter values
        """
        consolidated_params = {}
        
        params = self._normalize_parameters(parameters)
        old_params_dict = self._normalize_parameters(old_parameters)

        for param_name, current_params in params.items():
            if (param_name in old_params_dict and 
                param_name in self.importance_weights):

                old_params = old_params_dict[param_name]
                importance = self.importance_weights[param_name]

                # EWC consolidation: weighted average based on importance
                # High importance = preserve old, Low importance = use new
                consolidation_factor = 1.0 / (1.0 + importance)

                consolidated_params[param_name] = (
                    consolidation_factor * current_params + 
                    (1 - consolidation_factor) * old_params
                )

                try:
                    logger.debug(f"Consolidated {param_name}: factor={np.mean(consolidation_factor):.4f}")
                except Exception:
                    pass
            else:
                consolidated_params[param_name] = np.asarray(current_params)
        
        return consolidated_params
    
    def should_consolidate(self, parameters: Dict[str, np.ndarray]) -> bool:
        """
        Determine if parameters should be consolidated based on importance.
        
        Args:
            parameters: Current parameter values
            
        Returns:
            True if consolidation is needed
        """
        params = self._normalize_parameters(parameters)
        for param_name, param_values in params.items():
            if param_name in self.importance_weights:
                importance = self.importance_weights[param_name]
                try:
                    max_importance = np.max(importance)
                except Exception:
                    max_importance = float(np.mean(importance)) if hasattr(importance, 'mean') else 0.0

                if max_importance > self.consolidation_threshold:
                    logger.info(f"High importance detected in {param_name}: {max_importance:.4f}")
                    return True
        
        return False
    
    def save_ewc_state(self, filepath: Optional[str] = None) -> None:
        """Save EWC state to file."""
        if filepath is None:
            filepath = f"data/ewc_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            ewc_state = {
                'fisher_info': {k: v.tolist() for k, v in self.fisher_info.items()},
                'importance_weights': {k: v.tolist() for k, v in self.importance_weights.items()},
                'parameter_history': {k: v.tolist() for k, v in self.parameter_history.items()},
                'consolidation_threshold': self.consolidation_threshold,
                'learning_rate': self.learning_rate,
                'timestamp': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(ewc_state, f, indent=2)
            
            logger.info(f"EWC state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save EWC state: {e}")
    
    def load_ewc_state(self, filepath: str) -> bool:
        """Load EWC state from file."""
        try:
            with open(filepath, 'r') as f:
                ewc_state = json.load(f)
            
            self.fisher_info = {k: np.array(v) for k, v in ewc_state['fisher_info'].items()}
            self.importance_weights = {k: np.array(v) for k, v in ewc_state['importance_weights'].items()}
            self.parameter_history = {k: np.array(v) for k, v in ewc_state['parameter_history'].items()}
            self.consolidation_threshold = ewc_state.get('consolidation_threshold', 0.1)
            self.learning_rate = ewc_state.get('learning_rate', 0.01)
            
            logger.info(f"EWC state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load EWC state: {e}")
            return False
    
    def get_consolidation_report(self) -> Dict[str, Any]:
        """Generate a report on current consolidation state."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_parameters': len(self.importance_weights),
            'high_importance_params': [],
            'consolidation_needed': False
        }
        
        for param_name, importance in self.importance_weights.items():
            max_importance = np.max(importance)
            if max_importance > self.consolidation_threshold:
                report['high_importance_params'].append({
                    'name': param_name,
                    'max_importance': float(max_importance),
                    'mean_importance': float(np.mean(importance))
                })
                report['consolidation_needed'] = True
        
        return report
    
    async def enhanced_consolidate_weights(self, parameters: Dict[str, np.ndarray], 
                                         old_parameters: Dict[str, np.ndarray],
                                         context: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """
        Enhanced weight consolidation with monitoring and database integration.
        
        Args:
            parameters: Current parameter values
            old_parameters: Previous parameter values to preserve
            context: Additional context information
            
        Returns:
            Consolidated parameter values
        """
        start_time = datetime.now()
        
        try:
            # Perform standard consolidation
            consolidated_params = self.consolidate_weights(parameters, old_parameters)
            
            # Update monitoring systems
            if self.enable_monitoring:
                await self._update_monitoring_systems(parameters, consolidated_params, context)
            
            # Store operation data
            if self.enable_database_storage:
                await self._store_consolidation_data(parameters, consolidated_params, context, start_time)
            
            logger.info(f"Enhanced EWC consolidation completed: {len(consolidated_params)} parameters")
            return consolidated_params
            
        except Exception as e:
            logger.error(f"Enhanced EWC consolidation failed: {e}")
            return parameters  # Return original parameters on failure
    
    async def _update_monitoring_systems(self, parameters: Dict[str, np.ndarray], 
                                       consolidated_params: Dict[str, np.ndarray],
                                       context: Optional[Dict[str, Any]]) -> None:
        """Update cognitive monitoring systems."""
        try:
            if not self.enable_monitoring:
                return
            
            # Calculate consolidation metrics
            consolidation_metrics = self._calculate_consolidation_metrics(parameters, consolidated_params)
            
            # Update learning progress monitor
            if hasattr(self, 'learning_progress_monitor'):
                learning_data = {
                    'consolidation_quality': consolidation_metrics['consolidation_quality'],
                    'parameter_stability': consolidation_metrics['parameter_stability'],
                    'forgetting_prevention': consolidation_metrics['forgetting_prevention'],
                    'context': context
                }
                self.learning_progress_monitor.update_metrics(learning_data)
            
            # Update meta-learning monitor
            if hasattr(self, 'meta_learning_monitor'):
                meta_learning_data = {
                    'ewc_performance': consolidation_metrics['ewc_performance'],
                    'adaptation_rate': consolidation_metrics['adaptation_rate'],
                    'context': context
                }
                self.meta_learning_monitor.update_metrics(meta_learning_data)
            
        except Exception as e:
            logger.error(f"Failed to update monitoring systems: {e}")
    
    async def _store_consolidation_data(self, parameters: Dict[str, np.ndarray], 
                                      consolidated_params: Dict[str, np.ndarray],
                                      context: Optional[Dict[str, Any]], 
                                      start_time: datetime) -> None:
        """Store consolidation data in database."""
        try:
            if not self.enable_database_storage or not hasattr(self, 'integration'):
                return
            
            # Calculate metrics
            consolidation_metrics = self._calculate_consolidation_metrics(parameters, consolidated_params)
            
            # Create operation record
            operation_data = {
                'operation_type': 'ewc_consolidation',
                'parameter_count': len(parameters),
                'consolidation_quality': consolidation_metrics['consolidation_quality'],
                'parameter_stability': consolidation_metrics['parameter_stability'],
                'forgetting_prevention': consolidation_metrics['forgetting_prevention'],
                'ewc_performance': consolidation_metrics['ewc_performance'],
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'context': context or {}
            }
            
            await self.integration.log_system_event(
                LogLevel.INFO,
                Component.ARCHITECT,
                f"EWC consolidation: {len(parameters)} parameters",
                operation_data,
                "ewc_consolidation"
            )
            
        except Exception as e:
            logger.error(f"Failed to store consolidation data: {e}")
    
    def _calculate_consolidation_metrics(self, parameters: Dict[str, np.ndarray], 
                                       consolidated_params: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate consolidation performance metrics."""
        try:
            # Calculate parameter stability (how much parameters changed)
            total_change = 0.0
            total_params = 0
            
            params = self._normalize_parameters(parameters)
            cons = self._normalize_parameters(consolidated_params)

            for param_name in params:
                if param_name in cons:
                    param_diff = np.mean(np.abs(np.asarray(params[param_name]) - np.asarray(cons[param_name])))
                    total_change += param_diff
                    total_params += 1
            
            parameter_stability = 1.0 - (total_change / total_params) if total_params > 0 else 1.0
            
            # Calculate consolidation quality (based on importance weights)
            consolidation_quality = 0.0
            if self.importance_weights:
                total_importance = sum(np.mean(imp) for imp in self.importance_weights.values())
                consolidation_quality = min(1.0, total_importance / len(self.importance_weights))
            
            # Calculate forgetting prevention (based on Fisher information)
            forgetting_prevention = 0.0
            if self.fisher_info:
                total_fisher = sum(np.mean(fisher) for fisher in self.fisher_info.values())
                forgetting_prevention = min(1.0, total_fisher / len(self.fisher_info))
            
            # Calculate EWC performance (overall effectiveness)
            ewc_performance = (parameter_stability + consolidation_quality + forgetting_prevention) / 3.0
            
            # Calculate adaptation rate (how quickly the system adapts)
            adaptation_rate = min(1.0, self.learning_rate * 100)  # Scale learning rate to [0, 1]
            
            return {
                'parameter_stability': float(parameter_stability),
                'consolidation_quality': float(consolidation_quality),
                'forgetting_prevention': float(forgetting_prevention),
                'ewc_performance': float(ewc_performance),
                'adaptation_rate': float(adaptation_rate)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate consolidation metrics: {e}")
            return {
                'parameter_stability': 0.0,
                'consolidation_quality': 0.0,
                'forgetting_prevention': 0.0,
                'ewc_performance': 0.0,
                'adaptation_rate': 0.0
            }
    
    async def get_enhanced_report(self) -> Dict[str, Any]:
        """Get enhanced EWC report with monitoring data."""
        try:
            # Get base report
            base_report = self.get_consolidation_report()
            
            # Add monitoring data
            if self.enable_monitoring:
                base_report['monitoring_data'] = {
                    'learning_progress': getattr(self.learning_progress_monitor, 'metrics', {}),
                    'meta_learning': getattr(self.meta_learning_monitor, 'metrics', {})
                }
            
            # Add database integration status
            base_report['database_integration'] = {
                'enabled': self.enable_database_storage,
                'integration_available': hasattr(self, 'integration')
            }
            
            return base_report
            
        except Exception as e:
            logger.error(f"Failed to get enhanced report: {e}")
            return self.get_consolidation_report()
