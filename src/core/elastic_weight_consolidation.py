#!/usr/bin/env python3
"""
Elastic Weight Consolidation (EWC) for Tabula Rasa Architect
Prevents catastrophic forgetting during system evolution by preserving important weights.
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ElasticWeightConsolidation:
    """
    EWC implementation for the Architect system to prevent catastrophic forgetting.
    Maintains Fisher Information Matrix and importance weights for critical parameters.
    """
    
    def __init__(self, config_path: str = "data/config/ewc_config.json"):
        self.config_path = config_path
        self.fisher_info = {}  # Fisher Information Matrix
        self.importance_weights = {}  # Importance weights for each parameter
        self.parameter_history = {}  # Historical parameter values
        self.consolidation_threshold = 0.1  # Threshold for parameter importance
        self.learning_rate = 0.01  # EWC learning rate
        
        self._load_config()
        self._initialize_ewc()
    
    def _load_config(self):
        """Load EWC configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.consolidation_threshold = config.get('consolidation_threshold', 0.1)
                    self.learning_rate = config.get('learning_rate', 0.01)
        except Exception as e:
            logger.warning(f"Could not load EWC config: {e}")
    
    def _initialize_ewc(self):
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
    
    def compute_fisher_information(self, parameters: Dict[str, np.ndarray], 
                                 gradients: Dict[str, np.ndarray]) -> None:
        """
        Compute Fisher Information Matrix for current parameters.
        
        Args:
            parameters: Current parameter values
            gradients: Gradients of the loss function
        """
        for param_name, param_values in parameters.items():
            if param_name in self.fisher_info:
                # Fisher Information = E[gradient^2]
                fisher_update = gradients[param_name] ** 2
                
                # Exponential moving average for stability
                alpha = 0.9
                self.fisher_info[param_name] = (
                    alpha * self.fisher_info[param_name] + 
                    (1 - alpha) * fisher_update
                )
                
                logger.debug(f"Updated Fisher info for {param_name}: {fisher_update.mean():.6f}")
    
    def update_importance_weights(self, parameters: Dict[str, np.ndarray]) -> None:
        """
        Update importance weights based on Fisher Information.
        
        Args:
            parameters: Current parameter values
        """
        for param_name, param_values in parameters.items():
            if param_name in self.importance_weights:
                # Importance = Fisher Information * (current - old)^2
                if param_name in self.parameter_history:
                    old_params = self.parameter_history[param_name]
                    param_diff = (param_values - old_params) ** 2
                    importance_update = self.fisher_info[param_name] * param_diff
                    
                    # Update importance weights
                    self.importance_weights[param_name] = np.maximum(
                        self.importance_weights[param_name],
                        importance_update
                    )
                
                # Store current parameters for next update
                self.parameter_history[param_name] = param_values.copy()
    
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
        
        for param_name, current_params in parameters.items():
            if (param_name in old_parameters and 
                param_name in self.importance_weights):
                
                old_params = old_parameters[param_name]
                importance = self.importance_weights[param_name]
                
                # EWC loss = sum(importance * (current - old)^2)
                param_diff = current_params - old_params
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
        
        for param_name, current_params in parameters.items():
            if (param_name in old_parameters and 
                param_name in self.importance_weights):
                
                old_params = old_parameters[param_name]
                importance = self.importance_weights[param_name]
                
                # EWC consolidation: weighted average based on importance
                # High importance = preserve old, Low importance = use new
                consolidation_factor = 1.0 / (1.0 + importance)
                
                consolidated_params[param_name] = (
                    consolidation_factor * current_params + 
                    (1 - consolidation_factor) * old_params
                )
                
                logger.debug(f"Consolidated {param_name}: factor={consolidation_factor.mean():.4f}")
            else:
                consolidated_params[param_name] = current_params
        
        return consolidated_params
    
    def should_consolidate(self, parameters: Dict[str, np.ndarray]) -> bool:
        """
        Determine if parameters should be consolidated based on importance.
        
        Args:
            parameters: Current parameter values
            
        Returns:
            True if consolidation is needed
        """
        for param_name, param_values in parameters.items():
            if param_name in self.importance_weights:
                importance = self.importance_weights[param_name]
                max_importance = np.max(importance)
                
                if max_importance > self.consolidation_threshold:
                    logger.info(f"High importance detected in {param_name}: {max_importance:.4f}")
                    return True
        
        return False
    
    def save_ewc_state(self, filepath: str = None) -> None:
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
    
    def get_consolidation_report(self) -> Dict:
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
