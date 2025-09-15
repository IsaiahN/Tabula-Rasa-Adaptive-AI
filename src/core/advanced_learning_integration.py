#!/usr/bin/env python3
"""
Advanced Learning Integration for Tabula Rasa
Integrates EWC, Residual Learning, and ELMs with existing Conductor, Architect, and Governor systems.
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging

from .elastic_weight_consolidation import ElasticWeightConsolidation
from .residual_learning import ResidualLearningSystem
from .extreme_learning_machines import ExtremeLearningMachine, ConductorELMEnsemble

logger = logging.getLogger(__name__)

class AdvancedLearningIntegration:
    """
    Integrates advanced learning paradigms with Tabula Rasa's core systems.
    Coordinates EWC, Residual Learning, and ELMs across Conductor, Architect, and Governor.
    """
    
    def __init__(self, config_path: str = "data/config/advanced_learning_config.json"):
        self.config_path = config_path
        
        # Initialize advanced learning systems
        self.ewc_system = ElasticWeightConsolidation()
        self.residual_system = ResidualLearningSystem()
        self.elm_ensemble = ConductorELMEnsemble()
        
        # Integration state
        self.integration_active = True
        self.performance_history = []
        self.learning_metrics = {}
        
        self._load_config()
        self._initialize_integration()
    
    def _load_config(self):
        """Load advanced learning integration configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.integration_active = config.get('integration_active', True)
        except Exception as e:
            logger.warning(f"Could not load advanced learning config: {e}")
    
    def _initialize_integration(self):
        """Initialize the integration between advanced learning systems."""
        logger.info("Initializing advanced learning integration...")
        
        # Set up cross-system communication
        self._setup_conductor_elm_integration()
        self._setup_architect_ewc_integration()
        self._setup_governor_residual_integration()
        
        logger.info("Advanced learning integration initialized successfully")
    
    def _setup_conductor_elm_integration(self):
        """Set up ELM integration for the Conductor's meta-cognitive processes."""
        # Configure ELM ensemble for different Conductor tasks
        conductor_tasks = {
            'narrative_engine': {
                'input_size': 50,  # Context features
                'hidden_size': 100,
                'output_size': 25,  # Narrative decisions
                'activation_function': 'sigmoid'
            },
            'affective_agent': {
                'input_size': 30,  # Emotional state features
                'hidden_size': 60,
                'output_size': 15,  # Affective responses
                'activation_function': 'tanh'
            },
            'drive_agent': {
                'input_size': 40,  # Drive state features
                'hidden_size': 80,
                'output_size': 20,  # Drive priorities
                'activation_function': 'relu'
            },
            'social_simulant': {
                'input_size': 35,  # Social context features
                'hidden_size': 70,
                'output_size': 18,  # Social predictions
                'activation_function': 'sigmoid'
            }
        }
        
        # Initialize ELMs for each Conductor task
        for task_name, config in conductor_tasks.items():
            if task_name in self.elm_ensemble.elms:
                elm = self.elm_ensemble.elms[task_name]
                elm.input_size = config['input_size']
                elm.hidden_size = config['hidden_size']
                elm.output_size = config['output_size']
                elm.activation_function = config['activation_function']
                elm._initialize_elm()
        
        logger.info("Conductor ELM integration configured")
    
    def _setup_architect_ewc_integration(self):
        """Set up EWC integration for the Architect's evolution system."""
        # Configure EWC for different Architect components
        architect_components = {
            'action_selection_weights': (100, 100),
            'pattern_recognition_weights': (50, 50),
            'memory_consolidation_weights': (30, 30),
            'meta_learning_weights': (40, 40)
        }
        
        # Initialize EWC for each component
        for component_name, (rows, cols) in architect_components.items():
            if component_name in self.ewc_system.fisher_info:
                self.ewc_system.fisher_info[component_name] = np.zeros((rows, cols))
                self.ewc_system.importance_weights[component_name] = np.ones((rows, cols)) * 0.1
        
        logger.info("Architect EWC integration configured")
    
    def _setup_governor_residual_integration(self):
        """Set up Residual Learning integration for the Governor's decision-making."""
        # Configure residual learning for different Governor subsystems
        governor_subsystems = {
            'resource_allocator': {
                'input_size': 50,
                'hidden_size': 100,
                'output_size': 25,
                'residual_connections': [0, 2, 4]
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
            }
        }
        
        # Initialize residual learning for each subsystem
        for subsystem_name, config in governor_subsystems.items():
            if subsystem_name in self.residual_system.residual_layers:
                self.residual_system.residual_layers[subsystem_name] = config
                
                # Initialize skip connections
                self.residual_system.skip_connections[subsystem_name] = {}
                for conn_idx in config['residual_connections']:
                    self.residual_system.skip_connections[subsystem_name][conn_idx] = np.random.normal(
                        0, 0.1, (config['input_size'], config['output_size'])
                    )
        
        logger.info("Governor Residual Learning integration configured")
    
    def process_conductor_decision(self, context_features: np.ndarray, 
                                 decision_type: str = 'strategy') -> Dict[str, Any]:
        """
        Process a Conductor decision using ELM ensemble.
        
        Args:
            context_features: Context features for decision making
            decision_type: Type of decision to make
            
        Returns:
            Decision results with confidence scores
        """
        if not self.integration_active:
            return {'decision': None, 'confidence': 0.0, 'method': 'disabled'}
        
        try:
            # Get ELM prediction for the decision type
            if decision_type in self.elm_ensemble.elms:
                prediction = self.elm_ensemble.predict_ensemble(context_features, decision_type)
                confidence = np.mean(np.abs(prediction))  # Simple confidence measure
                
                return {
                    'decision': prediction.tolist(),
                    'confidence': float(confidence),
                    'method': 'elm_ensemble',
                    'decision_type': decision_type
                }
            else:
                # Fallback to ensemble prediction
                prediction = self.elm_ensemble.predict_ensemble(context_features)
                confidence = np.mean(np.abs(prediction))
                
                return {
                    'decision': prediction.tolist(),
                    'confidence': float(confidence),
                    'method': 'elm_ensemble_fallback',
                    'decision_type': 'ensemble'
                }
                
        except Exception as e:
            logger.error(f"Conductor decision processing failed: {e}")
            return {'decision': None, 'confidence': 0.0, 'method': 'error', 'error': str(e)}
    
    def process_architect_evolution(self, current_parameters: Dict[str, np.ndarray],
                                  evolution_direction: str = 'optimization') -> Dict[str, Any]:
        """
        Process Architect evolution using EWC to prevent catastrophic forgetting.
        
        Args:
            current_parameters: Current system parameters
            evolution_direction: Direction of evolution
            
        Returns:
            Evolution results with consolidation information
        """
        if not self.integration_active:
            return {'evolved_parameters': current_parameters, 'consolidation_applied': False}
        
        try:
            # Check if consolidation is needed
            should_consolidate = self.ewc_system.should_consolidate(current_parameters)
            
            if should_consolidate:
                # Apply EWC consolidation
                old_parameters = self.ewc_system.parameter_history.copy()
                consolidated_params = self.ewc_system.consolidate_weights(
                    current_parameters, old_parameters
                )
                
                # Update importance weights
                self.ewc_system.update_importance_weights(consolidated_params)
                
                logger.info(f"Applied EWC consolidation for {evolution_direction}")
                
                return {
                    'evolved_parameters': consolidated_params,
                    'consolidation_applied': True,
                    'method': 'ewc_consolidation',
                    'evolution_direction': evolution_direction
                }
            else:
                # No consolidation needed, update importance weights
                self.ewc_system.update_importance_weights(current_parameters)
                
                return {
                    'evolved_parameters': current_parameters,
                    'consolidation_applied': False,
                    'method': 'no_consolidation',
                    'evolution_direction': evolution_direction
                }
                
        except Exception as e:
            logger.error(f"Architect evolution processing failed: {e}")
            return {'evolved_parameters': current_parameters, 'consolidation_applied': False, 'error': str(e)}
    
    def process_governor_decision(self, input_data: np.ndarray, 
                                subsystem: str = 'decision_engine') -> Dict[str, Any]:
        """
        Process Governor decision using Residual Learning.
        
        Args:
            input_data: Input data for decision making
            subsystem: Governor subsystem to use
            
        Returns:
            Decision results with residual information
        """
        if not self.integration_active:
            return {'decision': None, 'residual_applied': False}
        
        try:
            if subsystem in self.residual_system.residual_layers:
                # Simulate layer processing with residual connections
                layer_config = self.residual_system.residual_layers[subsystem]
                
                # Create dummy weights and bias for simulation
                layer_weights = np.random.normal(0, 0.1, (layer_config['input_size'], layer_config['output_size']))
                layer_bias = np.random.normal(0, 0.1, (1, layer_config['output_size']))
                
                # Apply residual learning forward pass
                decision = self.residual_system.forward_pass(
                    subsystem, input_data, layer_weights, layer_bias
                )
                
                return {
                    'decision': decision.tolist(),
                    'residual_applied': True,
                    'method': 'residual_learning',
                    'subsystem': subsystem
                }
            else:
                # Fallback to simple processing
                decision = np.random.normal(0, 0.1, (input_data.shape[0], 10))
                
                return {
                    'decision': decision.tolist(),
                    'residual_applied': False,
                    'method': 'fallback',
                    'subsystem': subsystem
                }
                
        except Exception as e:
            logger.error(f"Governor decision processing failed: {e}")
            return {'decision': None, 'residual_applied': False, 'error': str(e)}
    
    def update_learning_metrics(self, performance_data: Dict[str, Any]) -> None:
        """
        Update learning metrics based on system performance.
        
        Args:
            performance_data: Performance data from all systems
        """
        # Update ELM ensemble weights based on performance
        if 'elm_performance' in performance_data:
            self.elm_ensemble.update_ensemble_weights(performance_data['elm_performance'])
        
        # Update residual learning parameters
        if 'residual_performance' in performance_data:
            self.residual_system.optimize_residual_architecture(
                performance_data['residual_performance']
            )
        
        # Update EWC parameters
        if 'ewc_performance' in performance_data:
            # EWC parameters are updated during evolution processing
            pass
        
        # Store performance history
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'performance_data': performance_data
        })
        
        # Keep only last 100 entries
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def get_integration_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report on the advanced learning integration."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'integration_active': self.integration_active,
            'ewc_report': self.ewc_system.get_consolidation_report(),
            'residual_report': self.residual_system.get_residual_report(),
            'elm_ensemble_report': self.elm_ensemble.get_ensemble_report(),
            'performance_history_length': len(self.performance_history),
            'recent_performance': self.performance_history[-5:] if self.performance_history else []
        }
        
        return report
    
    def save_integration_state(self, filepath: str = None) -> None:
        """Save the complete integration state."""
        if filepath is None:
            filepath = f"data/advanced_learning_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Save individual system states
            self.ewc_system.save_ewc_state()
            self.residual_system.save_residual_state()
            self.elm_ensemble.save_elm_state()
            
            # Save integration state
            integration_state = {
                'integration_active': self.integration_active,
                'performance_history': self.performance_history[-50:],  # Keep last 50 entries
                'timestamp': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(integration_state, f, indent=2)
            
            logger.info(f"Advanced learning integration state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save integration state: {e}")
    
    def load_integration_state(self, filepath: str) -> bool:
        """Load the complete integration state."""
        try:
            # Load individual system states
            self.ewc_system.load_ewc_state(f"data/ewc_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            self.residual_system.load_residual_state(f"data/residual_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            self.elm_ensemble.load_elm_state(f"data/elm_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            # Load integration state
            with open(filepath, 'r') as f:
                integration_state = json.load(f)
            
            self.integration_active = integration_state.get('integration_active', True)
            self.performance_history = integration_state.get('performance_history', [])
            
            logger.info(f"Advanced learning integration state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load integration state: {e}")
            return False
