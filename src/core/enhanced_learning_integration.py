"""
Enhanced Learning Integration API

Unified API for integrating EWC, Residual Learning, and ELMs with database storage
and cognitive monitoring systems.
"""

import sys
sys.dont_write_bytecode = True

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import numpy as np

from .elastic_weight_consolidation import ElasticWeightConsolidation
from .residual_learning import ResidualLearningSystem
from .extreme_learning_machines import ExtremeLearningMachine, DirectorELMEnsemble
from ..database.system_integration import get_system_integration
from ..database.api import Component, LogLevel
from ..core.cognitive_subsystems import CognitiveCoordinator

logger = logging.getLogger(__name__)


class EnhancedLearningIntegration:
    """
    Unified API for enhanced learning paradigms with database integration and monitoring.
    
    This class provides:
    - EWC for catastrophic forgetting prevention
    - Residual Learning for gradient flow optimization
    - ELMs for fast learning and decision-making
    - Comprehensive monitoring and database storage
    - Cognitive subsystem integration
    """
    
    def __init__(
        self,
        enable_monitoring: bool = True,
        enable_database_storage: bool = True,
        ewc_config: Optional[Dict[str, Any]] = None,
        residual_config: Optional[Dict[str, Any]] = None,
        elm_config: Optional[Dict[str, Any]] = None
    ):
        self.enable_monitoring = enable_monitoring
        self.enable_database_storage = enable_database_storage
        
        # Initialize learning systems
        self.ewc = ElasticWeightConsolidation(
            enable_monitoring=enable_monitoring,
            enable_database_storage=enable_database_storage
        )
        
        self.residual = ResidualLearningSystem(
            enable_monitoring=enable_monitoring,
            enable_database_storage=enable_database_storage
        )
        
        self.elm = ExtremeLearningMachine(
            enable_monitoring=enable_monitoring,
            enable_database_storage=enable_database_storage
        )
        
        self.elm_ensemble = DirectorELMEnsemble()
        
        # Initialize cognitive coordinator
        if enable_monitoring:
            self.cognitive_coordinator = CognitiveCoordinator()
        else:
            self.cognitive_coordinator = None
        
        # Database integration
        if enable_database_storage:
            self.integration = get_system_integration()
        else:
            self.integration = None
        
        # Session tracking
        self.session_id = f"enhanced_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_start_time = datetime.now()
        self.operation_count = 0
        
        # Performance tracking
        self.performance_history = []
        self.learning_metrics = {}
        
        logger.info(f"Enhanced Learning Integration initialized: session={self.session_id}")
    
    async def process_ewc_consolidation(
        self,
        parameters: Dict[str, np.ndarray],
        old_parameters: Dict[str, np.ndarray],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process EWC consolidation."""
        try:
            return await self.ewc.enhanced_consolidate_weights(parameters, old_parameters, context)
        except Exception as e:
            logger.error(f"EWC consolidation failed: {e}")
            return {'error': str(e)}

    async def process_residual_forward_pass(
        self,
        layer_name: str,
        input_data: np.ndarray,
        layer_weights: np.ndarray,
        layer_bias: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process residual forward pass."""
        try:
            output = await self.residual.enhanced_forward_pass(layer_name, input_data, layer_weights, layer_bias, context)
            return {'output': output}
        except Exception as e:
            logger.error(f"Residual forward pass failed: {e}")
            return {'error': str(e)}

    async def process_elm_training(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process ELM training."""
        try:
            return await self.elm.enhanced_train_batch(input_data, target_data, context)
        except Exception as e:
            logger.error(f"ELM training failed: {e}")
            return {'error': str(e)}

    async def process_elm_ensemble_training(
        self,
        training_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process ELM ensemble training."""
        try:
            return {'metrics': self.elm_ensemble.train_ensemble(training_data)}
        except Exception as e:
            logger.error(f"ELM ensemble training failed: {e}")
            return {'error': str(e)}

    async def initialize(self) -> bool:
        """Initialize the enhanced learning integration system."""
        try:
            # Initialize cognitive coordinator
            if self.cognitive_coordinator:
                await self.cognitive_coordinator.initialize_all_subsystems()
                logger.info("Cognitive subsystems initialized")
            
            # Log initialization
            if self.integration:
                await self.integration.log_system_event(
                    LogLevel.INFO,
                    Component.DIRECTOR,
                    f"Enhanced Learning Integration initialized: {self.session_id}",
                    {
                        'session_id': self.session_id,
                        'enable_monitoring': self.enable_monitoring,
                        'enable_database_storage': self.enable_database_storage,
                        'ewc_enabled': True,
                        'residual_enabled': True,
                        'elm_enabled': True,
                        'elm_ensemble_enabled': True
                    },
                    self.session_id
                )
            
            logger.info("Enhanced Learning Integration initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Learning Integration: {e}")
            return False

    async def process_learning_experience(self, learning_experience: Union[Dict[str, Any], str, float, int]) -> None:
        """Process a learning experience from game results with lifecycle pattern analysis.

        Args:
            learning_experience: Either a dictionary containing game result data for learning,
                               or a direct score value that can be a string, float or int
        """
        try:
            # Extract key metrics from the learning experience
            if isinstance(learning_experience, dict):
                game_score = learning_experience.get('final_score', 0.0)
                game_won = learning_experience.get('game_won', False)
                total_actions = learning_experience.get('total_actions', 0)
                game_duration = learning_experience.get('game_duration', 0.0)

                # Extract lifecycle pattern data if available
                lifecycle_patterns = learning_experience.get('lifecycle_patterns', {})
                failure_risk_score = lifecycle_patterns.get('failure_risk_score', 0.0)
                oscillation_detected = lifecycle_patterns.get('oscillation_detected', False)
                action_effectiveness = lifecycle_patterns.get('action_effectiveness', {})
            else:
                # Handle scalar value (str, float, int)
                try:
                    game_score = float(learning_experience)
                except (TypeError, ValueError):
                    game_score = 0.0
                game_won = game_score > 50  # Consider it a win if score > 50
                total_actions = 1  # Single action case
                game_duration = 0.0  # Duration unknown
                
                # Set default lifecycle metrics for scalar input
                failure_risk_score = 0.0
                oscillation_detected = False
                action_effectiveness = {}

            # Create enhanced learning input data with lifecycle features
            import torch
            hypotheses_generated = learning_experience.get('hypotheses_generated', 0) if isinstance(learning_experience, dict) else 0
            
            base_features = [
                game_score,
                1.0 if game_won else 0.0,
                total_actions,
                game_duration,
                hypotheses_generated
            ]

            # Add lifecycle pattern features
            lifecycle_features = [
                failure_risk_score,
                1.0 if oscillation_detected else 0.0,
                len(action_effectiveness),  # Number of action effectiveness patterns
                sum(action_effectiveness.values()) / max(len(action_effectiveness), 1)  # Average effectiveness
            ]

            # Pad features to match ELM input size (100)
            input_features = base_features + lifecycle_features  # 9 features
            padding_size = 100 - len(input_features)  # Pad to 100
            padded_features = input_features + [0.0] * padding_size
            
            learning_input = torch.tensor(padded_features, dtype=torch.float32).unsqueeze(0)

            # Create enhanced target for learning (success indicator with failure pattern weighting)
            success_score = 1.0 if game_won or game_score > 50 else 0.0
            # Reduce target score if high failure risk was detected but game still succeeded
            if success_score > 0 and failure_risk_score > 0.7:
                success_score *= 0.8  # Reduce success weight for risky success
            learning_target = torch.tensor([success_score], dtype=torch.float32).unsqueeze(0)

            # Convert PyTorch tensors to numpy for downstream components
            try:
                input_np = learning_input.detach().cpu().numpy() if hasattr(learning_input, 'detach') else np.array(learning_input)
            except Exception:
                input_np = np.array(learning_input)

            try:
                target_np = learning_target.detach().cpu().numpy() if hasattr(learning_target, 'detach') else np.array(learning_target)
            except Exception:
                target_np = np.array(learning_target)

            # Process through EWC if available (expecting dicts of numpy arrays)
            if self.ewc:
                try:
                    params = {'param_0': input_np}
                    old_params = {'param_0': target_np}
                    await self.process_ewc_consolidation(params, old_params)
                except Exception as e:
                    logger.warning(f"EWC processing skipped: {e}")

            # Process through residual learning if available (provide layer weights/bias placeholders)
            if self.residual:
                try:
                    # Ensure input dimension
                    if input_np.ndim == 2:
                        input_dim = input_np.shape[1]
                    else:
                        input_dim = int(np.prod(input_np.shape))

                    # Create small random weight matrix and zero bias to match shapes
                    layer_weights = np.random.normal(0, 0.1, (input_dim, 1))
                    layer_bias = np.zeros((1,))

                    await self.process_residual_forward_pass('learning_layer', input_np, layer_weights, layer_bias)
                except Exception as e:
                    logger.warning(f"Residual processing skipped: {e}")

            # Process through ELM if available
            if self.elm:
                try:
                    await self.process_elm_training(input_np, target_np)
                except Exception as e:
                    logger.warning(f"ELM processing skipped: {e}")

            # Process through ELM ensemble if available
            if self.elm_ensemble:
                try:
                    # ensemble expects mapping of tasks; provide a single-task dict
                    await self.process_elm_ensemble_training({'task_0': (input_np, target_np)})
                except Exception as e:
                    logger.warning(f"ELM ensemble processing skipped: {e}")

            # Update operation count and metrics
            self.operation_count += 1

            # Store enhanced performance metrics with lifecycle patterns
            performance_metrics = {
                'game_score': game_score,
                'game_won': game_won,
                'total_actions': total_actions,
                'game_duration': game_duration,
                'operation_count': self.operation_count,
                'timestamp': learning_experience.get('timestamp') if isinstance(learning_experience, dict) else datetime.now().isoformat(),
                # Lifecycle pattern metrics
                'failure_risk_score': failure_risk_score,
                'oscillation_detected': oscillation_detected,
                'action_effectiveness_count': len(action_effectiveness),
                'avg_action_effectiveness': sum(action_effectiveness.values()) / max(len(action_effectiveness), 1)
            }

            self.performance_history.append(performance_metrics)

            # Update enhanced learning metrics with lifecycle insights
            if len(self.performance_history) > 0:
                total_experiences = len(self.performance_history)
                self.learning_metrics.update({
                    'total_experiences_processed': self.operation_count,
                    'average_game_score': sum(exp.get('game_score', 0.0) for exp in self.performance_history) / total_experiences,
                    'win_rate': sum(1 for exp in self.performance_history if exp.get('game_won', False)) / total_experiences,
                    'average_failure_risk': sum(exp.get('failure_risk_score', 0.0) for exp in self.performance_history) / total_experiences,
                    'oscillation_frequency': sum(1 for exp in self.performance_history if exp.get('oscillation_detected', False)) / total_experiences,
                    'last_updated': performance_metrics['timestamp']
                })

            logger.info(f"Processed enhanced learning experience: score={game_score}, won={game_won}, "
                       f"risk={failure_risk_score:.2f}, oscillation={oscillation_detected}, operations={self.operation_count}")

        except Exception as e:
            logger.warning(f"Failed to process learning experience: {e}")
            raise


# Factory function for easy creation
def create_enhanced_learning_integration(
    enable_monitoring: bool = True,
    enable_database_storage: bool = True,
    ewc_config: Optional[Dict[str, Any]] = None,
    residual_config: Optional[Dict[str, Any]] = None,
    elm_config: Optional[Dict[str, Any]] = None
) -> EnhancedLearningIntegration:
    """
    Factory function to create an Enhanced Learning Integration instance.
    
    Args:
        enable_monitoring: Enable cognitive monitoring
        enable_database_storage: Enable database storage
        ewc_config: EWC configuration parameters
        residual_config: Residual learning configuration parameters
        elm_config: ELM configuration parameters
        
    Returns:
        Configured EnhancedLearningIntegration instance
    """
    return EnhancedLearningIntegration(
        enable_monitoring=enable_monitoring,
        enable_database_storage=enable_database_storage,
        ewc_config=ewc_config,
        residual_config=residual_config,
        elm_config=elm_config
    )