"""
Enhanced Learning Integration API

Unified API for integrating EWC, Residual Learning, and ELMs with database storage
and cognitive monitoring systems.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
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
    
    async def process_ewc_consolidation(
        self,
        parameters: Dict[str, np.ndarray],
        old_parameters: Dict[str, np.ndarray],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process EWC consolidation with monitoring and database integration.
        
        Args:
            parameters: Current parameter values
            old_parameters: Previous parameter values to preserve
            context: Additional context information
            
        Returns:
            Dictionary containing consolidation results and metadata
        """
        start_time = datetime.now()
        self.operation_count += 1
        
        try:
            # Add context information
            if context is None:
                context = {}
            
            context.update({
                'session_id': self.session_id,
                'operation_count': self.operation_count,
                'operation_type': 'ewc_consolidation',
                'timestamp': start_time.isoformat()
            })
            
            # Perform EWC consolidation
            consolidated_params = await self.ewc.enhanced_consolidate_weights(
                parameters, old_parameters, context
            )
            
            # Calculate consolidation metrics
            consolidation_metrics = self.ewc._calculate_consolidation_metrics(
                parameters, consolidated_params
            )
            
            # Prepare result
            result = {
                'consolidated_parameters': consolidated_params,
                'consolidation_metrics': consolidation_metrics,
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'operation_success': True,
                'session_id': self.session_id,
                'operation_count': self.operation_count
            }
            
            # Update performance history
            self.performance_history.append({
                'timestamp': start_time,
                'operation_type': 'ewc_consolidation',
                'metrics': consolidation_metrics,
                'processing_time': result['processing_time']
            })
            
            logger.info(f"EWC consolidation completed: {len(consolidated_params)} parameters")
            return result
            
        except Exception as e:
            logger.error(f"EWC consolidation failed: {e}")
            return {
                'consolidated_parameters': parameters,  # Return original on failure
                'consolidation_metrics': {},
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'operation_success': False,
                'error': str(e),
                'session_id': self.session_id,
                'operation_count': self.operation_count
            }
    
    async def process_residual_forward_pass(
        self,
        layer_name: str,
        input_data: np.ndarray,
        layer_weights: np.ndarray,
        layer_bias: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process residual forward pass with monitoring and database integration.
        
        Args:
            layer_name: Name of the residual layer
            input_data: Input data
            layer_weights: Layer weights
            layer_bias: Layer bias
            context: Additional context information
            
        Returns:
            Dictionary containing forward pass results and metadata
        """
        start_time = datetime.now()
        self.operation_count += 1
        
        try:
            # Add context information
            if context is None:
                context = {}
            
            context.update({
                'session_id': self.session_id,
                'operation_count': self.operation_count,
                'operation_type': 'residual_forward_pass',
                'timestamp': start_time.isoformat()
            })
            
            # Perform residual forward pass
            output = await self.residual.enhanced_forward_pass(
                layer_name, input_data, layer_weights, layer_bias, context
            )
            
            # Calculate residual metrics
            residual_metrics = self.residual._calculate_residual_metrics(
                layer_name, input_data, output
            )
            
            # Prepare result
            result = {
                'output': output,
                'residual_metrics': residual_metrics,
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'operation_success': True,
                'session_id': self.session_id,
                'operation_count': self.operation_count
            }
            
            # Update performance history
            self.performance_history.append({
                'timestamp': start_time,
                'operation_type': 'residual_forward_pass',
                'layer_name': layer_name,
                'metrics': residual_metrics,
                'processing_time': result['processing_time']
            })
            
            logger.debug(f"Residual forward pass completed: {layer_name}")
            return result
            
        except Exception as e:
            logger.error(f"Residual forward pass failed: {e}")
            return {
                'output': np.zeros_like(input_data),  # Fallback output
                'residual_metrics': {},
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'operation_success': False,
                'error': str(e),
                'session_id': self.session_id,
                'operation_count': self.operation_count
            }
    
    async def process_elm_training(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process ELM training with monitoring and database integration.
        
        Args:
            input_data: Input data for training
            target_data: Target data for training
            context: Additional context information
            
        Returns:
            Dictionary containing training results and metadata
        """
        start_time = datetime.now()
        self.operation_count += 1
        
        try:
            # Add context information
            if context is None:
                context = {}
            
            context.update({
                'session_id': self.session_id,
                'operation_count': self.operation_count,
                'operation_type': 'elm_training',
                'timestamp': start_time.isoformat()
            })
            
            # Perform ELM training
            metrics = await self.elm.enhanced_train_batch(input_data, target_data, context)
            
            # Calculate ELM metrics
            elm_metrics = self.elm._calculate_elm_metrics(input_data, target_data, metrics)
            
            # Prepare result
            result = {
                'training_metrics': metrics,
                'elm_metrics': elm_metrics,
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'operation_success': metrics['mse'] != float('inf'),
                'session_id': self.session_id,
                'operation_count': self.operation_count
            }
            
            # Update performance history
            self.performance_history.append({
                'timestamp': start_time,
                'operation_type': 'elm_training',
                'metrics': {**metrics, **elm_metrics},
                'processing_time': result['processing_time']
            })
            
            logger.info(f"ELM training completed: {metrics['training_samples']} samples")
            return result
            
        except Exception as e:
            logger.error(f"ELM training failed: {e}")
            return {
                'training_metrics': {'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf'), 'training_samples': 0},
                'elm_metrics': {},
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'operation_success': False,
                'error': str(e),
                'session_id': self.session_id,
                'operation_count': self.operation_count
            }
    
    async def process_elm_online_update(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process ELM online update with monitoring and database integration.
        
        Args:
            input_data: Single input sample
            target_data: Single target sample
            context: Additional context information
            
        Returns:
            Dictionary containing update results and metadata
        """
        start_time = datetime.now()
        self.operation_count += 1
        
        try:
            # Add context information
            if context is None:
                context = {}
            
            context.update({
                'session_id': self.session_id,
                'operation_count': self.operation_count,
                'operation_type': 'elm_online_update',
                'timestamp': start_time.isoformat()
            })
            
            # Perform ELM online update
            metrics = await self.elm.enhanced_online_update(input_data, target_data, context)
            
            # Calculate online metrics
            online_metrics = self.elm._calculate_online_metrics(input_data, target_data, metrics)
            
            # Prepare result
            result = {
                'update_metrics': metrics,
                'online_metrics': online_metrics,
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'operation_success': metrics['mse'] != float('inf'),
                'session_id': self.session_id,
                'operation_count': self.operation_count
            }
            
            # Update performance history
            self.performance_history.append({
                'timestamp': start_time,
                'operation_type': 'elm_online_update',
                'metrics': {**metrics, **online_metrics},
                'processing_time': result['processing_time']
            })
            
            logger.debug(f"ELM online update completed: {metrics['update_samples']} samples")
            return result
            
        except Exception as e:
            logger.error(f"ELM online update failed: {e}")
            return {
                'update_metrics': {'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf'), 'update_samples': 0},
                'online_metrics': {},
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'operation_success': False,
                'error': str(e),
                'session_id': self.session_id,
                'operation_count': self.operation_count
            }
    
    async def process_elm_ensemble_training(
        self,
        training_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process ELM ensemble training with monitoring and database integration.
        
        Args:
            training_data: Dictionary mapping task names to (input, target) tuples
            context: Additional context information
            
        Returns:
            Dictionary containing ensemble training results and metadata
        """
        start_time = datetime.now()
        self.operation_count += 1
        
        try:
            # Add context information
            if context is None:
                context = {}
            
            context.update({
                'session_id': self.session_id,
                'operation_count': self.operation_count,
                'operation_type': 'elm_ensemble_training',
                'timestamp': start_time.isoformat()
            })
            
            # Perform ELM ensemble training
            metrics = self.elm_ensemble.train_ensemble(training_data)
            
            # Calculate ensemble metrics
            ensemble_metrics = self._calculate_ensemble_metrics(metrics)
            
            # Prepare result
            result = {
                'ensemble_metrics': metrics,
                'ensemble_performance': ensemble_metrics,
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'operation_success': True,
                'session_id': self.session_id,
                'operation_count': self.operation_count
            }
            
            # Update performance history
            self.performance_history.append({
                'timestamp': start_time,
                'operation_type': 'elm_ensemble_training',
                'metrics': {**metrics, **ensemble_metrics},
                'processing_time': result['processing_time']
            })
            
            logger.info(f"ELM ensemble training completed: {len(training_data)} tasks")
            return result
            
        except Exception as e:
            logger.error(f"ELM ensemble training failed: {e}")
            return {
                'ensemble_metrics': {},
                'ensemble_performance': {},
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'operation_success': False,
                'error': str(e),
                'session_id': self.session_id,
                'operation_count': self.operation_count
            }
    
    def _calculate_ensemble_metrics(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate ensemble performance metrics."""
        try:
            if not metrics:
                return {}
            
            # Calculate average performance across all ELMs
            all_mse = [m['mse'] for m in metrics.values() if 'mse' in m]
            all_mae = [m['mae'] for m in metrics.values() if 'mae' in m]
            all_rmse = [m['rmse'] for m in metrics.values() if 'rmse' in m]
            
            ensemble_performance = {
                'average_mse': float(np.mean(all_mse)) if all_mse else 0.0,
                'average_mae': float(np.mean(all_mae)) if all_mae else 0.0,
                'average_rmse': float(np.mean(all_rmse)) if all_rmse else 0.0,
                'ensemble_size': len(metrics),
                'performance_variance': float(np.var(all_mse)) if all_mse else 0.0
            }
            
            return ensemble_performance
            
        except Exception as e:
            logger.error(f"Failed to calculate ensemble metrics: {e}")
            return {}
    
    async def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for all learning systems."""
        try:
            # Get individual system metrics
            ewc_report = await self.ewc.get_enhanced_report()
            residual_report = await self.residual.get_enhanced_report()
            elm_report = await self.elm.get_enhanced_report()
            ensemble_report = self.elm_ensemble.get_ensemble_report()
            
            # Get cognitive subsystem metrics
            cognitive_metrics = {}
            if self.cognitive_coordinator:
                cognitive_metrics = await self.cognitive_coordinator.get_all_subsystem_metrics()
            
            # Calculate overall performance
            overall_performance = self._calculate_overall_performance()
            
            # Prepare comprehensive result
            result = {
                'session_info': {
                    'session_id': self.session_id,
                    'session_start_time': self.session_start_time.isoformat(),
                    'operation_count': self.operation_count,
                    'session_duration': (datetime.now() - self.session_start_time).total_seconds()
                },
                'ewc_metrics': ewc_report,
                'residual_metrics': residual_report,
                'elm_metrics': elm_report,
                'ensemble_metrics': ensemble_report,
                'cognitive_metrics': cognitive_metrics,
                'overall_performance': overall_performance,
                'performance_history': self.performance_history[-100:]  # Last 100 operations
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive metrics: {e}")
            return {
                'session_info': {
                    'session_id': self.session_id,
                    'error': str(e)
                }
            }
    
    def _calculate_overall_performance(self) -> Dict[str, float]:
        """Calculate overall system performance metrics."""
        try:
            if not self.performance_history:
                return {}
            
            # Calculate performance trends
            recent_operations = self.performance_history[-10:]  # Last 10 operations
            success_rate = sum(1 for op in recent_operations if op.get('operation_success', False)) / len(recent_operations)
            
            # Calculate average processing time
            avg_processing_time = np.mean([op.get('processing_time', 0) for op in recent_operations])
            
            # Calculate operation distribution
            operation_types = [op.get('operation_type', 'unknown') for op in recent_operations]
            operation_distribution = {op_type: operation_types.count(op_type) for op_type in set(operation_types)}
            
            return {
                'success_rate': success_rate,
                'average_processing_time': avg_processing_time,
                'total_operations': len(self.performance_history),
                'operation_distribution': operation_distribution
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate overall performance: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup resources and finalize session."""
        try:
            # Get final metrics
            final_metrics = await self.get_comprehensive_metrics()
            
            # Store session summary
            if self.integration:
                await self.integration.log_system_event(
                    LogLevel.INFO,
                    Component.DIRECTOR,
                    f"Enhanced Learning session completed: {self.session_id}",
                    final_metrics,
                    self.session_id
                )
            
            # Cleanup cognitive coordinator
            if self.cognitive_coordinator:
                await self.cognitive_coordinator.cleanup_all_subsystems()
            
            logger.info(f"Enhanced Learning Integration cleanup completed: {self.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Enhanced Learning Integration: {e}")

    async def process_learning_experience(self, learning_experience: Dict[str, Any]) -> None:
        """Process a learning experience from game results with lifecycle pattern analysis.

        Args:
            learning_experience: Dictionary containing game result data for learning
        """
        try:
            # Extract key metrics from the learning experience
            game_score = learning_experience.get('final_score', 0.0)
            game_won = learning_experience.get('game_won', False)
            total_actions = learning_experience.get('total_actions', 0)
            game_duration = learning_experience.get('game_duration', 0.0)

            # Extract lifecycle pattern data if available
            lifecycle_patterns = learning_experience.get('lifecycle_patterns', {})
            failure_risk_score = lifecycle_patterns.get('failure_risk_score', 0.0)
            oscillation_detected = lifecycle_patterns.get('oscillation_detected', False)
            action_effectiveness = lifecycle_patterns.get('action_effectiveness', {})

            # Create enhanced learning input data with lifecycle features
            import torch
            base_features = [
                game_score,
                1.0 if game_won else 0.0,
                total_actions,
                game_duration,
                learning_experience.get('hypotheses_generated', 0)
            ]

            # Add lifecycle pattern features
            lifecycle_features = [
                failure_risk_score,
                1.0 if oscillation_detected else 0.0,
                len(action_effectiveness),  # Number of action effectiveness patterns
                sum(action_effectiveness.values()) / max(len(action_effectiveness), 1)  # Average effectiveness
            ]

            learning_input = torch.tensor(base_features + lifecycle_features, dtype=torch.float32).unsqueeze(0)

            # Create enhanced target for learning (success indicator with failure pattern weighting)
            success_score = 1.0 if game_won or game_score > 50 else 0.0
            # Reduce target score if high failure risk was detected but game still succeeded
            if success_score > 0 and failure_risk_score > 0.7:
                success_score *= 0.8  # Reduce success weight for risky success
            learning_target = torch.tensor([success_score], dtype=torch.float32).unsqueeze(0)

            # Process through EWC if available
            if self.ewc:
                await self.process_ewc_consolidation(learning_input, learning_target)

            # Process through residual learning if available
            if self.residual:
                await self.process_residual_forward_pass(learning_input, learning_target)

            # Process through ELM if available
            if self.elm:
                await self.process_elm_training(learning_input, learning_target)

            # Process through ELM ensemble if available
            if self.elm_ensemble:
                await self.process_elm_ensemble_training(learning_input, learning_target)

            # Update operation count and metrics
            self.operation_count += 1

            # Store enhanced performance metrics with lifecycle patterns
            performance_metrics = {
                'game_score': game_score,
                'game_won': game_won,
                'total_actions': total_actions,
                'game_duration': game_duration,
                'operation_count': self.operation_count,
                'timestamp': learning_experience.get('timestamp'),
                # Lifecycle pattern metrics
                'failure_risk_score': failure_risk_score,
                'oscillation_detected': oscillation_detected,
                'action_effectiveness_count': len(action_effectiveness),
                'avg_action_effectiveness': sum(action_effectiveness.values()) / max(len(action_effectiveness), 1)
            }

            self.performance_history.append(performance_metrics)

            # Update enhanced learning metrics with lifecycle insights
            self.learning_metrics.update({
                'total_experiences_processed': self.operation_count,
                'average_game_score': sum(exp['game_score'] for exp in self.performance_history) / len(self.performance_history),
                'win_rate': sum(1 for exp in self.performance_history if exp['game_won']) / len(self.performance_history),
                'average_failure_risk': sum(exp.get('failure_risk_score', 0) for exp in self.performance_history) / len(self.performance_history),
                'oscillation_frequency': sum(1 for exp in self.performance_history if exp.get('oscillation_detected', False)) / len(self.performance_history),
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
