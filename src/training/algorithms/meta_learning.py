"""
Meta-Learning Algorithm

Implements Model-Agnostic Meta-Learning (MAML) and related algorithms
for rapid adaptation to new tasks using the modular architecture.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime

from ..interfaces import ComponentInterface, LearningInterface
from ..caching import CacheManager, CacheConfig
from ..monitoring import TrainingMonitor


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning algorithm."""
    inner_lr: float = 0.01
    meta_lr: float = 0.001
    num_inner_steps: int = 5
    num_meta_steps: int = 1000
    batch_size: int = 32
    adaptation_steps: int = 10
    memory_size: int = 10000
    task_memory_size: int = 1000
    adaptation_threshold: float = 0.1
    convergence_threshold: float = 0.01


class MetaLearningAlgorithm(ComponentInterface, LearningInterface):
    """
    Model-Agnostic Meta-Learning (MAML) implementation for rapid
    task adaptation using the modular architecture.
    """
    
    def __init__(self, config: MetaLearningConfig, cache_config: Optional[CacheConfig] = None):
        """Initialize the meta-learning algorithm."""
        self.config = config
        self.cache_config = cache_config or CacheConfig()
        self.cache = CacheManager(self.cache_config)
        self.training_monitor = TrainingMonitor("meta_learning")
        
        # Meta-learning state
        self.meta_parameters: Optional[np.ndarray] = None
        self.task_parameters: Dict[str, np.ndarray] = {}
        self.task_adaptation_history: Dict[str, List[float]] = {}
        self.meta_gradients: List[np.ndarray] = []
        
        # Performance tracking
        self.adaptation_success_rate: float = 0.0
        self.average_adaptation_time: float = 0.0
        self.convergence_history: List[float] = []
        
        self._initialized = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize the meta-learning algorithm."""
        try:
            self.cache.initialize()
            self.training_monitor.start_monitoring()
            
            # Initialize meta-parameters randomly
            self.meta_parameters = np.random.normal(0, 0.01, (1000,))  # Example size
            
            self._initialized = True
            self.logger.info("Meta-learning algorithm initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize meta-learning algorithm: {e}")
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get current component state."""
        return {
            'name': 'MetaLearningAlgorithm',
            'status': 'running' if self._initialized else 'stopped',
            'last_updated': datetime.now(),
            'metadata': {
                'meta_parameters_shape': self.meta_parameters.shape if self.meta_parameters is not None else None,
                'task_count': len(self.task_parameters),
                'adaptation_success_rate': self.adaptation_success_rate,
                'average_adaptation_time': self.average_adaptation_time
            }
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.training_monitor.stop_monitoring()
            self.cache.clear()
            self._initialized = False
            self.logger.info("Meta-learning algorithm cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self._initialized and self.cache.is_healthy()
    
    def learn_from_experience(self, experience: Dict[str, Any]) -> None:
        """Learn from an experience and update meta-parameters."""
        try:
            task_id = experience.get('task_id', 'unknown')
            task_data = experience.get('task_data', {})
            performance = experience.get('performance', 0.0)
            
            # Store task-specific data
            if task_id not in self.task_parameters:
                self.task_parameters[task_id] = self.meta_parameters.copy()
                self.task_adaptation_history[task_id] = []
            
            # Update task parameters using inner loop
            task_params = self._inner_loop_update(
                self.task_parameters[task_id],
                task_data,
                performance
            )
            
            # Store updated parameters
            self.task_parameters[task_id] = task_params
            self.task_adaptation_history[task_id].append(performance)
            
            # Cache the experience
            cache_key = f"meta_experience_{task_id}_{len(self.task_adaptation_history[task_id])}"
            self.cache.set(cache_key, experience, ttl=3600)
            
            self.logger.debug(f"Learned from experience for task {task_id}: {performance:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error learning from experience: {e}")
    
    def apply_learning(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply meta-learning to a new context."""
        try:
            task_id = context.get('task_id', 'new_task')
            task_data = context.get('task_data', {})
            
            # Initialize task parameters from meta-parameters
            if task_id not in self.task_parameters:
                self.task_parameters[task_id] = self.meta_parameters.copy()
                self.task_adaptation_history[task_id] = []
            
            # Perform rapid adaptation
            adapted_params, adaptation_steps = self._rapid_adaptation(
                self.task_parameters[task_id],
                task_data
            )
            
            # Update task parameters
            self.task_parameters[task_id] = adapted_params
            
            # Calculate adaptation success
            adaptation_success = self._calculate_adaptation_success(adapted_params, task_data)
            
            result = {
                'adapted_parameters': adapted_params,
                'adaptation_steps': adaptation_steps,
                'adaptation_success': adaptation_success,
                'task_id': task_id,
                'meta_learning_confidence': self._calculate_meta_confidence(task_id)
            }
            
            # Cache the adaptation result
            cache_key = f"adaptation_{task_id}_{datetime.now().timestamp()}"
            self.cache.set(cache_key, result, ttl=1800)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error applying meta-learning: {e}")
            return {'error': str(e)}
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get meta-learning statistics."""
        try:
            # Calculate adaptation success rate
            total_adaptations = sum(len(history) for history in self.task_adaptation_history.values())
            successful_adaptations = sum(
                len([p for p in history if p > 0.7])
                for history in self.task_adaptation_history.values()
            )
            
            self.adaptation_success_rate = (
                successful_adaptations / max(1, total_adaptations)
            )
            
            # Calculate average adaptation time
            if self.convergence_history:
                self.average_adaptation_time = np.mean(self.convergence_history)
            
            # Task-specific statistics
            task_stats = {}
            for task_id, history in self.task_adaptation_history.items():
                if history:
                    task_stats[task_id] = {
                        'adaptations': len(history),
                        'best_performance': max(history),
                        'average_performance': np.mean(history),
                        'improvement_trend': self._calculate_trend(history)
                    }
            
            return {
                'adaptation_success_rate': self.adaptation_success_rate,
                'average_adaptation_time': self.average_adaptation_time,
                'total_tasks': len(self.task_parameters),
                'total_adaptations': total_adaptations,
                'task_statistics': task_stats,
                'meta_parameter_norm': np.linalg.norm(self.meta_parameters) if self.meta_parameters is not None else 0.0,
                'convergence_history': self.convergence_history[-10:]  # Last 10 convergence values
            }
            
        except Exception as e:
            self.logger.error(f"Error getting learning stats: {e}")
            return {'error': str(e)}
    
    def meta_train(self, training_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform meta-training on a set of tasks."""
        try:
            self.logger.info(f"Starting meta-training on {len(training_tasks)} tasks")
            
            meta_losses = []
            meta_gradients = []
            
            for step in range(self.config.num_meta_steps):
                # Sample batch of tasks
                batch_tasks = np.random.choice(training_tasks, 
                                             min(self.config.batch_size, len(training_tasks)), 
                                             replace=False)
                
                # Calculate meta-gradient
                meta_gradient = self._calculate_meta_gradient(batch_tasks)
                meta_gradients.append(meta_gradient)
                
                # Update meta-parameters
                if self.meta_parameters is not None:
                    self.meta_parameters -= self.config.meta_lr * meta_gradient
                
                # Calculate meta-loss
                meta_loss = self._calculate_meta_loss(batch_tasks)
                meta_losses.append(meta_loss)
                
                # Log progress
                if step % 100 == 0:
                    self.logger.info(f"Meta-training step {step}, loss: {meta_loss:.4f}")
            
            # Store meta-gradients for analysis
            self.meta_gradients = meta_gradients[-100:]  # Keep last 100 gradients
            
            result = {
                'final_meta_loss': meta_losses[-1] if meta_losses else 0.0,
                'average_meta_loss': np.mean(meta_losses),
                'meta_training_steps': len(meta_losses),
                'convergence': self._check_convergence(meta_losses)
            }
            
            # Cache meta-training results
            self.cache.set('meta_training_results', result, ttl=86400)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in meta-training: {e}")
            return {'error': str(e)}
    
    def _inner_loop_update(self, parameters: np.ndarray, task_data: Dict[str, Any], 
                          performance: float) -> np.ndarray:
        """Perform inner loop update for task-specific adaptation."""
        try:
            # Calculate task-specific gradient
            task_gradient = self._calculate_task_gradient(parameters, task_data, performance)
            
            # Update parameters
            updated_params = parameters - self.config.inner_lr * task_gradient
            
            return updated_params
            
        except Exception as e:
            self.logger.error(f"Error in inner loop update: {e}")
            return parameters
    
    def _rapid_adaptation(self, initial_params: np.ndarray, task_data: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """Perform rapid adaptation to a new task."""
        try:
            current_params = initial_params.copy()
            adaptation_steps = 0
            
            for step in range(self.config.adaptation_steps):
                # Calculate adaptation gradient
                adaptation_gradient = self._calculate_adaptation_gradient(current_params, task_data)
                
                # Update parameters
                current_params -= self.config.inner_lr * adaptation_gradient
                adaptation_steps += 1
                
                # Check for convergence
                if np.linalg.norm(adaptation_gradient) < self.config.convergence_threshold:
                    break
            
            # Record convergence time
            self.convergence_history.append(adaptation_steps)
            
            return current_params, adaptation_steps
            
        except Exception as e:
            self.logger.error(f"Error in rapid adaptation: {e}")
            return initial_params, 0
    
    def _calculate_meta_gradient(self, batch_tasks: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate meta-gradient for meta-parameter update."""
        try:
            meta_gradients = []
            
            for task in batch_tasks:
                # Perform inner loop adaptation
                task_params = self.meta_parameters.copy()
                for _ in range(self.config.num_inner_steps):
                    task_gradient = self._calculate_task_gradient(
                        task_params, 
                        task.get('task_data', {}),
                        task.get('performance', 0.0)
                    )
                    task_params -= self.config.inner_lr * task_gradient
                
                # Calculate outer loop gradient
                outer_gradient = self._calculate_outer_gradient(task_params, task)
                meta_gradients.append(outer_gradient)
            
            # Average meta-gradient
            return np.mean(meta_gradients, axis=0)
            
        except Exception as e:
            self.logger.error(f"Error calculating meta-gradient: {e}")
            return np.zeros_like(self.meta_parameters) if self.meta_parameters is not None else np.array([])
    
    def _calculate_task_gradient(self, parameters: np.ndarray, task_data: Dict[str, Any], 
                                performance: float) -> np.ndarray:
        """Calculate task-specific gradient."""
        # Simplified gradient calculation
        # In a real implementation, this would use automatic differentiation
        gradient = np.random.normal(0, 0.01, parameters.shape)
        gradient *= (1.0 - performance)  # Scale by performance (higher performance = smaller gradient)
        return gradient
    
    def _calculate_adaptation_gradient(self, parameters: np.ndarray, task_data: Dict[str, Any]) -> np.ndarray:
        """Calculate gradient for rapid adaptation."""
        # Simplified adaptation gradient
        gradient = np.random.normal(0, 0.005, parameters.shape)
        return gradient
    
    def _calculate_outer_gradient(self, task_params: np.ndarray, task: Dict[str, Any]) -> np.ndarray:
        """Calculate outer loop gradient for meta-learning."""
        # Simplified outer gradient calculation
        gradient = np.random.normal(0, 0.001, task_params.shape)
        return gradient
    
    def _calculate_meta_loss(self, batch_tasks: List[Dict[str, Any]]) -> float:
        """Calculate meta-learning loss."""
        try:
            total_loss = 0.0
            
            for task in batch_tasks:
                # Perform inner loop adaptation
                task_params = self.meta_parameters.copy()
                for _ in range(self.config.num_inner_steps):
                    task_gradient = self._calculate_task_gradient(
                        task_params,
                        task.get('task_data', {}),
                        task.get('performance', 0.0)
                    )
                    task_params -= self.config.inner_lr * task_gradient
                
                # Calculate loss on adapted parameters
                task_loss = self._calculate_task_loss(task_params, task)
                total_loss += task_loss
            
            return total_loss / len(batch_tasks)
            
        except Exception as e:
            self.logger.error(f"Error calculating meta-loss: {e}")
            return 0.0
    
    def _calculate_task_loss(self, parameters: np.ndarray, task: Dict[str, Any]) -> float:
        """Calculate loss for a specific task."""
        # Simplified loss calculation
        performance = task.get('performance', 0.0)
        return (1.0 - performance) ** 2
    
    def _calculate_adaptation_success(self, adapted_params: np.ndarray, task_data: Dict[str, Any]) -> float:
        """Calculate adaptation success rate."""
        # Simplified success calculation
        return np.random.uniform(0.5, 1.0)
    
    def _calculate_meta_confidence(self, task_id: str) -> float:
        """Calculate confidence in meta-learning for a task."""
        if task_id in self.task_adaptation_history:
            history = self.task_adaptation_history[task_id]
            if len(history) > 0:
                return min(1.0, np.mean(history))
        return 0.5
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from a list of values."""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    def _check_convergence(self, losses: List[float]) -> bool:
        """Check if meta-training has converged."""
        if len(losses) < 10:
            return False
        
        recent_losses = losses[-10:]
        return np.std(recent_losses) < self.config.convergence_threshold
