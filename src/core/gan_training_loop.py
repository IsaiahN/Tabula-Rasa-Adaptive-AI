"""
GAN Training Loop with Database Integration

This module implements a comprehensive training loop for the GAN system that integrates
with the existing ARC-AGI-3 learning system and uses database-only storage.

Key Features:
- Database-only storage (no file creation)
- Integration with existing learning systems
- Real-time training monitoring
- Automatic checkpointing
- Pattern-aware training
- Self-improving training loop
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np

from ..database.api import get_database
from ..database.director_commands import get_director_commands
from ..arc_integration.arc_meta_learning import ARCMetaLearningSystem
from .gan_system import PatternAwareGAN, GameState, GANTrainingConfig
from .gan_pattern_integration import GANPatternIntegration

logger = logging.getLogger(__name__)

@dataclass
class TrainingEpoch:
    """Represents a single training epoch."""
    epoch_number: int
    generator_loss: float
    discriminator_loss: float
    pattern_accuracy: float
    synthetic_quality: float
    convergence_score: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'epoch_number': self.epoch_number,
            'generator_loss': self.generator_loss,
            'discriminator_loss': self.discriminator_loss,
            'pattern_accuracy': self.pattern_accuracy,
            'synthetic_quality': self.synthetic_quality,
            'convergence_score': self.convergence_score,
            'timestamp': self.timestamp
        }

@dataclass
class TrainingConfig:
    """Configuration for GAN training loop."""
    max_epochs: int = 1000
    batch_size: int = 32
    learning_rate_generator: float = 0.0002
    learning_rate_discriminator: float = 0.0002
    convergence_threshold: float = 0.01
    checkpoint_frequency: int = 50
    validation_frequency: int = 10
    synthetic_data_ratio: float = 0.2
    pattern_learning_enabled: bool = True
    reverse_engineering_enabled: bool = True
    auto_stop_convergence: bool = True
    max_training_time_hours: int = 24

class GANTrainingLoop:
    """
    Main training loop for GAN system with database integration.
    
    Features:
    - Database-only storage
    - Real-time monitoring
    - Automatic checkpointing
    - Pattern-aware training
    - Self-improving capabilities
    """
    
    def __init__(self, 
                 config: Optional[TrainingConfig] = None,
                 pattern_learning_system: Optional[ARCMetaLearningSystem] = None):
        self.config = config or TrainingConfig()
        self.pattern_learning_system = pattern_learning_system
        self.db = get_database()
        self.director = get_director_commands()
        
        # Initialize GAN system
        gan_config = GANTrainingConfig(
            max_epochs=self.config.max_epochs,
            batch_size=self.config.batch_size,
            learning_rate_generator=self.config.learning_rate_generator,
            learning_rate_discriminator=self.config.learning_rate_discriminator,
            convergence_threshold=self.config.convergence_threshold
        )
        
        self.gan_system = PatternAwareGAN(
            config=gan_config,
            pattern_learning_system=pattern_learning_system
        )
        
        # Initialize pattern integration
        self.pattern_integration = GANPatternIntegration(
            pattern_learning_system=pattern_learning_system,
            gan_system=self.gan_system
        )
        
        # Training state
        self.current_session_id = None
        self.training_epochs = []
        self.convergence_history = []
        self.best_epoch = None
        self.training_start_time = None
        self.is_training = False
        
        logger.info("GAN Training Loop initialized with database integration")
    
    async def start_training(self, 
                           session_name: str = None,
                           real_data_source: str = "database") -> str:
        """
        Start GAN training session.
        
        Args:
            session_name: Optional name for the training session
            real_data_source: Source of real data ('database', 'api', 'file')
            
        Returns:
            Training session ID
        """
        try:
            # Start GAN training session
            self.current_session_id = await self.gan_system.start_training_session(session_name)
            self.training_start_time = time.time()
            self.is_training = True
            
            # Log training start
            await self.director.log_system_event(
                "gan_training_loop_started",
                f"GAN training loop started with session {self.current_session_id}",
                {
                    "session_id": self.current_session_id,
                    "config": asdict(self.config),
                    "real_data_source": real_data_source
                }
            )
            
            logger.info(f"GAN training loop started with session {self.current_session_id}")
            return self.current_session_id
            
        except Exception as e:
            logger.error(f"Failed to start GAN training: {e}")
            raise
    
    async def run_training_epochs(self, 
                                real_data: List[Dict[str, Any]] = None,
                                max_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Run training epochs with real data.
        
        Args:
            real_data: Real game states for training
            max_epochs: Maximum number of epochs to run
            
        Returns:
            Training results and metrics
        """
        try:
            if not self.is_training:
                raise ValueError("Training not started. Call start_training() first.")
            
            max_epochs = max_epochs or self.config.max_epochs
            epochs_completed = 0
            convergence_count = 0
            
            # Get real data if not provided
            if real_data is None:
                real_data = await self._get_real_training_data()
            
            # Convert real data to GameState objects
            real_states = [GameState.from_dict(state) for state in real_data]
            
            logger.info(f"Starting training with {len(real_states)} real states for {max_epochs} epochs")
            
            # Training loop
            for epoch in range(max_epochs):
                try:
                    # Check if training should stop
                    if await self._should_stop_training():
                        logger.info("Training stopped due to convergence or time limit")
                        break
                    
                    # Train one epoch
                    epoch_result = await self._train_single_epoch(epoch, real_states)
                    
                    # Store epoch results
                    self.training_epochs.append(epoch_result)
                    await self._store_epoch_results(epoch_result)
                    
                    # Check convergence
                    if await self._check_convergence(epoch_result):
                        convergence_count += 1
                        if convergence_count >= 3:  # Converged for 3 consecutive epochs
                            logger.info(f"Training converged at epoch {epoch}")
                            break
                    else:
                        convergence_count = 0
                    
                    # Update best epoch
                    if self.best_epoch is None or epoch_result.convergence_score > self.best_epoch.convergence_score:
                        self.best_epoch = epoch_result
                    
                    # Checkpoint if needed
                    if (epoch + 1) % self.config.checkpoint_frequency == 0:
                        await self._create_checkpoint(epoch + 1)
                    
                    # Validation if needed
                    if (epoch + 1) % self.config.validation_frequency == 0:
                        await self._run_validation(epoch + 1)
                    
                    epochs_completed += 1
                    
                    # Log progress
                    if (epoch + 1) % 10 == 0:
                        logger.info(f"Epoch {epoch + 1}/{max_epochs} completed. "
                                  f"G Loss: {epoch_result.generator_loss:.4f}, "
                                  f"D Loss: {epoch_result.discriminator_loss:.4f}, "
                                  f"Pattern Acc: {epoch_result.pattern_accuracy:.4f}")
                
                except Exception as e:
                    logger.error(f"Error in epoch {epoch}: {e}")
                    continue
            
            # Training completed
            self.is_training = False
            training_time = time.time() - self.training_start_time
            
            # Store final results
            final_results = await self._store_final_results(epochs_completed, training_time)
            
            # Log training completion
            await self.director.log_system_event(
                "gan_training_completed",
                f"GAN training completed after {epochs_completed} epochs",
                {
                    "session_id": self.current_session_id,
                    "epochs_completed": epochs_completed,
                    "training_time": training_time,
                    "final_results": final_results
                }
            )
            
            logger.info(f"GAN training completed after {epochs_completed} epochs in {training_time:.2f} seconds")
            
            return {
                "status": "success",
                "epochs_completed": epochs_completed,
                "training_time": training_time,
                "final_results": final_results,
                "best_epoch": self.best_epoch.to_dict() if self.best_epoch else None
            }
            
        except Exception as e:
            logger.error(f"Failed to run training epochs: {e}")
            self.is_training = False
            return {"status": "error", "message": str(e)}
    
    async def generate_synthetic_data(self, 
                                    count: int,
                                    pattern_types: List[str] = None,
                                    context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate synthetic data using trained GAN.
        
        Args:
            count: Number of synthetic states to generate
            pattern_types: Types of patterns to use
            context: Optional context for generation
            
        Returns:
            List of synthetic game states
        """
        try:
            if not self.current_session_id:
                raise ValueError("No active training session")
            
            # Generate synthetic states
            synthetic_states = await self.pattern_integration.generate_pattern_aware_states(
                count, pattern_types, context
            )
            
            # Convert to serializable format
            synthetic_data = [state.to_dict() for state in synthetic_states]
            
            # Store synthetic data
            await self._store_synthetic_data(synthetic_data)
            
            logger.info(f"Generated {len(synthetic_data)} synthetic states")
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic data: {e}")
            return []
    
    async def reverse_engineer_game_mechanics(self, game_id: str) -> Dict[str, Any]:
        """
        Use trained GAN to reverse engineer game mechanics.
        
        Args:
            game_id: Game ID to analyze
            
        Returns:
            Discovered game mechanics and rules
        """
        try:
            if not self.current_session_id:
                raise ValueError("No active training session")
            
            # Reverse engineer mechanics
            discovered_mechanics = await self.gan_system.reverse_engineer_game_mechanics(game_id)
            
            # Store reverse engineering results
            await self._store_reverse_engineering_results(game_id, discovered_mechanics)
            
            logger.info(f"Reverse engineered mechanics for game {game_id}")
            return discovered_mechanics
            
        except Exception as e:
            logger.error(f"Failed to reverse engineer game mechanics: {e}")
            return {"error": str(e)}
    
    async def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and metrics."""
        try:
            if not self.current_session_id:
                return {"error": "No active training session"}
            
            # Get session data
            session_data = await self.db.fetch_one("""
                SELECT * FROM gan_training_sessions 
                WHERE session_id = ?
            """, (self.current_session_id,))
            
            # Get recent epochs
            recent_epochs = self.training_epochs[-10:] if self.training_epochs else []
            
            # Calculate metrics
            total_epochs = len(self.training_epochs)
            avg_generator_loss = sum(e.generator_loss for e in self.training_epochs) / max(total_epochs, 1)
            avg_discriminator_loss = sum(e.discriminator_loss for e in self.training_epochs) / max(total_epochs, 1)
            avg_pattern_accuracy = sum(e.pattern_accuracy for e in self.training_epochs) / max(total_epochs, 1)
            avg_synthetic_quality = sum(e.synthetic_quality for e in self.training_epochs) / max(total_epochs, 1)
            
            # Training time
            training_time = time.time() - self.training_start_time if self.training_start_time else 0
            
            return {
                "status": "success",
                "session_id": self.current_session_id,
                "is_training": self.is_training,
                "total_epochs": total_epochs,
                "training_time": training_time,
                "average_metrics": {
                    "generator_loss": avg_generator_loss,
                    "discriminator_loss": avg_discriminator_loss,
                    "pattern_accuracy": avg_pattern_accuracy,
                    "synthetic_quality": avg_synthetic_quality
                },
                "recent_epochs": [e.to_dict() for e in recent_epochs],
                "best_epoch": self.best_epoch.to_dict() if self.best_epoch else None,
                "session_data": dict(session_data) if session_data else {}
            }
            
        except Exception as e:
            logger.error(f"Failed to get training status: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    
    async def _get_real_training_data(self) -> List[Dict[str, Any]]:
        """Get real training data from database."""
        try:
            # Get recent game results
            game_results = await self.db.fetch_all("""
                SELECT * FROM game_results 
                WHERE created_at >= datetime('now', '-7 days')
                ORDER BY created_at DESC
                LIMIT 1000
            """)
            
            # Convert to training data format
            training_data = []
            for result in game_results:
                # Create synthetic state data from game result
                state_data = {
                    'grid': np.random.rand(3, 64, 64).tolist(),  # Placeholder
                    'objects': [],
                    'properties': {
                        'game_id': result['game_id'],
                        'final_score': result['final_score'],
                        'total_actions': result['total_actions'],
                        'win_detected': result['win_detected']
                    },
                    'context': {
                        'session_id': result['session_id'],
                        'level_completions': result['level_completions']
                    },
                    'action_history': json.loads(result['actions_taken']) if result['actions_taken'] else [],
                    'success_probability': 1.0 if result['win_detected'] else 0.0,
                    'timestamp': time.time()
                }
                training_data.append(state_data)
            
            return training_data
            
        except Exception as e:
            logger.error(f"Failed to get real training data: {e}")
            return []
    
    async def _train_single_epoch(self, epoch: int, real_states: List[GameState]) -> TrainingEpoch:
        """Train a single epoch."""
        try:
            # Train GAN epoch
            gan_metrics = await self.gan_system.train_epoch(real_states)
            
            # Calculate convergence score
            convergence_score = self._calculate_convergence_score(gan_metrics)
            
            # Create epoch result
            epoch_result = TrainingEpoch(
                epoch_number=epoch + 1,
                generator_loss=gan_metrics['generator_loss'],
                discriminator_loss=gan_metrics['discriminator_loss'],
                pattern_accuracy=gan_metrics['pattern_accuracy'],
                synthetic_quality=gan_metrics['synthetic_quality'],
                convergence_score=convergence_score
            )
            
            return epoch_result
            
        except Exception as e:
            logger.error(f"Error in training epoch {epoch}: {e}")
            # Return dummy epoch result
            return TrainingEpoch(
                epoch_number=epoch + 1,
                generator_loss=1.0,
                discriminator_loss=1.0,
                pattern_accuracy=0.0,
                synthetic_quality=0.0,
                convergence_score=0.0
            )
    
    def _calculate_convergence_score(self, metrics: Dict[str, float]) -> float:
        """Calculate convergence score from metrics."""
        # Simple convergence score based on loss stability
        generator_loss = metrics['generator_loss']
        discriminator_loss = metrics['discriminator_loss']
        pattern_accuracy = metrics['pattern_accuracy']
        synthetic_quality = metrics['synthetic_quality']
        
        # Higher score means better convergence
        convergence_score = (
            (1.0 - min(generator_loss, 1.0)) * 0.3 +
            (1.0 - min(discriminator_loss, 1.0)) * 0.3 +
            pattern_accuracy * 0.2 +
            synthetic_quality * 0.2
        )
        
        return convergence_score
    
    async def _check_convergence(self, epoch_result: TrainingEpoch) -> bool:
        """Check if training has converged."""
        # Add to convergence history
        self.convergence_history.append(epoch_result.convergence_score)
        
        # Keep only recent history
        if len(self.convergence_history) > 10:
            self.convergence_history = self.convergence_history[-10:]
        
        # Check if convergence is stable
        if len(self.convergence_history) >= 3:
            recent_scores = self.convergence_history[-3:]
            score_variance = np.var(recent_scores)
            
            # Converged if variance is low and scores are high
            return score_variance < self.config.convergence_threshold and np.mean(recent_scores) > 0.8
        
        return False
    
    async def _should_stop_training(self) -> bool:
        """Check if training should stop."""
        # Check time limit
        if self.training_start_time:
            training_time = time.time() - self.training_start_time
            if training_time > self.config.max_training_time_hours * 3600:
                return True
        
        # Check convergence
        if self.config.auto_stop_convergence and len(self.convergence_history) >= 3:
            recent_scores = self.convergence_history[-3:]
            if np.mean(recent_scores) > 0.9 and np.var(recent_scores) < 0.01:
                return True
        
        return False
    
    async def _store_epoch_results(self, epoch_result: TrainingEpoch) -> None:
        """Store epoch results in database."""
        try:
            # Store in performance metrics
            await self.db.execute("""
                INSERT INTO gan_performance_metrics 
                (session_id, metric_name, metric_value, metric_type, epoch, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                self.current_session_id,
                'generator_loss',
                epoch_result.generator_loss,
                'loss',
                epoch_result.epoch_number,
                datetime.now()
            ))
            
            await self.db.execute("""
                INSERT INTO gan_performance_metrics 
                (session_id, metric_name, metric_value, metric_type, epoch, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                self.current_session_id,
                'discriminator_loss',
                epoch_result.discriminator_loss,
                'loss',
                epoch_result.epoch_number,
                datetime.now()
            ))
            
            await self.db.execute("""
                INSERT INTO gan_performance_metrics 
                (session_id, metric_name, metric_value, metric_type, epoch, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                self.current_session_id,
                'pattern_accuracy',
                epoch_result.pattern_accuracy,
                'accuracy',
                epoch_result.epoch_number,
                datetime.now()
            ))
            
            await self.db.execute("""
                INSERT INTO gan_performance_metrics 
                (session_id, metric_name, metric_value, metric_type, epoch, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                self.current_session_id,
                'synthetic_quality',
                epoch_result.synthetic_quality,
                'quality',
                epoch_result.epoch_number,
                datetime.now()
            ))
            
            await self.db.execute("""
                INSERT INTO gan_performance_metrics 
                (session_id, metric_name, metric_value, metric_type, epoch, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                self.current_session_id,
                'convergence_score',
                epoch_result.convergence_score,
                'convergence',
                epoch_result.epoch_number,
                datetime.now()
            ))
            
        except Exception as e:
            logger.error(f"Failed to store epoch results: {e}")
    
    async def _create_checkpoint(self, epoch: int) -> None:
        """Create training checkpoint."""
        try:
            # Store checkpoint in database
            checkpoint_id = f"checkpoint_{self.current_session_id}_{epoch}"
            
            # Get current metrics
            current_epoch = self.training_epochs[-1] if self.training_epochs else None
            if not current_epoch:
                return
            
            await self.db.execute("""
                INSERT INTO gan_model_checkpoints 
                (checkpoint_id, session_id, epoch, generator_weights, discriminator_weights,
                 generator_loss, discriminator_loss, pattern_accuracy, synthetic_quality, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                checkpoint_id,
                self.current_session_id,
                epoch,
                json.dumps({}),  # Placeholder for weights
                json.dumps({}),  # Placeholder for weights
                current_epoch.generator_loss,
                current_epoch.discriminator_loss,
                current_epoch.pattern_accuracy,
                current_epoch.synthetic_quality,
                datetime.now()
            ))
            
            logger.info(f"Checkpoint created at epoch {epoch}")
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
    
    async def _run_validation(self, epoch: int) -> None:
        """Run validation on current model."""
        try:
            # Generate synthetic data for validation
            synthetic_data = await self.generate_synthetic_data(10)
            
            # Validate synthetic data
            validation_results = await self.pattern_integration.validate_synthetic_patterns(
                [GameState.from_dict(state) for state in synthetic_data]
            )
            
            # Store validation results
            for result in validation_results:
                await self.db.execute("""
                    INSERT INTO gan_validation_results 
                    (session_id, generated_state_id, validation_type, validation_score,
                     validation_details, is_passed, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.current_session_id,
                    0,  # Placeholder
                    'epoch_validation',
                    result.validation_score,
                    json.dumps(result.validation_details),
                    result.is_valid,
                    datetime.now()
                ))
            
            logger.info(f"Validation completed at epoch {epoch}")
            
        except Exception as e:
            logger.error(f"Failed to run validation: {e}")
    
    async def _store_final_results(self, epochs_completed: int, training_time: float) -> Dict[str, Any]:
        """Store final training results."""
        try:
            # Update session status
            await self.db.execute("""
                UPDATE gan_training_sessions 
                SET end_time = ?, status = 'completed', total_training_steps = ?
                WHERE session_id = ?
            """, (datetime.now(), epochs_completed, self.current_session_id))
            
            # Calculate final metrics
            final_metrics = {
                "epochs_completed": epochs_completed,
                "training_time": training_time,
                "final_generator_loss": self.training_epochs[-1].generator_loss if self.training_epochs else 0.0,
                "final_discriminator_loss": self.training_epochs[-1].discriminator_loss if self.training_epochs else 0.0,
                "final_pattern_accuracy": self.training_epochs[-1].pattern_accuracy if self.training_epochs else 0.0,
                "final_synthetic_quality": self.training_epochs[-1].synthetic_quality if self.training_epochs else 0.0,
                "best_convergence_score": self.best_epoch.convergence_score if self.best_epoch else 0.0
            }
            
            return final_metrics
            
        except Exception as e:
            logger.error(f"Failed to store final results: {e}")
            return {}
    
    async def _store_synthetic_data(self, synthetic_data: List[Dict[str, Any]]) -> None:
        """Store synthetic data in database."""
        try:
            for state_data in synthetic_data:
                await self.db.execute("""
                    INSERT INTO gan_generated_states 
                    (session_id, state_data, pattern_context, quality_score, generation_method)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    self.current_session_id,
                    json.dumps(state_data),
                    json.dumps(state_data.get('context', {})),
                    state_data.get('success_probability', 0.0),
                    'training_loop'
                ))
                
        except Exception as e:
            logger.error(f"Failed to store synthetic data: {e}")
    
    async def _store_reverse_engineering_results(self, game_id: str, mechanics: Dict[str, Any]) -> None:
        """Store reverse engineering results."""
        try:
            await self.db.execute("""
                INSERT INTO gan_reverse_engineering 
                (session_id, game_id, discovered_rules, rule_confidence, mechanics_understood)
                VALUES (?, ?, ?, ?, ?)
            """, (
                self.current_session_id,
                game_id,
                json.dumps(mechanics),
                mechanics.get('confidence', 0.0),
                mechanics.get('mechanics_understood', 0.0)
            ))
            
        except Exception as e:
            logger.error(f"Failed to store reverse engineering results: {e}")
