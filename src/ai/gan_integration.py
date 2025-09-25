"""
GAN Integration Module for ARC-AGI-3 System
==========================================

This module integrates the GAN-based game mimicking system with the existing
continuous learning loop, providing enhanced action prediction and game state
understanding.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import torch

from .gan_game_mimicker import GANGameMimicker, GameStateTransition, create_transition_from_action_result

logger = logging.getLogger(__name__)

class GANEnhancedLearning:
    """Enhanced learning system that combines traditional RL with GAN-based game mimicking."""
    
    def __init__(self, continuous_learning_loop, enable_gpu: bool = True):
        self.continuous_learning_loop = continuous_learning_loop
        self.device = 'cuda' if enable_gpu and torch.cuda.is_available() else 'cpu'
        
        # Initialize GAN mimicker
        self.gan_mimicker = GANGameMimicker(
            frame_size=64,
            latent_dim=128,
            learning_rate=0.0002,
            device=self.device
        )
        
        # Training configuration
        self.gan_training_interval = 25  # Train GAN every N transitions
        self.min_transitions_for_training = 50
        self.transition_count = 0
        
        # Performance tracking
        self.gan_predictions_used = 0
        self.gan_predictions_successful = 0
        self.traditional_actions = 0
        self.gan_enhanced_actions = 0
        
        # Integration state
        self.is_integrated = False
        self.last_frame_cache = {}
        
        logger.info(f"Initialized GAN Enhanced Learning on {self.device}")
    
    def integrate_with_continuous_learning(self):
        """Integrate GAN system with the continuous learning loop."""
        if self.is_integrated:
            logger.warning("GAN integration already active")
            return
        
        # Store original methods
        self.original_execute_action = self.continuous_learning_loop._execute_action_with_api
        self.original_select_action = self.continuous_learning_loop._select_next_action
        
        # Replace with enhanced versions
        self.continuous_learning_loop._execute_action_with_api = self._enhanced_execute_action
        self.continuous_learning_loop._select_next_action = self._enhanced_select_action
        
        self.is_integrated = True
        logger.info(" GAN integration activated")
    
    def deintegrate(self):
        """Remove GAN integration and restore original methods."""
        if not self.is_integrated:
            return
        
        self.continuous_learning_loop._execute_action_with_api = self.original_execute_action
        self.continuous_learning_loop._select_next_action = self.original_select_action
        
        self.is_integrated = False
        logger.info(" GAN integration deactivated")
    
    async def _enhanced_execute_action(self, action: int, coordinates: Tuple[int, int], game_id: str) -> Dict[str, Any]:
        """Enhanced action execution that feeds transition data to GAN."""
        # Get current frame before action
        current_frame = getattr(self.continuous_learning_loop, '_last_frame', None)
        if current_frame:
            self.last_frame_cache[game_id] = current_frame
        
        # Execute original action
        result = await self.original_execute_action(action, coordinates, game_id)
        
        # Get frame after action
        new_frame = result.get('frame', None)
        if not new_frame and hasattr(self.continuous_learning_loop, '_last_frame'):
            new_frame = self.continuous_learning_loop._last_frame
        
        # Create transition for GAN training
        if current_frame and new_frame and current_frame != new_frame:
            try:
                transition = create_transition_from_action_result(
                    before_frame=current_frame,
                    after_frame=new_frame,
                    action=action,
                    coords=coordinates,
                    score_change=result.get('score_change', 0),
                    success=result.get('success', False),
                    game_id=game_id
                )
                
                self.gan_mimicker.add_transition(transition)
                self.transition_count += 1
                
                # Train GAN periodically
                if (self.transition_count % self.gan_training_interval == 0 and 
                    len(self.gan_mimicker.transition_buffer) >= self.min_transitions_for_training):
                    
                    await self._train_gan_async()
                
            except Exception as e:
                logger.error(f"Error creating GAN transition: {e}")
        
        # Track action type
        if hasattr(self, '_used_gan_prediction') and self._used_gan_prediction:
            self.gan_enhanced_actions += 1
            self._used_gan_prediction = False
        else:
            self.traditional_actions += 1
        
        return result
    
    async def _enhanced_select_action(self, game_id: str, available_actions: List[int], 
                                     current_frame: List[List[int]] = None) -> Tuple[int, Tuple[int, int]]:
        """Enhanced action selection that incorporates GAN predictions."""
        
        # Get traditional action selection
        traditional_action, traditional_coords = await self.original_select_action(
            game_id, available_actions, current_frame
        )
        
        # If we have enough training data, try GAN-based prediction
        if (len(self.gan_mimicker.transition_buffer) >= self.min_transitions_for_training and 
            current_frame is not None):
            
            try:
                # Get GAN prediction
                gan_action, gan_coords, confidence = self.gan_mimicker.predict_action(current_frame)
                
                # Use GAN prediction if:
                # 1. Action is available
                # 2. Confidence is high enough
                # 3. Traditional method seems to be struggling
                if (gan_action in available_actions and 
                    confidence > 0.7 and 
                    self._should_use_gan_prediction(game_id)):
                    
                    # Evaluate both options using GAN
                    traditional_quality = self.gan_mimicker.evaluate_action_quality(
                        current_frame, traditional_action, traditional_coords
                    )
                    gan_quality = self.gan_mimicker.evaluate_action_quality(
                        current_frame, gan_action, gan_coords
                    )
                    
                    # Use GAN prediction if it's significantly better
                    if gan_quality > traditional_quality + 0.1:
                        self._used_gan_prediction = True
                        self.gan_predictions_used += 1
                        
                        logger.info(f" GAN PREDICTION: Action {gan_action} at {gan_coords} "
                                  f"(confidence: {confidence:.3f}, quality: {gan_quality:.3f})")
                        
                        return gan_action, gan_coords
                
            except Exception as e:
                logger.error(f"Error in GAN action prediction: {e}")
        
        return traditional_action, traditional_coords
    
    def _should_use_gan_prediction(self, game_id: str) -> bool:
        """Determine if GAN prediction should be used based on current performance."""
        # Use GAN more often if traditional methods are struggling
        recent_effectiveness = getattr(self.continuous_learning_loop, 'recent_effectiveness', 0.5)
        
        # If effectiveness is low, be more willing to try GAN predictions
        if recent_effectiveness < 0.2:
            return True
        
        # Otherwise, use GAN predictions occasionally for exploration
        return np.random.random() < 0.3
    
    async def _train_gan_async(self):
        """Train GAN in a non-blocking way."""
        try:
            # Run training in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            metrics = await loop.run_in_executor(None, self.gan_mimicker.train_step, 32)
            
            if 'error' not in metrics:
                logger.info(f" GAN TRAINING: D_loss={metrics['discriminator_loss']:.3f}, "
                          f"G_loss={metrics['generator_loss']:.3f}, "
                          f"A_loss={metrics['action_predictor_loss']:.3f}, "
                          f"D_acc={metrics['discriminator_accuracy']:.3f}")
            
        except Exception as e:
            logger.error(f"Error in GAN training: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of GAN-enhanced learning."""
        gan_summary = self.gan_mimicker.get_training_summary()
        
        total_actions = self.traditional_actions + self.gan_enhanced_actions
        gan_success_rate = (self.gan_predictions_successful / max(1, self.gan_predictions_used))
        
        return {
            "integration_status": "active" if self.is_integrated else "inactive",
            "device": self.device,
            "total_transitions_collected": len(self.gan_mimicker.transition_buffer),
            "total_training_steps": gan_summary.get("total_training_steps", 0),
            "training_stability": gan_summary.get("training_stability", "unknown"),
            "action_distribution": {
                "traditional_actions": self.traditional_actions,
                "gan_enhanced_actions": self.gan_enhanced_actions,
                "gan_usage_rate": self.gan_enhanced_actions / max(1, total_actions)
            },
            "gan_prediction_stats": {
                "predictions_used": self.gan_predictions_used,
                "predictions_successful": self.gan_predictions_successful,
                "success_rate": gan_success_rate
            },
            "recent_performance": {
                "discriminator_loss": gan_summary.get("recent_discriminator_loss", 0),
                "generator_loss": gan_summary.get("recent_generator_loss", 0),
                "action_predictor_loss": gan_summary.get("recent_action_predictor_loss", 0),
                "discriminator_accuracy": gan_summary.get("recent_discriminator_accuracy", 0)
            }
        }
    
    def save_gan_model(self, filepath: str):
        """Save the GAN model and integration state."""
        # Save GAN model
        self.gan_mimicker.save_model(filepath)
        
        # Save integration state
        state_file = filepath.replace('.pth', '_state.json')
        import json
        with open(state_file, 'w') as f:
            json.dump({
                "transition_count": self.transition_count,
                "gan_predictions_used": self.gan_predictions_used,
                "gan_predictions_successful": self.gan_predictions_successful,
                "traditional_actions": self.traditional_actions,
                "gan_enhanced_actions": self.gan_enhanced_actions
            }, f)
        
        logger.info(f"Saved GAN integration state to {state_file}")
    
    def load_gan_model(self, filepath: str):
        """Load the GAN model and integration state."""
        # Load GAN model
        self.gan_mimicker.load_model(filepath)
        
        # Load integration state
        state_file = filepath.replace('.pth', '_state.json')
        try:
            import json
            with open(state_file, 'r') as f:
                state = json.load(f)
                
            self.transition_count = state.get("transition_count", 0)
            self.gan_predictions_used = state.get("gan_predictions_used", 0)
            self.gan_predictions_successful = state.get("gan_predictions_successful", 0)
            self.traditional_actions = state.get("traditional_actions", 0)
            self.gan_enhanced_actions = state.get("gan_enhanced_actions", 0)
            
            logger.info(f"Loaded GAN integration state from {state_file}")
            
        except FileNotFoundError:
            logger.warning(f"No integration state file found at {state_file}")
        except Exception as e:
            logger.error(f"Error loading integration state: {e}")

# Factory function for easy integration
def create_gan_enhanced_learning(continuous_learning_loop, enable_gpu: bool = True) -> GANEnhancedLearning:
    """Create and integrate GAN-enhanced learning system."""
    gan_system = GANEnhancedLearning(continuous_learning_loop, enable_gpu)
    gan_system.integrate_with_continuous_learning()
    return gan_system

# Utility functions for monitoring
async def monitor_gan_performance(gan_system: GANEnhancedLearning, interval: int = 300):
    """Monitor GAN performance and log periodic updates."""
    while gan_system.is_integrated:
        try:
            summary = gan_system.get_performance_summary()
            
            logger.info(f" GAN PERFORMANCE UPDATE:")
            logger.info(f"   Transitions: {summary['total_transitions_collected']}")
            logger.info(f"   Training Steps: {summary['total_training_steps']}")
            logger.info(f"   GAN Usage Rate: {summary['action_distribution']['gan_usage_rate']:.1%}")
            logger.info(f"   Prediction Success: {summary['gan_prediction_stats']['success_rate']:.1%}")
            logger.info(f"   Training Stability: {summary['training_stability']}")
            
            await asyncio.sleep(interval)
            
        except Exception as e:
            logger.error(f"Error in GAN performance monitoring: {e}")
            await asyncio.sleep(interval)

if __name__ == "__main__":
    # Example integration
    print("GAN Integration Module - Ready for integration with continuous learning loop")
    print("Use: gan_system = create_gan_enhanced_learning(your_continuous_learning_loop)")
