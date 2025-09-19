"""
GAN-Based Game Mimicking System for ARC-AGI-3
==============================================

This module implements a Generative Adversarial Network (GAN) to learn and mimic
the internal dynamics of ARC games, enabling better prediction of game state
transitions and reverse engineering of game logic.

Architecture:
- Generator: Learns to generate realistic game state transitions
- Discriminator: Learns to distinguish real vs fake transitions  
- Game State Encoder: Converts game frames to latent representations
- Action Predictor: Suggests optimal actions based on learned patterns
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional
import json
import asyncio
import time
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class GameStateTransition:
    """Represents a single game state transition."""
    before_state: np.ndarray  # Game frame before action
    action_taken: int         # Action that was taken (1-7)
    action_coords: Tuple[int, int]  # Coordinates where action was applied
    after_state: np.ndarray   # Game frame after action
    score_change: float       # Change in score
    success: bool            # Whether action was successful
    game_id: str            # Game identifier
    timestamp: float        # When transition occurred

class GameStateEncoder(nn.Module):
    """Encodes game frames into latent space representations."""
    
    def __init__(self, frame_size: int = 64, latent_dim: int = 128):
        super().__init__()
        self.frame_size = frame_size
        self.latent_dim = latent_dim
        
        # Convolutional encoder for spatial features
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 32x32 -> 16x16
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16x16 -> 8x8
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 8x8 -> 4x4
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        """Encode game frame to latent representation."""
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        return self.conv_encoder(x)

class TransitionGenerator(nn.Module):
    """Generates realistic game state transitions."""
    
    def __init__(self, latent_dim: int = 128, action_dim: int = 9):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim  # 7 actions + 2 coordinates
        
        # Generator network
        self.generator = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256 * 4 * 4),
            nn.ReLU()
        )
        
        # Deconvolutional decoder
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    # 32x32 -> 64x64
            nn.Tanh()
        )
        
    def forward(self, state_encoding, action_vector):
        """Generate next state given current state and action."""
        # Combine state and action
        combined_input = torch.cat([state_encoding, action_vector], dim=1)
        
        # Generate through fully connected layers
        x = self.generator(combined_input)
        x = x.view(-1, 256, 4, 4)
        
        # Decode to frame
        generated_frame = self.conv_decoder(x)
        return generated_frame

class TransitionDiscriminator(nn.Module):
    """Discriminates between real and generated transitions."""
    
    def __init__(self, frame_size: int = 64, action_dim: int = 9):
        super().__init__()
        self.frame_size = frame_size
        self.action_dim = action_dim
        
        # Frame encoder
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),  # 2 frames: before & after
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        
        # Combined discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(512 * 4 * 4 + action_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, before_frame, after_frame, action_vector):
        """Classify transition as real (1) or fake (0)."""
        # Combine before and after frames
        combined_frames = torch.cat([before_frame, after_frame], dim=1)
        
        # Encode frames
        frame_features = self.frame_encoder(combined_frames)
        
        # Combine with action
        combined_input = torch.cat([frame_features, action_vector], dim=1)
        
        # Classify
        return self.discriminator(combined_input)

class ActionPredictor(nn.Module):
    """Predicts optimal actions based on learned game dynamics."""
    
    def __init__(self, latent_dim: int = 128, num_actions: int = 7):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        
        # Action prediction network
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_actions),
            nn.Softmax(dim=1)
        )
        
        # Coordinate prediction network
        self.coord_predictor = nn.Sequential(
            nn.Linear(latent_dim + num_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # x, y coordinates
            nn.Sigmoid()  # Normalize to [0, 1]
        )
        
    def forward(self, state_encoding):
        """Predict optimal action and coordinates."""
        # Predict action probabilities
        action_probs = self.predictor(state_encoding)
        
        # Predict coordinates conditioned on action
        combined_input = torch.cat([state_encoding, action_probs], dim=1)
        coord_probs = self.coord_predictor(combined_input)
        
        return action_probs, coord_probs

class GANGameMimicker:
    """Main GAN-based game mimicking system."""
    
    def __init__(self, 
                 frame_size: int = 64,
                 latent_dim: int = 128,
                 learning_rate: float = 0.0002,
                 device: str = 'cpu'):
        
        self.frame_size = frame_size
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        
        # Initialize networks
        self.encoder = GameStateEncoder(frame_size, latent_dim).to(self.device)
        self.generator = TransitionGenerator(latent_dim).to(self.device)
        self.discriminator = TransitionDiscriminator(frame_size).to(self.device)
        self.action_predictor = ActionPredictor(latent_dim).to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.a_optimizer = optim.Adam(self.action_predictor.parameters(), lr=learning_rate)
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.reconstruction_loss = nn.MSELoss()
        self.action_loss = nn.CrossEntropyLoss()
        
        # Training history
        self.training_history = []
        self.transition_buffer = []
        
        logger.info(f"Initialized GAN Game Mimicker with {latent_dim}D latent space on {device}")
        
    def preprocess_frame(self, frame: List[List[int]]) -> torch.Tensor:
        """Convert game frame to tensor format."""
        if isinstance(frame, list):
            frame = np.array(frame, dtype=np.float32)
        
        # Normalize to [-1, 1] for GAN training
        frame = (frame - 7.5) / 7.5  # Assuming colors 0-15
        
        # Add batch and channel dimensions: [batch, channel, height, width]
        return torch.FloatTensor(frame).unsqueeze(0).unsqueeze(0).to(self.device)
    
    def encode_action(self, action: int, coords: Tuple[int, int]) -> torch.Tensor:
        """Encode action and coordinates as vector."""
        action_one_hot = torch.zeros(7)
        if 1 <= action <= 7:
            action_one_hot[action - 1] = 1.0
            
        # Normalize coordinates to [0, 1]
        norm_coords = torch.FloatTensor([coords[0] / self.frame_size, coords[1] / self.frame_size])
        
        action_vector = torch.cat([action_one_hot, norm_coords]).unsqueeze(0).to(self.device)
        return action_vector
    
    def add_transition(self, transition: GameStateTransition):
        """Add a transition to the training buffer."""
        self.transition_buffer.append(transition)
        
        # Keep buffer size manageable
        if len(self.transition_buffer) > 10000:
            self.transition_buffer = self.transition_buffer[-8000:]
            
        logger.debug(f"Added transition to buffer. Buffer size: {len(self.transition_buffer)}")
    
    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """Perform one training step on the GAN."""
        if len(self.transition_buffer) < batch_size:
            return {"error": "Not enough transitions for training"}
        
        # Sample batch
        batch_indices = np.random.choice(len(self.transition_buffer), batch_size, replace=False)
        batch = [self.transition_buffer[i] for i in batch_indices]
        
        # Prepare batch data
        before_states = []
        after_states = []
        action_vectors = []
        success_labels = []
        
        for transition in batch:
            before_states.append(self.preprocess_frame(transition.before_state))
            after_states.append(self.preprocess_frame(transition.after_state))
            action_vectors.append(self.encode_action(transition.action_taken, transition.action_coords))
            success_labels.append(float(transition.success))
        
        before_batch = torch.cat(before_states, dim=0)
        after_batch = torch.cat(after_states, dim=0)
        action_batch = torch.cat(action_vectors, dim=0)
        success_batch = torch.FloatTensor(success_labels).to(self.device)
        
        # Train discriminator
        self.d_optimizer.zero_grad()
        
        # Real transitions
        real_labels = torch.ones(batch_size, 1).to(self.device)
        real_output = self.discriminator(before_batch, after_batch, action_batch)
        d_loss_real = self.adversarial_loss(real_output, real_labels)
        
        # Fake transitions
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        before_encoded = self.encoder(before_batch)
        fake_after = self.generator(before_encoded, action_batch)
        fake_output = self.discriminator(before_batch, fake_after.detach(), action_batch)
        d_loss_fake = self.adversarial_loss(fake_output, fake_labels)
        
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train generator
        self.g_optimizer.zero_grad()
        
        fake_output = self.discriminator(before_batch, fake_after, action_batch)
        g_loss_adv = self.adversarial_loss(fake_output, real_labels)
        g_loss_rec = self.reconstruction_loss(fake_after, after_batch)
        g_loss = g_loss_adv + 10.0 * g_loss_rec  # Weighted reconstruction loss
        
        g_loss.backward()
        self.g_optimizer.step()
        
        # Train action predictor
        self.a_optimizer.zero_grad()
        
        action_probs, coord_probs = self.action_predictor(before_encoded.detach())
        action_targets = torch.LongTensor([t.action_taken - 1 for t in batch]).to(self.device)
        coord_targets = torch.FloatTensor([[t.action_coords[0] / self.frame_size, 
                                           t.action_coords[1] / self.frame_size] for t in batch]).to(self.device)
        
        a_loss_action = self.action_loss(action_probs, action_targets)
        a_loss_coord = self.reconstruction_loss(coord_probs, coord_targets)
        a_loss = a_loss_action + a_loss_coord
        
        a_loss.backward()
        self.a_optimizer.step()
        
        # Record training metrics
        metrics = {
            'discriminator_loss': d_loss.item(),
            'generator_loss': g_loss.item(),
            'action_predictor_loss': a_loss.item(),
            'discriminator_accuracy': ((real_output > 0.5).float().mean() + (fake_output < 0.5).float().mean()).item() / 2,
            'batch_size': batch_size
        }
        
        self.training_history.append(metrics)
        return metrics
    
    def predict_action(self, game_frame: List[List[int]]) -> Tuple[int, Tuple[int, int], float]:
        """Predict optimal action for given game state."""
        with torch.no_grad():
            # Encode current state
            frame_tensor = self.preprocess_frame(game_frame)
            state_encoding = self.encoder(frame_tensor)
            
            # Predict action and coordinates
            action_probs, coord_probs = self.action_predictor(state_encoding)
            
            # Get most likely action
            action_idx = torch.argmax(action_probs, dim=1).item()
            action = action_idx + 1  # Convert to 1-7 range
            
            # Get predicted coordinates
            coords = coord_probs[0].cpu().numpy()
            coord_x = int(coords[0] * self.frame_size)
            coord_y = int(coords[1] * self.frame_size)
            
            # Get confidence
            confidence = action_probs[0, action_idx].item()
            
            return action, (coord_x, coord_y), confidence
    
    def simulate_transition(self, game_frame: List[List[int]], action: int, coords: Tuple[int, int]) -> np.ndarray:
        """Simulate what would happen if we take the given action."""
        with torch.no_grad():
            # Encode current state
            frame_tensor = self.preprocess_frame(game_frame)
            state_encoding = self.encoder(frame_tensor)
            
            # Encode action
            action_vector = self.encode_action(action, coords)
            
            # Generate next state
            next_state_tensor = self.generator(state_encoding, action_vector)
            
            # Convert back to frame format
            next_state = next_state_tensor[0, 0].cpu().numpy()
            next_state = (next_state * 7.5) + 7.5  # Denormalize
            next_state = np.clip(next_state, 0, 15).astype(int)
            
            return next_state
    
    def evaluate_action_quality(self, game_frame: List[List[int]], action: int, coords: Tuple[int, int]) -> float:
        """Evaluate how good an action would be (0.0 = bad, 1.0 = good)."""
        with torch.no_grad():
            # Simulate the transition
            simulated_next = self.simulate_transition(game_frame, action, coords)
            
            # Encode both states
            current_tensor = self.preprocess_frame(game_frame)
            next_tensor = self.preprocess_frame(simulated_next.tolist())
            action_vector = self.encode_action(action, coords)
            
            # Use discriminator to evaluate realism
            realism_score = self.discriminator(current_tensor, next_tensor, action_vector).item()
            
            return realism_score
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        if not self.training_history:
            return {"status": "No training performed yet"}
        
        recent_history = self.training_history[-100:]  # Last 100 steps
        
        return {
            "total_training_steps": len(self.training_history),
            "transitions_collected": len(self.transition_buffer),
            "recent_discriminator_loss": np.mean([h['discriminator_loss'] for h in recent_history]),
            "recent_generator_loss": np.mean([h['generator_loss'] for h in recent_history]),
            "recent_action_predictor_loss": np.mean([h['action_predictor_loss'] for h in recent_history]),
            "recent_discriminator_accuracy": np.mean([h['discriminator_accuracy'] for h in recent_history]),
            "training_stability": "stable" if len(recent_history) > 10 and 
                                 np.std([h['discriminator_loss'] for h in recent_history]) < 0.5 else "unstable"
        }
    
    def save_model(self, filepath: str):
        """Save the trained models."""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'action_predictor': self.action_predictor.state_dict(),
            'training_history': self.training_history,
            'config': {
                'frame_size': self.frame_size,
                'latent_dim': self.latent_dim
            }
        }, filepath)
        logger.info(f"Saved GAN model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained models."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.action_predictor.load_state_dict(checkpoint['action_predictor'])
        self.training_history = checkpoint.get('training_history', [])
        
        logger.info(f"Loaded GAN model from {filepath}")

# Integration helper functions
def create_transition_from_action_result(
    before_frame: List[List[int]],
    after_frame: List[List[int]], 
    action: int,
    coords: Tuple[int, int],
    score_change: float,
    success: bool,
    game_id: str
) -> GameStateTransition:
    """Create a GameStateTransition from action execution results."""
    return GameStateTransition(
        before_state=np.array(before_frame),
        action_taken=action,
        action_coords=coords,
        after_state=np.array(after_frame),
        score_change=score_change,
        success=success,
        game_id=game_id,
        timestamp=time.time()
    )

def integrate_gan_with_continuous_learning(continuous_learning_loop, gan_mimicker: GANGameMimicker):
    """Integrate GAN mimicker with the continuous learning loop."""
    
    # Store original action execution method
    original_execute_action = continuous_learning_loop._execute_action_with_api
    
    async def enhanced_execute_action(action, coordinates, game_id):
        """Enhanced action execution that feeds data to GAN."""
        # Get current frame before action
        current_frame = getattr(continuous_learning_loop, '_last_frame', None)
        
        # Execute original action
        result = await original_execute_action(action, coordinates, game_id)
        
        # Get frame after action
        new_frame = result.get('frame', current_frame)
        
        # Create transition for GAN training
        if current_frame and new_frame:
            transition = create_transition_from_action_result(
                before_frame=current_frame,
                after_frame=new_frame,
                action=action,
                coords=coordinates,
                score_change=result.get('score_change', 0),
                success=result.get('success', False),
                game_id=game_id
            )
            gan_mimicker.add_transition(transition)
            
            # Train GAN periodically
            if len(gan_mimicker.transition_buffer) % 50 == 0:
                metrics = gan_mimicker.train_step()
                if 'error' not in metrics:
                    logger.info(f"GAN training: D_loss={metrics['discriminator_loss']:.3f}, "
                              f"G_loss={metrics['generator_loss']:.3f}, "
                              f"A_loss={metrics['action_predictor_loss']:.3f}")
        
        return result
    
    # Replace the method
    continuous_learning_loop._execute_action_with_api = enhanced_execute_action
    
    logger.info("Integrated GAN mimicker with continuous learning loop")

if __name__ == "__main__":
    # Example usage
    gan = GANGameMimicker(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simulate some training data
    for i in range(100):
        fake_before = np.random.randint(0, 16, (64, 64))
        fake_after = np.random.randint(0, 16, (64, 64))
        fake_transition = GameStateTransition(
            before_state=fake_before,
            action_taken=np.random.randint(1, 8),
            action_coords=(np.random.randint(0, 64), np.random.randint(0, 64)),
            after_state=fake_after,
            score_change=np.random.randn(),
            success=np.random.random() > 0.5,
            game_id=f"test_{i}",
            timestamp=time.time()
        )
        gan.add_transition(fake_transition)
    
    # Train for a few steps
    for _ in range(10):
        metrics = gan.train_step()
        print(f"Training metrics: {metrics}")
    
    # Test prediction
    test_frame = [[i % 16 for i in range(64)] for _ in range(64)]
    action, coords, confidence = gan.predict_action(test_frame)
    print(f"Predicted action: {action} at {coords} with confidence {confidence:.3f}")
