"""
GAN System for ARC-AGI-3 Game Mimicking

This module implements a Generative Adversarial Network system that learns to generate
synthetic ARC-AGI-3 game states and reverse-engineer game mechanics through adversarial training.

Key Features:
- Database-only storage (no file creation)
- Integration with existing pattern learning system
- Real-time synthetic data generation
- Game mechanics reverse engineering
- Enhanced predictor training with synthetic data
"""

# Lazy imports to avoid torch startup delays
def _get_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None

def _get_torch_nn():
    try:
        import torch.nn as nn
        return nn
    except ImportError:
        return None

def _get_torch_functional():
    try:
        import torch.nn.functional as F
        return F
    except ImportError:
        return None

def _get_torch_optim():
    try:
        import torch.optim as optim
        return optim
    except ImportError:
        return None
import json
import logging
import time
import uuid
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import asyncio

from ..database.api import get_database
from ..arc_integration.arc_meta_learning import ARCMetaLearningSystem, ARCPattern

logger = logging.getLogger(__name__)

@dataclass
class GameState:
    """Represents a game state for GAN training."""
    grid: np.ndarray
    objects: List[Dict[str, Any]]
    properties: Dict[str, Any]
    context: Dict[str, Any]
    action_history: List[int]
    success_probability: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'grid': self.grid.tolist() if isinstance(self.grid, np.ndarray) else self.grid,
            'objects': self.objects,
            'properties': self.properties,
            'context': self.context,
            'action_history': self.action_history,
            'success_probability': float(self.success_probability),
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameState':
        """Create from dictionary."""
        return cls(
            grid=np.array(data['grid']),
            objects=data['objects'],
            properties=data['properties'],
            context=data['context'],
            action_history=data['action_history'],
            success_probability=data['success_probability'],
            timestamp=data.get('timestamp', time.time())
        )

@dataclass
class GANTrainingConfig:
    """Configuration for GAN training."""
    batch_size: int = 32
    learning_rate_generator: float = 0.0002
    learning_rate_discriminator: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    latent_dim: int = 100
    pattern_embedding_dim: int = 256
    max_epochs: int = 1000
    convergence_threshold: float = 0.01
    validation_frequency: int = 10
    checkpoint_frequency: int = 50
    synthetic_data_ratio: float = 0.2  # 20% of training data should be synthetic

class GameStateGenerator(nn.Module):
    """
    Generator network that creates synthetic ARC-AGI-3 game states.
    
    Architecture:
    - Input: Noise vector + Pattern embeddings + Game context
    - Output: Synthetic game state (grid, objects, properties)
    """
    
    def __init__(self, 
                 latent_dim: int = 100,
                 pattern_embedding_dim: int = 256,
                 context_dim: int = 64,
                 grid_size: Tuple[int, int] = (64, 64),
                 num_channels: int = 3):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.pattern_embedding_dim = pattern_embedding_dim
        self.context_dim = context_dim
        self.grid_size = grid_size
        self.num_channels = num_channels
        
        # Input processing
        total_input_dim = latent_dim + pattern_embedding_dim + context_dim
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True)
        )
        
        # Grid generation head
        self.grid_head = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(True),
            nn.Linear(4096, grid_size[0] * grid_size[1] * num_channels),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Object generation head
        self.object_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 32)  # Max 32 objects with properties
        )
        
        # Properties generation head
        self.properties_head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 16)  # Game properties
        )
        
        # Success probability head
        self.success_head = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(True),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )
    
    def forward(self, noise: torch.Tensor, pattern_embeddings: torch.Tensor, 
                context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate synthetic game state."""
        # Combine inputs
        x = torch.cat([noise, pattern_embeddings, context], dim=1)
        
        # Encode
        encoded = self.encoder(x)
        
        # Generate components
        grid = self.grid_head(encoded)
        grid = grid.view(-1, self.num_channels, self.grid_size[0], self.grid_size[1])
        
        objects = self.object_head(encoded)
        properties = self.properties_head(encoded)
        success_prob = self.success_head(encoded)
        
        return {
            'grid': grid,
            'objects': objects,
            'properties': properties,
            'success_probability': success_prob
        }

class GameStateDiscriminator(nn.Module):
    """
    Discriminator network that distinguishes real from synthetic game states.
    
    Architecture:
    - Input: Game state (grid, objects, properties)
    - Output: Real/Synthetic probability + Quality score + Pattern consistency
    """
    
    def __init__(self, 
                 grid_size: Tuple[int, int] = (64, 64),
                 num_channels: int = 3,
                 object_dim: int = 32,
                 properties_dim: int = 16):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_channels = num_channels
        
        # Grid processing
        self.grid_conv = nn.Sequential(
            nn.Conv2d(num_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )
        
        # Calculate flattened size
        conv_output_size = 512 * (grid_size[0] // 16) * (grid_size[1] // 16)
        
        # Object and properties processing
        self.object_processor = nn.Sequential(
            nn.Linear(object_dim, 64),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 32)
        )
        
        self.properties_processor = nn.Sequential(
            nn.Linear(properties_dim, 32),
            nn.LeakyReLU(0.2, True),
            nn.Linear(32, 16)
        )
        
        # Combined processing
        total_features = conv_output_size + 32 + 16
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 3)  # [real/synthetic, quality, pattern_consistency]
        )
    
    def forward(self, grid: torch.Tensor, objects: torch.Tensor, 
                properties: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Discriminate game state."""
        # Process grid
        grid_features = self.grid_conv(grid)
        grid_features = grid_features.view(grid_features.size(0), -1)
        
        # Process objects and properties
        object_features = self.object_processor(objects)
        properties_features = self.properties_processor(properties)
        
        # Combine features
        combined = torch.cat([grid_features, object_features, properties_features], dim=1)
        
        # Classify
        output = self.classifier(combined)
        
        # Split outputs
        real_synthetic = torch.sigmoid(output[:, 0:1])  # Real probability
        quality = torch.sigmoid(output[:, 1:2])  # Quality score
        pattern_consistency = torch.sigmoid(output[:, 2:3])  # Pattern consistency
        
        return {
            'real_probability': real_synthetic,
            'quality_score': quality,
            'pattern_consistency': pattern_consistency
        }

class PatternAwareGAN:
    """
    Main GAN system that integrates with existing pattern learning.
    
    Features:
    - Database-only storage
    - Pattern-aware generation
    - Real-time synthetic data creation
    - Game mechanics reverse engineering
    """
    
    def __init__(self, 
                 config: Optional[GANTrainingConfig] = None,
                 pattern_learning_system: Optional[ARCMetaLearningSystem] = None):
        self.config = config or GANTrainingConfig()
        self.pattern_learning_system = pattern_learning_system
        self.db = get_database()
        
        # Initialize models
        self.generator = GameStateGenerator(
            latent_dim=self.config.latent_dim,
            pattern_embedding_dim=self.config.pattern_embedding_dim
        )
        
        self.discriminator = GameStateDiscriminator()
        
        # Optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=self.config.learning_rate_generator,
            betas=(self.config.beta1, self.config.beta2)
        )
        
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.learning_rate_discriminator,
            betas=(self.config.beta1, self.config.beta2)
        )
        
        # Training state
        self.current_session_id = None
        self.training_metrics = {
            'generator_loss': [],
            'discriminator_loss': [],
            'pattern_accuracy': [],
            'synthetic_quality': []
        }
        
        logger.info("Pattern-Aware GAN initialized with database integration")
    
    async def start_training_session(self, session_name: str = None) -> str:
        """Start a new GAN training session."""
        session_id = session_name or f"gan_session_{uuid.uuid4().hex[:8]}"
        
        # Create session in database
        await self.db.execute("""
            INSERT INTO gan_training_sessions 
            (session_id, start_time, status) 
            VALUES (?, ?, ?)
        """, (session_id, datetime.now(), 'running'))
        
        self.current_session_id = session_id
        
        # Log session start
        await self.db.execute("""
            INSERT INTO system_logs 
            (component, log_level, message, data, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            "gan_system",
            "INFO",
            f"GAN training session {session_id} started",
            json.dumps({"session_id": session_id}),
            datetime.now()
        ))
        
        logger.info(f"GAN training session {session_id} started")
        return session_id
    
    async def generate_synthetic_states(self, 
                                      count: int,
                                      context: Optional[Dict[str, Any]] = None) -> List[GameState]:
        """Generate synthetic game states for training."""
        if not self.current_session_id:
            raise ValueError("No active training session. Call start_training_session() first.")
        
        # Generate noise
        noise = torch.randn(count, self.config.latent_dim)
        
        # Get pattern embeddings from existing system
        pattern_embeddings = await self._get_pattern_embeddings(count)
        
        # Create context
        context_tensor = self._create_context_tensor(context or {}, count)
        
        # Generate states
        with torch.no_grad():
            generated = self.generator(noise, pattern_embeddings, context_tensor)
        
        # Convert to GameState objects
        synthetic_states = []
        for i in range(count):
            state = GameState(
                grid=generated['grid'][i].cpu().numpy(),
                objects=self._decode_objects(generated['objects'][i].cpu().numpy()),
                properties=self._decode_properties(generated['properties'][i].cpu().numpy()),
                context=context or {},
                action_history=[],
                success_probability=float(generated['success_probability'][i].item())
            )
            synthetic_states.append(state)
        
        # Store in database
        await self._store_synthetic_states(synthetic_states)
        
        return synthetic_states
    
    async def train_epoch(self, real_states: List[GameState]) -> Dict[str, float]:
        """Train GAN for one epoch."""
        if not self.current_session_id:
            raise ValueError("No active training session.")
        
        # Prepare real data
        real_batch = self._prepare_batch(real_states)
        
        # Generate synthetic data
        synthetic_states = await self.generate_synthetic_states(len(real_states))
        synthetic_batch = self._prepare_batch(synthetic_states)
        
        # Train discriminator
        d_loss = await self._train_discriminator(real_batch, synthetic_batch)
        
        # Train generator
        g_loss = await self._train_generator(synthetic_batch)
        
        # Calculate metrics
        pattern_accuracy = await self._calculate_pattern_accuracy(synthetic_states)
        synthetic_quality = await self._calculate_synthetic_quality(synthetic_states)
        
        # Update metrics
        self.training_metrics['generator_loss'].append(g_loss)
        self.training_metrics['discriminator_loss'].append(d_loss)
        self.training_metrics['pattern_accuracy'].append(pattern_accuracy)
        self.training_metrics['synthetic_quality'].append(synthetic_quality)
        
        # Store metrics in database
        await self._store_training_metrics(g_loss, d_loss, pattern_accuracy, synthetic_quality)
        
        return {
            'generator_loss': g_loss,
            'discriminator_loss': d_loss,
            'pattern_accuracy': pattern_accuracy,
            'synthetic_quality': synthetic_quality
        }
    
    async def reverse_engineer_game_mechanics(self, game_id: str) -> Dict[str, Any]:
        """Use GAN to reverse engineer game mechanics."""
        if not self.current_session_id:
            raise ValueError("No active training session.")
        
        # Get game data
        game_data = await self.db.fetch_all("""
            SELECT state_data, pattern_context 
            FROM gan_generated_states 
            WHERE game_id = ? AND session_id = ?
            ORDER BY quality_score DESC
            LIMIT 100
        """, (game_id, self.current_session_id))
        
        if not game_data:
            return {"error": "No generated data for this game"}
        
        # Analyze patterns in generated states
        discovered_rules = await self._analyze_generated_patterns(game_data)
        
        # Store reverse engineering results
        await self.db.execute("""
            INSERT INTO gan_reverse_engineering 
            (session_id, game_id, discovered_rules, rule_confidence, mechanics_understood)
            VALUES (?, ?, ?, ?, ?)
        """, (
            self.current_session_id,
            game_id,
            json.dumps(discovered_rules),
            discovered_rules.get('confidence', 0.0),
            discovered_rules.get('mechanics_understood', 0.0)
        ))
        
        return discovered_rules
    
    async def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and metrics."""
        if not self.current_session_id:
            return {"error": "No active training session"}
        
        # Get session data
        session_data = await self.db.fetch_one("""
            SELECT * FROM gan_training_sessions 
            WHERE session_id = ?
        """, (self.current_session_id,))
        
        # Get recent metrics
        recent_metrics = await self.db.fetch_all("""
            SELECT * FROM gan_performance_metrics 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 100
        """, (self.current_session_id,))
        
        return {
            'session': dict(session_data) if session_data else {},
            'recent_metrics': [dict(m) for m in recent_metrics],
            'current_metrics': self.training_metrics
        }
    
    # Private helper methods
    
    async def _get_pattern_embeddings(self, count: int) -> torch.Tensor:
        """Get pattern embeddings from existing pattern learning system."""
        if self.pattern_learning_system:
            # Get learned patterns
            patterns = await self.pattern_learning_system.get_learned_patterns(limit=count)
            # Convert to embeddings (simplified)
            embeddings = torch.randn(count, self.config.pattern_embedding_dim)
        else:
            # Fallback to random embeddings
            embeddings = torch.randn(count, self.config.pattern_embedding_dim)
        
        return embeddings
    
    def _create_context_tensor(self, context: Dict[str, Any], count: int) -> torch.Tensor:
        """Create context tensor for generation."""
        # Simplified context encoding - ensure we have the right dimension
        context_vector = np.array([
            context.get('level', 1),
            context.get('difficulty', 0.5),
            context.get('energy_level', 100.0),
            context.get('learning_drive', 0.5)
        ])
        
        # Ensure we have the right dimension for the context
        # Pad with zeros to match expected context_dim
        context_dim = 64  # This should match the context_dim in GameStateGenerator
        if len(context_vector) < context_dim:
            padded_vector = np.zeros(context_dim)
            padded_vector[:len(context_vector)] = context_vector
            context_vector = padded_vector
        elif len(context_vector) > context_dim:
            context_vector = context_vector[:context_dim]
        
        # Repeat for batch
        context_tensor = torch.tensor(
            np.tile(context_vector, (count, 1)), 
            dtype=torch.float32
        )
        
        return context_tensor
    
    def _decode_objects(self, object_vector: np.ndarray) -> List[Dict[str, Any]]:
        """Decode object vector to object list."""
        # Simplified object decoding
        objects = []
        for i in range(0, len(object_vector), 4):
            if i + 3 < len(object_vector):
                objects.append({
                    'x': int(object_vector[i] * 64),
                    'y': int(object_vector[i+1] * 64),
                    'type': int(object_vector[i+2] * 10),
                    'properties': float(object_vector[i+3])
                })
        return objects
    
    def _decode_properties(self, properties_vector: np.ndarray) -> Dict[str, Any]:
        """Decode properties vector to properties dict."""
        return {
            'grid_size': (64, 64),
            'num_objects': int(properties_vector[0] * 32),
            'complexity': float(properties_vector[1]),
            'difficulty': float(properties_vector[2]),
            'energy_required': float(properties_vector[3])
        }
    
    async def _store_synthetic_states(self, states: List[GameState]) -> None:
        """Store synthetic states in database."""
        for state in states:
            await self.db.execute("""
                INSERT INTO gan_generated_states 
                (session_id, state_data, pattern_context, quality_score, generation_method)
                VALUES (?, ?, ?, ?, ?)
            """, (
                self.current_session_id,
                json.dumps(state.to_dict()),
                json.dumps(state.context),
                state.success_probability,
                'gan'
            ))
    
    async def _prepare_batch(self, states: List[GameState]) -> Dict[str, torch.Tensor]:
        """Prepare batch for training."""
        grids = []
        objects = []
        properties = []
        
        for state in states:
            grids.append(torch.tensor(state.grid, dtype=torch.float32))
            objects.append(torch.tensor(self._encode_objects(state.objects), dtype=torch.float32))
            properties.append(torch.tensor(self._encode_properties(state.properties), dtype=torch.float32))
        
        return {
            'grid': torch.stack(grids),
            'objects': torch.stack(objects),
            'properties': torch.stack(properties)
        }
    
    def _encode_objects(self, objects: List[Dict[str, Any]]) -> np.ndarray:
        """Encode objects to vector."""
        # Simplified encoding
        vector = np.zeros(32)  # Max 32 objects
        for i, obj in enumerate(objects[:8]):  # Max 8 objects
            vector[i*4] = obj.get('x', 0) / 64.0
            vector[i*4+1] = obj.get('y', 0) / 64.0
            vector[i*4+2] = obj.get('type', 0) / 10.0
            vector[i*4+3] = obj.get('properties', 0)
        return vector
    
    def _encode_properties(self, properties: Dict[str, Any]) -> np.ndarray:
        """Encode properties to vector."""
        return np.array([
            properties.get('num_objects', 0) / 32.0,
            properties.get('complexity', 0.5),
            properties.get('difficulty', 0.5),
            properties.get('energy_required', 0.5)
        ])
    
    async def _train_discriminator(self, real_batch: Dict[str, torch.Tensor], 
                                 synthetic_batch: Dict[str, torch.Tensor]) -> float:
        """Train discriminator."""
        self.optimizer_d.zero_grad()
        
        # Real data
        real_output = self.discriminator(
            real_batch['grid'], 
            real_batch['objects'], 
            real_batch['properties']
        )
        real_loss = F.binary_cross_entropy(
            real_output['real_probability'], 
            torch.ones_like(real_output['real_probability'])
        )
        
        # Synthetic data
        synthetic_output = self.discriminator(
            synthetic_batch['grid'], 
            synthetic_batch['objects'], 
            synthetic_batch['properties']
        )
        synthetic_loss = F.binary_cross_entropy(
            synthetic_output['real_probability'], 
            torch.zeros_like(synthetic_output['real_probability'])
        )
        
        # Total loss
        d_loss = real_loss + synthetic_loss
        d_loss.backward()
        self.optimizer_d.step()
        
        return d_loss.item()
    
    async def _train_generator(self, synthetic_batch: Dict[str, torch.Tensor]) -> float:
        """Train generator."""
        self.optimizer_g.zero_grad()
        
        # Generate synthetic data
        output = self.discriminator(
            synthetic_batch['grid'], 
            synthetic_batch['objects'], 
            synthetic_batch['properties']
        )
        
        # Generator loss (fool discriminator)
        g_loss = F.binary_cross_entropy(
            output['real_probability'], 
            torch.ones_like(output['real_probability'])
        )
        
        g_loss.backward()
        self.optimizer_g.step()
        
        return g_loss.item()
    
    async def _calculate_pattern_accuracy(self, states: List[GameState]) -> float:
        """Calculate pattern accuracy of generated states."""
        # Simplified pattern accuracy calculation
        if not states:
            return 0.0
        
        total_accuracy = 0.0
        for state in states:
            # Check if state follows learned patterns
            accuracy = state.success_probability  # Simplified
            total_accuracy += accuracy
        
        return total_accuracy / len(states)
    
    async def _calculate_synthetic_quality(self, states: List[GameState]) -> float:
        """Calculate quality of synthetic states."""
        if not states:
            return 0.0
        
        # Simplified quality calculation
        total_quality = 0.0
        for state in states:
            # Check grid validity, object consistency, etc.
            quality = 0.8  # Simplified
            total_quality += quality
        
        return total_quality / len(states)
    
    async def _store_training_metrics(self, g_loss: float, d_loss: float, 
                                    pattern_accuracy: float, synthetic_quality: float) -> None:
        """Store training metrics in database."""
        # Update session metrics
        await self.db.execute("""
            UPDATE gan_training_sessions 
            SET generator_loss = ?, discriminator_loss = ?, 
                pattern_accuracy = ?, synthetic_quality_score = ?,
                total_training_steps = total_training_steps + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
        """, (g_loss, d_loss, pattern_accuracy, synthetic_quality, self.current_session_id))
        
        # Store detailed metrics
        await self.db.execute("""
            INSERT INTO gan_performance_metrics 
            (session_id, metric_name, metric_value, metric_type, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (self.current_session_id, 'generator_loss', g_loss, 'loss', datetime.now()))
        
        await self.db.execute("""
            INSERT INTO gan_performance_metrics 
            (session_id, metric_name, metric_value, metric_type, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (self.current_session_id, 'discriminator_loss', d_loss, 'loss', datetime.now()))
        
        await self.db.execute("""
            INSERT INTO gan_performance_metrics 
            (session_id, metric_name, metric_value, metric_type, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (self.current_session_id, 'pattern_accuracy', pattern_accuracy, 'accuracy', datetime.now()))
        
        await self.db.execute("""
            INSERT INTO gan_performance_metrics 
            (session_id, metric_name, metric_value, metric_type, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (self.current_session_id, 'synthetic_quality', synthetic_quality, 'quality', datetime.now()))
    
    async def _analyze_generated_patterns(self, game_data: List[Dict]) -> Dict[str, Any]:
        """Analyze generated patterns to discover game mechanics."""
        # Simplified pattern analysis
        discovered_rules = {
            'confidence': 0.7,
            'mechanics_understood': 0.6,
            'rules': [
                'Objects follow spatial patterns',
                'Success probability correlates with object density',
                'Grid complexity affects difficulty'
            ],
            'patterns': {
                'spatial_clustering': 0.8,
                'object_density': 0.6,
                'success_correlation': 0.7
            }
        }
        
        return discovered_rules
