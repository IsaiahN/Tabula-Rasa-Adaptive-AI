#!/usr/bin/env python3
"""
Aha! Moment Simulator for Tabula Rasa

Implements latent space exploration and restructuring simulation
for insight moments, building on existing simulation systems.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import math
from collections import deque

logger = logging.getLogger(__name__)

class RestructuringType(Enum):
    """Types of problem restructuring."""
    REPRESENTATION_CHANGE = "representation_change"
    CONSTRAINT_RELAXATION = "constraint_relaxation"
    PERSPECTIVE_SHIFT = "perspective_shift"
    PATTERN_RECOGNITION = "pattern_recognition"
    ANALOGY_MAPPING = "analogy_mapping"
    INSIGHT_BREAKTHROUGH = "insight_breakthrough"

class ExplorationStrategy(Enum):
    """Strategies for latent space exploration."""
    RANDOM_WALK = "random_walk"
    GRADIENT_ASCENT = "gradient_ascent"
    SIMULATED_ANNEALING = "simulated_annealing"
    GENETIC_ALGORITHM = "genetic_algorithm"
    DIFFUSION_SAMPLING = "diffusion_sampling"

@dataclass
class ProblemRepresentation:
    """Represents a problem in latent space."""
    latent_vector: np.ndarray
    features: Dict[str, Any]
    constraints: List[str]
    goals: List[str]
    difficulty: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class RestructuringEvent:
    """Represents a restructuring event."""
    restructuring_type: RestructuringType
    before_representation: ProblemRepresentation
    after_representation: ProblemRepresentation
    confidence_gain: float
    entropy_change: float
    timestamp: float
    exploration_strategy: ExplorationStrategy

@dataclass
class AhaMoment:
    """Represents an Aha! moment with full context."""
    timestamp: float
    restructuring_event: RestructuringEvent
    insight_quality: float
    solution_confidence: float
    problem_solved: bool
    context: Dict[str, Any]

class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for latent space exploration."""
    
    def __init__(self, input_dim: int = 512, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space."""
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to output."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

class DiffusionModel(nn.Module):
    """Diffusion model for latent space exploration."""
    
    def __init__(self, latent_dim: int = 128, num_timesteps: int = 1000):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps
        
        # Simple diffusion model
        self.noise_predictor = nn.Sequential(
            nn.Linear(latent_dim + 1, 256),  # +1 for timestep
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise at timestep t."""
        t_embed = t.unsqueeze(-1).expand(-1, 1)
        x_with_t = torch.cat([x, t_embed], dim=-1)
        return self.noise_predictor(x_with_t)
    
    def sample(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample from the diffusion model."""
        # Start with pure noise
        x = torch.randn(shape, device=device)
        
        # Denoise step by step
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((shape[0],), t, device=device)
            predicted_noise = self.forward(x, t_tensor)
            
            # Simple denoising step
            alpha = 0.01  # Learning rate
            x = x - alpha * predicted_noise
        
        return x

class RestructuringRewardSystem:
    """Reward system for evaluating restructuring success."""
    
    def __init__(self):
        self.restructuring_history = deque(maxlen=100)
        self.success_rates = {}
        
    def evaluate_restructuring(self, 
                              before_rep: ProblemRepresentation,
                              after_rep: ProblemRepresentation,
                              solution_quality: float) -> float:
        """Evaluate the quality of a restructuring."""
        
        # Calculate various reward components
        representation_change = self._calculate_representation_change(before_rep, after_rep)
        constraint_improvement = self._calculate_constraint_improvement(before_rep, after_rep)
        goal_progress = self._calculate_goal_progress(before_rep, after_rep)
        solution_improvement = solution_quality - before_rep.difficulty
        
        # Combine rewards
        total_reward = (
            representation_change * 0.3 +
            constraint_improvement * 0.3 +
            goal_progress * 0.2 +
            solution_improvement * 0.2
        )
        
        # Record for learning
        self.restructuring_history.append({
            'before': before_rep,
            'after': after_rep,
            'reward': total_reward,
            'timestamp': time.time()
        })
        
        return total_reward
    
    def _calculate_representation_change(self, 
                                       before: ProblemRepresentation, 
                                       after: ProblemRepresentation) -> float:
        """Calculate how much the representation changed."""
        # Calculate cosine similarity between latent vectors
        before_vec = torch.tensor(before.latent_vector, dtype=torch.float32)
        after_vec = torch.tensor(after.latent_vector, dtype=torch.float32)
        
        similarity = F.cosine_similarity(before_vec.unsqueeze(0), after_vec.unsqueeze(0))
        change_magnitude = 1.0 - similarity.item()
        
        return change_magnitude
    
    def _calculate_constraint_improvement(self, 
                                        before: ProblemRepresentation, 
                                        after: ProblemRepresentation) -> float:
        """Calculate improvement in constraint satisfaction."""
        # Simplified constraint improvement calculation
        before_constraints = len(before.constraints)
        after_constraints = len(after.constraints)
        
        if before_constraints == 0:
            return 0.0
        
        improvement = (before_constraints - after_constraints) / before_constraints
        return max(0.0, improvement)
    
    def _calculate_goal_progress(self, 
                               before: ProblemRepresentation, 
                               after: ProblemRepresentation) -> float:
        """Calculate progress toward goals."""
        # Simplified goal progress calculation
        before_goals = len(before.goals)
        after_goals = len(after.goals)
        
        if before_goals == 0:
            return 0.0
        
        progress = (before_goals - after_goals) / before_goals
        return max(0.0, progress)

class AhaMomentSimulator:
    """Main Aha! moment simulator with latent space exploration."""
    
    def __init__(self, 
                 latent_dim: int = 128,
                 input_dim: int = 512,
                 exploration_strategies: List[ExplorationStrategy] = None):
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        # Initialize models
        self.vae = VariationalAutoencoder(input_dim, latent_dim)
        self.diffusion_model = DiffusionModel(latent_dim)
        self.reward_system = RestructuringRewardSystem()
        
        # Exploration strategies
        self.exploration_strategies = exploration_strategies or list(ExplorationStrategy)
        
        # History and statistics
        self.aha_moments = deque(maxlen=50)
        self.restructuring_events = deque(maxlen=100)
        self.exploration_history = deque(maxlen=200)
        
        logger.info(f"Aha! Moment Simulator initialized with latent_dim={latent_dim}")
    
    def simulate_aha_moment(self, 
                           problem_representation: ProblemRepresentation,
                           context: Dict[str, Any]) -> Optional[AhaMoment]:
        """Simulate an Aha! moment through latent space exploration."""
        
        # Encode problem into latent space
        problem_tensor = torch.tensor(problem_representation.latent_vector, dtype=torch.float32)
        if problem_tensor.dim() == 1:
            problem_tensor = problem_tensor.unsqueeze(0)
        
        # Explore latent space for restructuring
        exploration_results = self._explore_latent_space(problem_tensor, context)
        
        if not exploration_results:
            return None
        
        # Find best restructuring
        best_restructuring = self._select_best_restructuring(exploration_results, problem_representation)
        
        if best_restructuring is None:
            return None
        
        # Create Aha! moment
        aha_moment = self._create_aha_moment(best_restructuring, problem_representation, context)
        
        # Record the moment
        self.aha_moments.append(aha_moment)
        self.restructuring_events.append(best_restructuring)
        
        return aha_moment
    
    def _explore_latent_space(self, 
                             problem_tensor: torch.Tensor, 
                             context: Dict[str, Any]) -> List[RestructuringEvent]:
        """Explore latent space using various strategies."""
        exploration_results = []
        
        for strategy in self.exploration_strategies:
            try:
                if strategy == ExplorationStrategy.RANDOM_WALK:
                    results = self._random_walk_exploration(problem_tensor, context)
                elif strategy == ExplorationStrategy.GRADIENT_ASCENT:
                    results = self._gradient_ascent_exploration(problem_tensor, context)
                elif strategy == ExplorationStrategy.SIMULATED_ANNEALING:
                    results = self._simulated_annealing_exploration(problem_tensor, context)
                elif strategy == ExplorationStrategy.DIFFUSION_SAMPLING:
                    results = self._diffusion_sampling_exploration(problem_tensor, context)
                else:
                    continue
                
                exploration_results.extend(results)
                
            except Exception as e:
                logger.warning(f"Exploration strategy {strategy.value} failed: {e}")
                continue
        
        return exploration_results
    
    def _random_walk_exploration(self, 
                                problem_tensor: torch.Tensor, 
                                context: Dict[str, Any]) -> List[RestructuringEvent]:
        """Random walk exploration in latent space."""
        results = []
        num_steps = context.get('random_walk_steps', 10)
        step_size = context.get('random_walk_step_size', 0.1)
        
        current_latent = problem_tensor.clone()
        
        for step in range(num_steps):
            # Generate random direction
            direction = torch.randn_like(current_latent)
            direction = direction / torch.norm(direction)
            
            # Take step
            new_latent = current_latent + step_size * direction
            
            # Create restructuring event
            restructuring = self._create_restructuring_event(
                problem_tensor, new_latent, RestructuringType.REPRESENTATION_CHANGE,
                ExplorationStrategy.RANDOM_WALK, context
            )
            
            if restructuring:
                results.append(restructuring)
                current_latent = new_latent
        
        return results
    
    def _gradient_ascent_exploration(self, 
                                   problem_tensor: torch.Tensor, 
                                   context: Dict[str, Any]) -> List[RestructuringEvent]:
        """Gradient ascent exploration in latent space."""
        results = []
        num_steps = context.get('gradient_steps', 5)
        learning_rate = context.get('gradient_lr', 0.01)
        
        current_latent = problem_tensor.clone()
        current_latent.requires_grad_(True)
        
        for step in range(num_steps):
            # Calculate gradient (simplified)
            # In practice, this would use a more sophisticated objective function
            objective = torch.norm(current_latent)  # Simplified objective
            objective.backward()
            
            if current_latent.grad is not None:
                # Update latent vector
                with torch.no_grad():
                    current_latent = current_latent + learning_rate * current_latent.grad
                    current_latent.grad.zero_()
                
                # Create restructuring event
                restructuring = self._create_restructuring_event(
                    problem_tensor, current_latent.detach(), 
                    RestructuringType.REPRESENTATION_CHANGE,
                    ExplorationStrategy.GRADIENT_ASCENT, context
                )
                
                if restructuring:
                    results.append(restructuring)
        
        return results
    
    def _simulated_annealing_exploration(self, 
                                       problem_tensor: torch.Tensor, 
                                       context: Dict[str, Any]) -> List[RestructuringEvent]:
        """Simulated annealing exploration in latent space."""
        results = []
        num_steps = context.get('annealing_steps', 20)
        initial_temp = context.get('initial_temp', 1.0)
        cooling_rate = context.get('cooling_rate', 0.95)
        
        current_latent = problem_tensor.clone()
        current_energy = self._calculate_energy(current_latent, context)
        temperature = initial_temp
        
        for step in range(num_steps):
            # Generate candidate
            candidate = current_latent + torch.randn_like(current_latent) * temperature
            
            # Calculate energy
            candidate_energy = self._calculate_energy(candidate, context)
            
            # Accept or reject
            if candidate_energy < current_energy or np.random.random() < np.exp(-(candidate_energy - current_energy) / temperature):
                current_latent = candidate
                current_energy = candidate_energy
                
                # Create restructuring event
                restructuring = self._create_restructuring_event(
                    problem_tensor, current_latent, 
                    RestructuringType.REPRESENTATION_CHANGE,
                    ExplorationStrategy.SIMULATED_ANNEALING, context
                )
                
                if restructuring:
                    results.append(restructuring)
            
            # Cool down
            temperature *= cooling_rate
        
        return results
    
    def _diffusion_sampling_exploration(self, 
                                      problem_tensor: torch.Tensor, 
                                      context: Dict[str, Any]) -> List[RestructuringEvent]:
        """Diffusion sampling exploration in latent space."""
        results = []
        num_samples = context.get('diffusion_samples', 5)
        
        for _ in range(num_samples):
            # Sample from diffusion model
            sampled_latent = self.diffusion_model.sample(problem_tensor.shape, problem_tensor.device)
            
            # Create restructuring event
            restructuring = self._create_restructuring_event(
                problem_tensor, sampled_latent, 
                RestructuringType.REPRESENTATION_CHANGE,
                ExplorationStrategy.DIFFUSION_SAMPLING, context
            )
            
            if restructuring:
                results.append(restructuring)
        
        return results
    
    def _create_restructuring_event(self, 
                                   original_latent: torch.Tensor,
                                   new_latent: torch.Tensor,
                                   restructuring_type: RestructuringType,
                                   exploration_strategy: ExplorationStrategy,
                                   context: Dict[str, Any]) -> Optional[RestructuringEvent]:
        """Create a restructuring event from latent space exploration."""
        
        # Create problem representations
        before_rep = ProblemRepresentation(
            latent_vector=original_latent.detach().numpy().flatten(),
            features=context.get('features', {}),
            constraints=context.get('constraints', []),
            goals=context.get('goals', []),
            difficulty=context.get('difficulty', 0.5),
            timestamp=time.time()
        )
        
        after_rep = ProblemRepresentation(
            latent_vector=new_latent.detach().numpy().flatten(),
            features=context.get('features', {}),
            constraints=context.get('constraints', []),
            goals=context.get('goals', []),
            difficulty=context.get('difficulty', 0.5),
            timestamp=time.time()
        )
        
        # Calculate confidence gain and entropy change
        confidence_gain = self._calculate_confidence_gain(before_rep, after_rep)
        entropy_change = self._calculate_entropy_change(before_rep, after_rep)
        
        # Create restructuring event
        restructuring = RestructuringEvent(
            restructuring_type=restructuring_type,
            before_representation=before_rep,
            after_representation=after_rep,
            confidence_gain=confidence_gain,
            entropy_change=entropy_change,
            timestamp=time.time(),
            exploration_strategy=exploration_strategy
        )
        
        return restructuring
    
    def _select_best_restructuring(self, 
                                  exploration_results: List[RestructuringEvent],
                                  original_problem: ProblemRepresentation) -> Optional[RestructuringEvent]:
        """Select the best restructuring from exploration results."""
        if not exploration_results:
            return None
        
        # Evaluate each restructuring
        best_restructuring = None
        best_score = -float('inf')
        
        for restructuring in exploration_results:
            # Calculate reward
            reward = self.reward_system.evaluate_restructuring(
                restructuring.before_representation,
                restructuring.after_representation,
                restructuring.confidence_gain
            )
            
            # Combine reward with other factors
            score = (
                reward * 0.4 +
                restructuring.confidence_gain * 0.3 +
                abs(restructuring.entropy_change) * 0.3
            )
            
            if score > best_score:
                best_score = score
                best_restructuring = restructuring
        
        return best_restructuring
    
    def _create_aha_moment(self, 
                          restructuring: RestructuringEvent,
                          original_problem: ProblemRepresentation,
                          context: Dict[str, Any]) -> AhaMoment:
        """Create an Aha! moment from restructuring event."""
        
        # Calculate insight quality
        insight_quality = (
            restructuring.confidence_gain * 0.4 +
            abs(restructuring.entropy_change) * 0.3 +
            self._calculate_representation_novelty(restructuring) * 0.3
        )
        
        # Calculate solution confidence
        solution_confidence = min(1.0, restructuring.confidence_gain + 0.5)
        
        # Determine if problem was solved
        problem_solved = solution_confidence > 0.7 and insight_quality > 0.6
        
        # Create Aha! moment
        aha_moment = AhaMoment(
            timestamp=time.time(),
            restructuring_event=restructuring,
            insight_quality=insight_quality,
            solution_confidence=solution_confidence,
            problem_solved=problem_solved,
            context=context
        )
        
        return aha_moment
    
    def _calculate_energy(self, latent_vector: torch.Tensor, context: Dict[str, Any]) -> float:
        """Calculate energy for simulated annealing."""
        # Simplified energy calculation
        return torch.norm(latent_vector).item()
    
    def _calculate_confidence_gain(self, before: ProblemRepresentation, after: ProblemRepresentation) -> float:
        """Calculate confidence gain from restructuring."""
        # Simplified confidence gain calculation
        return np.random.random() * 0.5  # Placeholder
    
    def _calculate_entropy_change(self, before: ProblemRepresentation, after: ProblemRepresentation) -> float:
        """Calculate entropy change from restructuring."""
        # Simplified entropy change calculation
        return np.random.random() * 0.3 - 0.15  # Placeholder
    
    def _calculate_representation_novelty(self, restructuring: RestructuringEvent) -> float:
        """Calculate novelty of the new representation."""
        # Simplified novelty calculation
        return np.random.random() * 0.4 + 0.3  # Placeholder
    
    def get_simulator_statistics(self) -> Dict[str, Any]:
        """Get statistics about the Aha! moment simulator."""
        return {
            'total_aha_moments': len(self.aha_moments),
            'total_restructuring_events': len(self.restructuring_events),
            'successful_aha_moments': sum(1 for moment in self.aha_moments if moment.problem_solved),
            'avg_insight_quality': np.mean([moment.insight_quality for moment in self.aha_moments]) if self.aha_moments else 0.0,
            'avg_solution_confidence': np.mean([moment.solution_confidence for moment in self.aha_moments]) if self.aha_moments else 0.0,
            'exploration_strategies_used': [strategy.value for strategy in self.exploration_strategies]
        }

# Factory function for easy integration
def create_aha_moment_simulator(**kwargs) -> AhaMomentSimulator:
    """Create a configured Aha! moment simulator."""
    return AhaMomentSimulator(**kwargs)
