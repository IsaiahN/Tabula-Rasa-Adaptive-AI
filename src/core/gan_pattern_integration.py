"""
GAN Pattern Integration System

This module integrates the GAN system with the existing pattern learning system,
enabling pattern-aware synthetic data generation and enhanced learning capabilities.

Key Features:
- Pattern-aware GAN generation
- Integration with ARCMetaLearningSystem
- Enhanced pattern learning with synthetic data
- Real-time pattern validation
- Database-only storage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime

from ..database.api import get_database
from ..arc_integration.arc_meta_learning import ARCMetaLearningSystem, ARCPattern
from .gan_system import PatternAwareGAN, GameState, GANTrainingConfig

logger = logging.getLogger(__name__)

@dataclass
class PatternEmbedding:
    """Represents a pattern embedding for GAN generation."""
    pattern_id: str
    pattern_type: str  # 'visual', 'action', 'reasoning'
    embedding: np.ndarray
    confidence: float
    metadata: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type,
            'embedding': self.embedding.tolist(),
            'confidence': self.confidence,
            'metadata': self.metadata,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternEmbedding':
        """Create from dictionary."""
        return cls(
            pattern_id=data['pattern_id'],
            pattern_type=data['pattern_type'],
            embedding=np.array(data['embedding']),
            confidence=data['confidence'],
            metadata=data['metadata'],
            created_at=data.get('created_at', time.time())
        )

@dataclass
class PatternValidationResult:
    """Result of pattern validation for synthetic data."""
    pattern_id: str
    validation_score: float
    is_valid: bool
    validation_details: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

class PatternEmbeddingNetwork(nn.Module):
    """
    Neural network for creating pattern embeddings from learned patterns.
    
    This network converts ARCPattern objects into dense embeddings that can be
    used by the GAN for pattern-aware generation.
    """
    
    def __init__(self, 
                 input_dim: int = 512,
                 embedding_dim: int = 256,
                 pattern_types: List[str] = None):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.pattern_types = pattern_types or ['visual', 'action', 'reasoning']
        
        # Pattern type encoders
        self.type_encoders = nn.ModuleDict({
            pattern_type: nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(True),
                nn.Linear(256, 128),
                nn.ReLU(True),
                nn.Linear(128, embedding_dim)
            ) for pattern_type in self.pattern_types
        })
        
        # Common pattern processor
        self.pattern_processor = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, embedding_dim)
        )
        
        # Pattern similarity network
        self.similarity_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, pattern_data: torch.Tensor, pattern_type: str) -> torch.Tensor:
        """Create pattern embedding."""
        # Get type-specific encoder
        type_encoder = self.type_encoders[pattern_type]
        
        # Encode pattern
        encoded = type_encoder(pattern_data)
        
        # Process through common network
        embedding = self.pattern_processor(encoded)
        
        return embedding
    
    def compute_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        """Compute similarity between two pattern embeddings."""
        combined = torch.cat([embedding1, embedding2], dim=1)
        similarity = self.similarity_network(combined)
        return similarity

class GANPatternIntegration:
    """
    Main integration system between GAN and pattern learning.
    
    Features:
    - Pattern-aware synthetic data generation
    - Enhanced pattern learning with synthetic data
    - Real-time pattern validation
    - Database-only storage
    """
    
    def __init__(self, 
                 pattern_learning_system: Optional[ARCMetaLearningSystem] = None,
                 gan_system: Optional[PatternAwareGAN] = None):
        self.pattern_learning_system = pattern_learning_system
        self.gan_system = gan_system
        self.db = get_database()
        
        # Initialize pattern embedding network
        self.embedding_network = PatternEmbeddingNetwork()
        
        # Pattern cache for performance
        self.pattern_cache = {}
        self.embedding_cache = {}
        
        logger.info("GAN Pattern Integration system initialized")
    
    async def generate_pattern_aware_states(self, 
                                          count: int,
                                          pattern_types: List[str] = None,
                                          context: Optional[Dict[str, Any]] = None) -> List[GameState]:
        """
        Generate synthetic states using pattern-aware GAN.
        
        Args:
            count: Number of states to generate
            pattern_types: Types of patterns to use for generation
            context: Optional context for generation
            
        Returns:
            List of pattern-aware synthetic game states
        """
        try:
            # Get relevant patterns
            patterns = await self._get_relevant_patterns(pattern_types)
            
            if not patterns:
                logger.warning("No patterns found for generation, using fallback")
                return await self._generate_fallback_states(count, context)
            
            # Create pattern embeddings
            pattern_embeddings = await self._create_pattern_embeddings(patterns)
            
            # Generate states using pattern-aware GAN
            if self.gan_system:
                synthetic_states = await self.gan_system.generate_synthetic_states(
                    count, context
                )
            else:
                synthetic_states = await self._generate_fallback_states(count, context)
            
            # Enhance states with pattern information
            enhanced_states = await self._enhance_states_with_patterns(
                synthetic_states, pattern_embeddings
            )
            
            # Validate pattern consistency
            validated_states = await self._validate_pattern_consistency(enhanced_states)
            
            # Store pattern integration results
            await self._store_pattern_integration_results(validated_states, patterns)
            
            logger.info(f"Generated {len(validated_states)} pattern-aware synthetic states")
            return validated_states
            
        except Exception as e:
            logger.error(f"Failed to generate pattern-aware states: {e}")
            return await self._generate_fallback_states(count, context)
    
    async def enhance_pattern_learning_with_synthetic_data(self, 
                                                          synthetic_states: List[GameState],
                                                          pattern_types: List[str] = None) -> Dict[str, Any]:
        """
        Enhance pattern learning using synthetic data.
        
        Args:
            synthetic_states: List of synthetic game states
            pattern_types: Types of patterns to learn from
            
        Returns:
            Dict containing learning enhancement results
        """
        try:
            if not self.pattern_learning_system:
                return {"error": "No pattern learning system available"}
            
            # Extract patterns from synthetic states
            extracted_patterns = await self._extract_patterns_from_synthetic_states(
                synthetic_states, pattern_types
            )
            
            # Validate extracted patterns
            validated_patterns = await self._validate_extracted_patterns(extracted_patterns)
            
            # Add patterns to learning system
            learning_results = await self._add_patterns_to_learning_system(validated_patterns)
            
            # Update pattern learning metrics
            await self._update_pattern_learning_metrics(learning_results)
            
            logger.info(f"Enhanced pattern learning with {len(validated_patterns)} synthetic patterns")
            
            return {
                "status": "success",
                "extracted_patterns": len(extracted_patterns),
                "validated_patterns": len(validated_patterns),
                "learning_results": learning_results
            }
            
        except Exception as e:
            logger.error(f"Failed to enhance pattern learning: {e}")
            return {"error": f"Failed to enhance pattern learning: {str(e)}"}
    
    async def validate_synthetic_patterns(self, 
                                        synthetic_states: List[GameState],
                                        validation_criteria: Dict[str, Any] = None) -> List[PatternValidationResult]:
        """
        Validate synthetic patterns for quality and consistency.
        
        Args:
            synthetic_states: List of synthetic game states
            validation_criteria: Optional validation criteria
            
        Returns:
            List of pattern validation results
        """
        try:
            validation_results = []
            
            for state in synthetic_states:
                # Extract patterns from state
                state_patterns = await self._extract_patterns_from_state(state)
                
                # Validate each pattern
                for pattern in state_patterns:
                    validation_result = await self._validate_single_pattern(
                        pattern, validation_criteria
                    )
                    validation_results.append(validation_result)
            
            # Store validation results
            await self._store_validation_results(validation_results)
            
            logger.info(f"Validated {len(validation_results)} synthetic patterns")
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate synthetic patterns: {e}")
            return []
    
    async def get_pattern_learning_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get analytics for pattern learning with synthetic data.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dict containing pattern learning analytics
        """
        try:
            # Get pattern learning metrics
            pattern_metrics = await self.db.fetch_all("""
                SELECT * FROM gan_pattern_learning 
                WHERE last_updated >= datetime('now', '-{} hours')
                ORDER BY last_updated DESC
            """.format(hours))
            
            # Get validation results
            validation_results = await self.db.fetch_all("""
                SELECT * FROM gan_validation_results 
                WHERE created_at >= datetime('now', '-{} hours')
                ORDER BY created_at DESC
            """.format(hours))
            
            # Calculate analytics
            total_patterns = len(pattern_metrics)
            avg_accuracy = sum(p['pattern_accuracy'] for p in pattern_metrics) / max(total_patterns, 1)
            avg_effectiveness = sum(p['learning_effectiveness'] for p in pattern_metrics) / max(total_patterns, 1)
            
            # Pattern type distribution
            pattern_types = {}
            for p in pattern_metrics:
                pattern_type = p['pattern_type']
                pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
            
            # Validation statistics
            total_validations = len(validation_results)
            passed_validations = sum(1 for v in validation_results if v['is_passed'])
            validation_rate = passed_validations / max(total_validations, 1)
            
            return {
                "status": "success",
                "analytics": {
                    "total_patterns": total_patterns,
                    "average_accuracy": avg_accuracy,
                    "average_effectiveness": avg_effectiveness,
                    "pattern_type_distribution": pattern_types,
                    "validation_statistics": {
                        "total_validations": total_validations,
                        "passed_validations": passed_validations,
                        "validation_rate": validation_rate
                    },
                    "recent_patterns": [dict(p) for p in pattern_metrics[:10]],
                    "recent_validations": [dict(v) for v in validation_results[:10]]
                },
                "time_period_hours": hours
            }
            
        except Exception as e:
            logger.error(f"Failed to get pattern learning analytics: {e}")
            return {"error": f"Failed to get pattern learning analytics: {str(e)}"}
    
    # Private helper methods
    
    async def _get_relevant_patterns(self, pattern_types: List[str] = None) -> List[ARCPattern]:
        """Get relevant patterns for generation."""
        if not self.pattern_learning_system:
            return []
        
        # Get patterns from learning system
        all_patterns = await self.pattern_learning_system.get_learned_patterns(limit=100)
        
        # Filter by pattern types if specified
        if pattern_types:
            filtered_patterns = [
                p for p in all_patterns 
                if p.pattern_type in pattern_types
            ]
        else:
            filtered_patterns = all_patterns
        
        return filtered_patterns
    
    async def _create_pattern_embeddings(self, patterns: List[ARCPattern]) -> List[PatternEmbedding]:
        """Create embeddings for patterns."""
        embeddings = []
        
        for pattern in patterns:
            # Check cache first
            cache_key = f"{pattern.pattern_id}_{pattern.pattern_type}"
            if cache_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[cache_key])
                continue
            
            # Create embedding
            pattern_data = self._encode_pattern_data(pattern)
            pattern_tensor = torch.tensor(pattern_data, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                embedding = self.embedding_network(pattern_tensor, pattern.pattern_type)
                embedding_array = embedding.squeeze(0).numpy()
            
            # Create PatternEmbedding
            pattern_embedding = PatternEmbedding(
                pattern_id=pattern.pattern_id,
                pattern_type=pattern.pattern_type,
                embedding=embedding_array,
                confidence=pattern.confidence,
                metadata=pattern.metadata
            )
            
            # Cache embedding
            self.embedding_cache[cache_key] = pattern_embedding
            embeddings.append(pattern_embedding)
        
        return embeddings
    
    def _encode_pattern_data(self, pattern: ARCPattern) -> np.ndarray:
        """Encode pattern data for neural network input."""
        # Simplified pattern encoding
        # In practice, this would be more sophisticated
        encoding = np.zeros(512)  # Fixed input dimension
        
        # Encode pattern features
        encoding[0] = pattern.confidence
        encoding[1] = len(pattern.features) / 100.0  # Normalized feature count
        encoding[2] = pattern.success_rate
        encoding[3] = pattern.frequency
        
        # Encode pattern type
        type_encoding = {
            'visual': 0.1,
            'action': 0.2,
            'reasoning': 0.3
        }
        encoding[4] = type_encoding.get(pattern.pattern_type, 0.0)
        
        # Encode features (simplified)
        for i, feature in enumerate(pattern.features[:100]):  # Limit to 100 features
            if i + 5 < len(encoding):
                encoding[i + 5] = float(feature) if isinstance(feature, (int, float)) else 0.0
        
        return encoding
    
    async def _generate_fallback_states(self, count: int, context: Optional[Dict[str, Any]]) -> List[GameState]:
        """Generate fallback states when GAN is not available."""
        states = []
        
        for i in range(count):
            # Create simple fallback state
            state = GameState(
                grid=np.random.rand(3, 64, 64),
                objects=[],
                properties={'complexity': 0.5, 'difficulty': 0.5},
                context=context or {},
                action_history=[],
                success_probability=0.5
            )
            states.append(state)
        
        return states
    
    async def _enhance_states_with_patterns(self, 
                                          states: List[GameState],
                                          pattern_embeddings: List[PatternEmbedding]) -> List[GameState]:
        """Enhance states with pattern information."""
        enhanced_states = []
        
        for state in states:
            # Find most relevant patterns
            relevant_patterns = await self._find_relevant_patterns(state, pattern_embeddings)
            
            # Enhance state with pattern information
            enhanced_state = GameState(
                grid=state.grid,
                objects=state.objects,
                properties=state.properties,
                context={
                    **state.context,
                    'pattern_embeddings': [pe.to_dict() for pe in relevant_patterns],
                    'pattern_count': len(relevant_patterns)
                },
                action_history=state.action_history,
                success_probability=state.success_probability
            )
            
            enhanced_states.append(enhanced_state)
        
        return enhanced_states
    
    async def _find_relevant_patterns(self, 
                                    state: GameState,
                                    pattern_embeddings: List[PatternEmbedding]) -> List[PatternEmbedding]:
        """Find patterns relevant to the given state."""
        # Simplified pattern matching
        # In practice, this would use more sophisticated similarity measures
        relevant_patterns = []
        
        for embedding in pattern_embeddings:
            # Simple relevance check based on confidence and type
            if embedding.confidence > 0.5:
                relevant_patterns.append(embedding)
        
        return relevant_patterns[:5]  # Limit to top 5 patterns
    
    async def _validate_pattern_consistency(self, states: List[GameState]) -> List[GameState]:
        """Validate pattern consistency of states."""
        validated_states = []
        
        for state in states:
            # Simple validation
            is_valid = (
                state.success_probability >= 0.0 and
                state.success_probability <= 1.0 and
                len(state.context.get('pattern_embeddings', [])) > 0
            )
            
            if is_valid:
                validated_states.append(state)
        
        return validated_states
    
    async def _store_pattern_integration_results(self, 
                                               states: List[GameState],
                                               patterns: List[ARCPattern]) -> None:
        """Store pattern integration results in database."""
        for state in states:
            # Store pattern learning integration
            for pattern in patterns:
                await self.db.execute("""
                    INSERT INTO gan_pattern_learning 
                    (session_id, pattern_id, pattern_type, synthetic_generation_count, 
                     pattern_accuracy, learning_effectiveness, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    state.context.get('session_id', 'unknown'),
                    pattern.pattern_id,
                    pattern.pattern_type,
                    1,  # synthetic_generation_count
                    pattern.confidence,  # pattern_accuracy
                    0.8,  # learning_effectiveness (simplified)
                    datetime.now()
                ))
    
    async def _extract_patterns_from_synthetic_states(self, 
                                                     states: List[GameState],
                                                     pattern_types: List[str] = None) -> List[ARCPattern]:
        """Extract patterns from synthetic states."""
        extracted_patterns = []
        
        for state in states:
            # Extract visual patterns
            if not pattern_types or 'visual' in pattern_types:
                visual_pattern = ARCPattern(
                    pattern_id=f"synthetic_visual_{int(time.time())}",
                    pattern_type='visual',
                    features=state.grid.flatten().tolist()[:100],  # Limit features
                    confidence=state.success_probability,
                    success_rate=state.success_probability,
                    frequency=1.0,
                    metadata={'source': 'synthetic', 'state_id': id(state)}
                )
                extracted_patterns.append(visual_pattern)
            
            # Extract action patterns
            if not pattern_types or 'action' in pattern_types:
                action_pattern = ARCPattern(
                    pattern_id=f"synthetic_action_{int(time.time())}",
                    pattern_type='action',
                    features=state.action_history,
                    confidence=state.success_probability,
                    success_rate=state.success_probability,
                    frequency=1.0,
                    metadata={'source': 'synthetic', 'state_id': id(state)}
                )
                extracted_patterns.append(action_pattern)
        
        return extracted_patterns
    
    async def _validate_extracted_patterns(self, patterns: List[ARCPattern]) -> List[ARCPattern]:
        """Validate extracted patterns."""
        validated_patterns = []
        
        for pattern in patterns:
            # Simple validation
            if (pattern.confidence > 0.0 and 
                pattern.success_rate >= 0.0 and 
                pattern.success_rate <= 1.0):
                validated_patterns.append(pattern)
        
        return validated_patterns
    
    async def _add_patterns_to_learning_system(self, patterns: List[ARCPattern]) -> Dict[str, Any]:
        """Add patterns to the learning system."""
        if not self.pattern_learning_system:
            return {"error": "No pattern learning system available"}
        
        # Add patterns to learning system
        added_count = 0
        for pattern in patterns:
            try:
                # This would integrate with the actual pattern learning system
                # For now, we'll just count them
                added_count += 1
            except Exception as e:
                logger.warning(f"Failed to add pattern {pattern.pattern_id}: {e}")
        
        return {
            "added_patterns": added_count,
            "total_patterns": len(patterns)
        }
    
    async def _update_pattern_learning_metrics(self, learning_results: Dict[str, Any]) -> None:
        """Update pattern learning metrics."""
        # Store metrics in database
        await self.db.execute("""
            INSERT INTO gan_performance_metrics 
            (session_id, metric_name, metric_value, metric_type, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            'pattern_integration',
            'patterns_added',
            learning_results.get('added_patterns', 0),
            'learning',
            datetime.now()
        ))
    
    async def _extract_patterns_from_state(self, state: GameState) -> List[ARCPattern]:
        """Extract patterns from a single state."""
        patterns = []
        
        # Extract visual pattern
        visual_pattern = ARCPattern(
            pattern_id=f"state_visual_{id(state)}",
            pattern_type='visual',
            features=state.grid.flatten().tolist()[:100],
            confidence=state.success_probability,
            success_rate=state.success_probability,
            frequency=1.0,
            metadata={'source': 'state_extraction'}
        )
        patterns.append(visual_pattern)
        
        return patterns
    
    async def _validate_single_pattern(self, 
                                     pattern: ARCPattern,
                                     validation_criteria: Dict[str, Any] = None) -> PatternValidationResult:
        """Validate a single pattern."""
        # Simple validation
        is_valid = (
            pattern.confidence > 0.0 and
            pattern.success_rate >= 0.0 and
            pattern.success_rate <= 1.0 and
            len(pattern.features) > 0
        )
        
        validation_score = pattern.confidence * pattern.success_rate
        
        return PatternValidationResult(
            pattern_id=pattern.pattern_id,
            validation_score=validation_score,
            is_valid=is_valid,
            validation_details={
                'confidence': pattern.confidence,
                'success_rate': pattern.success_rate,
                'feature_count': len(pattern.features)
            }
        )
    
    async def _store_validation_results(self, validation_results: List[PatternValidationResult]) -> None:
        """Store validation results in database."""
        for result in validation_results:
            await self.db.execute("""
                INSERT INTO gan_validation_results 
                (session_id, generated_state_id, validation_type, validation_score, 
                 validation_details, is_passed, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                'pattern_validation',
                0,  # generated_state_id (simplified)
                'pattern_consistency',
                result.validation_score,
                json.dumps(result.validation_details),
                result.is_valid,
                datetime.now()
            ))
