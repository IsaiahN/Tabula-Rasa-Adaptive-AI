"""
Attention Mechanisms for Vision

Advanced attention mechanisms for visual processing including
spatial attention, temporal attention, and multi-head attention.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..interfaces import ComponentInterface
from ...training.caching import CacheManager, CacheConfig
from ...training.monitoring import PerformanceMonitor


class AttentionType(Enum):
    """Available attention types."""
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    CHANNEL = "channel"
    MULTI_HEAD = "multi_head"
    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms."""
    attention_type: AttentionType = AttentionType.SPATIAL
    num_heads: int = 8
    attention_dim: int = 64
    dropout_rate: float = 0.1
    temperature: float = 1.0
    use_positional_encoding: bool = True
    enable_caching: bool = True
    cache_ttl: int = 300  # seconds


@dataclass
class AttentionResult:
    """Result of attention computation."""
    attention_weights: np.ndarray
    attended_features: np.ndarray
    attention_map: np.ndarray
    confidence: float
    timestamp: datetime


class AttentionMechanism(ComponentInterface):
    """
    Advanced attention mechanism for visual processing.
    Supports multiple attention types and configurations.
    """
    
    def __init__(self, config: AttentionConfig, cache_config: Optional[CacheConfig] = None):
        """Initialize the attention mechanism."""
        self.config = config
        self.cache_config = cache_config or CacheConfig()
        self.cache = CacheManager(self.cache_config)
        self.performance_monitor = PerformanceMonitor()
        
        # Attention state
        self.attention_weights: List[np.ndarray] = []
        self.attention_maps: List[np.ndarray] = []
        self.positional_encoding: Optional[np.ndarray] = None
        
        # Performance tracking
        self.computation_times: List[float] = []
        self.attention_scores: List[float] = []
        
        # Multi-head attention
        self.head_weights: List[np.ndarray] = []
        self.head_biases: List[np.ndarray] = []
        
        self._initialized = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize the attention mechanism."""
        try:
            self.cache.initialize()
            self.performance_monitor.start_monitoring()
            
            # Initialize attention weights
            self._initialize_attention_weights()
            
            # Initialize positional encoding if needed
            if self.config.use_positional_encoding:
                self._initialize_positional_encoding()
            
            self._initialized = True
            self.logger.info(f"Attention mechanism initialized with {self.config.attention_type.value}")
        except Exception as e:
            self.logger.error(f"Failed to initialize attention mechanism: {e}")
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get current component state."""
        return {
            'name': 'AttentionMechanism',
            'status': 'running' if self._initialized else 'stopped',
            'last_updated': datetime.now(),
            'metadata': {
                'attention_type': self.config.attention_type.value,
                'num_heads': self.config.num_heads,
                'attention_dim': self.config.attention_dim,
                'attention_weights_count': len(self.attention_weights),
                'attention_maps_count': len(self.attention_maps),
                'average_computation_time': np.mean(self.computation_times) if self.computation_times else 0.0
            }
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.performance_monitor.stop_monitoring()
            self.cache.clear()
            self._initialized = False
            self.logger.info("Attention mechanism cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self._initialized and self.cache.is_healthy()
    
    def compute_attention(self, query: np.ndarray, key: Optional[np.ndarray] = None, 
                         value: Optional[np.ndarray] = None) -> AttentionResult:
        """Compute attention for given inputs."""
        try:
            start_time = datetime.now()
            
            # Use query as key and value if not provided (self-attention)
            if key is None:
                key = query
            if value is None:
                value = query
            
            # Compute attention based on type
            if self.config.attention_type == AttentionType.SPATIAL:
                attention_weights, attended_features = self._spatial_attention(query, key, value)
            elif self.config.attention_type == AttentionType.TEMPORAL:
                attention_weights, attended_features = self._temporal_attention(query, key, value)
            elif self.config.attention_type == AttentionType.CHANNEL:
                attention_weights, attended_features = self._channel_attention(query, key, value)
            elif self.config.attention_type == AttentionType.MULTI_HEAD:
                attention_weights, attended_features = self._multi_head_attention(query, key, value)
            elif self.config.attention_type == AttentionType.SELF_ATTENTION:
                attention_weights, attended_features = self._self_attention(query)
            elif self.config.attention_type == AttentionType.CROSS_ATTENTION:
                attention_weights, attended_features = self._cross_attention(query, key, value)
            else:
                raise ValueError(f"Unsupported attention type: {self.config.attention_type}")
            
            # Create attention map
            attention_map = self._create_attention_map(attention_weights, query.shape)
            
            # Calculate confidence
            confidence = self._calculate_attention_confidence(attention_weights)
            
            # Create result
            result = AttentionResult(
                attention_weights=attention_weights,
                attended_features=attended_features,
                attention_map=attention_map,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            # Update performance metrics
            computation_time = (datetime.now() - start_time).total_seconds()
            self.computation_times.append(computation_time)
            self.attention_scores.append(confidence)
            
            # Store in history
            self.attention_weights.append(attention_weights)
            self.attention_maps.append(attention_map)
            
            # Cache result if enabled
            if self.config.enable_caching:
                cache_key = f"attention_{datetime.now().timestamp()}"
                self.cache.set(cache_key, result, ttl=self.config.cache_ttl)
            
            self.logger.debug(f"Computed {self.config.attention_type.value} attention in {computation_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error computing attention: {e}")
            raise
    
    def get_attention_statistics(self) -> Dict[str, Any]:
        """Get attention statistics and performance metrics."""
        try:
            if not self.computation_times:
                return {'error': 'No attention computations performed yet'}
            
            # Calculate statistics
            avg_computation_time = np.mean(self.computation_times)
            avg_attention_score = np.mean(self.attention_scores)
            
            # Attention weight statistics
            if self.attention_weights:
                all_weights = np.concatenate([w.flatten() for w in self.attention_weights])
                weight_stats = {
                    'mean': np.mean(all_weights),
                    'std': np.std(all_weights),
                    'min': np.min(all_weights),
                    'max': np.max(all_weights)
                }
            else:
                weight_stats = {}
            
            # Attention map statistics
            if self.attention_maps:
                all_maps = np.concatenate([m.flatten() for m in self.attention_maps])
                map_stats = {
                    'mean': np.mean(all_maps),
                    'std': np.std(all_maps),
                    'min': np.min(all_maps),
                    'max': np.max(all_maps)
                }
            else:
                map_stats = {}
            
            return {
                'total_computations': len(self.computation_times),
                'average_computation_time': avg_computation_time,
                'average_attention_score': avg_attention_score,
                'attention_type': self.config.attention_type.value,
                'num_heads': self.config.num_heads,
                'attention_dim': self.config.attention_dim,
                'weight_statistics': weight_stats,
                'map_statistics': map_stats,
                'positional_encoding_enabled': self.config.use_positional_encoding
            }
            
        except Exception as e:
            self.logger.error(f"Error getting attention statistics: {e}")
            return {'error': str(e)}
    
    def _initialize_attention_weights(self) -> None:
        """Initialize attention weights."""
        try:
            # Initialize weights for different attention types
            if self.config.attention_type == AttentionType.MULTI_HEAD:
                # Initialize weights for each head
                for head in range(self.config.num_heads):
                    # Query weights
                    q_weights = np.random.normal(0, 0.1, (self.config.attention_dim, self.config.attention_dim))
                    # Key weights
                    k_weights = np.random.normal(0, 0.1, (self.config.attention_dim, self.config.attention_dim))
                    # Value weights
                    v_weights = np.random.normal(0, 0.1, (self.config.attention_dim, self.config.attention_dim))
                    
                    self.head_weights.append({
                        'query': q_weights,
                        'key': k_weights,
                        'value': v_weights
                    })
                    
                    # Initialize biases
                    self.head_biases.append({
                        'query': np.zeros(self.config.attention_dim),
                        'key': np.zeros(self.config.attention_dim),
                        'value': np.zeros(self.config.attention_dim)
                    })
            else:
                # Initialize single attention weights
                self.head_weights = [{
                    'query': np.random.normal(0, 0.1, (self.config.attention_dim, self.config.attention_dim)),
                    'key': np.random.normal(0, 0.1, (self.config.attention_dim, self.config.attention_dim)),
                    'value': np.random.normal(0, 0.1, (self.config.attention_dim, self.config.attention_dim))
                }]
                
                self.head_biases = [{
                    'query': np.zeros(self.config.attention_dim),
                    'key': np.zeros(self.config.attention_dim),
                    'value': np.zeros(self.config.attention_dim)
                }]
            
            self.logger.info(f"Initialized attention weights for {self.config.num_heads} heads")
            
        except Exception as e:
            self.logger.error(f"Error initializing attention weights: {e}")
            raise
    
    def _initialize_positional_encoding(self) -> None:
        """Initialize positional encoding."""
        try:
            # Create positional encoding matrix
            max_length = 1000  # Maximum sequence length
            d_model = self.config.attention_dim
            
            pos_encoding = np.zeros((max_length, d_model))
            
            for pos in range(max_length):
                for i in range(0, d_model, 2):
                    pos_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
                    if i + 1 < d_model:
                        pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
            
            self.positional_encoding = pos_encoding
            self.logger.info("Positional encoding initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing positional encoding: {e}")
            raise
    
    def _spatial_attention(self, query: np.ndarray, key: np.ndarray, 
                          value: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute spatial attention."""
        try:
            # Reshape for spatial attention
            if len(query.shape) == 4:  # (batch, height, width, channels)
                batch_size, height, width, channels = query.shape
                query_flat = query.reshape(batch_size, height * width, channels)
                key_flat = key.reshape(batch_size, height * width, channels)
                value_flat = value.reshape(batch_size, height * width, channels)
            else:
                query_flat = query
                key_flat = key
                value_flat = value
            
            # Compute attention scores
            scores = np.matmul(query_flat, key_flat.transpose(-1, -2))
            scores = scores / np.sqrt(self.config.attention_dim)
            
            # Apply temperature
            scores = scores / self.config.temperature
            
            # Apply softmax
            attention_weights = self._softmax(scores, axis=-1)
            
            # Apply attention to values
            attended_features = np.matmul(attention_weights, value_flat)
            
            return attention_weights, attended_features
            
        except Exception as e:
            self.logger.error(f"Error computing spatial attention: {e}")
            raise
    
    def _temporal_attention(self, query: np.ndarray, key: np.ndarray, 
                           value: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute temporal attention."""
        try:
            # For temporal attention, we assume the first dimension is time
            # Compute attention scores
            scores = np.matmul(query, key.transpose(-1, -2))
            scores = scores / np.sqrt(self.config.attention_dim)
            
            # Apply temperature
            scores = scores / self.config.temperature
            
            # Apply softmax
            attention_weights = self._softmax(scores, axis=-1)
            
            # Apply attention to values
            attended_features = np.matmul(attention_weights, value)
            
            return attention_weights, attended_features
            
        except Exception as e:
            self.logger.error(f"Error computing temporal attention: {e}")
            raise
    
    def _channel_attention(self, query: np.ndarray, key: np.ndarray, 
                          value: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute channel attention."""
        try:
            # For channel attention, we focus on the channel dimension
            # Compute attention scores
            scores = np.matmul(query.transpose(-1, -2), key)
            scores = scores / np.sqrt(self.config.attention_dim)
            
            # Apply temperature
            scores = scores / self.config.temperature
            
            # Apply softmax
            attention_weights = self._softmax(scores, axis=-1)
            
            # Apply attention to values
            attended_features = np.matmul(value, attention_weights)
            
            return attention_weights, attended_features
            
        except Exception as e:
            self.logger.error(f"Error computing channel attention: {e}")
            raise
    
    def _multi_head_attention(self, query: np.ndarray, key: np.ndarray, 
                             value: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute multi-head attention."""
        try:
            # Split into heads
            head_size = self.config.attention_dim // self.config.num_heads
            heads = []
            
            for head in range(self.config.num_heads):
                # Get head weights
                head_weights = self.head_weights[head]
                head_biases = self.head_biases[head]
                
                # Apply linear transformations
                q_head = np.matmul(query, head_weights['query']) + head_biases['query']
                k_head = np.matmul(key, head_weights['key']) + head_biases['key']
                v_head = np.matmul(value, head_weights['value']) + head_biases['value']
                
                # Compute attention for this head
                scores = np.matmul(q_head, k_head.transpose(-1, -2))
                scores = scores / np.sqrt(head_size)
                scores = scores / self.config.temperature
                
                attention_weights = self._softmax(scores, axis=-1)
                attended_features = np.matmul(attention_weights, v_head)
                
                heads.append(attended_features)
            
            # Concatenate heads
            attended_features = np.concatenate(heads, axis=-1)
            
            # Average attention weights across heads
            attention_weights = np.mean([self._softmax(
                np.matmul(
                    np.matmul(query, self.head_weights[i]['query']),
                    np.matmul(key, self.head_weights[i]['key']).transpose(-1, -2)
                ) / np.sqrt(head_size) / self.config.temperature, axis=-1
            ) for i in range(self.config.num_heads)], axis=0)
            
            return attention_weights, attended_features
            
        except Exception as e:
            self.logger.error(f"Error computing multi-head attention: {e}")
            raise
    
    def _self_attention(self, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute self-attention."""
        return self._spatial_attention(query, query, query)
    
    def _cross_attention(self, query: np.ndarray, key: np.ndarray, 
                        value: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cross-attention."""
        return self._spatial_attention(query, key, value)
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax function."""
        try:
            # Subtract max for numerical stability
            x_max = np.max(x, axis=axis, keepdims=True)
            x_shifted = x - x_max
            
            # Compute exponentials
            exp_x = np.exp(x_shifted)
            
            # Compute softmax
            softmax_x = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
            
            return softmax_x
            
        except Exception as e:
            self.logger.error(f"Error computing softmax: {e}")
            return x
    
    def _create_attention_map(self, attention_weights: np.ndarray, 
                            input_shape: Tuple[int, ...]) -> np.ndarray:
        """Create attention map from attention weights."""
        try:
            # Reshape attention weights to match input spatial dimensions
            if len(input_shape) >= 3:  # Has spatial dimensions
                height, width = input_shape[1], input_shape[2]
                attention_map = attention_weights.reshape(-1, height, width)
            else:
                attention_map = attention_weights
            
            # Normalize to [0, 1]
            attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map) + 1e-8)
            
            return attention_map
            
        except Exception as e:
            self.logger.error(f"Error creating attention map: {e}")
            return attention_weights
    
    def _calculate_attention_confidence(self, attention_weights: np.ndarray) -> float:
        """Calculate confidence score for attention weights."""
        try:
            # Calculate entropy of attention distribution
            # Lower entropy means more focused attention (higher confidence)
            attention_flat = attention_weights.flatten()
            attention_flat = attention_flat / (np.sum(attention_flat) + 1e-8)  # Normalize
            
            # Calculate entropy
            entropy = -np.sum(attention_flat * np.log(attention_flat + 1e-8))
            
            # Convert entropy to confidence (0-1 scale)
            max_entropy = np.log(len(attention_flat))
            confidence = 1.0 - (entropy / max_entropy)
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating attention confidence: {e}")
            return 0.5
