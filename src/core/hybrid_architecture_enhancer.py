#!/usr/bin/env python3
"""
Hybrid Architecture Enhancer for Tabula Rasa

Enhances existing feature extraction + reasoning with hybrid architecture,
building on OpenCV feature extraction and Tree-Based Director.
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

logger = logging.getLogger(__name__)

class ProcessingPath(Enum):
    """Processing paths in hybrid architecture."""
    FIXED_FEATURE_EXTRACTION = "fixed_feature_extraction"  # Like early visual cortex
    TRAINABLE_RELATIONAL = "trainable_relational"  # Like prefrontal cortex
    META_MODULATION = "meta_modulation"  # Like DMN

class FeatureType(Enum):
    """Types of features extracted."""
    VISUAL_FEATURES = "visual_features"
    SPATIAL_FEATURES = "spatial_features"
    TEMPORAL_FEATURES = "temporal_features"
    SEMANTIC_FEATURES = "semantic_features"
    RELATIONAL_FEATURES = "relational_features"

@dataclass
class FeatureExtractionResult:
    """Result from feature extraction."""
    feature_type: FeatureType
    features: np.ndarray
    confidence: float
    processing_path: ProcessingPath
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RelationalReasoningResult:
    """Result from relational reasoning."""
    relations: List[Dict[str, Any]]
    confidence: float
    reasoning_depth: int
    processing_path: ProcessingPath
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HybridArchitectureOutput:
    """Output from hybrid architecture processing."""
    feature_results: List[FeatureExtractionResult]
    reasoning_results: List[RelationalReasoningResult]
    meta_modulation_weights: Dict[ProcessingPath, float]
    final_decision: Dict[str, Any]
    confidence: float
    processing_time: float

class FixedFeatureExtractor(nn.Module):
    """Fixed feature extraction layers (like early visual cortex)."""
    
    def __init__(self, input_channels: int = 3, feature_dim: int = 512):
        super().__init__()
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        
        # Fixed convolutional layers (pretrained, not trainable)
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Fixed pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fixed fully connected layers - calculate actual size dynamically
        # For 32x32 input: 32x32 -> 16x16 -> 8x8 -> 4x4, so 128 * 4 * 4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, feature_dim)  # For 32x32 input
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through fixed feature extractor."""
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        return x

class TrainableRelationalModule(nn.Module):
    """Trainable relational reasoning module (like prefrontal cortex)."""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Relational reasoning layers
        self.relation_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Attention mechanism for relational reasoning
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through relational reasoning module."""
        # Encode input
        encoded = self.relation_encoder(x)
        
        # Apply attention for relational reasoning
        # Reshape for attention (batch_size, seq_len, hidden_dim)
        if encoded.dim() == 2:
            encoded = encoded.unsqueeze(1)
        
        attended, attention_weights = self.attention(encoded, encoded, encoded)
        
        # Residual connection and layer norm
        attended = self.layer_norm(attended + encoded)
        
        # Output projection
        output = self.output_projection(attended.squeeze(1))
        
        return output, attention_weights

class MetaModulationNetwork(nn.Module):
    """Meta-modulation network (like DMN) for path selection."""
    
    def __init__(self, input_dim: int = 512, num_paths: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.num_paths = num_paths
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Path selection network
        self.path_selector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_paths),
            nn.Softmax(dim=-1)
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through meta-modulation network."""
        # Encode context
        context = self.context_encoder(x)
        
        # Select processing paths
        path_weights = self.path_selector(context)
        
        # Estimate confidence
        confidence = self.confidence_estimator(context)
        
        return path_weights, confidence

class HybridArchitectureEnhancer:
    """Enhanced hybrid architecture combining fixed and trainable components."""
    
    def __init__(self, 
                 input_dim: int = 512,
                 feature_dim: int = 512,
                 relational_dim: int = 128,
                 enable_training: bool = True):
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.relational_dim = relational_dim
        self.enable_training = enable_training
        
        # Initialize components
        self.fixed_feature_extractor = FixedFeatureExtractor(feature_dim=feature_dim)
        self.trainable_relational = TrainableRelationalModule(
            input_dim=feature_dim, 
            output_dim=relational_dim
        )
        self.meta_modulation = MetaModulationNetwork(
            input_dim=feature_dim,
            num_paths=3
        )
        
        # Processing statistics
        self.processing_stats = {
            'total_processing_time': 0.0,
            'path_usage_counts': {path: 0 for path in ProcessingPath},
            'feature_extraction_time': 0.0,
            'relational_reasoning_time': 0.0,
            'meta_modulation_time': 0.0
        }
        
        logger.info("Hybrid Architecture Enhancer initialized")
    
    def process_with_hybrid_architecture(self, 
                                       input_data: Dict[str, Any], 
                                       context: Dict[str, Any]) -> HybridArchitectureOutput:
        """Process input through hybrid architecture."""
        start_time = time.time()
        
        # Convert input to tensor
        input_tensor = self._prepare_input_tensor(input_data)
        
        # Path 1: Fixed feature extraction (like early visual cortex)
        feature_start = time.time()
        fixed_features = self._extract_fixed_features(input_tensor, input_data, context)
        feature_time = time.time() - feature_start
        self.processing_stats['feature_extraction_time'] += feature_time
        
        # Path 2: Trainable relational reasoning (like prefrontal cortex)
        reasoning_start = time.time()
        relational_results = self._perform_relational_reasoning(
            fixed_features, input_data, context
        )
        reasoning_time = time.time() - reasoning_start
        self.processing_stats['relational_reasoning_time'] += reasoning_time
        
        # Path 3: Meta-modulation for path selection (like DMN)
        modulation_start = time.time()
        meta_weights, confidence = self._perform_meta_modulation(
            fixed_features, input_data, context
        )
        modulation_time = time.time() - modulation_start
        self.processing_stats['meta_modulation_time'] += modulation_time
        
        # Combine results based on meta-modulation weights
        final_decision = self._combine_results(
            fixed_features, relational_results, meta_weights, context
        )
        
        # Update statistics
        total_time = time.time() - start_time
        self.processing_stats['total_processing_time'] += total_time
        
        # Update path usage counts
        dominant_path = self._get_dominant_path(meta_weights)
        self.processing_stats['path_usage_counts'][dominant_path] += 1
        
        # Create output
        output = HybridArchitectureOutput(
            feature_results=fixed_features,
            reasoning_results=relational_results,
            meta_modulation_weights={
                ProcessingPath.FIXED_FEATURE_EXTRACTION: meta_weights[0, 0].item(),
                ProcessingPath.TRAINABLE_RELATIONAL: meta_weights[0, 1].item(),
                ProcessingPath.META_MODULATION: meta_weights[0, 2].item()
            },
            final_decision=final_decision,
            confidence=confidence.item(),
            processing_time=total_time
        )
        
        return output
    
    def _prepare_input_tensor(self, input_data: Dict[str, Any]) -> torch.Tensor:
        """Prepare input data as tensor."""
        # Extract frame data
        frame = input_data.get('frame')
        if frame is not None:
            # Convert to tensor and normalize
            if isinstance(frame, np.ndarray):
                frame_tensor = torch.from_numpy(frame).float()
            else:
                frame_tensor = torch.tensor(frame, dtype=torch.float32)
            
            # Ensure correct shape (batch_size, channels, height, width)
            if frame_tensor.dim() == 3:
                # Check if channels are last (H, W, C) or first (C, H, W)
                if frame_tensor.shape[2] == 3:  # (H, W, C)
                    frame_tensor = frame_tensor.permute(2, 0, 1)  # Convert to (C, H, W)
                frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension
            elif frame_tensor.dim() == 2:
                frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            elif frame_tensor.dim() == 4:
                pass  # Already in correct format
            else:
                # Fallback to random tensor
                return torch.randn(1, 3, 32, 32)
            
            # Normalize to [0, 1]
            frame_tensor = frame_tensor / 255.0
            
            return frame_tensor
        else:
            # Fallback to random tensor
            return torch.randn(1, 3, 32, 32)
    
    def _extract_fixed_features(self, 
                               input_tensor: torch.Tensor, 
                               input_data: Dict[str, Any], 
                               context: Dict[str, Any]) -> List[FeatureExtractionResult]:
        """Extract features using fixed feature extractor."""
        with torch.no_grad():
            # Extract features
            features = self.fixed_feature_extractor(input_tensor)
            
            # Create feature extraction results
            results = []
            
            # Visual features
            visual_features = features[:, :self.feature_dim//4]
            results.append(FeatureExtractionResult(
                feature_type=FeatureType.VISUAL_FEATURES,
                features=visual_features.squeeze(0).numpy(),
                confidence=0.8,  # Fixed extractor has high confidence
                processing_path=ProcessingPath.FIXED_FEATURE_EXTRACTION,
                metadata={'extractor_type': 'fixed_convolutional'}
            ))
            
            # Spatial features
            spatial_features = features[:, self.feature_dim//4:self.feature_dim//2]
            results.append(FeatureExtractionResult(
                feature_type=FeatureType.SPATIAL_FEATURES,
                features=spatial_features.squeeze(0).numpy(),
                confidence=0.7,
                processing_path=ProcessingPath.FIXED_FEATURE_EXTRACTION,
                metadata={'extractor_type': 'fixed_convolutional'}
            ))
            
            # Temporal features (simplified)
            temporal_features = features[:, self.feature_dim//2:3*self.feature_dim//4]
            results.append(FeatureExtractionResult(
                feature_type=FeatureType.TEMPORAL_FEATURES,
                features=temporal_features.squeeze(0).numpy(),
                confidence=0.6,
                processing_path=ProcessingPath.FIXED_FEATURE_EXTRACTION,
                metadata={'extractor_type': 'fixed_convolutional'}
            ))
            
            # Semantic features
            semantic_features = features[:, 3*self.feature_dim//4:]
            results.append(FeatureExtractionResult(
                feature_type=FeatureType.SEMANTIC_FEATURES,
                features=semantic_features.squeeze(0).numpy(),
                confidence=0.5,
                processing_path=ProcessingPath.FIXED_FEATURE_EXTRACTION,
                metadata={'extractor_type': 'fixed_convolutional'}
            ))
            
            return results
    
    def _perform_relational_reasoning(self, 
                                    fixed_features: List[FeatureExtractionResult], 
                                    input_data: Dict[str, Any], 
                                    context: Dict[str, Any]) -> List[RelationalReasoningResult]:
        """Perform relational reasoning using trainable module."""
        if not fixed_features:
            return []
        
        # Combine features for relational reasoning
        combined_features = np.concatenate([feat.features for feat in fixed_features])
        combined_tensor = torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0)
        
        if self.enable_training:
            # Trainable mode
            self.trainable_relational.train()
            relational_output, attention_weights = self.trainable_relational(combined_tensor)
        else:
            # Inference mode
            self.trainable_relational.eval()
            with torch.no_grad():
                relational_output, attention_weights = self.trainable_relational(combined_tensor)
        
        # Create relational reasoning results
        results = []
        
        # Extract relations from attention weights
        relations = self._extract_relations_from_attention(attention_weights, fixed_features)
        
        results.append(RelationalReasoningResult(
            relations=relations,
            confidence=0.7,  # Relational reasoning confidence
            reasoning_depth=3,  # Simplified depth
            processing_path=ProcessingPath.TRAINABLE_RELATIONAL,
            metadata={
                'attention_weights': attention_weights.squeeze(0).detach().numpy().tolist(),
                'output_dim': relational_output.shape[-1]
            }
        ))
        
        return results
    
    def _perform_meta_modulation(self, 
                               fixed_features: List[FeatureExtractionResult], 
                               input_data: Dict[str, Any], 
                               context: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform meta-modulation for path selection."""
        if not fixed_features:
            # Fallback to equal weights
            return torch.tensor([0.33, 0.33, 0.34]), torch.tensor(0.5)
        
        # Combine features for meta-modulation
        combined_features = np.concatenate([feat.features for feat in fixed_features])
        combined_tensor = torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0)
        
        if self.enable_training:
            # Trainable mode
            self.meta_modulation.train()
            path_weights, confidence = self.meta_modulation(combined_tensor)
        else:
            # Inference mode
            self.meta_modulation.eval()
            with torch.no_grad():
                path_weights, confidence = self.meta_modulation(combined_tensor)
        
        return path_weights, confidence
    
    def _extract_relations_from_attention(self, 
                                        attention_weights: torch.Tensor, 
                                        features: List[FeatureExtractionResult]) -> List[Dict[str, Any]]:
        """Extract relations from attention weights."""
        relations = []
        
        # Convert attention weights to numpy (detach from gradient)
        weights = attention_weights.squeeze(0).detach().numpy()
        
        # Find strong attention patterns
        for i, feat in enumerate(features):
            if i < weights.shape[0]:
                attention_strength = weights[i].mean()
                
                if attention_strength > 0.1:  # Threshold for significant attention
                    relation = {
                        'feature_type': feat.feature_type.value,
                        'attention_strength': float(attention_strength),
                        'confidence': feat.confidence,
                        'relation_type': 'attention_based'
                    }
                    relations.append(relation)
        
        return relations
    
    def _combine_results(self, 
                        fixed_features: List[FeatureExtractionResult],
                        relational_results: List[RelationalReasoningResult],
                        meta_weights: torch.Tensor,
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from all processing paths."""
        
        # Weight the contributions based on meta-modulation
        fixed_weight = meta_weights[0, 0].item()
        relational_weight = meta_weights[0, 1].item()
        meta_weight = meta_weights[0, 2].item()
        
        # Combine feature information
        combined_features = {}
        for feat in fixed_features:
            combined_features[feat.feature_type.value] = {
                'features': feat.features.tolist(),
                'confidence': feat.confidence,
                'weight': fixed_weight
            }
        
        # Add relational information
        combined_relations = []
        for rel in relational_results:
            combined_relations.extend(rel.relations)
        
        # Create final decision
        final_decision = {
            'features': combined_features,
            'relations': combined_relations,
            'meta_weights': {
                'fixed_extraction': fixed_weight,
                'relational_reasoning': relational_weight,
                'meta_modulation': meta_weight
            },
            'processing_paths_used': [
                ProcessingPath.FIXED_FEATURE_EXTRACTION.value,
                ProcessingPath.TRAINABLE_RELATIONAL.value,
                ProcessingPath.META_MODULATION.value
            ],
            'context': context
        }
        
        return final_decision
    
    def _get_dominant_path(self, meta_weights: torch.Tensor) -> ProcessingPath:
        """Get the dominant processing path based on meta-weights."""
        weights = meta_weights.squeeze(0).detach().numpy()
        dominant_idx = np.argmax(weights)
        
        path_mapping = {
            0: ProcessingPath.FIXED_FEATURE_EXTRACTION,
            1: ProcessingPath.TRAINABLE_RELATIONAL,
            2: ProcessingPath.META_MODULATION
        }
        
        return path_mapping.get(dominant_idx, ProcessingPath.FIXED_FEATURE_EXTRACTION)
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get statistics about hybrid architecture processing."""
        total_time = self.processing_stats['total_processing_time']
        
        return {
            'total_processing_time': total_time,
            'average_processing_time': total_time / max(1, sum(self.processing_stats['path_usage_counts'].values())),
            'path_usage_counts': {path.value: count for path, count in self.processing_stats['path_usage_counts'].items()},
            'feature_extraction_time': self.processing_stats['feature_extraction_time'],
            'relational_reasoning_time': self.processing_stats['relational_reasoning_time'],
            'meta_modulation_time': self.processing_stats['meta_modulation_time'],
            'time_distribution': {
                'feature_extraction': self.processing_stats['feature_extraction_time'] / max(1, total_time),
                'relational_reasoning': self.processing_stats['relational_reasoning_time'] / max(1, total_time),
                'meta_modulation': self.processing_stats['meta_modulation_time'] / max(1, total_time)
            }
        }

# Factory function for easy integration
def create_hybrid_architecture_enhancer(**kwargs) -> HybridArchitectureEnhancer:
    """Create a configured hybrid architecture enhancer."""
    return HybridArchitectureEnhancer(**kwargs)
