#!/usr/bin/env python3
"""
Global Workspace System for Tabula Rasa

Implements global workspace-like dynamics with attention mechanisms,
building on existing Director system and message bus architecture.
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

class AttentionType(Enum):
    """Types of attention mechanisms."""
    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"
    TEMPORAL_ATTENTION = "temporal_attention"
    SPATIAL_ATTENTION = "spatial_attention"

class ModuleType(Enum):
    """Types of specialized modules."""
    VISION = "vision"
    REASONING = "reasoning"
    MEMORY = "memory"
    ACTION = "action"
    EMOTION = "emotion"
    PLANNING = "planning"

@dataclass
class ModuleOutput:
    """Output from a specialized module."""
    module_type: ModuleType
    content: Dict[str, Any]
    confidence: float
    relevance_score: float
    timestamp: float
    attention_weight: float = 0.0

@dataclass
class GlobalWorkspaceState:
    """Current state of the global workspace."""
    active_modules: List[ModuleType]
    attention_weights: Dict[ModuleType, float]
    broadcast_content: Dict[str, Any]
    coherence_score: float
    timestamp: float

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for global workspace."""
    
    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of multi-head attention."""
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Output projection
        output = self.w_o(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        
        return output, attention_weights.mean(dim=1)  # Average across heads

class TransformerAttention(nn.Module):
    """Transformer-based attention for global workspace."""
    
    def __init__(self, d_model: int = 512, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)
        ])
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through transformer attention."""
        attention_weights = None
        
        for attention_layer in self.attention_layers:
            x, attn_weights = attention_layer(x, x, x, mask)
            if attention_weights is None:
                attention_weights = attn_weights
            else:
                attention_weights = attention_weights + attn_weights
        
        # Feed forward
        x = x + self.feed_forward(x)
        
        # Average attention weights across layers
        if attention_weights is not None:
            attention_weights = attention_weights / len(self.attention_layers)
        
        return x, attention_weights

class SpecializedModule:
    """Base class for specialized modules in the global workspace."""
    
    def __init__(self, module_type: ModuleType, output_dim: int = 512):
        self.module_type = module_type
        self.output_dim = output_dim
        self.last_output = None
        self.confidence = 0.0
        self.relevance_score = 0.0
        
    def process(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> ModuleOutput:
        """Process input data and return module output."""
        # This is a base implementation - subclasses should override
        content = self._extract_content(input_data, context)
        confidence = self._calculate_confidence(content, context)
        relevance_score = self._calculate_relevance(content, context)
        
        output = ModuleOutput(
            module_type=self.module_type,
            content=content,
            confidence=confidence,
            relevance_score=relevance_score,
            timestamp=time.time()
        )
        
        self.last_output = output
        return output
    
    def _extract_content(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from input data."""
        return {"raw_data": input_data, "context": context}
    
    def _calculate_confidence(self, content: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate confidence in the output."""
        return 0.5  # Base confidence
    
    def _calculate_relevance(self, content: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate relevance score for the output."""
        return 0.5  # Base relevance

class VisionModule(SpecializedModule):
    """Vision processing module."""
    
    def __init__(self, output_dim: int = 512):
        super().__init__(ModuleType.VISION, output_dim)
    
    def _extract_content(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract visual features and patterns."""
        frame = input_data.get('frame')
        if frame is not None:
            # Extract visual features (simplified)
            features = {
                'shape_features': self._extract_shape_features(frame),
                'color_features': self._extract_color_features(frame),
                'spatial_features': self._extract_spatial_features(frame),
                'pattern_features': self._extract_pattern_features(frame)
            }
            return features
        return {"error": "No frame data available"}
    
    def _extract_shape_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract shape-related features."""
        # Simplified shape feature extraction
        return {"shapes_detected": 0, "edges": 0, "corners": 0}
    
    def _extract_color_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract color-related features."""
        # Simplified color feature extraction
        return {"dominant_colors": [], "color_distribution": {}}
    
    def _extract_spatial_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract spatial relationship features."""
        # Simplified spatial feature extraction
        return {"spatial_relationships": [], "object_positions": []}
    
    def _extract_pattern_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract pattern-related features."""
        # Simplified pattern feature extraction
        return {"patterns_detected": [], "symmetry": 0.0}

class ReasoningModule(SpecializedModule):
    """Reasoning and logic processing module."""
    
    def __init__(self, output_dim: int = 512):
        super().__init__(ModuleType.REASONING, output_dim)
    
    def _extract_content(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract reasoning patterns and logical relationships."""
        return {
            'logical_rules': self._extract_logical_rules(input_data, context),
            'causal_relationships': self._extract_causal_relationships(input_data, context),
            'inference_chains': self._extract_inference_chains(input_data, context),
            'constraints': self._extract_constraints(input_data, context)
        }
    
    def _extract_logical_rules(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Extract logical rules from the data."""
        return []  # Simplified implementation
    
    def _extract_causal_relationships(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract causal relationships."""
        return []  # Simplified implementation
    
    def _extract_inference_chains(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract inference chains."""
        return []  # Simplified implementation
    
    def _extract_constraints(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Extract constraints from the data."""
        return []  # Simplified implementation

class MemoryModule(SpecializedModule):
    """Memory and knowledge processing module."""
    
    def __init__(self, output_dim: int = 512):
        super().__init__(ModuleType.MEMORY, output_dim)
    
    def _extract_content(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant memories and knowledge."""
        return {
            'relevant_memories': self._retrieve_relevant_memories(input_data, context),
            'knowledge_patterns': self._extract_knowledge_patterns(input_data, context),
            'associations': self._find_associations(input_data, context),
            'contextual_information': self._extract_contextual_info(input_data, context)
        }
    
    def _retrieve_relevant_memories(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve relevant memories."""
        return []  # Simplified implementation
    
    def _extract_knowledge_patterns(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract knowledge patterns."""
        return []  # Simplified implementation
    
    def _find_associations(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find associations with current data."""
        return []  # Simplified implementation
    
    def _extract_contextual_info(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract contextual information."""
        return context

class GlobalWorkspaceSystem:
    """Global workspace system with attention mechanisms."""
    
    def __init__(self, 
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 coherence_threshold: float = 0.6):
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.coherence_threshold = coherence_threshold
        
        # Initialize attention mechanism
        self.attention_mechanism = TransformerAttention(d_model, num_heads, num_layers)
        
        # Initialize specialized modules
        self.specialized_modules = {
            ModuleType.VISION: VisionModule(d_model),
            ModuleType.REASONING: ReasoningModule(d_model),
            ModuleType.MEMORY: MemoryModule(d_model)
        }
        
        # Global workspace state
        self.current_state = GlobalWorkspaceState(
            active_modules=[],
            attention_weights={},
            broadcast_content={},
            coherence_score=0.0,
            timestamp=time.time()
        )
        
        # History for temporal attention
        self.module_output_history = []
        self.attention_history = []
        
        logger.info(f"Global Workspace System initialized with {len(self.specialized_modules)} modules")
    
    def process_global_workspace(self, 
                                input_data: Dict[str, Any], 
                                context: Dict[str, Any]) -> GlobalWorkspaceState:
        """Process input through global workspace with attention."""
        
        # Get outputs from specialized modules
        module_outputs = {}
        for module_type, module in self.specialized_modules.items():
            try:
                output = module.process(input_data, context)
                module_outputs[module_type] = output
            except Exception as e:
                logger.warning(f"Module {module_type.value} failed: {e}")
                continue
        
        if not module_outputs:
            logger.warning("No module outputs available")
            return self.current_state
        
        # Convert module outputs to tensor representation
        module_tensors = self._convert_to_tensors(module_outputs)
        
        # Apply attention mechanism
        attended_outputs, attention_weights = self.attention_mechanism(module_tensors)
        
        # Calculate attention weights for each module
        module_attention_weights = self._calculate_module_attention_weights(
            attention_weights, list(module_outputs.keys())
        )
        
        # Determine which modules are active (high attention)
        active_modules = [
            module_type for module_type, weight in module_attention_weights.items()
            if weight > self.coherence_threshold
        ]
        
        # Create broadcast content from attended outputs
        broadcast_content = self._create_broadcast_content(
            attended_outputs, module_outputs, active_modules
        )
        
        # Calculate coherence score
        coherence_score = self._calculate_coherence_score(
            module_outputs, module_attention_weights
        )
        
        # Update global workspace state
        self.current_state = GlobalWorkspaceState(
            active_modules=active_modules,
            attention_weights=module_attention_weights,
            broadcast_content=broadcast_content,
            coherence_score=coherence_score,
            timestamp=time.time()
        )
        
        # Update history
        self.module_output_history.append(module_outputs)
        self.attention_history.append(attention_weights)
        
        # Keep history manageable
        if len(self.module_output_history) > 100:
            self.module_output_history = self.module_output_history[-50:]
            self.attention_history = self.attention_history[-50:]
        
        return self.current_state
    
    def _convert_to_tensors(self, module_outputs: Dict[ModuleType, ModuleOutput]) -> torch.Tensor:
        """Convert module outputs to tensor representation."""
        # Create a simple tensor representation
        # In practice, this would be more sophisticated
        tensor_list = []
        
        for module_type, output in module_outputs.items():
            # Create a simple feature vector from module output
            features = []
            
            # Add confidence and relevance
            features.extend([output.confidence, output.relevance_score])
            
            # Add content features (simplified)
            content = output.content
            if isinstance(content, dict):
                # Add some basic features from content
                features.extend([len(str(content)), hash(str(content)) % 1000 / 1000.0])
            else:
                features.extend([0.0, 0.0])
            
            # Pad to d_model dimensions
            while len(features) < self.d_model:
                features.append(0.0)
            
            features = features[:self.d_model]
            tensor_list.append(features)
        
        # Convert to tensor
        tensor = torch.tensor(tensor_list, dtype=torch.float32)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def _calculate_module_attention_weights(self, 
                                          attention_weights: torch.Tensor, 
                                          module_types: List[ModuleType]) -> Dict[ModuleType, float]:
        """Calculate attention weights for each module."""
        weights = {}
        
        if attention_weights is not None and len(module_types) > 0:
            # Average attention weights across the sequence
            avg_weights = attention_weights.mean(dim=1).squeeze(0)
            
            # Normalize weights
            if len(avg_weights) >= len(module_types):
                normalized_weights = F.softmax(avg_weights[:len(module_types)], dim=0)
                
                for i, module_type in enumerate(module_types):
                    weights[module_type] = normalized_weights[i].item()
            else:
                # Fallback to equal weights
                for module_type in module_types:
                    weights[module_type] = 1.0 / len(module_types)
        else:
            # Fallback to equal weights
            for module_type in module_types:
                weights[module_type] = 1.0 / len(module_types)
        
        return weights
    
    def _create_broadcast_content(self, 
                                 attended_outputs: torch.Tensor,
                                 module_outputs: Dict[ModuleType, ModuleOutput],
                                 active_modules: List[ModuleType]) -> Dict[str, Any]:
        """Create broadcast content from attended outputs."""
        broadcast_content = {
            'attended_features': attended_outputs.detach().numpy().tolist(),
            'active_modules': [m.value for m in active_modules],
            'module_insights': {}
        }
        
        # Add insights from active modules
        for module_type in active_modules:
            if module_type in module_outputs:
                output = module_outputs[module_type]
                broadcast_content['module_insights'][module_type.value] = {
                    'content': output.content,
                    'confidence': output.confidence,
                    'relevance': output.relevance_score
                }
        
        return broadcast_content
    
    def _calculate_coherence_score(self, 
                                  module_outputs: Dict[ModuleType, ModuleOutput],
                                  attention_weights: Dict[ModuleType, float]) -> float:
        """Calculate coherence score of the global workspace."""
        if not module_outputs:
            return 0.0
        
        # Calculate coherence based on attention distribution and module confidence
        total_attention = sum(attention_weights.values())
        if total_attention == 0:
            return 0.0
        
        # Weighted average of module confidences
        weighted_confidence = sum(
            output.confidence * attention_weights.get(module_type, 0.0)
            for module_type, output in module_outputs.items()
        ) / total_attention
        
        # Attention concentration (higher is more coherent)
        attention_variance = np.var(list(attention_weights.values()))
        attention_concentration = 1.0 / (1.0 + attention_variance)
        
        # Combine confidence and attention concentration
        coherence_score = (weighted_confidence + attention_concentration) / 2.0
        
        return coherence_score
    
    def get_workspace_statistics(self) -> Dict[str, Any]:
        """Get statistics about the global workspace."""
        return {
            'current_state': {
                'active_modules': [m.value for m in self.current_state.active_modules],
                'attention_weights': {k.value: v for k, v in self.current_state.attention_weights.items()},
                'coherence_score': self.current_state.coherence_score,
                'timestamp': self.current_state.timestamp
            },
            'module_count': len(self.specialized_modules),
            'history_length': len(self.module_output_history),
            'avg_coherence': np.mean([self.current_state.coherence_score]) if hasattr(self, 'current_state') else 0.0
        }

# Factory function for easy integration
def create_global_workspace_system(**kwargs) -> GlobalWorkspaceSystem:
    """Create a configured global workspace system."""
    return GlobalWorkspaceSystem(**kwargs)
