"""
Enhanced Differentiable Neural Computer (DNC) with Meta-Cognitive Monitoring

This enhanced DNC implementation integrates with the 37 cognitive subsystems
and provides comprehensive database storage for all memory operations and metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List, Any
import numpy as np
import logging
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

# Import cognitive monitoring systems
from ..core.cognitive_subsystems import (
    MemoryAccessMonitor, MemoryConsolidationMonitor, MemoryFragmentationMonitor,
    SalientMemoryRetrievalMonitor, MemoryRegularizationMonitor, TemporalMemoryMonitor,
    LearningProgressMonitor, MetaLearningMonitor, PatternRecognitionMonitor
)
from ..database.system_integration import get_system_integration
from ..database.api import Component, LogLevel

logger = logging.getLogger(__name__)

class MemoryOperationType(Enum):
    """Types of memory operations for monitoring."""
    READ = "read"
    WRITE = "write"
    ERASE = "erase"
    ALLOCATE = "allocate"
    RETRIEVE = "retrieve"
    CONSOLIDATE = "consolidate"

@dataclass
class MemoryMetrics:
    """Comprehensive memory metrics for monitoring."""
    memory_utilization: float
    average_usage: float
    max_usage: float
    link_matrix_norm: float
    memory_diversity: float
    read_efficiency: float
    write_efficiency: float
    retrieval_accuracy: float
    consolidation_quality: float
    fragmentation_level: float
    salience_distribution: float
    temporal_coherence: float
    operation_count: int
    error_rate: float
    timestamp: datetime

@dataclass
class MemoryOperation:
    """Record of a memory operation for analysis."""
    operation_type: MemoryOperationType
    memory_indices: List[int]
    salience_values: List[float]
    success: bool
    processing_time: float
    context_hash: str
    timestamp: datetime
    metadata: Dict[str, Any]

class EnhancedDNCMemory(nn.Module):
    """
    Enhanced DNC with comprehensive meta-cognitive monitoring and database integration.
    
    This implementation extends the basic DNC with:
    - Real-time monitoring via 37 cognitive subsystems
    - Database storage for all operations and metrics
    - Advanced salience tracking and retrieval
    - Memory consolidation and fragmentation monitoring
    - Performance analytics and optimization
    """
    
    def __init__(
        self,
        memory_size: int = 512,
        word_size: int = 64,
        num_read_heads: int = 4,
        num_write_heads: int = 1,
        controller_size: int = 256,
        enable_monitoring: bool = True,
        enable_database_storage: bool = True
    ):
        super().__init__()
        
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.controller_size = controller_size
        self.enable_monitoring = enable_monitoring
        self.enable_database_storage = enable_database_storage
        
        # Core DNC components
        self.register_buffer('memory_matrix', torch.zeros(memory_size, word_size))
        self.register_buffer('usage_vector', torch.zeros(memory_size))
        self.register_buffer('write_weights_history', torch.zeros(memory_size))
        self.register_buffer('read_weights_history', torch.zeros(memory_size))
        self.register_buffer('link_matrix', torch.zeros(memory_size, memory_size))
        self.register_buffer('precedence_weights', torch.zeros(memory_size))
        self.register_buffer('memory_salience_map', torch.zeros(memory_size))
        
        # Enhanced tracking
        self.register_buffer('access_frequency', torch.zeros(memory_size))
        self.register_buffer('temporal_links', torch.zeros(memory_size, memory_size))
        self.register_buffer('consolidation_weights', torch.zeros(memory_size))
        self.register_buffer('fragmentation_map', torch.zeros(memory_size))
        
        # Controller networks (initialized on first forward pass)
        self.controller = None
        self.interface_layer = None
        
        # Meta-cognitive monitoring systems
        if self.enable_monitoring:
            self._initialize_monitoring_systems()
        
        # Database integration
        if self.enable_database_storage:
            self._initialize_database_integration()
        
        # Operation tracking
        self.operation_history = []
        self.metrics_history = []
        self.performance_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'average_processing_time': 0.0,
            'memory_hit_rate': 0.0
        }
        
        # Initialize memory
        self._reset_memory()
        
        logger.info(f"Enhanced DNC initialized: {memory_size}x{word_size}, "
                   f"monitoring={enable_monitoring}, database={enable_database_storage}")
    
    def _initialize_monitoring_systems(self):
        """Initialize all cognitive monitoring subsystems."""
        try:
            # Memory-specific monitors
            self.memory_access_monitor = MemoryAccessMonitor()
            self.memory_consolidation_monitor = MemoryConsolidationMonitor()
            self.memory_fragmentation_monitor = MemoryFragmentationMonitor()
            self.salient_memory_retrieval_monitor = SalientMemoryRetrievalMonitor()
            self.memory_regularization_monitor = MemoryRegularizationMonitor()
            self.temporal_memory_monitor = TemporalMemoryMonitor()
            
            # Learning monitors
            self.learning_progress_monitor = LearningProgressMonitor()
            self.meta_learning_monitor = MetaLearningMonitor()
            self.pattern_recognition_monitor = PatternRecognitionMonitor()
            
            logger.info("Cognitive monitoring systems initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring systems: {e}")
            self.enable_monitoring = False
    
    def _initialize_database_integration(self):
        """Initialize database integration for persistent storage."""
        try:
            self.integration = get_system_integration()
            logger.info("Database integration initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database integration: {e}")
            self.enable_database_storage = False
    
    def _reset_memory(self):
        """Reset memory to initial state."""
        self.memory_matrix.fill_(0.0)
        self.usage_vector.fill_(0.0)
        self.write_weights_history.fill_(0.0)
        self.read_weights_history.fill_(0.0)
        self.link_matrix.fill_(0.0)
        self.precedence_weights.fill_(0.0)
        self.memory_salience_map.fill_(0.0)
        self.access_frequency.fill_(0.0)
        self.temporal_links.fill_(0.0)
        self.consolidation_weights.fill_(0.0)
        self.fragmentation_map.fill_(0.0)
    
    def forward(
        self, 
        input_data: torch.Tensor,
        prev_reads: torch.Tensor,
        controller_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """
        Enhanced forward pass with comprehensive monitoring and database integration.
        """
        start_time = datetime.now()
        operation_success = True
        error_message = None
        
        try:
            # Initialize controller if needed
            if self.controller is None:
                self._initialize_controller(input_data, prev_reads)
            
            # Perform core DNC operations
            read_vectors, controller_output, new_controller_state, debug_info = self._core_dnc_forward(
                input_data, prev_reads, controller_state
            )
            
            # Enhanced monitoring and analysis
            if self.enable_monitoring:
                self._update_monitoring_systems(debug_info, context)
            
            # Database storage (skip in sync context)
            if self.enable_database_storage and asyncio.get_event_loop().is_running():
                # Only run async operations if we're in an async context
                asyncio.create_task(self._store_operation_data(debug_info, context, start_time))
            
            # Update performance statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_stats(True, processing_time)
            
            return read_vectors, controller_output, new_controller_state, debug_info
            
        except Exception as e:
            operation_success = False
            error_message = str(e)
            logger.error(f"DNC forward pass failed: {e}")
            
            # Update performance statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_stats(False, processing_time)
            
            # Return fallback values
            batch_size = input_data.size(0)
            fallback_reads = torch.zeros(batch_size, self.num_read_heads * self.word_size)
            fallback_controller = torch.zeros(batch_size, self.controller_size)
            fallback_state = (torch.zeros(1, batch_size, self.controller_size), 
                            torch.zeros(1, batch_size, self.controller_size))
            fallback_debug = {'error': error_message, 'operation_success': False}
            
            return fallback_reads, fallback_controller, fallback_state, fallback_debug
    
    def _initialize_controller(self, input_data: torch.Tensor, prev_reads: torch.Tensor):
        """Initialize controller and interface layer on first forward pass."""
        controller_input_size = input_data.size(-1) + prev_reads.size(-1)
        
        self.controller = nn.LSTM(
            input_size=controller_input_size,
            hidden_size=self.controller_size,
            batch_first=True
        )
        
        # Calculate interface layer size
        interface_size = (
            self.num_read_heads * self.word_size +  # Read keys
            self.num_read_heads +                   # Read strengths
            self.num_write_heads * self.word_size + # Write keys
            self.num_write_heads +                  # Write strengths
            self.num_write_heads * self.word_size + # Write vectors
            self.num_write_heads * self.word_size + # Erase vectors
            self.num_write_heads +                  # Free gates
            self.num_write_heads +                  # Allocation gates
            self.num_write_heads +                  # Write gates
            self.num_read_heads * 3                 # Read modes
        )
        
        self.interface_layer = nn.Linear(self.controller_size, interface_size)
        
        logger.info(f"Controller initialized: input_size={controller_input_size}, "
                   f"interface_size={interface_size}")
    
    def _core_dnc_forward(
        self,
        input_data: torch.Tensor,
        prev_reads: torch.Tensor,
        controller_state: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """Core DNC forward pass logic."""
        batch_size = input_data.size(0)
        
        # Controller forward pass
        controller_input = torch.cat([input_data, prev_reads], dim=-1)
        controller_output, new_controller_state = self.controller(
            controller_input.unsqueeze(1), controller_state
        )
        controller_output = controller_output.squeeze(1)
        
        # Generate interface parameters
        interface_params = self.interface_layer(controller_output)
        params = self._parse_interface_params(interface_params)
        
        # Memory operations
        read_vectors, write_info = self._enhanced_memory_operations(params, batch_size)
        
        # Update memory state
        self._update_memory_state(write_info)
        
        # Prepare debug info
        debug_info = self._generate_debug_info(write_info)
        
        return read_vectors, controller_output, new_controller_state, debug_info
    
    def _enhanced_memory_operations(
        self, 
        params: Dict[str, torch.Tensor], 
        batch_size: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Enhanced memory operations with monitoring."""
        
        # Content-based addressing
        write_content_weights = self._content_addressing(
            params['write_keys'], params['write_strengths']
        )
        read_content_weights = self._content_addressing(
            params['read_keys'], params['read_strengths']
        )
        
        # Allocation weights for writing
        allocation_weights = self._allocation_addressing()
        
        # Enhanced write weights with consolidation awareness
        write_weights = self._calculate_enhanced_write_weights(
            write_content_weights, allocation_weights, params
        )
        
        # Enhanced read weights with salience and temporal awareness
        read_weights = self._calculate_enhanced_read_weights(
            read_content_weights, params
        )
        
        # Perform reads
        read_vectors = self._perform_enhanced_reads(read_weights, batch_size)
        
        # Perform writes
        self._perform_enhanced_writes(
            write_weights, params['erase_vectors'], params['write_vectors'],
            params['free_gates']
        )
        
        # Update access tracking
        self._update_access_tracking(read_weights, write_weights)
        
        write_info = {
            'write_weights': write_weights,
            'read_weights': read_weights,
            'allocation_weights': allocation_weights,
            'read_vectors': read_vectors
        }
        
        return read_vectors.view(batch_size, -1), write_info
    
    def _calculate_enhanced_write_weights(
        self,
        content_weights: torch.Tensor,
        allocation_weights: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate write weights with consolidation awareness."""
        
        # Base write weights
        write_weights = (
            params['write_gates'].unsqueeze(-1) * (
                params['allocation_gates'].unsqueeze(-1) * allocation_weights.unsqueeze(0) +
                (1 - params['allocation_gates'].unsqueeze(-1)) * content_weights
            )
        )
        
        # Apply consolidation weights if monitoring is enabled
        if self.enable_monitoring and hasattr(self, 'consolidation_weights'):
            consolidation_factor = self.consolidation_weights.unsqueeze(0).unsqueeze(0)
            write_weights = write_weights * (1 + 0.1 * consolidation_factor)
        
        return write_weights
    
    def _calculate_enhanced_read_weights(
        self,
        content_weights: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate read weights with salience and temporal awareness."""
        
        # Temporal addressing
        read_weights = self._temporal_addressing(content_weights, params['read_modes'])
        
        # Apply salience weighting
        if hasattr(self, 'memory_salience_map'):
            salience_weights = self.memory_salience_map.unsqueeze(0).unsqueeze(0)
            read_weights = read_weights * (1 + 0.2 * salience_weights)
        
        # Apply fragmentation awareness
        if hasattr(self, 'fragmentation_map'):
            fragmentation_penalty = self.fragmentation_map.unsqueeze(0).unsqueeze(0)
            read_weights = read_weights * (1 - 0.1 * fragmentation_penalty)
        
        return read_weights
    
    def _perform_enhanced_reads(
        self,
        read_weights: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """Perform enhanced read operations with monitoring."""
        
        # Ensure correct dimensions
        if read_weights.size(0) != batch_size:
            read_weights = read_weights.expand(batch_size, -1, -1)
        
        # Perform reads
        read_vectors = torch.matmul(
            read_weights.unsqueeze(-2),
            self.memory_matrix.unsqueeze(0).unsqueeze(0).expand(
                batch_size, self.num_read_heads, -1, -1
            )
        ).squeeze(-2)
        
        # Update read tracking
        self.read_weights_history = read_weights.mean(dim=0).mean(dim=0)
        
        return read_vectors
    
    def _perform_enhanced_writes(
        self,
        write_weights: torch.Tensor,
        erase_vectors: torch.Tensor,
        write_vectors: torch.Tensor,
        free_gates: torch.Tensor
    ):
        """Perform enhanced write operations with monitoring."""
        
        batch_size = write_weights.size(0)
        
        # Average across batch for memory update
        avg_write_weights = write_weights.mean(dim=0)
        avg_erase_vectors = erase_vectors.mean(dim=0)
        avg_write_vectors = write_vectors.mean(dim=0)
        avg_free_gates = free_gates.mean(dim=0)
        
        # Perform writes with consolidation awareness
        for head in range(self.num_write_heads):
            write_weight = avg_write_weights[head].unsqueeze(-1)
            erase_vector = avg_erase_vectors[head].unsqueeze(0)
            write_vector = avg_write_vectors[head].unsqueeze(0)
            free_gate = avg_free_gates[head]
            
            # Erase operation
            self.memory_matrix = self.memory_matrix * (
                1 - write_weight * erase_vector
            )
            
            # Write operation
            self.memory_matrix = self.memory_matrix + write_weight * write_vector
            
            # Update write tracking
            self.write_weights_history = torch.max(
                self.write_weights_history, write_weight.squeeze()
            )
    
    def _update_access_tracking(self, read_weights: torch.Tensor, write_weights: torch.Tensor):
        """Update access frequency tracking for monitoring."""
        
        # Update read access frequency
        read_access = read_weights.mean(dim=0).mean(dim=0)
        self.access_frequency = 0.9 * self.access_frequency + 0.1 * read_access
        
        # Update write access frequency
        write_access = write_weights.mean(dim=0).mean(dim=0)
        self.access_frequency = 0.9 * self.access_frequency + 0.1 * write_access
    
    def _update_monitoring_systems(self, debug_info: Dict[str, Any], context: Optional[Dict[str, Any]]):
        """Update all cognitive monitoring systems."""
        
        try:
            # Update memory access monitoring
            if hasattr(self, 'memory_access_monitor'):
                access_data = {
                    'memory_utilization': debug_info.get('memory_utilization', 0.0),
                    'access_frequency': self.access_frequency.mean().item(),
                    'read_weights': debug_info.get('read_weights'),
                    'write_weights': debug_info.get('write_weights')
                }
                self.memory_access_monitor.update_metrics(access_data)
            
            # Update consolidation monitoring
            if hasattr(self, 'memory_consolidation_monitor'):
                consolidation_data = {
                    'consolidation_weights': self.consolidation_weights,
                    'memory_matrix': self.memory_matrix,
                    'usage_vector': self.usage_vector
                }
                self.memory_consolidation_monitor.update_metrics(consolidation_data)
            
            # Update fragmentation monitoring
            if hasattr(self, 'memory_fragmentation_monitor'):
                fragmentation_data = {
                    'fragmentation_map': self.fragmentation_map,
                    'memory_utilization': debug_info.get('memory_utilization', 0.0),
                    'access_patterns': self.access_frequency
                }
                self.memory_fragmentation_monitor.update_metrics(fragmentation_data)
            
            # Update salience monitoring
            if hasattr(self, 'salient_memory_retrieval_monitor'):
                salience_data = {
                    'salience_map': self.memory_salience_map,
                    'retrieval_accuracy': debug_info.get('retrieval_accuracy', 0.0),
                    'context': context
                }
                self.salient_memory_retrieval_monitor.update_metrics(salience_data)
            
        except Exception as e:
            logger.error(f"Failed to update monitoring systems: {e}")
    
    async def _store_operation_data(
        self, 
        debug_info: Dict[str, Any], 
        context: Optional[Dict[str, Any]], 
        start_time: datetime
    ):
        """Store operation data in database."""
        
        try:
            if not hasattr(self, 'integration'):
                return
            
            # Create operation record
            operation = MemoryOperation(
                operation_type=MemoryOperationType.READ,  # Default to read
                memory_indices=list(range(self.memory_size)),
                salience_values=self.memory_salience_map.tolist(),
                success=debug_info.get('operation_success', True),
                processing_time=(datetime.now() - start_time).total_seconds(),
                context_hash=hash(str(context)) if context else 0,
                timestamp=start_time,
                metadata=debug_info
            )
            
            # Store in database
            await self.integration.log_system_event(
                LogLevel.INFO,
                Component.MEMORY_SYSTEM,
                f"DNC operation: {operation.operation_type.value}",
                asdict(operation),
                "enhanced_dnc"
            )
            
        except Exception as e:
            logger.error(f"Failed to store operation data: {e}")
    
    def _update_performance_stats(self, success: bool, processing_time: float):
        """Update performance statistics."""
        
        self.performance_stats['total_operations'] += 1
        
        if success:
            self.performance_stats['successful_operations'] += 1
        else:
            self.performance_stats['failed_operations'] += 1
        
        # Update average processing time
        total_ops = self.performance_stats['total_operations']
        current_avg = self.performance_stats['average_processing_time']
        self.performance_stats['average_processing_time'] = (
            (current_avg * (total_ops - 1) + processing_time) / total_ops
        )
        
        # Update success rate
        self.performance_stats['memory_hit_rate'] = (
            self.performance_stats['successful_operations'] / total_ops
        )
    
    def _generate_debug_info(self, write_info: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Generate comprehensive debug information."""
        
        return {
            'memory_utilization': float((self.usage_vector > 0.1).float().mean().detach()),
            'average_usage': float(self.usage_vector.mean().detach()),
            'max_usage': float(self.usage_vector.max().detach()),
            'link_matrix_norm': float(self.link_matrix.norm().detach()),
            'memory_diversity': float(torch.std(self.memory_matrix, dim=0).mean().detach()),
            'read_efficiency': float(self.access_frequency.mean().detach()),
            'write_efficiency': float(self.write_weights_history.mean().detach()),
            'salience_distribution': float(self.memory_salience_map.std().detach()),
            'temporal_coherence': float(self.temporal_links.mean().detach()),
            'operation_success': True,
            'performance_stats': self.performance_stats.copy()
        }
    
    def get_comprehensive_metrics(self) -> MemoryMetrics:
        """Get comprehensive memory metrics for analysis."""
        
        return MemoryMetrics(
            memory_utilization=float((self.usage_vector > 0.1).float().mean().detach()),
            average_usage=float(self.usage_vector.mean().detach()),
            max_usage=float(self.usage_vector.max().detach()),
            link_matrix_norm=float(self.link_matrix.norm().detach()),
            memory_diversity=float(torch.std(self.memory_matrix, dim=0).mean().detach()),
            read_efficiency=float(self.access_frequency.mean().detach()),
            write_efficiency=float(self.write_weights_history.mean().detach()),
            retrieval_accuracy=self.performance_stats['memory_hit_rate'],
            consolidation_quality=float(self.consolidation_weights.mean().detach()),
            fragmentation_level=float(self.fragmentation_map.mean().detach()),
            salience_distribution=float(self.memory_salience_map.std().detach()),
            temporal_coherence=float(self.temporal_links.mean().detach()),
            operation_count=self.performance_stats['total_operations'],
            error_rate=1.0 - self.performance_stats['memory_hit_rate'],
            timestamp=datetime.now()
        )
    
    def retrieve_salient_memories(
        self, 
        current_context: torch.Tensor, 
        salience_threshold: float = 0.6,
        max_retrievals: int = 5
    ) -> List[Tuple[torch.Tensor, float]]:
        """Enhanced salient memory retrieval with monitoring."""
        
        try:
            # Compute context similarity
            current_context_norm = F.normalize(current_context.unsqueeze(0), dim=-1)
            memory_norm = F.normalize(self.memory_matrix, dim=-1)
            similarities = torch.matmul(current_context_norm, memory_norm.t()).squeeze(0)
            
            # Weight by salience
            salience_weighted_similarities = similarities * self.memory_salience_map
            
            # Filter by threshold
            valid_mask = self.memory_salience_map >= salience_threshold
            if not valid_mask.any():
                return []
            
            # Get top retrievals
            valid_similarities = salience_weighted_similarities[valid_mask]
            valid_indices = torch.nonzero(valid_mask).squeeze(-1)
            sorted_indices = torch.argsort(valid_similarities, descending=True)
            top_indices = valid_indices[sorted_indices[:max_retrievals]]
            
            # Return results
            retrieved_memories = []
            for idx in top_indices:
                memory_vector = self.memory_matrix[idx]
                relevance_score = salience_weighted_similarities[idx].item()
                retrieved_memories.append((memory_vector, relevance_score))
            
            # Update monitoring
            if self.enable_monitoring and hasattr(self, 'salient_memory_retrieval_monitor'):
                retrieval_data = {
                    'retrieval_count': len(retrieved_memories),
                    'average_relevance': np.mean([score for _, score in retrieved_memories]) if retrieved_memories else 0.0,
                    'salience_threshold': salience_threshold
                }
                self.salient_memory_retrieval_monitor.update_metrics(retrieval_data)
            
            return retrieved_memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve salient memories: {e}")
            return []
    
    def update_memory_salience(self, memory_indices: torch.Tensor, salience_values: torch.Tensor):
        """Update salience values with monitoring."""
        
        try:
            alpha = 0.1
            for idx, salience in zip(memory_indices, salience_values):
                if 0 <= idx < self.memory_size:
                    current_salience = self.memory_salience_map[idx]
                    # For the first update, use the new value directly
                    if current_salience == 0.0:
                        self.memory_salience_map[idx] = salience
                    else:
                        self.memory_salience_map[idx] = (
                            alpha * salience + (1 - alpha) * current_salience
                        )
            
            # Update monitoring
            if self.enable_monitoring and hasattr(self, 'salient_memory_retrieval_monitor'):
                salience_data = {
                    'salience_map': self.memory_salience_map,
                    'updated_indices': memory_indices.tolist(),
                    'new_values': salience_values.tolist()
                }
                self.salient_memory_retrieval_monitor.update_metrics(salience_data)
                
        except Exception as e:
            logger.error(f"Failed to update memory salience: {e}")
    
    def consolidate_memory(self, consolidation_strength: float = 0.5):
        """Perform memory consolidation with monitoring."""
        
        try:
            # Calculate consolidation weights based on usage and salience
            usage_weights = self.usage_vector / (self.usage_vector.sum() + 1e-8)
            salience_weights = self.memory_salience_map / (self.memory_salience_map.sum() + 1e-8)
            consolidation_weights = (usage_weights + salience_weights) / 2
            
            # Apply consolidation
            self.consolidation_weights = consolidation_weights
            
            # Update monitoring
            if self.enable_monitoring and hasattr(self, 'memory_consolidation_monitor'):
                consolidation_data = {
                    'consolidation_weights': consolidation_weights,
                    'consolidation_strength': consolidation_strength,
                    'memory_matrix': self.memory_matrix,
                    'usage_vector': self.usage_vector
                }
                self.memory_consolidation_monitor.update_metrics(consolidation_data)
            
            logger.info(f"Memory consolidation completed with strength {consolidation_strength}")
            
        except Exception as e:
            logger.error(f"Failed to consolidate memory: {e}")
    
    def analyze_fragmentation(self) -> Dict[str, float]:
        """Analyze memory fragmentation with monitoring."""
        
        try:
            # Calculate fragmentation metrics
            usage_variance = torch.var(self.usage_vector)
            access_variance = torch.var(self.access_frequency)
            fragmentation_score = float(usage_variance + access_variance)
            
            # Update fragmentation map
            self.fragmentation_map = torch.abs(self.usage_vector - self.usage_vector.mean())
            
            # Update monitoring
            if self.enable_monitoring and hasattr(self, 'memory_fragmentation_monitor'):
                fragmentation_data = {
                    'fragmentation_score': fragmentation_score,
                    'fragmentation_map': self.fragmentation_map,
                    'usage_variance': float(usage_variance),
                    'access_variance': float(access_variance)
                }
                self.memory_fragmentation_monitor.update_metrics(fragmentation_data)
            
            return {
                'fragmentation_score': fragmentation_score,
                'usage_variance': float(usage_variance),
                'access_variance': float(access_variance),
                'average_fragmentation': float(self.fragmentation_map.mean())
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze fragmentation: {e}")
            return {'fragmentation_score': 0.0, 'error': str(e)}
    
    def reset_memory(self):
        """Reset memory with monitoring."""
        
        try:
            self._reset_memory()
            
            # Reset performance stats
            self.performance_stats = {
                'total_operations': 0,
                'successful_operations': 0,
                'failed_operations': 0,
                'average_processing_time': 0.0,
                'memory_hit_rate': 0.0
            }
            
            # Clear operation history
            self.operation_history.clear()
            self.metrics_history.clear()
            
            logger.info("Memory reset completed")
            
        except Exception as e:
            logger.error(f"Failed to reset memory: {e}")
    
    # Include all the original DNC methods for compatibility
    def _parse_interface_params(self, interface_params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Parse interface parameters from controller output."""
        batch_size = interface_params.size(0)
        
        # Split interface parameters
        splits = []
        offset = 0
        
        # Read parameters
        read_keys_size = self.num_read_heads * self.word_size
        read_keys = interface_params[:, offset:offset + read_keys_size]
        read_keys = read_keys.view(batch_size, self.num_read_heads, self.word_size)
        offset += read_keys_size
        
        read_strengths = F.softplus(interface_params[:, offset:offset + self.num_read_heads]) + 1
        offset += self.num_read_heads
        
        # Write parameters
        write_keys_size = self.num_write_heads * self.word_size
        write_keys = interface_params[:, offset:offset + write_keys_size]
        write_keys = write_keys.view(batch_size, self.num_write_heads, self.word_size)
        offset += write_keys_size
        
        write_strengths = F.softplus(interface_params[:, offset:offset + self.num_write_heads]) + 1
        offset += self.num_write_heads
        
        write_vectors = interface_params[:, offset:offset + write_keys_size]
        write_vectors = write_vectors.view(batch_size, self.num_write_heads, self.word_size)
        offset += write_keys_size
        
        erase_vectors = torch.sigmoid(interface_params[:, offset:offset + write_keys_size])
        erase_vectors = erase_vectors.view(batch_size, self.num_write_heads, self.word_size)
        offset += write_keys_size
        
        # Gates
        free_gates = torch.sigmoid(interface_params[:, offset:offset + self.num_write_heads])
        offset += self.num_write_heads
        
        allocation_gates = torch.sigmoid(interface_params[:, offset:offset + self.num_write_heads])
        offset += self.num_write_heads
        
        write_gates = torch.sigmoid(interface_params[:, offset:offset + self.num_write_heads])
        offset += self.num_write_heads
        
        # Read modes
        read_modes = F.softmax(
            interface_params[:, offset:offset + self.num_read_heads * 3].view(
                batch_size, self.num_read_heads, 3
            ), dim=-1
        )
        
        return {
            'read_keys': read_keys,
            'read_strengths': read_strengths,
            'write_keys': write_keys,
            'write_strengths': write_strengths,
            'write_vectors': write_vectors,
            'erase_vectors': erase_vectors,
            'free_gates': free_gates,
            'allocation_gates': allocation_gates,
            'write_gates': write_gates,
            'read_modes': read_modes
        }
    
    def _content_addressing(
        self, 
        keys: torch.Tensor, 
        strengths: torch.Tensor
    ) -> torch.Tensor:
        """Content-based addressing using cosine similarity."""
        batch_size, num_heads, word_size = keys.shape
        
        # Normalize keys and memory
        keys_norm = F.normalize(keys, dim=-1)
        memory_norm = F.normalize(self.memory_matrix, dim=-1)
        
        # Compute cosine similarities
        similarities = torch.matmul(
            keys_norm.view(batch_size * num_heads, word_size),
            memory_norm.t()
        ).view(batch_size, num_heads, self.memory_size)
        
        # Apply strength and softmax
        weights = F.softmax(similarities * strengths.unsqueeze(-1), dim=-1)
        
        return weights
    
    def _allocation_addressing(self) -> torch.Tensor:
        """Allocation-based addressing for finding free memory locations."""
        # Sort usage vector to find least used locations
        sorted_usage, indices = torch.sort(self.usage_vector)
        
        # Create allocation weights (prefer least used locations)
        allocation_weights = torch.zeros_like(self.usage_vector)
        
        # Allocate to least used locations
        num_allocate = min(10, self.memory_size)  # Allocate to top 10 least used
        for i in range(num_allocate):
            idx = indices[i]
            allocation_weights[idx] = (num_allocate - i) / num_allocate
            
        # Normalize
        allocation_weights = allocation_weights / (allocation_weights.sum() + 1e-8)
        
        return allocation_weights
    
    def _temporal_addressing(
        self, 
        content_weights: torch.Tensor, 
        read_modes: torch.Tensor
    ) -> torch.Tensor:
        """Temporal addressing using link matrix."""
        batch_size, num_heads, _ = content_weights.shape
        
        # Forward and backward weights from link matrix
        forward_weights = torch.matmul(
            content_weights.view(batch_size * num_heads, 1, self.memory_size),
            self.link_matrix.unsqueeze(0).expand(batch_size * num_heads, -1, -1)
        ).squeeze(1).view(batch_size, num_heads, self.memory_size)
        
        backward_weights = torch.matmul(
            content_weights.view(batch_size * num_heads, 1, self.memory_size),
            self.link_matrix.t().unsqueeze(0).expand(batch_size * num_heads, -1, -1)
        ).squeeze(1).view(batch_size, num_heads, self.memory_size)
        
        # Combine using read modes
        read_weights = (
            read_modes[:, :, 0:1] * backward_weights +
            read_modes[:, :, 1:2] * content_weights +
            read_modes[:, :, 2:3] * forward_weights
        )
        
        return read_weights
    
    def _update_memory_state(self, write_info: Dict[str, torch.Tensor]):
        """Update memory usage and temporal links."""
        write_weights = write_info['write_weights'].mean(dim=0)  # Average across batch
        read_weights = write_info['read_weights'].mean(dim=0)    # Average across batch
        
        # Update usage vector
        for head in range(self.num_write_heads):
            self.usage_vector = (
                self.usage_vector + write_weights[head] - 
                self.usage_vector * write_weights[head]
            )
            
        # Update precedence weights and link matrix
        for head in range(self.num_write_heads):
            # Update precedence
            self.precedence_weights = (
                (1 - write_weights[head].sum()) * self.precedence_weights +
                write_weights[head]
            )
            
            # Update link matrix
            write_weights_expanded = write_weights[head].unsqueeze(-1)
            precedence_expanded = self.precedence_weights.unsqueeze(0)
            
            self.link_matrix = (
                (1 - write_weights_expanded - write_weights_expanded.t()) * self.link_matrix +
                write_weights_expanded * precedence_expanded
            )
            
        # Decay link matrix to prevent overflow
        self.link_matrix = self.link_matrix * 0.99
        
        # Update temporal links for monitoring
        self.temporal_links = self.link_matrix
