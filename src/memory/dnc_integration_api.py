"""
DNC Integration API

Provides a unified API for integrating the Enhanced DNC with the broader system,
including database storage, monitoring, and cognitive subsystem coordination.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import torch
import numpy as np

from .enhanced_dnc import EnhancedDNCMemory, MemoryMetrics, MemoryOperation, MemoryOperationType
from ..core.cognitive_subsystems import CognitiveCoordinator
from ..database.system_integration import get_system_integration
from ..database.api import Component, LogLevel

logger = logging.getLogger(__name__)

class DNCIntegrationAPI:
    """
    Unified API for DNC integration with meta-cognitive monitoring and database storage.
    
    This API provides:
    - Easy DNC initialization and configuration
    - Automatic monitoring and database integration
    - Cognitive subsystem coordination
    - Performance analytics and reporting
    - Memory optimization and maintenance
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
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.controller_size = controller_size
        self.enable_monitoring = enable_monitoring
        self.enable_database_storage = enable_database_storage
        
        # Initialize DNC
        self.dnc = EnhancedDNCMemory(
            memory_size=memory_size,
            word_size=word_size,
            num_read_heads=num_read_heads,
            num_write_heads=num_write_heads,
            controller_size=controller_size,
            enable_monitoring=enable_monitoring,
            enable_database_storage=enable_database_storage
        )
        
        # Initialize cognitive coordinator
        if enable_monitoring:
            self.cognitive_coordinator = CognitiveCoordinator()
        else:
            self.cognitive_coordinator = None
        
        # Database integration
        if enable_database_storage:
            self.integration = get_system_integration()
        else:
            self.integration = None
        
        # Session tracking
        self.session_id = f"dnc_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_start_time = datetime.now()
        self.operation_count = 0
        
        logger.info(f"DNC Integration API initialized: session={self.session_id}")
    
    async def initialize(self) -> bool:
        """Initialize the DNC integration system."""
        try:
            # Initialize cognitive coordinator
            if self.cognitive_coordinator:
                await self.cognitive_coordinator.initialize_all_subsystems()
                logger.info("Cognitive subsystems initialized")
            
            # Log initialization
            if self.integration:
                await self.integration.log_system_event(
                    LogLevel.INFO,
                    Component.MEMORY_SYSTEM,
                    f"DNC Integration API initialized: {self.session_id}",
                    {
                        'session_id': self.session_id,
                        'memory_size': self.memory_size,
                        'word_size': self.word_size,
                        'num_read_heads': self.num_read_heads,
                        'num_write_heads': self.num_write_heads,
                        'controller_size': self.controller_size,
                        'enable_monitoring': self.enable_monitoring,
                        'enable_database_storage': self.enable_database_storage
                    },
                    self.session_id
                )
            
            logger.info("DNC Integration API initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DNC Integration API: {e}")
            return False
    
    async def process_input(
        self,
        input_data: torch.Tensor,
        prev_reads: torch.Tensor,
        controller_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process input through the DNC with full monitoring and database integration.
        
        Args:
            input_data: Input tensor
            prev_reads: Previous read vectors
            controller_state: Previous controller state
            context: Additional context information
            
        Returns:
            Dictionary containing results and metadata
        """
        start_time = datetime.now()
        self.operation_count += 1
        
        try:
            # Add context information
            if context is None:
                context = {}
            
            context.update({
                'session_id': self.session_id,
                'operation_count': self.operation_count,
                'timestamp': start_time.isoformat()
            })
            
            # Process through DNC
            read_vectors, controller_output, new_controller_state, debug_info = self.dnc(
                input_data, prev_reads, controller_state, context
            )
            
            # Update cognitive monitoring
            if self.cognitive_coordinator:
                await self._update_cognitive_monitoring(debug_info, context)
            
            # Store operation data
            if self.integration:
                await self._store_operation_data(debug_info, context, start_time)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare result
            result = {
                'read_vectors': read_vectors,
                'controller_output': controller_output,
                'new_controller_state': new_controller_state,
                'debug_info': debug_info,
                'processing_time': processing_time,
                'operation_success': debug_info.get('operation_success', True),
                'session_id': self.session_id,
                'operation_count': self.operation_count
            }
            
            logger.debug(f"DNC processing completed: {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"DNC processing failed: {e}")
            return {
                'read_vectors': torch.zeros(input_data.size(0), self.num_read_heads * self.word_size),
                'controller_output': torch.zeros(input_data.size(0), self.controller_size),
                'new_controller_state': (torch.zeros(1, input_data.size(0), self.controller_size),
                                       torch.zeros(1, input_data.size(0), self.controller_size)),
                'debug_info': {'error': str(e), 'operation_success': False},
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'operation_success': False,
                'session_id': self.session_id,
                'operation_count': self.operation_count
            }
    
    async def retrieve_memories(
        self,
        query: torch.Tensor,
        salience_threshold: float = 0.6,
        max_retrievals: int = 5,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve memories based on query with monitoring and database integration.
        
        Args:
            query: Query tensor for memory retrieval
            salience_threshold: Minimum salience for retrieval
            max_retrievals: Maximum number of memories to retrieve
            context: Additional context information
            
        Returns:
            Dictionary containing retrieved memories and metadata
        """
        start_time = datetime.now()
        
        try:
            # Add context information
            if context is None:
                context = {}
            
            context.update({
                'session_id': self.session_id,
                'retrieval_timestamp': start_time.isoformat(),
                'salience_threshold': salience_threshold,
                'max_retrievals': max_retrievals
            })
            
            # Retrieve memories
            retrieved_memories = self.dnc.retrieve_salient_memories(
                query, salience_threshold, max_retrievals
            )
            
            # Prepare result
            result = {
                'retrieved_memories': retrieved_memories,
                'memory_count': len(retrieved_memories),
                'average_relevance': np.mean([score for _, score in retrieved_memories]) if retrieved_memories else 0.0,
                'salience_threshold': salience_threshold,
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'session_id': self.session_id
            }
            
            # Store retrieval data
            if self.integration:
                await self.integration.log_system_event(
                    LogLevel.INFO,
                    Component.MEMORY_SYSTEM,
                    f"Memory retrieval: {len(retrieved_memories)} memories",
                    result,
                    self.session_id
                )
            
            logger.debug(f"Memory retrieval completed: {len(retrieved_memories)} memories")
            return result
            
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return {
                'retrieved_memories': [],
                'memory_count': 0,
                'average_relevance': 0.0,
                'salience_threshold': salience_threshold,
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'error': str(e),
                'session_id': self.session_id
            }
    
    async def update_memory_salience(
        self,
        memory_indices: List[int],
        salience_values: List[float],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update memory salience values with monitoring and database integration.
        
        Args:
            memory_indices: Indices of memories to update
            salience_values: New salience values
            context: Additional context information
            
        Returns:
            Dictionary containing update results and metadata
        """
        start_time = datetime.now()
        
        try:
            # Convert to tensors
            indices_tensor = torch.tensor(memory_indices, dtype=torch.long)
            values_tensor = torch.tensor(salience_values, dtype=torch.float32)
            
            # Add context information
            if context is None:
                context = {}
            
            context.update({
                'session_id': self.session_id,
                'update_timestamp': start_time.isoformat(),
                'memory_indices': memory_indices,
                'salience_values': salience_values
            })
            
            # Update salience
            self.dnc.update_memory_salience(indices_tensor, values_tensor)
            
            # Prepare result
            result = {
                'updated_indices': memory_indices,
                'updated_values': salience_values,
                'update_count': len(memory_indices),
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'session_id': self.session_id
            }
            
            # Store update data
            if self.integration:
                await self.integration.log_system_event(
                    LogLevel.INFO,
                    Component.MEMORY_SYSTEM,
                    f"Memory salience updated: {len(memory_indices)} memories",
                    result,
                    self.session_id
                )
            
            logger.debug(f"Memory salience update completed: {len(memory_indices)} memories")
            return result
            
        except Exception as e:
            logger.error(f"Memory salience update failed: {e}")
            return {
                'updated_indices': [],
                'updated_values': [],
                'update_count': 0,
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'error': str(e),
                'session_id': self.session_id
            }
    
    async def consolidate_memory(
        self,
        consolidation_strength: float = 0.5,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform memory consolidation with monitoring and database integration.
        
        Args:
            consolidation_strength: Strength of consolidation (0.0 to 1.0)
            context: Additional context information
            
        Returns:
            Dictionary containing consolidation results and metadata
        """
        start_time = datetime.now()
        
        try:
            # Add context information
            if context is None:
                context = {}
            
            context.update({
                'session_id': self.session_id,
                'consolidation_timestamp': start_time.isoformat(),
                'consolidation_strength': consolidation_strength
            })
            
            # Perform consolidation
            self.dnc.consolidate_memory(consolidation_strength)
            
            # Get metrics after consolidation
            metrics = self.dnc.get_comprehensive_metrics()
            
            # Prepare result
            result = {
                'consolidation_strength': consolidation_strength,
                'consolidation_quality': metrics.consolidation_quality,
                'memory_utilization': metrics.memory_utilization,
                'memory_diversity': metrics.memory_diversity,
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'session_id': self.session_id
            }
            
            # Store consolidation data
            if self.integration:
                await self.integration.log_system_event(
                    LogLevel.INFO,
                    Component.MEMORY_SYSTEM,
                    f"Memory consolidation completed: strength={consolidation_strength}",
                    result,
                    self.session_id
                )
            
            logger.info(f"Memory consolidation completed: strength={consolidation_strength}")
            return result
            
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
            return {
                'consolidation_strength': consolidation_strength,
                'consolidation_quality': 0.0,
                'memory_utilization': 0.0,
                'memory_diversity': 0.0,
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'error': str(e),
                'session_id': self.session_id
            }
    
    async def analyze_fragmentation(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze memory fragmentation with monitoring and database integration.
        
        Args:
            context: Additional context information
            
        Returns:
            Dictionary containing fragmentation analysis and metadata
        """
        start_time = datetime.now()
        
        try:
            # Add context information
            if context is None:
                context = {}
            
            context.update({
                'session_id': self.session_id,
                'analysis_timestamp': start_time.isoformat()
            })
            
            # Analyze fragmentation
            fragmentation_analysis = self.dnc.analyze_fragmentation()
            
            # Prepare result
            result = {
                'fragmentation_analysis': fragmentation_analysis,
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'session_id': self.session_id
            }
            
            # Store analysis data
            if self.integration:
                await self.integration.log_system_event(
                    LogLevel.INFO,
                    Component.MEMORY_SYSTEM,
                    f"Memory fragmentation analysis completed",
                    result,
                    self.session_id
                )
            
            logger.debug(f"Memory fragmentation analysis completed")
            return result
            
        except Exception as e:
            logger.error(f"Memory fragmentation analysis failed: {e}")
            return {
                'fragmentation_analysis': {'fragmentation_score': 0.0, 'error': str(e)},
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'error': str(e),
                'session_id': self.session_id
            }
    
    async def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for the DNC system."""
        try:
            # Get DNC metrics
            dnc_metrics = self.dnc.get_comprehensive_metrics()
            
            # Get cognitive subsystem metrics
            cognitive_metrics = {}
            if self.cognitive_coordinator:
                cognitive_metrics = await self.cognitive_coordinator.get_all_subsystem_metrics()
            
            # Prepare comprehensive result
            result = {
                'dnc_metrics': {
                    'memory_utilization': dnc_metrics.memory_utilization,
                    'average_usage': dnc_metrics.average_usage,
                    'max_usage': dnc_metrics.max_usage,
                    'link_matrix_norm': dnc_metrics.link_matrix_norm,
                    'memory_diversity': dnc_metrics.memory_diversity,
                    'read_efficiency': dnc_metrics.read_efficiency,
                    'write_efficiency': dnc_metrics.write_efficiency,
                    'retrieval_accuracy': dnc_metrics.retrieval_accuracy,
                    'consolidation_quality': dnc_metrics.consolidation_quality,
                    'fragmentation_level': dnc_metrics.fragmentation_level,
                    'salience_distribution': dnc_metrics.salience_distribution,
                    'temporal_coherence': dnc_metrics.temporal_coherence,
                    'operation_count': dnc_metrics.operation_count,
                    'error_rate': dnc_metrics.error_rate,
                    'timestamp': dnc_metrics.timestamp.isoformat()
                },
                'cognitive_metrics': cognitive_metrics,
                'session_info': {
                    'session_id': self.session_id,
                    'session_start_time': self.session_start_time.isoformat(),
                    'operation_count': self.operation_count,
                    'session_duration': (datetime.now() - self.session_start_time).total_seconds()
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive metrics: {e}")
            return {
                'dnc_metrics': {},
                'cognitive_metrics': {},
                'session_info': {
                    'session_id': self.session_id,
                    'error': str(e)
                }
            }
    
    async def _update_cognitive_monitoring(
        self, 
        debug_info: Dict[str, Any], 
        context: Dict[str, Any]
    ):
        """Update cognitive monitoring systems."""
        try:
            if not self.cognitive_coordinator:
                return
            
            # Update all subsystems with DNC data
            monitoring_data = {
                'memory_utilization': debug_info.get('memory_utilization', 0.0),
                'memory_diversity': debug_info.get('memory_diversity', 0.0),
                'read_efficiency': debug_info.get('read_efficiency', 0.0),
                'write_efficiency': debug_info.get('write_efficiency', 0.0),
                'operation_success': debug_info.get('operation_success', True),
                'context': context
            }
            
            await self.cognitive_coordinator.update_all_subsystems(monitoring_data)
            
        except Exception as e:
            logger.error(f"Failed to update cognitive monitoring: {e}")
    
    async def _store_operation_data(
        self, 
        debug_info: Dict[str, Any], 
        context: Dict[str, Any], 
        start_time: datetime
    ):
        """Store operation data in database."""
        try:
            if not self.integration:
                return
            
            # Create operation record
            operation_data = {
                'session_id': self.session_id,
                'operation_count': self.operation_count,
                'start_time': start_time.isoformat(),
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'debug_info': debug_info,
                'context': context
            }
            
            await self.integration.log_system_event(
                LogLevel.INFO,
                Component.MEMORY_SYSTEM,
                f"DNC operation {self.operation_count}",
                operation_data,
                self.session_id
            )
            
        except Exception as e:
            logger.error(f"Failed to store operation data: {e}")
    
    async def cleanup(self):
        """Cleanup resources and finalize session."""
        try:
            # Get final metrics
            final_metrics = await self.get_comprehensive_metrics()
            
            # Store session summary
            if self.integration:
                await self.integration.log_system_event(
                    LogLevel.INFO,
                    Component.MEMORY_SYSTEM,
                    f"DNC session completed: {self.session_id}",
                    final_metrics,
                    self.session_id
                )
            
            # Cleanup cognitive coordinator
            if self.cognitive_coordinator:
                await self.cognitive_coordinator.cleanup_all_subsystems()
            
            logger.info(f"DNC Integration API cleanup completed: {self.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup DNC Integration API: {e}")
    
    def reset_memory(self):
        """Reset DNC memory state."""
        try:
            self.dnc.reset_memory()
            self.operation_count = 0
            logger.info("DNC memory reset completed")
        except Exception as e:
            logger.error(f"Failed to reset DNC memory: {e}")


# Factory function for easy creation
def create_dnc_integration(
    memory_size: int = 512,
    word_size: int = 64,
    num_read_heads: int = 4,
    num_write_heads: int = 1,
    controller_size: int = 256,
    enable_monitoring: bool = True,
    enable_database_storage: bool = True
) -> DNCIntegrationAPI:
    """
    Factory function to create a DNC Integration API instance.
    
    Args:
        memory_size: Size of the memory matrix
        word_size: Size of each memory word
        num_read_heads: Number of read heads
        num_write_heads: Number of write heads
        controller_size: Size of the controller LSTM
        enable_monitoring: Enable cognitive monitoring
        enable_database_storage: Enable database storage
        
    Returns:
        Configured DNCIntegrationAPI instance
    """
    return DNCIntegrationAPI(
        memory_size=memory_size,
        word_size=word_size,
        num_read_heads=num_read_heads,
        num_write_heads=num_write_heads,
        controller_size=controller_size,
        enable_monitoring=enable_monitoring,
        enable_database_storage=enable_database_storage
    )
