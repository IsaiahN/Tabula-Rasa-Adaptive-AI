"""
Test DNC Integration with Meta-Cognitive Monitoring

Comprehensive tests for the enhanced DNC system with database integration
and cognitive subsystem monitoring.
"""

import pytest
import torch
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from src.memory.enhanced_dnc import EnhancedDNCMemory, MemoryMetrics, MemoryOperationType
from src.memory.dnc_integration_api import DNCIntegrationAPI, create_dnc_integration


class TestEnhancedDNCMemory:
    """Test the Enhanced DNC Memory system."""
    
    def test_initialization(self):
        """Test DNC initialization with monitoring."""
        dnc = EnhancedDNCMemory(
            memory_size=256,
            word_size=32,
            num_read_heads=2,
            num_write_heads=1,
            controller_size=128,
            enable_monitoring=True,
            enable_database_storage=True
        )
        
        assert dnc.memory_size == 256
        assert dnc.word_size == 32
        assert dnc.num_read_heads == 2
        assert dnc.num_write_heads == 1
        assert dnc.controller_size == 128
        assert dnc.enable_monitoring == True
        assert dnc.enable_database_storage == True
        
        # Check buffer shapes
        assert dnc.memory_matrix.shape == (256, 32)
        assert dnc.usage_vector.shape == (256,)
        assert dnc.memory_salience_map.shape == (256,)
    
    def test_forward_pass(self):
        """Test DNC forward pass."""
        dnc = EnhancedDNCMemory(memory_size=64, word_size=16, enable_monitoring=False)
        
        batch_size = 2
        input_size = 32
        input_data = torch.randn(batch_size, input_size)
        prev_reads = torch.randn(batch_size, dnc.num_read_heads * dnc.word_size)
        
        # First forward pass (initializes controller)
        read_vectors, controller_output, new_state, debug_info = dnc(
            input_data, prev_reads, context={'test': True}
        )
        
        assert read_vectors.shape == (batch_size, dnc.num_read_heads * dnc.word_size)
        assert controller_output.shape == (batch_size, dnc.controller_size)
        assert len(new_state) == 2  # LSTM state tuple
        assert isinstance(debug_info, dict)
        assert 'operation_success' in debug_info
    
    def test_memory_retrieval(self):
        """Test salient memory retrieval."""
        dnc = EnhancedDNCMemory(memory_size=64, word_size=16, enable_monitoring=False)
        
        # Initialize some memory
        dnc.memory_matrix = torch.randn(64, 16)
        dnc.memory_salience_map = torch.rand(64)
        
        # Test retrieval
        query = torch.randn(16)
        memories = dnc.retrieve_salient_memories(query, salience_threshold=0.5, max_retrievals=3)
        
        assert isinstance(memories, list)
        assert len(memories) <= 3
        
        for memory_vector, relevance_score in memories:
            assert memory_vector.shape == (16,)
            assert isinstance(relevance_score, float)
            assert 0.0 <= relevance_score <= 1.0
    
    def test_salience_update(self):
        """Test memory salience update."""
        dnc = EnhancedDNCMemory(memory_size=64, word_size=16, enable_monitoring=False)
        
        # Update salience for specific indices
        indices = torch.tensor([0, 1, 2])
        values = torch.tensor([0.8, 0.9, 0.7])
        
        dnc.update_memory_salience(indices, values)
        
        # Check that salience was updated
        assert torch.allclose(dnc.memory_salience_map[0], torch.tensor(0.8), atol=1e-6)
        assert torch.allclose(dnc.memory_salience_map[1], torch.tensor(0.9), atol=1e-6)
        assert torch.allclose(dnc.memory_salience_map[2], torch.tensor(0.7), atol=1e-6)
    
    def test_memory_consolidation(self):
        """Test memory consolidation."""
        dnc = EnhancedDNCMemory(memory_size=64, word_size=16, enable_monitoring=False)
        
        # Initialize some usage and salience
        dnc.usage_vector = torch.rand(64)
        dnc.memory_salience_map = torch.rand(64)
        
        # Perform consolidation
        dnc.consolidate_memory(consolidation_strength=0.5)
        
        # Check that consolidation weights were calculated
        assert dnc.consolidation_weights.shape == (64,)
        assert torch.all(dnc.consolidation_weights >= 0.0)
        assert torch.all(dnc.consolidation_weights <= 1.0)
    
    def test_fragmentation_analysis(self):
        """Test memory fragmentation analysis."""
        dnc = EnhancedDNCMemory(memory_size=64, word_size=16, enable_monitoring=False)
        
        # Initialize some usage patterns
        dnc.usage_vector = torch.rand(64)
        dnc.access_frequency = torch.rand(64)
        
        # Analyze fragmentation
        analysis = dnc.analyze_fragmentation()
        
        assert isinstance(analysis, dict)
        assert 'fragmentation_score' in analysis
        assert 'usage_variance' in analysis
        assert 'access_variance' in analysis
        assert 'average_fragmentation' in analysis
        
        assert analysis['fragmentation_score'] >= 0.0
        assert analysis['usage_variance'] >= 0.0
        assert analysis['access_variance'] >= 0.0
    
    def test_comprehensive_metrics(self):
        """Test comprehensive metrics generation."""
        dnc = EnhancedDNCMemory(memory_size=64, word_size=16, enable_monitoring=False)
        
        # Initialize some data
        dnc.usage_vector = torch.rand(64)
        dnc.memory_matrix = torch.randn(64, 16)
        dnc.memory_salience_map = torch.rand(64)
        
        # Get metrics
        metrics = dnc.get_comprehensive_metrics()
        
        assert isinstance(metrics, MemoryMetrics)
        assert 0.0 <= metrics.memory_utilization <= 1.0
        assert metrics.average_usage >= 0.0
        assert metrics.max_usage >= 0.0
        assert metrics.link_matrix_norm >= 0.0
        assert metrics.memory_diversity >= 0.0
        assert isinstance(metrics.timestamp, datetime)


class TestDNCIntegrationAPI:
    """Test the DNC Integration API."""
    
    @pytest.fixture
    def mock_integration(self):
        """Mock database integration."""
        mock = Mock()
        mock.log_system_event = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_cognitive_coordinator(self):
        """Mock cognitive coordinator."""
        mock = Mock()
        mock.initialize_all_subsystems = AsyncMock()
        mock.update_all_subsystems = AsyncMock()
        mock.get_all_subsystem_metrics = AsyncMock(return_value={})
        mock.cleanup_all_subsystems = AsyncMock()
        return mock
    
    @pytest.mark.asyncio
    async def test_api_initialization(self, mock_integration, mock_cognitive_coordinator):
        """Test API initialization."""
        with patch('src.memory.dnc_integration_api.get_system_integration', return_value=mock_integration), \
             patch('src.memory.dnc_integration_api.CognitiveCoordinator', return_value=mock_cognitive_coordinator):
            
            api = DNCIntegrationAPI(
                memory_size=128,
                word_size=32,
                enable_monitoring=True,
                enable_database_storage=True
            )
            
            result = await api.initialize()
            
            assert result == True
            assert api.session_id is not None
            assert api.operation_count == 0
            mock_cognitive_coordinator.initialize_all_subsystems.assert_called_once()
            mock_integration.log_system_event.assert_called()
    
    @pytest.mark.asyncio
    async def test_process_input(self, mock_integration, mock_cognitive_coordinator):
        """Test input processing through API."""
        with patch('src.memory.dnc_integration_api.get_system_integration', return_value=mock_integration), \
             patch('src.memory.dnc_integration_api.CognitiveCoordinator', return_value=mock_cognitive_coordinator):
            
            api = DNCIntegrationAPI(enable_monitoring=True, enable_database_storage=True)
            await api.initialize()
            
            # Test input processing
            input_data = torch.randn(2, 32)
            prev_reads = torch.randn(2, api.num_read_heads * api.word_size)
            context = {'test_context': True}
            
            result = await api.process_input(input_data, prev_reads, context=context)
            
            assert 'read_vectors' in result
            assert 'controller_output' in result
            assert 'new_controller_state' in result
            assert 'debug_info' in result
            assert 'processing_time' in result
            assert 'operation_success' in result
            assert 'session_id' in result
            assert 'operation_count' in result
            
            assert result['read_vectors'].shape == (2, api.num_read_heads * api.word_size)
            assert result['controller_output'].shape == (2, api.controller_size)
            assert result['operation_count'] == 1
    
    @pytest.mark.asyncio
    async def test_retrieve_memories(self, mock_integration, mock_cognitive_coordinator):
        """Test memory retrieval through API."""
        with patch('src.memory.dnc_integration_api.get_system_integration', return_value=mock_integration), \
             patch('src.memory.dnc_integration_api.CognitiveCoordinator', return_value=mock_cognitive_coordinator):
            
            api = DNCIntegrationAPI(enable_monitoring=True, enable_database_storage=True)
            await api.initialize()
            
            # Initialize some memory
            api.dnc.memory_matrix = torch.randn(api.memory_size, api.word_size)
            api.dnc.memory_salience_map = torch.rand(api.memory_size)
            
            # Test retrieval
            query = torch.randn(api.word_size)
            result = await api.retrieve_memories(query, salience_threshold=0.5, max_retrievals=3)
            
            assert 'retrieved_memories' in result
            assert 'memory_count' in result
            assert 'average_relevance' in result
            assert 'salience_threshold' in result
            assert 'processing_time' in result
            assert 'session_id' in result
            
            assert result['memory_count'] <= 3
            assert 0.0 <= result['average_relevance'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_update_memory_salience(self, mock_integration, mock_cognitive_coordinator):
        """Test memory salience update through API."""
        with patch('src.memory.dnc_integration_api.get_system_integration', return_value=mock_integration), \
             patch('src.memory.dnc_integration_api.CognitiveCoordinator', return_value=mock_cognitive_coordinator):
            
            api = DNCIntegrationAPI(enable_monitoring=True, enable_database_storage=True)
            await api.initialize()
            
            # Test salience update
            indices = [0, 1, 2]
            values = [0.8, 0.9, 0.7]
            result = await api.update_memory_salience(indices, values)
            
            assert 'updated_indices' in result
            assert 'updated_values' in result
            assert 'update_count' in result
            assert 'processing_time' in result
            assert 'session_id' in result
            
            assert result['updated_indices'] == indices
            assert result['updated_values'] == values
            assert result['update_count'] == len(indices)
    
    @pytest.mark.asyncio
    async def test_consolidate_memory(self, mock_integration, mock_cognitive_coordinator):
        """Test memory consolidation through API."""
        with patch('src.memory.dnc_integration_api.get_system_integration', return_value=mock_integration), \
             patch('src.memory.dnc_integration_api.CognitiveCoordinator', return_value=mock_cognitive_coordinator):
            
            api = DNCIntegrationAPI(enable_monitoring=True, enable_database_storage=True)
            await api.initialize()
            
            # Test consolidation
            result = await api.consolidate_memory(consolidation_strength=0.5)
            
            assert 'consolidation_strength' in result
            assert 'consolidation_quality' in result
            assert 'memory_utilization' in result
            assert 'memory_diversity' in result
            assert 'processing_time' in result
            assert 'session_id' in result
            
            assert result['consolidation_strength'] == 0.5
            assert 0.0 <= result['consolidation_quality'] <= 1.0
            assert 0.0 <= result['memory_utilization'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_analyze_fragmentation(self, mock_integration, mock_cognitive_coordinator):
        """Test fragmentation analysis through API."""
        with patch('src.memory.dnc_integration_api.get_system_integration', return_value=mock_integration), \
             patch('src.memory.dnc_integration_api.CognitiveCoordinator', return_value=mock_cognitive_coordinator):
            
            api = DNCIntegrationAPI(enable_monitoring=True, enable_database_storage=True)
            await api.initialize()
            
            # Test fragmentation analysis
            result = await api.analyze_fragmentation()
            
            assert 'fragmentation_analysis' in result
            assert 'processing_time' in result
            assert 'session_id' in result
            
            analysis = result['fragmentation_analysis']
            assert 'fragmentation_score' in analysis
            assert 'usage_variance' in analysis
            assert 'access_variance' in analysis
            assert 'average_fragmentation' in analysis
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_metrics(self, mock_integration, mock_cognitive_coordinator):
        """Test comprehensive metrics retrieval."""
        with patch('src.memory.dnc_integration_api.get_system_integration', return_value=mock_integration), \
             patch('src.memory.dnc_integration_api.CognitiveCoordinator', return_value=mock_cognitive_coordinator):
            
            api = DNCIntegrationAPI(enable_monitoring=True, enable_database_storage=True)
            await api.initialize()
            
            # Test metrics retrieval
            result = await api.get_comprehensive_metrics()
            
            assert 'dnc_metrics' in result
            assert 'cognitive_metrics' in result
            assert 'session_info' in result
            
            dnc_metrics = result['dnc_metrics']
            assert 'memory_utilization' in dnc_metrics
            assert 'average_usage' in dnc_metrics
            assert 'max_usage' in dnc_metrics
            assert 'operation_count' in dnc_metrics
            assert 'error_rate' in dnc_metrics
            
            session_info = result['session_info']
            assert 'session_id' in session_info
            assert 'session_start_time' in session_info
            assert 'operation_count' in session_info
            assert 'session_duration' in session_info
    
    @pytest.mark.asyncio
    async def test_cleanup(self, mock_integration, mock_cognitive_coordinator):
        """Test API cleanup."""
        with patch('src.memory.dnc_integration_api.get_system_integration', return_value=mock_integration), \
             patch('src.memory.dnc_integration_api.CognitiveCoordinator', return_value=mock_cognitive_coordinator):
            
            api = DNCIntegrationAPI(enable_monitoring=True, enable_database_storage=True)
            await api.initialize()
            
            # Test cleanup
            await api.cleanup()
            
            mock_cognitive_coordinator.cleanup_all_subsystems.assert_called_once()
            # Should have logged session completion
            assert mock_integration.log_system_event.call_count >= 2  # Initialization + cleanup
    
    def test_factory_function(self):
        """Test the factory function for creating DNC integration."""
        api = create_dnc_integration(
            memory_size=256,
            word_size=32,
            num_read_heads=2,
            num_write_heads=1,
            controller_size=128,
            enable_monitoring=False,
            enable_database_storage=False
        )
        
        assert isinstance(api, DNCIntegrationAPI)
        assert api.memory_size == 256
        assert api.word_size == 32
        assert api.num_read_heads == 2
        assert api.num_write_heads == 1
        assert api.controller_size == 128
        assert api.enable_monitoring == False
        assert api.enable_database_storage == False


class TestIntegrationCompatibility:
    """Test integration with existing systems."""
    
    def test_import_compatibility(self):
        """Test that all imports work correctly."""
        from src.memory import (
            DNCMemory, EnhancedDNCMemory, MemoryMetrics, 
            MemoryOperation, MemoryOperationType, DNCIntegrationAPI, create_dnc_integration
        )
        
        # Test that classes can be instantiated
        basic_dnc = DNCMemory()
        enhanced_dnc = EnhancedDNCMemory(enable_monitoring=False, enable_database_storage=False)
        api = create_dnc_integration(enable_monitoring=False, enable_database_storage=False)
        
        assert basic_dnc is not None
        assert enhanced_dnc is not None
        assert api is not None
    
    def test_memory_operation_type_enum(self):
        """Test MemoryOperationType enum."""
        from src.memory import MemoryOperationType
        
        assert MemoryOperationType.READ.value == "read"
        assert MemoryOperationType.WRITE.value == "write"
        assert MemoryOperationType.ERASE.value == "erase"
        assert MemoryOperationType.ALLOCATE.value == "allocate"
        assert MemoryOperationType.RETRIEVE.value == "retrieve"
        assert MemoryOperationType.CONSOLIDATE.value == "consolidate"
    
    def test_memory_metrics_dataclass(self):
        """Test MemoryMetrics dataclass."""
        from src.memory import MemoryMetrics
        from datetime import datetime
        
        metrics = MemoryMetrics(
            memory_utilization=0.5,
            average_usage=0.3,
            max_usage=0.8,
            link_matrix_norm=1.2,
            memory_diversity=0.7,
            read_efficiency=0.6,
            write_efficiency=0.4,
            retrieval_accuracy=0.9,
            consolidation_quality=0.8,
            fragmentation_level=0.2,
            salience_distribution=0.5,
            temporal_coherence=0.6,
            operation_count=100,
            error_rate=0.01,
            timestamp=datetime.now()
        )
        
        assert metrics.memory_utilization == 0.5
        assert metrics.average_usage == 0.3
        assert metrics.max_usage == 0.8
        assert metrics.operation_count == 100
        assert metrics.error_rate == 0.01


if __name__ == "__main__":
    # Run basic tests
    print("Running DNC Integration Tests...")
    
    # Test basic DNC
    print("Testing Enhanced DNC Memory...")
    test_dnc = TestEnhancedDNCMemory()
    test_dnc.test_initialization()
    test_dnc.test_forward_pass()
    test_dnc.test_memory_retrieval()
    test_dnc.test_salience_update()
    test_dnc.test_memory_consolidation()
    test_dnc.test_fragmentation_analysis()
    test_dnc.test_comprehensive_metrics()
    print(" Enhanced DNC Memory tests passed")
    
    # Test integration compatibility
    print("Testing Integration Compatibility...")
    test_compat = TestIntegrationCompatibility()
    test_compat.test_import_compatibility()
    test_compat.test_memory_operation_type_enum()
    test_compat.test_memory_metrics_dataclass()
    print(" Integration compatibility tests passed")
    
    print("All basic tests passed! Use pytest for full async testing.")
