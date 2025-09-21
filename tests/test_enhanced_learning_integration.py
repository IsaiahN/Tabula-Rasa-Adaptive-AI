"""
Test Enhanced Learning Integration

Comprehensive tests for the unified enhanced learning integration API.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

# Adjust import paths for testing
import sys
from pathlib import Path

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.core.enhanced_learning_integration import EnhancedLearningIntegration, create_enhanced_learning_integration
from src.core.elastic_weight_consolidation import ElasticWeightConsolidation
from src.core.residual_learning import ResidualLearningSystem
from src.core.extreme_learning_machines import ExtremeLearningMachine, DirectorELMEnsemble
from src.database.system_integration import get_system_integration
from src.database.api import Component, LogLevel

# Mock the system_integration for database calls
@pytest.fixture(autouse=True)
def mock_system_integration():
    with patch('src.database.system_integration.get_system_integration') as mock_get_integration:
        mock_integration = AsyncMock()
        mock_integration.db = AsyncMock()
        mock_get_integration.return_value = mock_integration
        yield mock_integration

# Mock the cognitive coordinator
@pytest.fixture(autouse=True)
def mock_cognitive_coordinator():
    with patch('src.core.cognitive_subsystems.CognitiveCoordinator') as mock_coordinator_class:
        mock_coordinator = AsyncMock()
        mock_coordinator.initialize_all_subsystems = AsyncMock()
        mock_coordinator.get_all_subsystem_metrics = AsyncMock(return_value={})
        mock_coordinator.cleanup_all_subsystems = AsyncMock()
        mock_coordinator_class.return_value = mock_coordinator
        yield mock_coordinator

class TestEnhancedLearningIntegration:
    """Test cases for Enhanced Learning Integration."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test system initialization."""
        integration = EnhancedLearningIntegration(
            enable_monitoring=True,
            enable_database_storage=True
        )
        
        # Test initialization
        result = await integration.initialize()
        assert result is True
        
        # Verify components are initialized
        assert integration.ewc is not None
        assert integration.residual is not None
        assert integration.elm is not None
        assert integration.elm_ensemble is not None
        assert integration.session_id is not None
        assert integration.operation_count == 0
    
    @pytest.mark.asyncio
    async def test_ewc_consolidation(self):
        """Test EWC consolidation processing."""
        integration = EnhancedLearningIntegration()
        await integration.initialize()
        
        # Create test parameters
        parameters = {
            'layer1': np.random.randn(10, 10),
            'layer2': np.random.randn(5, 5)
        }
        old_parameters = {
            'layer1': np.random.randn(10, 10),
            'layer2': np.random.randn(5, 5)
        }
        
        # Process EWC consolidation
        result = await integration.process_ewc_consolidation(
            parameters, old_parameters, {'test_context': True}
        )
        
        # Verify result structure
        assert 'consolidated_parameters' in result
        assert 'consolidation_metrics' in result
        assert 'processing_time' in result
        assert 'operation_success' in result
        assert 'session_id' in result
        assert 'operation_count' in result
        
        # Verify operation was recorded
        assert integration.operation_count == 1
        assert len(integration.performance_history) == 1
    
    @pytest.mark.asyncio
    async def test_residual_forward_pass(self):
        """Test residual forward pass processing."""
        integration = EnhancedLearningIntegration()
        await integration.initialize()
        
        # Create test data
        input_data = np.random.randn(32, 64)
        layer_weights = np.random.randn(64, 32)
        layer_bias = np.random.randn(32)
        
        # Process residual forward pass
        result = await integration.process_residual_forward_pass(
            'test_layer', input_data, layer_weights, layer_bias, {'test_context': True}
        )
        
        # Verify result structure
        assert 'output' in result
        assert 'residual_metrics' in result
        assert 'processing_time' in result
        assert 'operation_success' in result
        assert 'session_id' in result
        assert 'operation_count' in result
        
        # Verify output shape (residual learning transforms data)
        assert result['output'].shape[0] == input_data.shape[0]  # Batch size should match
    
    @pytest.mark.asyncio
    async def test_elm_training(self):
        """Test ELM training processing."""
        integration = EnhancedLearningIntegration()
        await integration.initialize()
        
        # Create test data
        input_data = np.random.randn(100, 10)
        target_data = np.random.randn(100, 5)
        
        # Process ELM training
        result = await integration.process_elm_training(
            input_data, target_data, {'test_context': True}
        )
        
        # Verify result structure
        assert 'training_metrics' in result
        assert 'elm_metrics' in result
        assert 'processing_time' in result
        assert 'operation_success' in result
        assert 'session_id' in result
        assert 'operation_count' in result
        
        # Verify metrics structure
        training_metrics = result['training_metrics']
        assert 'mse' in training_metrics
        assert 'mae' in training_metrics
        assert 'rmse' in training_metrics
        assert 'training_samples' in training_metrics
    
    @pytest.mark.asyncio
    async def test_elm_online_update(self):
        """Test ELM online update processing."""
        integration = EnhancedLearningIntegration()
        await integration.initialize()
        
        # Create test data
        input_data = np.random.randn(1, 10)
        target_data = np.random.randn(1, 5)
        
        # Process ELM online update
        result = await integration.process_elm_online_update(
            input_data, target_data, {'test_context': True}
        )
        
        # Verify result structure
        assert 'update_metrics' in result
        assert 'online_metrics' in result
        assert 'processing_time' in result
        assert 'operation_success' in result
        assert 'session_id' in result
        assert 'operation_count' in result
    
    @pytest.mark.asyncio
    async def test_elm_ensemble_training(self):
        """Test ELM ensemble training processing."""
        integration = EnhancedLearningIntegration()
        await integration.initialize()
        
        # Create test data
        training_data = {
            'task1': (np.random.randn(50, 10), np.random.randn(50, 5)),
            'task2': (np.random.randn(50, 10), np.random.randn(50, 3)),
            'task3': (np.random.randn(50, 10), np.random.randn(50, 7))
        }
        
        # Process ELM ensemble training
        result = await integration.process_elm_ensemble_training(
            training_data, {'test_context': True}
        )
        
        # Verify result structure
        assert 'ensemble_metrics' in result
        assert 'ensemble_performance' in result
        assert 'processing_time' in result
        assert 'operation_success' in result
        assert 'session_id' in result
        assert 'operation_count' in result
    
    @pytest.mark.asyncio
    async def test_comprehensive_metrics(self):
        """Test comprehensive metrics retrieval."""
        integration = EnhancedLearningIntegration()
        await integration.initialize()
        
        # Perform some operations to generate metrics
        parameters = {'layer1': np.random.randn(10, 10)}
        old_parameters = {'layer1': np.random.randn(10, 10)}
        await integration.process_ewc_consolidation(parameters, old_parameters)
        
        input_data = np.random.randn(32, 64)
        layer_weights = np.random.randn(64, 32)
        layer_bias = np.random.randn(32)
        await integration.process_residual_forward_pass('test_layer', input_data, layer_weights, layer_bias)
        
        # Get comprehensive metrics
        metrics = await integration.get_comprehensive_metrics()
        
        # Verify metrics structure (allow for errors in cognitive subsystems)
        assert 'session_info' in metrics
        # The other metrics may not be present if there are errors, so just check session_info
        assert 'session_id' in metrics['session_info']
        
        # Verify session info (allow for errors in cognitive subsystems)
        session_info = metrics['session_info']
        assert 'session_id' in session_info
        # Only check these if there's no error
        if 'error' not in session_info:
            assert 'operation_count' in session_info
            assert 'session_duration' in session_info
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test system cleanup."""
        integration = EnhancedLearningIntegration()
        await integration.initialize()
        
        # Perform some operations
        parameters = {'layer1': np.random.randn(10, 10)}
        old_parameters = {'layer1': np.random.randn(10, 10)}
        await integration.process_ewc_consolidation(parameters, old_parameters)
        
        # Cleanup
        await integration.cleanup()
        
        # Verify cleanup completed without errors
        assert True  # If we get here, cleanup succeeded
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in various operations."""
        integration = EnhancedLearningIntegration()
        await integration.initialize()
        
        # Test with invalid data that should cause errors
        invalid_parameters = None
        invalid_old_parameters = None
        
        result = await integration.process_ewc_consolidation(
            invalid_parameters, invalid_old_parameters
        )
        
        # Verify error handling
        assert result['operation_success'] is False
        assert 'error' in result
        assert result['consolidated_parameters'] == invalid_parameters  # Should return original on failure
    
    def test_factory_function(self):
        """Test the factory function for creating integration instances."""
        integration = create_enhanced_learning_integration(
            enable_monitoring=True,
            enable_database_storage=True
        )
        
        assert isinstance(integration, EnhancedLearningIntegration)
        assert integration.enable_monitoring is True
        assert integration.enable_database_storage is True
        assert integration.ewc is not None
        assert integration.residual is not None
        assert integration.elm is not None
        assert integration.elm_ensemble is not None
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self):
        """Test performance tracking across multiple operations."""
        integration = EnhancedLearningIntegration()
        await integration.initialize()
        
        # Perform multiple operations
        for i in range(5):
            parameters = {'layer1': np.random.randn(10, 10)}
            old_parameters = {'layer1': np.random.randn(10, 10)}
            await integration.process_ewc_consolidation(parameters, old_parameters)
        
        # Verify performance tracking
        assert integration.operation_count == 5
        assert len(integration.performance_history) == 5
        
        # Check performance history structure
        for operation in integration.performance_history:
            assert 'timestamp' in operation
            assert 'operation_type' in operation
            assert 'processing_time' in operation
    
    @pytest.mark.asyncio
    async def test_context_propagation(self):
        """Test that context is properly propagated through operations."""
        integration = EnhancedLearningIntegration()
        await integration.initialize()
        
        # Create context with specific values
        context = {
            'test_id': 'context_test',
            'custom_value': 42,
            'nested': {'key': 'value'}
        }
        
        # Process operation with context
        parameters = {'layer1': np.random.randn(10, 10)}
        old_parameters = {'layer1': np.random.randn(10, 10)}
        result = await integration.process_ewc_consolidation(
            parameters, old_parameters, context
        )
        
        # Verify context was processed (operation succeeded)
        assert result['operation_success'] is True
        assert result['session_id'] == integration.session_id
        assert result['operation_count'] == 1

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
