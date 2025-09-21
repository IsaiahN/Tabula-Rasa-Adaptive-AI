#!/usr/bin/env python3
"""
Comprehensive test suite for the Enhanced Frame Analysis System.

This module tests the enhanced frame analysis capabilities including:
- Advanced object detection and recognition
- Multi-scale pattern recognition
- Spatial and temporal reasoning
- Attention mechanisms
- Real-time processing
- Integration with existing systems
"""

import unittest
import numpy as np
import time
from typing import Dict, List, Any, Optional
import tempfile
import os

# Import the enhanced frame analysis system
from src.vision.enhanced_frame_analyzer import (
    EnhancedFrameAnalyzer, 
    EnhancedFrameAnalysisConfig, 
    AnalysisMode,
    VisualPattern,
    FrameAnalysisResult,
    create_enhanced_frame_analyzer
)
from src.vision.frame_analysis_integration import (
    FrameAnalysisIntegration,
    FrameAnalysisIntegrationConfig,
    create_frame_analysis_integration
)


class TestEnhancedFrameAnalyzer(unittest.TestCase):
    """Test the EnhancedFrameAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = EnhancedFrameAnalysisConfig(
            analysis_mode=AnalysisMode.ENHANCED,
            enable_object_detection=True,
            enable_pattern_recognition=True,
            enable_spatial_analysis=True,
            enable_attention_mechanisms=True,
            enable_visual_reasoning=True,
            enable_change_detection=True
        )
        self.analyzer = EnhancedFrameAnalyzer(self.config)
        
        # Create test frames
        self.test_frame_rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        self.test_frame_gray = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        self.test_frame_2d = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
        self.test_frame_3d = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
    
    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertIsNotNone(self.analyzer.config)
        self.assertEqual(self.analyzer.config.analysis_mode, AnalysisMode.ENHANCED)
        self.assertTrue(self.analyzer.config.enable_object_detection)
        self.assertTrue(self.analyzer.config.enable_pattern_recognition)
        self.assertTrue(self.analyzer.config.enable_spatial_analysis)
        self.assertTrue(self.analyzer.config.enable_attention_mechanisms)
        self.assertTrue(self.analyzer.config.enable_visual_reasoning)
        self.assertTrue(self.analyzer.config.enable_change_detection)
    
    def test_analyze_rgb_frame(self):
        """Test analysis of RGB frame."""
        result = self.analyzer.analyze_frame(self.test_frame_rgb, "test_game")
        
        self.assertIsInstance(result, FrameAnalysisResult)
        self.assertEqual(result.analysis_mode, AnalysisMode.ENHANCED)
        self.assertIsInstance(result.objects, list)
        self.assertIsInstance(result.patterns, list)
        self.assertIsInstance(result.changes, list)
        self.assertIsInstance(result.spatial_analysis, dict)
        self.assertIsInstance(result.reasoning_results, list)
        self.assertIsInstance(result.insights, list)
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 1.0)
        self.assertGreaterEqual(result.quality_score, 0.0)
        self.assertLessEqual(result.quality_score, 1.0)
        self.assertGreater(result.processing_time, 0.0)
    
    def test_analyze_gray_frame(self):
        """Test analysis of grayscale frame."""
        result = self.analyzer.analyze_frame(self.test_frame_gray, "test_game")
        
        self.assertIsInstance(result, FrameAnalysisResult)
        self.assertEqual(result.analysis_mode, AnalysisMode.ENHANCED)
        self.assertIsInstance(result.objects, list)
        self.assertIsInstance(result.patterns, list)
        self.assertIsInstance(result.changes, list)
        self.assertIsInstance(result.spatial_analysis, dict)
        self.assertIsInstance(result.reasoning_results, list)
        self.assertIsInstance(result.insights, list)
    
    def test_analyze_2d_list_frame(self):
        """Test analysis of 2D list frame."""
        result = self.analyzer.analyze_frame(self.test_frame_2d, "test_game")
        
        self.assertIsInstance(result, FrameAnalysisResult)
        self.assertEqual(result.analysis_mode, AnalysisMode.ENHANCED)
        self.assertIsInstance(result.objects, list)
        self.assertIsInstance(result.patterns, list)
        self.assertIsInstance(result.changes, list)
        self.assertIsInstance(result.spatial_analysis, dict)
        self.assertIsInstance(result.reasoning_results, list)
        self.assertIsInstance(result.insights, list)
    
    def test_analyze_3d_list_frame(self):
        """Test analysis of 3D list frame."""
        result = self.analyzer.analyze_frame(self.test_frame_3d, "test_game")
        
        self.assertIsInstance(result, FrameAnalysisResult)
        self.assertEqual(result.analysis_mode, AnalysisMode.ENHANCED)
        self.assertIsInstance(result.objects, list)
        self.assertIsInstance(result.patterns, list)
        self.assertIsInstance(result.changes, list)
        self.assertIsInstance(result.spatial_analysis, dict)
        self.assertIsInstance(result.reasoning_results, list)
        self.assertIsInstance(result.insights, list)
    
    def test_frame_preprocessing(self):
        """Test frame preprocessing functionality."""
        # Test RGB frame preprocessing
        processed_rgb = self.analyzer._preprocess_frame(self.test_frame_rgb)
        self.assertIsInstance(processed_rgb, np.ndarray)
        self.assertEqual(len(processed_rgb.shape), 3)
        self.assertEqual(processed_rgb.dtype, np.uint8)
        
        # Test grayscale frame preprocessing
        processed_gray = self.analyzer._preprocess_frame(self.test_frame_gray)
        self.assertIsInstance(processed_gray, np.ndarray)
        self.assertEqual(len(processed_gray.shape), 3)
        self.assertEqual(processed_gray.dtype, np.uint8)
        
        # Test 2D list preprocessing
        processed_2d = self.analyzer._preprocess_frame(self.test_frame_2d)
        self.assertIsInstance(processed_2d, np.ndarray)
        self.assertEqual(len(processed_2d.shape), 3)
        self.assertEqual(processed_2d.dtype, np.uint8)
    
    def test_pattern_recognition(self):
        """Test pattern recognition functionality."""
        # Create a frame with a simple pattern
        pattern_frame = np.zeros((32, 32, 3), dtype=np.uint8)
        pattern_frame[10:20, 10:20] = [255, 255, 255]  # White square
        
        result = self.analyzer.analyze_frame(pattern_frame, "pattern_test")
        
        self.assertIsInstance(result.patterns, list)
        # Patterns might be empty if no patterns are detected, which is fine
    
    def test_spatial_analysis(self):
        """Test spatial analysis functionality."""
        result = self.analyzer.analyze_frame(self.test_frame_rgb, "spatial_test")
        
        self.assertIsInstance(result.spatial_analysis, dict)
        self.assertIsInstance(result.spatial_relationships, list)
    
    def test_attention_mechanisms(self):
        """Test attention mechanism functionality."""
        result = self.analyzer.analyze_frame(self.test_frame_rgb, "attention_test")
        
        # Attention map might be None if attention mechanisms are disabled
        if result.attention_map is not None:
            self.assertIsInstance(result.attention_map, np.ndarray)
        
        self.assertIsInstance(result.focus_areas, list)
    
    def test_visual_reasoning(self):
        """Test visual reasoning functionality."""
        result = self.analyzer.analyze_frame(self.test_frame_rgb, "reasoning_test")
        
        self.assertIsInstance(result.reasoning_results, list)
        self.assertIsInstance(result.insights, list)
    
    def test_change_detection(self):
        """Test change detection functionality."""
        # Analyze first frame
        result1 = self.analyzer.analyze_frame(self.test_frame_rgb, "change_test")
        
        # Create a slightly different frame
        modified_frame = self.test_frame_rgb.copy()
        modified_frame[10:20, 10:20] = [255, 0, 0]  # Red square
        
        # Analyze second frame
        result2 = self.analyzer.analyze_frame(modified_frame, "change_test")
        
        self.assertIsInstance(result1.changes, list)
        self.assertIsInstance(result2.changes, list)
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        result = self.analyzer.analyze_frame(self.test_frame_rgb, "confidence_test")
        
        self.assertIsInstance(result.confidence_score, float)
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 1.0)
    
    def test_quality_calculation(self):
        """Test quality score calculation."""
        result = self.analyzer.analyze_frame(self.test_frame_rgb, "quality_test")
        
        self.assertIsInstance(result.quality_score, float)
        self.assertGreaterEqual(result.quality_score, 0.0)
        self.assertLessEqual(result.quality_score, 1.0)
    
    def test_performance_stats(self):
        """Test performance statistics tracking."""
        # Analyze a few frames
        for i in range(3):
            self.analyzer.analyze_frame(self.test_frame_rgb, f"perf_test_{i}")
        
        stats = self.analyzer.get_performance_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_frames_processed', stats)
        self.assertIn('average_processing_time', stats)
        self.assertIn('cache_hit_rate', stats)
        self.assertIn('detection_accuracy', stats)
        
        self.assertGreater(stats['total_frames_processed'], 0)
        self.assertGreaterEqual(stats['average_processing_time'], 0.0)
    
    def test_reset_for_new_game(self):
        """Test reset functionality for new game."""
        # Analyze some frames
        self.analyzer.analyze_frame(self.test_frame_rgb, "reset_test")
        
        # Reset for new game
        self.analyzer.reset_for_new_game("reset_test")
        
        # Check that frame history is cleared
        history = self.analyzer.get_frame_history("reset_test")
        self.assertEqual(len(history), 0)
        
        # Check that pattern tracking is cleared
        patterns = self.analyzer.get_pattern_history("reset_test")
        self.assertEqual(len(patterns), 0)
    
    def test_frame_history_tracking(self):
        """Test frame history tracking."""
        # Analyze multiple frames
        for i in range(5):
            frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            self.analyzer.analyze_frame(frame, "history_test")
        
        # Check frame history
        history = self.analyzer.get_frame_history("history_test")
        self.assertEqual(len(history), 5)
        
        for frame in history:
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(len(frame.shape), 3)
    
    def test_pattern_tracking(self):
        """Test pattern tracking across frames."""
        # Analyze multiple frames
        for i in range(3):
            frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            self.analyzer.analyze_frame(frame, "pattern_tracking_test")
        
        # Check pattern history
        patterns = self.analyzer.get_pattern_history("pattern_tracking_test")
        self.assertIsInstance(patterns, list)
        
        for pattern in patterns:
            self.assertIsInstance(pattern, VisualPattern)
            self.assertIsInstance(pattern.pattern_id, str)
            self.assertIsInstance(pattern.pattern_type, str)
            self.assertIsInstance(pattern.description, str)
            self.assertIsInstance(pattern.confidence, float)
            self.assertIsInstance(pattern.bounding_box, tuple)
            self.assertIsInstance(pattern.features, dict)


class TestFrameAnalysisIntegration(unittest.TestCase):
    """Test the FrameAnalysisIntegration class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = FrameAnalysisIntegrationConfig(
            enable_enhanced_analysis=True,
            enable_legacy_compatibility=True,
            enable_database_logging=False,  # Disable for testing
            enable_cognitive_monitoring=False  # Disable for testing
        )
        self.integration = FrameAnalysisIntegration(self.config)
        
        # Create test frame
        self.test_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    def test_initialization(self):
        """Test integration initialization."""
        self.assertIsNotNone(self.integration.config)
        self.assertTrue(self.integration.config.enable_enhanced_analysis)
        self.assertTrue(self.integration.config.enable_legacy_compatibility)
        self.assertFalse(self.integration.config.enable_database_logging)
        self.assertFalse(self.integration.config.enable_cognitive_monitoring)
    
    def test_enhanced_analysis(self):
        """Test enhanced analysis mode."""
        result = self.integration.analyze_frame(
            self.test_frame, 
            "integration_test",
            use_enhanced=True
        )
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("analysis_type"), "enhanced")
        self.assertIn("analysis_mode", result)
        self.assertIn("objects", result)
        self.assertIn("patterns", result)
        self.assertIn("spatial_analysis", result)
        self.assertIn("reasoning_results", result)
        self.assertIn("insights", result)
        self.assertIn("processing_time", result)
        self.assertIn("confidence_score", result)
        self.assertIn("quality_score", result)
    
    def test_legacy_analysis(self):
        """Test legacy analysis mode."""
        result = self.integration.analyze_frame(
            self.test_frame, 
            "integration_test",
            use_enhanced=False
        )
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("analysis_type"), "legacy")
        self.assertIn("analysis_mode", result)
    
    def test_adaptive_mode_selection(self):
        """Test adaptive mode selection."""
        # Test with complex frame (should use enhanced)
        complex_frame = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = self.integration.analyze_frame(complex_frame, "adaptive_test")
        
        self.assertIsInstance(result, dict)
        self.assertIn("analysis_type", result)
    
    def test_analysis_stats(self):
        """Test analysis statistics tracking."""
        # Perform some analyses
        for i in range(3):
            self.integration.analyze_frame(self.test_frame, f"stats_test_{i}")
        
        stats = self.integration.get_analysis_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_analyses', stats)
        self.assertIn('enhanced_analyses', stats)
        self.assertIn('legacy_analyses', stats)
        self.assertIn('average_processing_time', stats)
        self.assertIn('success_rate', stats)
        self.assertIn('error_count', stats)
        
        self.assertGreater(stats['total_analyses'], 0)
        self.assertGreaterEqual(stats['success_rate'], 0.0)
        self.assertLessEqual(stats['success_rate'], 1.0)
    
    def test_enhanced_analyzer_stats(self):
        """Test enhanced analyzer statistics."""
        stats = self.integration.get_enhanced_analyzer_stats()
        
        self.assertIsInstance(stats, dict)
        # Stats might be empty if enhanced analyzer is not used
    
    def test_reset_for_new_game(self):
        """Test reset functionality for new game."""
        # Perform some analyses
        self.integration.analyze_frame(self.test_frame, "reset_test")
        
        # Reset for new game
        self.integration.reset_for_new_game("reset_test")
        
        # This should not raise an exception
        self.assertTrue(True)
    
    def test_pattern_history(self):
        """Test pattern history retrieval."""
        # Perform some analyses
        for i in range(3):
            self.integration.analyze_frame(self.test_frame, "pattern_test")
        
        # Get pattern history
        patterns = self.integration.get_pattern_history("pattern_test")
        
        self.assertIsInstance(patterns, list)
        # Patterns might be empty if no patterns are detected
    
    def test_frame_history(self):
        """Test frame history retrieval."""
        # Perform some analyses
        for i in range(3):
            self.integration.analyze_frame(self.test_frame, "frame_test")
        
        # Get frame history
        history = self.integration.get_frame_history("frame_test")
        
        self.assertIsInstance(history, list)
        # History might be empty if enhanced analyzer is not used


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions."""
    
    def test_create_enhanced_frame_analyzer(self):
        """Test enhanced frame analyzer factory function."""
        analyzer = create_enhanced_frame_analyzer()
        
        self.assertIsInstance(analyzer, EnhancedFrameAnalyzer)
        self.assertIsNotNone(analyzer.config)
    
    def test_create_enhanced_frame_analyzer_with_config(self):
        """Test enhanced frame analyzer factory function with config."""
        config = EnhancedFrameAnalysisConfig(
            analysis_mode=AnalysisMode.REALTIME,
            enable_object_detection=False
        )
        analyzer = create_enhanced_frame_analyzer(config)
        
        self.assertIsInstance(analyzer, EnhancedFrameAnalyzer)
        self.assertEqual(analyzer.config.analysis_mode, AnalysisMode.REALTIME)
        self.assertFalse(analyzer.config.enable_object_detection)
    
    def test_create_frame_analysis_integration(self):
        """Test frame analysis integration factory function."""
        integration = create_frame_analysis_integration()
        
        self.assertIsInstance(integration, FrameAnalysisIntegration)
        self.assertIsNotNone(integration.config)
    
    def test_create_frame_analysis_integration_with_config(self):
        """Test frame analysis integration factory function with config."""
        config = FrameAnalysisIntegrationConfig(
            enable_enhanced_analysis=False,
            enable_legacy_compatibility=True
        )
        integration = create_frame_analysis_integration(config)
        
        self.assertIsInstance(integration, FrameAnalysisIntegration)
        self.assertFalse(integration.config.enable_enhanced_analysis)
        self.assertTrue(integration.config.enable_legacy_compatibility)


class TestIntegration(unittest.TestCase):
    """Test integration between components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.integration = create_frame_analysis_integration()
        self.test_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    def test_end_to_end_analysis(self):
        """Test complete end-to-end analysis workflow."""
        # Perform analysis
        result = self.integration.analyze_frame(self.test_frame, "e2e_test")
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn("analysis_type", result)
        self.assertIn("processing_time", result)
        
        # Verify performance metrics
        stats = self.integration.get_analysis_stats()
        self.assertGreater(stats['total_analyses'], 0)
        self.assertGreaterEqual(stats['success_rate'], 0.0)
    
    def test_multiple_game_analysis(self):
        """Test analysis across multiple games."""
        games = ["game1", "game2", "game3"]
        
        for game_id in games:
            result = self.integration.analyze_frame(self.test_frame, game_id)
            self.assertIsInstance(result, dict)
            self.assertIn("analysis_type", result)
        
        # Verify all games are tracked
        stats = self.integration.get_analysis_stats()
        self.assertGreater(stats['total_analyses'], 0)
    
    def test_error_handling(self):
        """Test error handling with invalid input."""
        # Test with invalid frame
        invalid_frame = "not_a_frame"
        result = self.integration.analyze_frame(invalid_frame, "error_test")
        
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
        # The system falls back to legacy mode when there's an error
        self.assertEqual(result["analysis_type"], "legacy")
    
    def test_performance_under_load(self):
        """Test performance under multiple concurrent analyses."""
        start_time = time.time()
        
        # Perform multiple analyses
        for i in range(10):
            frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            result = self.integration.analyze_frame(frame, f"load_test_{i}")
            self.assertIsInstance(result, dict)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify reasonable performance (should complete in reasonable time)
        self.assertLess(total_time, 10.0)  # Should complete within 10 seconds
        
        # Verify all analyses were processed
        stats = self.integration.get_analysis_stats()
        self.assertGreaterEqual(stats['total_analyses'], 10)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
