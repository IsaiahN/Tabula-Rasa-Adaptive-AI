"""
Vision Package

Modular computer vision components for ARC puzzle analysis.
"""

from .object_detection import DetectedObject, ObjectDetector
from .spatial_analysis import SpatialRelationship, SpatialAnalyzer
from .pattern_recognition import PatternInfo, PatternRecognizer
from .change_detection import ChangeInfo, ChangeDetector
from .feature_extraction import ActionableTarget, FeatureExtractor
from .position_tracking import PositionTracker
from .movement_detection import MovementDetector
from .pattern_analysis import PatternAnalyzer
from .frame_processing import FrameProcessor
from .frame_analyzer import FrameAnalyzer

# Enhanced frame analysis system
from .enhanced_frame_analyzer import (
    EnhancedFrameAnalyzer,
    EnhancedFrameAnalysisConfig,
    AnalysisMode,
    VisualPattern,
    FrameAnalysisResult,
    create_enhanced_frame_analyzer
)
from .frame_analysis_integration import (
    FrameAnalysisIntegration,
    FrameAnalysisIntegrationConfig,
    create_frame_analysis_integration
)

# Enhanced vision components
from .enhanced import (
    AdvancedObjectDetector,
    DetectionConfig,
    DetectionMethod,
    Detection,
    RealTimeProcessor,
    ProcessingConfig,
    ProcessingMode,
    ProcessedFrame,
    AttentionMechanism,
    AttentionConfig,
    AttentionType,
    AttentionResult,
    VisualReasoningEngine,
    ReasoningConfig,
    ReasoningType,
    ReasoningResult,
    SpatialRelation
)

__all__ = [
    # Core vision components
    'DetectedObject',
    'ObjectDetector', 
    'SpatialRelationship',
    'SpatialAnalyzer',
    'PatternInfo',
    'PatternRecognizer',
    'ChangeInfo',
    'ChangeDetector',
    'ActionableTarget',
    'FeatureExtractor',
    'PositionTracker',
    'MovementDetector',
    'PatternAnalyzer',
    'FrameProcessor',
    'FrameAnalyzer',
    
    # Enhanced frame analysis system
    'EnhancedFrameAnalyzer',
    'EnhancedFrameAnalysisConfig',
    'AnalysisMode',
    'VisualPattern',
    'FrameAnalysisResult',
    'create_enhanced_frame_analyzer',
    'FrameAnalysisIntegration',
    'FrameAnalysisIntegrationConfig',
    'create_frame_analysis_integration',
    
    # Enhanced vision components
    'AdvancedObjectDetector',
    'DetectionConfig',
    'DetectionMethod',
    'Detection',
    'RealTimeProcessor',
    'ProcessingConfig',
    'ProcessingMode',
    'ProcessedFrame',
    'AttentionMechanism',
    'AttentionConfig',
    'AttentionType',
    'AttentionResult',
    'VisualReasoningEngine',
    'ReasoningConfig',
    'ReasoningType',
    'ReasoningResult',
    'SpatialRelation'
]
