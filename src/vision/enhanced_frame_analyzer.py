"""
Enhanced Frame Analysis System with Advanced Visual Pattern Recognition

This system provides comprehensive visual analysis capabilities including:
- Advanced object detection and recognition
- Multi-scale pattern recognition
- Spatial and temporal reasoning
- Attention mechanisms for focus optimization
- Real-time visual processing
- Cross-frame pattern tracking
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque
import time

# Import enhanced components with fallback
try:
    from .enhanced.advanced_detection import AdvancedObjectDetector, DetectionConfig, DetectionMethod
except ImportError:
    AdvancedObjectDetector = None
    DetectionConfig = None
    DetectionMethod = None

try:
    from .enhanced.real_time_processing import RealTimeProcessor, ProcessingConfig, ProcessingMode
except ImportError:
    RealTimeProcessor = None
    ProcessingConfig = None
    ProcessingMode = None

try:
    from .enhanced.attention_mechanisms import AttentionMechanism, AttentionConfig, AttentionType
except ImportError:
    AttentionMechanism = None
    AttentionConfig = None
    AttentionType = None

try:
    from .enhanced.visual_reasoning import VisualReasoningEngine, ReasoningConfig, ReasoningType
except ImportError:
    VisualReasoningEngine = None
    ReasoningConfig = None
    ReasoningType = None

try:
    from .pattern_recognition.recognizer import PatternRecognizer, PatternInfo
except ImportError:
    PatternRecognizer = None
    PatternInfo = None

try:
    from .spatial_analysis.analyzer import SpatialAnalyzer
except ImportError:
    SpatialAnalyzer = None

try:
    from .object_detection.detector import ObjectDetector
except ImportError:
    ObjectDetector = None

try:
    from .change_detection.detector import ChangeDetector
except ImportError:
    ChangeDetector = None

logger = logging.getLogger(__name__)


class AnalysisMode(Enum):
    """Available analysis modes."""
    BASIC = "basic"
    ENHANCED = "enhanced"
    REALTIME = "realtime"
    DEEP = "deep"
    ADAPTIVE = "adaptive"


@dataclass
class EnhancedFrameAnalysisConfig:
    """Configuration for enhanced frame analysis."""
    analysis_mode: AnalysisMode = AnalysisMode.ENHANCED
    enable_object_detection: bool = True
    enable_pattern_recognition: bool = True
    enable_spatial_analysis: bool = True
    enable_temporal_tracking: bool = True
    enable_attention_mechanisms: bool = True
    enable_visual_reasoning: bool = True
    enable_change_detection: bool = True
    
    # Performance settings
    max_frame_history: int = 50
    processing_timeout: float = 0.1  # seconds
    enable_caching: bool = True
    cache_ttl: int = 300  # seconds
    
    # Detection settings
    detection_confidence_threshold: float = 0.6
    pattern_confidence_threshold: float = 0.7
    spatial_resolution: int = 32
    temporal_window: int = 10
    
    # Attention settings
    attention_focus_areas: int = 5
    attention_decay_rate: float = 0.95
    attention_boost_threshold: float = 0.8


@dataclass
class VisualPattern:
    """Represents a detected visual pattern."""
    pattern_id: str
    pattern_type: str  # 'object', 'shape', 'texture', 'motion', 'spatial'
    description: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    features: Dict[str, Any]
    temporal_stability: float = 0.0
    spatial_relationships: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameAnalysisResult:
    """Result of enhanced frame analysis."""
    frame_id: str
    timestamp: float
    analysis_mode: AnalysisMode
    
    # Detection results
    objects: List[Dict[str, Any]] = field(default_factory=list)
    patterns: List[VisualPattern] = field(default_factory=list)
    changes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Spatial analysis
    spatial_analysis: Dict[str, Any] = field(default_factory=dict)
    spatial_relationships: List[Dict[str, Any]] = field(default_factory=list)
    
    # Attention and focus
    attention_map: Optional[np.ndarray] = None
    focus_areas: List[Dict[str, Any]] = field(default_factory=list)
    
    # Visual reasoning
    reasoning_results: List[Dict[str, Any]] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    
    # Performance metrics
    processing_time: float = 0.0
    confidence_score: float = 0.0
    quality_score: float = 0.0
    
    # Metadata
    frame_quality: Dict[str, Any] = field(default_factory=dict)
    temporal_context: Dict[str, Any] = field(default_factory=dict)


class EnhancedFrameAnalyzer:
    """
    Enhanced frame analysis system with advanced visual pattern recognition.
    
    This system provides comprehensive visual analysis capabilities including:
    - Multi-scale object detection and recognition
    - Advanced pattern recognition across spatial and temporal dimensions
    - Attention mechanisms for focus optimization
    - Visual reasoning and inference
    - Real-time processing optimization
    - Cross-frame pattern tracking and temporal analysis
    """
    
    def __init__(self, config: Optional[EnhancedFrameAnalysisConfig] = None):
        self.config = config or EnhancedFrameAnalysisConfig()
        
        # Initialize core components
        self._initialize_detection_systems()
        self._initialize_analysis_engines()
        self._initialize_tracking_systems()
        
        # Frame history and caching
        self.frame_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.max_frame_history))
        self.analysis_cache: Dict[str, FrameAnalysisResult] = {}
        self.pattern_tracking: Dict[str, List[VisualPattern]] = defaultdict(list)
        
        # Performance tracking
        self.performance_stats = {
            'total_frames_processed': 0,
            'average_processing_time': 0.0,
            'cache_hit_rate': 0.0,
            'detection_accuracy': 0.0
        }
        
        logger.info("Enhanced Frame Analyzer initialized")
    
    def _initialize_detection_systems(self):
        """Initialize object detection and change detection systems."""
        # Advanced object detection
        if self.config.enable_object_detection and AdvancedObjectDetector is not None:
            try:
                detection_config = DetectionConfig(
                    confidence_threshold=self.config.detection_confidence_threshold,
                    enable_attention=self.config.enable_attention_mechanisms,
                    multi_scale=True,
                    temporal_consistency=True
                )
                self.object_detector = AdvancedObjectDetector(detection_config)
            except Exception as e:
                logger.warning(f"Failed to initialize advanced object detector: {e}")
                self.object_detector = None
        else:
            self.object_detector = None
        
        # Change detection
        if self.config.enable_change_detection and ChangeDetector is not None:
            try:
                self.change_detector = ChangeDetector()
            except Exception as e:
                logger.warning(f"Failed to initialize change detector: {e}")
                self.change_detector = None
        else:
            self.change_detector = None
    
    def _initialize_analysis_engines(self):
        """Initialize pattern recognition and analysis engines."""
        # Pattern recognition
        if self.config.enable_pattern_recognition and PatternRecognizer is not None:
            try:
                self.pattern_recognizer = PatternRecognizer()
            except Exception as e:
                logger.warning(f"Failed to initialize pattern recognizer: {e}")
                self.pattern_recognizer = None
        else:
            self.pattern_recognizer = None
        
        # Spatial analysis
        if self.config.enable_spatial_analysis and SpatialAnalyzer is not None:
            try:
                self.spatial_analyzer = SpatialAnalyzer()
            except Exception as e:
                logger.warning(f"Failed to initialize spatial analyzer: {e}")
                self.spatial_analyzer = None
        else:
            self.spatial_analyzer = None
        
        # Visual reasoning
        if self.config.enable_visual_reasoning and VisualReasoningEngine is not None:
            try:
                reasoning_config = ReasoningConfig(
                    spatial_resolution=self.config.spatial_resolution,
                    temporal_window=self.config.temporal_window,
                    confidence_threshold=self.config.pattern_confidence_threshold
                )
                self.visual_reasoning = VisualReasoningEngine(reasoning_config)
            except Exception as e:
                logger.warning(f"Failed to initialize visual reasoning: {e}")
                self.visual_reasoning = None
        else:
            self.visual_reasoning = None
    
    def _initialize_tracking_systems(self):
        """Initialize attention and processing systems."""
        # Attention mechanisms
        if self.config.enable_attention_mechanisms and AttentionMechanism is not None:
            try:
                attention_config = AttentionConfig(
                    focus_areas=self.config.attention_focus_areas,
                    decay_rate=self.config.attention_decay_rate,
                    boost_threshold=self.config.attention_boost_threshold
                )
                self.attention_mechanism = AttentionMechanism(attention_config)
            except Exception as e:
                logger.warning(f"Failed to initialize attention mechanism: {e}")
                self.attention_mechanism = None
        else:
            self.attention_mechanism = None
        
        # Real-time processing
        if RealTimeProcessor is not None:
            try:
                processing_config = ProcessingConfig(
                    mode=ProcessingMode.REALTIME if self.config.analysis_mode == AnalysisMode.REALTIME else ProcessingMode.BALANCED,
                    timeout=self.config.processing_timeout,
                    enable_caching=self.config.enable_caching
                )
                self.real_time_processor = RealTimeProcessor(processing_config)
            except Exception as e:
                logger.warning(f"Failed to initialize real-time processor: {e}")
                self.real_time_processor = None
        else:
            self.real_time_processor = None
    
    def analyze_frame(self, 
                     frame: Union[np.ndarray, List[List[int]], List[List[List[int]]]], 
                     game_id: str = "default",
                     frame_id: Optional[str] = None) -> FrameAnalysisResult:
        """
        Perform comprehensive frame analysis with advanced pattern recognition.
        
        Args:
            frame: Input frame (numpy array, 2D list, or 3D list)
            game_id: Game identifier for tracking
            frame_id: Optional frame identifier
            
        Returns:
            FrameAnalysisResult with comprehensive analysis data
        """
        start_time = time.time()
        
        # Generate frame ID if not provided
        if frame_id is None:
            frame_id = f"{game_id}_{int(time.time() * 1000)}"
        
        # Check cache first
        if self.config.enable_caching and frame_id in self.analysis_cache:
            cached_result = self.analysis_cache[frame_id]
            if time.time() - cached_result.timestamp < self.config.cache_ttl:
                self.performance_stats['cache_hit_rate'] = (
                    self.performance_stats['cache_hit_rate'] * 0.9 + 0.1
                )
                return cached_result
        
        # Preprocess frame
        processed_frame = self._preprocess_frame(frame)
        
        # Initialize result
        result = FrameAnalysisResult(
            frame_id=frame_id,
            timestamp=time.time(),
            analysis_mode=self.config.analysis_mode
        )
        
        try:
            # Object detection
            if self.object_detector:
                objects = self._detect_objects(processed_frame, game_id)
                result.objects = objects
            
            # Pattern recognition
            if self.pattern_recognizer:
                patterns = self._recognize_patterns(processed_frame, game_id)
                result.patterns = patterns
            
            # Change detection
            if self.change_detector and game_id in self.frame_history:
                changes = self._detect_changes(processed_frame, game_id)
                result.changes = changes
            
            # Spatial analysis
            if self.spatial_analyzer:
                spatial_analysis = self._analyze_spatial_properties(processed_frame, game_id)
                result.spatial_analysis = spatial_analysis
                result.spatial_relationships = spatial_analysis.get('relationships', [])
            
            # Attention mechanisms
            if self.attention_mechanism:
                attention_result = self._compute_attention(processed_frame, game_id)
                result.attention_map = attention_result.get('attention_map')
                result.focus_areas = attention_result.get('focus_areas', [])
            
            # Visual reasoning
            if self.visual_reasoning:
                reasoning_results = self._perform_visual_reasoning(processed_frame, game_id, result)
                result.reasoning_results = reasoning_results
                result.insights = self._extract_insights(reasoning_results)
            
            # Update frame history
            self.frame_history[game_id].append(processed_frame)
            
            # Calculate performance metrics
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            result.confidence_score = self._calculate_confidence_score(result)
            result.quality_score = self._calculate_quality_score(processed_frame)
            
            # Update performance stats
            self._update_performance_stats(processing_time)
            
            # Cache result
            if self.config.enable_caching:
                self.analysis_cache[frame_id] = result
                # Clean old cache entries
                self._cleanup_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"Frame analysis failed: {e}")
            result.processing_time = time.time() - start_time
            result.confidence_score = 0.0
            return result
    
    def _preprocess_frame(self, frame: Union[np.ndarray, List, Tuple]) -> np.ndarray:
        """Preprocess frame to standard format."""
        try:
            # Convert to numpy array if needed
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            
            # Handle different input formats
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # RGB image
                frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            elif len(frame.shape) == 2:
                # Grayscale
                frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 1:
                # Single channel
                frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
            # Ensure proper data type
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # Resize if too large (for performance)
            if frame.shape[0] > 512 or frame.shape[1] > 512:
                scale = min(512 / frame.shape[0], 512 / frame.shape[1])
                new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
                frame = cv2.resize(frame, new_size)
            
            return frame
            
        except Exception as e:
            logger.error(f"Frame preprocessing failed: {e}")
            # Return a default frame
            return np.zeros((64, 64, 3), dtype=np.uint8)
    
    def _detect_objects(self, frame: np.ndarray, game_id: str) -> List[Dict[str, Any]]:
        """Detect objects in the frame."""
        try:
            detections = self.object_detector.detect(frame)
            
            objects = []
            for detection in detections:
                obj = {
                    'class_id': detection.class_id,
                    'class_name': detection.class_name,
                    'confidence': detection.confidence,
                    'bounding_box': detection.bounding_box,
                    'features': detection.features,
                    'temporal_stability': detection.temporal_stability
                }
                objects.append(obj)
            
            return objects
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    def _recognize_patterns(self, frame: np.ndarray, game_id: str) -> List[VisualPattern]:
        """Recognize visual patterns in the frame."""
        try:
            # Convert to grayscale for pattern recognition
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect patterns using the pattern recognizer
            pattern_infos = self.pattern_recognizer.detect_patterns(gray_frame.tolist())
            
            patterns = []
            for i, pattern_info in enumerate(pattern_infos):
                pattern = VisualPattern(
                    pattern_id=f"{game_id}_pattern_{i}",
                    pattern_type=pattern_info.pattern_type,
                    description=pattern_info.description,
                    confidence=pattern_info.confidence,
                    bounding_box=(0, 0, frame.shape[1], frame.shape[0]),  # Full frame for now
                    features={
                        'locations': pattern_info.locations,
                        'properties': pattern_info.properties,
                        'size': pattern_info.size
                    }
                )
                patterns.append(pattern)
            
            # Track patterns across frames
            self.pattern_tracking[game_id].extend(patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern recognition failed: {e}")
            return []
    
    def _detect_changes(self, frame: np.ndarray, game_id: str) -> List[Dict[str, Any]]:
        """Detect changes from previous frame."""
        try:
            if len(self.frame_history[game_id]) < 2:
                return []
            
            previous_frame = self.frame_history[game_id][-2]
            changes = self.change_detector.detect_changes(previous_frame, frame)
            
            return changes
            
        except Exception as e:
            logger.error(f"Change detection failed: {e}")
            return []
    
    def _analyze_spatial_properties(self, frame: np.ndarray, game_id: str) -> Dict[str, Any]:
        """Analyze spatial properties of the frame."""
        try:
            spatial_analysis = self.spatial_analyzer.analyze_spatial_properties(frame)
            return spatial_analysis
            
        except Exception as e:
            logger.error(f"Spatial analysis failed: {e}")
            return {}
    
    def _compute_attention(self, frame: np.ndarray, game_id: str) -> Dict[str, Any]:
        """Compute attention map and focus areas."""
        try:
            attention_result = self.attention_mechanism.compute_attention(frame)
            return attention_result
            
        except Exception as e:
            logger.error(f"Attention computation failed: {e}")
            return {}
    
    def _perform_visual_reasoning(self, frame: np.ndarray, game_id: str, 
                                 analysis_result: FrameAnalysisResult) -> List[Dict[str, Any]]:
        """Perform visual reasoning on the frame."""
        try:
            # Prepare context for reasoning
            context = {
                'objects': analysis_result.objects,
                'patterns': analysis_result.patterns,
                'spatial_analysis': analysis_result.spatial_analysis,
                'frame_history': list(self.frame_history[game_id])[-5:]  # Last 5 frames
            }
            
            reasoning_results = self.visual_reasoning.reason(frame, context)
            return reasoning_results
            
        except Exception as e:
            logger.error(f"Visual reasoning failed: {e}")
            return []
    
    def _extract_insights(self, reasoning_results: List[Dict[str, Any]]) -> List[str]:
        """Extract insights from reasoning results."""
        insights = []
        for result in reasoning_results:
            if 'conclusion' in result:
                insights.append(result['conclusion'])
        return insights
    
    def _calculate_confidence_score(self, result: FrameAnalysisResult) -> float:
        """Calculate overall confidence score for the analysis."""
        try:
            scores = []
            
            # Object detection confidence
            if result.objects:
                obj_confidences = [obj.get('confidence', 0.0) for obj in result.objects]
                scores.extend(obj_confidences)
            
            # Pattern recognition confidence
            if result.patterns:
                pattern_confidences = [pattern.confidence for pattern in result.patterns]
                scores.extend(pattern_confidences)
            
            # Reasoning confidence
            if result.reasoning_results:
                reasoning_confidences = [r.get('confidence', 0.0) for r in result.reasoning_results]
                scores.extend(reasoning_confidences)
            
            if scores:
                return sum(scores) / len(scores)
            else:
                return 0.5  # Default confidence
                
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.0
    
    def _calculate_quality_score(self, frame: np.ndarray) -> float:
        """Calculate frame quality score."""
        try:
            # Calculate image sharpness using Laplacian variance
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 range
            quality_score = min(laplacian_var / 1000.0, 1.0)
            return quality_score
            
        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            return 0.5
    
    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics."""
        self.performance_stats['total_frames_processed'] += 1
        
        # Update average processing time
        current_avg = self.performance_stats['average_processing_time']
        total_frames = self.performance_stats['total_frames_processed']
        self.performance_stats['average_processing_time'] = (
            (current_avg * (total_frames - 1) + processing_time) / total_frames
        )
    
    def _cleanup_cache(self):
        """Clean up old cache entries."""
        current_time = time.time()
        to_remove = []
        
        for frame_id, result in self.analysis_cache.items():
            if current_time - result.timestamp > self.config.cache_ttl:
                to_remove.append(frame_id)
        
        for frame_id in to_remove:
            del self.analysis_cache[frame_id]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.performance_stats.copy()
    
    def reset_for_new_game(self, game_id: str):
        """Reset analysis state for a new game."""
        if game_id in self.frame_history:
            self.frame_history[game_id].clear()
        
        if game_id in self.pattern_tracking:
            self.pattern_tracking[game_id].clear()
        
        logger.info(f"Enhanced frame analyzer reset for game {game_id}")
    
    def get_pattern_history(self, game_id: str) -> List[VisualPattern]:
        """Get pattern history for a specific game."""
        return self.pattern_tracking.get(game_id, [])
    
    def get_frame_history(self, game_id: str) -> List[np.ndarray]:
        """Get frame history for a specific game."""
        return list(self.frame_history.get(game_id, []))


def create_enhanced_frame_analyzer(config: Optional[EnhancedFrameAnalysisConfig] = None) -> EnhancedFrameAnalyzer:
    """Create an enhanced frame analyzer with the given configuration."""
    return EnhancedFrameAnalyzer(config)
