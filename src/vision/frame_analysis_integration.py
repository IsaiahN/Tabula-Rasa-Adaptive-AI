"""
Frame Analysis Integration Module

This module provides integration between the enhanced frame analysis system
and the existing ARC-AGI-3 training infrastructure.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from .enhanced_frame_analyzer import (
    EnhancedFrameAnalyzer, 
    EnhancedFrameAnalysisConfig, 
    AnalysisMode,
    FrameAnalysisResult,
    VisualPattern
)
from .frame_analyzer import FrameAnalyzer

# Import with fallback for optional dependencies
try:
    from ..core.cognitive_subsystems.visual_subsystems import FrameAnalysisMonitor
except ImportError:
    FrameAnalysisMonitor = None

try:
    from ..database.system_integration import get_system_integration
    from ..database.api import Component, LogLevel
except ImportError:
    get_system_integration = None
    Component = None
    LogLevel = None

logger = logging.getLogger(__name__)


@dataclass
class FrameAnalysisIntegrationConfig:
    """Configuration for frame analysis integration."""
    enable_enhanced_analysis: bool = True
    enable_legacy_compatibility: bool = True
    enable_database_logging: bool = True
    enable_cognitive_monitoring: bool = True
    
    # Analysis mode selection
    default_analysis_mode: AnalysisMode = AnalysisMode.ENHANCED
    adaptive_mode_selection: bool = True
    
    # Performance settings
    max_concurrent_analyses: int = 5
    analysis_timeout: float = 2.0  # seconds
    enable_result_caching: bool = True
    cache_size: int = 1000


class FrameAnalysisIntegration:
    """
    Integration layer between enhanced frame analysis and existing systems.
    
    This class provides:
    - Seamless integration with existing FrameAnalyzer
    - Database logging and persistence
    - Cognitive subsystem monitoring
    - Adaptive analysis mode selection
    - Performance optimization
    """
    
    def __init__(self, config: Optional[FrameAnalysisIntegrationConfig] = None):
        self.config = config or FrameAnalysisIntegrationConfig()
        
        # Initialize analysis systems
        self._initialize_analysis_systems()
        
        # Initialize monitoring and logging
        self._initialize_monitoring()
        
        # Performance tracking
        self.analysis_stats = {
            'total_analyses': 0,
            'enhanced_analyses': 0,
            'legacy_analyses': 0,
            'average_processing_time': 0.0,
            'success_rate': 0.0,
            'error_count': 0
        }
        
        logger.info("Frame Analysis Integration initialized")
    
    def _initialize_analysis_systems(self):
        """Initialize both enhanced and legacy analysis systems."""
        # Enhanced frame analyzer
        if self.config.enable_enhanced_analysis:
            enhanced_config = EnhancedFrameAnalysisConfig(
                analysis_mode=self.config.default_analysis_mode,
                enable_caching=self.config.enable_result_caching
            )
            self.enhanced_analyzer = EnhancedFrameAnalyzer(enhanced_config)
        else:
            self.enhanced_analyzer = None
        
        # Legacy frame analyzer for compatibility
        if self.config.enable_legacy_compatibility:
            self.legacy_analyzer = FrameAnalyzer()
        else:
            self.legacy_analyzer = None
    
    def _initialize_monitoring(self):
        """Initialize monitoring and logging systems."""
        # Cognitive subsystem monitoring
        if self.config.enable_cognitive_monitoring and FrameAnalysisMonitor is not None:
            try:
                self.frame_monitor = FrameAnalysisMonitor()
            except Exception as e:
                logger.warning(f"Failed to initialize frame monitor: {e}")
                self.frame_monitor = None
        else:
            self.frame_monitor = None
        
        # Database integration
        if self.config.enable_database_logging and get_system_integration is not None:
            try:
                self.integration = get_system_integration()
            except Exception as e:
                logger.warning(f"Failed to initialize database integration: {e}")
                self.integration = None
        else:
            self.integration = None
    
    def analyze_frame(self, 
                     frame: Union[np.ndarray, List[List[int]], List[List[List[int]]]], 
                     game_id: str = "default",
                     analysis_mode: Optional[AnalysisMode] = None,
                     use_enhanced: Optional[bool] = None) -> Dict[str, Any]:
        """
        Analyze a frame using the appropriate analysis system.
        
        Args:
            frame: Input frame data
            game_id: Game identifier
            analysis_mode: Specific analysis mode to use
            use_enhanced: Whether to use enhanced analysis (overrides adaptive selection)
            
        Returns:
            Dictionary containing analysis results
        """
        start_time = time.time()
        
        try:
            # Determine which analyzer to use
            if use_enhanced is None:
                use_enhanced = self._should_use_enhanced_analysis(game_id, frame)
            
            # Select analysis mode
            if analysis_mode is None:
                analysis_mode = self._select_analysis_mode(game_id, frame, use_enhanced)
            
            # Perform analysis
            if use_enhanced and self.enhanced_analyzer:
                result = self._perform_enhanced_analysis(frame, game_id, analysis_mode)
                self.analysis_stats['enhanced_analyses'] += 1
            else:
                result = self._perform_legacy_analysis(frame, game_id)
                self.analysis_stats['legacy_analyses'] += 1
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_analysis_stats(processing_time, True)
            
            # Log to database
            if self.integration:
                self._log_analysis_to_database(result, game_id, processing_time)
            
            # Update cognitive monitoring
            if self.frame_monitor:
                self._update_cognitive_monitoring(result, game_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Frame analysis failed: {e}")
            self._update_analysis_stats(time.time() - start_time, False)
            return {"error": str(e), "analysis_type": "failed"}
    
    def _should_use_enhanced_analysis(self, game_id: str, frame: Any) -> bool:
        """Determine whether to use enhanced analysis based on adaptive selection."""
        if not self.config.adaptive_mode_selection:
            return self.config.enable_enhanced_analysis
        
        # Simple heuristic: use enhanced analysis for complex frames
        try:
            if hasattr(frame, 'shape'):
                # Complex frame (likely numpy array)
                return True
            elif isinstance(frame, list) and len(frame) > 0:
                # Check if it's a 3D list (complex structure)
                if isinstance(frame[0], list) and len(frame[0]) > 0:
                    if isinstance(frame[0][0], list):
                        return True  # 3D list
            return False
        except:
            return False
    
    def _select_analysis_mode(self, game_id: str, frame: Any, use_enhanced: bool) -> AnalysisMode:
        """Select the appropriate analysis mode."""
        if not use_enhanced:
            return AnalysisMode.BASIC
        
        # Adaptive mode selection based on frame complexity
        try:
            if hasattr(frame, 'shape'):
                height, width = frame.shape[:2]
                if height * width > 10000:  # Large frame
                    return AnalysisMode.DEEP
                elif height * width > 1000:  # Medium frame
                    return AnalysisMode.ENHANCED
                else:  # Small frame
                    return AnalysisMode.REALTIME
            else:
                return AnalysisMode.ENHANCED
        except:
            return self.config.default_analysis_mode
    
    def _perform_enhanced_analysis(self, frame: Any, game_id: str, analysis_mode: AnalysisMode) -> Dict[str, Any]:
        """Perform enhanced frame analysis."""
        # Update analyzer config if needed
        if self.enhanced_analyzer.config.analysis_mode != analysis_mode:
            self.enhanced_analyzer.config.analysis_mode = analysis_mode
        
        # Perform analysis
        result = self.enhanced_analyzer.analyze_frame(frame, game_id)
        
        # Convert to dictionary format for compatibility
        return {
            "analysis_type": "enhanced",
            "analysis_mode": analysis_mode.value,
            "frame_id": result.frame_id,
            "timestamp": result.timestamp,
            "objects": result.objects,
            "patterns": [self._pattern_to_dict(p) for p in result.patterns],
            "changes": result.changes,
            "spatial_analysis": result.spatial_analysis,
            "spatial_relationships": result.spatial_relationships,
            "attention_map": result.attention_map.tolist() if result.attention_map is not None else None,
            "focus_areas": result.focus_areas,
            "reasoning_results": result.reasoning_results,
            "insights": result.insights,
            "processing_time": result.processing_time,
            "confidence_score": result.confidence_score,
            "quality_score": result.quality_score,
            "frame_quality": result.frame_quality,
            "temporal_context": result.temporal_context
        }
    
    def _perform_legacy_analysis(self, frame: Any, game_id: str) -> Dict[str, Any]:
        """Perform legacy frame analysis for compatibility."""
        try:
            # Use the legacy analyzer
            result = self.legacy_analyzer.analyze_frame_for_action6_targets(frame, game_id)
            
            # Add analysis type identifier
            result["analysis_type"] = "legacy"
            result["analysis_mode"] = "basic"
            
            return result
            
        except Exception as e:
            logger.error(f"Legacy analysis failed: {e}")
            return {
                "analysis_type": "legacy",
                "analysis_mode": "basic",
                "error": str(e)
            }
    
    def _pattern_to_dict(self, pattern: VisualPattern) -> Dict[str, Any]:
        """Convert VisualPattern to dictionary."""
        return {
            "pattern_id": pattern.pattern_id,
            "pattern_type": pattern.pattern_type,
            "description": pattern.description,
            "confidence": pattern.confidence,
            "bounding_box": pattern.bounding_box,
            "features": pattern.features,
            "temporal_stability": pattern.temporal_stability,
            "spatial_relationships": pattern.spatial_relationships,
            "metadata": pattern.metadata
        }
    
    def _update_analysis_stats(self, processing_time: float, success: bool):
        """Update analysis statistics."""
        self.analysis_stats['total_analyses'] += 1
        
        if success:
            # Update average processing time
            current_avg = self.analysis_stats['average_processing_time']
            total = self.analysis_stats['total_analyses']
            self.analysis_stats['average_processing_time'] = (
                (current_avg * (total - 1) + processing_time) / total
            )
            
            # Update success rate
            success_count = self.analysis_stats['total_analyses'] - self.analysis_stats['error_count']
            self.analysis_stats['success_rate'] = success_count / self.analysis_stats['total_analyses']
        else:
            self.analysis_stats['error_count'] += 1
            self.analysis_stats['success_rate'] = (
                (self.analysis_stats['total_analyses'] - self.analysis_stats['error_count']) / 
                self.analysis_stats['total_analyses']
            )
    
    def _log_analysis_to_database(self, result: Dict[str, Any], game_id: str, processing_time: float):
        """Log analysis results to database."""
        try:
            if self.integration and Component is not None and LogLevel is not None:
                # Prepare log data
                log_data = {
                    "game_id": game_id,
                    "analysis_type": result.get("analysis_type", "unknown"),
                    "analysis_mode": result.get("analysis_mode", "unknown"),
                    "processing_time": processing_time,
                    "confidence_score": result.get("confidence_score", 0.0),
                    "quality_score": result.get("quality_score", 0.0),
                    "object_count": len(result.get("objects", [])),
                    "pattern_count": len(result.get("patterns", [])),
                    "insight_count": len(result.get("insights", [])),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Log to database
                self.integration.log_system_event(
                    LogLevel.INFO,
                    Component.FRAME_ANALYSIS,
                    f"Frame analysis completed for game {game_id}",
                    log_data,
                    game_id
                )
                
        except Exception as e:
            logger.error(f"Database logging failed: {e}")
    
    def _update_cognitive_monitoring(self, result: Dict[str, Any], game_id: str):
        """Update cognitive subsystem monitoring."""
        try:
            if self.frame_monitor and hasattr(self.frame_monitor, 'update_metrics'):
                # Update frame analysis metrics
                metrics = {
                    "processing_time": result.get("processing_time", 0.0),
                    "confidence_score": result.get("confidence_score", 0.0),
                    "quality_score": result.get("quality_score", 0.0),
                    "object_count": len(result.get("objects", [])),
                    "pattern_count": len(result.get("patterns", [])),
                    "analysis_type": result.get("analysis_type", "unknown")
                }
                
                self.frame_monitor.update_metrics(metrics)
                
        except Exception as e:
            logger.error(f"Cognitive monitoring update failed: {e}")
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get current analysis statistics."""
        return self.analysis_stats.copy()
    
    def get_enhanced_analyzer_stats(self) -> Dict[str, Any]:
        """Get enhanced analyzer performance statistics."""
        if self.enhanced_analyzer:
            return self.enhanced_analyzer.get_performance_stats()
        else:
            return {}
    
    def reset_for_new_game(self, game_id: str):
        """Reset analysis state for a new game."""
        if self.enhanced_analyzer:
            self.enhanced_analyzer.reset_for_new_game(game_id)
        
        if self.legacy_analyzer:
            self.legacy_analyzer.reset_for_new_game(game_id)
        
        logger.info(f"Frame analysis integration reset for game {game_id}")
    
    def get_pattern_history(self, game_id: str) -> List[Dict[str, Any]]:
        """Get pattern history for a specific game."""
        if self.enhanced_analyzer:
            patterns = self.enhanced_analyzer.get_pattern_history(game_id)
            return [self._pattern_to_dict(p) for p in patterns]
        else:
            return []
    
    def get_frame_history(self, game_id: str) -> List[Any]:
        """Get frame history for a specific game."""
        if self.enhanced_analyzer:
            return self.enhanced_analyzer.get_frame_history(game_id)
        else:
            return []


def create_frame_analysis_integration(config: Optional[FrameAnalysisIntegrationConfig] = None) -> FrameAnalysisIntegration:
    """Create a frame analysis integration instance."""
    return FrameAnalysisIntegration(config)
