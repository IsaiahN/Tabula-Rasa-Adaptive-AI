"""
Real-Time Vision Processing

High-performance real-time vision processing with frame buffering,
parallel processing, and adaptive quality control.
"""

import cv2
import numpy as np
import threading
import queue
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..interfaces import ComponentInterface
from ...training.caching import CacheManager, CacheConfig
from ...training.monitoring import PerformanceMonitor


class ProcessingMode(Enum):
    """Available processing modes."""
    REALTIME = "realtime"
    BATCH = "batch"
    ADAPTIVE = "adaptive"


@dataclass
class ProcessingConfig:
    """Configuration for real-time processing."""
    mode: ProcessingMode = ProcessingMode.ADAPTIVE
    max_fps: float = 30.0
    buffer_size: int = 10
    parallel_workers: int = 4
    quality_threshold: float = 0.8
    adaptive_quality: bool = True
    enable_caching: bool = True
    cache_ttl: int = 60  # seconds
    frame_skip_threshold: float = 0.1  # Skip frames if processing is too slow


@dataclass
class ProcessedFrame:
    """Processed frame result."""
    frame_id: int
    timestamp: datetime
    processing_time: float
    quality_score: float
    features: Dict[str, Any]
    metadata: Dict[str, Any]


class RealTimeProcessor(ComponentInterface):
    """
    Real-time vision processing system with adaptive quality control
    and parallel processing capabilities.
    """
    
    def __init__(self, config: ProcessingConfig, cache_config: Optional[CacheConfig] = None):
        """Initialize the real-time processor."""
        self.config = config
        self.cache_config = cache_config or CacheConfig()
        self.cache = CacheManager(self.cache_config)
        self.performance_monitor = PerformanceMonitor()
        
        # Processing state
        self.frame_buffer = queue.Queue(maxsize=config.buffer_size)
        self.processed_frames = queue.Queue(maxsize=config.buffer_size)
        self.executor = ThreadPoolExecutor(max_workers=config.parallel_workers)
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times: List[float] = []
        self.quality_scores: List[float] = []
        self.fps_history: List[float] = []
        
        # Adaptive quality control
        self.current_quality = 1.0
        self.quality_adjustment_rate = 0.1
        self.target_processing_time = 1.0 / config.max_fps
        
        # Threading
        self.processing_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        # Callbacks
        self.frame_callbacks: List[Callable[[ProcessedFrame], None]] = []
        self.error_callbacks: List[Callable[[Exception], None]] = []
        
        self._initialized = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize the real-time processor."""
        try:
            self.cache.initialize()
            self.performance_monitor.start_monitoring()
            
            # Start processing thread
            self.running = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            self._initialized = True
            self.logger.info(f"Real-time processor initialized with {self.config.mode.value} mode")
        except Exception as e:
            self.logger.error(f"Failed to initialize real-time processor: {e}")
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get current component state."""
        return {
            'name': 'RealTimeProcessor',
            'status': 'running' if self._initialized else 'stopped',
            'last_updated': datetime.now(),
            'metadata': {
                'mode': self.config.mode.value,
                'frame_count': self.frame_count,
                'buffer_size': self.frame_buffer.qsize(),
                'processed_frames': self.processed_frames.qsize(),
                'current_quality': self.current_quality,
                'average_fps': np.mean(self.fps_history) if self.fps_history else 0.0,
                'average_processing_time': np.mean(self.processing_times) if self.processing_times else 0.0
            }
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.running = False
            
            # Wait for processing thread to finish
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Clear queues
            while not self.frame_buffer.empty():
                try:
                    self.frame_buffer.get_nowait()
                except queue.Empty:
                    break
            
            while not self.processed_frames.empty():
                try:
                    self.processed_frames.get_nowait()
                except queue.Empty:
                    break
            
            self.performance_monitor.stop_monitoring()
            self.cache.clear()
            self._initialized = False
            self.logger.info("Real-time processor cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return (self._initialized and 
                self.cache.is_healthy() and 
                self.running and 
                self.frame_buffer.qsize() < self.config.buffer_size * 0.9)
    
    def process_frame(self, frame: np.ndarray, 
                     frame_id: Optional[int] = None) -> Optional[ProcessedFrame]:
        """Process a single frame."""
        try:
            if frame_id is None:
                frame_id = self.frame_count
                self.frame_count += 1
            
            # Check if we should skip this frame
            if self._should_skip_frame():
                self.logger.debug(f"Skipping frame {frame_id} due to performance constraints")
                return None
            
            # Add frame to buffer
            if self.config.mode == ProcessingMode.REALTIME:
                # Process immediately
                return self._process_frame_immediate(frame, frame_id)
            else:
                # Add to buffer for batch processing
                self.frame_buffer.put((frame, frame_id, datetime.now()))
                return None
                
        except Exception as e:
            self.logger.error(f"Error processing frame {frame_id}: {e}")
            self._notify_error_callbacks(e)
            return None
    
    def get_processed_frame(self, timeout: float = 1.0) -> Optional[ProcessedFrame]:
        """Get the next processed frame."""
        try:
            return self.processed_frames.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def add_frame_callback(self, callback: Callable[[ProcessedFrame], None]) -> None:
        """Add a callback for processed frames."""
        self.frame_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Add a callback for errors."""
        self.error_callbacks.append(callback)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            current_time = time.time()
            
            # Calculate FPS
            if len(self.fps_history) > 0:
                avg_fps = np.mean(self.fps_history)
                current_fps = len(self.fps_history) / (current_time - self.fps_history[0]) if len(self.fps_history) > 1 else 0
            else:
                avg_fps = 0
                current_fps = 0
            
            # Calculate processing statistics
            avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
            avg_quality = np.mean(self.quality_scores) if self.quality_scores else 0
            
            # Calculate buffer utilization
            buffer_utilization = self.frame_buffer.qsize() / self.config.buffer_size
            processed_utilization = self.processed_frames.qsize() / self.config.buffer_size
            
            return {
                'frame_count': self.frame_count,
                'current_fps': current_fps,
                'average_fps': avg_fps,
                'average_processing_time': avg_processing_time,
                'average_quality': avg_quality,
                'current_quality': self.current_quality,
                'buffer_utilization': buffer_utilization,
                'processed_utilization': processed_utilization,
                'processing_mode': self.config.mode.value,
                'parallel_workers': self.config.parallel_workers,
                'adaptive_quality_enabled': self.config.adaptive_quality
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {'error': str(e)}
    
    def _processing_loop(self) -> None:
        """Main processing loop for batch and adaptive modes."""
        try:
            while self.running:
                try:
                    # Get frame from buffer
                    frame, frame_id, timestamp = self.frame_buffer.get(timeout=1.0)
                    
                    # Process frame
                    processed_frame = self._process_frame_immediate(frame, frame_id)
                    
                    if processed_frame:
                        # Add to processed frames queue
                        self.processed_frames.put(processed_frame)
                        
                        # Notify callbacks
                        self._notify_frame_callbacks(processed_frame)
                        
                        # Update adaptive quality
                        if self.config.adaptive_quality:
                            self._update_adaptive_quality(processed_frame)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in processing loop: {e}")
                    self._notify_error_callbacks(e)
                    
        except Exception as e:
            self.logger.error(f"Fatal error in processing loop: {e}")
            self._notify_error_callbacks(e)
    
    def _process_frame_immediate(self, frame: np.ndarray, frame_id: int) -> ProcessedFrame:
        """Process a frame immediately."""
        try:
            start_time = time.time()
            
            # Apply quality adjustment
            if self.config.adaptive_quality and self.current_quality < 1.0:
                frame = self._adjust_frame_quality(frame, self.current_quality)
            
            # Extract features
            features = self._extract_features(frame)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(frame, features)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create processed frame
            processed_frame = ProcessedFrame(
                frame_id=frame_id,
                timestamp=datetime.now(),
                processing_time=processing_time,
                quality_score=quality_score,
                features=features,
                metadata={
                    'original_shape': frame.shape,
                    'quality_adjustment': self.current_quality,
                    'processing_mode': self.config.mode.value
                }
            )
            
            # Update performance metrics
            self.processing_times.append(processing_time)
            self.quality_scores.append(quality_score)
            
            # Update FPS history
            current_time = time.time()
            self.fps_history.append(current_time)
            if len(self.fps_history) > 100:  # Keep last 100 timestamps
                self.fps_history = self.fps_history[-100:]
            
            # Cache result if enabled
            if self.config.enable_caching:
                cache_key = f"frame_{frame_id}_{current_time}"
                self.cache.set(cache_key, processed_frame, ttl=self.config.cache_ttl)
            
            return processed_frame
            
        except Exception as e:
            self.logger.error(f"Error processing frame {frame_id}: {e}")
            raise
    
    def _extract_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract features from frame."""
        try:
            features = {}
            
            # Basic frame properties
            features['shape'] = frame.shape
            features['dtype'] = str(frame.dtype)
            features['mean_intensity'] = np.mean(frame)
            features['std_intensity'] = np.std(frame)
            
            # Edge detection
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / edges.size
            
            # Corner detection
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
            features['corner_count'] = len(corners) if corners is not None else 0
            
            # Texture analysis
            features['texture_variance'] = np.var(gray)
            
            # Color analysis (if color image)
            if len(frame.shape) == 3:
                features['color_channels'] = frame.shape[2]
                features['color_mean'] = np.mean(frame, axis=(0, 1)).tolist()
                features['color_std'] = np.std(frame, axis=(0, 1)).tolist()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return {}
    
    def _calculate_quality_score(self, frame: np.ndarray, features: Dict[str, Any]) -> float:
        """Calculate quality score for the frame."""
        try:
            score = 1.0
            
            # Sharpness score (based on edge density)
            if 'edge_density' in features:
                edge_density = features['edge_density']
                # Higher edge density generally means sharper image
                sharpness_score = min(edge_density * 10, 1.0)
                score *= sharpness_score
            
            # Brightness score
            if 'mean_intensity' in features:
                mean_intensity = features['mean_intensity']
                # Optimal brightness is around 0.5 (for normalized images)
                brightness_score = 1.0 - abs(mean_intensity - 0.5) * 2
                score *= max(brightness_score, 0.1)
            
            # Contrast score (based on standard deviation)
            if 'std_intensity' in features:
                std_intensity = features['std_intensity']
                # Higher contrast generally means better quality
                contrast_score = min(std_intensity * 2, 1.0)
                score *= contrast_score
            
            # Texture score
            if 'texture_variance' in features:
                texture_variance = features['texture_variance']
                # Some texture variance is good, but not too much
                texture_score = min(texture_variance * 10, 1.0)
                score *= texture_score
            
            return max(score, 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {e}")
            return 0.5
    
    def _adjust_frame_quality(self, frame: np.ndarray, quality: float) -> np.ndarray:
        """Adjust frame quality for performance."""
        try:
            if quality >= 1.0:
                return frame
            
            # Resize frame based on quality
            height, width = frame.shape[:2]
            new_height = int(height * quality)
            new_width = int(width * quality)
            
            # Resize frame
            resized = cv2.resize(frame, (new_width, new_height))
            
            # Resize back to original size
            result = cv2.resize(resized, (width, height))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error adjusting frame quality: {e}")
            return frame
    
    def _should_skip_frame(self) -> bool:
        """Determine if current frame should be skipped."""
        try:
            if not self.processing_times:
                return False
            
            # Calculate average processing time
            avg_processing_time = np.mean(self.processing_times[-10:])  # Last 10 frames
            
            # Skip if processing is too slow
            if avg_processing_time > self.target_processing_time * (1 + self.config.frame_skip_threshold):
                return True
            
            # Skip if buffer is getting full
            if self.frame_buffer.qsize() > self.config.buffer_size * 0.8:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking if frame should be skipped: {e}")
            return False
    
    def _update_adaptive_quality(self, processed_frame: ProcessedFrame) -> None:
        """Update adaptive quality based on processing performance."""
        try:
            if not self.config.adaptive_quality:
                return
            
            # Calculate target processing time
            target_time = self.target_processing_time
            
            # Adjust quality based on processing time
            if processed_frame.processing_time > target_time * 1.1:
                # Processing too slow, reduce quality
                self.current_quality = max(0.3, self.current_quality - self.quality_adjustment_rate)
            elif processed_frame.processing_time < target_time * 0.9:
                # Processing fast enough, can increase quality
                self.current_quality = min(1.0, self.current_quality + self.quality_adjustment_rate)
            
            # Adjust quality based on quality score
            if processed_frame.quality_score < self.config.quality_threshold:
                # Quality too low, increase quality
                self.current_quality = min(1.0, self.current_quality + self.quality_adjustment_rate * 0.5)
            
            self.logger.debug(f"Updated adaptive quality to {self.current_quality:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error updating adaptive quality: {e}")
    
    def _notify_frame_callbacks(self, processed_frame: ProcessedFrame) -> None:
        """Notify frame callbacks."""
        try:
            for callback in self.frame_callbacks:
                try:
                    callback(processed_frame)
                except Exception as e:
                    self.logger.error(f"Error in frame callback: {e}")
        except Exception as e:
            self.logger.error(f"Error notifying frame callbacks: {e}")
    
    def _notify_error_callbacks(self, error: Exception) -> None:
        """Notify error callbacks."""
        try:
            for callback in self.error_callbacks:
                try:
                    callback(error)
                except Exception as e:
                    self.logger.error(f"Error in error callback: {e}")
        except Exception as e:
            self.logger.error(f"Error notifying error callbacks: {e}")
