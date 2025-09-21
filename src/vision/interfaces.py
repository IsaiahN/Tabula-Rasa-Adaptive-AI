"""
Vision Component Interfaces

Defines common interfaces for vision components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class ComponentInterface(ABC):
    """Base interface for vision components."""
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data and return results."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get component status information."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset component state."""
        pass


class FrameProcessorInterface(ComponentInterface):
    """Interface for frame processing components."""
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame."""
        pass
    
    @abstractmethod
    def process_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Process multiple frames."""
        pass


class DetectorInterface(ComponentInterface):
    """Interface for detection components."""
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects/features in a frame."""
        pass
    
    @abstractmethod
    def get_detection_config(self) -> Dict[str, Any]:
        """Get current detection configuration."""
        pass
    
    @abstractmethod
    def set_detection_config(self, config: Dict[str, Any]) -> None:
        """Set detection configuration."""
        pass


class AnalyzerInterface(ComponentInterface):
    """Interface for analysis components."""
    
    @abstractmethod
    def analyze(self, frame: np.ndarray, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze a frame with optional context."""
        pass
    
    @abstractmethod
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get current analysis configuration."""
        pass
    
    @abstractmethod
    def set_analysis_config(self, config: Dict[str, Any]) -> None:
        """Set analysis configuration."""
        pass


class TrackerInterface(ComponentInterface):
    """Interface for tracking components."""
    
    @abstractmethod
    def track(self, frame: np.ndarray, previous_frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Track objects/features across frames."""
        pass
    
    @abstractmethod
    def get_tracking_history(self) -> List[Dict[str, Any]]:
        """Get tracking history."""
        pass
    
    @abstractmethod
    def clear_tracking_history(self) -> None:
        """Clear tracking history."""
        pass
