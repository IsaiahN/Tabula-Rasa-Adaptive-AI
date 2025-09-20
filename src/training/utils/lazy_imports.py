"""
Lazy Import Utilities

Provides lazy loading of heavy dependencies to improve startup performance.
"""

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Global caches for lazy imports
_torch = None
_opencv_available = False
_opencv_extractor = None

class LazyImports:
    """Manages lazy imports for performance optimization."""
    
    def __init__(self):
        self._torch = None
        self._opencv_available = False
        self._opencv_extractor = None
    
    def get_torch(self) -> Optional[Any]:
        """Lazy import of torch to avoid startup overhead."""
        if self._torch is None:
            try:
                import torch
                self._torch = torch
                logger.debug("PyTorch loaded successfully")
            except ImportError:
                self._torch = False
                logger.warning("PyTorch not available")
        return self._torch if self._torch is not False else None
    
    def get_opencv_extractor(self) -> Optional[Any]:
        """Lazy import of OpenCV feature extractor."""
        if self._opencv_extractor is None:
            try:
                from src.arc_integration.opencv_feature_extractor import OpenCVFeatureExtractor
                self._opencv_extractor = OpenCVFeatureExtractor
                self._opencv_available = True
                logger.debug("OpenCV feature extractor loaded successfully")
            except ImportError:
                self._opencv_available = False
                self._opencv_extractor = False
                logger.warning("OpenCV feature extractor not available - falling back to basic analysis")
        return self._opencv_extractor if self._opencv_extractor is not False else None
    
    def is_opencv_available(self) -> bool:
        """Check if OpenCV is available."""
        if self._opencv_extractor is None:
            self.get_opencv_extractor()
        return self._opencv_available

# Global lazy import functions for backward compatibility
def get_torch():
    """Global function for lazy torch import."""
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            _torch = False
    return _torch if _torch is not False else None

def get_opencv_extractor():
    """Global function for lazy OpenCV import."""
    global _opencv_available, _opencv_extractor
    if _opencv_extractor is None:
        try:
            from src.arc_integration.opencv_feature_extractor import OpenCVFeatureExtractor
            _opencv_extractor = OpenCVFeatureExtractor
            _opencv_available = True
        except ImportError:
            _opencv_available = False
            _opencv_extractor = False
            print("⚠️ OpenCV feature extractor not available - falling back to basic analysis")
    return _opencv_extractor if _opencv_extractor is not False else None
