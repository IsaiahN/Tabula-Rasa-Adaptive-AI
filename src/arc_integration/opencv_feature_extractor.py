"""
OpenCV Feature Extractor for ARC-AGI-3

This module provides computer vision capabilities for analyzing ARC puzzle grids.

This file has been modularized. The main functionality is now in src/vision/.
"""

# Import from the new modular structure
from src.vision import (
    ObjectDetector, DetectedObject,
    SpatialAnalyzer, SpatialRelationship,
    PatternRecognizer, PatternInfo,
    FeatureExtractor,
    ChangeDetector, ChangeInfo, ActionableTarget
)

# Re-export for backward compatibility
__all__ = [
    'ObjectDetector', 'DetectedObject',
    'SpatialAnalyzer', 'SpatialRelationship',
    'PatternRecognizer', 'PatternInfo',
    'FeatureExtractor',
    'ChangeDetector', 'ChangeInfo', 'ActionableTarget'
]

# For direct execution, create a simple wrapper
if __name__ == "__main__":
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def main():
        """Main entry point for feature extraction."""
        try:
            # Create feature extractor
            extractor = FeatureExtractor()
            logger.info("Feature extractor created successfully")
            return extractor
        except Exception as e:
            logger.error(f"Error creating feature extractor: {e}")
            raise
    
    # Run the main function
    main()
