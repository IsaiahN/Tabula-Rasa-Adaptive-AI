#!/usr/bin/env python3
"""
Test script for OpenCV feature extraction integration.
"""

import sys
import os
sys.path.append('src')

from src.arc_integration.opencv_feature_extractor import FeatureExtractor

def test_opencv_integration():
    """Test the OpenCV feature extraction system."""
    print(" Testing OpenCV Feature Extraction Integration")
    
    # Create a simple test grid (3x3 with some objects)
    test_grid = [
        [0, 1, 0],
        [1, 1, 1], 
        [0, 1, 0]
    ]
    
    try:
        # Initialize the feature extractor
        extractor = FeatureExtractor()
        print(" OpenCV feature extractor initialized successfully")
        
        # Extract features from the test grid
        features = extractor.extract_features(test_grid, "test_game")
        
        if 'error' in features:
            print(f" Feature extraction failed: {features['error']}")
            return False
        
        print(" Feature extraction completed successfully")
        print(f" Detected {len(features.get('objects', []))} objects")
        print(f" Found {len(features.get('relationships', []))} relationships")
        print(f" Identified {len(features.get('patterns', []))} patterns")
        print(f" Summary: {features.get('summary', 'No summary available')}")
        
        # Test change detection
        output_grid = [
            [0, 2, 0],
            [2, 2, 2],
            [0, 2, 0]
        ]
        
        changes = extractor.detect_changes(test_grid, output_grid, "test_game")
        
        if 'error' in changes:
            print(f" Change detection failed: {changes['error']}")
            return False
        
        print(" Change detection completed successfully")
        print(f" Detected {len(changes.get('changes', []))} changes")
        print(f" Change summary: {changes.get('change_summary', 'No summary available')}")
        
        print(" OpenCV integration test PASSED!")
        return True
        
    except Exception as e:
        print(f" OpenCV integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_opencv_integration()
    sys.exit(0 if success else 1)
