"""
OpenCV Utility Module for Tabula Rasa
=====================================

Centralized OpenCV utilities for consistent computer vision operations
across all systems (Director, Architect, Governor).

Features:
- Frame preprocessing and enhancement
- Motion detection and optical flow
- Object detection and tracking
- Pattern recognition and matching
- Edge detection and feature extraction
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class OpenCVProcessor:
    """Centralized OpenCV processor for consistent computer vision operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_detector = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def _ensure_proper_format(self, frame: np.ndarray) -> np.ndarray:
        """Ensure frame is in proper format for OpenCV operations."""
        try:
            # Handle different input formats
            if len(frame.shape) == 1:
                # 1D array - try to reshape to square
                size = int(np.sqrt(len(frame)))
                if size * size == len(frame):
                    frame = frame.reshape(size, size)
                else:
                    # Pad to make it square
                    size = int(np.ceil(np.sqrt(len(frame))))
                    padded = np.zeros(size * size, dtype=frame.dtype)
                    padded[:len(frame)] = frame
                    frame = padded.reshape(size, size)
            
            # CRITICAL FIX: Handle 3D arrays properly
            if len(frame.shape) == 3:
                # If it's (H, W, C) or (C, H, W), convert to 2D
                if frame.shape[0] == 64 and frame.shape[1] == 64:
                    # Likely (H, W, C) - take first channel or convert to grayscale
                    if frame.shape[2] == 1:
                        frame = frame[:, :, 0]
                    else:
                        # Convert to grayscale
                        frame = np.mean(frame, axis=2)
                elif frame.shape[2] == 64 and frame.shape[0] == 64:
                    # Likely (C, H, W) - take first channel
                    frame = frame[0, :, :]
                else:
                    # Squeeze and take first 2D slice
                    frame = frame.squeeze()
                    if len(frame.shape) > 2:
                        frame = frame[:, :, 0] if frame.shape[2] > 0 else frame[:, :]
            
            # Ensure 2D array
            if len(frame.shape) > 2:
                frame = frame.squeeze()
            
            # Convert to uint8
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # Final validation - ensure it's 2D
            if len(frame.shape) != 2:
                self.logger.warning(f"Frame still not 2D after processing: {frame.shape}")
                # Force 2D by taking first slice
                if len(frame.shape) == 3:
                    frame = frame[:, :, 0] if frame.shape[2] > 0 else frame[:, :]
                else:
                    # Create a default 64x64 frame
                    frame = np.zeros((64, 64), dtype=np.uint8)
            
            return frame
        except Exception as e:
            self.logger.warning(f"Frame format conversion failed: {e}")
            # Return a default 64x64 grayscale frame
            return np.zeros((64, 64), dtype=np.uint8)
        
    def preprocess_frame(self, frame: np.ndarray, 
                        blur_kernel: Tuple[int, int] = (5, 5),
                        enhance_contrast: bool = True) -> np.ndarray:
        """Preprocess frame for better analysis."""
        try:
            # Use proper format validation
            frame_processed = self._ensure_proper_format(frame)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(frame_processed, blur_kernel, 0)
            
            # Enhance contrast if requested
            if enhance_contrast:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(blurred)
                return enhanced
            
            return blurred
        except Exception as e:
            self.logger.warning(f"Frame preprocessing failed: {e}")
            return np.zeros((64, 64), dtype=np.uint8)
    
    def detect_edges(self, frame: np.ndarray, 
                    low_threshold: int = 50, 
                    high_threshold: int = 150) -> np.ndarray:
        """Detect edges using Canny edge detection."""
        try:
            frame_processed = self._ensure_proper_format(frame)
            edges = cv2.Canny(frame_processed, low_threshold, high_threshold)
            return edges
        except Exception as e:
            self.logger.warning(f"Edge detection failed: {e}")
            return np.zeros((64, 64), dtype=np.uint8)
    
    def find_contours(self, frame: np.ndarray, 
                     min_area: int = 10,
                     max_area: int = 10000) -> List[np.ndarray]:
        """Find contours with filtering."""
        try:
            frame_processed = self._ensure_proper_format(frame)
            
            # Use adaptive thresholding for better contour detection
            thresh = cv2.adaptiveThreshold(frame_processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    filtered_contours.append(contour)
            
            return filtered_contours
        except Exception as e:
            self.logger.warning(f"Contour detection failed: {e}")
            return []
    
    def detect_motion(self, prev_frame: np.ndarray, 
                     curr_frame: np.ndarray) -> Dict[str, Any]:
        """Detect motion between frames using optical flow."""
        try:
            prev_gray = self._ensure_proper_format(prev_frame)
            curr_gray = self._ensure_proper_format(curr_frame)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, None, None)
            
            if flow[0] is not None and flow[1] is not None:
                good_old = flow[0]
                good_new = flow[1]
                
                if len(good_old) > 0 and len(good_new) > 0:
                    # Calculate motion vectors
                    displacement = good_new - good_old
                    motion_magnitude = np.linalg.norm(displacement, axis=1)
                    avg_motion = np.mean(motion_magnitude)
                    
                    # Calculate motion direction
                    motion_direction = np.arctan2(displacement[:, 1], displacement[:, 0])
                    dominant_direction = np.median(motion_direction)
                    
                    return {
                        'motion_detected': avg_motion > 1.0,
                        'motion_magnitude': float(avg_motion),
                        'motion_direction': float(dominant_direction),
                        'motion_consistency': float(np.std(motion_direction)),
                        'feature_points': len(good_old)
                    }
            
            return {
                'motion_detected': False,
                'motion_magnitude': 0.0,
                'motion_direction': 0.0,
                'motion_consistency': 0.0,
                'feature_points': 0
            }
        except Exception as e:
            self.logger.warning(f"Motion detection failed: {e}")
            return {
                'motion_detected': False,
                'motion_magnitude': 0.0,
                'motion_direction': 0.0,
                'motion_consistency': 0.0,
                'feature_points': 0
            }
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in frame using contour analysis."""
        try:
            contours = self.find_contours(frame)
            objects = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # Calculate object properties
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 1.0
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Classify object shape
                epsilon = 0.02 * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                vertices = len(approx)
                
                shape_type = "unknown"
                if vertices == 3:
                    shape_type = "triangle"
                elif vertices == 4:
                    shape_type = "rectangle"
                elif circularity > 0.8:
                    shape_type = "circle"
                elif vertices > 8:
                    shape_type = "polygon"
                
                objects.append({
                    'contour': contour,
                    'area': int(area),
                    'perimeter': float(perimeter),
                    'bounding_box': (x, y, w, h),
                    'aspect_ratio': float(aspect_ratio),
                    'circularity': float(circularity),
                    'shape_type': shape_type,
                    'vertices': vertices,
                    'center': (x + w//2, y + h//2)
                })
            
            return objects
        except Exception as e:
            self.logger.warning(f"Object detection failed: {e}")
            return []
    
    def detect_lines(self, frame: np.ndarray, 
                    min_line_length: int = 30,
                    max_line_gap: int = 10) -> List[Dict[str, Any]]:
        """Detect lines in frame using Hough line detection."""
        try:
            edges = self.detect_edges(frame)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                  minLineLength=min_line_length, maxLineGap=max_line_gap)
            
            detected_lines = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    angle = np.arctan2(y2-y1, x2-x1)
                    
                    # Classify line orientation
                    orientation = "horizontal" if abs(angle) < np.pi/4 or abs(angle) > 3*np.pi/4 else "vertical"
                    
                    detected_lines.append({
                        'start': (x1, y1),
                        'end': (x2, y2),
                        'length': float(length),
                        'angle': float(angle),
                        'orientation': orientation
                    })
            
            return detected_lines
        except Exception as e:
            self.logger.warning(f"Line detection failed: {e}")
            return []
    
    def calculate_frame_complexity(self, frame: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive frame complexity metrics."""
        try:
            frame_processed = self._ensure_proper_format(frame)
            
            # Edge density
            edges = self.detect_edges(frame_processed)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Texture complexity using Laplacian variance
            laplacian_var = cv2.Laplacian(frame_processed, cv2.CV_64F).var()
            
            # Color complexity
            unique_colors = len(set(frame.flatten()))
            
            # Object count
            objects = self.detect_objects(frame_processed)
            object_count = len(objects)
            
            # Line count
            lines = self.detect_lines(frame_processed)
            line_count = len(lines)
            
            return {
                'edge_density': float(edge_density),
                'texture_complexity': float(laplacian_var),
                'color_complexity': float(unique_colors),
                'object_count': int(object_count),
                'line_count': int(line_count),
                'overall_complexity': float(edge_density * 1000 + laplacian_var * 0.1 + unique_colors * 0.01)
            }
        except Exception as e:
            self.logger.warning(f"Complexity calculation failed: {e}")
            return {
                'edge_density': 0.0,
                'texture_complexity': 0.0,
                'color_complexity': 0.0,
                'object_count': 0,
                'line_count': 0,
                'overall_complexity': 0.0
            }
    
    def template_match(self, frame: np.ndarray, 
                      template: np.ndarray, 
                      threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find template matches in frame."""
        try:
            processed_frame = self.preprocess_frame(frame)
            processed_template = self.preprocess_frame(template)
            
            # Perform template matching
            result = cv2.matchTemplate(processed_frame, processed_template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            
            matches = []
            for pt in zip(*locations[::-1]):
                matches.append({
                    'position': pt,
                    'confidence': float(result[pt[1], pt[0]]),
                    'template_size': processed_template.shape
                })
            
            return matches
        except Exception as e:
            self.logger.warning(f"Template matching failed: {e}")
            return []

# Global instance for consistent usage
opencv_processor = OpenCVProcessor()
