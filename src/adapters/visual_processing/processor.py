"""
ARC Visual Processor

Processes ARC visual grids into format compatible with Adaptive Learning Agent.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class ARCVisualProcessor(nn.Module):
    """
    Processes ARC visual grids into format compatible with Adaptive Learning Agent.
    
    ARC grids are typically small (up to 64x64) with discrete color values (0-9).
    This processor converts them to the expected visual format.
    """
    
    def __init__(self, target_size: Tuple[int, int, int] = (3, 64, 64)):
        super().__init__()
        self.target_size = target_size
        self.color_embedding = nn.Embedding(10, 3)  # 10 colors -> 3 channels
        
    def forward(self, arc_frame: List[List[List[int]]]) -> torch.Tensor:
        """
        Convert ARC frame to visual tensor.
        
        Args:
            arc_frame: ARC frame data [height][width][channels] with values 0-9
            
        Returns:
            visual_tensor: [batch_size, channels, height, width]
        """
        if not arc_frame:
            # Return empty frame
            return torch.zeros(1, *self.target_size)
            
        # Convert to numpy array
        frame_array = np.array(arc_frame)
        
        # Handle different input formats
        if len(frame_array.shape) == 2:
            # Single channel grid
            frame_array = frame_array[:, :, np.newaxis]
        elif len(frame_array.shape) == 3 and frame_array.shape[2] == 1:
            # Already has channel dimension
            pass
        else:
            logger.warning(f"Unexpected frame shape: {frame_array.shape}")
            
        # Convert to tensor
        frame_tensor = torch.tensor(frame_array, dtype=torch.long)
        
        # Embed colors to RGB-like representation
        if frame_tensor.dim() == 3:
            embedded = self.color_embedding(frame_tensor)  # [H, W, C, 3]
            embedded = embedded.mean(dim=2)  # Average across original channels
            embedded = embedded.permute(2, 0, 1)  # [3, H, W]
        else:
            embedded = self.color_embedding(frame_tensor)  # [H, W, 3]
            embedded = embedded.permute(2, 0, 1)  # [3, H, W]
            
        # Resize to target size if needed
        if embedded.shape[1:] != self.target_size[1:]:
            embedded = torch.nn.functional.interpolate(
                embedded.unsqueeze(0), 
                size=self.target_size[1:], 
                mode='nearest'
            ).squeeze(0)
            
        return embedded.unsqueeze(0)  # Add batch dimension
    
    def preprocess_frame(self, frame_data: dict) -> torch.Tensor:
        """Preprocess frame data for the processor."""
        try:
            # Extract grid data
            if 'grid' in frame_data:
                grid = frame_data['grid']
            elif 'input' in frame_data:
                grid = frame_data['input']
            else:
                logger.warning("No grid data found in frame")
                return torch.zeros(1, *self.target_size)
            
            return self.forward(grid)
            
        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}")
            return torch.zeros(1, *self.target_size)
    
    def get_feature_maps(self, arc_frame: List[List[List[int]]]) -> dict:
        """Extract feature maps from ARC frame."""
        try:
            visual_tensor = self.forward(arc_frame)
            
            # Extract different feature representations
            features = {
                'visual_tensor': visual_tensor,
                'shape': visual_tensor.shape,
                'mean': visual_tensor.mean().item(),
                'std': visual_tensor.std().item(),
                'min': visual_tensor.min().item(),
                'max': visual_tensor.max().item()
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting feature maps: {e}")
            return {'error': str(e)}
    
    def resize_frame(self, frame: List[List[int]], target_size: Tuple[int, int]) -> List[List[int]]:
        """Resize frame to target size."""
        try:
            if not frame:
                return [[0] * target_size[1] for _ in range(target_size[0])]
            
            # Convert to numpy array
            frame_array = np.array(frame)
            
            # Resize using OpenCV if available
            try:
                import cv2
                resized = cv2.resize(frame_array, target_size, interpolation=cv2.INTER_NEAREST)
                return resized.tolist()
            except ImportError:
                # Fallback to simple resizing
                logger.warning("OpenCV not available, using simple resizing")
                return self._simple_resize(frame, target_size)
                
        except Exception as e:
            logger.error(f"Error resizing frame: {e}")
            return frame
    
    def _simple_resize(self, frame: List[List[int]], target_size: Tuple[int, int]) -> List[List[int]]:
        """Simple fallback resizing without OpenCV."""
        try:
            if not frame:
                return [[0] * target_size[1] for _ in range(target_size[0])]
            
            height, width = len(frame), len(frame[0])
            target_height, target_width = target_size
            
            # Create new frame
            new_frame = [[0] * target_width for _ in range(target_height)]
            
            # Simple nearest neighbor resizing
            for i in range(target_height):
                for j in range(target_width):
                    # Map to original coordinates
                    orig_i = int(i * height / target_height)
                    orig_j = int(j * width / target_width)
                    
                    # Clamp to valid range
                    orig_i = min(orig_i, height - 1)
                    orig_j = min(orig_j, width - 1)
                    
                    new_frame[i][j] = frame[orig_i][orig_j]
            
            return new_frame
            
        except Exception as e:
            logger.error(f"Error in simple resize: {e}")
            return frame
