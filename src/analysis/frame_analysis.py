"""Analysis utilities for detecting frame patterns and stagnation."""

from typing import List, Tuple, Dict, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class FrameAnalysisResult:
    """Results from analyzing frame patterns."""
    is_oscillating: bool
    is_stagnating: bool
    frame_similarity: float
    action_diversity: float
    net_progress_rate: float
    cycles_detected: List[List[str]]
    repeated_regions: List[Tuple[int, int, int, int]]  # x1, y1, x2, y2 of repeated regions
    action_frequencies: Dict[str, float]

def compare_frames(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Compare two frames and return similarity score (0-1)."""
    if frame1.shape != frame2.shape:
        return 0.0
    return np.mean(frame1 == frame2)

def detect_action_cycles(actions: List[str], window_size: int = 8) -> List[List[str]]:
    """Detect repeating patterns in action sequences."""
    cycles = []
    n = len(actions)
    
    for size in range(2, min(window_size + 1, n // 2 + 1)):
        for i in range(n - size * 2):
            window = actions[i:i + size]
            next_window = actions[i + size:i + size * 2]
            if window == next_window and window not in cycles:
                cycles.append(window)
    
    return cycles

def calculate_action_diversity(actions: List[str]) -> float:
    """Calculate diversity of actions (0-1)."""
    if not actions:
        return 0.0
        
    unique_actions = len(set(actions))
    total_actions = len(actions)
    diversity = unique_actions / total_actions
    
    # Penalize for immediate repetitions
    repetitions = sum(1 for i in range(len(actions)-1) if actions[i] == actions[i+1])
    diversity *= (1 - repetitions/total_actions)
    
    return diversity

def detect_repeated_regions(frames: List[np.ndarray], 
                          threshold: float = 0.95) -> List[Tuple[int, int, int, int]]:
    """Detect regions that remain unchanged across multiple frames."""
    if not frames or len(frames) < 2:
        return []
    
    height, width = frames[0].shape
    regions = []
    min_region_size = 3  # Minimum size to consider as a meaningful region
    
    for y in range(height - min_region_size):
        for x in range(width - min_region_size):
            for h in range(min_region_size, height - y + 1):
                for w in range(min_region_size, width - x + 1):
                    region = [frame[y:y+h, x:x+w] for frame in frames]
                    if all(np.array_equal(region[0], r) for r in region[1:]):
                        # Found an unchanged region
                        regions.append((x, y, x+w, y+h))
    
    # Merge overlapping regions
    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(regions):
            j = i + 1
            while j < len(regions):
                r1 = regions[i]
                r2 = regions[j]
                if (max(r1[0], r2[0]) < min(r1[2], r2[2]) and 
                    max(r1[1], r2[1]) < min(r1[3], r2[3])):
                    # Regions overlap, merge them
                    merged_region = (
                        min(r1[0], r2[0]),
                        min(r1[1], r2[1]),
                        max(r1[2], r2[2]),
                        max(r1[3], r2[3])
                    )
                    regions.pop(j)
                    regions[i] = merged_region
                    merged = True
                else:
                    j += 1
            if not merged:
                i += 1
    
    return regions

def analyze_frame_sequence(frames: List[np.ndarray], 
                         actions: List[str],
                         min_progress: float = 0.1) -> FrameAnalysisResult:
    """Analyze a sequence of frames for stagnation and oscillation patterns."""
    
    # Initialize metrics
    frame_similarities = []
    progress_rates = []
    
    # Compare consecutive frames
    for i in range(len(frames) - 1):
        similarity = compare_frames(frames[i], frames[i + 1])
        frame_similarities.append(similarity)
        
        # Calculate progress rate as percentage of cells changed
        progress = 1.0 - similarity
        progress_rates.append(progress)
    
    # Calculate action diversity
    action_diversity = calculate_action_diversity(actions)
    
    # Detect action cycles
    cycles = detect_action_cycles(actions)
    
    # Calculate action frequencies
    action_counts = {}
    for action in actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    total_actions = len(actions)
    action_frequencies = {
        action: count/total_actions 
        for action, count in action_counts.items()
    }
    
    # Detect repeated regions
    repeated_regions = detect_repeated_regions(frames)
    
    # Calculate average progress rate
    avg_progress = np.mean(progress_rates) if progress_rates else 0.0
    
    # Determine if stagnating
    is_stagnating = (
        avg_progress < min_progress or  # Low overall progress
        len(repeated_regions) > len(frames[0]) * 0.5 or  # Many unchanged regions
        action_diversity < 0.3  # Low action diversity
    )
    
    # Determine if oscillating
    is_oscillating = bool(cycles) and avg_progress < min_progress * 2
    
    # Calculate frame similarity as average similarity between consecutive frames
    frame_similarity = np.mean(frame_similarities) if frame_similarities else 1.0
    
    return FrameAnalysisResult(
        is_oscillating=bool(is_oscillating),
        is_stagnating=bool(is_stagnating),
        frame_similarity=float(frame_similarity),
        action_diversity=float(action_diversity),
        net_progress_rate=float(avg_progress),
        cycles_detected=cycles,
        repeated_regions=repeated_regions,
        action_frequencies=action_frequencies
    )