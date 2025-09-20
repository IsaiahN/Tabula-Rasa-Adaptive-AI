"""
ARC Pattern Recognizer

Recognizes and learns patterns from ARC tasks.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import time
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ARCPattern:
    """Represents a learned pattern from ARC tasks."""
    pattern_type: str  # 'visual', 'spatial', 'logical', 'sequential'
    description: str
    conditions: Dict[str, Any]
    actions: List[str]
    success_rate: float
    confidence: float
    games_seen: List[str]
    timestamp: float

class ARCPatternRecognizer:
    """Recognizes and learns patterns from ARC tasks."""
    
    def __init__(self, pattern_memory_size: int = 1000):
        self.pattern_memory_size = pattern_memory_size
        self.patterns = deque(maxlen=pattern_memory_size)
        
        # Pattern storage by type
        self.visual_patterns = defaultdict(int)
        self.spatial_patterns = defaultdict(int)
        self.logical_patterns = defaultdict(int)
        self.sequential_patterns = defaultdict(int)
        
        # Pattern recognition statistics
        self.stats = {
            'patterns_discovered': 0,
            'patterns_validated': 0,
            'patterns_applied': 0
        }
    
    def recognize_patterns(self, episode_data: Dict[str, Any]) -> List[ARCPattern]:
        """Recognize patterns from episode data."""
        try:
            discovered_patterns = []
            
            # Extract different types of patterns
            visual_patterns = self._extract_visual_patterns(episode_data)
            spatial_patterns = self._extract_spatial_patterns(episode_data)
            logical_patterns = self._extract_logical_patterns(episode_data)
            sequential_patterns = self._extract_sequential_patterns(episode_data)
            
            # Combine all patterns
            all_patterns = (visual_patterns + spatial_patterns + 
                          logical_patterns + sequential_patterns)
            
            # Validate and store patterns
            for pattern in all_patterns:
                if self._validate_pattern(pattern):
                    self._store_pattern(pattern)
                    discovered_patterns.append(pattern)
                    self.stats['patterns_discovered'] += 1
            
            logger.info(f"Recognized {len(discovered_patterns)} patterns from episode")
            return discovered_patterns
            
        except Exception as e:
            logger.error(f"Error recognizing patterns: {e}")
            return []
    
    def _extract_visual_patterns(self, episode_data: Dict[str, Any]) -> List[ARCPattern]:
        """Extract visual patterns from episode data."""
        patterns = []
        
        try:
            frames = episode_data.get('frames', [])
            actions = episode_data.get('actions', [])
            success = episode_data.get('success', False)
            game_id = episode_data.get('game_id', 'unknown')
            
            if not frames or not actions:
                return patterns
            
            # Look for visual consistency patterns
            for i in range(len(frames) - 1):
                frame1 = frames[i]
                frame2 = frames[i + 1]
                action = actions[i] if i < len(actions) else 'unknown'
                
                # Check for visual changes
                visual_changes = self._analyze_visual_changes(frame1, frame2)
                
                if visual_changes:
                    pattern = ARCPattern(
                        pattern_type='visual',
                        description=f"Visual change pattern: {visual_changes['description']}",
                        conditions={
                            'action': action,
                            'change_type': visual_changes['type'],
                            'change_magnitude': visual_changes['magnitude']
                        },
                        actions=[action],
                        success_rate=1.0 if success else 0.0,
                        confidence=visual_changes['confidence'],
                        games_seen=[game_id],
                        timestamp=time.time()
                    )
                    patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"Error extracting visual patterns: {e}")
        
        return patterns
    
    def _extract_spatial_patterns(self, episode_data: Dict[str, Any]) -> List[ARCPattern]:
        """Extract spatial patterns from episode data."""
        patterns = []
        
        try:
            frames = episode_data.get('frames', [])
            actions = episode_data.get('actions', [])
            coordinates = episode_data.get('coordinates', [])
            success = episode_data.get('success', False)
            game_id = episode_data.get('game_id', 'unknown')
            
            if not frames or not actions:
                return patterns
            
            # Look for spatial movement patterns
            for i in range(len(actions) - 1):
                if i < len(coordinates) - 1:
                    coord1 = coordinates[i]
                    coord2 = coordinates[i + 1]
                    action1 = actions[i]
                    action2 = actions[i + 1]
                    
                    # Calculate movement vector
                    movement = self._calculate_movement_vector(coord1, coord2)
                    
                    if movement['magnitude'] > 0:
                        pattern = ARCPattern(
                            pattern_type='spatial',
                            description=f"Spatial movement: {movement['direction']} with magnitude {movement['magnitude']:.2f}",
                            conditions={
                                'action_sequence': [action1, action2],
                                'movement_vector': movement,
                                'distance': movement['magnitude']
                            },
                            actions=[action1, action2],
                            success_rate=1.0 if success else 0.0,
                            confidence=min(1.0, movement['magnitude'] / 10.0),
                            games_seen=[game_id],
                            timestamp=time.time()
                        )
                        patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"Error extracting spatial patterns: {e}")
        
        return patterns
    
    def _extract_logical_patterns(self, episode_data: Dict[str, Any]) -> List[ARCPattern]:
        """Extract logical patterns from episode data."""
        patterns = []
        
        try:
            actions = episode_data.get('actions', [])
            success = episode_data.get('success', False)
            game_id = episode_data.get('game_id', 'unknown')
            
            if len(actions) < 2:
                return patterns
            
            # Look for logical action sequences
            for i in range(len(actions) - 2):
                action_sequence = actions[i:i + 3]
                
                # Check for logical patterns
                if self._is_logical_sequence(action_sequence):
                    pattern = ARCPattern(
                        pattern_type='logical',
                        description=f"Logical sequence: {' -> '.join(action_sequence)}",
                        conditions={
                            'action_sequence': action_sequence,
                            'sequence_length': len(action_sequence)
                        },
                        actions=action_sequence,
                        success_rate=1.0 if success else 0.0,
                        confidence=0.8,  # High confidence for logical sequences
                        games_seen=[game_id],
                        timestamp=time.time()
                    )
                    patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"Error extracting logical patterns: {e}")
        
        return patterns
    
    def _extract_sequential_patterns(self, episode_data: Dict[str, Any]) -> List[ARCPattern]:
        """Extract sequential patterns from episode data."""
        patterns = []
        
        try:
            actions = episode_data.get('actions', [])
            success = episode_data.get('success', False)
            game_id = episode_data.get('game_id', 'unknown')
            
            if len(actions) < 3:
                return patterns
            
            # Look for repeating sequences
            for length in range(2, min(len(actions), 6)):
                for i in range(len(actions) - length + 1):
                    sequence = actions[i:i + length]
                    
                    # Check if this sequence appears elsewhere
                    if self._count_sequence_occurrences(sequence, actions) > 1:
                        pattern = ARCPattern(
                            pattern_type='sequential',
                            description=f"Repeating sequence: {' -> '.join(sequence)}",
                            conditions={
                                'sequence': sequence,
                                'occurrences': self._count_sequence_occurrences(sequence, actions),
                                'sequence_length': length
                            },
                            actions=sequence,
                            success_rate=1.0 if success else 0.0,
                            confidence=0.7,
                            games_seen=[game_id],
                            timestamp=time.time()
                        )
                        patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"Error extracting sequential patterns: {e}")
        
        return patterns
    
    def _analyze_visual_changes(self, frame1: List[List[int]], frame2: List[List[int]]) -> Optional[Dict[str, Any]]:
        """Analyze visual changes between two frames."""
        try:
            if not frame1 or not frame2:
                return None
            
            # Convert to numpy arrays for easier comparison
            arr1 = np.array(frame1)
            arr2 = np.array(frame2)
            
            # Ensure same shape
            if arr1.shape != arr2.shape:
                return None
            
            # Calculate differences
            diff = arr1 != arr2
            change_count = np.sum(diff)
            total_pixels = arr1.size
            
            if change_count == 0:
                return None
            
            change_ratio = change_count / total_pixels
            
            # Determine change type
            if change_ratio < 0.1:
                change_type = 'minor'
            elif change_ratio < 0.5:
                change_type = 'moderate'
            else:
                change_type = 'major'
            
            return {
                'type': change_type,
                'magnitude': change_ratio,
                'description': f"{change_type} visual change ({change_ratio:.1%} of pixels)",
                'confidence': min(1.0, change_ratio * 2)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing visual changes: {e}")
            return None
    
    def _calculate_movement_vector(self, coord1: Dict[str, int], coord2: Dict[str, int]) -> Dict[str, Any]:
        """Calculate movement vector between two coordinates."""
        try:
            x1, y1 = coord1.get('x', 0), coord1.get('y', 0)
            x2, y2 = coord2.get('x', 0), coord2.get('y', 0)
            
            dx = x2 - x1
            dy = y2 - y1
            magnitude = np.sqrt(dx**2 + dy**2)
            
            # Determine direction
            if magnitude == 0:
                direction = 'none'
            elif abs(dx) > abs(dy):
                direction = 'horizontal'
            else:
                direction = 'vertical'
            
            return {
                'dx': dx,
                'dy': dy,
                'magnitude': magnitude,
                'direction': direction
            }
            
        except Exception as e:
            logger.error(f"Error calculating movement vector: {e}")
            return {'dx': 0, 'dy': 0, 'magnitude': 0, 'direction': 'none'}
    
    def _is_logical_sequence(self, sequence: List[str]) -> bool:
        """Check if a sequence represents a logical pattern."""
        try:
            # Define logical patterns
            logical_patterns = [
                ['ACTION1', 'ACTION2', 'ACTION3'],  # Sequential progression
                ['ACTION4', 'ACTION5', 'ACTION6'],  # Alternative progression
                ['ACTION1', 'ACTION1', 'ACTION2'],  # Repetition followed by change
                ['ACTION2', 'ACTION1', 'ACTION2'],  # Alternating pattern
            ]
            
            return sequence in logical_patterns
            
        except Exception as e:
            logger.error(f"Error checking logical sequence: {e}")
            return False
    
    def _count_sequence_occurrences(self, sequence: List[str], actions: List[str]) -> int:
        """Count how many times a sequence appears in actions."""
        try:
            count = 0
            for i in range(len(actions) - len(sequence) + 1):
                if actions[i:i + len(sequence)] == sequence:
                    count += 1
            return count
            
        except Exception as e:
            logger.error(f"Error counting sequence occurrences: {e}")
            return 0
    
    def _validate_pattern(self, pattern: ARCPattern) -> bool:
        """Validate a pattern before storing it."""
        try:
            # Check minimum confidence
            if pattern.confidence < 0.3:
                return False
            
            # Check if pattern is too similar to existing ones
            for existing_pattern in self.patterns:
                if self._patterns_similar(pattern, existing_pattern):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating pattern: {e}")
            return False
    
    def _patterns_similar(self, pattern1: ARCPattern, pattern2: ARCPattern) -> bool:
        """Check if two patterns are similar."""
        try:
            # Same type and similar actions
            if (pattern1.pattern_type == pattern2.pattern_type and 
                pattern1.actions == pattern2.actions):
                return True
            
            # Similar descriptions
            if self._text_similarity(pattern1.description, pattern2.description) > 0.8:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking pattern similarity: {e}")
            return False
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity between two strings."""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union)
            
        except Exception as e:
            logger.error(f"Error calculating text similarity: {e}")
            return 0.0
    
    def _store_pattern(self, pattern: ARCPattern):
        """Store a validated pattern."""
        try:
            self.patterns.append(pattern)
            
            # Update pattern type counters
            if pattern.pattern_type == 'visual':
                self.visual_patterns[tuple(pattern.actions)] += 1
            elif pattern.pattern_type == 'spatial':
                self.spatial_patterns[tuple(pattern.actions)] += 1
            elif pattern.pattern_type == 'logical':
                self.logical_patterns[tuple(pattern.actions)] += 1
            elif pattern.pattern_type == 'sequential':
                self.sequential_patterns[tuple(pattern.actions)] += 1
            
        except Exception as e:
            logger.error(f"Error storing pattern: {e}")
    
    def get_patterns_by_type(self, pattern_type: str) -> List[ARCPattern]:
        """Get patterns of a specific type."""
        return [p for p in self.patterns if p.pattern_type == pattern_type]
    
    def get_patterns_by_game(self, game_id: str) -> List[ARCPattern]:
        """Get patterns seen in a specific game."""
        return [p for p in self.patterns if game_id in p.games_seen]
    
    def get_high_confidence_patterns(self, min_confidence: float = 0.7) -> List[ARCPattern]:
        """Get patterns with high confidence."""
        return [p for p in self.patterns if p.confidence >= min_confidence]
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about recognized patterns."""
        try:
            total_patterns = len(self.patterns)
            type_counts = defaultdict(int)
            confidence_scores = []
            
            for pattern in self.patterns:
                type_counts[pattern.pattern_type] += 1
                confidence_scores.append(pattern.confidence)
            
            return {
                'total_patterns': total_patterns,
                'type_distribution': dict(type_counts),
                'average_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
                'high_confidence_count': len([c for c in confidence_scores if c >= 0.7]),
                'stats': self.stats
            }
            
        except Exception as e:
            logger.error(f"Error getting pattern statistics: {e}")
            return {}
