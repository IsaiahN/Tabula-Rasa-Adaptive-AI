"""
Sleep Breakthrough Detection Algorithms

Advanced algorithms for detecting and processing breakthrough moments during sleep cycles.
These breakthroughs represent significant learning events that should be prioritized
for memory consolidation and pattern strengthening.
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)

class BreakthroughDetector:
    """
    Detects breakthrough moments in learning and experience data.
    
    Breakthroughs are defined as:
    1. Significant performance improvements
    2. Novel pattern discoveries
    3. Strategic insights
    4. Successful problem-solving sequences
    5. Energy and motivation spikes
    """
    
    def __init__(
        self,
        breakthrough_threshold: float = 0.7,
        novelty_threshold: float = 0.6,
        performance_window: int = 50,
        pattern_window: int = 100,
        energy_spike_threshold: float = 0.3
    ):
        self.breakthrough_threshold = breakthrough_threshold
        self.novelty_threshold = novelty_threshold
        self.performance_window = performance_window
        self.pattern_window = pattern_window
        self.energy_spike_threshold = energy_spike_threshold
        
        # Breakthrough tracking
        self.breakthrough_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=performance_window)
        self.pattern_history = deque(maxlen=pattern_window)
        self.energy_history = deque(maxlen=50)
        
        # Breakthrough types
        self.breakthrough_types = {
            'performance_spike': 0,
            'novel_pattern': 0,
            'strategic_insight': 0,
            'energy_breakthrough': 0,
            'learning_acceleration': 0,
            'problem_solving': 0
        }
        
        # Pattern recognition for breakthroughs
        self.pattern_embeddings = []
        self.novel_patterns = set()
        
    def detect_breakthroughs(
        self, 
        experience_data: Dict[str, Any],
        current_performance: float,
        learning_progress: float,
        energy_level: float,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect breakthrough moments in the given experience data.
        
        Args:
            experience_data: Experience data to analyze
            current_performance: Current performance metric
            learning_progress: Learning progress value
            energy_level: Current energy level
            context: Additional context data
            
        Returns:
            List of detected breakthroughs with metadata
        """
        breakthroughs = []
        
        # Update history
        self.performance_history.append(current_performance)
        self.energy_history.append(energy_level)
        
        # 1. Performance Spike Detection
        performance_breakthrough = self._detect_performance_spike(current_performance)
        if performance_breakthrough:
            breakthroughs.append(performance_breakthrough)
            
        # 2. Novel Pattern Detection
        pattern_breakthrough = self._detect_novel_pattern(experience_data)
        if pattern_breakthrough:
            breakthroughs.append(pattern_breakthrough)
            
        # 3. Strategic Insight Detection
        insight_breakthrough = self._detect_strategic_insight(experience_data, context)
        if insight_breakthrough:
            breakthroughs.append(insight_breakthrough)
            
        # 4. Energy Breakthrough Detection
        energy_breakthrough = self._detect_energy_breakthrough(energy_level)
        if energy_breakthrough:
            breakthroughs.append(energy_breakthrough)
            
        # 5. Learning Acceleration Detection
        learning_breakthrough = self._detect_learning_acceleration(learning_progress)
        if learning_breakthrough:
            breakthroughs.append(learning_breakthrough)
            
        # 6. Problem Solving Breakthrough Detection
        problem_solving_breakthrough = self._detect_problem_solving_breakthrough(experience_data)
        if problem_solving_breakthrough:
            breakthroughs.append(problem_solving_breakthrough)
        
        # Update breakthrough history
        for breakthrough in breakthroughs:
            self.breakthrough_history.append(breakthrough)
            self.breakthrough_types[breakthrough['type']] += 1
            
        return breakthroughs
    
    def _detect_performance_spike(self, current_performance: float) -> Optional[Dict[str, Any]]:
        """Detect significant performance improvements."""
        if len(self.performance_history) < 10:
            return None
            
        # Calculate performance trend
        recent_performance = list(self.performance_history)[-10:]
        baseline_performance = np.mean(list(self.performance_history)[:-10]) if len(self.performance_history) > 10 else np.mean(recent_performance)
        
        # Check for significant improvement
        performance_improvement = (current_performance - baseline_performance) / max(baseline_performance, 0.01)
        
        if performance_improvement > self.breakthrough_threshold:
            return {
                'type': 'performance_spike',
                'magnitude': performance_improvement,
                'current_performance': current_performance,
                'baseline_performance': baseline_performance,
                'confidence': min(performance_improvement, 1.0),
                'timestamp': datetime.now().isoformat()
            }
            
        return None
    
    def _detect_novel_pattern(self, experience_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect novel patterns in experience data."""
        # Extract pattern features
        pattern_features = self._extract_pattern_features(experience_data)
        
        if not pattern_features:
            return None
            
        # Check for novelty against existing patterns
        novelty_score = self._calculate_pattern_novelty(pattern_features)
        
        if novelty_score > self.novelty_threshold:
            # Store as novel pattern
            pattern_id = f"pattern_{len(self.pattern_embeddings)}"
            self.pattern_embeddings.append(pattern_features)
            self.novel_patterns.add(pattern_id)
            
            return {
                'type': 'novel_pattern',
                'pattern_id': pattern_id,
                'novelty_score': novelty_score,
                'features': pattern_features,
                'confidence': novelty_score,
                'timestamp': datetime.now().isoformat()
            }
            
        return None
    
    def _detect_strategic_insight(self, experience_data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Detect strategic insights from experience data."""
        if not context:
            return None
            
        # Look for strategic indicators
        strategic_indicators = []
        
        # Check for action effectiveness insights
        if 'action_effectiveness' in context:
            action_eff = context['action_effectiveness']
            for action_id, effectiveness in action_eff.items():
                if effectiveness.get('success_rate', 0) > 0.8:
                    strategic_indicators.append(f"high_success_action_{action_id}")
                    
        # Check for coordinate intelligence insights
        if 'coordinate_intelligence' in context:
            coord_intel = context['coordinate_intelligence']
            if coord_intel.get('success_zones', 0) > 5:
                strategic_indicators.append('coordinate_success_zones')
                
        # Check for boundary detection insights
        if 'boundary_data' in context:
            boundary_data = context['boundary_data']
            if boundary_data.get('detection_accuracy', 0) > 0.9:
                strategic_indicators.append('boundary_detection_accuracy')
        
        if len(strategic_indicators) >= 2:
            return {
                'type': 'strategic_insight',
                'indicators': strategic_indicators,
                'insight_count': len(strategic_indicators),
                'confidence': min(len(strategic_indicators) / 5.0, 1.0),
                'timestamp': datetime.now().isoformat()
            }
            
        return None
    
    def _detect_energy_breakthrough(self, energy_level: float) -> Optional[Dict[str, Any]]:
        """Detect energy breakthroughs and motivation spikes."""
        if len(self.energy_history) < 5:
            return None
            
        # Calculate energy trend
        recent_energy = list(self.energy_history)[-5:]
        baseline_energy = np.mean(list(self.energy_history)[:-5]) if len(self.energy_history) > 5 else np.mean(recent_energy)
        
        # Check for energy spike
        energy_change = (energy_level - baseline_energy) / max(baseline_energy, 0.01)
        
        if energy_change > self.energy_spike_threshold:
            return {
                'type': 'energy_breakthrough',
                'energy_change': energy_change,
                'current_energy': energy_level,
                'baseline_energy': baseline_energy,
                'confidence': min(energy_change, 1.0),
                'timestamp': datetime.now().isoformat()
            }
            
        return None
    
    def _detect_learning_acceleration(self, learning_progress: float) -> Optional[Dict[str, Any]]:
        """Detect learning acceleration patterns."""
        if learning_progress <= 0:
            return None
            
        # Check for sustained learning progress
        if learning_progress > 0.5:  # High learning progress
            return {
                'type': 'learning_acceleration',
                'learning_progress': learning_progress,
                'acceleration_rate': learning_progress * 2.0,
                'confidence': min(learning_progress, 1.0),
                'timestamp': datetime.now().isoformat()
            }
            
        return None
    
    def _detect_problem_solving_breakthrough(self, experience_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect problem-solving breakthroughs."""
        # Look for successful problem-solving indicators
        if 'reward' in experience_data and experience_data['reward'] > 0.8:
            return {
                'type': 'problem_solving',
                'reward': experience_data['reward'],
                'success_level': experience_data['reward'],
                'confidence': experience_data['reward'],
                'timestamp': datetime.now().isoformat()
            }
            
        return None
    
    def _extract_pattern_features(self, experience_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract pattern features from experience data."""
        features = []
        
        # Extract numerical features
        if 'learning_progress' in experience_data:
            features.append(experience_data['learning_progress'])
        if 'reward' in experience_data:
            features.append(experience_data['reward'])
        if 'action' in experience_data:
            features.append(experience_data['action'])
            
        # Extract state features if available
        if 'state' in experience_data and hasattr(experience_data['state'], 'visual'):
            visual_tensor = experience_data['state'].visual
            if isinstance(visual_tensor, torch.Tensor):
                # Extract basic visual features
                features.extend([
                    torch.mean(visual_tensor).item(),
                    torch.std(visual_tensor).item(),
                    torch.max(visual_tensor).item()
                ])
        
        return np.array(features) if features else None
    
    def _calculate_pattern_novelty(self, pattern_features: np.ndarray) -> float:
        """Calculate novelty score for a pattern."""
        if not self.pattern_embeddings:
            return 1.0  # First pattern is always novel
            
        # Calculate distances to existing patterns
        distances = []
        for existing_pattern in self.pattern_embeddings:
            if len(existing_pattern) == len(pattern_features):
                distance = np.linalg.norm(pattern_features - existing_pattern)
                distances.append(distance)
        
        if not distances:
            return 1.0
            
        # Novelty is inverse of minimum distance
        min_distance = min(distances)
        novelty = 1.0 / (1.0 + min_distance)
        
        return novelty
    
    def get_breakthrough_statistics(self) -> Dict[str, Any]:
        """Get breakthrough detection statistics."""
        total_breakthroughs = len(self.breakthrough_history)
        
        return {
            'total_breakthroughs': total_breakthroughs,
            'breakthrough_types': self.breakthrough_types.copy(),
            'novel_patterns_discovered': len(self.novel_patterns),
            'recent_breakthroughs': list(self.breakthrough_history)[-10:],
            'breakthrough_rate': total_breakthroughs / max(len(self.performance_history), 1),
            'average_confidence': np.mean([b.get('confidence', 0) for b in self.breakthrough_history]) if self.breakthrough_history else 0.0
        }


class SleepBreakthroughProcessor:
    """
    Processes breakthrough moments during sleep cycles for enhanced memory consolidation.
    """
    
    def __init__(self, breakthrough_detector: BreakthroughDetector):
        self.breakthrough_detector = breakthrough_detector
        self.breakthrough_consolidation_strength = 2.0
        self.novel_pattern_boost = 1.5
        self.strategic_insight_boost = 2.5
        
    def process_breakthroughs_during_sleep(
        self, 
        experiences: List[Dict[str, Any]],
        memory_matrix: torch.Tensor,
        usage_vector: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Process breakthrough moments during sleep for enhanced consolidation.
        
        Args:
            experiences: List of experiences to analyze
            memory_matrix: Memory matrix to consolidate
            usage_vector: Memory usage vector
            
        Returns:
            Consolidation results with breakthrough processing
        """
        breakthrough_results = {
            'breakthroughs_detected': 0,
            'breakthroughs_processed': 0,
            'consolidation_operations': 0,
            'memory_strengthening_applied': 0.0,
            'novel_patterns_consolidated': 0,
            'strategic_insights_consolidated': 0
        }
        
        # Detect breakthroughs in experiences
        all_breakthroughs = []
        for exp in experiences:
            breakthroughs = self.breakthrough_detector.detect_breakthroughs(
                experience_data=exp,
                current_performance=exp.get('performance', 0.0),
                learning_progress=exp.get('learning_progress', 0.0),
                energy_level=exp.get('energy_level', 50.0),
                context=exp.get('context', {})
            )
            all_breakthroughs.extend(breakthroughs)
        
        breakthrough_results['breakthroughs_detected'] = len(all_breakthroughs)
        
        if not all_breakthroughs:
            return breakthrough_results
        
        # Process each breakthrough for memory consolidation
        for breakthrough in all_breakthroughs:
            consolidation_result = self._consolidate_breakthrough_memory(
                breakthrough, memory_matrix, usage_vector
            )
            
            breakthrough_results['breakthroughs_processed'] += 1
            breakthrough_results['consolidation_operations'] += consolidation_result['operations']
            breakthrough_results['memory_strengthening_applied'] += consolidation_result['strengthening']
            
            if breakthrough['type'] == 'novel_pattern':
                breakthrough_results['novel_patterns_consolidated'] += 1
            elif breakthrough['type'] == 'strategic_insight':
                breakthrough_results['strategic_insights_consolidated'] += 1
        
        return breakthrough_results
    
    def _consolidate_breakthrough_memory(
        self, 
        breakthrough: Dict[str, Any], 
        memory_matrix: torch.Tensor, 
        usage_vector: torch.Tensor
    ) -> Dict[str, Any]:
        """Consolidate memory based on breakthrough type."""
        breakthrough_type = breakthrough['type']
        confidence = breakthrough.get('confidence', 0.5)
        
        # Calculate consolidation strength based on breakthrough type
        if breakthrough_type == 'performance_spike':
            consolidation_strength = self.breakthrough_consolidation_strength * confidence
        elif breakthrough_type == 'novel_pattern':
            consolidation_strength = self.novel_pattern_boost * confidence
        elif breakthrough_type == 'strategic_insight':
            consolidation_strength = self.strategic_insight_boost * confidence
        elif breakthrough_type == 'energy_breakthrough':
            consolidation_strength = 1.5 * confidence
        elif breakthrough_type == 'learning_acceleration':
            consolidation_strength = 2.0 * confidence
        elif breakthrough_type == 'problem_solving':
            consolidation_strength = 2.5 * confidence
        else:
            consolidation_strength = 1.0 * confidence
        
        # Apply consolidation to high-usage memory locations
        high_usage_mask = usage_vector > 0.1
        operations = 0
        strengthening = 0.0
        
        if high_usage_mask.any():
            # Apply breakthrough-specific consolidation
            memory_matrix[high_usage_mask] *= (1.0 + consolidation_strength)
            operations = high_usage_mask.sum().item()
            strengthening = consolidation_strength
            
            logger.info(f"Breakthrough consolidation: {breakthrough_type} with strength {consolidation_strength:.2f}")
        
        return {
            'operations': operations,
            'strengthening': strengthening,
            'breakthrough_type': breakthrough_type,
            'consolidation_strength': consolidation_strength
        }
    
    def get_breakthrough_consolidation_report(self) -> Dict[str, Any]:
        """Get report on breakthrough consolidation processing."""
        stats = self.breakthrough_detector.get_breakthrough_statistics()
        
        return {
            'breakthrough_detector_stats': stats,
            'consolidation_strength': self.breakthrough_consolidation_strength,
            'novel_pattern_boost': self.novel_pattern_boost,
            'strategic_insight_boost': self.strategic_insight_boost,
            'total_breakthroughs_processed': stats['total_breakthroughs']
        }


def create_sleep_breakthrough_system(
    breakthrough_threshold: float = 0.7,
    novelty_threshold: float = 0.6,
    performance_window: int = 50
) -> Tuple[BreakthroughDetector, SleepBreakthroughProcessor]:
    """
    Create a complete sleep breakthrough detection and processing system.
    
    Args:
        breakthrough_threshold: Threshold for breakthrough detection
        novelty_threshold: Threshold for novel pattern detection
        performance_window: Window size for performance analysis
        
    Returns:
        Tuple of (BreakthroughDetector, SleepBreakthroughProcessor)
    """
    detector = BreakthroughDetector(
        breakthrough_threshold=breakthrough_threshold,
        novelty_threshold=novelty_threshold,
        performance_window=performance_window
    )
    
    processor = SleepBreakthroughProcessor(detector)
    
    return detector, processor
