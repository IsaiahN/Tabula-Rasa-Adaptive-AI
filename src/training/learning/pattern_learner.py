"""
Pattern Learner

Specialized pattern recognition and learning algorithms.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np

logger = logging.getLogger(__name__)

class PatternLearner:
    """Learns and recognizes patterns in training data."""
    
    def __init__(self, pattern_threshold: float = 0.7, max_patterns: int = 1000):
        self.pattern_threshold = pattern_threshold
        self.max_patterns = max_patterns
        self.learned_patterns = {}
        self.pattern_frequencies = defaultdict(int)
        self.pattern_effectiveness = defaultdict(list)
        self.pattern_confidence = defaultdict(float)
        self.pattern_history = []
    
    def learn_pattern(self, pattern_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Learn a new pattern from data."""
        try:
            pattern_id = self._generate_pattern_id(pattern_data)
            
            # Extract pattern features
            features = self._extract_pattern_features(pattern_data)
            
            # Calculate pattern confidence
            confidence = self._calculate_pattern_confidence(features, pattern_data)
            
            if confidence < self.pattern_threshold:
                logger.debug(f"Pattern confidence {confidence:.2f} below threshold {self.pattern_threshold}")
                return None
            
            # Store pattern
            pattern = {
                'id': pattern_id,
                'features': features,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'effectiveness': pattern_data.get('effectiveness', 0.0),
                'context': pattern_data.get('context', {}),
                'frequency': 1
            }
            
            self.learned_patterns[pattern_id] = pattern
            self.pattern_frequencies[pattern_id] = 1
            self.pattern_effectiveness[pattern_id] = [pattern['effectiveness']]
            self.pattern_confidence[pattern_id] = confidence
            
            # Record pattern learning
            self.pattern_history.append({
                'timestamp': datetime.now(),
                'pattern_id': pattern_id,
                'confidence': confidence,
                'effectiveness': pattern['effectiveness']
            })
            
            logger.info(f"Learned pattern {pattern_id} with confidence {confidence:.2f}")
            return pattern
            
        except Exception as e:
            logger.error(f"Error learning pattern: {e}")
            return None
    
    def recognize_pattern(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recognize patterns in input data."""
        try:
            input_features = self._extract_pattern_features(input_data)
            matches = []
            
            for pattern_id, pattern in self.learned_patterns.items():
                similarity = self._calculate_similarity(input_features, pattern['features'])
                
                if similarity >= self.pattern_threshold:
                    match = {
                        'pattern_id': pattern_id,
                        'similarity': similarity,
                        'confidence': pattern['confidence'],
                        'effectiveness': self._get_average_effectiveness(pattern_id),
                        'pattern': pattern
                    }
                    matches.append(match)
            
            # Sort by similarity
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.debug(f"Recognized {len(matches)} patterns in input data")
            return matches
            
        except Exception as e:
            logger.error(f"Error recognizing patterns: {e}")
            return []
    
    def update_pattern_effectiveness(self, pattern_id: str, effectiveness: float) -> None:
        """Update pattern effectiveness based on new data."""
        try:
            if pattern_id in self.learned_patterns:
                # Update effectiveness history
                self.pattern_effectiveness[pattern_id].append(effectiveness)
                
                # Keep only recent effectiveness values
                if len(self.pattern_effectiveness[pattern_id]) > 100:
                    self.pattern_effectiveness[pattern_id] = self.pattern_effectiveness[pattern_id][-50:]
                
                # Update pattern frequency
                self.pattern_frequencies[pattern_id] += 1
                
                # Update pattern in learned patterns
                self.learned_patterns[pattern_id]['frequency'] = self.pattern_frequencies[pattern_id]
                self.learned_patterns[pattern_id]['effectiveness'] = self._get_average_effectiveness(pattern_id)
                
                logger.debug(f"Updated effectiveness for pattern {pattern_id}: {effectiveness:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating pattern effectiveness: {e}")
    
    def _generate_pattern_id(self, pattern_data: Dict[str, Any]) -> str:
        """Generate a unique ID for a pattern."""
        try:
            # Create a hashable representation of the pattern
            pattern_type = pattern_data.get('type', 'unknown')
            features = pattern_data.get('features', {})
            context = pattern_data.get('context', {})
            
            # Create a simple hash
            pattern_str = f"{pattern_type}_{str(sorted(features.items()))}_{str(sorted(context.items()))}"
            return f"pattern_{hash(pattern_str)}"
            
        except Exception as e:
            logger.error(f"Error generating pattern ID: {e}")
            return f"pattern_{int(datetime.now().timestamp())}"
    
    def _extract_pattern_features(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from pattern data."""
        try:
            features = {}
            
            # Extract basic features
            features['type'] = pattern_data.get('type', 'unknown')
            features['complexity'] = pattern_data.get('complexity', 0.0)
            features['size'] = pattern_data.get('size', 0)
            
            # Extract action features
            actions = pattern_data.get('actions', [])
            features['action_count'] = len(actions)
            features['action_types'] = [action.get('type', 'unknown') for action in actions]
            
            # Extract coordinate features
            coordinates = pattern_data.get('coordinates', [])
            features['coordinate_count'] = len(coordinates)
            if coordinates:
                features['coordinate_bounds'] = {
                    'min_x': min(coord[0] for coord in coordinates),
                    'max_x': max(coord[0] for coord in coordinates),
                    'min_y': min(coord[1] for coord in coordinates),
                    'max_y': max(coord[1] for coord in coordinates)
                }
            
            # Extract context features
            context = pattern_data.get('context', {})
            features['context_keys'] = list(context.keys())
            features['context_size'] = len(context)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting pattern features: {e}")
            return {}
    
    def _calculate_pattern_confidence(self, features: Dict[str, Any], pattern_data: Dict[str, Any]) -> float:
        """Calculate confidence in a pattern."""
        try:
            confidence = 0.5  # Base confidence
            
            # Increase confidence based on feature completeness
            feature_count = len(features)
            if feature_count > 5:
                confidence += 0.2
            elif feature_count > 3:
                confidence += 0.1
            
            # Increase confidence based on pattern complexity
            complexity = features.get('complexity', 0.0)
            confidence += complexity * 0.2
            
            # Increase confidence based on action count
            action_count = features.get('action_count', 0)
            if action_count > 3:
                confidence += 0.1
            elif action_count > 1:
                confidence += 0.05
            
            # Increase confidence based on effectiveness
            effectiveness = pattern_data.get('effectiveness', 0.0)
            confidence += effectiveness * 0.3
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating pattern confidence: {e}")
            return 0.0
    
    def _calculate_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate similarity between two feature sets."""
        try:
            if not features1 or not features2:
                return 0.0
            
            # Calculate Jaccard similarity for categorical features
            categorical_features = ['type', 'action_types', 'context_keys']
            jaccard_similarity = 0.0
            
            for feature in categorical_features:
                if feature in features1 and feature in features2:
                    set1 = set(features1[feature]) if isinstance(features1[feature], list) else {features1[feature]}
                    set2 = set(features2[feature]) if isinstance(features2[feature], list) else {features2[feature]}
                    
                    if set1 and set2:
                        intersection = len(set1.intersection(set2))
                        union = len(set1.union(set2))
                        jaccard_similarity += intersection / union if union > 0 else 0.0
            
            jaccard_similarity /= len(categorical_features)
            
            # Calculate numerical similarity
            numerical_features = ['complexity', 'size', 'action_count', 'coordinate_count']
            numerical_similarity = 0.0
            
            for feature in numerical_features:
                if feature in features1 and feature in features2:
                    val1 = features1[feature]
                    val2 = features2[feature]
                    
                    if val1 == 0 and val2 == 0:
                        numerical_similarity += 1.0
                    elif val1 != 0 and val2 != 0:
                        numerical_similarity += 1.0 - abs(val1 - val2) / max(val1, val2)
            
            numerical_similarity /= len(numerical_features)
            
            # Combine similarities
            total_similarity = (jaccard_similarity + numerical_similarity) / 2.0
            
            return min(1.0, max(0.0, total_similarity))
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _get_average_effectiveness(self, pattern_id: str) -> float:
        """Get average effectiveness for a pattern."""
        try:
            effectiveness_values = self.pattern_effectiveness.get(pattern_id, [])
            if not effectiveness_values:
                return 0.0
            return sum(effectiveness_values) / len(effectiveness_values)
        except Exception as e:
            logger.error(f"Error calculating average effectiveness: {e}")
            return 0.0
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get pattern learning statistics."""
        try:
            total_patterns = len(self.learned_patterns)
            high_confidence_patterns = sum(1 for p in self.learned_patterns.values() if p['confidence'] > 0.8)
            effective_patterns = sum(1 for p in self.learned_patterns.values() if p['effectiveness'] > 0.7)
            
            return {
                'total_patterns': total_patterns,
                'high_confidence_patterns': high_confidence_patterns,
                'effective_patterns': effective_patterns,
                'pattern_threshold': self.pattern_threshold,
                'max_patterns': self.max_patterns,
                'pattern_learning_history': len(self.pattern_history)
            }
        except Exception as e:
            logger.error(f"Error getting pattern statistics: {e}")
            return {}
    
    def cleanup_old_patterns(self, max_age_days: int = 7) -> int:
        """Clean up old patterns to prevent memory bloat."""
        try:
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            patterns_to_remove = []
            for pattern_id, pattern in self.learned_patterns.items():
                if pattern['timestamp'] < cutoff_date:
                    patterns_to_remove.append(pattern_id)
            
            for pattern_id in patterns_to_remove:
                del self.learned_patterns[pattern_id]
                del self.pattern_frequencies[pattern_id]
                del self.pattern_effectiveness[pattern_id]
                del self.pattern_confidence[pattern_id]
            
            logger.info(f"Cleaned up {len(patterns_to_remove)} old patterns")
            return len(patterns_to_remove)
            
        except Exception as e:
            logger.error(f"Error cleaning up old patterns: {e}")
            return 0
    
    def reset_pattern_learner(self) -> None:
        """Reset pattern learner state."""
        self.learned_patterns.clear()
        self.pattern_frequencies.clear()
        self.pattern_effectiveness.clear()
        self.pattern_confidence.clear()
        self.pattern_history.clear()
        logger.info("Pattern learner reset")
