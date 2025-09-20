"""
Knowledge Transfer

Handles knowledge transfer between different ARC tasks.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class KnowledgeTransfer:
    """Handles knowledge transfer between different ARC tasks."""
    
    def __init__(self, transfer_threshold: float = 0.6):
        self.transfer_threshold = transfer_threshold
        self.transfer_history = []
        self.successful_transfers = []
        
        # Transfer statistics
        self.stats = {
            'transfer_attempts': 0,
            'successful_transfers': 0,
            'failed_transfers': 0,
            'transfer_effectiveness': 0.0
        }
    
    def transfer_knowledge(self, source_patterns: List[Any], target_game_id: str, 
                          target_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Transfer knowledge from source patterns to target game."""
        try:
            self.stats['transfer_attempts'] += 1
            
            # Find applicable patterns for transfer
            applicable_patterns = self._find_applicable_patterns(source_patterns, target_context)
            
            if not applicable_patterns:
                logger.info(f"No applicable patterns found for transfer to {target_game_id}")
                return []
            
            # Generate transfer recommendations
            transfer_recommendations = self._generate_transfer_recommendations(
                applicable_patterns, target_context
            )
            
            # Record transfer attempt
            transfer_record = {
                'target_game_id': target_game_id,
                'applicable_patterns': len(applicable_patterns),
                'recommendations': len(transfer_recommendations),
                'timestamp': self._get_timestamp()
            }
            self.transfer_history.append(transfer_record)
            
            logger.info(f"Generated {len(transfer_recommendations)} transfer recommendations for {target_game_id}")
            return transfer_recommendations
            
        except Exception as e:
            logger.error(f"Error transferring knowledge: {e}")
            return []
    
    def _find_applicable_patterns(self, source_patterns: List[Any], target_context: Dict[str, Any]) -> List[Any]:
        """Find patterns that are applicable to the target context."""
        try:
            applicable_patterns = []
            
            for pattern in source_patterns:
                if self._is_pattern_applicable(pattern, target_context):
                    applicable_patterns.append(pattern)
            
            return applicable_patterns
            
        except Exception as e:
            logger.error(f"Error finding applicable patterns: {e}")
            return []
    
    def _is_pattern_applicable(self, pattern: Any, target_context: Dict[str, Any]) -> bool:
        """Check if a pattern is applicable to the target context."""
        try:
            # Check pattern confidence
            if pattern.confidence < self.transfer_threshold:
                return False
            
            # Check pattern success rate
            if pattern.success_rate < 0.5:
                return False
            
            # Check context compatibility
            target_actions = target_context.get('available_actions', [])
            if target_actions:
                # Check if pattern actions are available in target
                pattern_actions = pattern.actions
                if not all(action in target_actions for action in pattern_actions):
                    return False
            
            # Check pattern type compatibility
            target_pattern_types = target_context.get('pattern_types', [])
            if target_pattern_types and pattern.pattern_type not in target_pattern_types:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking pattern applicability: {e}")
            return False
    
    def _generate_transfer_recommendations(self, applicable_patterns: List[Any], 
                                         target_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate transfer recommendations from applicable patterns."""
        try:
            recommendations = []
            
            # Group patterns by type
            patterns_by_type = defaultdict(list)
            for pattern in applicable_patterns:
                patterns_by_type[pattern.pattern_type].append(pattern)
            
            # Generate recommendations for each pattern type
            for pattern_type, patterns in patterns_by_type.items():
                if not patterns:
                    continue
                
                # Calculate average success rate for this pattern type
                success_rates = [p.success_rate for p in patterns]
                avg_success_rate = np.mean(success_rates)
                
                # Calculate average confidence
                confidences = [p.confidence for p in patterns]
                avg_confidence = np.mean(confidences)
                
                # Generate recommendation
                recommendation = {
                    'pattern_type': pattern_type,
                    'recommended_actions': self._extract_recommended_actions(patterns),
                    'success_rate': avg_success_rate,
                    'confidence': avg_confidence,
                    'pattern_count': len(patterns),
                    'transfer_strategy': self._get_transfer_strategy(pattern_type),
                    'adaptation_notes': self._generate_adaptation_notes(patterns, target_context)
                }
                
                recommendations.append(recommendation)
            
            # Sort by success rate and confidence
            recommendations.sort(key=lambda x: x['success_rate'] * x['confidence'], reverse=True)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating transfer recommendations: {e}")
            return []
    
    def _extract_recommended_actions(self, patterns: List[Any]) -> List[str]:
        """Extract recommended actions from patterns."""
        try:
            # Count action frequencies
            action_counts = defaultdict(int)
            for pattern in patterns:
                for action in pattern.actions:
                    action_counts[action] += 1
            
            # Sort by frequency
            sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Return most frequent actions
            return [action for action, count in sorted_actions[:5]]
            
        except Exception as e:
            logger.error(f"Error extracting recommended actions: {e}")
            return []
    
    def _get_transfer_strategy(self, pattern_type: str) -> str:
        """Get transfer strategy for a pattern type."""
        strategies = {
            'visual': 'Focus on visual pattern recognition and apply similar visual processing',
            'spatial': 'Use spatial reasoning and coordinate-based actions',
            'logical': 'Apply logical sequence patterns and reasoning steps',
            'sequential': 'Follow established action sequences and timing patterns'
        }
        
        return strategies.get(pattern_type, 'Apply general pattern-based approach')
    
    def _generate_adaptation_notes(self, patterns: List[Any], target_context: Dict[str, Any]) -> List[str]:
        """Generate adaptation notes for patterns."""
        try:
            notes = []
            
            # Check for action availability
            target_actions = target_context.get('available_actions', [])
            if target_actions:
                pattern_actions = set()
                for pattern in patterns:
                    pattern_actions.update(pattern.actions)
                
                unavailable_actions = pattern_actions - set(target_actions)
                if unavailable_actions:
                    notes.append(f"Note: Actions {list(unavailable_actions)} may not be available in target context")
            
            # Check for pattern complexity
            avg_pattern_length = np.mean([len(p.actions) for p in patterns])
            if avg_pattern_length > 5:
                notes.append("Note: Patterns are complex - consider breaking into smaller steps")
            
            # Check for success rate consistency
            success_rates = [p.success_rate for p in patterns]
            if np.std(success_rates) > 0.3:
                notes.append("Note: Success rates vary significantly - test patterns carefully")
            
            return notes
            
        except Exception as e:
            logger.error(f"Error generating adaptation notes: {e}")
            return []
    
    def evaluate_transfer_success(self, target_game_id: str, success: bool, 
                                performance_improvement: float = 0.0):
        """Evaluate the success of a knowledge transfer."""
        try:
            # Record transfer result
            transfer_result = {
                'target_game_id': target_game_id,
                'success': success,
                'performance_improvement': performance_improvement,
                'timestamp': self._get_timestamp()
            }
            
            if success:
                self.successful_transfers.append(transfer_result)
                self.stats['successful_transfers'] += 1
            else:
                self.stats['failed_transfers'] += 1
            
            # Update transfer effectiveness
            total_attempts = self.stats['transfer_attempts']
            if total_attempts > 0:
                self.stats['transfer_effectiveness'] = self.stats['successful_transfers'] / total_attempts
            
            logger.info(f"Transfer evaluation: {target_game_id} - Success: {success}, Improvement: {performance_improvement:.2f}")
            
        except Exception as e:
            logger.error(f"Error evaluating transfer success: {e}")
    
    def get_transfer_statistics(self) -> Dict[str, Any]:
        """Get statistics about knowledge transfers."""
        try:
            return {
                'total_attempts': self.stats['transfer_attempts'],
                'successful_transfers': self.stats['successful_transfers'],
                'failed_transfers': self.stats['failed_transfers'],
                'transfer_effectiveness': self.stats['transfer_effectiveness'],
                'recent_transfers': self.transfer_history[-10:] if self.transfer_history else [],
                'successful_transfer_games': [t['target_game_id'] for t in self.successful_transfers]
            }
            
        except Exception as e:
            logger.error(f"Error getting transfer statistics: {e}")
            return {}
    
    def get_transfer_recommendations(self, game_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get transfer recommendations for a specific game."""
        try:
            # This would typically use stored patterns and insights
            # For now, return a placeholder
            return [{
                'pattern_type': 'general',
                'recommended_actions': ['ACTION1', 'ACTION2', 'ACTION3'],
                'success_rate': 0.7,
                'confidence': 0.8,
                'transfer_strategy': 'Apply general pattern-based approach',
                'adaptation_notes': ['Test patterns carefully in new context']
            }]
            
        except Exception as e:
            logger.error(f"Error getting transfer recommendations: {e}")
            return []
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        import time
        return str(time.time())
