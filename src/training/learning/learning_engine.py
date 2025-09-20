"""
Learning Engine

Core learning algorithms and adaptive learning mechanisms.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class LearningEngine:
    """Core learning engine for adaptive learning."""
    
    def __init__(self, learning_rate: float = 0.1, memory_decay: float = 0.95):
        self.learning_rate = learning_rate
        self.memory_decay = memory_decay
        self.learning_history = []
        self.adaptation_metrics = {
            'total_learnings': 0,
            'successful_adaptations': 0,
            'failed_adaptations': 0,
            'learning_confidence': 0.5
        }
        self.learning_strategies = {
            'exploration': 0.3,
            'exploitation': 0.7,
            'random': 0.1
        }
    
    def learn_from_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from a single experience."""
        try:
            learning_result = {
                'timestamp': datetime.now(),
                'experience_id': experience.get('id', 'unknown'),
                'learning_type': 'experience',
                'insights': [],
                'adaptations': [],
                'confidence': 0.0
            }
            
            # Extract learning insights
            insights = self._extract_insights(experience)
            learning_result['insights'] = insights
            
            # Generate adaptations
            adaptations = self._generate_adaptations(experience, insights)
            learning_result['adaptations'] = adaptations
            
            # Calculate learning confidence
            confidence = self._calculate_learning_confidence(experience, insights)
            learning_result['confidence'] = confidence
            
            # Update learning metrics
            self._update_learning_metrics(learning_result)
            
            # Record learning
            self.learning_history.append(learning_result)
            
            logger.debug(f"Learned from experience: {len(insights)} insights, {len(adaptations)} adaptations")
            return learning_result
            
        except Exception as e:
            logger.error(f"Error learning from experience: {e}")
            return {
                'timestamp': datetime.now(),
                'experience_id': experience.get('id', 'unknown'),
                'learning_type': 'error',
                'insights': [],
                'adaptations': [],
                'confidence': 0.0,
                'error': str(e)
            }
    
    def adapt_strategy(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt learning strategy based on performance."""
        try:
            adaptation = {
                'timestamp': datetime.now(),
                'adaptation_type': 'strategy',
                'changes': {},
                'reasoning': '',
                'confidence': 0.0
            }
            
            # Analyze performance
            performance_score = performance_metrics.get('overall_score', 0.0)
            action_effectiveness = performance_metrics.get('action_effectiveness', 0.0)
            learning_progress = performance_metrics.get('learning_progress', 0.0)
            
            # Adjust learning rate
            if performance_score > 0.8:
                # High performance - increase exploitation
                self.learning_strategies['exploitation'] = min(0.9, self.learning_strategies['exploitation'] + 0.1)
                self.learning_strategies['exploration'] = max(0.1, self.learning_strategies['exploration'] - 0.05)
                adaptation['changes']['exploitation'] = self.learning_strategies['exploitation']
                adaptation['reasoning'] += "High performance detected, increasing exploitation. "
            elif performance_score < 0.3:
                # Low performance - increase exploration
                self.learning_strategies['exploration'] = min(0.5, self.learning_strategies['exploration'] + 0.1)
                self.learning_strategies['exploitation'] = max(0.3, self.learning_strategies['exploitation'] - 0.05)
                adaptation['changes']['exploration'] = self.learning_strategies['exploration']
                adaptation['reasoning'] += "Low performance detected, increasing exploration. "
            
            # Adjust learning rate
            if action_effectiveness > 0.7:
                self.learning_rate = min(0.2, self.learning_rate + 0.01)
                adaptation['changes']['learning_rate'] = self.learning_rate
                adaptation['reasoning'] += "High action effectiveness, increasing learning rate. "
            elif action_effectiveness < 0.3:
                self.learning_rate = max(0.01, self.learning_rate - 0.01)
                adaptation['changes']['learning_rate'] = self.learning_rate
                adaptation['reasoning'] += "Low action effectiveness, decreasing learning rate. "
            
            # Calculate adaptation confidence
            adaptation['confidence'] = self._calculate_adaptation_confidence(performance_metrics)
            
            # Update metrics
            if adaptation['confidence'] > 0.5:
                self.adaptation_metrics['successful_adaptations'] += 1
            else:
                self.adaptation_metrics['failed_adaptations'] += 1
            
            logger.info(f"Adapted strategy: {adaptation['reasoning']}")
            return adaptation
            
        except Exception as e:
            logger.error(f"Error adapting strategy: {e}")
            return {
                'timestamp': datetime.now(),
                'adaptation_type': 'error',
                'changes': {},
                'reasoning': f'Error in adaptation: {e}',
                'confidence': 0.0
            }
    
    def _extract_insights(self, experience: Dict[str, Any]) -> List[str]:
        """Extract learning insights from an experience."""
        insights = []
        
        try:
            # Analyze action effectiveness
            action_effectiveness = experience.get('action_effectiveness', 0.0)
            if action_effectiveness > 0.8:
                insights.append(f"High action effectiveness: {action_effectiveness:.2f}")
            elif action_effectiveness < 0.2:
                insights.append(f"Low action effectiveness: {action_effectiveness:.2f}")
            
            # Analyze pattern recognition
            patterns_detected = experience.get('patterns_detected', 0)
            if patterns_detected > 0:
                insights.append(f"Detected {patterns_detected} patterns")
            
            # Analyze learning progress
            learning_progress = experience.get('learning_progress', 0.0)
            if learning_progress > 0.5:
                insights.append(f"Significant learning progress: {learning_progress:.2f}")
            
            # Analyze error patterns
            errors = experience.get('errors', [])
            if errors:
                insights.append(f"Encountered {len(errors)} errors - learning opportunity")
            
        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            insights.append(f"Error in insight extraction: {e}")
        
        return insights
    
    def _generate_adaptations(self, experience: Dict[str, Any], insights: List[str]) -> List[Dict[str, Any]]:
        """Generate adaptations based on experience and insights."""
        adaptations = []
        
        try:
            # Memory adaptations
            if any("High action effectiveness" in insight for insight in insights):
                adaptations.append({
                    'type': 'memory',
                    'action': 'strengthen_memory',
                    'target': 'action_patterns',
                    'description': 'Strengthen memory of effective action patterns'
                })
            
            # Strategy adaptations
            if any("Low action effectiveness" in insight for insight in insights):
                adaptations.append({
                    'type': 'strategy',
                    'action': 'adjust_action_selection',
                    'target': 'action_algorithm',
                    'description': 'Adjust action selection algorithm'
                })
            
            # Learning rate adaptations
            if any("Significant learning progress" in insight for insight in insights):
                adaptations.append({
                    'type': 'learning',
                    'action': 'increase_learning_rate',
                    'target': 'global_learning',
                    'description': 'Increase learning rate due to good progress'
                })
            
        except Exception as e:
            logger.error(f"Error generating adaptations: {e}")
        
        return adaptations
    
    def _calculate_learning_confidence(self, experience: Dict[str, Any], insights: List[str]) -> float:
        """Calculate confidence in learning from experience."""
        try:
            confidence = 0.5  # Base confidence
            
            # Increase confidence based on insight quality
            insight_count = len(insights)
            if insight_count > 3:
                confidence += 0.2
            elif insight_count > 1:
                confidence += 0.1
            
            # Increase confidence based on experience quality
            action_effectiveness = experience.get('action_effectiveness', 0.0)
            confidence += action_effectiveness * 0.3
            
            # Increase confidence based on learning progress
            learning_progress = experience.get('learning_progress', 0.0)
            confidence += learning_progress * 0.2
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating learning confidence: {e}")
            return 0.0
    
    def _calculate_adaptation_confidence(self, performance_metrics: Dict[str, Any]) -> float:
        """Calculate confidence in strategy adaptation."""
        try:
            confidence = 0.5  # Base confidence
            
            # Increase confidence based on performance consistency
            performance_score = performance_metrics.get('overall_score', 0.0)
            confidence += performance_score * 0.3
            
            # Increase confidence based on learning progress
            learning_progress = performance_metrics.get('learning_progress', 0.0)
            confidence += learning_progress * 0.2
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating adaptation confidence: {e}")
            return 0.0
    
    def _update_learning_metrics(self, learning_result: Dict[str, Any]) -> None:
        """Update learning metrics."""
        try:
            self.adaptation_metrics['total_learnings'] += 1
            
            # Update learning confidence
            if learning_result['confidence'] > 0.7:
                self.adaptation_metrics['learning_confidence'] = min(1.0,
                    self.adaptation_metrics['learning_confidence'] + 0.01)
            elif learning_result['confidence'] < 0.3:
                self.adaptation_metrics['learning_confidence'] = max(0.0,
                    self.adaptation_metrics['learning_confidence'] - 0.01)
            
        except Exception as e:
            logger.error(f"Error updating learning metrics: {e}")
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status."""
        return {
            'learning_rate': self.learning_rate,
            'memory_decay': self.memory_decay,
            'strategies': self.learning_strategies.copy(),
            'metrics': self.adaptation_metrics.copy(),
            'total_learnings': len(self.learning_history),
            'recent_learnings': self.learning_history[-5:] if self.learning_history else []
        }
    
    def get_learning_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get learning history."""
        if limit is None:
            return self.learning_history.copy()
        return self.learning_history[-limit:]
    
    def reset_learning_engine(self) -> None:
        """Reset learning engine state."""
        self.learning_history.clear()
        self.adaptation_metrics = {
            'total_learnings': 0,
            'successful_adaptations': 0,
            'failed_adaptations': 0,
            'learning_confidence': 0.5
        }
        self.learning_strategies = {
            'exploration': 0.3,
            'exploitation': 0.7,
            'random': 0.1
        }
        logger.info("Learning engine reset")
