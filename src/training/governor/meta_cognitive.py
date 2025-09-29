"""
Meta-Cognitive Controller

Handles meta-cognitive processes including self-reflection, learning adaptation,
and system monitoring.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)

# Global singleton instance
_meta_cognitive_controller_instance = None

class MetaCognitiveController:
    """Controls meta-cognitive processes and self-reflection."""
    
    def __init__(self, max_reflections: int = 1000):
        self.max_reflections = max_reflections
        self.reflections = deque(maxlen=max_reflections)
        self.learning_adaptations = deque(maxlen=max_reflections)
        self.performance_history = deque(maxlen=100)
        self.meta_cognitive_state = {
            'self_awareness_level': 0.5,
            'learning_confidence': 0.5,
            'adaptation_rate': 1.0,
            'reflection_frequency': 10,  # Reflect every 10 actions
            'last_reflection': None
        }
        self.architect = None
        self._initialize_architect()
    
    def _initialize_architect(self) -> None:
        """Initialize the architect system."""
        try:
            from src.core.architect import create_architect
            self.architect = create_architect()
            # Note: No need to log here as Architect logs its own initialization
            logger.debug("MetaCognitiveController connected to Architect singleton")
        except ImportError as e:
            logger.warning(f"Architect not available: {e}")
            self.architect = None
        except Exception as e:
            logger.error(f"Error initializing architect: {e}")
            self.architect = None
    
    def should_reflect(self, action_count: int) -> bool:
        """Determine if the system should perform self-reflection."""
        reflection_freq = self.meta_cognitive_state['reflection_frequency']
        return action_count % reflection_freq == 0
    
    def perform_reflection(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform meta-cognitive self-reflection."""
        try:
            reflection = {
                'timestamp': datetime.now(),
                'context': context,
                'self_awareness_level': self.meta_cognitive_state['self_awareness_level'],
                'learning_confidence': self.meta_cognitive_state['learning_confidence'],
                'insights': [],
                'adaptations': [],
                'recommendations': []
            }
            
            # Analyze performance
            performance_insights = self._analyze_performance(context)
            reflection['insights'].extend(performance_insights)
            
            # Analyze learning patterns
            learning_insights = self._analyze_learning_patterns(context)
            reflection['insights'].extend(learning_insights)
            
            # Generate adaptations
            adaptations = self._generate_adaptations(context, reflection['insights'])
            reflection['adaptations'].extend(adaptations)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(context, reflection['insights'])
            reflection['recommendations'].extend(recommendations)
            
            # Update meta-cognitive state
            self._update_meta_cognitive_state(reflection)
            
            # Record reflection
            self.reflections.append(reflection)
            
            logger.info(f"Performed reflection with {len(reflection['insights'])} insights")
            return reflection
            
        except Exception as e:
            logger.error(f"Error performing reflection: {e}")
            return {
                'timestamp': datetime.now(),
                'context': context,
                'error': str(e),
                'insights': [],
                'adaptations': [],
                'recommendations': []
            }
    
    def _analyze_performance(self, context: Dict[str, Any]) -> List[str]:
        """Analyze current performance and generate insights."""
        insights = []
        
        try:
            # Analyze memory usage
            memory_usage = context.get('memory_usage', 0)
            if memory_usage > 1000:
                insights.append(f"High memory usage detected: {memory_usage:.1f}MB")
            elif memory_usage < 100:
                insights.append(f"Low memory usage: {memory_usage:.1f}MB - may indicate underutilization")
            
            # Analyze action effectiveness
            action_effectiveness = context.get('action_effectiveness', 0.0)
            if action_effectiveness > 0.8:
                insights.append(f"High action effectiveness: {action_effectiveness:.2f}")
            elif action_effectiveness < 0.3:
                insights.append(f"Low action effectiveness: {action_effectiveness:.2f} - may need strategy adjustment")
            
            # Analyze learning progress
            learning_progress = context.get('learning_progress', 0.0)
            if learning_progress > 0.7:
                insights.append(f"Strong learning progress: {learning_progress:.2f}")
            elif learning_progress < 0.2:
                insights.append(f"Slow learning progress: {learning_progress:.2f} - may need approach change")
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            insights.append(f"Error in performance analysis: {e}")
        
        return insights
    
    def _analyze_learning_patterns(self, context: Dict[str, Any]) -> List[str]:
        """Analyze learning patterns and generate insights."""
        insights = []
        
        try:
            # Analyze pattern recognition
            patterns_learned = context.get('patterns_learned', 0)
            if patterns_learned > 10:
                insights.append(f"Strong pattern recognition: {patterns_learned} patterns learned")
            elif patterns_learned < 3:
                insights.append(f"Limited pattern recognition: {patterns_learned} patterns - may need more diverse training")
            
            # Analyze adaptation rate
            adaptation_rate = self.meta_cognitive_state['adaptation_rate']
            if adaptation_rate > 1.5:
                insights.append(f"High adaptation rate: {adaptation_rate:.2f} - system is very responsive")
            elif adaptation_rate < 0.5:
                insights.append(f"Low adaptation rate: {adaptation_rate:.2f} - system may be too conservative")
            
        except Exception as e:
            logger.error(f"Error analyzing learning patterns: {e}")
            insights.append(f"Error in learning pattern analysis: {e}")
        
        return insights
    
    def _generate_adaptations(self, context: Dict[str, Any], insights: List[str]) -> List[Dict[str, Any]]:
        """Generate system adaptations based on insights."""
        adaptations = []
        
        try:
            # Memory optimization adaptations
            if any("High memory usage" in insight for insight in insights):
                adaptations.append({
                    'type': 'memory_optimization',
                    'action': 'reduce_memory_footprint',
                    'priority': 'high',
                    'description': 'Reduce memory usage through cleanup and optimization'
                })
            
            # Learning rate adaptations
            if any("Low learning progress" in insight for insight in insights):
                adaptations.append({
                    'type': 'learning_rate',
                    'action': 'increase_learning_rate',
                    'priority': 'medium',
                    'description': 'Increase learning rate to improve progress'
                })
            
            # Strategy adaptations
            if any("Low action effectiveness" in insight for insight in insights):
                adaptations.append({
                    'type': 'strategy',
                    'action': 'adjust_action_strategy',
                    'priority': 'high',
                    'description': 'Adjust action selection strategy for better effectiveness'
                })
            
        except Exception as e:
            logger.error(f"Error generating adaptations: {e}")
        
        return adaptations
    
    def _generate_recommendations(self, context: Dict[str, Any], insights: List[str]) -> List[str]:
        """Generate recommendations based on insights."""
        recommendations = []
        
        try:
            # Performance recommendations
            if any("High memory usage" in insight for insight in insights):
                recommendations.append("Consider implementing more aggressive memory cleanup")
            
            if any("Low action effectiveness" in insight for insight in insights):
                recommendations.append("Review and update action selection algorithms")
            
            if any("Slow learning progress" in insight for insight in insights):
                recommendations.append("Increase training diversity and complexity")
            
            # General recommendations
            if len(insights) < 3:
                recommendations.append("Increase monitoring granularity for better insights")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _update_meta_cognitive_state(self, reflection: Dict[str, Any]) -> None:
        """Update meta-cognitive state based on reflection."""
        try:
            # Update self-awareness based on insight quality
            insight_count = len(reflection.get('insights', []))
            if insight_count > 5:
                self.meta_cognitive_state['self_awareness_level'] = min(1.0, 
                    self.meta_cognitive_state['self_awareness_level'] + 0.1)
            elif insight_count < 2:
                self.meta_cognitive_state['self_awareness_level'] = max(0.0,
                    self.meta_cognitive_state['self_awareness_level'] - 0.05)
            
            # Update learning confidence based on adaptations
            adaptation_count = len(reflection.get('adaptations', []))
            if adaptation_count > 0:
                self.meta_cognitive_state['learning_confidence'] = min(1.0,
                    self.meta_cognitive_state['learning_confidence'] + 0.05)
            
            # Update reflection frequency based on system state
            if self.meta_cognitive_state['self_awareness_level'] > 0.8:
                self.meta_cognitive_state['reflection_frequency'] = max(5,
                    self.meta_cognitive_state['reflection_frequency'] - 1)
            elif self.meta_cognitive_state['self_awareness_level'] < 0.3:
                self.meta_cognitive_state['reflection_frequency'] = min(20,
                    self.meta_cognitive_state['reflection_frequency'] + 2)
            
            self.meta_cognitive_state['last_reflection'] = reflection['timestamp']
            
        except Exception as e:
            logger.error(f"Error updating meta-cognitive state: {e}")
    
    def get_meta_cognitive_status(self) -> Dict[str, Any]:
        """Get current meta-cognitive status."""
        return {
            'state': self.meta_cognitive_state.copy(),
            'total_reflections': len(self.reflections),
            'total_adaptations': len(self.learning_adaptations),
            'architect_available': self.architect is not None,
            'recent_reflections': list(self.reflections)[-5:] if self.reflections else []
        }
    
    def get_reflection_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get reflection history."""
        if limit is None:
            return list(self.reflections)
        return list(self.reflections)[-limit:]
    
    def reset_meta_cognitive_state(self) -> None:
        """Reset meta-cognitive state."""
        self.reflections.clear()
        self.learning_adaptations.clear()
        self.performance_history.clear()
        self.meta_cognitive_state = {
            'self_awareness_level': 0.5,
            'learning_confidence': 0.5,
            'adaptation_rate': 1.0,
            'reflection_frequency': 10,
            'last_reflection': None
        }
        logger.info("Meta-cognitive state reset")


def create_meta_cognitive_controller(max_reflections: int = 1000) -> MetaCognitiveController:
    """Create or get the singleton MetaCognitiveController instance."""
    global _meta_cognitive_controller_instance
    if _meta_cognitive_controller_instance is None:
        print("  LEARNING MANAGER: Creating singleton MetaCognitiveController instance")
        _meta_cognitive_controller_instance = MetaCognitiveController(max_reflections=max_reflections)
    return _meta_cognitive_controller_instance


def get_meta_cognitive_controller() -> Optional[MetaCognitiveController]:
    """Get the singleton MetaCognitiveController instance if it exists."""
    return _meta_cognitive_controller_instance
