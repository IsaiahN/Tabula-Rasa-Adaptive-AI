"""
ARC Meta Learning System

Enhanced meta-learning system specifically designed for ARC tasks.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import time
import numpy as np

from ..pattern_recognition import ARCPatternRecognizer
from ..insight_extraction import ARCInsightExtractor
from ..knowledge_transfer import KnowledgeTransfer

logger = logging.getLogger(__name__)

class ARCMetaLearningSystem:
    """
    Enhanced meta-learning system specifically designed for ARC tasks.
    
    This system learns patterns across different ARC games and develops
    general reasoning strategies that can be applied to new tasks.
    """
    
    def __init__(
        self,
        pattern_memory_size: int = 1000,
        insight_threshold: float = 0.7,
        cross_validation_threshold: int = 3
    ):
        self.pattern_memory_size = pattern_memory_size
        self.insight_threshold = insight_threshold
        self.cross_validation_threshold = cross_validation_threshold
        
        # Initialize components
        self.pattern_recognizer = ARCPatternRecognizer(pattern_memory_size)
        self.insight_extractor = ARCInsightExtractor(insight_threshold)
        self.knowledge_transfer = KnowledgeTransfer()
        
        # ARC-specific storage
        self.game_histories = defaultdict(list)
        self.strategy_effectiveness = defaultdict(list)
        
        # Learning statistics
        self.stats = {
            'patterns_discovered': 0,
            'insights_generated': 0,
            'successful_transfers': 0,
            'games_analyzed': 0
        }
        
        logger.info("ARC Meta-Learning System initialized")
    
    def analyze_game_episode(
        self,
        game_id: str,
        episode_data: Dict[str, Any],
        success: bool,
        final_score: int
    ) -> Dict[str, Any]:
        """
        Analyze a completed game episode to extract patterns and insights.
        
        Args:
            game_id: Identifier for the ARC game
            episode_data: Complete episode data including frames, actions, reasoning
            success: Whether the episode was successful
            final_score: Final score achieved
            
        Returns:
            Analysis results including patterns and insights
        """
        try:
            # Store episode in game history
            episode_record = {
                'episode_data': episode_data,
                'success': success,
                'final_score': final_score,
                'timestamp': time.time()
            }
            self.game_histories[game_id].append(episode_record)
            self.stats['games_analyzed'] += 1
            
            # Extract patterns
            patterns = self.pattern_recognizer.recognize_patterns(episode_data)
            self.stats['patterns_discovered'] += len(patterns)
            
            # Extract insights
            insights = self.insight_extractor.extract_insights(patterns, self.game_histories)
            self.stats['insights_generated'] += len(insights)
            
            # Update strategy effectiveness
            self._update_strategy_effectiveness(game_id, success, final_score)
            
            # Generate analysis summary
            analysis_summary = {
                'game_id': game_id,
                'success': success,
                'final_score': final_score,
                'patterns_discovered': len(patterns),
                'insights_generated': len(insights),
                'patterns': [self._pattern_to_dict(p) for p in patterns],
                'insights': [self._insight_to_dict(i) for i in insights],
                'timestamp': time.time()
            }
            
            logger.info(f"Analyzed episode for {game_id}: {len(patterns)} patterns, {len(insights)} insights")
            return analysis_summary
            
        except Exception as e:
            logger.error(f"Error analyzing game episode: {e}")
            return {'error': str(e)}
    
    def get_learning_recommendations(self, game_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get learning recommendations for a specific game."""
        try:
            # Get transfer recommendations
            transfer_recommendations = self.knowledge_transfer.get_transfer_recommendations(game_id, context)
            
            # Get applicable patterns
            applicable_patterns = self._get_applicable_patterns(game_id, context)
            
            # Get applicable insights
            applicable_insights = self._get_applicable_insights(game_id, context)
            
            # Generate strategy recommendations
            strategy_recommendations = self._generate_strategy_recommendations(game_id, context)
            
            return {
                'game_id': game_id,
                'transfer_recommendations': transfer_recommendations,
                'applicable_patterns': applicable_patterns,
                'applicable_insights': applicable_insights,
                'strategy_recommendations': strategy_recommendations,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting learning recommendations: {e}")
            return {'error': str(e)}
    
    def transfer_knowledge(self, source_game_id: str, target_game_id: str, 
                          target_context: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer knowledge from source game to target game."""
        try:
            # Get patterns from source game
            source_patterns = self.pattern_recognizer.get_patterns_by_game(source_game_id)
            
            # Transfer knowledge
            transfer_recommendations = self.knowledge_transfer.transfer_knowledge(
                source_patterns, target_game_id, target_context
            )
            
            # Record transfer attempt
            self.stats['successful_transfers'] += 1
            
            return {
                'source_game_id': source_game_id,
                'target_game_id': target_game_id,
                'transfer_recommendations': transfer_recommendations,
                'patterns_transferred': len(source_patterns),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error transferring knowledge: {e}")
            return {'error': str(e)}
    
    def _update_strategy_effectiveness(self, game_id: str, success: bool, final_score: int):
        """Update strategy effectiveness tracking."""
        try:
            # Get recent patterns for this game
            recent_patterns = self.pattern_recognizer.get_patterns_by_game(game_id)
            
            if recent_patterns:
                # Calculate strategy effectiveness
                strategy_key = f"{game_id}_recent"
                effectiveness = {
                    'success': success,
                    'score': final_score,
                    'pattern_count': len(recent_patterns),
                    'timestamp': time.time()
                }
                
                self.strategy_effectiveness[strategy_key].append(effectiveness)
                
                # Keep only recent data
                if len(self.strategy_effectiveness[strategy_key]) > 100:
                    self.strategy_effectiveness[strategy_key] = self.strategy_effectiveness[strategy_key][-50:]
                    
        except Exception as e:
            logger.error(f"Error updating strategy effectiveness: {e}")
    
    def _get_applicable_patterns(self, game_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get patterns applicable to the game context."""
        try:
            # Get all patterns
            all_patterns = list(self.pattern_recognizer.patterns)
            
            # Filter by applicability
            applicable_patterns = []
            for pattern in all_patterns:
                if self._is_pattern_applicable_to_context(pattern, context):
                    applicable_patterns.append(self._pattern_to_dict(pattern))
            
            # Sort by confidence and success rate
            applicable_patterns.sort(key=lambda x: x['confidence'] * x['success_rate'], reverse=True)
            
            return applicable_patterns[:10]  # Return top 10
            
        except Exception as e:
            logger.error(f"Error getting applicable patterns: {e}")
            return []
    
    def _get_applicable_insights(self, game_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get insights applicable to the game context."""
        try:
            # Get all insights
            all_insights = self.insight_extractor.insights
            
            # Filter by applicability
            applicable_insights = []
            for insight in all_insights:
                if insight.applicability_score >= self.insight_threshold:
                    applicable_insights.append(self._insight_to_dict(insight))
            
            # Sort by applicability score
            applicable_insights.sort(key=lambda x: x['applicability_score'], reverse=True)
            
            return applicable_insights[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"Error getting applicable insights: {e}")
            return []
    
    def _generate_strategy_recommendations(self, game_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate strategy recommendations for the game."""
        try:
            recommendations = []
            
            # Get game history
            game_episodes = self.game_histories.get(game_id, [])
            
            if game_episodes:
                # Calculate success rate
                successful_episodes = [e for e in game_episodes if e['success']]
                success_rate = len(successful_episodes) / len(game_episodes)
                
                if success_rate < 0.5:
                    recommendations.append({
                        'type': 'strategy',
                        'content': 'Success rate is low. Consider using high-confidence patterns and insights.',
                        'priority': 'high'
                    })
                
                # Get most effective patterns for this game
                game_patterns = self.pattern_recognizer.get_patterns_by_game(game_id)
                if game_patterns:
                    high_confidence_patterns = [p for p in game_patterns if p.confidence > 0.8]
                    if high_confidence_patterns:
                        recommendations.append({
                            'type': 'pattern',
                            'content': f'Use {len(high_confidence_patterns)} high-confidence patterns identified for this game.',
                            'priority': 'medium'
                        })
            
            # Get general recommendations
            if not game_episodes:
                recommendations.append({
                    'type': 'general',
                    'content': 'No previous data for this game. Start with general patterns and adapt based on results.',
                    'priority': 'medium'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating strategy recommendations: {e}")
            return []
    
    def _is_pattern_applicable_to_context(self, pattern: Any, context: Dict[str, Any]) -> bool:
        """Check if a pattern is applicable to the given context."""
        try:
            # Check if pattern actions are available in context
            available_actions = context.get('available_actions', [])
            if available_actions:
                if not all(action in available_actions for action in pattern.actions):
                    return False
            
            # Check pattern confidence
            if pattern.confidence < 0.5:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking pattern applicability: {e}")
            return False
    
    def _pattern_to_dict(self, pattern: Any) -> Dict[str, Any]:
        """Convert pattern to dictionary."""
        try:
            return {
                'pattern_type': pattern.pattern_type,
                'description': pattern.description,
                'actions': pattern.actions,
                'success_rate': pattern.success_rate,
                'confidence': pattern.confidence,
                'games_seen': pattern.games_seen,
                'timestamp': pattern.timestamp
            }
        except Exception as e:
            logger.error(f"Error converting pattern to dict: {e}")
            return {}
    
    def _insight_to_dict(self, insight: Any) -> Dict[str, Any]:
        """Convert insight to dictionary."""
        try:
            return {
                'insight_type': insight.insight_type,
                'content': insight.content,
                'supporting_evidence': insight.supporting_evidence,
                'applicability_score': insight.applicability_score,
                'validation_count': insight.validation_count,
                'timestamp': insight.timestamp
            }
        except Exception as e:
            logger.error(f"Error converting insight to dict: {e}")
            return {}
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            pattern_stats = self.pattern_recognizer.get_pattern_statistics()
            insight_stats = self.insight_extractor.get_insight_statistics()
            transfer_stats = self.knowledge_transfer.get_transfer_statistics()
            
            return {
                'overall_stats': self.stats,
                'pattern_statistics': pattern_stats,
                'insight_statistics': insight_stats,
                'transfer_statistics': transfer_stats,
                'games_analyzed': len(self.game_histories),
                'total_episodes': sum(len(episodes) for episodes in self.game_histories.values())
            }
            
        except Exception as e:
            logger.error(f"Error getting system statistics: {e}")
            return {'error': str(e)}
