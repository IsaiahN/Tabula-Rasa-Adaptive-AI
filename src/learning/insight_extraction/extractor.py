"""
ARC Insight Extractor

Extracts high-level insights from ARC tasks and patterns.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ARCInsight:
    """Represents a high-level insight about ARC reasoning."""
    insight_type: str  # 'strategy', 'heuristic', 'failure_mode', 'success_pattern'
    content: str
    supporting_evidence: List[Dict[str, Any]]
    applicability_score: float
    validation_count: int
    timestamp: float

class ARCInsightExtractor:
    """Extracts high-level insights from ARC tasks and patterns."""
    
    def __init__(self, insight_threshold: float = 0.7):
        self.insight_threshold = insight_threshold
        self.insights = []
        self.insight_templates = self._initialize_insight_templates()
        
        # Insight statistics
        self.stats = {
            'insights_generated': 0,
            'insights_validated': 0,
            'insights_applied': 0
        }
    
    def _initialize_insight_templates(self) -> Dict[str, str]:
        """Initialize templates for generating insights."""
        return {
            'strategy': "Strategy discovered: {strategy} with {success_rate:.1%} success rate across {game_count} games",
            'heuristic': "Heuristic identified: {heuristic} - {description}",
            'failure_mode': "Failure mode detected: {failure_mode} occurs in {failure_rate:.1%} of failed attempts",
            'success_pattern': "Success pattern found: {pattern} leads to success in {success_rate:.1%} of cases"
        }
    
    def extract_insights(self, patterns: List[Any], game_histories: Dict[str, List[Dict[str, Any]]]) -> List[ARCInsight]:
        """Extract insights from patterns and game histories."""
        try:
            insights = []
            
            # Extract different types of insights
            strategy_insights = self._extract_strategy_insights(patterns, game_histories)
            heuristic_insights = self._extract_heuristic_insights(patterns, game_histories)
            failure_insights = self._extract_failure_insights(patterns, game_histories)
            success_insights = self._extract_success_insights(patterns, game_histories)
            
            # Combine all insights
            all_insights = (strategy_insights + heuristic_insights + 
                          failure_insights + success_insights)
            
            # Validate and store insights
            for insight in all_insights:
                if self._validate_insight(insight):
                    self._store_insight(insight)
                    insights.append(insight)
                    self.stats['insights_generated'] += 1
            
            logger.info(f"Extracted {len(insights)} insights from patterns and histories")
            return insights
            
        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            return []
    
    def _extract_strategy_insights(self, patterns: List[Any], game_histories: Dict[str, List[Dict[str, Any]]]) -> List[ARCInsight]:
        """Extract strategy insights from patterns and histories."""
        insights = []
        
        try:
            # Group patterns by type and analyze success rates
            pattern_groups = defaultdict(list)
            for pattern in patterns:
                pattern_groups[pattern.pattern_type].append(pattern)
            
            for pattern_type, type_patterns in pattern_groups.items():
                if len(type_patterns) < 3:  # Need sufficient data
                    continue
                
                # Calculate success rate for this pattern type
                success_rates = [p.success_rate for p in type_patterns]
                avg_success_rate = np.mean(success_rates)
                
                if avg_success_rate > 0.7:  # High success rate
                    # Count games where this pattern type was used
                    games_with_pattern = set()
                    for pattern in type_patterns:
                        games_with_pattern.update(pattern.games_seen)
                    
                    insight = ARCInsight(
                        insight_type='strategy',
                        content=self.insight_templates['strategy'].format(
                            strategy=f"{pattern_type} pattern usage",
                            success_rate=avg_success_rate,
                            game_count=len(games_with_pattern)
                        ),
                        supporting_evidence=[{
                            'pattern_type': pattern_type,
                            'success_rate': avg_success_rate,
                            'pattern_count': len(type_patterns),
                            'games_affected': list(games_with_pattern)
                        }],
                        applicability_score=avg_success_rate,
                        validation_count=len(type_patterns),
                        timestamp=time.time()
                    )
                    insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error extracting strategy insights: {e}")
        
        return insights
    
    def _extract_heuristic_insights(self, patterns: List[Any], game_histories: Dict[str, List[Dict[str, Any]]]) -> List[ARCInsight]:
        """Extract heuristic insights from patterns and histories."""
        insights = []
        
        try:
            # Look for common action sequences that lead to success
            action_sequences = defaultdict(list)
            
            for pattern in patterns:
                if pattern.pattern_type == 'sequential' and pattern.success_rate > 0.8:
                    sequence_key = tuple(pattern.actions)
                    action_sequences[sequence_key].append(pattern)
            
            for sequence, sequence_patterns in action_sequences.items():
                if len(sequence_patterns) >= 2:  # Need multiple occurrences
                    success_rates = [p.success_rate for p in sequence_patterns]
                    avg_success_rate = np.mean(success_rates)
                    
                    if avg_success_rate > 0.8:
                        insight = ARCInsight(
                            insight_type='heuristic',
                            content=self.insight_templates['heuristic'].format(
                                heuristic=f"Action sequence: {' -> '.join(sequence)}",
                                description=f"High success rate of {avg_success_rate:.1%}"
                            ),
                            supporting_evidence=[{
                                'sequence': list(sequence),
                                'success_rate': avg_success_rate,
                                'occurrences': len(sequence_patterns)
                            }],
                            applicability_score=avg_success_rate,
                            validation_count=len(sequence_patterns),
                            timestamp=time.time()
                        )
                        insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error extracting heuristic insights: {e}")
        
        return insights
    
    def _extract_failure_insights(self, patterns: List[Any], game_histories: Dict[str, List[Dict[str, Any]]]) -> List[ARCInsight]:
        """Extract failure mode insights from patterns and histories."""
        insights = []
        
        try:
            # Analyze failed episodes
            failed_episodes = []
            for game_id, episodes in game_histories.items():
                for episode in episodes:
                    if not episode.get('success', False):
                        failed_episodes.append(episode)
            
            if len(failed_episodes) < 3:  # Need sufficient failure data
                return insights
            
            # Look for common patterns in failures
            failure_patterns = defaultdict(int)
            total_failures = len(failed_episodes)
            
            for episode in failed_episodes:
                actions = episode.get('episode_data', {}).get('actions', [])
                if len(actions) >= 2:
                    # Look for common action pairs in failures
                    for i in range(len(actions) - 1):
                        action_pair = (actions[i], actions[i + 1])
                        failure_patterns[action_pair] += 1
            
            # Find most common failure patterns
            for action_pair, count in failure_patterns.items():
                failure_rate = count / total_failures
                
                if failure_rate > 0.3:  # Common in failures
                    insight = ARCInsight(
                        insight_type='failure_mode',
                        content=self.insight_templates['failure_mode'].format(
                            failure_mode=f"Action sequence: {' -> '.join(action_pair)}",
                            failure_rate=failure_rate
                        ),
                        supporting_evidence=[{
                            'action_pair': list(action_pair),
                            'failure_rate': failure_rate,
                            'occurrences': count,
                            'total_failures': total_failures
                        }],
                        applicability_score=1.0 - failure_rate,  # Lower applicability for failure modes
                        validation_count=count,
                        timestamp=time.time()
                    )
                    insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error extracting failure insights: {e}")
        
        return insights
    
    def _extract_success_insights(self, patterns: List[Any], game_histories: Dict[str, List[Dict[str, Any]]]) -> List[ARCInsight]:
        """Extract success pattern insights from patterns and histories."""
        insights = []
        
        try:
            # Analyze successful episodes
            successful_episodes = []
            for game_id, episodes in game_histories.items():
                for episode in episodes:
                    if episode.get('success', False):
                        successful_episodes.append(episode)
            
            if len(successful_episodes) < 3:  # Need sufficient success data
                return insights
            
            # Look for common patterns in successes
            success_patterns = defaultdict(int)
            total_successes = len(successful_episodes)
            
            for episode in successful_episodes:
                actions = episode.get('episode_data', {}).get('actions', [])
                if len(actions) >= 2:
                    # Look for common action pairs in successes
                    for i in range(len(actions) - 1):
                        action_pair = (actions[i], actions[i + 1])
                        success_patterns[action_pair] += 1
            
            # Find most common success patterns
            for action_pair, count in success_patterns.items():
                success_rate = count / total_successes
                
                if success_rate > 0.5:  # Common in successes
                    insight = ARCInsight(
                        insight_type='success_pattern',
                        content=self.insight_templates['success_pattern'].format(
                            pattern=f"Action sequence: {' -> '.join(action_pair)}",
                            success_rate=success_rate
                        ),
                        supporting_evidence=[{
                            'action_pair': list(action_pair),
                            'success_rate': success_rate,
                            'occurrences': count,
                            'total_successes': total_successes
                        }],
                        applicability_score=success_rate,
                        validation_count=count,
                        timestamp=time.time()
                    )
                    insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error extracting success insights: {e}")
        
        return insights
    
    def _validate_insight(self, insight: ARCInsight) -> bool:
        """Validate an insight before storing it."""
        try:
            # Check minimum applicability score
            if insight.applicability_score < self.insight_threshold:
                return False
            
            # Check minimum validation count
            if insight.validation_count < 2:
                return False
            
            # Check if insight is too similar to existing ones
            for existing_insight in self.insights:
                if self._insights_similar(insight, existing_insight):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating insight: {e}")
            return False
    
    def _insights_similar(self, insight1: ARCInsight, insight2: ARCInsight) -> bool:
        """Check if two insights are similar."""
        try:
            # Same type and similar content
            if (insight1.insight_type == insight2.insight_type and 
                self._text_similarity(insight1.content, insight2.content) > 0.8):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking insight similarity: {e}")
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
    
    def _store_insight(self, insight: ARCInsight):
        """Store a validated insight."""
        try:
            self.insights.append(insight)
        except Exception as e:
            logger.error(f"Error storing insight: {e}")
    
    def get_insights_by_type(self, insight_type: str) -> List[ARCInsight]:
        """Get insights of a specific type."""
        return [i for i in self.insights if i.insight_type == insight_type]
    
    def get_high_applicability_insights(self, min_applicability: float = 0.8) -> List[ARCInsight]:
        """Get insights with high applicability scores."""
        return [i for i in self.insights if i.applicability_score >= min_applicability]
    
    def get_insight_statistics(self) -> Dict[str, Any]:
        """Get statistics about extracted insights."""
        try:
            total_insights = len(self.insights)
            type_counts = defaultdict(int)
            applicability_scores = []
            
            for insight in self.insights:
                type_counts[insight.insight_type] += 1
                applicability_scores.append(insight.applicability_score)
            
            return {
                'total_insights': total_insights,
                'type_distribution': dict(type_counts),
                'average_applicability': np.mean(applicability_scores) if applicability_scores else 0.0,
                'high_applicability_count': len([a for a in applicability_scores if a >= 0.8]),
                'stats': self.stats
            }
            
        except Exception as e:
            logger.error(f"Error getting insight statistics: {e}")
            return {}
