"""
ARC-specific Meta-Learning System

This module extends the base meta-learning system with ARC-specific pattern recognition,
insight extraction, and cross-task knowledge transfer capabilities.
"""

import torch
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass
import time

from src.core.meta_learning import MetaLearningSystem

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


@dataclass
class ARCInsight:
    """Represents a high-level insight about ARC reasoning."""
    insight_type: str  # 'strategy', 'heuristic', 'failure_mode', 'success_pattern'
    content: str
    supporting_evidence: List[Dict[str, Any]]
    applicability_score: float
    validation_count: int
    timestamp: float


class ARCMetaLearningSystem:
    """
    Enhanced meta-learning system specifically designed for ARC tasks.
    
    This system learns patterns across different ARC games and develops
    general reasoning strategies that can be applied to new tasks.
    """
    
    def __init__(
        self,
        base_meta_learning: MetaLearningSystem,
        pattern_memory_size: int = 1000,
        insight_threshold: float = 0.7,
        cross_validation_threshold: int = 3
    ):
        self.base_system = base_meta_learning
        self.pattern_memory_size = pattern_memory_size
        self.insight_threshold = insight_threshold
        self.cross_validation_threshold = cross_validation_threshold
        
        # ARC-specific storage
        self.patterns = deque(maxlen=pattern_memory_size)
        self.insights = []
        self.game_histories = defaultdict(list)
        self.strategy_effectiveness = defaultdict(list)
        
        # Pattern recognition
        self.visual_patterns = defaultdict(int)
        self.action_sequences = defaultdict(int)
        self.success_conditions = defaultdict(list)
        
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
    ) -> List[ARCPattern]:
        """
        Analyze a completed game episode to extract patterns and insights.
        
        Args:
            game_id: Identifier for the ARC game
            episode_data: Complete episode data including frames, actions, reasoning
            success: Whether the episode was successful
            final_score: Final score achieved
            
        Returns:
            List of discovered patterns
        """
        discovered_patterns = []
        
        # Store episode in game history
        episode_record = {
            'episode_data': episode_data,
            'success': success,
            'final_score': final_score,
            'timestamp': time.time()
        }
        self.game_histories[game_id].append(episode_record)
        self.stats['games_analyzed'] += 1
        
        # Extract visual patterns
        visual_patterns = self._extract_visual_patterns(episode_data)
        discovered_patterns.extend(visual_patterns)
        
        # Extract action patterns
        action_patterns = self._extract_action_patterns(episode_data, success)
        discovered_patterns.extend(action_patterns)
        
        # Extract reasoning patterns
        reasoning_patterns = self._extract_reasoning_patterns(episode_data, success)
        discovered_patterns.extend(reasoning_patterns)
        
        # Update pattern memory
        for pattern in discovered_patterns:
            self.patterns.append(pattern)
            self.stats['patterns_discovered'] += 1
            
        # Generate insights if we have enough data
        if len(self.game_histories[game_id]) >= self.cross_validation_threshold:
            new_insights = self._generate_insights(game_id)
            self.insights.extend(new_insights)
            self.stats['insights_generated'] += len(new_insights)
            
        return discovered_patterns
        
    def _extract_visual_patterns(self, episode_data: Dict[str, Any]) -> List[ARCPattern]:
        """Extract patterns from visual frame sequences."""
        patterns = []
        
        if 'frames' not in episode_data:
            return patterns
            
        frames = episode_data['frames']
        if len(frames) < 2:
            return patterns
            
        # Analyze frame transitions
        for i in range(len(frames) - 1):
            current_frame = frames[i].frame if hasattr(frames[i], 'frame') else []
            next_frame = frames[i + 1].frame if hasattr(frames[i + 1], 'frame') else []
            
            if current_frame and next_frame:
                # Detect grid changes
                changes = self._detect_grid_changes(current_frame, next_frame)
                if changes:
                    pattern = ARCPattern(
                        pattern_type='visual',
                        description=f"Grid change pattern: {changes['type']}",
                        conditions={'grid_size': changes.get('grid_size', 'unknown')},
                        actions=[episode_data['actions'][i].name if i < len(episode_data['actions']) else 'unknown'],
                        success_rate=0.5,  # Will be updated with more data
                        confidence=0.6,
                        games_seen=[episode_data.get('game_id', 'unknown')],
                        timestamp=time.time()
                    )
                    patterns.append(pattern)
                    
        return patterns
        
    def _extract_action_patterns(self, episode_data: Dict[str, Any], success: bool) -> List[ARCPattern]:
        """Extract patterns from action sequences."""
        patterns = []
        
        if 'actions' not in episode_data:
            return patterns
            
        actions = episode_data['actions']
        if len(actions) < 3:
            return patterns
            
        # Look for action sequences
        for i in range(len(actions) - 2):
            sequence = [actions[i].name, actions[i + 1].name, actions[i + 2].name]
            sequence_key = ' -> '.join(sequence)
            
            self.action_sequences[sequence_key] += 1
            
            # Create pattern for frequently seen sequences
            if self.action_sequences[sequence_key] >= 2:
                pattern = ARCPattern(
                    pattern_type='sequential',
                    description=f"Action sequence: {sequence_key}",
                    conditions={'sequence_length': 3},
                    actions=sequence,
                    success_rate=1.0 if success else 0.0,
                    confidence=min(0.9, self.action_sequences[sequence_key] / 10.0),
                    games_seen=[episode_data.get('game_id', 'unknown')],
                    timestamp=time.time()
                )
                patterns.append(pattern)
                
        return patterns
        
    def _extract_reasoning_patterns(self, episode_data: Dict[str, Any], success: bool) -> List[ARCPattern]:
        """Extract patterns from reasoning data."""
        patterns = []
        
        if 'reasoning' not in episode_data:
            return patterns
            
        reasoning_data = episode_data['reasoning']
        
        # Analyze confidence patterns
        confidences = [r.get('confidence', 0.5) for r in reasoning_data if isinstance(r, dict)]
        if confidences:
            avg_confidence = np.mean(confidences)
            confidence_trend = 'increasing' if len(confidences) > 5 and confidences[-3:] > confidences[:3] else 'stable'
            
            pattern = ARCPattern(
                pattern_type='logical',
                description=f"Confidence pattern: avg={avg_confidence:.2f}, trend={confidence_trend}",
                conditions={'avg_confidence': avg_confidence, 'trend': confidence_trend},
                actions=['confidence_based_action'],
                success_rate=1.0 if success else 0.0,
                confidence=avg_confidence,
                games_seen=[episode_data.get('game_id', 'unknown')],
                timestamp=time.time()
            )
            patterns.append(pattern)
            
        return patterns
        
    def _detect_grid_changes(self, grid1: List[List[List[int]]], grid2: List[List[List[int]]]) -> Optional[Dict[str, Any]]:
        """Detect and categorize changes between two grids."""
        if not grid1 or not grid2:
            return None
            
        try:
            arr1 = np.array(grid1)
            arr2 = np.array(grid2)
            
            if arr1.shape != arr2.shape:
                return {'type': 'size_change', 'grid_size': f"{arr1.shape} -> {arr2.shape}"}
                
            diff = np.sum(arr1 != arr2)
            total_cells = arr1.size
            
            if diff == 0:
                return {'type': 'no_change', 'grid_size': arr1.shape}
            elif diff < total_cells * 0.1:
                return {'type': 'minor_change', 'changed_cells': int(diff), 'grid_size': arr1.shape}
            elif diff < total_cells * 0.5:
                return {'type': 'moderate_change', 'changed_cells': int(diff), 'grid_size': arr1.shape}
            else:
                return {'type': 'major_change', 'changed_cells': int(diff), 'grid_size': arr1.shape}
                
        except Exception as e:
            logger.warning(f"Error detecting grid changes: {e}")
            return None
            
    def _generate_insights(self, game_id: str) -> List[ARCInsight]:
        """Generate high-level insights from accumulated game data."""
        insights = []
        
        game_history = self.game_histories[game_id]
        if len(game_history) < self.cross_validation_threshold:
            return insights
            
        # Analyze success patterns
        successful_episodes = [ep for ep in game_history if ep['success']]
        failed_episodes = [ep for ep in game_history if not ep['success']]
        
        if successful_episodes and failed_episodes:
            # Compare successful vs failed strategies
            success_insight = self._compare_strategies(successful_episodes, failed_episodes, game_id)
            if success_insight:
                insights.append(success_insight)
                
        # Analyze learning progression
        if len(game_history) >= 5:
            progression_insight = self._analyze_learning_progression(game_history, game_id)
            if progression_insight:
                insights.append(progression_insight)
                
        return insights
        
    def _compare_strategies(
        self,
        successful_episodes: List[Dict[str, Any]],
        failed_episodes: List[Dict[str, Any]],
        game_id: str
    ) -> Optional[ARCInsight]:
        """Compare strategies between successful and failed episodes."""
        
        # Extract action patterns from successful episodes
        success_actions = []
        for episode in successful_episodes:
            if 'episode_data' in episode and 'actions' in episode['episode_data']:
                actions = [a.name for a in episode['episode_data']['actions']]
                success_actions.extend(actions)
                
        # Extract action patterns from failed episodes
        fail_actions = []
        for episode in failed_episodes:
            if 'episode_data' in episode and 'actions' in episode['episode_data']:
                actions = [a.name for a in episode['episode_data']['actions']]
                fail_actions.extend(actions)
                
        if not success_actions or not fail_actions:
            return None
            
        # Find actions that appear more in successful episodes
        success_freq = defaultdict(int)
        fail_freq = defaultdict(int)
        
        for action in success_actions:
            success_freq[action] += 1
        for action in fail_actions:
            fail_freq[action] += 1
            
        # Identify beneficial actions
        beneficial_actions = []
        for action in success_freq:
            success_rate = success_freq[action] / (success_freq[action] + fail_freq.get(action, 0))
            if success_rate > 0.7:  # Action leads to success 70% of the time
                beneficial_actions.append((action, success_rate))
                
        if beneficial_actions:
            insight = ARCInsight(
                insight_type='strategy',
                content=f"Beneficial actions for {game_id}: {[a[0] for a in beneficial_actions]}",
                supporting_evidence=[
                    {'action': action, 'success_rate': rate} for action, rate in beneficial_actions
                ],
                applicability_score=0.8,
                validation_count=len(successful_episodes),
                timestamp=time.time()
            )
            return insight
            
        return None
        
    def _analyze_learning_progression(
        self,
        game_history: List[Dict[str, Any]],
        game_id: str
    ) -> Optional[ARCInsight]:
        """Analyze how performance changes over time."""
        
        scores = [episode['final_score'] for episode in game_history]
        timestamps = [episode['timestamp'] for episode in game_history]
        
        if len(scores) < 5:
            return None
            
        # Calculate learning trend
        early_scores = scores[:len(scores)//2]
        late_scores = scores[len(scores)//2:]
        
        early_avg = np.mean(early_scores)
        late_avg = np.mean(late_scores)
        
        improvement = late_avg - early_avg
        
        if improvement > 5:  # Significant improvement
            insight = ARCInsight(
                insight_type='success_pattern',
                content=f"Learning progression detected for {game_id}: {improvement:.1f} point improvement",
                supporting_evidence=[
                    {'early_average': early_avg, 'late_average': late_avg, 'improvement': improvement}
                ],
                applicability_score=0.9,
                validation_count=len(game_history),
                timestamp=time.time()
            )
            return insight
            
        return None
        
    def get_applicable_patterns(self, game_id: str, current_context: Dict[str, Any]) -> List[ARCPattern]:
        """Get patterns that might be applicable to the current game context."""
        applicable_patterns = []
        
        for pattern in self.patterns:
            # Check if pattern is from the same game
            if game_id in pattern.games_seen:
                applicable_patterns.append(pattern)
                continue
                
            # Check if pattern conditions match current context
            if self._pattern_matches_context(pattern, current_context):
                applicable_patterns.append(pattern)
                
        # Sort by confidence and success rate
        applicable_patterns.sort(key=lambda p: p.confidence * p.success_rate, reverse=True)
        
        return applicable_patterns[:10]  # Return top 10 most relevant patterns
        
    def _pattern_matches_context(self, pattern: ARCPattern, context: Dict[str, Any]) -> bool:
        """Check if a pattern's conditions match the current context."""
        if not pattern.conditions:
            return False
            
        # Simple matching based on available context
        for key, value in pattern.conditions.items():
            if key in context:
                if context[key] != value:
                    return False
            else:
                # If context doesn't have the key, assume partial match
                continue
                
        return True
        
    def get_strategic_recommendations(self, game_id: str) -> List[str]:
        """Get strategic recommendations based on learned insights."""
        recommendations = []
        
        # Get insights that might apply to this game
        applicable_insights = [
            insight for insight in self.insights
            if insight.applicability_score > self.insight_threshold
        ]
        
        for insight in applicable_insights:
            if insight.insight_type == 'strategy':
                recommendations.append(insight.content)
            elif insight.insight_type == 'success_pattern':
                recommendations.append(f"Apply successful pattern: {insight.content}")
                
        # Add pattern-based recommendations
        recent_patterns = [p for p in self.patterns if p.success_rate > 0.7][-5:]
        for pattern in recent_patterns:
            if pattern.pattern_type == 'sequential':
                recommendations.append(f"Consider action sequence: {' -> '.join(pattern.actions)}")
                
        return recommendations[:5]  # Return top 5 recommendations
        
    def save_learning_state(self, filepath: str):
        """Save the current learning state to disk."""
        save_data = {
            'patterns': [
                {
                    'pattern_type': p.pattern_type,
                    'description': p.description,
                    'conditions': p.conditions,
                    'actions': p.actions,
                    'success_rate': p.success_rate,
                    'confidence': p.confidence,
                    'games_seen': p.games_seen,
                    'timestamp': p.timestamp
                } for p in self.patterns
            ],
            'insights': [
                {
                    'insight_type': i.insight_type,
                    'content': i.content,
                    'supporting_evidence': i.supporting_evidence,
                    'applicability_score': i.applicability_score,
                    'validation_count': i.validation_count,
                    'timestamp': i.timestamp
                } for i in self.insights
            ],
            'stats': self.stats,
            'timestamp': time.time()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            logger.info(f"ARC meta-learning state saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save learning state: {e}")
            
    def load_learning_state(self, filepath: str):
        """Load learning state from disk."""
        try:
            with open(filepath, 'r') as f:
                save_data = json.load(f)
                
            # Restore patterns
            self.patterns.clear()
            for p_data in save_data.get('patterns', []):
                pattern = ARCPattern(**p_data)
                self.patterns.append(pattern)
                
            # Restore insights
            self.insights.clear()
            for i_data in save_data.get('insights', []):
                insight = ARCInsight(**i_data)
                self.insights.append(insight)
                
            # Restore stats
            self.stats.update(save_data.get('stats', {}))
            
            logger.info(f"ARC meta-learning state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load learning state: {e}")
            
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of the current learning state."""
        return {
            'total_patterns': len(self.patterns),
            'total_insights': len(self.insights),
            'games_analyzed': self.stats['games_analyzed'],
            'patterns_discovered': self.stats['patterns_discovered'],
            'insights_generated': self.stats['insights_generated'],
            'successful_transfers': self.stats['successful_transfers'],
            'pattern_types': {
                pattern_type: sum(1 for p in self.patterns if p.pattern_type == pattern_type)
                for pattern_type in ['visual', 'spatial', 'logical', 'sequential']
            },
            'average_pattern_confidence': np.mean([p.confidence for p in self.patterns]) if self.patterns else 0.0,
            'high_confidence_patterns': sum(1 for p in self.patterns if p.confidence > 0.8)
        }
