"""
Pathway-Based Learning System for ARC-AGI-3 Training
Tracks action sequences and weights them based on score improvements.
Integrates with existing memory consolidation system.
"""
from typing import Dict, List, Tuple, Any, Optional
from collections import deque, defaultdict
import json
import numpy as np
from datetime import datetime


class PathwayLearningSystem:
    """
    Advanced learning system that tracks action sequences (pathways) and
    weights them based on their contribution to score improvements.
    
    Integrates with the existing memory system for consolidation and prioritization.
    """
    
    def __init__(self, memory_system=None):
        self.memory_system = memory_system
        
        # Core pathway tracking
        self.current_pathway = deque(maxlen=20)  # Current action sequence
        self.pathway_scores = {}  # pathway_hash -> score_info
        self.pathway_frequency = defaultdict(int)  # How often pathways appear
        
        # Score-based learning
        self.score_history = []  # Track score changes over time
        self.baseline_score = 0
        self.last_score = 0
        self.win_score_target = 0
        
        # Pathway analysis
        self.successful_pathways = []  # High-scoring sequences
        self.failed_pathways = []     # Low-scoring sequences
        self.pathway_patterns = {}    # Learned patterns in successful paths
        
        # Adaptive thresholds - start high, get more selective
        self.score_improvement_threshold = 1.0  # Minimum score gain to consider success
        self.pathway_relevance_decay = 0.95     # How quickly old pathways lose relevance
        self.min_pathway_length = 2             # Minimum sequence length to track
        self.max_pathway_length = 15            # Maximum sequence length to track
        
        # Memory integration
        self.consolidation_ready = []  # Pathways ready for memory consolidation
        
    def track_action(self, action: int, action_data: Dict[str, Any], 
                   score_before: int, score_after: int, win_score: int,
                   game_id: str, frame_data: Any = None) -> Dict[str, Any]:
        """
        Track a single action and its impact on scoring.
        
        Args:
            action: Action number (1-7)
            action_data: Additional action data (coordinates for ACTION6, etc.)
            score_before: Score before this action
            score_after: Score after this action  
            win_score: Target score for victory
            game_id: Current game identifier
            frame_data: Frame analysis data (optional)
            
        Returns:
            Analysis of this action's contribution to pathways
        """
        # Create action record
        action_record = {
            'action': action,
            'data': action_data,
            'score_before': score_before,
            'score_after': score_after,
            'score_improvement': score_after - score_before,
            'timestamp': datetime.now().isoformat(),
            'game_id': game_id,
            'frame_data': frame_data
        }
        
        # Add to current pathway
        self.current_pathway.append(action_record)
        
        # Update score tracking
        self.last_score = score_after
        self.win_score_target = win_score
        self.score_history.append({
            'score': score_after,
            'improvement': score_after - score_before,
            'timestamp': action_record['timestamp'],
            'pathway_length': len(self.current_pathway)
        })
        
        # Analyze pathway contribution
        analysis = self._analyze_current_pathway(action_record)
        
        # Check if pathway should be consolidated into memory
        if analysis.get('should_consolidate', False):
            self._prepare_for_consolidation(analysis)
        
        return analysis
    
    def _analyze_current_pathway(self, latest_action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the current pathway for learning opportunities.
        """
        if len(self.current_pathway) < self.min_pathway_length:
            return {'status': 'pathway_too_short', 'pathway_length': len(self.current_pathway)}
        
        # Calculate pathway metrics
        pathway_list = list(self.current_pathway)
        pathway_hash = self._hash_pathway(pathway_list)
        
        # Calculate total score improvement for this pathway
        total_improvement = sum(action['score_improvement'] for action in pathway_list)
        pathway_length = len(pathway_list)
        
        # Calculate efficiency (score per action)
        efficiency = total_improvement / pathway_length if pathway_length > 0 else 0
        
        # Calculate progress toward win condition
        current_score = latest_action['score_after']
        progress_ratio = current_score / self.win_score_target if self.win_score_target > 0 else 0
        
        # Determine if this pathway is successful
        is_successful = total_improvement >= self.score_improvement_threshold
        
        analysis = {
            'pathway_hash': pathway_hash,
            'pathway_length': pathway_length,
            'total_score_improvement': total_improvement,
            'efficiency': efficiency,
            'progress_ratio': progress_ratio,
            'is_successful': is_successful,
            'actions_sequence': [a['action'] for a in pathway_list],
            'score_progression': [a['score_after'] for a in pathway_list]
        }
        
        # Update pathway tracking
        self._update_pathway_records(pathway_hash, analysis, pathway_list)
        
        # Determine if consolidation is needed
        analysis['should_consolidate'] = self._should_consolidate_pathway(analysis)
        
        return analysis
    
    def _hash_pathway(self, pathway: List[Dict[str, Any]]) -> str:
        """
        Create a hash for a pathway sequence for tracking purposes.
        """
        # Create signature from actions and key data
        signature_parts = []
        for action in pathway:
            part = str(action['action'])
            
            # Include coordinate data for ACTION6
            if action['action'] == 6 and 'data' in action and action['data']:
                if 'x' in action['data'] and 'y' in action['data']:
                    part += f"({action['data']['x']},{action['data']['y']})"
            
            signature_parts.append(part)
        
        return "_".join(signature_parts)
    
    def _update_pathway_records(self, pathway_hash: str, analysis: Dict[str, Any], 
                              pathway: List[Dict[str, Any]]):
        """
        Update records for this pathway based on its performance.
        """
        # Update frequency tracking
        self.pathway_frequency[pathway_hash] += 1
        
        # Update score tracking
        if pathway_hash not in self.pathway_scores:
            self.pathway_scores[pathway_hash] = {
                'first_seen': datetime.now().isoformat(),
                'total_attempts': 0,
                'total_improvement': 0,
                'best_improvement': 0,
                'average_improvement': 0,
                'success_rate': 0,
                'pathway_data': pathway.copy()
            }
        
        record = self.pathway_scores[pathway_hash]
        record['total_attempts'] += 1
        record['total_improvement'] += analysis['total_score_improvement']
        record['best_improvement'] = max(record['best_improvement'], analysis['total_score_improvement'])
        record['average_improvement'] = record['total_improvement'] / record['total_attempts']
        
        # Update success rate
        if analysis['is_successful']:
            record['success_rate'] = (record['success_rate'] * (record['total_attempts'] - 1) + 1) / record['total_attempts']
        else:
            record['success_rate'] = (record['success_rate'] * (record['total_attempts'] - 1)) / record['total_attempts']
        
        # Categorize pathway
        if analysis['is_successful'] and record['average_improvement'] > self.score_improvement_threshold:
            if pathway_hash not in [p['hash'] for p in self.successful_pathways]:
                self.successful_pathways.append({
                    'hash': pathway_hash,
                    'pathway': pathway.copy(),
                    'analysis': analysis.copy(),
                    'record': record.copy()
                })
        elif record['average_improvement'] < 0:
            if pathway_hash not in [p['hash'] for p in self.failed_pathways]:
                self.failed_pathways.append({
                    'hash': pathway_hash,
                    'pathway': pathway.copy(),
                    'analysis': analysis.copy(),
                    'record': record.copy()
                })
    
    def _should_consolidate_pathway(self, analysis: Dict[str, Any]) -> bool:
        """
        Determine if a pathway should be consolidated into long-term memory.
        """
        # Consolidate if highly successful
        if analysis['is_successful'] and analysis['efficiency'] > 1.0:
            return True
            
        # Consolidate if we've seen this pattern multiple times
        pathway_hash = analysis['pathway_hash']
        if self.pathway_frequency[pathway_hash] >= 3:
            return True
            
        # Consolidate if it's a long sequence with some success
        if analysis['pathway_length'] >= 10 and analysis['total_score_improvement'] > 0:
            return True
            
        return False
    
    def _prepare_for_consolidation(self, analysis: Dict[str, Any]):
        """
        Prepare pathway data for memory system consolidation.
        """
        consolidation_data = {
            'type': 'pathway',
            'pathway_hash': analysis['pathway_hash'],
            'analysis': analysis,
            'timestamp': datetime.now().isoformat(),
            'priority': self._calculate_consolidation_priority(analysis),
            'game_context': self._extract_game_context()
        }
        
        self.consolidation_ready.append(consolidation_data)
        
        # Integrate with memory system if available
        if self.memory_system and hasattr(self.memory_system, 'add_consolidation_candidate'):
            self.memory_system.add_consolidation_candidate(consolidation_data)
    
    def _calculate_consolidation_priority(self, analysis: Dict[str, Any]) -> float:
        """
        Calculate how important this pathway is for memory consolidation.
        """
        priority = 0.0
        
        # High efficiency pathways get high priority
        priority += analysis['efficiency'] * 0.4
        
        # Progress toward win condition
        priority += analysis['progress_ratio'] * 0.3
        
        # Pathway length bonus (longer sequences are more valuable if successful)
        if analysis['is_successful']:
            priority += min(analysis['pathway_length'] / 20.0, 0.2)
        
        # Frequency bonus (commonly seen patterns)
        frequency = self.pathway_frequency[analysis['pathway_hash']]
        priority += min(frequency / 10.0, 0.1)
        
        return priority
    
    def _extract_game_context(self) -> Dict[str, Any]:
        """
        Extract relevant context about current game state for memory storage.
        """
        return {
            'current_score': self.last_score,
            'win_score_target': self.win_score_target,
            'progress_ratio': self.last_score / self.win_score_target if self.win_score_target > 0 else 0,
            'score_history_length': len(self.score_history),
            'pathway_count': len(self.current_pathway)
        }
    
    def get_pathway_recommendations(self, available_actions: List[int], 
                                  current_score: int, game_id: str) -> Dict[str, Any]:
        """
        Get action recommendations based on learned pathways.
        
        Returns:
            Recommendations with weights for each available action
        """
        recommendations = {
            'action_weights': {},
            'pathway_suggestions': [],
            'reasoning': [],
            'confidence': 0.0
        }
        
        # Weight actions based on successful pathways
        for action in available_actions:
            weight = self._calculate_action_weight(action, current_score, game_id)
            recommendations['action_weights'][action] = weight
        
        # Find relevant pathway suggestions
        relevant_pathways = self._find_relevant_pathways(available_actions, current_score)
        recommendations['pathway_suggestions'] = relevant_pathways
        
        # Generate reasoning
        recommendations['reasoning'] = self._generate_pathway_reasoning(
            recommendations['action_weights'], relevant_pathways
        )
        
        # Calculate confidence based on data quality
        recommendations['confidence'] = self._calculate_recommendation_confidence()
        
        return recommendations
    
    def _calculate_action_weight(self, action: int, current_score: int, game_id: str) -> float:
        """
        Calculate weight for a specific action based on pathway learning.
        """
        base_weight = 0.5  # Default weight
        
        # Analyze successful pathways that included this action
        action_success_data = self._analyze_action_in_pathways(action)
        
        # Weight based on average improvement when this action was used
        if action_success_data['total_occurrences'] > 0:
            avg_improvement = action_success_data['total_improvement'] / action_success_data['total_occurrences']
            success_rate = action_success_data['successful_occurrences'] / action_success_data['total_occurrences']
            
            # Adjust weight based on performance
            performance_multiplier = (avg_improvement + 1) * success_rate
            base_weight *= performance_multiplier
        
        # Context-based adjustments
        progress_ratio = current_score / self.win_score_target if self.win_score_target > 0 else 0
        
        # If we're close to winning, prefer actions that have worked in similar score ranges
        if progress_ratio > 0.8:
            high_score_performance = self._get_action_performance_in_score_range(action, 0.8, 1.0)
            base_weight *= (1 + high_score_performance)
        
        # Ensure minimum weight for exploration (anti-bias protection)
        return max(0.15, min(2.0, base_weight))
    
    def _analyze_action_in_pathways(self, action: int) -> Dict[str, int]:
        """
        Analyze how this action has performed across all tracked pathways.
        """
        data = {
            'total_occurrences': 0,
            'successful_occurrences': 0,
            'total_improvement': 0,
            'in_successful_pathways': 0
        }
        
        # Check successful pathways
        for pathway_info in self.successful_pathways:
            pathway = pathway_info['pathway']
            for action_record in pathway:
                if action_record['action'] == action:
                    data['total_occurrences'] += 1
                    data['total_improvement'] += action_record['score_improvement']
                    if action_record['score_improvement'] > 0:
                        data['successful_occurrences'] += 1
                    data['in_successful_pathways'] += 1
        
        # Check all pathway records
        for pathway_hash, record in self.pathway_scores.items():
            for action_record in record['pathway_data']:
                if action_record['action'] == action:
                    if pathway_hash not in [p['hash'] for p in self.successful_pathways]:
                        data['total_occurrences'] += 1
                        data['total_improvement'] += action_record['score_improvement']
                        if action_record['score_improvement'] > 0:
                            data['successful_occurrences'] += 1
        
        return data
    
    def _get_action_performance_in_score_range(self, action: int, min_ratio: float, max_ratio: float) -> float:
        """
        Get action performance in a specific score range.
        """
        # This would analyze action performance when the game progress was in a specific range
        # Implementation would look at score_history and correlate with action performance
        return 0.0  # Placeholder - would need historical data analysis
    
    def _find_relevant_pathways(self, available_actions: List[int], current_score: int) -> List[Dict[str, Any]]:
        """
        Find pathways that might be relevant to the current situation.
        """
        relevant = []
        
        for pathway_info in self.successful_pathways[:10]:  # Top 10 successful pathways
            pathway = pathway_info['pathway']
            
            # Check if pathway starts with an available action
            if pathway and pathway[0]['action'] in available_actions:
                relevance_score = self._calculate_pathway_relevance(pathway_info, current_score)
                
                relevant.append({
                    'pathway_hash': pathway_info['hash'],
                    'first_action': pathway[0]['action'],
                    'sequence': [a['action'] for a in pathway],
                    'relevance_score': relevance_score,
                    'expected_improvement': pathway_info['record']['average_improvement']
                })
        
        # Sort by relevance
        relevant.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant[:5]  # Return top 5 most relevant
    
    def _calculate_pathway_relevance(self, pathway_info: Dict[str, Any], current_score: int) -> float:
        """
        Calculate how relevant a pathway is to the current situation.
        """
        relevance = 0.0
        
        # Base relevance from success rate
        relevance += pathway_info['record']['success_rate'] * 0.4
        
        # Relevance from average improvement
        relevance += min(pathway_info['record']['average_improvement'] / 10.0, 0.3)
        
        # Recent usage bonus (pathways used recently are more relevant)
        frequency = self.pathway_frequency[pathway_info['hash']]
        relevance += min(frequency / 20.0, 0.2)
        
        # Score context relevance (pathways that worked at similar scores)
        # This would need more sophisticated analysis of score contexts
        relevance += 0.1  # Placeholder
        
        return relevance
    
    def _generate_pathway_reasoning(self, action_weights: Dict[int, float], 
                                  pathway_suggestions: List[Dict[str, Any]]) -> List[str]:
        """
        Generate human-readable reasoning for pathway recommendations.
        """
        reasoning = []
        
        # Explain action weights
        sorted_actions = sorted(action_weights.items(), key=lambda x: x[1], reverse=True)
        best_action, best_weight = sorted_actions[0]
        reasoning.append(f"Action {best_action} has highest weight ({best_weight:.2f}) based on pathway analysis")
        
        # Explain pathway suggestions
        if pathway_suggestions:
            best_pathway = pathway_suggestions[0]
            reasoning.append(f"Most relevant pathway starts with Action {best_pathway['first_action']} "
                           f"(sequence: {best_pathway['sequence'][:5]}..., expected improvement: {best_pathway['expected_improvement']:.1f})")
        
        # Explain learning state
        total_pathways = len(self.successful_pathways) + len(self.failed_pathways)
        reasoning.append(f"Learning from {total_pathways} pathways ({len(self.successful_pathways)} successful, {len(self.failed_pathways)} failed)")
        
        return reasoning
    
    def _calculate_recommendation_confidence(self) -> float:
        """
        Calculate confidence in pathway-based recommendations.
        """
        confidence = 0.0
        
        # More data = more confidence
        total_pathways = len(self.pathway_scores)
        confidence += min(total_pathways / 50.0, 0.4)
        
        # More successful pathways = more confidence
        successful_ratio = len(self.successful_pathways) / max(total_pathways, 1)
        confidence += successful_ratio * 0.3
        
        # Recent experience = more confidence
        recent_actions = len(self.score_history[-20:])  # Last 20 actions
        confidence += min(recent_actions / 20.0, 0.3)
        
        return confidence
    
    def reset_for_new_game(self, game_id: str):
        """
        Reset pathway tracking for a new game while preserving learned patterns.
        """
        # Clear current pathway but keep learned data
        self.current_pathway.clear()
        self.last_score = 0
        self.baseline_score = 0
        self.win_score_target = 0
        
        # Keep successful/failed pathways for transfer learning
        # but reduce their weights slightly for new game context
        for pathway_info in self.successful_pathways:
            # Reduce relevance for new game (transfer learning with decay)
            if 'transfer_learning_decay' not in pathway_info:
                pathway_info['transfer_learning_decay'] = 0.8
            else:
                pathway_info['transfer_learning_decay'] *= 0.9
    
    def get_memory_consolidation_data(self) -> List[Dict[str, Any]]:
        """
        Get data ready for memory system consolidation.
        """
        ready_data = self.consolidation_ready.copy()
        self.consolidation_ready.clear()
        return ready_data
    
    def integrate_memory_insights(self, memory_insights: Dict[str, Any]):
        """
        Integrate insights from the memory system back into pathway learning.
        """
        # This would allow the memory system to provide feedback
        # about which pathways should be prioritized or forgotten
        if 'priority_pathways' in memory_insights:
            for pathway_hash in memory_insights['priority_pathways']:
                if pathway_hash in self.pathway_scores:
                    # Boost priority of these pathways
                    self.pathway_scores[pathway_hash]['memory_boost'] = True
        
        if 'deprecated_pathways' in memory_insights:
            for pathway_hash in memory_insights['deprecated_pathways']:
                # Reduce weight of deprecated pathways
                if pathway_hash in self.pathway_scores:
                    self.pathway_scores[pathway_hash]['deprecated'] = True
