#!/usr/bin/env python3
"""
Prediction Method Learner for Simulation Intelligence

This module implements a system that learns which prediction methods work best
for different game types and contexts. It uses A/B testing and performance
tracking to optimize the choice of search algorithms.
"""

import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

class PredictionMethod(Enum):
    """Available prediction methods."""
    BFS = "breadth_first_search"
    DFS = "depth_first_search"
    HYBRID = "hybrid_search"
    BAYESIAN = "bayesian_search"
    ADAPTIVE = "adaptive_search"

@dataclass
class MethodPerformance:
    """Performance metrics for a prediction method."""
    method: PredictionMethod
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    average_confidence: float = 0.0
    average_depth: float = 0.0
    success_rate: float = 0.0
    energy_efficiency: float = 0.0
    learning_efficiency: float = 0.0
    last_updated: float = field(default_factory=time.time)
    
    # Context-specific performance
    context_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Time-based performance (recent vs historical)
    recent_accuracy: float = 0.0
    historical_accuracy: float = 0.0
    trend: str = "stable"  # "improving", "declining", "stable"

@dataclass
class ContextProfile:
    """Profile of a specific game context."""
    context_key: str
    game_type: str
    energy_level: float
    learning_drive: float
    boredom_level: float
    action_count: int
    recent_success_rate: float
    pattern_complexity: float
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

class PredictionMethodLearner:
    """
    Learns which prediction methods work best for different contexts.
    
    This class implements the core learning mechanism that allows the system
    to adaptively choose the best prediction method based on historical
    performance and current context.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 confidence_threshold: float = 0.7,
                 min_samples: int = 10,
                 adaptation_rate: float = 0.05):
        self.learning_rate = learning_rate
        self.confidence_threshold = confidence_threshold
        self.min_samples = min_samples
        self.adaptation_rate = adaptation_rate
        
        # Method performance tracking
        self.method_performance = {
            method: MethodPerformance(method=method)
            for method in PredictionMethod
        }
        
        # Context profiles
        self.context_profiles: Dict[str, ContextProfile] = {}
        
        # A/B testing framework
        self.ab_tests = {}
        self.active_tests = {}
        
        # Learning statistics
        self.learning_stats = {
            'total_evaluations': 0,
            'method_switches': 0,
            'accuracy_improvements': 0,
            'context_discoveries': 0,
            'ab_test_results': 0
        }
        
        # Method selection strategy
        self.selection_strategy = "adaptive"  # "adaptive", "best_overall", "context_aware"
        
        logger.info("PredictionMethodLearner initialized")
    
    def select_best_method(self, 
                          context: Optional[Dict[str, Any]] = None,
                          available_methods: Optional[List[PredictionMethod]] = None) -> PredictionMethod:
        """
        Select the best prediction method for the given context.
        
        Args:
            context: Current game context
            available_methods: List of available methods (default: all)
            
        Returns:
            Best prediction method for the context
        """
        
        if available_methods is None:
            available_methods = list(PredictionMethod)
        
        # Create context profile
        context_profile = self._create_context_profile(context)
        
        # Select method based on strategy
        if self.selection_strategy == "adaptive":
            return self._select_adaptive_method(context_profile, available_methods)
        elif self.selection_strategy == "best_overall":
            return self._select_best_overall_method(available_methods)
        elif self.selection_strategy == "context_aware":
            return self._select_context_aware_method(context_profile, available_methods)
        else:
            return self._select_adaptive_method(context_profile, available_methods)
    
    def _select_adaptive_method(self, 
                               context_profile: ContextProfile,
                               available_methods: List[PredictionMethod]) -> PredictionMethod:
        """Select method using adaptive learning approach."""
        
        # Calculate scores for each method
        method_scores = {}
        
        for method in available_methods:
            if method not in self.method_performance:
                continue
            
            performance = self.method_performance[method]
            
            # Base score from overall performance
            base_score = performance.accuracy * performance.average_confidence
            
            # Context-specific adjustment
            context_score = self._calculate_context_score(method, context_profile)
            
            # Recent performance weight
            recent_weight = 0.7
            historical_weight = 0.3
            
            recent_score = performance.recent_accuracy * recent_weight
            historical_score = performance.historical_accuracy * historical_weight
            
            # Combined score
            total_score = (base_score + context_score + recent_score + historical_score) / 4.0
            
            method_scores[method] = total_score
        
        if not method_scores:
            return available_methods[0] if available_methods else PredictionMethod.BFS
        
        # Select best method with some exploration
        best_method = max(method_scores, key=method_scores.get)
        
        # Add exploration (epsilon-greedy)
        if np.random.random() < 0.1:  # 10% exploration
            exploration_methods = [m for m in available_methods if m != best_method]
            if exploration_methods:
                best_method = np.random.choice(exploration_methods)
        
        return best_method
    
    def _select_best_overall_method(self, available_methods: List[PredictionMethod]) -> PredictionMethod:
        """Select method based on overall best performance."""
        
        best_method = available_methods[0]
        best_score = 0.0
        
        for method in available_methods:
            if method not in self.method_performance:
                continue
            
            performance = self.method_performance[method]
            score = performance.accuracy * performance.average_confidence
            
            if score > best_score:
                best_score = score
                best_method = method
        
        return best_method
    
    def _select_context_aware_method(self, 
                                   context_profile: ContextProfile,
                                   available_methods: List[PredictionMethod]) -> PredictionMethod:
        """Select method based on context-specific performance."""
        
        # Find similar contexts
        similar_contexts = self._find_similar_contexts(context_profile)
        
        if not similar_contexts:
            return self._select_best_overall_method(available_methods)
        
        # Calculate context-specific scores
        method_scores = {}
        
        for method in available_methods:
            if method not in self.method_performance:
                continue
            
            performance = self.method_performance[method]
            context_scores = []
            
            for similar_context in similar_contexts:
                if similar_context in performance.context_performance:
                    context_perf = performance.context_performance[similar_context]
                    context_scores.append(context_perf.get('accuracy', 0.0))
            
            if context_scores:
                method_scores[method] = statistics.mean(context_scores)
            else:
                method_scores[method] = performance.accuracy
        
        if not method_scores:
            return available_methods[0] if available_methods else PredictionMethod.BFS
        
        return max(method_scores, key=method_scores.get)
    
    def _calculate_context_score(self, 
                               method: PredictionMethod,
                               context_profile: ContextProfile) -> float:
        """Calculate context-specific score for a method."""
        
        if method not in self.method_performance:
            return 0.0
        
        performance = self.method_performance[method]
        
        # Check if we have context-specific data
        if context_profile.context_key in performance.context_performance:
            context_perf = performance.context_performance[context_profile.context_key]
            return context_perf.get('accuracy', 0.0)
        
        # Use overall performance as fallback
        return performance.accuracy
    
    def _create_context_profile(self, context: Optional[Dict[str, Any]]) -> ContextProfile:
        """Create a context profile from the given context."""
        
        if context is None:
            context = {}
        
        # Extract context features
        game_type = context.get('game_type', 'unknown')
        energy_level = context.get('energy', 100.0)
        learning_drive = context.get('learning_drive', 0.5)
        boredom_level = context.get('boredom_level', 0.0)
        action_count = context.get('action_count', 0)
        recent_success_rate = context.get('recent_success_rate', 0.5)
        
        # Calculate pattern complexity
        pattern_complexity = self._calculate_pattern_complexity(context)
        
        # Create context key
        context_key = self._create_context_key(
            game_type, energy_level, learning_drive, boredom_level, action_count
        )
        
        # Check if profile exists
        if context_key in self.context_profiles:
            profile = self.context_profiles[context_key]
            profile.last_updated = time.time()
            return profile
        
        # Create new profile
        profile = ContextProfile(
            context_key=context_key,
            game_type=game_type,
            energy_level=energy_level,
            learning_drive=learning_drive,
            boredom_level=boredom_level,
            action_count=action_count,
            recent_success_rate=recent_success_rate,
            pattern_complexity=pattern_complexity
        )
        
        self.context_profiles[context_key] = profile
        self.learning_stats['context_discoveries'] += 1
        
        return profile
    
    def _create_context_key(self, 
                           game_type: str,
                           energy_level: float,
                           learning_drive: float,
                           boredom_level: float,
                           action_count: int) -> str:
        """Create a key for context identification."""
        
        # Quantize values for grouping similar contexts
        energy_bucket = int(energy_level // 20) * 20  # 0, 20, 40, 60, 80, 100
        drive_bucket = round(learning_drive, 1)  # 0.0, 0.1, 0.2, ..., 1.0
        boredom_bucket = round(boredom_level, 1)
        count_bucket = int(action_count // 10) * 10  # 0, 10, 20, 30, ...
        
        return f"{game_type}_{energy_bucket}_{drive_bucket}_{boredom_bucket}_{count_bucket}"
    
    def _calculate_pattern_complexity(self, context: Dict[str, Any]) -> float:
        """Calculate the complexity of the current context."""
        
        complexity = 0.0
        
        # Energy level complexity (lower energy = more complex)
        energy = context.get('energy', 100.0)
        complexity += (100.0 - energy) / 100.0 * 0.3
        
        # Learning drive complexity (higher drive = more complex)
        learning_drive = context.get('learning_drive', 0.5)
        complexity += learning_drive * 0.2
        
        # Boredom level complexity (higher boredom = more complex)
        boredom = context.get('boredom_level', 0.0)
        complexity += boredom * 0.2
        
        # Action count complexity (more actions = more complex)
        action_count = context.get('action_count', 0)
        complexity += min(action_count / 100.0, 1.0) * 0.3
        
        return min(1.0, complexity)
    
    def _find_similar_contexts(self, context_profile: ContextProfile) -> List[str]:
        """Find contexts similar to the given profile."""
        
        similar_contexts = []
        
        for context_key, profile in self.context_profiles.items():
            if context_key == context_profile.context_key:
                continue
            
            similarity = self._calculate_context_similarity(context_profile, profile)
            if similarity > 0.7:  # Similarity threshold
                similar_contexts.append(context_key)
        
        return similar_contexts
    
    def _calculate_context_similarity(self, 
                                    profile1: ContextProfile,
                                    profile2: ContextProfile) -> float:
        """Calculate similarity between two context profiles."""
        
        similarity = 0.0
        
        # Game type similarity
        if profile1.game_type == profile2.game_type:
            similarity += 0.3
        
        # Energy level similarity
        energy_sim = 1.0 - abs(profile1.energy_level - profile2.energy_level) / 100.0
        similarity += energy_sim * 0.2
        
        # Learning drive similarity
        drive_sim = 1.0 - abs(profile1.learning_drive - profile2.learning_drive)
        similarity += drive_sim * 0.2
        
        # Boredom level similarity
        boredom_sim = 1.0 - abs(profile1.boredom_level - profile2.boredom_level)
        similarity += boredom_sim * 0.1
        
        # Action count similarity
        count_sim = 1.0 - abs(profile1.action_count - profile2.action_count) / 100.0
        similarity += count_sim * 0.2
        
        return similarity
    
    def update_method_performance(self, 
                                 method: PredictionMethod,
                                 prediction: Dict[str, Any],
                                 actual_outcome: bool,
                                 context: Optional[Dict[str, Any]] = None):
        """Update performance metrics for a prediction method."""
        
        if method not in self.method_performance:
            self.method_performance[method] = MethodPerformance(method=method)
        
        performance = self.method_performance[method]
        
        # Update basic metrics
        performance.total_predictions += 1
        if actual_outcome:
            performance.correct_predictions += 1
        
        # Update accuracy
        performance.accuracy = performance.correct_predictions / performance.total_predictions
        
        # Update confidence
        predicted_confidence = prediction.get('confidence', 0.5)
        performance.average_confidence = (
            (performance.average_confidence * (performance.total_predictions - 1) + predicted_confidence)
            / performance.total_predictions
        )
        
        # Update depth
        predicted_depth = prediction.get('depth', 0)
        performance.average_depth = (
            (performance.average_depth * (performance.total_predictions - 1) + predicted_depth)
            / performance.total_predictions
        )
        
        # Update success rate
        if actual_outcome:
            performance.success_rate = (
                (performance.success_rate * (performance.total_predictions - 1) + 1.0)
                / performance.total_predictions
            )
        else:
            performance.success_rate = (
                (performance.success_rate * (performance.total_predictions - 1) + 0.0)
                / performance.total_predictions
            )
        
        # Update context-specific performance
        if context:
            context_profile = self._create_context_profile(context)
            self._update_context_performance(performance, context_profile, actual_outcome)
        
        # Update recent vs historical accuracy
        self._update_temporal_performance(performance)
        
        performance.last_updated = time.time()
        
        # Update learning statistics
        self.learning_stats['total_evaluations'] += 1
        
        logger.debug(f"Updated {method.value} performance: accuracy={performance.accuracy:.3f}")
    
    def _update_context_performance(self, 
                                   performance: MethodPerformance,
                                   context_profile: ContextProfile,
                                   actual_outcome: bool):
        """Update context-specific performance metrics."""
        
        context_key = context_profile.context_key
        
        if context_key not in performance.context_performance:
            performance.context_performance[context_key] = {
                'total_predictions': 0,
                'correct_predictions': 0,
                'accuracy': 0.0,
                'success_rate': 0.0
            }
        
        context_perf = performance.context_performance[context_key]
        context_perf['total_predictions'] += 1
        
        if actual_outcome:
            context_perf['correct_predictions'] += 1
        
        context_perf['accuracy'] = context_perf['correct_predictions'] / context_perf['total_predictions']
        
        if actual_outcome:
            context_perf['success_rate'] = (
                (context_perf['success_rate'] * (context_perf['total_predictions'] - 1) + 1.0)
                / context_perf['total_predictions']
            )
        else:
            context_perf['success_rate'] = (
                (context_perf['success_rate'] * (context_perf['total_predictions'] - 1) + 0.0)
                / context_perf['total_predictions']
            )
    
    def _update_temporal_performance(self, performance: MethodPerformance):
        """Update recent vs historical performance metrics."""
        
        # This is a simplified implementation
        # In practice, you'd maintain separate time windows for recent vs historical
        
        # For now, use a simple moving average approach
        if performance.total_predictions > 10:
            # Recent accuracy (last 10 predictions)
            recent_window = min(10, performance.total_predictions)
            recent_correct = max(0, performance.correct_predictions - (performance.total_predictions - recent_window))
            performance.recent_accuracy = recent_correct / recent_window
            
            # Historical accuracy (all predictions)
            performance.historical_accuracy = performance.accuracy
            
            # Determine trend
            if performance.recent_accuracy > performance.historical_accuracy + 0.05:
                performance.trend = "improving"
            elif performance.recent_accuracy < performance.historical_accuracy - 0.05:
                performance.trend = "declining"
            else:
                performance.trend = "stable"
    
    def start_ab_test(self, 
                     test_name: str,
                     method_a: PredictionMethod,
                     method_b: PredictionMethod,
                     context: Optional[Dict[str, Any]] = None) -> str:
        """Start an A/B test between two methods."""
        
        test_id = f"{test_name}_{int(time.time())}"
        
        self.ab_tests[test_id] = {
            'test_name': test_name,
            'method_a': method_a,
            'method_b': method_b,
            'context': context,
            'start_time': time.time(),
            'results_a': {'total': 0, 'successes': 0},
            'results_b': {'total': 0, 'successes': 0},
            'status': 'active'
        }
        
        self.active_tests[test_id] = self.ab_tests[test_id]
        
        logger.info(f"Started A/B test {test_id}: {method_a.value} vs {method_b.value}")
        return test_id
    
    def record_ab_test_result(self, 
                             test_id: str,
                             method: PredictionMethod,
                             success: bool):
        """Record a result for an active A/B test."""
        
        if test_id not in self.ab_tests:
            logger.warning(f"A/B test {test_id} not found")
            return
        
        test = self.ab_tests[test_id]
        
        if test['status'] != 'active':
            logger.warning(f"A/B test {test_id} is not active")
            return
        
        # Record result
        if method == test['method_a']:
            test['results_a']['total'] += 1
            if success:
                test['results_a']['successes'] += 1
        elif method == test['method_b']:
            test['results_b']['total'] += 1
            if success:
                test['results_b']['successes'] += 1
        
        # Check if test should be concluded
        total_results = test['results_a']['total'] + test['results_b']['total']
        if total_results >= 100:  # Minimum sample size
            self._conclude_ab_test(test_id)
    
    def _conclude_ab_test(self, test_id: str):
        """Conclude an A/B test and determine the winner."""
        
        if test_id not in self.ab_tests:
            return
        
        test = self.ab_tests[test_id]
        test['status'] = 'completed'
        test['end_time'] = time.time()
        
        # Calculate success rates
        success_rate_a = test['results_a']['successes'] / max(1, test['results_a']['total'])
        success_rate_b = test['results_b']['successes'] / max(1, test['results_b']['total'])
        
        # Determine winner
        if success_rate_a > success_rate_b + 0.05:  # 5% significance threshold
            winner = test['method_a']
            confidence = abs(success_rate_a - success_rate_b)
        elif success_rate_b > success_rate_a + 0.05:
            winner = test['method_b']
            confidence = abs(success_rate_b - success_rate_a)
        else:
            winner = None
            confidence = 0.0
        
        test['winner'] = winner
        test['confidence'] = confidence
        test['success_rate_a'] = success_rate_a
        test['success_rate_b'] = success_rate_b
        
        # Remove from active tests
        if test_id in self.active_tests:
            del self.active_tests[test_id]
        
        self.learning_stats['ab_test_results'] += 1
        
        logger.info(f"Concluded A/B test {test_id}: winner={winner.value if winner else 'tie'}, "
                   f"confidence={confidence:.3f}")
    
    def get_method_recommendations(self, 
                                 context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get recommendations for which methods to use in different contexts."""
        
        recommendations = {}
        
        # Overall best methods
        sorted_methods = sorted(
            self.method_performance.items(),
            key=lambda x: x[1].accuracy * x[1].average_confidence,
            reverse=True
        )
        
        recommendations['overall_best'] = [
            {
                'method': method.value,
                'accuracy': perf.accuracy,
                'confidence': perf.average_confidence,
                'total_predictions': perf.total_predictions
            }
            for method, perf in sorted_methods[:3]
        ]
        
        # Context-specific recommendations
        if context:
            context_profile = self._create_context_profile(context)
            similar_contexts = self._find_similar_contexts(context_profile)
            
            context_methods = {}
            for method, performance in self.method_performance.items():
                context_scores = []
                for similar_context in similar_contexts:
                    if similar_context in performance.context_performance:
                        context_perf = performance.context_performance[similar_context]
                        context_scores.append(context_perf.get('accuracy', 0.0))
                
                if context_scores:
                    context_methods[method.value] = statistics.mean(context_scores)
            
            if context_methods:
                sorted_context_methods = sorted(
                    context_methods.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                recommendations['context_specific'] = [
                    {
                        'method': method,
                        'accuracy': accuracy,
                        'contexts_considered': len(similar_contexts)
                    }
                    for method, accuracy in sorted_context_methods[:3]
                ]
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about method learning."""
        
        return {
            'learning_stats': self.learning_stats,
            'method_performance': {
                method.value: {
                    'accuracy': perf.accuracy,
                    'confidence': perf.average_confidence,
                    'total_predictions': perf.total_predictions,
                    'success_rate': perf.success_rate,
                    'trend': perf.trend,
                    'contexts_tracked': len(perf.context_performance)
                }
                for method, perf in self.method_performance.items()
            },
            'context_profiles': len(self.context_profiles),
            'active_ab_tests': len(self.active_tests),
            'completed_ab_tests': len([t for t in self.ab_tests.values() if t['status'] == 'completed']),
            'selection_strategy': self.selection_strategy
        }
