#!/usr/bin/env python3
"""
Conscious-Like Behavior Evaluator for Tabula Rasa

Implements formal evaluation metrics for conscious-like behavior,
building on existing consciousness metrics and performance tracking.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import math
from collections import deque
import json

logger = logging.getLogger(__name__)

class BehaviorMetric(Enum):
    """Types of conscious-like behavior metrics."""
    FLEXIBILITY = "flexibility"
    INTROSPECTION = "introspection"
    GENERALIZATION = "generalization"
    ROBUSTNESS = "robustness"
    COHERENCE = "coherence"
    ADAPTABILITY = "adaptability"
    CREATIVITY = "creativity"
    SELF_AWARENESS = "self_awareness"

class EvaluationContext(Enum):
    """Contexts for behavior evaluation."""
    TASK_SWITCHING = "task_switching"
    NOVEL_SITUATIONS = "novel_situations"
    ADVERSARIAL_INPUTS = "adversarial_inputs"
    UNCERTAINTY_HANDLING = "uncertainty_handling"
    ERROR_RECOVERY = "error_recovery"
    STRATEGY_SELECTION = "strategy_selection"

@dataclass
class BehaviorObservation:
    """Single observation of system behavior."""
    timestamp: float
    context: EvaluationContext
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    performance_metrics: Dict[str, float]
    confidence_level: float
    strategy_used: str
    error_occurred: bool
    recovery_time: Optional[float] = None

@dataclass
class BehaviorEvaluation:
    """Evaluation of conscious-like behavior."""
    metric: BehaviorMetric
    score: float
    confidence: float
    context: EvaluationContext
    observations_used: int
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConsciousBehaviorProfile:
    """Complete profile of conscious-like behavior."""
    overall_score: float
    metric_scores: Dict[BehaviorMetric, float]
    strengths: List[BehaviorMetric]
    weaknesses: List[BehaviorMetric]
    recommendations: List[str]
    evaluation_timestamp: float
    total_observations: int

class FlexibilityMetrics:
    """Metrics for evaluating behavioral flexibility."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.strategy_history = deque(maxlen=window_size)
        self.performance_history = deque(maxlen=window_size)
        self.context_history = deque(maxlen=window_size)
    
    def add_observation(self, observation: BehaviorObservation):
        """Add new observation for flexibility analysis."""
        self.strategy_history.append(observation.strategy_used)
        self.performance_history.append(observation.performance_metrics.get('overall_score', 0.0))
        self.context_history.append(observation.context)
    
    def evaluate_flexibility(self) -> BehaviorEvaluation:
        """Evaluate behavioral flexibility."""
        if len(self.strategy_history) < 5:
            return BehaviorEvaluation(
                metric=BehaviorMetric.FLEXIBILITY,
                score=0.0,
                confidence=0.0,
                context=EvaluationContext.TASK_SWITCHING,
                observations_used=len(self.strategy_history),
                timestamp=time.time()
            )
        
        # Calculate strategy diversity
        unique_strategies = len(set(self.strategy_history))
        total_strategies = len(self.strategy_history)
        strategy_diversity = unique_strategies / total_strategies
        
        # Calculate context adaptation
        context_adaptation = self._calculate_context_adaptation()
        
        # Calculate performance consistency across strategies
        performance_consistency = self._calculate_performance_consistency()
        
        # Calculate strategy switching frequency
        switching_frequency = self._calculate_switching_frequency()
        
        # Combine metrics
        flexibility_score = (
            strategy_diversity * 0.3 +
            context_adaptation * 0.3 +
            performance_consistency * 0.2 +
            switching_frequency * 0.2
        )
        
        confidence = min(1.0, len(self.strategy_history) / self.window_size)
        
        return BehaviorEvaluation(
            metric=BehaviorMetric.FLEXIBILITY,
            score=flexibility_score,
            confidence=confidence,
            context=EvaluationContext.TASK_SWITCHING,
            observations_used=len(self.strategy_history),
            timestamp=time.time(),
            details={
                'strategy_diversity': strategy_diversity,
                'context_adaptation': context_adaptation,
                'performance_consistency': performance_consistency,
                'switching_frequency': switching_frequency
            }
        )
    
    def _calculate_context_adaptation(self) -> float:
        """Calculate how well the system adapts to different contexts."""
        if len(self.context_history) < 2:
            return 0.0
        
        # Calculate performance variance across contexts
        context_performance = {}
        for i, context in enumerate(self.context_history):
            if context not in context_performance:
                context_performance[context] = []
            context_performance[context].append(self.performance_history[i])
        
        if len(context_performance) < 2:
            return 0.0
        
        # Calculate average performance per context
        context_avg_performance = {
            ctx: np.mean(perfs) for ctx, perfs in context_performance.items()
        }
        
        # Calculate variance in performance across contexts
        performance_values = list(context_avg_performance.values())
        performance_variance = np.var(performance_values)
        
        # Higher variance indicates better adaptation (assuming good performance)
        adaptation_score = min(1.0, performance_variance * 2)
        
        return adaptation_score
    
    def _calculate_performance_consistency(self) -> float:
        """Calculate consistency of performance across strategies."""
        if len(self.performance_history) < 3:
            return 0.0
        
        # Calculate coefficient of variation
        mean_performance = np.mean(self.performance_history)
        std_performance = np.std(self.performance_history)
        
        if mean_performance == 0:
            return 0.0
        
        cv = std_performance / mean_performance
        consistency_score = max(0.0, 1.0 - cv)
        
        return consistency_score
    
    def _calculate_switching_frequency(self) -> float:
        """Calculate frequency of strategy switching."""
        if len(self.strategy_history) < 2:
            return 0.0
        
        switches = 0
        for i in range(1, len(self.strategy_history)):
            if self.strategy_history[i] != self.strategy_history[i-1]:
                switches += 1
        
        switching_rate = switches / (len(self.strategy_history) - 1)
        
        # Optimal switching rate (not too high, not too low)
        optimal_rate = 0.3
        switching_score = 1.0 - abs(switching_rate - optimal_rate) / optimal_rate
        
        return max(0.0, switching_score)

class IntrospectionMetrics:
    """Metrics for evaluating introspective capabilities."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.confidence_history = deque(maxlen=window_size)
        self.error_history = deque(maxlen=window_size)
        self.recovery_history = deque(maxlen=window_size)
        self.self_reports = deque(maxlen=window_size)
    
    def add_observation(self, observation: BehaviorObservation):
        """Add new observation for introspection analysis."""
        self.confidence_history.append(observation.confidence_level)
        self.error_history.append(observation.error_occurred)
        if observation.recovery_time is not None:
            self.recovery_history.append(observation.recovery_time)
        # Add self-report if available
        if 'self_report' in observation.output_data:
            self.self_reports.append(observation.output_data['self_report'])
    
    def evaluate_introspection(self) -> BehaviorEvaluation:
        """Evaluate introspective capabilities."""
        if len(self.confidence_history) < 5:
            return BehaviorEvaluation(
                metric=BehaviorMetric.INTROSPECTION,
                score=0.0,
                confidence=0.0,
                context=EvaluationContext.UNCERTAINTY_HANDLING,
                observations_used=len(self.confidence_history),
                timestamp=time.time()
            )
        
        # Calculate confidence calibration
        confidence_calibration = self._calculate_confidence_calibration()
        
        # Calculate error detection accuracy
        error_detection = self._calculate_error_detection()
        
        # Calculate recovery effectiveness
        recovery_effectiveness = self._calculate_recovery_effectiveness()
        
        # Calculate self-report quality
        self_report_quality = self._calculate_self_report_quality()
        
        # Combine metrics
        introspection_score = (
            confidence_calibration * 0.3 +
            error_detection * 0.3 +
            recovery_effectiveness * 0.2 +
            self_report_quality * 0.2
        )
        
        confidence = min(1.0, len(self.confidence_history) / self.window_size)
        
        return BehaviorEvaluation(
            metric=BehaviorMetric.INTROSPECTION,
            score=introspection_score,
            confidence=confidence,
            context=EvaluationContext.UNCERTAINTY_HANDLING,
            observations_used=len(self.confidence_history),
            timestamp=time.time(),
            details={
                'confidence_calibration': confidence_calibration,
                'error_detection': error_detection,
                'recovery_effectiveness': recovery_effectiveness,
                'self_report_quality': self_report_quality
            }
        )
    
    def _calculate_confidence_calibration(self) -> float:
        """Calculate how well confidence correlates with actual performance."""
        if len(self.confidence_history) < 3:
            return 0.0
        
        # Calculate correlation between confidence and performance
        # This is simplified - in practice, you'd need performance data
        confidence_values = list(self.confidence_history)
        performance_values = [0.8] * len(confidence_values)  # Placeholder
        
        correlation = np.corrcoef(confidence_values, performance_values)[0, 1]
        
        # Convert correlation to 0-1 score
        calibration_score = (correlation + 1) / 2
        
        return max(0.0, calibration_score)
    
    def _calculate_error_detection(self) -> float:
        """Calculate accuracy of error detection."""
        if len(self.error_history) < 3:
            return 0.0
        
        # Calculate error rate
        error_rate = sum(self.error_history) / len(self.error_history)
        
        # Calculate how well the system detects its own errors
        # This is simplified - in practice, you'd need more sophisticated error detection
        error_detection_score = 1.0 - error_rate
        
        return max(0.0, error_detection_score)
    
    def _calculate_recovery_effectiveness(self) -> float:
        """Calculate effectiveness of error recovery."""
        if len(self.recovery_history) < 2:
            return 0.0
        
        # Calculate average recovery time
        avg_recovery_time = np.mean(self.recovery_history)
        
        # Shorter recovery time is better
        max_recovery_time = 10.0  # seconds
        recovery_score = max(0.0, 1.0 - (avg_recovery_time / max_recovery_time))
        
        return recovery_score
    
    def _calculate_self_report_quality(self) -> float:
        """Calculate quality of self-reports."""
        if len(self.self_reports) < 2:
            return 0.0
        
        # Calculate diversity and consistency of self-reports
        unique_reports = len(set(str(report) for report in self.self_reports))
        total_reports = len(self.self_reports)
        
        diversity_score = unique_reports / total_reports
        
        # Calculate consistency (simplified)
        consistency_score = 0.7  # Placeholder
        
        quality_score = (diversity_score + consistency_score) / 2
        
        return quality_score

class GeneralizationMetrics:
    """Metrics for evaluating generalization capabilities."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.task_performance = {}
        self.novel_task_performance = {}
        self.transfer_scores = {}
    
    def add_observation(self, observation: BehaviorObservation):
        """Add new observation for generalization analysis."""
        task_type = observation.input_data.get('task_type', 'unknown')
        is_novel = observation.input_data.get('is_novel', False)
        performance = observation.performance_metrics.get('overall_score', 0.0)
        
        if is_novel:
            if task_type not in self.novel_task_performance:
                self.novel_task_performance[task_type] = []
            self.novel_task_performance[task_type].append(performance)
        else:
            if task_type not in self.task_performance:
                self.task_performance[task_type] = []
            self.task_performance[task_type].append(performance)
    
    def evaluate_generalization(self) -> BehaviorEvaluation:
        """Evaluate generalization capabilities."""
        if len(self.task_performance) < 2:
            return BehaviorEvaluation(
                metric=BehaviorMetric.GENERALIZATION,
                score=0.0,
                confidence=0.0,
                context=EvaluationContext.NOVEL_SITUATIONS,
                observations_used=len(self.task_performance),
                timestamp=time.time()
            )
        
        # Calculate performance on novel tasks
        novel_performance = self._calculate_novel_task_performance()
        
        # Calculate transfer learning effectiveness
        transfer_effectiveness = self._calculate_transfer_effectiveness()
        
        # Calculate task similarity adaptation
        similarity_adaptation = self._calculate_similarity_adaptation()
        
        # Combine metrics
        generalization_score = (
            novel_performance * 0.4 +
            transfer_effectiveness * 0.3 +
            similarity_adaptation * 0.3
        )
        
        total_observations = sum(len(perfs) for perfs in self.task_performance.values()) + \
                           sum(len(perfs) for perfs in self.novel_task_performance.values())
        confidence = min(1.0, total_observations / (self.window_size * 2))
        
        return BehaviorEvaluation(
            metric=BehaviorMetric.GENERALIZATION,
            score=generalization_score,
            confidence=confidence,
            context=EvaluationContext.NOVEL_SITUATIONS,
            observations_used=total_observations,
            timestamp=time.time(),
            details={
                'novel_performance': novel_performance,
                'transfer_effectiveness': transfer_effectiveness,
                'similarity_adaptation': similarity_adaptation
            }
        )
    
    def _calculate_novel_task_performance(self) -> float:
        """Calculate performance on novel tasks."""
        if not self.novel_task_performance:
            return 0.0
        
        all_novel_performances = []
        for perfs in self.novel_task_performance.values():
            all_novel_performances.extend(perfs)
        
        if not all_novel_performances:
            return 0.0
        
        avg_novel_performance = np.mean(all_novel_performances)
        return avg_novel_performance
    
    def _calculate_transfer_effectiveness(self) -> float:
        """Calculate effectiveness of transfer learning."""
        if not self.task_performance or not self.novel_task_performance:
            return 0.0
        
        # Calculate average performance on known tasks
        all_known_performances = []
        for perfs in self.task_performance.values():
            all_known_performances.extend(perfs)
        
        if not all_known_performances:
            return 0.0
        
        avg_known_performance = np.mean(all_known_performances)
        
        # Calculate average performance on novel tasks
        all_novel_performances = []
        for perfs in self.novel_task_performance.values():
            all_novel_performances.extend(perfs)
        
        if not all_novel_performances:
            return 0.0
        
        avg_novel_performance = np.mean(all_novel_performances)
        
        # Transfer effectiveness is the ratio of novel to known performance
        transfer_ratio = avg_novel_performance / avg_known_performance
        transfer_effectiveness = min(1.0, transfer_ratio)
        
        return transfer_effectiveness
    
    def _calculate_similarity_adaptation(self) -> float:
        """Calculate adaptation to similar tasks."""
        # Simplified similarity adaptation calculation
        return 0.7  # Placeholder

class RobustnessMetrics:
    """Metrics for evaluating robustness to adversarial inputs."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.adversarial_performance = deque(maxlen=window_size)
        self.normal_performance = deque(maxlen=window_size)
        self.degradation_scores = deque(maxlen=window_size)
    
    def add_observation(self, observation: BehaviorObservation):
        """Add new observation for robustness analysis."""
        is_adversarial = observation.input_data.get('is_adversarial', False)
        performance = observation.performance_metrics.get('overall_score', 0.0)
        
        if is_adversarial:
            self.adversarial_performance.append(performance)
        else:
            self.normal_performance.append(performance)
    
    def evaluate_robustness(self) -> BehaviorEvaluation:
        """Evaluate robustness to adversarial inputs."""
        if len(self.adversarial_performance) < 3 or len(self.normal_performance) < 3:
            return BehaviorEvaluation(
                metric=BehaviorMetric.ROBUSTNESS,
                score=0.0,
                confidence=0.0,
                context=EvaluationContext.ADVERSARIAL_INPUTS,
                observations_used=len(self.adversarial_performance) + len(self.normal_performance),
                timestamp=time.time()
            )
        
        # Calculate performance degradation under adversarial inputs
        avg_adversarial = np.mean(self.adversarial_performance)
        avg_normal = np.mean(self.normal_performance)
        
        if avg_normal == 0:
            degradation_ratio = 1.0
        else:
            degradation_ratio = avg_adversarial / avg_normal
        
        # Calculate robustness score
        robustness_score = min(1.0, degradation_ratio)
        
        # Calculate confidence
        total_observations = len(self.adversarial_performance) + len(self.normal_performance)
        confidence = min(1.0, total_observations / (self.window_size * 2))
        
        return BehaviorEvaluation(
            metric=BehaviorMetric.ROBUSTNESS,
            score=robustness_score,
            confidence=confidence,
            context=EvaluationContext.ADVERSARIAL_INPUTS,
            observations_used=total_observations,
            timestamp=time.time(),
            details={
                'adversarial_performance': avg_adversarial,
                'normal_performance': avg_normal,
                'degradation_ratio': degradation_ratio
            }
        )

class ConsciousBehaviorEvaluator:
    """Main evaluator for conscious-like behavior."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        
        # Initialize metric evaluators
        self.flexibility_metrics = FlexibilityMetrics(window_size)
        self.introspection_metrics = IntrospectionMetrics(window_size)
        self.generalization_metrics = GeneralizationMetrics(window_size)
        self.robustness_metrics = RobustnessMetrics(window_size)
        
        # History
        self.observations = deque(maxlen=window_size * 2)
        self.evaluations = deque(maxlen=100)
        
        logger.info("Conscious Behavior Evaluator initialized")
    
    def add_observation(self, observation: BehaviorObservation):
        """Add new behavior observation."""
        self.observations.append(observation)
        
        # Add to metric evaluators
        self.flexibility_metrics.add_observation(observation)
        self.introspection_metrics.add_observation(observation)
        self.generalization_metrics.add_observation(observation)
        self.robustness_metrics.add_observation(observation)
    
    def evaluate_conscious_behavior(self) -> ConsciousBehaviorProfile:
        """Evaluate conscious-like behavior across all metrics."""
        
        # Evaluate each metric
        flexibility_eval = self.flexibility_metrics.evaluate_flexibility()
        introspection_eval = self.introspection_metrics.evaluate_introspection()
        generalization_eval = self.generalization_metrics.evaluate_generalization()
        robustness_eval = self.robustness_metrics.evaluate_robustness()
        
        # Store evaluations
        evaluations = [flexibility_eval, introspection_eval, generalization_eval, robustness_eval]
        self.evaluations.extend(evaluations)
        
        # Calculate metric scores
        metric_scores = {
            BehaviorMetric.FLEXIBILITY: flexibility_eval.score,
            BehaviorMetric.INTROSPECTION: introspection_eval.score,
            BehaviorMetric.GENERALIZATION: generalization_eval.score,
            BehaviorMetric.ROBUSTNESS: robustness_eval.score
        }
        
        # Calculate overall score
        overall_score = np.mean(list(metric_scores.values()))
        
        # Identify strengths and weaknesses
        strengths = [metric for metric, score in metric_scores.items() if score > 0.7]
        weaknesses = [metric for metric, score in metric_scores.items() if score < 0.4]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metric_scores)
        
        # Create behavior profile
        profile = ConsciousBehaviorProfile(
            overall_score=overall_score,
            metric_scores=metric_scores,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            evaluation_timestamp=time.time(),
            total_observations=len(self.observations)
        )
        
        return profile
    
    def _generate_recommendations(self, metric_scores: Dict[BehaviorMetric, float]) -> List[str]:
        """Generate recommendations based on metric scores."""
        recommendations = []
        
        if metric_scores[BehaviorMetric.FLEXIBILITY] < 0.5:
            recommendations.append("Improve strategy diversity and context adaptation")
        
        if metric_scores[BehaviorMetric.INTROSPECTION] < 0.5:
            recommendations.append("Enhance confidence calibration and error detection")
        
        if metric_scores[BehaviorMetric.GENERALIZATION] < 0.5:
            recommendations.append("Improve transfer learning and novel task performance")
        
        if metric_scores[BehaviorMetric.ROBUSTNESS] < 0.5:
            recommendations.append("Strengthen robustness to adversarial inputs")
        
        if not recommendations:
            recommendations.append("Continue current approach - all metrics are performing well")
        
        return recommendations
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get statistics about behavior evaluations."""
        if not self.evaluations:
            return {}
        
        # Calculate average scores by metric
        metric_averages = {}
        for metric in BehaviorMetric:
            metric_evals = [eval for eval in self.evaluations if eval.metric == metric]
            if metric_evals:
                metric_averages[metric.value] = np.mean([eval.score for eval in metric_evals])
        
        return {
            'total_evaluations': len(self.evaluations),
            'total_observations': len(self.observations),
            'metric_averages': metric_averages,
            'recent_evaluations': [
                {
                    'metric': eval.metric.value,
                    'score': eval.score,
                    'confidence': eval.confidence,
                    'timestamp': eval.timestamp
                }
                for eval in list(self.evaluations)[-10:]
            ]
        }

# Factory function for easy integration
def create_conscious_behavior_evaluator(**kwargs) -> ConsciousBehaviorEvaluator:
    """Create a configured conscious behavior evaluator."""
    return ConsciousBehaviorEvaluator(**kwargs)
