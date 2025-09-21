"""
Adaptive Learning Controller

Intelligent learning system that adapts strategies based on performance
and environmental conditions using the modular architecture.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from ..interfaces import ComponentInterface, LearningInterface
from ..caching import CacheManager, CacheConfig
from ..monitoring import TrainingMonitor, SystemMonitor


class LearningStrategy(Enum):
    """Available learning strategies."""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"
    META_LEARNING = "meta_learning"


@dataclass
class LearningContext:
    """Context for learning decisions."""
    current_performance: float
    recent_trend: str  # 'improving', 'stable', 'declining'
    resource_availability: float
    time_constraints: float
    complexity_level: int
    previous_strategies: List[LearningStrategy]
    success_rate: float


class AdaptiveLearningController(ComponentInterface, LearningInterface):
    """
    Intelligent adaptive learning controller that dynamically adjusts
    learning strategies based on performance and context.
    """
    
    def __init__(self, cache_config: Optional[CacheConfig] = None):
        """Initialize the adaptive learning controller."""
        self.cache_config = cache_config or CacheConfig()
        self.cache = CacheManager(self.cache_config)
        self.training_monitor = TrainingMonitor("adaptive_learning")
        self.system_monitor = SystemMonitor()
        
        # Learning state
        self.current_strategy = LearningStrategy.BALANCED
        self.strategy_history: List[Tuple[LearningStrategy, float, datetime]] = []
        self.performance_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'average': 0.5,
            'poor': 0.3
        }
        
        # Strategy effectiveness tracking
        self.strategy_effectiveness: Dict[LearningStrategy, List[float]] = {
            strategy: [] for strategy in LearningStrategy
        }
        
        self._initialized = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize the adaptive learning controller."""
        try:
            self.cache.initialize()
            self.training_monitor.start_monitoring()
            self.system_monitor.start_monitoring()
            self._initialized = True
            self.logger.info("Adaptive learning controller initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize adaptive learning controller: {e}")
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get current component state."""
        return {
            'name': 'AdaptiveLearningController',
            'status': 'running' if self._initialized else 'stopped',
            'last_updated': datetime.now(),
            'metadata': {
                'current_strategy': self.current_strategy.value,
                'strategy_count': len(self.strategy_history),
                'cache_stats': self.cache.get_stats()
            }
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.training_monitor.stop_monitoring()
            self.system_monitor.stop_monitoring()
            self.cache.clear()
            self._initialized = False
            self.logger.info("Adaptive learning controller cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self._initialized and self.cache.is_healthy()
    
    def learn_from_experience(self, experience: Dict[str, Any]) -> None:
        """Learn from an experience and update strategy effectiveness."""
        try:
            # Extract learning metrics
            strategy = LearningStrategy(experience.get('strategy', 'balanced'))
            performance = experience.get('performance', 0.0)
            duration = experience.get('duration', 0.0)
            
            # Update strategy effectiveness
            self.strategy_effectiveness[strategy].append(performance)
            
            # Keep only recent data (last 100 experiences per strategy)
            if len(self.strategy_effectiveness[strategy]) > 100:
                self.strategy_effectiveness[strategy] = self.strategy_effectiveness[strategy][-100:]
            
            # Record in strategy history
            self.strategy_history.append((strategy, performance, datetime.now()))
            
            # Clean old history (keep last 1000 entries)
            if len(self.strategy_history) > 1000:
                self.strategy_history = self.strategy_history[-1000:]
            
            # Cache the experience for future analysis
            cache_key = f"experience_{len(self.strategy_history)}"
            self.cache.set(cache_key, experience, ttl=3600)
            
            self.logger.debug(f"Learned from experience: {strategy.value} -> {performance:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error learning from experience: {e}")
    
    def apply_learning(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply learned knowledge to a context and recommend strategy."""
        try:
            # Create learning context
            learning_context = self._create_learning_context(context)
            
            # Select optimal strategy
            recommended_strategy = self._select_strategy(learning_context)
            
            # Generate learning parameters
            parameters = self._generate_parameters(recommended_strategy, learning_context)
            
            # Update current strategy
            self.current_strategy = recommended_strategy
            
            result = {
                'strategy': recommended_strategy.value,
                'parameters': parameters,
                'confidence': self._calculate_confidence(recommended_strategy),
                'reasoning': self._generate_reasoning(recommended_strategy, learning_context),
                'alternatives': self._get_alternative_strategies(learning_context)
            }
            
            # Cache the decision
            decision_key = f"decision_{datetime.now().timestamp()}"
            self.cache.set(decision_key, result, ttl=1800)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error applying learning: {e}")
            return {'strategy': 'balanced', 'parameters': {}, 'error': str(e)}
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics and insights."""
        try:
            # Calculate strategy effectiveness
            strategy_stats = {}
            for strategy, performances in self.strategy_effectiveness.items():
                if performances:
                    strategy_stats[strategy.value] = {
                        'count': len(performances),
                        'average': sum(performances) / len(performances),
                        'best': max(performances),
                        'worst': min(performances),
                        'trend': self._calculate_trend(performances[-10:]) if len(performances) >= 10 else 'insufficient_data'
                    }
                else:
                    strategy_stats[strategy.value] = {
                        'count': 0,
                        'average': 0.0,
                        'best': 0.0,
                        'worst': 0.0,
                        'trend': 'no_data'
                    }
            
            # Recent performance analysis
            recent_performances = [perf for _, perf, _ in self.strategy_history[-50:]]
            recent_trend = self._calculate_trend(recent_performances) if recent_performances else 'no_data'
            
            return {
                'current_strategy': self.current_strategy.value,
                'strategy_effectiveness': strategy_stats,
                'recent_trend': recent_trend,
                'total_experiences': len(self.strategy_history),
                'cache_stats': self.cache.get_stats(),
                'system_health': self.system_monitor.get_current_health().status
            }
            
        except Exception as e:
            self.logger.error(f"Error getting learning stats: {e}")
            return {'error': str(e)}
    
    def _create_learning_context(self, context: Dict[str, Any]) -> LearningContext:
        """Create a learning context from raw context data."""
        # Get recent performance trend
        recent_performances = [perf for _, perf, _ in self.strategy_history[-10:]]
        trend = self._calculate_trend(recent_performances) if recent_performances else 'stable'
        
        # Get system health
        system_health = self.system_monitor.get_current_health()
        resource_availability = 1.0 - (system_health.cpu_usage + system_health.memory_usage) / 200.0
        
        # Calculate success rate
        recent_strategies = [strategy for strategy, _, _ in self.strategy_history[-20:]]
        success_rate = len([p for _, p, _ in self.strategy_history[-20:] if p > 0.7]) / max(1, len(self.strategy_history[-20:]))
        
        return LearningContext(
            current_performance=context.get('performance', 0.5),
            recent_trend=trend,
            resource_availability=max(0.0, min(1.0, resource_availability)),
            time_constraints=context.get('time_constraints', 1.0),
            complexity_level=context.get('complexity_level', 1),
            previous_strategies=recent_strategies,
            success_rate=success_rate
        )
    
    def _select_strategy(self, context: LearningContext) -> LearningStrategy:
        """Select the optimal learning strategy based on context."""
        # If performance is excellent, exploit current knowledge
        if context.current_performance >= self.performance_thresholds['excellent']:
            if context.recent_trend == 'improving':
                return LearningStrategy.EXPLOITATION
            else:
                return LearningStrategy.BALANCED
        
        # If performance is poor, explore new approaches
        elif context.current_performance < self.performance_thresholds['poor']:
            return LearningStrategy.EXPLORATION
        
        # If resources are limited, use meta-learning
        elif context.resource_availability < 0.3:
            return LearningStrategy.META_LEARNING
        
        # If time constraints are high, use adaptive strategy
        elif context.time_constraints < 0.5:
            return LearningStrategy.ADAPTIVE
        
        # Default to balanced approach
        else:
            return LearningStrategy.BALANCED
    
    def _generate_parameters(self, strategy: LearningStrategy, context: LearningContext) -> Dict[str, Any]:
        """Generate parameters for the selected strategy."""
        base_params = {
            'learning_rate': 0.01,
            'batch_size': 32,
            'patience': 10,
            'exploration_rate': 0.1
        }
        
        if strategy == LearningStrategy.EXPLORATION:
            return {
                **base_params,
                'learning_rate': 0.05,
                'exploration_rate': 0.3,
                'patience': 5,
                'diversity_weight': 0.8
            }
        elif strategy == LearningStrategy.EXPLOITATION:
            return {
                **base_params,
                'learning_rate': 0.001,
                'exploration_rate': 0.01,
                'patience': 20,
                'exploitation_weight': 0.9
            }
        elif strategy == LearningStrategy.META_LEARNING:
            return {
                **base_params,
                'learning_rate': 0.02,
                'meta_learning_rate': 0.1,
                'adaptation_steps': 5,
                'memory_size': 1000
            }
        elif strategy == LearningStrategy.ADAPTIVE:
            return {
                **base_params,
                'learning_rate': 0.01 * context.resource_availability,
                'exploration_rate': 0.1 * (1.0 - context.current_performance),
                'patience': int(10 * context.time_constraints),
                'adaptation_rate': 0.1
            }
        else:  # BALANCED
            return base_params
    
    def _calculate_confidence(self, strategy: LearningStrategy) -> float:
        """Calculate confidence in strategy recommendation."""
        if strategy not in self.strategy_effectiveness:
            return 0.5
        
        performances = self.strategy_effectiveness[strategy]
        if not performances:
            return 0.5
        
        # Confidence based on consistency and recent performance
        avg_performance = sum(performances) / len(performances)
        consistency = 1.0 - (max(performances) - min(performances)) if len(performances) > 1 else 1.0
        recency_bonus = min(1.0, len(performances) / 10.0)
        
        return min(1.0, (avg_performance * 0.5 + consistency * 0.3 + recency_bonus * 0.2))
    
    def _generate_reasoning(self, strategy: LearningStrategy, context: LearningContext) -> str:
        """Generate human-readable reasoning for strategy selection."""
        reasons = []
        
        if context.current_performance >= self.performance_thresholds['excellent']:
            reasons.append("High performance detected")
        elif context.current_performance < self.performance_thresholds['poor']:
            reasons.append("Low performance requires exploration")
        
        if context.recent_trend == 'improving':
            reasons.append("Positive trend suggests exploitation")
        elif context.recent_trend == 'declining':
            reasons.append("Declining trend suggests exploration")
        
        if context.resource_availability < 0.3:
            reasons.append("Limited resources favor meta-learning")
        
        if context.time_constraints < 0.5:
            reasons.append("Time constraints favor adaptive approach")
        
        if not reasons:
            reasons.append("Balanced approach for stable conditions")
        
        return f"Selected {strategy.value}: {', '.join(reasons)}"
    
    def _get_alternative_strategies(self, context: LearningContext) -> List[Dict[str, Any]]:
        """Get alternative strategy recommendations."""
        alternatives = []
        for strategy in LearningStrategy:
            if strategy != self.current_strategy:
                confidence = self._calculate_confidence(strategy)
                if confidence > 0.3:  # Only include viable alternatives
                    alternatives.append({
                        'strategy': strategy.value,
                        'confidence': confidence,
                        'parameters': self._generate_parameters(strategy, context)
                    })
        
        return sorted(alternatives, key=lambda x: x['confidence'], reverse=True)
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from a list of values."""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend calculation
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * val for i, val in enumerate(values))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'
