"""
Exploration-Related Cognitive Subsystems

Implements 5 exploration-focused subsystems for comprehensive exploration monitoring and management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np

from .base_subsystem import BaseCognitiveSubsystem, SubsystemHealth

logger = logging.getLogger(__name__)

class ExplorationStrategyMonitor(BaseCognitiveSubsystem):
    """Monitors exploration strategies and effectiveness."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="exploration_strategy",
            name="Exploration Strategy Monitor",
            description="Tracks exploration strategies and their effectiveness"
        )
        self.exploration_events = []
        self.strategy_effectiveness = []
        self.exploration_coverage = []
        self.strategy_switches = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize exploration strategy monitoring."""
        self.exploration_events = []
        self.strategy_effectiveness = []
        self.exploration_coverage = []
        self.strategy_switches = []
        logger.info("Exploration Strategy Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect exploration strategy metrics."""
        current_time = datetime.now()
        
        # Calculate strategy effectiveness
        strategy_effectiveness = np.mean(self.strategy_effectiveness) if self.strategy_effectiveness else 0.0
        
        # Calculate exploration coverage
        exploration_coverage = np.mean(self.exploration_coverage) if self.exploration_coverage else 0.0
        
        # Count strategy switches
        total_switches = len(self.strategy_switches)
        recent_switches = len([
            switch for switch in self.strategy_switches
            if (current_time - switch.get('timestamp', current_time)).seconds < 1800
        ])
        
        # Calculate strategy diversity
        active_strategies = set([event.get('strategy', 'unknown') for event in self.exploration_events])
        strategy_diversity = len(active_strategies) / max(len(self.exploration_events), 1)
        
        # Calculate exploration frequency
        total_explorations = len(self.exploration_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        exploration_frequency = total_explorations / hours_elapsed
        
        # Calculate coverage trend
        coverage_trend = 0.0
        if len(self.exploration_coverage) > 1:
            recent_coverage = self.exploration_coverage[-10:]
            if len(recent_coverage) > 1:
                coverage_trend = np.polyfit(range(len(recent_coverage)), recent_coverage, 1)[0]
        
        return {
            'strategy_effectiveness': strategy_effectiveness,
            'exploration_coverage': exploration_coverage,
            'strategy_diversity': strategy_diversity,
            'coverage_trend': coverage_trend,
            'total_explorations': total_explorations,
            'total_strategy_switches': total_switches,
            'recent_strategy_switches': recent_switches,
            'exploration_frequency': exploration_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze exploration strategy health."""
        effectiveness = metrics['strategy_effectiveness']
        coverage = metrics['exploration_coverage']
        diversity = metrics['strategy_diversity']
        
        if effectiveness < 0.3 or coverage < 0.2 or diversity < 0.1:
            return SubsystemHealth.CRITICAL
        elif effectiveness < 0.5 or coverage < 0.4 or diversity < 0.3:
            return SubsystemHealth.WARNING
        elif effectiveness < 0.7:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on effectiveness and coverage."""
        effectiveness = metrics['strategy_effectiveness']
        coverage = metrics['exploration_coverage']
        
        return (effectiveness + coverage) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on diversity and trend."""
        diversity = metrics['strategy_diversity']
        trend = metrics['coverage_trend']
        
        # Normalize trend (positive is good)
        trend_score = max(0, min(1.0, trend * 10))
        
        return (diversity + trend_score) / 2

class BoredomDetectionMonitor(BaseCognitiveSubsystem):
    """Monitors boredom detection and response."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="boredom_detection",
            name="Boredom Detection Monitor",
            description="Tracks boredom detection and response mechanisms"
        )
        self.boredom_events = []
        self.boredom_levels = []
        self.response_effectiveness = []
        self.boredom_duration = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize boredom detection monitoring."""
        self.boredom_events = []
        self.boredom_levels = []
        self.response_effectiveness = []
        self.boredom_duration = []
        logger.info("Boredom Detection Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect boredom detection metrics."""
        current_time = datetime.now()
        
        # Calculate average boredom level
        avg_boredom = np.mean(self.boredom_levels) if self.boredom_levels else 0.0
        
        # Calculate response effectiveness
        response_effectiveness = np.mean(self.response_effectiveness) if self.response_effectiveness else 0.0
        
        # Calculate average boredom duration
        avg_duration = np.mean(self.boredom_duration) if self.boredom_duration else 0.0
        
        # Count recent boredom events
        recent_boredom = len([
            event for event in self.boredom_events
            if (current_time - event.get('timestamp', current_time)).seconds < 3600
        ])
        
        # Calculate boredom frequency
        total_boredom_events = len(self.boredom_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        boredom_frequency = total_boredom_events / hours_elapsed
        
        # Calculate boredom trend
        boredom_trend = 0.0
        if len(self.boredom_levels) > 1:
            recent_boredom_levels = self.boredom_levels[-10:]
            if len(recent_boredom_levels) > 1:
                boredom_trend = np.polyfit(range(len(recent_boredom_levels)), recent_boredom_levels, 1)[0]
        
        # Calculate detection accuracy
        detection_accuracy = 0.0
        if self.boredom_events:
            accurate_detections = len([event for event in self.boredom_events if event.get('accurate', False)])
            detection_accuracy = accurate_detections / len(self.boredom_events)
        
        return {
            'avg_boredom_level': avg_boredom,
            'response_effectiveness': response_effectiveness,
            'avg_boredom_duration': avg_duration,
            'boredom_trend': boredom_trend,
            'detection_accuracy': detection_accuracy,
            'total_boredom_events': total_boredom_events,
            'recent_boredom_events': recent_boredom,
            'boredom_frequency': boredom_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze boredom detection health."""
        boredom_level = metrics['avg_boredom_level']
        response_effectiveness = metrics['response_effectiveness']
        detection_accuracy = metrics['detection_accuracy']
        
        if boredom_level > 0.8 or response_effectiveness < 0.3 or detection_accuracy < 0.5:
            return SubsystemHealth.CRITICAL
        elif boredom_level > 0.6 or response_effectiveness < 0.5 or detection_accuracy < 0.7:
            return SubsystemHealth.WARNING
        elif boredom_level > 0.4:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on response effectiveness and detection accuracy."""
        response_effectiveness = metrics['response_effectiveness']
        detection_accuracy = metrics['detection_accuracy']
        
        return (response_effectiveness + detection_accuracy) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on boredom level and trend."""
        boredom_level = metrics['avg_boredom_level']
        trend = metrics['boredom_trend']
        
        # Lower boredom is better
        boredom_score = max(0, 1 - boredom_level)
        
        # Negative trend is better (decreasing boredom)
        trend_score = max(0, 1 - abs(trend) / 0.1) if trend < 0 else max(0, 1 - trend / 0.1)
        
        return (boredom_score + trend_score) / 2

class StagnationDetectionMonitor(BaseCognitiveSubsystem):
    """Monitors stagnation detection and response."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="stagnation_detection",
            name="Stagnation Detection Monitor",
            description="Tracks stagnation detection and response mechanisms"
        )
        self.stagnation_events = []
        self.stagnation_levels = []
        self.recovery_effectiveness = []
        self.stagnation_duration = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize stagnation detection monitoring."""
        self.stagnation_events = []
        self.stagnation_levels = []
        self.recovery_effectiveness = []
        self.stagnation_duration = []
        logger.info("Stagnation Detection Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect stagnation detection metrics."""
        current_time = datetime.now()
        
        # Calculate average stagnation level
        avg_stagnation = np.mean(self.stagnation_levels) if self.stagnation_levels else 0.0
        
        # Calculate recovery effectiveness
        recovery_effectiveness = np.mean(self.recovery_effectiveness) if self.recovery_effectiveness else 0.0
        
        # Calculate average stagnation duration
        avg_duration = np.mean(self.stagnation_duration) if self.stagnation_duration else 0.0
        
        # Count recent stagnation events
        recent_stagnation = len([
            event for event in self.stagnation_events
            if (current_time - event.get('timestamp', current_time)).seconds < 3600
        ])
        
        # Calculate stagnation frequency
        total_stagnation_events = len(self.stagnation_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        stagnation_frequency = total_stagnation_events / hours_elapsed
        
        # Calculate stagnation trend
        stagnation_trend = 0.0
        if len(self.stagnation_levels) > 1:
            recent_stagnation_levels = self.stagnation_levels[-10:]
            if len(recent_stagnation_levels) > 1:
                stagnation_trend = np.polyfit(range(len(recent_stagnation_levels)), recent_stagnation_levels, 1)[0]
        
        # Calculate detection accuracy
        detection_accuracy = 0.0
        if self.stagnation_events:
            accurate_detections = len([event for event in self.stagnation_events if event.get('accurate', False)])
            detection_accuracy = accurate_detections / len(self.stagnation_events)
        
        return {
            'avg_stagnation_level': avg_stagnation,
            'recovery_effectiveness': recovery_effectiveness,
            'avg_stagnation_duration': avg_duration,
            'stagnation_trend': stagnation_trend,
            'detection_accuracy': detection_accuracy,
            'total_stagnation_events': total_stagnation_events,
            'recent_stagnation_events': recent_stagnation,
            'stagnation_frequency': stagnation_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze stagnation detection health."""
        stagnation_level = metrics['avg_stagnation_level']
        recovery_effectiveness = metrics['recovery_effectiveness']
        detection_accuracy = metrics['detection_accuracy']
        
        if stagnation_level > 0.8 or recovery_effectiveness < 0.3 or detection_accuracy < 0.5:
            return SubsystemHealth.CRITICAL
        elif stagnation_level > 0.6 or recovery_effectiveness < 0.5 or detection_accuracy < 0.7:
            return SubsystemHealth.WARNING
        elif stagnation_level > 0.4:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on recovery effectiveness and detection accuracy."""
        recovery_effectiveness = metrics['recovery_effectiveness']
        detection_accuracy = metrics['detection_accuracy']
        
        return (recovery_effectiveness + detection_accuracy) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on stagnation level and trend."""
        stagnation_level = metrics['avg_stagnation_level']
        trend = metrics['stagnation_trend']
        
        # Lower stagnation is better
        stagnation_score = max(0, 1 - stagnation_level)
        
        # Negative trend is better (decreasing stagnation)
        trend_score = max(0, 1 - abs(trend) / 0.1) if trend < 0 else max(0, 1 - trend / 0.1)
        
        return (stagnation_score + trend_score) / 2

class ContrarianStrategyMonitor(BaseCognitiveSubsystem):
    """Monitors contrarian strategy effectiveness."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="contrarian_strategy",
            name="Contrarian Strategy Monitor",
            description="Tracks contrarian strategy effectiveness and counter-trend behavior"
        )
        self.contrarian_events = []
        self.contrarian_effectiveness = []
        self.counter_trend_success = []
        self.contrarian_frequency = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize contrarian strategy monitoring."""
        self.contrarian_events = []
        self.contrarian_effectiveness = []
        self.counter_trend_success = []
        self.contrarian_frequency = []
        logger.info("Contrarian Strategy Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect contrarian strategy metrics."""
        current_time = datetime.now()
        
        # Calculate contrarian effectiveness
        contrarian_effectiveness = np.mean(self.contrarian_effectiveness) if self.contrarian_effectiveness else 0.0
        
        # Calculate counter-trend success rate
        counter_trend_success = np.mean(self.counter_trend_success) if self.counter_trend_success else 0.0
        
        # Calculate contrarian frequency
        avg_frequency = np.mean(self.contrarian_frequency) if self.contrarian_frequency else 0.0
        
        # Count recent contrarian events
        recent_contrarian = len([
            event for event in self.contrarian_events
            if (current_time - event.get('timestamp', current_time)).seconds < 1800
        ])
        
        # Calculate contrarian diversity
        contrarian_types = set([event.get('contrarian_type', 'unknown') for event in self.contrarian_events])
        contrarian_diversity = len(contrarian_types) / max(len(self.contrarian_events), 1)
        
        # Calculate total contrarian events
        total_contrarian_events = len(self.contrarian_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        contrarian_frequency = total_contrarian_events / hours_elapsed
        
        # Calculate contrarian trend
        contrarian_trend = 0.0
        if len(self.contrarian_effectiveness) > 1:
            recent_effectiveness = self.contrarian_effectiveness[-10:]
            if len(recent_effectiveness) > 1:
                contrarian_trend = np.polyfit(range(len(recent_effectiveness)), recent_effectiveness, 1)[0]
        
        return {
            'contrarian_effectiveness': contrarian_effectiveness,
            'counter_trend_success': counter_trend_success,
            'avg_frequency': avg_frequency,
            'contrarian_diversity': contrarian_diversity,
            'contrarian_trend': contrarian_trend,
            'total_contrarian_events': total_contrarian_events,
            'recent_contrarian_events': recent_contrarian,
            'contrarian_frequency': contrarian_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze contrarian strategy health."""
        effectiveness = metrics['contrarian_effectiveness']
        counter_trend_success = metrics['counter_trend_success']
        diversity = metrics['contrarian_diversity']
        
        if effectiveness < 0.3 or counter_trend_success < 0.4 or diversity < 0.1:
            return SubsystemHealth.CRITICAL
        elif effectiveness < 0.5 or counter_trend_success < 0.6 or diversity < 0.3:
            return SubsystemHealth.WARNING
        elif effectiveness < 0.7:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on effectiveness and counter-trend success."""
        effectiveness = metrics['contrarian_effectiveness']
        counter_trend_success = metrics['counter_trend_success']
        
        return (effectiveness + counter_trend_success) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on diversity and trend."""
        diversity = metrics['contrarian_diversity']
        trend = metrics['contrarian_trend']
        
        # Normalize trend (positive is good)
        trend_score = max(0, min(1.0, trend * 10))
        
        return (diversity + trend_score) / 2

class GoalInventionMonitor(BaseCognitiveSubsystem):
    """Monitors goal invention and creativity."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="goal_invention",
            name="Goal Invention Monitor",
            description="Tracks goal invention, creativity, and novel objective generation"
        )
        self.goal_invention_events = []
        self.goal_creativity_scores = []
        self.goal_achievement_rates = []
        self.novel_goal_frequency = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize goal invention monitoring."""
        self.goal_invention_events = []
        self.goal_creativity_scores = []
        self.goal_achievement_rates = []
        self.novel_goal_frequency = []
        logger.info("Goal Invention Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect goal invention metrics."""
        current_time = datetime.now()
        
        # Calculate goal creativity score
        creativity_score = np.mean(self.goal_creativity_scores) if self.goal_creativity_scores else 0.0
        
        # Calculate goal achievement rate
        achievement_rate = np.mean(self.goal_achievement_rates) if self.goal_achievement_rates else 0.0
        
        # Calculate novel goal frequency
        novel_frequency = np.mean(self.novel_goal_frequency) if self.novel_goal_frequency else 0.0
        
        # Count recent goal inventions
        recent_inventions = len([
            event for event in self.goal_invention_events
            if (current_time - event.get('timestamp', current_time)).seconds < 3600
        ])
        
        # Calculate goal diversity
        goal_types = set([event.get('goal_type', 'unknown') for event in self.goal_invention_events])
        goal_diversity = len(goal_types) / max(len(self.goal_invention_events), 1)
        
        # Calculate total goal inventions
        total_inventions = len(self.goal_invention_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        invention_frequency = total_inventions / hours_elapsed
        
        # Calculate creativity trend
        creativity_trend = 0.0
        if len(self.goal_creativity_scores) > 1:
            recent_creativity = self.goal_creativity_scores[-10:]
            if len(recent_creativity) > 1:
                creativity_trend = np.polyfit(range(len(recent_creativity)), recent_creativity, 1)[0]
        
        return {
            'creativity_score': creativity_score,
            'achievement_rate': achievement_rate,
            'novel_frequency': novel_frequency,
            'goal_diversity': goal_diversity,
            'creativity_trend': creativity_trend,
            'total_inventions': total_inventions,
            'recent_inventions': recent_inventions,
            'invention_frequency': invention_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze goal invention health."""
        creativity_score = metrics['creativity_score']
        achievement_rate = metrics['achievement_rate']
        novel_frequency = metrics['novel_frequency']
        
        if creativity_score < 0.3 or achievement_rate < 0.4 or novel_frequency < 0.1:
            return SubsystemHealth.CRITICAL
        elif creativity_score < 0.5 or achievement_rate < 0.6 or novel_frequency < 0.3:
            return SubsystemHealth.WARNING
        elif creativity_score < 0.7:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on creativity and achievement rate."""
        creativity_score = metrics['creativity_score']
        achievement_rate = metrics['achievement_rate']
        
        return (creativity_score + achievement_rate) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on diversity and trend."""
        goal_diversity = metrics['goal_diversity']
        creativity_trend = metrics['creativity_trend']
        
        # Normalize creativity trend (positive is good)
        trend_score = max(0, min(1.0, creativity_trend * 10))
        
        return (goal_diversity + trend_score) / 2
