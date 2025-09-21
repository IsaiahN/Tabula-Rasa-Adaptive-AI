"""
Action-Related Cognitive Subsystems

Implements 5 action-focused subsystems for comprehensive action monitoring and management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np

from .base_subsystem import BaseCognitiveSubsystem, SubsystemHealth

logger = logging.getLogger(__name__)

class ActionIntelligenceMonitor(BaseCognitiveSubsystem):
    """Monitors action intelligence and semantic understanding."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="action_intelligence",
            name="Action Intelligence Monitor",
            description="Tracks action intelligence, semantic understanding, and action effectiveness"
        )
        self.action_events = []
        self.semantic_accuracy = []
        self.action_effectiveness = []
        self.coordinate_success_rates = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize action intelligence monitoring."""
        self.action_events = []
        self.semantic_accuracy = []
        self.action_effectiveness = []
        self.coordinate_success_rates = []
        logger.info("Action Intelligence Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect action intelligence metrics."""
        current_time = datetime.now()
        
        # Calculate semantic accuracy
        semantic_accuracy = np.mean(self.semantic_accuracy) if self.semantic_accuracy else 0.0
        
        # Calculate action effectiveness
        action_effectiveness = np.mean(self.action_effectiveness) if self.action_effectiveness else 0.0
        
        # Calculate coordinate success rate
        coordinate_success = np.mean(self.coordinate_success_rates) if self.coordinate_success_rates else 0.0
        
        # Count recent actions
        recent_actions = len([
            event for event in self.action_events
            if (current_time - event.get('timestamp', current_time)).seconds < 300
        ])
        
        # Calculate action diversity
        action_types = set([event.get('action_type', 'unknown') for event in self.action_events])
        action_diversity = len(action_types) / max(len(self.action_events), 1)
        
        # Calculate action frequency
        total_actions = len(self.action_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        action_frequency = total_actions / hours_elapsed
        
        # Calculate learning progress
        learning_progress = 0.0
        if len(self.semantic_accuracy) > 1:
            recent_accuracy = self.semantic_accuracy[-10:]
            if len(recent_accuracy) > 1:
                learning_progress = np.polyfit(range(len(recent_accuracy)), recent_accuracy, 1)[0]
        
        return {
            'semantic_accuracy': semantic_accuracy,
            'action_effectiveness': action_effectiveness,
            'coordinate_success_rate': coordinate_success,
            'action_diversity': action_diversity,
            'learning_progress': learning_progress,
            'total_actions': total_actions,
            'recent_actions': recent_actions,
            'action_frequency': action_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze action intelligence health."""
        semantic_accuracy = metrics['semantic_accuracy']
        action_effectiveness = metrics['action_effectiveness']
        coordinate_success = metrics['coordinate_success_rate']
        
        if semantic_accuracy < 0.5 or action_effectiveness < 0.4 or coordinate_success < 0.3:
            return SubsystemHealth.CRITICAL
        elif semantic_accuracy < 0.7 or action_effectiveness < 0.6 or coordinate_success < 0.5:
            return SubsystemHealth.WARNING
        elif semantic_accuracy < 0.8:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on accuracy and effectiveness."""
        semantic_accuracy = metrics['semantic_accuracy']
        action_effectiveness = metrics['action_effectiveness']
        
        return (semantic_accuracy + action_effectiveness) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on diversity and learning progress."""
        action_diversity = metrics['action_diversity']
        learning_progress = metrics['learning_progress']
        
        # Normalize learning progress (positive is good)
        progress_score = max(0, min(1.0, learning_progress * 10))
        
        return (action_diversity + progress_score) / 2

class ActionExperimentationMonitor(BaseCognitiveSubsystem):
    """Monitors action experimentation and discovery."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="action_experimentation",
            name="Action Experimentation Monitor",
            description="Tracks action experimentation, discovery, and innovation"
        )
        self.experimentation_events = []
        self.discovery_rates = []
        self.innovation_scores = []
        self.experiment_success_rates = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize action experimentation monitoring."""
        self.experimentation_events = []
        self.discovery_rates = []
        self.innovation_scores = []
        self.experiment_success_rates = []
        logger.info("Action Experimentation Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect action experimentation metrics."""
        current_time = datetime.now()
        
        # Calculate discovery rate
        discovery_rate = np.mean(self.discovery_rates) if self.discovery_rates else 0.0
        
        # Calculate innovation score
        innovation_score = np.mean(self.innovation_scores) if self.innovation_scores else 0.0
        
        # Calculate experiment success rate
        experiment_success = np.mean(self.experiment_success_rates) if self.experiment_success_rates else 0.0
        
        # Count recent experiments
        recent_experiments = len([
            event for event in self.experimentation_events
            if (current_time - event.get('timestamp', current_time)).seconds < 1800
        ])
        
        # Calculate experiment diversity
        experiment_types = set([event.get('experiment_type', 'unknown') for event in self.experimentation_events])
        experiment_diversity = len(experiment_types) / max(len(self.experimentation_events), 1)
        
        # Calculate experimentation frequency
        total_experiments = len(self.experimentation_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        experimentation_frequency = total_experiments / hours_elapsed
        
        # Calculate innovation trend
        innovation_trend = 0.0
        if len(self.innovation_scores) > 1:
            recent_innovation = self.innovation_scores[-10:]
            if len(recent_innovation) > 1:
                innovation_trend = np.polyfit(range(len(recent_innovation)), recent_innovation, 1)[0]
        
        return {
            'discovery_rate': discovery_rate,
            'innovation_score': innovation_score,
            'experiment_success_rate': experiment_success,
            'experiment_diversity': experiment_diversity,
            'innovation_trend': innovation_trend,
            'total_experiments': total_experiments,
            'recent_experiments': recent_experiments,
            'experimentation_frequency': experimentation_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze action experimentation health."""
        discovery_rate = metrics['discovery_rate']
        innovation_score = metrics['innovation_score']
        experiment_success = metrics['experiment_success_rate']
        
        if discovery_rate < 0.1 or innovation_score < 0.2 or experiment_success < 0.3:
            return SubsystemHealth.CRITICAL
        elif discovery_rate < 0.3 or innovation_score < 0.4 or experiment_success < 0.5:
            return SubsystemHealth.WARNING
        elif discovery_rate < 0.5:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on discovery and innovation."""
        discovery_rate = metrics['discovery_rate']
        innovation_score = metrics['innovation_score']
        
        return (discovery_rate + innovation_score) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on diversity and success rate."""
        experiment_diversity = metrics['experiment_diversity']
        experiment_success = metrics['experiment_success_rate']
        
        return (experiment_diversity + experiment_success) / 2

class EmergencyMovementMonitor(BaseCognitiveSubsystem):
    """Monitors emergency movement and crisis response."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="emergency_movement",
            name="Emergency Movement Monitor",
            description="Tracks emergency movement responses and crisis management"
        )
        self.emergency_events = []
        self.response_times = []
        self.crisis_resolution_rates = []
        self.emergency_effectiveness = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize emergency movement monitoring."""
        self.emergency_events = []
        self.response_times = []
        self.crisis_resolution_rates = []
        self.emergency_effectiveness = []
        logger.info("Emergency Movement Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect emergency movement metrics."""
        current_time = datetime.now()
        
        # Calculate average response time
        avg_response_time = np.mean(self.response_times) if self.response_times else 0.0
        
        # Calculate crisis resolution rate
        resolution_rate = np.mean(self.crisis_resolution_rates) if self.crisis_resolution_rates else 0.0
        
        # Calculate emergency effectiveness
        emergency_effectiveness = np.mean(self.emergency_effectiveness) if self.emergency_effectiveness else 0.0
        
        # Count recent emergencies
        recent_emergencies = len([
            event for event in self.emergency_events
            if (current_time - event.get('timestamp', current_time)).seconds < 3600
        ])
        
        # Calculate emergency types
        emergency_types = set([event.get('emergency_type', 'unknown') for event in self.emergency_events])
        emergency_diversity = len(emergency_types) / max(len(self.emergency_events), 1)
        
        # Calculate emergency frequency
        total_emergencies = len(self.emergency_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        emergency_frequency = total_emergencies / hours_elapsed
        
        # Calculate response time trend
        response_trend = 0.0
        if len(self.response_times) > 1:
            recent_responses = self.response_times[-10:]
            if len(recent_responses) > 1:
                response_trend = np.polyfit(range(len(recent_responses)), recent_responses, 1)[0]
        
        return {
            'avg_response_time': avg_response_time,
            'crisis_resolution_rate': resolution_rate,
            'emergency_effectiveness': emergency_effectiveness,
            'emergency_diversity': emergency_diversity,
            'response_trend': response_trend,
            'total_emergencies': total_emergencies,
            'recent_emergencies': recent_emergencies,
            'emergency_frequency': emergency_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze emergency movement health."""
        response_time = metrics['avg_response_time']
        resolution_rate = metrics['crisis_resolution_rate']
        effectiveness = metrics['emergency_effectiveness']
        
        if response_time > 5000 or resolution_rate < 0.5 or effectiveness < 0.4:
            return SubsystemHealth.CRITICAL
        elif response_time > 3000 or resolution_rate < 0.7 or effectiveness < 0.6:
            return SubsystemHealth.WARNING
        elif response_time > 2000:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on resolution rate and effectiveness."""
        resolution_rate = metrics['crisis_resolution_rate']
        effectiveness = metrics['emergency_effectiveness']
        
        return (resolution_rate + effectiveness) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on response time and trend."""
        response_time = metrics['avg_response_time']
        response_trend = metrics['response_trend']
        
        # Normalize response time (lower is better)
        time_score = max(0, 1 - (response_time / 5000))
        
        # Normalize trend (negative is better for response time)
        trend_score = max(0, 1 - abs(response_trend) / 1000)
        
        return (time_score + trend_score) / 2

class PredictiveCoordinatesMonitor(BaseCognitiveSubsystem):
    """Monitors predictive coordinate selection and optimization."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="predictive_coordinates",
            name="Predictive Coordinates Monitor",
            description="Tracks predictive coordinate selection and spatial optimization"
        )
        self.coordinate_events = []
        self.prediction_accuracy = []
        self.coordinate_effectiveness = []
        self.spatial_optimization = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize predictive coordinates monitoring."""
        self.coordinate_events = []
        self.prediction_accuracy = []
        self.coordinate_effectiveness = []
        self.spatial_optimization = []
        logger.info("Predictive Coordinates Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect predictive coordinates metrics."""
        current_time = datetime.now()
        
        # Calculate prediction accuracy
        prediction_accuracy = np.mean(self.prediction_accuracy) if self.prediction_accuracy else 0.0
        
        # Calculate coordinate effectiveness
        coordinate_effectiveness = np.mean(self.coordinate_effectiveness) if self.coordinate_effectiveness else 0.0
        
        # Calculate spatial optimization
        spatial_optimization = np.mean(self.spatial_optimization) if self.spatial_optimization else 0.0
        
        # Count recent coordinate events
        recent_coordinates = len([
            event for event in self.coordinate_events
            if (current_time - event.get('timestamp', current_time)).seconds < 300
        ])
        
        # Calculate coordinate diversity
        coordinate_regions = set([event.get('region', 'unknown') for event in self.coordinate_events])
        coordinate_diversity = len(coordinate_regions) / max(len(self.coordinate_events), 1)
        
        # Calculate coordinate frequency
        total_coordinates = len(self.coordinate_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        coordinate_frequency = total_coordinates / hours_elapsed
        
        # Calculate prediction trend
        prediction_trend = 0.0
        if len(self.prediction_accuracy) > 1:
            recent_accuracy = self.prediction_accuracy[-10:]
            if len(recent_accuracy) > 1:
                prediction_trend = np.polyfit(range(len(recent_accuracy)), recent_accuracy, 1)[0]
        
        return {
            'prediction_accuracy': prediction_accuracy,
            'coordinate_effectiveness': coordinate_effectiveness,
            'spatial_optimization': spatial_optimization,
            'coordinate_diversity': coordinate_diversity,
            'prediction_trend': prediction_trend,
            'total_coordinates': total_coordinates,
            'recent_coordinates': recent_coordinates,
            'coordinate_frequency': coordinate_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze predictive coordinates health."""
        prediction_accuracy = metrics['prediction_accuracy']
        coordinate_effectiveness = metrics['coordinate_effectiveness']
        spatial_optimization = metrics['spatial_optimization']
        
        if prediction_accuracy < 0.5 or coordinate_effectiveness < 0.4 or spatial_optimization < 0.3:
            return SubsystemHealth.CRITICAL
        elif prediction_accuracy < 0.7 or coordinate_effectiveness < 0.6 or spatial_optimization < 0.5:
            return SubsystemHealth.WARNING
        elif prediction_accuracy < 0.8:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on accuracy and effectiveness."""
        prediction_accuracy = metrics['prediction_accuracy']
        coordinate_effectiveness = metrics['coordinate_effectiveness']
        
        return (prediction_accuracy + coordinate_effectiveness) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on optimization and diversity."""
        spatial_optimization = metrics['spatial_optimization']
        coordinate_diversity = metrics['coordinate_diversity']
        
        return (spatial_optimization + coordinate_diversity) / 2

class CoordinateSuccessMonitor(BaseCognitiveSubsystem):
    """Monitors coordinate success rates and spatial reasoning."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="coordinate_success",
            name="Coordinate Success Monitor",
            description="Tracks coordinate success rates and spatial reasoning effectiveness"
        )
        self.coordinate_attempts = []
        self.success_rates = []
        self.spatial_reasoning_scores = []
        self.coordinate_learning = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize coordinate success monitoring."""
        self.coordinate_attempts = []
        self.success_rates = []
        self.spatial_reasoning_scores = []
        self.coordinate_learning = []
        logger.info("Coordinate Success Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect coordinate success metrics."""
        current_time = datetime.now()
        
        # Calculate success rate
        success_rate = np.mean(self.success_rates) if self.success_rates else 0.0
        
        # Calculate spatial reasoning score
        spatial_reasoning = np.mean(self.spatial_reasoning_scores) if self.spatial_reasoning_scores else 0.0
        
        # Calculate coordinate learning progress
        coordinate_learning = np.mean(self.coordinate_learning) if self.coordinate_learning else 0.0
        
        # Count recent attempts
        recent_attempts = len([
            attempt for attempt in self.coordinate_attempts
            if (current_time - attempt.get('timestamp', current_time)).seconds < 300
        ])
        
        # Calculate coordinate regions
        coordinate_regions = set([attempt.get('region', 'unknown') for attempt in self.coordinate_attempts])
        region_diversity = len(coordinate_regions) / max(len(self.coordinate_attempts), 1)
        
        # Calculate attempt frequency
        total_attempts = len(self.coordinate_attempts)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        attempt_frequency = total_attempts / hours_elapsed
        
        # Calculate learning trend
        learning_trend = 0.0
        if len(self.coordinate_learning) > 1:
            recent_learning = self.coordinate_learning[-10:]
            if len(recent_learning) > 1:
                learning_trend = np.polyfit(range(len(recent_learning)), recent_learning, 1)[0]
        
        return {
            'success_rate': success_rate,
            'spatial_reasoning': spatial_reasoning,
            'coordinate_learning': coordinate_learning,
            'region_diversity': region_diversity,
            'learning_trend': learning_trend,
            'total_attempts': total_attempts,
            'recent_attempts': recent_attempts,
            'attempt_frequency': attempt_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze coordinate success health."""
        success_rate = metrics['success_rate']
        spatial_reasoning = metrics['spatial_reasoning']
        coordinate_learning = metrics['coordinate_learning']
        
        if success_rate < 0.3 or spatial_reasoning < 0.4 or coordinate_learning < 0.2:
            return SubsystemHealth.CRITICAL
        elif success_rate < 0.5 or spatial_reasoning < 0.6 or coordinate_learning < 0.4:
            return SubsystemHealth.WARNING
        elif success_rate < 0.7:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on success rate and spatial reasoning."""
        success_rate = metrics['success_rate']
        spatial_reasoning = metrics['spatial_reasoning']
        
        return (success_rate + spatial_reasoning) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on learning and diversity."""
        coordinate_learning = metrics['coordinate_learning']
        region_diversity = metrics['region_diversity']
        learning_trend = metrics['learning_trend']
        
        # Normalize learning trend (positive is good)
        trend_score = max(0, min(1.0, learning_trend * 10))
        
        return (coordinate_learning + region_diversity + trend_score) / 3
