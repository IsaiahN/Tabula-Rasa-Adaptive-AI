"""
System-Related Cognitive Subsystems

Implements 8 system-focused subsystems for comprehensive system monitoring and management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np

from .base_subsystem import BaseCognitiveSubsystem, SubsystemHealth

logger = logging.getLogger(__name__)

class ResourceUtilizationMonitor(BaseCognitiveSubsystem):
    """Monitors system resource utilization and efficiency."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="resource_utilization",
            name="Resource Utilization Monitor",
            description="Tracks CPU, memory, and system resource utilization"
        )
        self.resource_events = []
        self.cpu_utilization = []
        self.memory_utilization = []
        self.resource_efficiency = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize resource utilization monitoring."""
        self.resource_events = []
        self.cpu_utilization = []
        self.memory_utilization = []
        self.resource_efficiency = []
        logger.info("Resource Utilization Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect resource utilization metrics."""
        current_time = datetime.now()
        
        # Calculate CPU utilization
        cpu_util = np.mean(self.cpu_utilization) if self.cpu_utilization else 0.0
        
        # Calculate memory utilization
        memory_util = np.mean(self.memory_utilization) if self.memory_utilization else 0.0
        
        # Calculate resource efficiency
        efficiency = np.mean(self.resource_efficiency) if self.resource_efficiency else 1.0
        
        # Count recent resource events
        recent_events = len([
            event for event in self.resource_events
            if (current_time - event.get('timestamp', current_time)).seconds < 300
        ])
        
        # Calculate resource balance
        resource_balance = 0.0
        if cpu_util > 0 and memory_util > 0:
            # Balance is about having similar utilization across resources
            balance_diff = abs(cpu_util - memory_util)
            resource_balance = max(0, 1 - balance_diff / 100)  # Normalize to 0-1
        
        # Calculate utilization trend
        utilization_trend = 0.0
        if len(self.cpu_utilization) > 1 and len(self.memory_utilization) > 1:
            recent_cpu = self.cpu_utilization[-10:]
            recent_memory = self.memory_utilization[-10:]
            if len(recent_cpu) > 1 and len(recent_memory) > 1:
                cpu_trend = np.polyfit(range(len(recent_cpu)), recent_cpu, 1)[0]
                memory_trend = np.polyfit(range(len(recent_memory)), recent_memory, 1)[0]
                utilization_trend = (cpu_trend + memory_trend) / 2
        
        # Calculate total resource events
        total_events = len(self.resource_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        event_frequency = total_events / hours_elapsed
        
        return {
            'cpu_utilization': cpu_util,
            'memory_utilization': memory_util,
            'resource_efficiency': efficiency,
            'resource_balance': resource_balance,
            'utilization_trend': utilization_trend,
            'total_events': total_events,
            'recent_events': recent_events,
            'event_frequency': event_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze resource utilization health."""
        cpu_util = metrics['cpu_utilization']
        memory_util = metrics['memory_utilization']
        efficiency = metrics['resource_efficiency']
        
        if cpu_util > 90 or memory_util > 90 or efficiency < 0.3:
            return SubsystemHealth.CRITICAL
        elif cpu_util > 80 or memory_util > 80 or efficiency < 0.5:
            return SubsystemHealth.WARNING
        elif cpu_util > 70:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on efficiency and balance."""
        efficiency = metrics['resource_efficiency']
        balance = metrics['resource_balance']
        
        return (efficiency + balance) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on utilization and trend."""
        cpu_util = metrics['cpu_utilization']
        memory_util = metrics['memory_utilization']
        trend = metrics['utilization_trend']
        
        # Optimal utilization is around 70-80%
        cpu_score = max(0, 1 - abs(cpu_util - 75) / 75)
        memory_score = max(0, 1 - abs(memory_util - 75) / 75)
        
        # Normalize trend (stable is good)
        trend_score = max(0, 1 - abs(trend) / 10)
        
        return (cpu_score + memory_score + trend_score) / 3

class GradientFlowMonitor(BaseCognitiveSubsystem):
    """Monitors gradient flow and learning dynamics."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="gradient_flow",
            name="Gradient Flow Monitor",
            description="Tracks gradient flow, learning dynamics, and optimization"
        )
        self.gradient_events = []
        self.gradient_magnitudes = []
        self.gradient_directions = []
        self.learning_rates = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize gradient flow monitoring."""
        self.gradient_events = []
        self.gradient_magnitudes = []
        self.gradient_directions = []
        self.learning_rates = []
        logger.info("Gradient Flow Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect gradient flow metrics."""
        current_time = datetime.now()
        
        # Calculate average gradient magnitude
        avg_magnitude = np.mean(self.gradient_magnitudes) if self.gradient_magnitudes else 0.0
        
        # Calculate gradient direction consistency
        direction_consistency = 0.0
        if len(self.gradient_directions) > 1:
            # Calculate consistency of gradient directions
            direction_std = np.std(self.gradient_directions)
            direction_consistency = max(0, 1 - direction_std / 180)  # Normalize to 0-1
        
        # Calculate average learning rate
        avg_learning_rate = np.mean(self.learning_rates) if self.learning_rates else 0.0
        
        # Count recent gradient events
        recent_events = len([
            event for event in self.gradient_events
            if (current_time - event.get('timestamp', current_time)).seconds < 300
        ])
        
        # Calculate gradient stability
        gradient_stability = 0.0
        if len(self.gradient_magnitudes) > 1:
            magnitude_std = np.std(self.gradient_magnitudes)
            gradient_stability = max(0, 1 - magnitude_std / 10)  # Normalize to 0-1
        
        # Calculate learning rate trend
        learning_rate_trend = 0.0
        if len(self.learning_rates) > 1:
            recent_rates = self.learning_rates[-10:]
            if len(recent_rates) > 1:
                learning_rate_trend = np.polyfit(range(len(recent_rates)), recent_rates, 1)[0]
        
        # Calculate total gradient events
        total_events = len(self.gradient_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        event_frequency = total_events / hours_elapsed
        
        return {
            'avg_magnitude': avg_magnitude,
            'direction_consistency': direction_consistency,
            'avg_learning_rate': avg_learning_rate,
            'gradient_stability': gradient_stability,
            'learning_rate_trend': learning_rate_trend,
            'total_events': total_events,
            'recent_events': recent_events,
            'event_frequency': event_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze gradient flow health."""
        magnitude = metrics['avg_magnitude']
        consistency = metrics['direction_consistency']
        stability = metrics['gradient_stability']
        
        if magnitude < 0.01 or consistency < 0.3 or stability < 0.2:
            return SubsystemHealth.CRITICAL
        elif magnitude < 0.05 or consistency < 0.5 or stability < 0.4:
            return SubsystemHealth.WARNING
        elif magnitude < 0.1:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on magnitude and consistency."""
        magnitude = metrics['avg_magnitude']
        consistency = metrics['direction_consistency']
        
        # Normalize magnitude (moderate is good)
        magnitude_score = max(0, 1 - abs(magnitude - 0.1) / 0.1)
        
        return (magnitude_score + consistency) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on stability and trend."""
        stability = metrics['gradient_stability']
        trend = metrics['learning_rate_trend']
        
        # Normalize trend (stable is good)
        trend_score = max(0, 1 - abs(trend) / 0.01)
        
        return (stability + trend_score) / 2

class UsageTrackingMonitor(BaseCognitiveSubsystem):
    """Monitors system usage patterns and optimization opportunities."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="usage_tracking",
            name="Usage Tracking Monitor",
            description="Tracks system usage patterns and optimization opportunities"
        )
        self.usage_events = []
        self.usage_patterns = []
        self.optimization_opportunities = []
        self.usage_efficiency = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize usage tracking monitoring."""
        self.usage_events = []
        self.usage_patterns = []
        self.optimization_opportunities = []
        self.usage_efficiency = []
        logger.info("Usage Tracking Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect usage tracking metrics."""
        current_time = datetime.now()
        
        # Calculate usage efficiency
        usage_efficiency = np.mean(self.usage_efficiency) if self.usage_efficiency else 0.0
        
        # Calculate optimization opportunities
        optimization_opportunities = np.mean(self.optimization_opportunities) if self.optimization_opportunities else 0.0
        
        # Count recent usage events
        recent_events = len([
            event for event in self.usage_events
            if (current_time - event.get('timestamp', current_time)).seconds < 300
        ])
        
        # Calculate usage pattern diversity
        pattern_diversity = len(set(self.usage_patterns)) / max(len(self.usage_patterns), 1)
        
        # Calculate usage frequency
        total_events = len(self.usage_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        event_frequency = total_events / hours_elapsed
        
        # Calculate usage trend
        usage_trend = 0.0
        if len(self.usage_efficiency) > 1:
            recent_efficiency = self.usage_efficiency[-10:]
            if len(recent_efficiency) > 1:
                usage_trend = np.polyfit(range(len(recent_efficiency)), recent_efficiency, 1)[0]
        
        # Calculate optimization potential
        optimization_potential = 0.0
        if optimization_opportunities > 0:
            optimization_potential = min(1.0, optimization_opportunities / 10)  # Normalize to 0-1
        
        return {
            'usage_efficiency': usage_efficiency,
            'optimization_opportunities': optimization_opportunities,
            'optimization_potential': optimization_potential,
            'pattern_diversity': pattern_diversity,
            'usage_trend': usage_trend,
            'total_events': total_events,
            'recent_events': recent_events,
            'event_frequency': event_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze usage tracking health."""
        efficiency = metrics['usage_efficiency']
        optimization_potential = metrics['optimization_potential']
        pattern_diversity = metrics['pattern_diversity']
        
        if efficiency < 0.3 or optimization_potential < 0.2 or pattern_diversity < 0.1:
            return SubsystemHealth.CRITICAL
        elif efficiency < 0.5 or optimization_potential < 0.4 or pattern_diversity < 0.3:
            return SubsystemHealth.WARNING
        elif efficiency < 0.7:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on efficiency and optimization potential."""
        efficiency = metrics['usage_efficiency']
        optimization_potential = metrics['optimization_potential']
        
        return (efficiency + optimization_potential) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on pattern diversity and trend."""
        pattern_diversity = metrics['pattern_diversity']
        trend = metrics['usage_trend']
        
        # Normalize trend (positive is good)
        trend_score = max(0, min(1.0, trend * 10))
        
        return (pattern_diversity + trend_score) / 2

class AntiBiasWeightingMonitor(BaseCognitiveSubsystem):
    """Monitors anti-bias weighting and fairness mechanisms."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="anti_bias_weighting",
            name="Anti-Bias Weighting Monitor",
            description="Tracks anti-bias weighting and fairness mechanisms"
        )
        self.bias_events = []
        self.bias_detection_accuracy = []
        self.fairness_scores = []
        self.bias_correction_effectiveness = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize anti-bias weighting monitoring."""
        self.bias_events = []
        self.bias_detection_accuracy = []
        self.fairness_scores = []
        self.bias_correction_effectiveness = []
        logger.info("Anti-Bias Weighting Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect anti-bias weighting metrics."""
        current_time = datetime.now()
        
        # Calculate bias detection accuracy
        detection_accuracy = np.mean(self.bias_detection_accuracy) if self.bias_detection_accuracy else 0.0
        
        # Calculate fairness scores
        fairness_score = np.mean(self.fairness_scores) if self.fairness_scores else 0.0
        
        # Calculate bias correction effectiveness
        correction_effectiveness = np.mean(self.bias_correction_effectiveness) if self.bias_correction_effectiveness else 0.0
        
        # Count recent bias events
        recent_events = len([
            event for event in self.bias_events
            if (current_time - event.get('timestamp', current_time)).seconds < 3600
        ])
        
        # Calculate bias types
        bias_types = set([event.get('bias_type', 'unknown') for event in self.bias_events])
        bias_diversity = len(bias_types) / max(len(self.bias_events), 1)
        
        # Calculate bias frequency
        total_events = len(self.bias_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        event_frequency = total_events / hours_elapsed
        
        # Calculate fairness trend
        fairness_trend = 0.0
        if len(self.fairness_scores) > 1:
            recent_fairness = self.fairness_scores[-10:]
            if len(recent_fairness) > 1:
                fairness_trend = np.polyfit(range(len(recent_fairness)), recent_fairness, 1)[0]
        
        # Calculate overall bias mitigation
        bias_mitigation = (detection_accuracy + fairness_score + correction_effectiveness) / 3
        
        return {
            'detection_accuracy': detection_accuracy,
            'fairness_score': fairness_score,
            'correction_effectiveness': correction_effectiveness,
            'bias_mitigation': bias_mitigation,
            'bias_diversity': bias_diversity,
            'fairness_trend': fairness_trend,
            'total_events': total_events,
            'recent_events': recent_events,
            'event_frequency': event_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze anti-bias weighting health."""
        detection_accuracy = metrics['detection_accuracy']
        fairness_score = metrics['fairness_score']
        bias_mitigation = metrics['bias_mitigation']
        
        if detection_accuracy < 0.5 or fairness_score < 0.4 or bias_mitigation < 0.3:
            return SubsystemHealth.CRITICAL
        elif detection_accuracy < 0.7 or fairness_score < 0.6 or bias_mitigation < 0.5:
            return SubsystemHealth.WARNING
        elif detection_accuracy < 0.8:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on detection accuracy and fairness."""
        detection_accuracy = metrics['detection_accuracy']
        fairness_score = metrics['fairness_score']
        
        return (detection_accuracy + fairness_score) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on bias mitigation and trend."""
        bias_mitigation = metrics['bias_mitigation']
        trend = metrics['fairness_trend']
        
        # Normalize trend (positive is good)
        trend_score = max(0, min(1.0, trend * 10))
        
        return (bias_mitigation + trend_score) / 2

class ClusterFormationMonitor(BaseCognitiveSubsystem):
    """Monitors cluster formation and organization."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="cluster_formation",
            name="Cluster Formation Monitor",
            description="Tracks cluster formation, organization, and effectiveness"
        )
        self.cluster_events = []
        self.cluster_quality_scores = []
        self.cluster_stability = []
        self.cluster_effectiveness = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize cluster formation monitoring."""
        self.cluster_events = []
        self.cluster_quality_scores = []
        self.cluster_stability = []
        self.cluster_effectiveness = []
        logger.info("Cluster Formation Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect cluster formation metrics."""
        current_time = datetime.now()
        
        # Calculate cluster quality
        cluster_quality = np.mean(self.cluster_quality_scores) if self.cluster_quality_scores else 0.0
        
        # Calculate cluster stability
        cluster_stability = np.mean(self.cluster_stability) if self.cluster_stability else 0.0
        
        # Calculate cluster effectiveness
        cluster_effectiveness = np.mean(self.cluster_effectiveness) if self.cluster_effectiveness else 0.0
        
        # Count recent cluster events
        recent_events = len([
            event for event in self.cluster_events
            if (current_time - event.get('timestamp', current_time)).seconds < 1800
        ])
        
        # Calculate cluster types
        cluster_types = set([event.get('cluster_type', 'unknown') for event in self.cluster_events])
        cluster_diversity = len(cluster_types) / max(len(self.cluster_events), 1)
        
        # Calculate cluster frequency
        total_events = len(self.cluster_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        event_frequency = total_events / hours_elapsed
        
        # Calculate cluster formation trend
        formation_trend = 0.0
        if len(self.cluster_quality_scores) > 1:
            recent_quality = self.cluster_quality_scores[-10:]
            if len(recent_quality) > 1:
                formation_trend = np.polyfit(range(len(recent_quality)), recent_quality, 1)[0]
        
        # Calculate overall cluster performance
        cluster_performance = (cluster_quality + cluster_stability + cluster_effectiveness) / 3
        
        return {
            'cluster_quality': cluster_quality,
            'cluster_stability': cluster_stability,
            'cluster_effectiveness': cluster_effectiveness,
            'cluster_performance': cluster_performance,
            'cluster_diversity': cluster_diversity,
            'formation_trend': formation_trend,
            'total_events': total_events,
            'recent_events': recent_events,
            'event_frequency': event_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze cluster formation health."""
        cluster_quality = metrics['cluster_quality']
        cluster_stability = metrics['cluster_stability']
        cluster_performance = metrics['cluster_performance']
        
        if cluster_quality < 0.4 or cluster_stability < 0.3 or cluster_performance < 0.3:
            return SubsystemHealth.CRITICAL
        elif cluster_quality < 0.6 or cluster_stability < 0.5 or cluster_performance < 0.5:
            return SubsystemHealth.WARNING
        elif cluster_quality < 0.8:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on cluster performance."""
        return metrics['cluster_performance']
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on diversity and trend."""
        diversity = metrics['cluster_diversity']
        trend = metrics['formation_trend']
        
        # Normalize trend (positive is good)
        trend_score = max(0, min(1.0, trend * 10))
        
        return (diversity + trend_score) / 2

class DangerZoneAvoidanceMonitor(BaseCognitiveSubsystem):
    """Monitors danger zone avoidance and safety mechanisms."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="danger_zone_avoidance",
            name="Danger Zone Avoidance Monitor",
            description="Tracks danger zone avoidance and safety mechanisms"
        )
        self.danger_events = []
        self.avoidance_success_rates = []
        self.danger_detection_accuracy = []
        self.safety_scores = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize danger zone avoidance monitoring."""
        self.danger_events = []
        self.avoidance_success_rates = []
        self.danger_detection_accuracy = []
        self.safety_scores = []
        logger.info("Danger Zone Avoidance Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect danger zone avoidance metrics."""
        current_time = datetime.now()
        
        # Calculate avoidance success rate
        avoidance_success = np.mean(self.avoidance_success_rates) if self.avoidance_success_rates else 0.0
        
        # Calculate danger detection accuracy
        detection_accuracy = np.mean(self.danger_detection_accuracy) if self.danger_detection_accuracy else 0.0
        
        # Calculate safety scores
        safety_score = np.mean(self.safety_scores) if self.safety_scores else 0.0
        
        # Count recent danger events
        recent_events = len([
            event for event in self.danger_events
            if (current_time - event.get('timestamp', current_time)).seconds < 3600
        ])
        
        # Calculate danger types
        danger_types = set([event.get('danger_type', 'unknown') for event in self.danger_events])
        danger_diversity = len(danger_types) / max(len(self.danger_events), 1)
        
        # Calculate danger frequency
        total_events = len(self.danger_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        event_frequency = total_events / hours_elapsed
        
        # Calculate safety trend
        safety_trend = 0.0
        if len(self.safety_scores) > 1:
            recent_safety = self.safety_scores[-10:]
            if len(recent_safety) > 1:
                safety_trend = np.polyfit(range(len(recent_safety)), recent_safety, 1)[0]
        
        # Calculate overall safety performance
        safety_performance = (avoidance_success + detection_accuracy + safety_score) / 3
        
        return {
            'avoidance_success': avoidance_success,
            'detection_accuracy': detection_accuracy,
            'safety_score': safety_score,
            'safety_performance': safety_performance,
            'danger_diversity': danger_diversity,
            'safety_trend': safety_trend,
            'total_events': total_events,
            'recent_events': recent_events,
            'event_frequency': event_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze danger zone avoidance health."""
        avoidance_success = metrics['avoidance_success']
        detection_accuracy = metrics['detection_accuracy']
        safety_performance = metrics['safety_performance']
        
        if avoidance_success < 0.5 or detection_accuracy < 0.6 or safety_performance < 0.4:
            return SubsystemHealth.CRITICAL
        elif avoidance_success < 0.7 or detection_accuracy < 0.8 or safety_performance < 0.6:
            return SubsystemHealth.WARNING
        elif avoidance_success < 0.8:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on safety performance."""
        return metrics['safety_performance']
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on diversity and trend."""
        diversity = metrics['danger_diversity']
        trend = metrics['safety_trend']
        
        # Normalize trend (positive is good)
        trend_score = max(0, min(1.0, trend * 10))
        
        return (diversity + trend_score) / 2

class SwarmIntelligenceMonitor(BaseCognitiveSubsystem):
    """Monitors swarm intelligence and collective behavior."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="swarm_intelligence",
            name="Swarm Intelligence Monitor",
            description="Tracks swarm intelligence and collective behavior patterns"
        )
        self.swarm_events = []
        self.collective_effectiveness = []
        self.swarm_coordination = []
        self.emergent_behavior_scores = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize swarm intelligence monitoring."""
        self.swarm_events = []
        self.collective_effectiveness = []
        self.swarm_coordination = []
        self.emergent_behavior_scores = []
        logger.info("Swarm Intelligence Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect swarm intelligence metrics."""
        current_time = datetime.now()
        
        # Calculate collective effectiveness
        collective_effectiveness = np.mean(self.collective_effectiveness) if self.collective_effectiveness else 0.0
        
        # Calculate swarm coordination
        swarm_coordination = np.mean(self.swarm_coordination) if self.swarm_coordination else 0.0
        
        # Calculate emergent behavior scores
        emergent_behavior = np.mean(self.emergent_behavior_scores) if self.emergent_behavior_scores else 0.0
        
        # Count recent swarm events
        recent_events = len([
            event for event in self.swarm_events
            if (current_time - event.get('timestamp', current_time)).seconds < 1800
        ])
        
        # Calculate swarm types
        swarm_types = set([event.get('swarm_type', 'unknown') for event in self.swarm_events])
        swarm_diversity = len(swarm_types) / max(len(self.swarm_events), 1)
        
        # Calculate swarm frequency
        total_events = len(self.swarm_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        event_frequency = total_events / hours_elapsed
        
        # Calculate swarm intelligence trend
        intelligence_trend = 0.0
        if len(self.collective_effectiveness) > 1:
            recent_effectiveness = self.collective_effectiveness[-10:]
            if len(recent_effectiveness) > 1:
                intelligence_trend = np.polyfit(range(len(recent_effectiveness)), recent_effectiveness, 1)[0]
        
        # Calculate overall swarm performance
        swarm_performance = (collective_effectiveness + swarm_coordination + emergent_behavior) / 3
        
        return {
            'collective_effectiveness': collective_effectiveness,
            'swarm_coordination': swarm_coordination,
            'emergent_behavior': emergent_behavior,
            'swarm_performance': swarm_performance,
            'swarm_diversity': swarm_diversity,
            'intelligence_trend': intelligence_trend,
            'total_events': total_events,
            'recent_events': recent_events,
            'event_frequency': event_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze swarm intelligence health."""
        collective_effectiveness = metrics['collective_effectiveness']
        swarm_coordination = metrics['swarm_coordination']
        swarm_performance = metrics['swarm_performance']
        
        if collective_effectiveness < 0.3 or swarm_coordination < 0.4 or swarm_performance < 0.3:
            return SubsystemHealth.CRITICAL
        elif collective_effectiveness < 0.5 or swarm_coordination < 0.6 or swarm_performance < 0.5:
            return SubsystemHealth.WARNING
        elif collective_effectiveness < 0.7:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on swarm performance."""
        return metrics['swarm_performance']
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on diversity and trend."""
        diversity = metrics['swarm_diversity']
        trend = metrics['intelligence_trend']
        
        # Normalize trend (positive is good)
        trend_score = max(0, min(1.0, trend * 10))
        
        return (diversity + trend_score) / 2

class HebbianBonusesMonitor(BaseCognitiveSubsystem):
    """Monitors Hebbian bonuses and learning reinforcement."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="hebbian_bonuses",
            name="Hebbian Bonuses Monitor",
            description="Tracks Hebbian bonuses and learning reinforcement mechanisms"
        )
        self.hebbian_events = []
        self.bonus_effectiveness = []
        self.learning_reinforcement = []
        self.hebbian_strength = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize Hebbian bonuses monitoring."""
        self.hebbian_events = []
        self.bonus_effectiveness = []
        self.learning_reinforcement = []
        self.hebbian_strength = []
        logger.info("Hebbian Bonuses Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect Hebbian bonuses metrics."""
        current_time = datetime.now()
        
        # Calculate bonus effectiveness
        bonus_effectiveness = np.mean(self.bonus_effectiveness) if self.bonus_effectiveness else 0.0
        
        # Calculate learning reinforcement
        learning_reinforcement = np.mean(self.learning_reinforcement) if self.learning_reinforcement else 0.0
        
        # Calculate Hebbian strength
        hebbian_strength = np.mean(self.hebbian_strength) if self.hebbian_strength else 0.0
        
        # Count recent Hebbian events
        recent_events = len([
            event for event in self.hebbian_events
            if (current_time - event.get('timestamp', current_time)).seconds < 300
        ])
        
        # Calculate Hebbian types
        hebbian_types = set([event.get('hebbian_type', 'unknown') for event in self.hebbian_events])
        hebbian_diversity = len(hebbian_types) / max(len(self.hebbian_events), 1)
        
        # Calculate Hebbian frequency
        total_events = len(self.hebbian_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        event_frequency = total_events / hours_elapsed
        
        # Calculate Hebbian trend
        hebbian_trend = 0.0
        if len(self.hebbian_strength) > 1:
            recent_strength = self.hebbian_strength[-10:]
            if len(recent_strength) > 1:
                hebbian_trend = np.polyfit(range(len(recent_strength)), recent_strength, 1)[0]
        
        # Calculate overall Hebbian performance
        hebbian_performance = (bonus_effectiveness + learning_reinforcement + hebbian_strength) / 3
        
        return {
            'bonus_effectiveness': bonus_effectiveness,
            'learning_reinforcement': learning_reinforcement,
            'hebbian_strength': hebbian_strength,
            'hebbian_performance': hebbian_performance,
            'hebbian_diversity': hebbian_diversity,
            'hebbian_trend': hebbian_trend,
            'total_events': total_events,
            'recent_events': recent_events,
            'event_frequency': event_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze Hebbian bonuses health."""
        bonus_effectiveness = metrics['bonus_effectiveness']
        learning_reinforcement = metrics['learning_reinforcement']
        hebbian_performance = metrics['hebbian_performance']
        
        if bonus_effectiveness < 0.3 or learning_reinforcement < 0.4 or hebbian_performance < 0.3:
            return SubsystemHealth.CRITICAL
        elif bonus_effectiveness < 0.5 or learning_reinforcement < 0.6 or hebbian_performance < 0.5:
            return SubsystemHealth.WARNING
        elif bonus_effectiveness < 0.7:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on Hebbian performance."""
        return metrics['hebbian_performance']
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on diversity and trend."""
        diversity = metrics['hebbian_diversity']
        trend = metrics['hebbian_trend']
        
        # Normalize trend (positive is good)
        trend_score = max(0, min(1.0, trend * 10))
        
        return (diversity + trend_score) / 2
