"""
Learning-Related Cognitive Subsystems

Implements 6 learning-focused subsystems for comprehensive learning monitoring and management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np

from .base_subsystem import BaseCognitiveSubsystem, SubsystemHealth

logger = logging.getLogger(__name__)

class LearningProgressMonitor(BaseCognitiveSubsystem):
    """Monitors learning progress and effectiveness."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="learning_progress",
            name="Learning Progress Monitor",
            description="Tracks learning progress, skill acquisition, and knowledge growth"
        )
        self.learning_events = []
        self.skill_levels = {}
        self.knowledge_growth = []
        self.learning_rates = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize learning progress monitoring."""
        self.learning_events = []
        self.skill_levels = {}
        self.knowledge_growth = []
        self.learning_rates = []
        logger.info("Learning Progress Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect learning progress metrics."""
        current_time = datetime.now()
        
        # Calculate average learning rate
        avg_learning_rate = np.mean(self.learning_rates) if self.learning_rates else 0.0
        
        # Calculate knowledge growth rate
        growth_rate = 0.0
        if len(self.knowledge_growth) > 1:
            recent_growth = self.knowledge_growth[-10:]
            if len(recent_growth) > 1:
                growth_rate = np.polyfit(range(len(recent_growth)), recent_growth, 1)[0]
        
        # Count active skills
        active_skills = len([skill for skill, level in self.skill_levels.items() if level > 0.1])
        
        # Count recent learning events
        recent_events = len([
            event for event in self.learning_events
            if (current_time - event.get('timestamp', current_time)).seconds < 3600
        ])
        
        # Calculate skill diversity
        skill_diversity = len(self.skill_levels) / max(active_skills, 1)
        
        return {
            'avg_learning_rate': avg_learning_rate,
            'knowledge_growth_rate': growth_rate,
            'active_skills': active_skills,
            'total_skills': len(self.skill_levels),
            'skill_diversity': skill_diversity,
            'total_learning_events': len(self.learning_events),
            'recent_learning_events': recent_events,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze learning progress health."""
        learning_rate = metrics['avg_learning_rate']
        growth_rate = metrics['knowledge_growth_rate']
        active_skills = metrics['active_skills']
        
        if learning_rate < 0.01 or growth_rate < 0.001 or active_skills < 2:
            return SubsystemHealth.CRITICAL
        elif learning_rate < 0.05 or growth_rate < 0.005 or active_skills < 5:
            return SubsystemHealth.WARNING
        elif learning_rate < 0.1:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on learning rate and skill diversity."""
        learning_rate = metrics['avg_learning_rate']
        skill_diversity = metrics['skill_diversity']
        
        # Normalize learning rate
        rate_score = min(1.0, learning_rate * 10)  # Scale to 0-1
        
        return (rate_score + skill_diversity) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on growth rate and recent activity."""
        growth_rate = metrics['knowledge_growth_rate']
        recent_events = metrics['recent_learning_events']
        
        # Normalize growth rate
        growth_score = max(0, min(1.0, growth_rate * 100))  # Scale to 0-1
        
        # Normalize recent activity
        activity_score = min(1.0, recent_events / 10)  # Normalize to 10 events per hour
        
        return (growth_score + activity_score) / 2

class MetaLearningMonitor(BaseCognitiveSubsystem):
    """Monitors meta-learning processes and effectiveness."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="meta_learning",
            name="Meta-Learning Monitor",
            description="Tracks meta-learning processes, learning-to-learn, and adaptation"
        )
        self.meta_learning_events = []
        self.adaptation_success_rates = []
        self.learning_strategies = []
        self.transfer_effectiveness = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize meta-learning monitoring."""
        self.meta_learning_events = []
        self.adaptation_success_rates = []
        self.learning_strategies = []
        self.transfer_effectiveness = []
        logger.info("Meta-Learning Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect meta-learning metrics."""
        current_time = datetime.now()
        
        # Calculate adaptation success rate
        adaptation_success = np.mean(self.adaptation_success_rates) if self.adaptation_success_rates else 0.0
        
        # Calculate transfer effectiveness
        transfer_effectiveness = np.mean(self.transfer_effectiveness) if self.transfer_effectiveness else 0.0
        
        # Count active strategies
        active_strategies = len(set(self.learning_strategies))
        
        # Count recent meta-learning events
        recent_events = len([
            event for event in self.meta_learning_events
            if (current_time - event.get('timestamp', current_time)).seconds < 1800
        ])
        
        # Calculate strategy diversity
        strategy_diversity = active_strategies / max(len(self.learning_strategies), 1)
        
        # Calculate meta-learning frequency
        total_events = len(self.meta_learning_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        meta_learning_frequency = total_events / hours_elapsed
        
        return {
            'adaptation_success_rate': adaptation_success,
            'transfer_effectiveness': transfer_effectiveness,
            'active_strategies': active_strategies,
            'strategy_diversity': strategy_diversity,
            'total_meta_events': total_events,
            'recent_meta_events': recent_events,
            'meta_learning_frequency': meta_learning_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze meta-learning health."""
        adaptation_success = metrics['adaptation_success_rate']
        transfer_effectiveness = metrics['transfer_effectiveness']
        strategy_diversity = metrics['strategy_diversity']
        
        if adaptation_success < 0.4 or transfer_effectiveness < 0.3 or strategy_diversity < 0.2:
            return SubsystemHealth.CRITICAL
        elif adaptation_success < 0.6 or transfer_effectiveness < 0.5 or strategy_diversity < 0.4:
            return SubsystemHealth.WARNING
        elif adaptation_success < 0.8:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on adaptation and transfer effectiveness."""
        adaptation_success = metrics['adaptation_success_rate']
        transfer_effectiveness = metrics['transfer_effectiveness']
        
        return (adaptation_success + transfer_effectiveness) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on strategy diversity and frequency."""
        strategy_diversity = metrics['strategy_diversity']
        frequency = metrics['meta_learning_frequency']
        
        # Normalize frequency (moderate is good)
        freq_score = max(0, 1 - abs(frequency - 5) / 20)  # Optimal around 5 per hour
        
        return (strategy_diversity + freq_score) / 2

class KnowledgeTransferMonitor(BaseCognitiveSubsystem):
    """Monitors knowledge transfer between tasks and domains."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="knowledge_transfer",
            name="Knowledge Transfer Monitor",
            description="Tracks knowledge transfer between tasks, domains, and sessions"
        )
        self.transfer_events = []
        self.transfer_success_rates = []
        self.domain_mappings = {}
        self.cross_domain_effectiveness = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize knowledge transfer monitoring."""
        self.transfer_events = []
        self.transfer_success_rates = []
        self.domain_mappings = {}
        self.cross_domain_effectiveness = []
        logger.info("Knowledge Transfer Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect knowledge transfer metrics."""
        current_time = datetime.now()
        
        # Calculate transfer success rate
        transfer_success = np.mean(self.transfer_success_rates) if self.transfer_success_rates else 0.0
        
        # Calculate cross-domain effectiveness
        cross_domain_effectiveness = np.mean(self.cross_domain_effectiveness) if self.cross_domain_effectiveness else 0.0
        
        # Count active domains
        active_domains = len([domain for domain, mapping in self.domain_mappings.items() if mapping.get('active', False)])
        
        # Count recent transfers
        recent_transfers = len([
            event for event in self.transfer_events
            if (current_time - event.get('timestamp', current_time)).seconds < 3600
        ])
        
        # Calculate domain diversity
        domain_diversity = active_domains / max(len(self.domain_mappings), 1)
        
        # Calculate transfer frequency
        total_transfers = len(self.transfer_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        transfer_frequency = total_transfers / hours_elapsed
        
        return {
            'transfer_success_rate': transfer_success,
            'cross_domain_effectiveness': cross_domain_effectiveness,
            'active_domains': active_domains,
            'domain_diversity': domain_diversity,
            'total_transfers': total_transfers,
            'recent_transfers': recent_transfers,
            'transfer_frequency': transfer_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze knowledge transfer health."""
        transfer_success = metrics['transfer_success_rate']
        cross_domain_effectiveness = metrics['cross_domain_effectiveness']
        domain_diversity = metrics['domain_diversity']
        
        if transfer_success < 0.3 or cross_domain_effectiveness < 0.2 or domain_diversity < 0.1:
            return SubsystemHealth.CRITICAL
        elif transfer_success < 0.5 or cross_domain_effectiveness < 0.4 or domain_diversity < 0.3:
            return SubsystemHealth.WARNING
        elif transfer_success < 0.7:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on transfer success and cross-domain effectiveness."""
        transfer_success = metrics['transfer_success_rate']
        cross_domain_effectiveness = metrics['cross_domain_effectiveness']
        
        return (transfer_success + cross_domain_effectiveness) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on domain diversity and transfer frequency."""
        domain_diversity = metrics['domain_diversity']
        frequency = metrics['transfer_frequency']
        
        # Normalize frequency (moderate is good)
        freq_score = max(0, 1 - abs(frequency - 3) / 15)  # Optimal around 3 per hour
        
        return (domain_diversity + freq_score) / 2

class PatternRecognitionMonitor(BaseCognitiveSubsystem):
    """Monitors pattern recognition capabilities and effectiveness."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="pattern_recognition",
            name="Pattern Recognition Monitor",
            description="Tracks pattern recognition capabilities and effectiveness"
        )
        self.pattern_events = []
        self.recognition_accuracy = []
        self.pattern_complexity = []
        self.recognition_times = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize pattern recognition monitoring."""
        self.pattern_events = []
        self.recognition_accuracy = []
        self.pattern_complexity = []
        self.recognition_times = []
        logger.info("Pattern Recognition Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect pattern recognition metrics."""
        current_time = datetime.now()
        
        # Calculate recognition accuracy
        accuracy = np.mean(self.recognition_accuracy) if self.recognition_accuracy else 0.0
        
        # Calculate average recognition time
        avg_recognition_time = np.mean(self.recognition_times) if self.recognition_times else 0.0
        
        # Calculate average pattern complexity
        avg_complexity = np.mean(self.pattern_complexity) if self.pattern_complexity else 0.0
        
        # Count recent recognitions
        recent_recognitions = len([
            event for event in self.pattern_events
            if (current_time - event.get('timestamp', current_time)).seconds < 300
        ])
        
        # Calculate pattern diversity
        pattern_types = set([event.get('pattern_type', 'unknown') for event in self.pattern_events])
        pattern_diversity = len(pattern_types) / max(len(self.pattern_events), 1)
        
        # Calculate recognition frequency
        total_recognitions = len(self.pattern_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        recognition_frequency = total_recognitions / hours_elapsed
        
        return {
            'recognition_accuracy': accuracy,
            'avg_recognition_time': avg_recognition_time,
            'avg_pattern_complexity': avg_complexity,
            'pattern_diversity': pattern_diversity,
            'total_recognitions': total_recognitions,
            'recent_recognitions': recent_recognitions,
            'recognition_frequency': recognition_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze pattern recognition health."""
        accuracy = metrics['recognition_accuracy']
        recognition_time = metrics['avg_recognition_time']
        complexity = metrics['avg_pattern_complexity']
        
        if accuracy < 0.6 or recognition_time > 2000 or complexity < 0.3:
            return SubsystemHealth.CRITICAL
        elif accuracy < 0.8 or recognition_time > 1000 or complexity < 0.5:
            return SubsystemHealth.WARNING
        elif accuracy < 0.9:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on accuracy and complexity."""
        accuracy = metrics['recognition_accuracy']
        complexity = metrics['avg_pattern_complexity']
        
        # Normalize complexity (higher is better, up to a point)
        complexity_score = min(1.0, complexity)
        
        return (accuracy + complexity_score) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on recognition time and diversity."""
        recognition_time = metrics['avg_recognition_time']
        diversity = metrics['pattern_diversity']
        
        # Normalize recognition time (lower is better)
        time_score = max(0, 1 - (recognition_time / 2000))
        
        return (time_score + diversity) / 2

class CurriculumLearningMonitor(BaseCognitiveSubsystem):
    """Monitors curriculum learning progression and effectiveness."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="curriculum_learning",
            name="Curriculum Learning Monitor",
            description="Tracks curriculum learning progression and difficulty adaptation"
        )
        self.curriculum_events = []
        self.difficulty_levels = []
        self.progression_rates = []
        self.mastery_scores = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize curriculum learning monitoring."""
        self.curriculum_events = []
        self.difficulty_levels = []
        self.progression_rates = []
        self.mastery_scores = []
        logger.info("Curriculum Learning Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect curriculum learning metrics."""
        current_time = datetime.now()
        
        # Calculate average difficulty level
        avg_difficulty = np.mean(self.difficulty_levels) if self.difficulty_levels else 0.0
        
        # Calculate progression rate
        progression_rate = np.mean(self.progression_rates) if self.progression_rates else 0.0
        
        # Calculate mastery score
        mastery_score = np.mean(self.mastery_scores) if self.mastery_scores else 0.0
        
        # Count recent curriculum events
        recent_events = len([
            event for event in self.curriculum_events
            if (current_time - event.get('timestamp', current_time)).seconds < 1800
        ])
        
        # Calculate difficulty progression
        difficulty_progression = 0.0
        if len(self.difficulty_levels) > 1:
            recent_difficulties = self.difficulty_levels[-10:]
            if len(recent_difficulties) > 1:
                difficulty_progression = np.polyfit(range(len(recent_difficulties)), recent_difficulties, 1)[0]
        
        # Calculate curriculum efficiency
        total_events = len(self.curriculum_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        curriculum_frequency = total_events / hours_elapsed
        
        return {
            'avg_difficulty': avg_difficulty,
            'progression_rate': progression_rate,
            'mastery_score': mastery_score,
            'difficulty_progression': difficulty_progression,
            'total_curriculum_events': total_events,
            'recent_curriculum_events': recent_events,
            'curriculum_frequency': curriculum_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze curriculum learning health."""
        progression_rate = metrics['progression_rate']
        mastery_score = metrics['mastery_score']
        difficulty_progression = metrics['difficulty_progression']
        
        if progression_rate < 0.1 or mastery_score < 0.3 or difficulty_progression < 0.01:
            return SubsystemHealth.CRITICAL
        elif progression_rate < 0.3 or mastery_score < 0.6 or difficulty_progression < 0.05:
            return SubsystemHealth.WARNING
        elif progression_rate < 0.5:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on progression rate and mastery."""
        progression_rate = metrics['progression_rate']
        mastery_score = metrics['mastery_score']
        
        return (progression_rate + mastery_score) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on difficulty progression and frequency."""
        difficulty_progression = metrics['difficulty_progression']
        frequency = metrics['curriculum_frequency']
        
        # Normalize difficulty progression (positive is good)
        progression_score = max(0, min(1.0, difficulty_progression * 10))
        
        # Normalize frequency (moderate is good)
        freq_score = max(0, 1 - abs(frequency - 2) / 10)  # Optimal around 2 per hour
        
        return (progression_score + freq_score) / 2

class CrossSessionLearningMonitor(BaseCognitiveSubsystem):
    """Monitors cross-session learning persistence and effectiveness."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="cross_session_learning",
            name="Cross-Session Learning Monitor",
            description="Tracks learning persistence across sessions and knowledge retention"
        )
        self.session_transitions = []
        self.retention_rates = []
        self.knowledge_persistence = []
        self.session_continuity = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize cross-session learning monitoring."""
        self.session_transitions = []
        self.retention_rates = []
        self.knowledge_persistence = []
        self.session_continuity = []
        logger.info("Cross-Session Learning Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect cross-session learning metrics."""
        current_time = datetime.now()
        
        # Calculate retention rate
        retention_rate = np.mean(self.retention_rates) if self.retention_rates else 0.0
        
        # Calculate knowledge persistence
        knowledge_persistence = np.mean(self.knowledge_persistence) if self.knowledge_persistence else 0.0
        
        # Calculate session continuity
        session_continuity = np.mean(self.session_continuity) if self.session_continuity else 0.0
        
        # Count recent session transitions
        recent_transitions = len([
            transition for transition in self.session_transitions
            if (current_time - transition.get('timestamp', current_time)).seconds < 86400  # 24 hours
        ])
        
        # Calculate session frequency
        total_transitions = len(self.session_transitions)
        days_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 86400)
        session_frequency = total_transitions / days_elapsed
        
        # Calculate learning continuity
        learning_continuity = 0.0
        if len(self.session_continuity) > 1:
            recent_continuity = self.session_continuity[-5:]
            learning_continuity = np.mean(recent_continuity)
        
        return {
            'retention_rate': retention_rate,
            'knowledge_persistence': knowledge_persistence,
            'session_continuity': session_continuity,
            'learning_continuity': learning_continuity,
            'total_session_transitions': total_transitions,
            'recent_session_transitions': recent_transitions,
            'session_frequency': session_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze cross-session learning health."""
        retention_rate = metrics['retention_rate']
        knowledge_persistence = metrics['knowledge_persistence']
        session_continuity = metrics['session_continuity']
        
        if retention_rate < 0.4 or knowledge_persistence < 0.3 or session_continuity < 0.2:
            return SubsystemHealth.CRITICAL
        elif retention_rate < 0.6 or knowledge_persistence < 0.5 or session_continuity < 0.4:
            return SubsystemHealth.WARNING
        elif retention_rate < 0.8:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on retention and persistence."""
        retention_rate = metrics['retention_rate']
        knowledge_persistence = metrics['knowledge_persistence']
        
        return (retention_rate + knowledge_persistence) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on continuity and frequency."""
        session_continuity = metrics['session_continuity']
        learning_continuity = metrics['learning_continuity']
        frequency = metrics['session_frequency']
        
        # Normalize frequency (moderate is good)
        freq_score = max(0, 1 - abs(frequency - 1) / 5)  # Optimal around 1 per day
        
        continuity_score = (session_continuity + learning_continuity) / 2
        
        return (continuity_score + freq_score) / 2
