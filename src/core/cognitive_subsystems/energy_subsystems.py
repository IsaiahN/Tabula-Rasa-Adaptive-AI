"""
Energy-Related Cognitive Subsystems

Implements 4 energy-focused subsystems for comprehensive energy monitoring and management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np

from .base_subsystem import BaseCognitiveSubsystem, SubsystemHealth

logger = logging.getLogger(__name__)

class EnergySystemMonitor(BaseCognitiveSubsystem):
    """Monitors energy system health and efficiency."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="energy_system",
            name="Energy System Monitor",
            description="Tracks energy levels, consumption, and system efficiency"
        )
        self.energy_events = []
        self.energy_levels = []
        self.energy_consumption = []
        self.energy_efficiency = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize energy system monitoring."""
        self.energy_events = []
        self.energy_levels = []
        self.energy_consumption = []
        self.energy_efficiency = []
        logger.info("Energy System Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect energy system metrics."""
        current_time = datetime.now()
        
        # Calculate current energy level
        current_energy = np.mean(self.energy_levels) if self.energy_levels else 100.0
        
        # Calculate energy consumption rate
        consumption_rate = np.mean(self.energy_consumption) if self.energy_consumption else 0.0
        
        # Calculate energy efficiency
        efficiency = np.mean(self.energy_efficiency) if self.energy_efficiency else 1.0
        
        # Count recent energy events
        recent_events = len([
            event for event in self.energy_events
            if (current_time - event.get('timestamp', current_time)).seconds < 300
        ])
        
        # Calculate energy trend
        energy_trend = 0.0
        if len(self.energy_levels) > 1:
            recent_energy = self.energy_levels[-10:]
            if len(recent_energy) > 1:
                energy_trend = np.polyfit(range(len(recent_energy)), recent_energy, 1)[0]
        
        # Calculate energy stability
        energy_stability = 0.0
        if len(self.energy_levels) > 1:
            energy_std = np.std(self.energy_levels)
            energy_stability = max(0, 1 - energy_std / 50)  # Normalize to 0-1
        
        # Calculate total energy events
        total_events = len(self.energy_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        event_frequency = total_events / hours_elapsed
        
        return {
            'current_energy': current_energy,
            'consumption_rate': consumption_rate,
            'efficiency': efficiency,
            'energy_trend': energy_trend,
            'energy_stability': energy_stability,
            'total_energy_events': total_events,
            'recent_energy_events': recent_events,
            'event_frequency': event_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze energy system health."""
        current_energy = metrics['current_energy']
        efficiency = metrics['efficiency']
        stability = metrics['energy_stability']
        
        if current_energy < 20 or efficiency < 0.3 or stability < 0.2:
            return SubsystemHealth.CRITICAL
        elif current_energy < 40 or efficiency < 0.5 or stability < 0.4:
            return SubsystemHealth.WARNING
        elif current_energy < 60:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on energy level and efficiency."""
        current_energy = metrics['current_energy']
        efficiency = metrics['efficiency']
        
        # Normalize energy level (0-100 to 0-1)
        energy_score = current_energy / 100.0
        
        return (energy_score + efficiency) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on consumption and stability."""
        consumption_rate = metrics['consumption_rate']
        stability = metrics['energy_stability']
        
        # Normalize consumption rate (lower is better)
        consumption_score = max(0, 1 - consumption_rate / 10)  # Normalize to 0-1
        
        return (consumption_score + stability) / 2

class SleepCycleMonitor(BaseCognitiveSubsystem):
    """Monitors sleep cycles and consolidation effectiveness."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="sleep_cycle",
            name="Sleep Cycle Monitor",
            description="Tracks sleep cycles, consolidation, and memory processing"
        )
        self.sleep_events = []
        self.sleep_durations = []
        self.consolidation_effectiveness = []
        self.sleep_quality_scores = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize sleep cycle monitoring."""
        self.sleep_events = []
        self.sleep_durations = []
        self.consolidation_effectiveness = []
        self.sleep_quality_scores = []
        logger.info("Sleep Cycle Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect sleep cycle metrics."""
        current_time = datetime.now()
        
        # Calculate average sleep duration
        avg_duration = np.mean(self.sleep_durations) if self.sleep_durations else 0.0
        
        # Calculate consolidation effectiveness
        consolidation_effectiveness = np.mean(self.consolidation_effectiveness) if self.consolidation_effectiveness else 0.0
        
        # Calculate sleep quality
        sleep_quality = np.mean(self.sleep_quality_scores) if self.sleep_quality_scores else 0.0
        
        # Count recent sleep events
        recent_sleep = len([
            event for event in self.sleep_events
            if (current_time - event.get('timestamp', current_time)).seconds < 86400  # 24 hours
        ])
        
        # Calculate sleep frequency
        total_sleep_events = len(self.sleep_events)
        days_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 86400)
        sleep_frequency = total_sleep_events / days_elapsed
        
        # Calculate sleep trend
        sleep_trend = 0.0
        if len(self.sleep_quality_scores) > 1:
            recent_quality = self.sleep_quality_scores[-5:]
            if len(recent_quality) > 1:
                sleep_trend = np.polyfit(range(len(recent_quality)), recent_quality, 1)[0]
        
        # Calculate sleep regularity
        sleep_regularity = 0.0
        if len(self.sleep_durations) > 1:
            duration_std = np.std(self.sleep_durations)
            sleep_regularity = max(0, 1 - duration_std / 1000)  # Normalize to 0-1
        
        return {
            'avg_sleep_duration': avg_duration,
            'consolidation_effectiveness': consolidation_effectiveness,
            'sleep_quality': sleep_quality,
            'sleep_trend': sleep_trend,
            'sleep_regularity': sleep_regularity,
            'total_sleep_events': total_sleep_events,
            'recent_sleep_events': recent_sleep,
            'sleep_frequency': sleep_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze sleep cycle health."""
        sleep_quality = metrics['sleep_quality']
        consolidation_effectiveness = metrics['consolidation_effectiveness']
        sleep_regularity = metrics['sleep_regularity']
        
        if sleep_quality < 0.3 or consolidation_effectiveness < 0.4 or sleep_regularity < 0.2:
            return SubsystemHealth.CRITICAL
        elif sleep_quality < 0.5 or consolidation_effectiveness < 0.6 or sleep_regularity < 0.4:
            return SubsystemHealth.WARNING
        elif sleep_quality < 0.7:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on sleep quality and consolidation."""
        sleep_quality = metrics['sleep_quality']
        consolidation_effectiveness = metrics['consolidation_effectiveness']
        
        return (sleep_quality + consolidation_effectiveness) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on regularity and trend."""
        sleep_regularity = metrics['sleep_regularity']
        sleep_trend = metrics['sleep_trend']
        
        # Normalize sleep trend (positive is good)
        trend_score = max(0, min(1.0, sleep_trend * 10))
        
        return (sleep_regularity + trend_score) / 2

class MidGameSleepMonitor(BaseCognitiveSubsystem):
    """Monitors mid-game sleep and micro-consolidation."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="mid_game_sleep",
            name="Mid-Game Sleep Monitor",
            description="Tracks mid-game sleep, micro-consolidation, and real-time memory processing"
        )
        self.mid_game_sleep_events = []
        self.micro_consolidation_effectiveness = []
        self.sleep_trigger_accuracy = []
        self.mid_game_sleep_durations = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize mid-game sleep monitoring."""
        self.mid_game_sleep_events = []
        self.micro_consolidation_effectiveness = []
        self.sleep_trigger_accuracy = []
        self.mid_game_sleep_durations = []
        logger.info("Mid-Game Sleep Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect mid-game sleep metrics."""
        current_time = datetime.now()
        
        # Calculate micro-consolidation effectiveness
        micro_consolidation = np.mean(self.micro_consolidation_effectiveness) if self.micro_consolidation_effectiveness else 0.0
        
        # Calculate sleep trigger accuracy
        trigger_accuracy = np.mean(self.sleep_trigger_accuracy) if self.sleep_trigger_accuracy else 0.0
        
        # Calculate average mid-game sleep duration
        avg_duration = np.mean(self.mid_game_sleep_durations) if self.mid_game_sleep_durations else 0.0
        
        # Count recent mid-game sleep events
        recent_mid_game_sleep = len([
            event for event in self.mid_game_sleep_events
            if (current_time - event.get('timestamp', current_time)).seconds < 3600
        ])
        
        # Calculate mid-game sleep frequency
        total_mid_game_sleep = len(self.mid_game_sleep_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        mid_game_sleep_frequency = total_mid_game_sleep / hours_elapsed
        
        # Calculate trigger sensitivity
        trigger_sensitivity = 0.0
        if self.sleep_trigger_accuracy:
            accurate_triggers = len([acc for acc in self.sleep_trigger_accuracy if acc > 0.8])
            trigger_sensitivity = accurate_triggers / len(self.sleep_trigger_accuracy)
        
        # Calculate duration efficiency
        duration_efficiency = 0.0
        if self.mid_game_sleep_durations:
            optimal_duration = 100  # Optimal mid-game sleep duration
            duration_efficiency = max(0, 1 - abs(avg_duration - optimal_duration) / optimal_duration)
        
        return {
            'micro_consolidation_effectiveness': micro_consolidation,
            'trigger_accuracy': trigger_accuracy,
            'avg_duration': avg_duration,
            'trigger_sensitivity': trigger_sensitivity,
            'duration_efficiency': duration_efficiency,
            'total_mid_game_sleep': total_mid_game_sleep,
            'recent_mid_game_sleep': recent_mid_game_sleep,
            'mid_game_sleep_frequency': mid_game_sleep_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze mid-game sleep health."""
        micro_consolidation = metrics['micro_consolidation_effectiveness']
        trigger_accuracy = metrics['trigger_accuracy']
        duration_efficiency = metrics['duration_efficiency']
        
        if micro_consolidation < 0.3 or trigger_accuracy < 0.5 or duration_efficiency < 0.4:
            return SubsystemHealth.CRITICAL
        elif micro_consolidation < 0.5 or trigger_accuracy < 0.7 or duration_efficiency < 0.6:
            return SubsystemHealth.WARNING
        elif micro_consolidation < 0.7:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on consolidation and trigger accuracy."""
        micro_consolidation = metrics['micro_consolidation_effectiveness']
        trigger_accuracy = metrics['trigger_accuracy']
        
        return (micro_consolidation + trigger_accuracy) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on duration efficiency and sensitivity."""
        duration_efficiency = metrics['duration_efficiency']
        trigger_sensitivity = metrics['trigger_sensitivity']
        
        return (duration_efficiency + trigger_sensitivity) / 2

class DeathManagerMonitor(BaseCognitiveSubsystem):
    """Monitors death management and recovery processes."""
    
    def __init__(self):
        super().__init__(
            subsystem_id="death_manager",
            name="Death Manager Monitor",
            description="Tracks death events, recovery processes, and system resilience"
        )
        self.death_events = []
        self.recovery_times = []
        self.recovery_success_rates = []
        self.death_frequencies = []
    
    async def _initialize_subsystem(self) -> None:
        """Initialize death manager monitoring."""
        self.death_events = []
        self.recovery_times = []
        self.recovery_success_rates = []
        self.death_frequencies = []
        logger.info("Death Manager Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect death manager metrics."""
        current_time = datetime.now()
        
        # Calculate average recovery time
        avg_recovery_time = np.mean(self.recovery_times) if self.recovery_times else 0.0
        
        # Calculate recovery success rate
        recovery_success = np.mean(self.recovery_success_rates) if self.recovery_success_rates else 0.0
        
        # Calculate death frequency
        death_frequency = np.mean(self.death_frequencies) if self.death_frequencies else 0.0
        
        # Count recent death events
        recent_deaths = len([
            event for event in self.death_events
            if (current_time - event.get('timestamp', current_time)).seconds < 3600
        ])
        
        # Calculate total death events
        total_deaths = len(self.death_events)
        hours_elapsed = max(1, (current_time - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600)
        total_death_frequency = total_deaths / hours_elapsed
        
        # Calculate death trend
        death_trend = 0.0
        if len(self.death_frequencies) > 1:
            recent_frequencies = self.death_frequencies[-10:]
            if len(recent_frequencies) > 1:
                death_trend = np.polyfit(range(len(recent_frequencies)), recent_frequencies, 1)[0]
        
        # Calculate system resilience
        system_resilience = 0.0
        if recovery_success > 0 and avg_recovery_time > 0:
            # Higher success rate and lower recovery time = higher resilience
            system_resilience = recovery_success * (1 - min(1, avg_recovery_time / 1000))
        
        # Calculate death prevention effectiveness
        death_prevention = 0.0
        if total_death_frequency > 0:
            # Lower death frequency = higher prevention effectiveness
            death_prevention = max(0, 1 - total_death_frequency / 10)  # Normalize to 0-1
        
        return {
            'avg_recovery_time': avg_recovery_time,
            'recovery_success_rate': recovery_success,
            'death_frequency': death_frequency,
            'death_trend': death_trend,
            'system_resilience': system_resilience,
            'death_prevention': death_prevention,
            'total_deaths': total_deaths,
            'recent_deaths': recent_deaths,
            'total_death_frequency': total_death_frequency,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze death manager health."""
        recovery_success = metrics['recovery_success_rate']
        system_resilience = metrics['system_resilience']
        death_prevention = metrics['death_prevention']
        
        if recovery_success < 0.5 or system_resilience < 0.3 or death_prevention < 0.2:
            return SubsystemHealth.CRITICAL
        elif recovery_success < 0.7 or system_resilience < 0.5 or death_prevention < 0.4:
            return SubsystemHealth.WARNING
        elif recovery_success < 0.8:
            return SubsystemHealth.GOOD
        else:
            return SubsystemHealth.EXCELLENT
    
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score based on recovery success and system resilience."""
        recovery_success = metrics['recovery_success_rate']
        system_resilience = metrics['system_resilience']
        
        return (recovery_success + system_resilience) / 2
    
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score based on death prevention and recovery time."""
        death_prevention = metrics['death_prevention']
        recovery_time = metrics['avg_recovery_time']
        
        # Normalize recovery time (lower is better)
        recovery_score = max(0, 1 - recovery_time / 1000)  # Normalize to 0-1
        
        return (death_prevention + recovery_score) / 2
