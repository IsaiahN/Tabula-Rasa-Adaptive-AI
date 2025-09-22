"""
Unified Automation System - Phase 1 Integration

This system integrates all Phase 1 automation components:
- Self-Healing System
- Autonomous System Monitor
- Self-Configuring System

Provides a unified interface for managing all automation systems.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .self_healing_system import start_self_healing, stop_self_healing, get_healing_status
from .autonomous_system_monitor import start_autonomous_monitoring, stop_autonomous_monitoring, get_monitoring_status
from .self_configuring_system import start_self_configuring, stop_self_configuring, get_configuring_status
from ..database.system_integration import get_system_integration

logger = logging.getLogger(__name__)

class AutomationPhase(Enum):
    """Automation implementation phases."""
    PHASE_1 = "phase_1"  # Self-healing, monitoring, configuring
    PHASE_2 = "phase_2"  # Meta-learning, testing
    PHASE_3 = "phase_3"  # Code evolution, architecture improvement
    PHASE_4 = "phase_4"  # Complete self-sufficiency

class SystemHealth(Enum):
    """Overall system health levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class AutomationStatus:
    """Status of automation systems."""
    phase: AutomationPhase
    overall_health: SystemHealth
    systems_active: Dict[str, bool]
    metrics: Dict[str, Any]
    last_update: float

class UnifiedAutomationSystem:
    """
    Unified Automation System that coordinates all automation components.
    
    Phase 1 Features:
    - Self-Healing System (automatic error recovery)
    - Autonomous System Monitor (health monitoring and auto-actions)
    - Self-Configuring System (automatic configuration optimization)
    - Unified control and monitoring
    - Cross-system coordination
    """
    
    def __init__(self):
        self.integration = get_system_integration()
        
        # System state
        self.automation_active = False
        self.current_phase = AutomationPhase.PHASE_1
        self.systems = {
            'self_healing': False,
            'monitoring': False,
            'self_configuring': False
        }
        
        # Health tracking
        self.health_history = []
        self.last_health_check = 0
        self.health_check_interval = 30  # seconds
        
        # Performance metrics
        self.metrics = {
            "total_automation_cycles": 0,
            "errors_auto_fixed": 0,
            "threshold_violations_handled": 0,
            "config_changes_applied": 0,
            "system_health_improvements": 0,
            "cross_system_coordinations": 0,
            "emergency_interventions": 0
        }
        
        # Coordination
        self.coordination_active = False
        self.last_coordination = 0
        self.coordination_interval = 60  # seconds
        
    async def start_automation(self, phase: AutomationPhase = AutomationPhase.PHASE_1):
        """Start the unified automation system."""
        if self.automation_active:
            logger.warning("Automation system already active")
            return
        
        self.automation_active = True
        self.current_phase = phase
        logger.info(f"🚀 Starting Unified Automation System - {phase.value}")
        
        # Start Phase 1 systems
        if phase == AutomationPhase.PHASE_1:
            await self._start_phase_1_systems()
        
        # Start coordination
        asyncio.create_task(self._coordination_loop())
        
        # Start health monitoring
        asyncio.create_task(self._health_monitoring_loop())
        
        # Start metrics collection
        asyncio.create_task(self._metrics_collection_loop())
        
        logger.info("✅ Unified Automation System started successfully")
    
    async def stop_automation(self):
        """Stop the unified automation system."""
        self.automation_active = False
        logger.info("🛑 Stopping Unified Automation System")
        
        # Stop all systems
        await self._stop_all_systems()
        
        logger.info("✅ Unified Automation System stopped")
    
    async def _start_phase_1_systems(self):
        """Start Phase 1 automation systems."""
        try:
            # Start Self-Healing System
            logger.info("🩺 Starting Self-Healing System")
            await start_self_healing()
            self.systems['self_healing'] = True
            
            # Start Autonomous System Monitor
            logger.info("📊 Starting Autonomous System Monitor")
            await start_autonomous_monitoring()
            self.systems['monitoring'] = True
            
            # Start Self-Configuring System
            logger.info("⚙️ Starting Self-Configuring System")
            await start_self_configuring()
            self.systems['self_configuring'] = True
            
            logger.info("✅ Phase 1 systems started successfully")
            
        except Exception as e:
            logger.error(f"Error starting Phase 1 systems: {e}")
            raise
    
    async def _stop_all_systems(self):
        """Stop all automation systems."""
        try:
            # Stop Self-Healing System
            if self.systems['self_healing']:
                await stop_self_healing()
                self.systems['self_healing'] = False
            
            # Stop Autonomous System Monitor
            if self.systems['monitoring']:
                await stop_autonomous_monitoring()
                self.systems['monitoring'] = False
            
            # Stop Self-Configuring System
            if self.systems['self_configuring']:
                await stop_self_configuring()
                self.systems['self_configuring'] = False
            
        except Exception as e:
            logger.error(f"Error stopping systems: {e}")
    
    async def _coordination_loop(self):
        """Main coordination loop for cross-system communication."""
        while self.automation_active:
            try:
                current_time = time.time()
                
                if current_time - self.last_coordination >= self.coordination_interval:
                    await self._coordinate_systems()
                    self.last_coordination = current_time
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(30)
    
    async def _coordinate_systems(self):
        """Coordinate between different automation systems."""
        try:
            # Get status from all systems
            healing_status = get_healing_status()
            monitoring_status = get_monitoring_status()
            configuring_status = get_configuring_status()
            
            # Coordinate based on system status
            await self._coordinate_healing_and_monitoring(healing_status, monitoring_status)
            await self._coordinate_monitoring_and_configuring(monitoring_status, configuring_status)
            await self._coordinate_healing_and_configuring(healing_status, configuring_status)
            
            # Update metrics
            self.metrics["cross_system_coordinations"] += 1
            
        except Exception as e:
            logger.error(f"Error coordinating systems: {e}")
    
    async def _coordinate_healing_and_monitoring(self, healing_status: Dict[str, Any], monitoring_status: Dict[str, Any]):
        """Coordinate between healing and monitoring systems."""
        try:
            # If monitoring detects issues, ensure healing is active
            if monitoring_status.get('monitoring_active', False):
                violations = monitoring_status.get('metrics', {}).get('threshold_violations', 0)
                if violations > 0 and not healing_status.get('healing_active', False):
                    logger.info("🔄 Coordinating: Activating healing system due to monitoring violations")
                    await start_self_healing()
                    self.systems['self_healing'] = True
            
            # If healing is fixing many errors, ensure monitoring is active
            if healing_status.get('healing_active', False):
                errors_fixed = healing_status.get('metrics', {}).get('errors_auto_fixed', 0)
                if errors_fixed > 10 and not monitoring_status.get('monitoring_active', False):
                    logger.info("🔄 Coordinating: Activating monitoring due to high error fixing activity")
                    await start_autonomous_monitoring()
                    self.systems['monitoring'] = True
            
        except Exception as e:
            logger.error(f"Error coordinating healing and monitoring: {e}")
    
    async def _coordinate_monitoring_and_configuring(self, monitoring_status: Dict[str, Any], configuring_status: Dict[str, Any]):
        """Coordinate between monitoring and configuring systems."""
        try:
            # If monitoring detects performance issues, ensure configuring is active
            if monitoring_status.get('monitoring_active', False):
                auto_actions = monitoring_status.get('metrics', {}).get('auto_actions_taken', 0)
                if auto_actions > 5 and not configuring_status.get('configuring_active', False):
                    logger.info("🔄 Coordinating: Activating configuring due to frequent monitoring actions")
                    await start_self_configuring()
                    self.systems['self_configuring'] = True
            
            # If configuring is making many changes, ensure monitoring is active
            if configuring_status.get('configuring_active', False):
                config_changes = configuring_status.get('metrics', {}).get('config_changes_applied', 0)
                if config_changes > 3 and not monitoring_status.get('monitoring_active', False):
                    logger.info("🔄 Coordinating: Activating monitoring due to configuration changes")
                    await start_autonomous_monitoring()
                    self.systems['monitoring'] = True
            
        except Exception as e:
            logger.error(f"Error coordinating monitoring and configuring: {e}")
    
    async def _coordinate_healing_and_configuring(self, healing_status: Dict[str, Any], configuring_status: Dict[str, Any]):
        """Coordinate between healing and configuring systems."""
        try:
            # If healing is fixing many errors, consider configuration changes
            if healing_status.get('healing_active', False) and configuring_status.get('configuring_active', False):
                errors_fixed = healing_status.get('metrics', {}).get('errors_auto_fixed', 0)
                config_changes = configuring_status.get('metrics', {}).get('config_changes_applied', 0)
                
                # If many errors are being fixed but few config changes, suggest more aggressive configuring
                if errors_fixed > 20 and config_changes < 5:
                    logger.info("🔄 Coordinating: Suggesting more aggressive configuration due to high error rate")
                    # This would trigger more aggressive configuration changes
            
        except Exception as e:
            logger.error(f"Error coordinating healing and configuring: {e}")
    
    async def _health_monitoring_loop(self):
        """Health monitoring loop for the automation system."""
        while self.automation_active:
            try:
                current_time = time.time()
                
                if current_time - self.last_health_check >= self.health_check_interval:
                    await self._check_automation_health()
                    self.last_health_check = current_time
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _check_automation_health(self):
        """Check overall automation system health."""
        try:
            # Get individual system health
            healing_status = get_healing_status()
            monitoring_status = get_monitoring_status()
            configuring_status = get_configuring_status()
            
            # Calculate overall health
            overall_health = self._calculate_overall_health(healing_status, monitoring_status, configuring_status)
            
            # Create health status
            health_status = AutomationStatus(
                phase=self.current_phase,
                overall_health=overall_health,
                systems_active=self.systems.copy(),
                metrics=self.metrics.copy(),
                last_update=time.time()
            )
            
            # Store health history
            self.health_history.append(health_status)
            if len(self.health_history) > 100:  # Keep last 100 health checks
                self.health_history = self.health_history[-100:]
            
            # Handle health issues
            await self._handle_health_issues(health_status)
            
            # Log health status
            logger.debug(f"🏥 Automation health: {overall_health.value}")
            
        except Exception as e:
            logger.error(f"Error checking automation health: {e}")
    
    def _calculate_overall_health(self, healing_status: Dict[str, Any], monitoring_status: Dict[str, Any], configuring_status: Dict[str, Any]) -> SystemHealth:
        """Calculate overall automation system health."""
        try:
            # Check if systems are active
            systems_active = sum(self.systems.values())
            total_systems = len(self.systems)
            
            if systems_active == 0:
                return SystemHealth.EMERGENCY
            elif systems_active < total_systems:
                return SystemHealth.CRITICAL
            elif systems_active == total_systems:
                # Check individual system health
                healing_health = self._assess_system_health(healing_status)
                monitoring_health = self._assess_system_health(monitoring_status)
                configuring_health = self._assess_system_health(configuring_status)
                
                # Calculate average health
                avg_health = (healing_health + monitoring_health + configuring_health) / 3
                
                if avg_health >= 0.9:
                    return SystemHealth.EXCELLENT
                elif avg_health >= 0.7:
                    return SystemHealth.GOOD
                elif avg_health >= 0.5:
                    return SystemHealth.WARNING
                else:
                    return SystemHealth.CRITICAL
            
            return SystemHealth.WARNING
            
        except Exception as e:
            logger.error(f"Error calculating overall health: {e}")
            return SystemHealth.CRITICAL
    
    def _assess_system_health(self, system_status: Dict[str, Any]) -> float:
        """Assess health of an individual system."""
        try:
            if not system_status.get('healing_active', False) and not system_status.get('monitoring_active', False) and not system_status.get('configuring_active', False):
                return 0.0
            
            # Get metrics
            metrics = system_status.get('metrics', {})
            
            # Calculate health score based on metrics
            if 'healing_active' in system_status:
                # Healing system health
                errors_detected = metrics.get('errors_detected', 0)
                errors_fixed = metrics.get('errors_auto_fixed', 0)
                
                if errors_detected > 0:
                    success_rate = errors_fixed / errors_detected
                else:
                    success_rate = 1.0
                
                return min(1.0, success_rate)
            
            elif 'monitoring_active' in system_status:
                # Monitoring system health
                violations = metrics.get('threshold_violations', 0)
                actions_taken = metrics.get('auto_actions_taken', 0)
                
                if violations > 0:
                    action_rate = actions_taken / violations
                else:
                    action_rate = 1.0
                
                return min(1.0, action_rate)
            
            elif 'configuring_active' in system_status:
                # Configuring system health
                changes_applied = metrics.get('config_changes_applied', 0)
                changes_rolled_back = metrics.get('config_changes_rolled_back', 0)
                
                if changes_applied > 0:
                    success_rate = (changes_applied - changes_rolled_back) / changes_applied
                else:
                    success_rate = 1.0
                
                return min(1.0, success_rate)
            
            return 0.5  # Default health score
            
        except Exception as e:
            logger.error(f"Error assessing system health: {e}")
            return 0.0
    
    async def _handle_health_issues(self, health_status: AutomationStatus):
        """Handle health issues in the automation system."""
        try:
            if health_status.overall_health == SystemHealth.EMERGENCY:
                await self._handle_emergency_health()
            elif health_status.overall_health == SystemHealth.CRITICAL:
                await self._handle_critical_health()
            elif health_status.overall_health == SystemHealth.WARNING:
                await self._handle_warning_health()
            
        except Exception as e:
            logger.error(f"Error handling health issues: {e}")
    
    async def _handle_emergency_health(self):
        """Handle emergency health situation."""
        try:
            logger.critical("🚨 EMERGENCY: Automation system health critical")
            
            # Restart all systems
            await self._restart_all_systems()
            
            # Update metrics
            self.metrics["emergency_interventions"] += 1
            
        except Exception as e:
            logger.error(f"Error handling emergency health: {e}")
    
    async def _handle_critical_health(self):
        """Handle critical health situation."""
        try:
            logger.warning("⚠️ CRITICAL: Automation system health degraded")
            
            # Restart failed systems
            await self._restart_failed_systems()
            
        except Exception as e:
            logger.error(f"Error handling critical health: {e}")
    
    async def _handle_warning_health(self):
        """Handle warning health situation."""
        try:
            logger.info("⚠️ WARNING: Automation system health needs attention")
            
            # Monitor more closely
            self.health_check_interval = 15  # Check every 15 seconds instead of 30
            
        except Exception as e:
            logger.error(f"Error handling warning health: {e}")
    
    async def _restart_all_systems(self):
        """Restart all automation systems."""
        try:
            logger.info("🔄 Restarting all automation systems")
            
            # Stop all systems
            await self._stop_all_systems()
            
            # Wait a moment
            await asyncio.sleep(5)
            
            # Start all systems
            await self._start_phase_1_systems()
            
        except Exception as e:
            logger.error(f"Error restarting all systems: {e}")
    
    async def _restart_failed_systems(self):
        """Restart failed automation systems."""
        try:
            # Check which systems are not active
            healing_status = get_healing_status()
            monitoring_status = get_monitoring_status()
            configuring_status = get_configuring_status()
            
            if not healing_status.get('healing_active', False):
                logger.info("🔄 Restarting self-healing system")
                await start_self_healing()
                self.systems['self_healing'] = True
            
            if not monitoring_status.get('monitoring_active', False):
                logger.info("🔄 Restarting monitoring system")
                await start_autonomous_monitoring()
                self.systems['monitoring'] = True
            
            if not configuring_status.get('configuring_active', False):
                logger.info("🔄 Restarting configuring system")
                await start_self_configuring()
                self.systems['self_configuring'] = True
            
        except Exception as e:
            logger.error(f"Error restarting failed systems: {e}")
    
    async def _metrics_collection_loop(self):
        """Metrics collection loop."""
        while self.automation_active:
            try:
                # Collect metrics from all systems
                await self._collect_metrics()
                
                await asyncio.sleep(60)  # Collect metrics every minute
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(30)
    
    async def _collect_metrics(self):
        """Collect metrics from all automation systems."""
        try:
            # Get metrics from individual systems
            healing_status = get_healing_status()
            monitoring_status = get_monitoring_status()
            configuring_status = get_configuring_status()
            
            # Update unified metrics
            self.metrics["errors_auto_fixed"] = healing_status.get('metrics', {}).get('errors_auto_fixed', 0)
            self.metrics["threshold_violations_handled"] = monitoring_status.get('metrics', {}).get('threshold_violations', 0)
            self.metrics["config_changes_applied"] = configuring_status.get('metrics', {}).get('config_changes_applied', 0)
            
            # Calculate system health improvements
            if len(self.health_history) >= 2:
                current_health = self.health_history[-1]
                previous_health = self.health_history[-2]
                
                if self._health_improved(current_health, previous_health):
                    self.metrics["system_health_improvements"] += 1
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    def _health_improved(self, current: AutomationStatus, previous: AutomationStatus) -> bool:
        """Check if system health has improved."""
        try:
            health_levels = {
                SystemHealth.EMERGENCY: 0,
                SystemHealth.CRITICAL: 1,
                SystemHealth.WARNING: 2,
                SystemHealth.GOOD: 3,
                SystemHealth.EXCELLENT: 4
            }
            
            current_level = health_levels.get(current.overall_health, 0)
            previous_level = health_levels.get(previous.overall_health, 0)
            
            return current_level > previous_level
            
        except Exception as e:
            logger.error(f"Error checking health improvement: {e}")
            return False
    
    def get_automation_status(self) -> Dict[str, Any]:
        """Get comprehensive automation system status."""
        return {
            "automation_active": self.automation_active,
            "current_phase": self.current_phase.value,
            "systems_active": self.systems,
            "metrics": self.metrics,
            "health_history_size": len(self.health_history),
            "last_health_check": self.last_health_check,
            "coordination_active": self.coordination_active,
            "last_coordination": self.last_coordination
        }
    
    def get_phase_1_status(self) -> Dict[str, Any]:
        """Get Phase 1 specific status."""
        return {
            "self_healing": get_healing_status(),
            "monitoring": get_monitoring_status(),
            "self_configuring": get_configuring_status()
        }

# Global unified automation system instance
unified_automation = UnifiedAutomationSystem()

async def start_unified_automation(phase: str = "phase_1"):
    """Start the unified automation system."""
    automation_phase = AutomationPhase(phase)
    await unified_automation.start_automation(automation_phase)

async def stop_unified_automation():
    """Stop the unified automation system."""
    await unified_automation.stop_automation()

def get_automation_status():
    """Get automation system status."""
    return unified_automation.get_automation_status()

def get_phase_1_status():
    """Get Phase 1 status."""
    return unified_automation.get_phase_1_status()
