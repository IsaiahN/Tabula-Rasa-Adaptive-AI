"""
Autonomous System Manager

This manager coordinates the autonomous Governor and Architect systems,
providing a unified interface for Director to interact with the autonomous subsystems.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .autonomous_governor import start_autonomous_governor, stop_autonomous_governor, get_autonomous_governor_status
from .autonomous_architect import start_autonomous_architect, stop_autonomous_architect, get_autonomous_architect_status
from .governor_architect_bridge import start_governor_architect_communication, stop_governor_architect_communication, get_bridge_status
from ..database.system_integration import get_system_integration

logger = logging.getLogger(__name__)

class SystemMode(Enum):
    """System operation modes."""
    AUTONOMOUS = "autonomous"     # Full autonomy - Governor and Architect operate independently
    COLLABORATIVE = "collaborative"  # Governor and Architect collaborate with Director
    DIRECTED = "directed"         # Director has full control
    EMERGENCY = "emergency"       # Emergency mode - minimal autonomy

@dataclass
class SystemStatus:
    """Overall system status."""
    mode: SystemMode
    governor_active: bool
    architect_active: bool
    bridge_active: bool
    overall_health: float
    autonomy_level: float
    last_update: float
    metrics: Dict[str, Any]

class AutonomousSystemManager:
    """
    Manager for the autonomous Governor and Architect systems.
    
    This manager provides:
    1. Unified control interface for Director
    2. System health monitoring
    3. Mode switching capabilities
    4. Performance tracking
    5. Emergency handling
    """
    
    def __init__(self):
        self.integration = get_system_integration()
        
        # System state
        self.current_mode = SystemMode.DIRECTED
        self.system_active = False
        self.emergency_mode = False
        
        # Component status
        self.governor_status = {}
        self.architect_status = {}
        self.bridge_status = {}
        
        # Performance tracking
        self.performance_metrics = {
            "total_autonomous_decisions": 0,
            "total_autonomous_evolutions": 0,
            "successful_collaborations": 0,
            "system_uptime": 0,
            "mode_switches": 0,
            "emergency_events": 0
        }
        
        # Health monitoring
        self.health_history = []
        self.last_health_check = 0
        self.health_check_interval = 30  # seconds
        
        # Director interface
        self.director_notifications = []
        self.director_requests = []
        
    async def start_autonomous_system(self, mode: SystemMode = SystemMode.AUTONOMOUS):
        """Start the autonomous system in the specified mode."""
        try:
            logger.info(f"🚀 Starting autonomous system in {mode.value} mode")
            
            # Set mode
            self.current_mode = mode
            self.system_active = True
            
            # Start components based on mode
            if mode in [SystemMode.AUTONOMOUS, SystemMode.COLLABORATIVE]:
                # Start Governor
                await start_autonomous_governor()
                logger.info("✅ Autonomous Governor started")
                
                # Start Architect
                await start_autonomous_architect()
                logger.info("✅ Autonomous Architect started")
                
                # Start communication bridge
                await start_governor_architect_communication()
                logger.info("✅ Governor-Architect bridge started")
                
            elif mode == SystemMode.DIRECTED:
                # In directed mode, components are available but not autonomous
                logger.info("📋 System in directed mode - components available but not autonomous")
                
            elif mode == SystemMode.EMERGENCY:
                # Emergency mode - minimal functionality
                logger.info("🚨 System in emergency mode - minimal functionality")
            
            # Start health monitoring
            asyncio.create_task(self._health_monitoring_loop())
            
            # Start performance tracking
            asyncio.create_task(self._performance_tracking_loop())
            
            # Start Director notification loop
            asyncio.create_task(self._director_notification_loop())
            
            # Update metrics
            self.performance_metrics["mode_switches"] += 1
            
            logger.info(f"🎯 Autonomous system started successfully in {mode.value} mode")
            
        except Exception as e:
            logger.error(f"Failed to start autonomous system: {e}")
            raise
    
    async def stop_autonomous_system(self):
        """Stop the autonomous system."""
        try:
            logger.info("🛑 Stopping autonomous system")
            
            # Stop components
            if self.current_mode in [SystemMode.AUTONOMOUS, SystemMode.COLLABORATIVE]:
                await stop_governor_architect_communication()
                await stop_autonomous_architect()
                await stop_autonomous_governor()
            
            # Update state
            self.system_active = False
            
            logger.info("✅ Autonomous system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping autonomous system: {e}")
    
    async def switch_mode(self, new_mode: SystemMode):
        """Switch system mode."""
        try:
            logger.info(f"🔄 Switching from {self.current_mode.value} to {new_mode.value} mode")
            
            # Stop current mode
            if self.system_active:
                await self.stop_autonomous_system()
            
            # Start new mode
            await self.start_autonomous_system(new_mode)
            
            # Update metrics
            self.performance_metrics["mode_switches"] += 1
            
            logger.info(f"✅ Successfully switched to {new_mode.value} mode")
            
        except Exception as e:
            logger.error(f"Error switching mode: {e}")
            raise
    
    async def _health_monitoring_loop(self):
        """Monitor system health."""
        while self.system_active:
            try:
                current_time = time.time()
                
                if current_time - self.last_health_check >= self.health_check_interval:
                    await self._perform_health_check()
                    self.last_health_check = current_time
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _performance_tracking_loop(self):
        """Track system performance."""
        while self.system_active:
            try:
                await self._update_performance_metrics()
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in performance tracking loop: {e}")
                await asyncio.sleep(10)
    
    async def _director_notification_loop(self):
        """Send notifications to Director."""
        while self.system_active:
            try:
                await self._process_director_notifications()
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in director notification loop: {e}")
                await asyncio.sleep(5)
    
    async def _perform_health_check(self):
        """Perform comprehensive health check."""
        try:
            # Get component status
            self.governor_status = get_autonomous_governor_status()
            self.architect_status = get_autonomous_architect_status()
            self.bridge_status = get_bridge_status()
            
            # Calculate overall health
            overall_health = self._calculate_overall_health()
            
            # Calculate autonomy level
            autonomy_level = self._calculate_autonomy_level()
            
            # Create health status
            health_status = SystemStatus(
                mode=self.current_mode,
                governor_active=self.governor_status.get("autonomous_cycle_active", False),
                architect_active=self.architect_status.get("autonomous_cycle_active", False),
                bridge_active=self.bridge_status.get("communication_active", False),
                overall_health=overall_health,
                autonomy_level=autonomy_level,
                last_update=time.time(),
                metrics=self.performance_metrics
            )
            
            # Store health history
            self.health_history.append(health_status)
            if len(self.health_history) > 100:  # Keep last 100 health checks
                self.health_history = self.health_history[-100:]
            
            # Check for health issues
            await self._check_health_issues(health_status)
            
            # Log health status
            logger.debug(f"🏥 Health check: overall={overall_health:.2f}, autonomy={autonomy_level:.2f}")
            
        except Exception as e:
            logger.error(f"Error performing health check: {e}")
    
    def _calculate_overall_health(self) -> float:
        """Calculate overall system health."""
        try:
            # Get component health scores
            governor_health = 1.0 if self.governor_status.get("autonomous_cycle_active", False) else 0.0
            architect_health = 1.0 if self.architect_status.get("autonomous_cycle_active", False) else 0.0
            bridge_health = 1.0 if self.bridge_status.get("communication_active", False) else 0.0
            
            # Calculate weighted average
            overall_health = (governor_health * 0.4 + architect_health * 0.4 + bridge_health * 0.2)
            
            return overall_health
            
        except Exception as e:
            logger.error(f"Error calculating overall health: {e}")
            return 0.0
    
    def _calculate_autonomy_level(self) -> float:
        """Calculate current autonomy level."""
        try:
            if self.current_mode == SystemMode.AUTONOMOUS:
                return 1.0
            elif self.current_mode == SystemMode.COLLABORATIVE:
                return 0.7
            elif self.current_mode == SystemMode.DIRECTED:
                return 0.3
            elif self.current_mode == SystemMode.EMERGENCY:
                return 0.1
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating autonomy level: {e}")
            return 0.0
    
    async def _check_health_issues(self, health_status: SystemStatus):
        """Check for health issues and take action."""
        try:
            # Check for critical health issues
            if health_status.overall_health < 0.3:
                await self._handle_critical_health_issue(health_status)
            
            # Check for component failures
            if not health_status.governor_active and self.current_mode in [SystemMode.AUTONOMOUS, SystemMode.COLLABORATIVE]:
                await self._handle_component_failure("governor", health_status)
            
            if not health_status.architect_active and self.current_mode in [SystemMode.AUTONOMOUS, SystemMode.COLLABORATIVE]:
                await self._handle_component_failure("architect", health_status)
            
            if not health_status.bridge_active and self.current_mode in [SystemMode.AUTONOMOUS, SystemMode.COLLABORATIVE]:
                await self._handle_component_failure("bridge", health_status)
            
        except Exception as e:
            logger.error(f"Error checking health issues: {e}")
    
    async def _handle_critical_health_issue(self, health_status: SystemStatus):
        """Handle critical health issues."""
        try:
            logger.warning(f"🚨 Critical health issue detected: overall_health={health_status.overall_health:.2f}")
            
            # Switch to emergency mode
            if self.current_mode != SystemMode.EMERGENCY:
                await self.switch_mode(SystemMode.EMERGENCY)
            
            # Notify Director
            await self._notify_director({
                "type": "critical_health_issue",
                "severity": "critical",
                "overall_health": health_status.overall_health,
                "action_taken": "switched_to_emergency_mode",
                "timestamp": time.time()
            })
            
            # Update metrics
            self.performance_metrics["emergency_events"] += 1
            
        except Exception as e:
            logger.error(f"Error handling critical health issue: {e}")
    
    async def _handle_component_failure(self, component: str, health_status: SystemStatus):
        """Handle component failure."""
        try:
            logger.warning(f"⚠️ Component failure detected: {component}")
            
            # Notify Director
            await self._notify_director({
                "type": "component_failure",
                "severity": "high",
                "component": component,
                "overall_health": health_status.overall_health,
                "timestamp": time.time()
            })
            
            # Attempt recovery
            await self._attempt_component_recovery(component)
            
        except Exception as e:
            logger.error(f"Error handling component failure: {e}")
    
    async def _attempt_component_recovery(self, component: str):
        """Attempt to recover a failed component."""
        try:
            logger.info(f"🔧 Attempting recovery for component: {component}")
            
            if component == "governor":
                await start_autonomous_governor()
            elif component == "architect":
                await start_autonomous_architect()
            elif component == "bridge":
                await start_governor_architect_communication()
            
            logger.info(f"✅ Recovery attempted for component: {component}")
            
        except Exception as e:
            logger.error(f"Error attempting component recovery: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics."""
        try:
            # Update autonomous decision counts
            governor_decisions = self.governor_status.get("decisions_made", 0)
            architect_evolutions = self.architect_status.get("evolutions_made", 0)
            bridge_collaborations = self.bridge_status.get("successful_collaborations", 0)
            
            self.performance_metrics["total_autonomous_decisions"] = governor_decisions
            self.performance_metrics["total_autonomous_evolutions"] = architect_evolutions
            self.performance_metrics["successful_collaborations"] = bridge_collaborations
            
            # Update system uptime
            if self.system_active:
                self.performance_metrics["system_uptime"] = time.time() - self.performance_metrics.get("start_time", time.time())
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _process_director_notifications(self):
        """Process notifications for Director."""
        try:
            # Process any pending notifications
            while self.director_notifications:
                notification = self.director_notifications.pop(0)
                await self._send_director_notification(notification)
            
        except Exception as e:
            logger.error(f"Error processing director notifications: {e}")
    
    async def _notify_director(self, notification: Dict[str, Any]):
        """Notify Director of important events."""
        try:
            # Add to notification queue
            self.director_notifications.append(notification)
            
            # Log to database
            await self.integration.log_system_event(
                "INFO", "AUTONOMOUS_SYSTEM_MANAGER",
                f"Director notification: {notification.get('type')}",
                notification
            )
            
        except Exception as e:
            logger.error(f"Error notifying Director: {e}")
    
    async def _send_director_notification(self, notification: Dict[str, Any]):
        """Send notification to Director."""
        try:
            # This would integrate with actual Director notification system
            logger.info(f"📢 Director notification: {notification.get('type')} - {notification.get('severity', 'info')}")
            
        except Exception as e:
            logger.error(f"Error sending director notification: {e}")
    
    # Director interface methods
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for Director."""
        try:
            # Get current health status
            if self.health_history:
                current_health = self.health_history[-1]
            else:
                current_health = SystemStatus(
                    mode=self.current_mode,
                    governor_active=False,
                    architect_active=False,
                    bridge_active=False,
                    overall_health=0.0,
                    autonomy_level=0.0,
                    last_update=time.time(),
                    metrics=self.performance_metrics
                )
            
            return {
                "system_active": self.system_active,
                "current_mode": self.current_mode.value,
                "emergency_mode": self.emergency_mode,
                "overall_health": current_health.overall_health,
                "autonomy_level": current_health.autonomy_level,
                "governor_status": self.governor_status,
                "architect_status": self.architect_status,
                "bridge_status": self.bridge_status,
                "performance_metrics": self.performance_metrics,
                "health_history_length": len(self.health_history),
                "pending_notifications": len(self.director_notifications),
                "last_update": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    async def request_director_action(self, action_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Request action from Director."""
        try:
            # Add to request queue
            request = {
                "action_type": action_type,
                "parameters": parameters,
                "timestamp": time.time(),
                "request_id": f"req_{int(time.time() * 1000)}"
            }
            
            self.director_requests.append(request)
            
            # Notify Director
            await self._notify_director({
                "type": "director_action_request",
                "action_type": action_type,
                "parameters": parameters,
                "request_id": request["request_id"],
                "timestamp": time.time()
            })
            
            return {"status": "request_sent", "request_id": request["request_id"]}
            
        except Exception as e:
            logger.error(f"Error requesting director action: {e}")
            return {"error": str(e)}
    
    async def execute_director_command(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command from Director."""
        try:
            logger.info(f"🎯 Executing Director command: {command}")
            
            if command == "switch_mode":
                new_mode = SystemMode(parameters.get("mode", "directed"))
                await self.switch_mode(new_mode)
                return {"status": "success", "new_mode": new_mode.value}
            
            elif command == "start_autonomous":
                await self.start_autonomous_system(SystemMode.AUTONOMOUS)
                return {"status": "success", "mode": "autonomous"}
            
            elif command == "stop_autonomous":
                await self.stop_autonomous_system()
                return {"status": "success", "mode": "stopped"}
            
            elif command == "emergency_mode":
                await self.switch_mode(SystemMode.EMERGENCY)
                return {"status": "success", "mode": "emergency"}
            
            elif command == "get_status":
                status = await self.get_system_status()
                return {"status": "success", "data": status}
            
            else:
                return {"status": "error", "message": f"Unknown command: {command}"}
            
        except Exception as e:
            logger.error(f"Error executing director command: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_autonomy_summary(self) -> Dict[str, Any]:
        """Get summary of autonomy capabilities and current state."""
        return {
            "current_mode": self.current_mode.value,
            "autonomy_level": self._calculate_autonomy_level(),
            "governor_autonomy": {
                "active": self.governor_status.get("autonomous_cycle_active", False),
                "decisions_made": self.governor_status.get("decisions_made", 0),
                "success_rate": self.governor_status.get("successful_decisions", 0) / max(1, self.governor_status.get("decisions_made", 1))
            },
            "architect_autonomy": {
                "active": self.architect_status.get("autonomous_cycle_active", False),
                "evolutions_made": self.architect_status.get("evolutions_made", 0),
                "success_rate": self.architect_status.get("successful_evolutions", 0) / max(1, self.architect_status.get("evolutions_made", 1))
            },
            "collaboration": {
                "active": self.bridge_status.get("communication_active", False),
                "collaborative_decisions": self.bridge_status.get("collaborative_decisions", 0),
                "messages_exchanged": self.bridge_status.get("messages_exchanged", 0)
            },
            "overall_health": self._calculate_overall_health(),
            "performance_metrics": self.performance_metrics
        }

# Global autonomous system manager instance
autonomous_system_manager = AutonomousSystemManager()

async def start_autonomous_system(mode: str = "autonomous"):
    """Start the autonomous system."""
    system_mode = SystemMode(mode)
    await autonomous_system_manager.start_autonomous_system(system_mode)

async def stop_autonomous_system():
    """Stop the autonomous system."""
    await autonomous_system_manager.stop_autonomous_system()

async def switch_system_mode(mode: str):
    """Switch system mode."""
    system_mode = SystemMode(mode)
    await autonomous_system_manager.switch_mode(system_mode)

async def get_autonomous_system_status():
    """Get autonomous system status."""
    return await autonomous_system_manager.get_system_status()

async def execute_director_command(command: str, parameters: Dict[str, Any] = None):
    """Execute a Director command."""
    if parameters is None:
        parameters = {}
    return await autonomous_system_manager.execute_director_command(command, parameters)

def get_autonomy_summary():
    """Get autonomy summary."""
    return autonomous_system_manager.get_autonomy_summary()
