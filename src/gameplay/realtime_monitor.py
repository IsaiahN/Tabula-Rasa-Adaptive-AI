"""
Real-time Gameplay Monitoring System

Monitors gameplay in real-time and applies automatic fixes.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

@dataclass
class GameplayEvent:
    """Represents a gameplay event."""
    timestamp: datetime
    event_type: str
    data: Dict[str, Any]
    severity: str
    auto_fixed: bool = False

class RealTimeGameplayMonitor:
    """Real-time gameplay monitoring and auto-fixing system."""
    
    def __init__(self, check_interval: float = 1.0):
        self.check_interval = check_interval
        self.is_monitoring = False
        self.events = []
        self.monitoring_callbacks = []
        self.auto_fix_enabled = True
        
        # Import the automation systems
        from .error_automation import gameplay_automation
        from .action_corrector import action_corrector
        
        self.error_automation = gameplay_automation
        self.action_corrector = action_corrector
    
    async def start_monitoring(self, game_state_callback: Callable[[], Dict[str, Any]]):
        """Start real-time monitoring."""
        self.is_monitoring = True
        logger.info(" Starting real-time gameplay monitoring...")
        
        while self.is_monitoring:
            try:
                # Get current game state
                game_state = game_state_callback()
                
                # Process monitoring cycle
                await self._monitoring_cycle(game_state)
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {e}")
                await asyncio.sleep(5)  # Wait 5 seconds before retrying
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.is_monitoring = False
        logger.info(" Stopped real-time gameplay monitoring")
    
    async def _monitoring_cycle(self, game_state: Dict[str, Any]):
        """Process a single monitoring cycle."""
        try:
            # Get recent action history
            action_history = game_state.get('action_history', [])
            frame_data = game_state.get('frame_data')
            api_responses = game_state.get('api_responses', [])
            
            # Process gameplay errors
            if self.auto_fix_enabled:
                error_result = await self.error_automation.process_gameplay_cycle(
                    game_state, action_history, frame_data, api_responses
                )
                
                # Log significant events
                if error_result['errors_detected'] > 0:
                    self._log_event("error_detected", {
                        "errors": error_result['errors'],
                        "fixes": error_result['fixes']
                    }, "warning")
                
                if error_result['fixes_applied'] > 0:
                    self._log_event("auto_fix_applied", {
                        "fixes": error_result['fixes']
                    }, "info", auto_fixed=True)
            
            # Monitor system health
            health_status = self._monitor_system_health(game_state)
            if health_status['status'] != 'healthy':
                self._log_event("health_warning", health_status, "warning")
            
            # Monitor performance
            performance_status = self._monitor_performance(game_state)
            if performance_status['issues']:
                self._log_event("performance_issue", performance_status, "info")
            
            # Notify callbacks
            for callback in self.monitoring_callbacks:
                try:
                    await callback(game_state, self.events[-10:])  # Last 10 events
                except Exception as e:
                    logger.error(f"Error in monitoring callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")
            self._log_event("monitoring_error", {"error": str(e)}, "critical")
    
    def _log_event(self, event_type: str, data: Dict[str, Any], severity: str, auto_fixed: bool = False):
        """Log a gameplay event."""
        event = GameplayEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            data=data,
            severity=severity,
            auto_fixed=auto_fixed
        )
        
        self.events.append(event)
        
        # Keep only last 1000 events
        if len(self.events) > 1000:
            self.events = self.events[-1000:]
        
        # Log based on severity
        if severity == "critical":
            logger.critical(f" {event_type}: {data}")
        elif severity == "warning":
            logger.warning(f" {event_type}: {data}")
        elif severity == "info":
            logger.info(f"ℹ {event_type}: {data}")
    
    def _monitor_system_health(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor overall system health."""
        health_issues = []
        
        # Check action success rate
        action_history = game_state.get('action_history', [])
        if len(action_history) > 10:
            recent_actions = action_history[-10:]
            success_rate = sum(1 for a in recent_actions if a.get('success', False)) / len(recent_actions)
            
            if success_rate < 0.3:
                health_issues.append(f"Low success rate: {success_rate:.2f}")
        
        # Check for stagnation
        if len(action_history) > 20:
            recent_scores = [a.get('score_after', 0) for a in action_history[-20:]]
            if len(set(recent_scores)) == 1 and recent_scores[0] == 0:
                health_issues.append("No score progress in 20 actions")
        
        # Check API error rate
        api_responses = game_state.get('api_responses', [])
        if api_responses:
            error_rate = sum(1 for r in api_responses if 'error' in r) / len(api_responses)
            if error_rate > 0.2:
                health_issues.append(f"High API error rate: {error_rate:.2f}")
        
        return {
            "status": "healthy" if not health_issues else "warning",
            "issues": health_issues,
            "timestamp": datetime.now().isoformat()
        }
    
    def _monitor_performance(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor performance metrics."""
        performance_issues = []
        
        # Check action frequency
        action_history = game_state.get('action_history', [])
        if len(action_history) > 5:
            recent_actions = action_history[-5:]
            time_span = (recent_actions[-1].get('timestamp', 0) - 
                        recent_actions[0].get('timestamp', 0))
            
            if time_span > 0:
                actions_per_second = len(recent_actions) / time_span
                if actions_per_second < 0.1:  # Less than 1 action per 10 seconds
                    performance_issues.append(f"Slow action rate: {actions_per_second:.2f} actions/sec")
        
        # Check memory usage (if available)
        if 'memory_usage' in game_state:
            memory_usage = game_state['memory_usage']
            if memory_usage > 0.8:  # 80% memory usage
                performance_issues.append(f"High memory usage: {memory_usage:.2f}")
        
        return {
            "issues": performance_issues,
            "timestamp": datetime.now().isoformat()
        }
    
    def add_monitoring_callback(self, callback: Callable[[Dict[str, Any], List[GameplayEvent]], None]):
        """Add a callback for monitoring events."""
        self.monitoring_callbacks.append(callback)
    
    def get_recent_events(self, count: int = 10) -> List[GameplayEvent]:
        """Get recent gameplay events."""
        return self.events[-count:]
    
    def get_event_summary(self) -> Dict[str, Any]:
        """Get summary of all events."""
        if not self.events:
            return {"total_events": 0}
        
        # Count events by type and severity
        event_counts = {}
        severity_counts = {}
        auto_fixed_count = 0
        
        for event in self.events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
            if event.auto_fixed:
                auto_fixed_count += 1
        
        return {
            "total_events": len(self.events),
            "event_types": event_counts,
            "severity_counts": severity_counts,
            "auto_fixed_count": auto_fixed_count,
            "auto_fix_rate": auto_fixed_count / len(self.events) if self.events else 0,
            "recent_events": [self._event_to_dict(e) for e in self.events[-5:]]
        }
    
    def _event_to_dict(self, event: GameplayEvent) -> Dict[str, Any]:
        """Convert GameplayEvent to dictionary."""
        return {
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "data": event.data,
            "severity": event.severity,
            "auto_fixed": event.auto_fixed
        }
    
    def enable_auto_fix(self):
        """Enable automatic error fixing."""
        self.auto_fix_enabled = True
        logger.info(" Auto-fix enabled")
    
    def disable_auto_fix(self):
        """Disable automatic error fixing."""
        self.auto_fix_enabled = False
        logger.info(" Auto-fix disabled")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            "is_monitoring": self.is_monitoring,
            "auto_fix_enabled": self.auto_fix_enabled,
            "check_interval": self.check_interval,
            "total_events": len(self.events),
            "callbacks_registered": len(self.monitoring_callbacks)
        }

# Global monitoring instance
realtime_monitor = RealTimeGameplayMonitor()

async def start_gameplay_monitoring(game_state_callback: Callable[[], Dict[str, Any]]):
    """Start real-time gameplay monitoring."""
    await realtime_monitor.start_monitoring(game_state_callback)

def stop_gameplay_monitoring():
    """Stop real-time gameplay monitoring."""
    realtime_monitor.stop_monitoring()

def get_gameplay_events(count: int = 10) -> List[GameplayEvent]:
    """Get recent gameplay events."""
    return realtime_monitor.get_recent_events(count)

def get_monitoring_status() -> Dict[str, Any]:
    """Get monitoring status."""
    return realtime_monitor.get_monitoring_status()

def enable_auto_fix():
    """Enable automatic error fixing."""
    realtime_monitor.enable_auto_fix()

def disable_auto_fix():
    """Disable automatic error fixing."""
    realtime_monitor.disable_auto_fix()
