"""
Alert Manager

Manages alerts and notifications for the monitoring system.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert definition."""
    id: str
    level: AlertLevel
    message: str
    source: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class AlertManager:
    """
    Manages alerts and notifications.
    """
    
    def __init__(self):
        self._alerts: List[Alert] = []
        self._callbacks: List[Callable[[Alert], None]] = []
        self._alert_id_counter = 0
        self._lock = threading.Lock()
        
        # Alert rules
        self._rules: List[Dict[str, Any]] = []
        self._suppressed_alerts: set = set()
    
    def add_rule(self, 
                 condition: Callable[[Dict[str, Any]], bool],
                 level: AlertLevel,
                 message_template: str,
                 source: str) -> None:
        """Add an alert rule."""
        rule = {
            'condition': condition,
            'level': level,
            'message_template': message_template,
            'source': source
        }
        self._rules.append(rule)
    
    def check_conditions(self, data: Dict[str, Any]) -> List[Alert]:
        """Check all alert conditions against data."""
        new_alerts = []
        
        with self._lock:
            for rule in self._rules:
                try:
                    if rule['condition'](data):
                        alert_id = f"{rule['source']}_{self._alert_id_counter}"
                        self._alert_id_counter += 1
                        
                        message = rule['message_template'].format(**data)
                        alert = Alert(
                            id=alert_id,
                            level=rule['level'],
                            message=message,
                            source=rule['source'],
                            timestamp=datetime.now(),
                            metadata=data
                        )
                        
                        # Check if alert is suppressed
                        if alert_id not in self._suppressed_alerts:
                            self._alerts.append(alert)
                            new_alerts.append(alert)
                            
                            # Notify callbacks
                            for callback in self._callbacks:
                                try:
                                    callback(alert)
                                except Exception as e:
                                    print(f"Error in alert callback: {e}")
                except Exception as e:
                    print(f"Error checking alert rule: {e}")
        
        return new_alerts
    
    def create_alert(self, 
                    level: AlertLevel,
                    message: str,
                    source: str,
                    metadata: Optional[Dict[str, Any]] = None) -> Alert:
        """Create a manual alert."""
        with self._lock:
            alert_id = f"{source}_{self._alert_id_counter}"
            self._alert_id_counter += 1
            
            alert = Alert(
                id=alert_id,
                level=level,
                message=message,
                source=source,
                timestamp=datetime.now(),
                metadata=metadata
            )
            
            self._alerts.append(alert)
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    print(f"Error in alert callback: {e}")
            
            return alert
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    return True
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        with self._lock:
            return [alert for alert in self._alerts if not alert.resolved]
    
    def get_alerts_by_level(self, level: AlertLevel) -> List[Alert]:
        """Get alerts by severity level."""
        with self._lock:
            return [alert for alert in self._alerts if alert.level == level]
    
    def get_alerts_by_source(self, source: str) -> List[Alert]:
        """Get alerts by source."""
        with self._lock:
            return [alert for alert in self._alerts if alert.source == source]
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get alerts from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        with self._lock:
            return [alert for alert in self._alerts if alert.timestamp >= cutoff_time]
    
    def suppress_alert(self, alert_id: str) -> None:
        """Suppress an alert (prevent it from being created)."""
        self._suppressed_alerts.add(alert_id)
    
    def unsuppress_alert(self, alert_id: str) -> None:
        """Remove alert suppression."""
        self._suppressed_alerts.discard(alert_id)
    
    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add a callback for new alerts."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[Alert], None]) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def clear_old_alerts(self, days: int = 7) -> int:
        """Clear alerts older than specified days."""
        cutoff_time = datetime.now() - timedelta(days=days)
        with self._lock:
            old_alerts = [alert for alert in self._alerts if alert.timestamp < cutoff_time]
            self._alerts = [alert for alert in self._alerts if alert.timestamp >= cutoff_time]
            return len(old_alerts)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        with self._lock:
            active_alerts = [alert for alert in self._alerts if not alert.resolved]
            resolved_alerts = [alert for alert in self._alerts if alert.resolved]
            
            level_counts = {}
            for level in AlertLevel:
                level_counts[level.value] = len([alert for alert in active_alerts if alert.level == level])
            
            return {
                'total_alerts': len(self._alerts),
                'active_alerts': len(active_alerts),
                'resolved_alerts': len(resolved_alerts),
                'level_counts': level_counts,
                'suppressed_count': len(self._suppressed_alerts)
            }
    
    def create_system_alert_rules(self) -> None:
        """Create default system alert rules."""
        # High CPU usage
        self.add_rule(
            condition=lambda data: data.get('cpu_usage', 0) > 90,
            level=AlertLevel.CRITICAL,
            message_template="CPU usage critical: {cpu_usage:.1f}%",
            source="system_monitor"
        )
        
        # High memory usage
        self.add_rule(
            condition=lambda data: data.get('memory_usage', 0) > 90,
            level=AlertLevel.CRITICAL,
            message_template="Memory usage critical: {memory_usage:.1f}%",
            source="system_monitor"
        )
        
        # Low win rate
        self.add_rule(
            condition=lambda data: data.get('win_rate', 1) < 0.1 and data.get('games_completed', 0) > 10,
            level=AlertLevel.WARNING,
            message_template="Low win rate: {win_rate:.1%} after {games_completed} games",
            source="training_monitor"
        )
        
        # Training session error
        self.add_rule(
            condition=lambda data: data.get('status') == 'error',
            level=AlertLevel.ERROR,
            message_template="Training session error: {error_message}",
            source="training_monitor"
        )
