#!/usr/bin/env python3
"""
Enhanced Alerting System

Provides sophisticated alerting with intelligent routing, escalation,
suppression, and integration with multiple notification channels.
"""

import logging
import time
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum
import threading
import asyncio

from ..core.caching_system import UnifiedCachingSystem, CacheConfig
from ..core.unified_performance_monitor import AlertLevel


class AlertChannel(Enum):
    """Alert notification channels."""
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DISCORD = "discord"
    CONSOLE = "console"
    DATABASE = "database"


class AlertState(Enum):
    """Alert states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"


class EscalationLevel(Enum):
    """Escalation levels."""
    LEVEL_1 = "level_1"  # Immediate
    LEVEL_2 = "level_2"  # 5 minutes
    LEVEL_3 = "level_3"  # 15 minutes
    LEVEL_4 = "level_4"  # 30 minutes
    LEVEL_5 = "level_5"  # 1 hour


@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertLevel
    channels: List[AlertChannel]
    escalation_level: EscalationLevel
    suppression_rules: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Enhanced alert with additional metadata."""
    alert_id: str
    rule_id: str
    severity: AlertLevel
    title: str
    message: str
    source: str
    timestamp: datetime
    state: AlertState = AlertState.ACTIVE
    escalation_level: EscalationLevel = EscalationLevel.LEVEL_1
    channels: List[AlertChannel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    suppression_reason: Optional[str] = None


@dataclass
class AlertGroup:
    """Group of related alerts."""
    group_id: str
    alerts: List[Alert]
    created_at: datetime
    last_updated: datetime
    group_type: str = "related"
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertSuppressor:
    """Handles alert suppression logic."""
    
    def __init__(self):
        self.suppression_rules = []
        self.active_suppressions = {}
        
    def add_suppression_rule(self, rule: Dict[str, Any]) -> None:
        """Add a suppression rule."""
        self.suppression_rules.append(rule)
    
    def should_suppress(self, alert: Alert) -> Tuple[bool, Optional[str]]:
        """Check if an alert should be suppressed."""
        for rule in self.suppression_rules:
            if self._matches_suppression_rule(alert, rule):
                return True, rule.get('reason', 'Suppressed by rule')
        
        return False, None
    
    def _matches_suppression_rule(self, alert: Alert, rule: Dict[str, Any]) -> bool:
        """Check if alert matches a suppression rule."""
        # Check source
        if 'source' in rule and alert.source != rule['source']:
            return False
        
        # Check severity
        if 'max_severity' in rule and alert.severity.value not in self._get_severity_levels(rule['max_severity']):
            return False
        
        # Check time window
        if 'time_window_minutes' in rule:
            window_start = datetime.now() - timedelta(minutes=rule['time_window_minutes'])
            if alert.timestamp < window_start:
                return False
        
        # Check frequency
        if 'max_frequency' in rule and 'time_window_minutes' in rule:
            recent_alerts = self._count_recent_alerts(alert.source, rule['time_window_minutes'])
            if recent_alerts >= rule['max_frequency']:
                return False
        
        return True
    
    def _get_severity_levels(self, max_severity: str) -> List[str]:
        """Get list of severity levels up to max severity."""
        severity_order = ['info', 'warning', 'error', 'critical']
        max_index = severity_order.index(max_severity) if max_severity in severity_order else 0
        return severity_order[:max_index + 1]
    
    def _count_recent_alerts(self, source: str, window_minutes: int) -> int:
        """Count recent alerts from a source."""
        # This would typically query a database or cache
        # For now, return 0
        return 0


class AlertEscalator:
    """Handles alert escalation logic."""
    
    def __init__(self):
        self.escalation_rules = {}
        self.escalation_timers = {}
        
    def add_escalation_rule(self, severity: AlertLevel, 
                          escalation_schedule: List[Tuple[int, EscalationLevel]]) -> None:
        """Add escalation rule for a severity level."""
        self.escalation_rules[severity] = escalation_schedule
    
    def should_escalate(self, alert: Alert) -> Tuple[bool, Optional[EscalationLevel]]:
        """Check if an alert should be escalated."""
        if alert.state != AlertState.ACTIVE:
            return False, None
        
        # Check if escalation schedule exists for this severity
        if alert.severity not in self.escalation_rules:
            return False, None
        
        schedule = self.escalation_rules[alert.severity]
        time_since_creation = (datetime.now() - alert.timestamp).total_seconds() / 60
        
        # Find the appropriate escalation level
        for minutes, level in schedule:
            if time_since_creation >= minutes and alert.escalation_level.value < level.value:
                return True, level
        
        return False, None
    
    def escalate_alert(self, alert: Alert, new_level: EscalationLevel) -> Alert:
        """Escalate an alert to a new level."""
        alert.escalation_level = new_level
        alert.metadata['escalated_at'] = datetime.now()
        alert.metadata['escalation_count'] = alert.metadata.get('escalation_count', 0) + 1
        return alert


class AlertGrouper:
    """Groups related alerts together."""
    
    def __init__(self):
        self.grouping_rules = []
        self.active_groups = {}
        
    def add_grouping_rule(self, rule: Dict[str, Any]) -> None:
        """Add a grouping rule."""
        self.grouping_rules.append(rule)
    
    def should_group(self, alert: Alert, existing_alerts: List[Alert]) -> Tuple[bool, Optional[str]]:
        """Check if an alert should be grouped with existing alerts."""
        for rule in self.grouping_rules:
            group_id = self._check_grouping_rule(alert, existing_alerts, rule)
            if group_id:
                return True, group_id
        
        return False, None
    
    def _check_grouping_rule(self, alert: Alert, existing_alerts: List[Alert], 
                           rule: Dict[str, Any]) -> Optional[str]:
        """Check if alert matches a grouping rule."""
        # Check source grouping
        if rule.get('group_by_source', False):
            source_alerts = [a for a in existing_alerts if a.source == alert.source]
            if source_alerts:
                # Check time window
                time_window = rule.get('time_window_minutes', 5)
                recent_alerts = [a for a in source_alerts 
                               if (alert.timestamp - a.timestamp).total_seconds() / 60 <= time_window]
                if recent_alerts:
                    return f"source_{alert.source}_{int(alert.timestamp.timestamp())}"
        
        # Check severity grouping
        if rule.get('group_by_severity', False):
            severity_alerts = [a for a in existing_alerts if a.severity == alert.severity]
            if severity_alerts:
                time_window = rule.get('time_window_minutes', 10)
                recent_alerts = [a for a in severity_alerts 
                               if (alert.timestamp - a.timestamp).total_seconds() / 60 <= time_window]
                if recent_alerts:
                    return f"severity_{alert.severity.value}_{int(alert.timestamp.timestamp())}"
        
        return None


class NotificationChannel:
    """Base class for notification channels."""
    
    def __init__(self, channel_type: AlertChannel, config: Dict[str, Any]):
        self.channel_type = channel_type
        self.config = config
        self.enabled = True
        
    def send_alert(self, alert: Alert) -> bool:
        """Send alert through this channel."""
        raise NotImplementedError
    
    def test_connection(self) -> bool:
        """Test channel connection."""
        return True


class LogChannel(NotificationChannel):
    """Log-based notification channel."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AlertChannel.LOG, config)
        self.logger = logging.getLogger(f"alerting.{self.channel_type.value}")
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert to log."""
        try:
            log_level = {
                AlertLevel.INFO: logging.INFO,
                AlertLevel.WARNING: logging.WARNING,
                AlertLevel.ERROR: logging.ERROR,
                AlertLevel.CRITICAL: logging.CRITICAL
            }.get(alert.severity, logging.INFO)
            
            self.logger.log(log_level, f"ALERT [{alert.alert_id}] {alert.title}: {alert.message}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send alert to log: {e}")
            return False


class ConsoleChannel(NotificationChannel):
    """Console-based notification channel."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AlertChannel.CONSOLE, config)
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert to console."""
        try:
            severity_color = {
                AlertLevel.INFO: "\033[94m",      # Blue
                AlertLevel.WARNING: "\033[93m",   # Yellow
                AlertLevel.ERROR: "\033[91m",     # Red
                AlertLevel.CRITICAL: "\033[95m"   # Magenta
            }.get(alert.severity, "\033[0m")
            
            reset_color = "\033[0m"
            
            print(f"{severity_color}[{alert.severity.value.upper()}] {alert.title}{reset_color}")
            print(f"  Source: {alert.source}")
            print(f"  Time: {alert.timestamp}")
            print(f"  Message: {alert.message}")
            print(f"  ID: {alert.alert_id}")
            print("-" * 50)
            
            return True
        except Exception as e:
            print(f"Failed to send alert to console: {e}")
            return False


class WebhookChannel(NotificationChannel):
    """Webhook-based notification channel."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AlertChannel.WEBHOOK, config)
        self.webhook_url = config.get('url')
        self.headers = config.get('headers', {})
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        try:
            import requests
            
            payload = {
                'alert_id': alert.alert_id,
                'severity': alert.severity.value,
                'title': alert.title,
                'message': alert.message,
                'source': alert.source,
                'timestamp': alert.timestamp.isoformat(),
                'metadata': alert.metadata
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            return response.status_code == 200
        except Exception as e:
            print(f"Failed to send webhook alert: {e}")
            return False


class EnhancedAlertingSystem:
    """Main enhanced alerting system."""
    
    def __init__(self, cache_config: Optional[CacheConfig] = None):
        self.cache_config = cache_config or CacheConfig()
        self.cache = UnifiedCachingSystem(self.cache_config)
        
        # Core components
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_groups: Dict[str, AlertGroup] = {}
        
        # Processing components
        self.suppressor = AlertSuppressor()
        self.escalator = AlertEscalator()
        self.grouper = AlertGrouper()
        
        # Notification channels
        self.channels: Dict[AlertChannel, NotificationChannel] = {}
        
        # Callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.group_callbacks: List[Callable[[AlertGroup], None]] = []
        
        # Threading
        self.lock = threading.Lock()
        self.escalation_thread = None
        self.running = False
        
        # Statistics
        self.alert_stats = {
            'total_alerts': 0,
            'active_alerts': 0,
            'resolved_alerts': 0,
            'suppressed_alerts': 0,
            'escalated_alerts': 0
        }
        
        self._initialized = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize the enhanced alerting system."""
        try:
            # Set up default channels
            self._setup_default_channels()
            
            # Set up default rules
            self._setup_default_rules()
            
            # Start escalation thread
            self.running = True
            self.escalation_thread = threading.Thread(target=self._escalation_loop, daemon=True)
            self.escalation_thread.start()
            
            self._initialized = True
            self.logger.info("Enhanced alerting system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize alerting system: {e}")
            raise
    
    def _setup_default_channels(self) -> None:
        """Set up default notification channels."""
        # Log channel
        self.channels[AlertChannel.LOG] = LogChannel({})
        
        # Console channel
        self.channels[AlertChannel.CONSOLE] = ConsoleChannel({})
    
    def _setup_default_rules(self) -> None:
        """Set up default alert rules."""
        # High CPU usage rule
        self.add_alert_rule(AlertRule(
            rule_id="high_cpu_usage",
            name="High CPU Usage",
            condition=lambda data: data.get('cpu_usage', 0) > 0.9,
            severity=AlertLevel.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
            escalation_level=EscalationLevel.LEVEL_1
        ))
        
        # High memory usage rule
        self.add_alert_rule(AlertRule(
            rule_id="high_memory_usage",
            name="High Memory Usage",
            condition=lambda data: data.get('memory_usage', 0) > 0.9,
            severity=AlertLevel.CRITICAL,
            channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
            escalation_level=EscalationLevel.LEVEL_1
        ))
        
        # Error rate spike rule
        self.add_alert_rule(AlertRule(
            rule_id="error_rate_spike",
            name="Error Rate Spike",
            condition=lambda data: data.get('error_rate', 0) > 0.1,
            severity=AlertLevel.ERROR,
            channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
            escalation_level=EscalationLevel.LEVEL_2
        ))
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        with self.lock:
            self.alert_rules[rule.rule_id] = rule
            self.logger.info(f"Added alert rule: {rule.name}")
    
    def add_notification_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        self.channels[channel.channel_type] = channel
        self.logger.info(f"Added notification channel: {channel.channel_type.value}")
    
    def process_data(self, data: Dict[str, Any], source: str = "unknown") -> List[Alert]:
        """Process data and generate alerts based on rules."""
        alerts = []
        
        with self.lock:
            for rule_id, rule in self.alert_rules.items():
                try:
                    if rule.condition(data):
                        alert = self._create_alert(rule, data, source)
                        
                        # Check suppression
                        should_suppress, reason = self.suppressor.should_suppress(alert)
                        if should_suppress:
                            alert.state = AlertState.SUPPRESSED
                            alert.suppression_reason = reason
                            self.alert_stats['suppressed_alerts'] += 1
                        else:
                            # Send alert
                            self._send_alert(alert)
                            alerts.append(alert)
                        
                        # Store alert
                        self.active_alerts[alert.alert_id] = alert
                        self.alert_stats['total_alerts'] += 1
                        self.alert_stats['active_alerts'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Error processing rule {rule_id}: {e}")
        
        return alerts
    
    def _create_alert(self, rule: AlertRule, data: Dict[str, Any], source: str) -> Alert:
        """Create an alert from a rule and data."""
        alert_id = f"{rule.rule_id}_{int(time.time())}"
        
        # Generate title and message
        title = f"{rule.name} Alert"
        message = f"Condition triggered in {source}: {data}"
        
        return Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            severity=rule.severity,
            title=title,
            message=message,
            source=source,
            timestamp=datetime.now(),
            channels=rule.channels,
            metadata={
                'data': data,
                'rule_name': rule.name
            }
        )
    
    def _send_alert(self, alert: Alert) -> None:
        """Send alert through configured channels."""
        for channel_type in alert.channels:
            if channel_type in self.channels:
                channel = self.channels[channel_type]
                try:
                    success = channel.send_alert(alert)
                    if success:
                        self.logger.debug(f"Alert {alert.alert_id} sent via {channel_type.value}")
                    else:
                        self.logger.warning(f"Failed to send alert {alert.alert_id} via {channel_type.value}")
                except Exception as e:
                    self.logger.error(f"Error sending alert via {channel_type.value}: {e}")
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.state = AlertState.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                
                self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
        
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.state = AlertState.RESOLVED
                alert.resolved_at = datetime.now()
                
                self.alert_stats['active_alerts'] -= 1
                self.alert_stats['resolved_alerts'] += 1
                
                self.logger.info(f"Alert {alert_id} resolved")
                return True
        
        return False
    
    def _escalation_loop(self) -> None:
        """Background thread for handling alert escalation."""
        while self.running:
            try:
                with self.lock:
                    for alert in self.active_alerts.values():
                        if alert.state == AlertState.ACTIVE:
                            should_escalate, new_level = self.escalator.should_escalate(alert)
                            if should_escalate:
                                self.escalator.escalate_alert(alert, new_level)
                                self.alert_stats['escalated_alerts'] += 1
                                
                                # Re-send escalated alert
                                self._send_alert(alert)
                
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in escalation loop: {e}")
                time.sleep(60)
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add a callback for alerts."""
        self.alert_callbacks.append(callback)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alerts."""
        with self.lock:
            return {
                'stats': self.alert_stats.copy(),
                'active_alerts': len([a for a in self.active_alerts.values() if a.state == AlertState.ACTIVE]),
                'acknowledged_alerts': len([a for a in self.active_alerts.values() if a.state == AlertState.ACKNOWLEDGED]),
                'resolved_alerts': len([a for a in self.active_alerts.values() if a.state == AlertState.RESOLVED]),
                'suppressed_alerts': len([a for a in self.active_alerts.values() if a.state == AlertState.SUPPRESSED]),
                'channels': list(self.channels.keys()),
                'rules': list(self.alert_rules.keys())
            }
    
    def get_recent_alerts(self, limit: int = 50) -> List[Alert]:
        """Get recent alerts."""
        with self.lock:
            alerts = list(self.active_alerts.values())
            alerts.sort(key=lambda x: x.timestamp, reverse=True)
            return alerts[:limit]
    
    def cleanup_old_alerts(self, max_age_hours: int = 168) -> None:  # 1 week default
        """Clean up old alerts."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self.lock:
            old_alerts = [aid for aid, alert in self.active_alerts.items() 
                         if alert.timestamp < cutoff_time and alert.state == AlertState.RESOLVED]
            
            for alert_id in old_alerts:
                del self.active_alerts[alert_id]
            
            self.logger.info(f"Cleaned up {len(old_alerts)} old alerts")


# Global instance
_alerting_system = None

def get_alerting_system() -> EnhancedAlertingSystem:
    """Get the global alerting system instance."""
    global _alerting_system
    if _alerting_system is None:
        _alerting_system = EnhancedAlertingSystem()
        _alerting_system.initialize()
    return _alerting_system
