#!/usr/bin/env python3
"""
Integrity Monitoring System - Continuous system integrity monitoring.

This module provides:
- System health monitoring
- Component integrity verification
- Security event detection
- Anomaly detection
- Performance monitoring
"""

import logging
import time
import threading
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import json
import psutil
import os

logger = logging.getLogger(__name__)


class IntegrityStatus(Enum):
    """System integrity status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    COMPROMISED = "compromised"


class SecurityEventType(Enum):
    """Types of security events."""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    SYSTEM_INTRUSION = "system_intrusion"
    MALWARE_DETECTED = "malware_detected"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    PERFORMANCE_ANOMALY = "performance_anomaly"
    MEMORY_ANOMALY = "memory_anomaly"
    NETWORK_ANOMALY = "network_anomaly"
    FILE_SYSTEM_ANOMALY = "file_system_anomaly"


@dataclass
class SecurityEvent:
    """Security event information."""
    event_id: str
    event_type: SecurityEventType
    severity: IntegrityStatus
    timestamp: datetime
    description: str
    source: str
    details: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class IntegrityMetrics:
    """System integrity metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    process_count: int
    file_descriptors: int
    system_load: float
    uptime: float
    security_events: int
    integrity_score: float


class IntegrityMonitor:
    """
    Comprehensive system integrity monitoring.
    
    Monitors:
    - System resources (CPU, memory, disk)
    - Process health
    - Network activity
    - File system integrity
    - Security events
    - Performance metrics
    """
    
    def __init__(self, monitoring_interval: float = 5.0, alert_thresholds: Optional[Dict] = None):
        self.monitoring_interval = monitoring_interval
        self.alert_thresholds = alert_thresholds or self._get_default_thresholds()
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.metrics_history = deque(maxlen=1000)
        self.security_events = deque(maxlen=1000)
        self.integrity_callbacks = []
        
        # Baseline metrics
        self.baseline_metrics = None
        self.anomaly_detector = None
        
        # Security monitoring
        self.suspicious_processes = set()
        self.blocked_ips = set()
        self.file_integrity_hashes = {}
        
        # Initialize monitoring
        self._initialize_monitoring()
    
    def _get_default_thresholds(self) -> Dict[str, float]:
        """Get default alert thresholds."""
        return {
            'cpu_usage': 80.0,  # 80% CPU usage
            'memory_usage': 85.0,  # 85% memory usage
            'disk_usage': 90.0,  # 90% disk usage
            'process_count': 1000,  # 1000 processes
            'file_descriptors': 10000,  # 10000 file descriptors
            'system_load': 5.0,  # Load average of 5.0
            'integrity_score': 0.7,  # 70% integrity score
        }
    
    def _initialize_monitoring(self):
        """Initialize monitoring components."""
        # Initialize anomaly detector
        self.anomaly_detector = SystemAnomalyDetector()
        
        # Initialize baseline metrics
        self._establish_baseline()
    
    def start_monitoring(self):
        """Start continuous integrity monitoring."""
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Integrity monitoring started")
    
    def stop_monitoring(self):
        """Stop integrity monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Integrity monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for anomalies
                anomalies = self._detect_anomalies(metrics)
                
                # Check for security events
                security_events = self._detect_security_events(metrics)
                
                # Process events
                for event in security_events:
                    self._process_security_event(event)
                
                # Update integrity status
                integrity_status = self._calculate_integrity_status(metrics, anomalies)
                
                # Notify callbacks
                self._notify_callbacks(metrics, integrity_status, security_events)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
            
            time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> IntegrityMetrics:
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network_io = psutil.net_io_counters()._asdict()
            
            # Process count
            process_count = len(psutil.pids())
            
            # File descriptors
            try:
                file_descriptors = len(os.listdir('/proc/self/fd'))
            except:
                file_descriptors = 0
            
            # System load
            system_load = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
            
            # Uptime
            uptime = time.time() - psutil.boot_time()
            
            # Security events count
            security_events_count = len(self.security_events)
            
            # Calculate integrity score
            integrity_score = self._calculate_integrity_score(cpu_usage, memory_usage, disk_usage)
            
            return IntegrityMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                process_count=process_count,
                file_descriptors=file_descriptors,
                system_load=system_load,
                uptime=uptime,
                security_events=security_events_count,
                integrity_score=integrity_score
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            return IntegrityMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                process_count=0,
                file_descriptors=0,
                system_load=0.0,
                uptime=0.0,
                security_events=0,
                integrity_score=0.0
            )
    
    def _detect_anomalies(self, metrics: IntegrityMetrics) -> List[str]:
        """Detect anomalies in system metrics."""
        anomalies = []
        
        if not self.baseline_metrics:
            return anomalies
        
        # CPU anomaly
        if metrics.cpu_usage > self.baseline_metrics.cpu_usage * 2:
            anomalies.append("High CPU usage anomaly")
        
        # Memory anomaly
        if metrics.memory_usage > self.baseline_metrics.memory_usage * 1.5:
            anomalies.append("High memory usage anomaly")
        
        # Process count anomaly
        if metrics.process_count > self.baseline_metrics.process_count * 1.5:
            anomalies.append("High process count anomaly")
        
        # File descriptor anomaly
        if metrics.file_descriptors > self.baseline_metrics.file_descriptors * 2:
            anomalies.append("High file descriptor usage anomaly")
        
        # System load anomaly
        if metrics.system_load > self.baseline_metrics.system_load * 2:
            anomalies.append("High system load anomaly")
        
        return anomalies
    
    def _detect_security_events(self, metrics: IntegrityMetrics) -> List[SecurityEvent]:
        """Detect security events."""
        events = []
        
        # Check for suspicious processes
        suspicious_processes = self._detect_suspicious_processes()
        for process in suspicious_processes:
            event = SecurityEvent(
                event_id=f"proc_{int(time.time())}_{process['pid']}",
                event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                severity=IntegrityStatus.WARNING,
                timestamp=datetime.now(),
                description=f"Suspicious process detected: {process['name']}",
                source="process_monitor",
                details=process
            )
            events.append(event)
        
        # Check for performance anomalies
        if metrics.integrity_score < 0.5:
            event = SecurityEvent(
                event_id=f"perf_{int(time.time())}",
                event_type=SecurityEventType.PERFORMANCE_ANOMALY,
                severity=IntegrityStatus.DEGRADED,
                timestamp=datetime.now(),
                description="Performance anomaly detected",
                source="performance_monitor",
                details={"integrity_score": metrics.integrity_score}
            )
            events.append(event)
        
        # Check for memory anomalies
        if metrics.memory_usage > 90.0:
            event = SecurityEvent(
                event_id=f"mem_{int(time.time())}",
                event_type=SecurityEventType.MEMORY_ANOMALY,
                severity=IntegrityStatus.CRITICAL,
                timestamp=datetime.now(),
                description="Memory usage critical",
                source="memory_monitor",
                details={"memory_usage": metrics.memory_usage}
            )
            events.append(event)
        
        return events
    
    def _detect_suspicious_processes(self) -> List[Dict[str, Any]]:
        """Detect suspicious processes."""
        suspicious = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    proc_info = proc.info
                    
                    # Check for high resource usage
                    if (proc_info['cpu_percent'] > 50.0 or 
                        proc_info['memory_percent'] > 20.0):
                        
                        # Check if process is in suspicious list
                        if proc_info['name'] in self.suspicious_processes:
                            suspicious.append(proc_info)
                        
                        # Check for unusual process names
                        if self._is_suspicious_process_name(proc_info['name']):
                            suspicious.append(proc_info)
                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        
        except Exception as e:
            self.logger.error(f"Error detecting suspicious processes: {e}")
        
        return suspicious
    
    def _is_suspicious_process_name(self, name: str) -> bool:
        """Check if process name is suspicious."""
        suspicious_patterns = [
            'nc', 'netcat', 'ncat',
            'wget', 'curl',
            'python', 'perl', 'ruby',
            'sh', 'bash', 'zsh',
            'ssh', 'telnet',
            'ftp', 'tftp',
            'nmap', 'masscan',
            'hydra', 'john',
            'metasploit', 'msfconsole'
        ]
        
        return any(pattern in name.lower() for pattern in suspicious_patterns)
    
    def _calculate_integrity_score(self, cpu_usage: float, memory_usage: float, 
                                  disk_usage: float) -> float:
        """Calculate system integrity score."""
        # Normalize metrics to 0-1 scale
        cpu_score = max(0, 1 - cpu_usage / 100)
        memory_score = max(0, 1 - memory_usage / 100)
        disk_score = max(0, 1 - disk_usage / 100)
        
        # Weighted average
        integrity_score = (cpu_score * 0.4 + memory_score * 0.4 + disk_score * 0.2)
        
        return max(0, min(1, integrity_score))
    
    def _calculate_integrity_status(self, metrics: IntegrityMetrics, 
                                   anomalies: List[str]) -> IntegrityStatus:
        """Calculate overall system integrity status."""
        # Check critical thresholds
        if (metrics.cpu_usage > 95 or metrics.memory_usage > 95 or 
            metrics.disk_usage > 95):
            return IntegrityStatus.CRITICAL
        
        # Check for many anomalies
        if len(anomalies) > 5:
            return IntegrityStatus.COMPROMISED
        
        # Check integrity score
        if metrics.integrity_score < 0.3:
            return IntegrityStatus.CRITICAL
        elif metrics.integrity_score < 0.5:
            return IntegrityStatus.DEGRADED
        elif metrics.integrity_score < 0.7:
            return IntegrityStatus.WARNING
        
        return IntegrityStatus.HEALTHY
    
    def _establish_baseline(self):
        """Establish baseline metrics for anomaly detection."""
        try:
            # Collect metrics over a short period
            baseline_metrics = []
            for _ in range(5):
                metrics = self._collect_metrics()
                baseline_metrics.append(metrics)
                time.sleep(1)
            
            # Calculate average baseline
            if baseline_metrics:
                self.baseline_metrics = IntegrityMetrics(
                    timestamp=datetime.now(),
                    cpu_usage=sum(m.cpu_usage for m in baseline_metrics) / len(baseline_metrics),
                    memory_usage=sum(m.memory_usage for m in baseline_metrics) / len(baseline_metrics),
                    disk_usage=sum(m.disk_usage for m in baseline_metrics) / len(baseline_metrics),
                    network_io={},
                    process_count=sum(m.process_count for m in baseline_metrics) // len(baseline_metrics),
                    file_descriptors=sum(m.file_descriptors for m in baseline_metrics) // len(baseline_metrics),
                    system_load=sum(m.system_load for m in baseline_metrics) / len(baseline_metrics),
                    uptime=0.0,
                    security_events=0,
                    integrity_score=sum(m.integrity_score for m in baseline_metrics) / len(baseline_metrics)
                )
                
                self.logger.info("Baseline metrics established")
        
        except Exception as e:
            self.logger.error(f"Error establishing baseline: {e}")
    
    def _process_security_event(self, event: SecurityEvent):
        """Process a security event."""
        self.security_events.append(event)
        
        # Log the event
        self.logger.warning(f"Security event: {event.event_type.value} - {event.description}")
        
        # Apply immediate response based on severity
        if event.severity == IntegrityStatus.CRITICAL:
            self._handle_critical_event(event)
        elif event.severity == IntegrityStatus.DEGRADED:
            self._handle_degraded_event(event)
    
    def _handle_critical_event(self, event: SecurityEvent):
        """Handle critical security events."""
        # Implementation would include:
        # - Immediate system lockdown
        # - Alert administrators
        # - Isolate affected components
        # - Start incident response
        self.logger.critical(f"CRITICAL SECURITY EVENT: {event.description}")
    
    def _handle_degraded_event(self, event: SecurityEvent):
        """Handle degraded security events."""
        # Implementation would include:
        # - Increase monitoring
        # - Apply additional security measures
        # - Log detailed information
        self.logger.warning(f"Degraded security event: {event.description}")
    
    def _notify_callbacks(self, metrics: IntegrityMetrics, status: IntegrityStatus,
                         events: List[SecurityEvent]):
        """Notify registered callbacks."""
        for callback in self.integrity_callbacks:
            try:
                callback(metrics, status, events)
            except Exception as e:
                self.logger.error(f"Callback notification failed: {e}")
    
    def add_integrity_callback(self, callback: Callable[[IntegrityMetrics, IntegrityStatus, List[SecurityEvent]], None]):
        """Add integrity monitoring callback."""
        self.integrity_callbacks.append(callback)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system integrity status."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        latest_metrics = self.metrics_history[-1]
        anomalies = self._detect_anomalies(latest_metrics)
        status = self._calculate_integrity_status(latest_metrics, anomalies)
        
        return {
            "status": status.value,
            "integrity_score": latest_metrics.integrity_score,
            "cpu_usage": latest_metrics.cpu_usage,
            "memory_usage": latest_metrics.memory_usage,
            "disk_usage": latest_metrics.disk_usage,
            "anomalies": anomalies,
            "security_events": len(self.security_events),
            "timestamp": latest_metrics.timestamp.isoformat()
        }
    
    def get_security_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent security events."""
        events = list(self.security_events)[-limit:]
        return [
            {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "severity": event.severity.value,
                "timestamp": event.timestamp.isoformat(),
                "description": event.description,
                "source": event.source,
                "resolved": event.resolved
            }
            for event in events
        ]
    
    def get_integrity_stats(self) -> Dict[str, Any]:
        """Get integrity monitoring statistics."""
        if not self.metrics_history:
            return {"total_metrics": 0}
        
        total_metrics = len(self.metrics_history)
        avg_integrity_score = sum(m.integrity_score for m in self.metrics_history) / total_metrics
        
        # Calculate status distribution
        status_counts = {}
        for metrics in self.metrics_history:
            anomalies = self._detect_anomalies(metrics)
            status = self._calculate_integrity_status(metrics, anomalies)
            status_counts[status.value] = status_counts.get(status.value, 0) + 1
        
        return {
            "total_metrics": total_metrics,
            "average_integrity_score": avg_integrity_score,
            "status_distribution": status_counts,
            "security_events": len(self.security_events),
            "monitoring_active": self.is_monitoring
        }


class SystemAnomalyDetector:
    """System anomaly detection helper."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_anomalies(self, metrics: IntegrityMetrics) -> List[str]:
        """Detect system anomalies."""
        anomalies = []
        
        # Implementation would include more sophisticated anomaly detection
        # using statistical methods, machine learning, etc.
        
        return anomalies
