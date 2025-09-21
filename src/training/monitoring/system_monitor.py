"""
System Monitor

Monitors system resources and health metrics.
"""

import psutil
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class SystemMetric(Enum):
    """System metrics to monitor."""
    CPU_PERCENT = "cpu_percent"
    MEMORY_PERCENT = "memory_percent"
    DISK_USAGE = "disk_usage"
    NETWORK_IO = "network_io"
    PROCESS_COUNT = "process_count"
    LOAD_AVERAGE = "load_average"


@dataclass
class SystemHealth:
    """System health status."""
    status: str  # 'healthy', 'warning', 'critical'
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    timestamp: datetime
    alerts: List[str]


class SystemMonitor:
    """
    Monitors system resources and health.
    """
    
    def __init__(self, check_interval: int = 5):
        self.check_interval = check_interval
        self._running = False
        self._thread = None
        self._callbacks: List[Callable[[SystemHealth], None]] = []
        self._metrics_history: List[Dict[str, Any]] = []
        self._max_history = 1000
        
        # Thresholds
        self.cpu_warning_threshold = 80.0
        self.cpu_critical_threshold = 95.0
        self.memory_warning_threshold = 80.0
        self.memory_critical_threshold = 95.0
        self.disk_warning_threshold = 85.0
        self.disk_critical_threshold = 95.0
    
    def start_monitoring(self) -> None:
        """Start system monitoring."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self._running = False
        if self._thread:
            self._thread.join()
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                health = self._check_system_health()
                self._metrics_history.append({
                    'timestamp': health.timestamp,
                    'cpu_usage': health.cpu_usage,
                    'memory_usage': health.memory_usage,
                    'disk_usage': health.disk_usage,
                    'status': health.status
                })
                
                # Keep only recent history
                if len(self._metrics_history) > self._max_history:
                    self._metrics_history = self._metrics_history[-self._max_history:]
                
                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(health)
                    except Exception as e:
                        print(f"Error in system monitor callback: {e}")
                
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Error in system monitoring: {e}")
                time.sleep(self.check_interval)
    
    def _check_system_health(self) -> SystemHealth:
        """Check current system health."""
        # Get system metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Determine status
        alerts = []
        status = 'healthy'
        
        if cpu_usage >= self.cpu_critical_threshold:
            status = 'critical'
            alerts.append(f"CPU usage critical: {cpu_usage:.1f}%")
        elif cpu_usage >= self.cpu_warning_threshold:
            status = 'warning' if status == 'healthy' else status
            alerts.append(f"CPU usage high: {cpu_usage:.1f}%")
        
        if memory.percent >= self.memory_critical_threshold:
            status = 'critical'
            alerts.append(f"Memory usage critical: {memory.percent:.1f}%")
        elif memory.percent >= self.memory_warning_threshold:
            status = 'warning' if status == 'healthy' else status
            alerts.append(f"Memory usage high: {memory.percent:.1f}%")
        
        if disk.percent >= self.disk_critical_threshold:
            status = 'critical'
            alerts.append(f"Disk usage critical: {disk.percent:.1f}%")
        elif disk.percent >= self.disk_warning_threshold:
            status = 'warning' if status == 'healthy' else status
            alerts.append(f"Disk usage high: {disk.percent:.1f}%")
        
        return SystemHealth(
            status=status,
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            timestamp=datetime.now(),
            alerts=alerts
        )
    
    def add_callback(self, callback: Callable[[SystemHealth], None]) -> None:
        """Add a callback for health updates."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[SystemHealth], None]) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def get_current_health(self) -> SystemHealth:
        """Get current system health."""
        return self._check_system_health()
    
    def get_metrics_history(self, duration_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get metrics history for specified duration."""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        return [m for m in self._metrics_history if m['timestamp'] >= cutoff_time]
    
    def get_average_metrics(self, duration_minutes: int = 60) -> Dict[str, float]:
        """Get average metrics for specified duration."""
        history = self.get_metrics_history(duration_minutes)
        if not history:
            return {}
        
        return {
            'avg_cpu_usage': sum(m['cpu_usage'] for m in history) / len(history),
            'avg_memory_usage': sum(m['memory_usage'] for m in history) / len(history),
            'avg_disk_usage': sum(m['disk_usage'] for m in history) / len(history),
            'sample_count': len(history)
        }
    
    def set_thresholds(self, 
                      cpu_warning: Optional[float] = None,
                      cpu_critical: Optional[float] = None,
                      memory_warning: Optional[float] = None,
                      memory_critical: Optional[float] = None,
                      disk_warning: Optional[float] = None,
                      disk_critical: Optional[float] = None) -> None:
        """Set monitoring thresholds."""
        if cpu_warning is not None:
            self.cpu_warning_threshold = cpu_warning
        if cpu_critical is not None:
            self.cpu_critical_threshold = cpu_critical
        if memory_warning is not None:
            self.memory_warning_threshold = memory_warning
        if memory_critical is not None:
            self.memory_critical_threshold = memory_critical
        if disk_warning is not None:
            self.disk_warning_threshold = disk_warning
        if disk_critical is not None:
            self.disk_critical_threshold = disk_critical
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get detailed system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_total': psutil.disk_usage('/').total,
            'disk_free': psutil.disk_usage('/').free,
            'boot_time': datetime.fromtimestamp(psutil.boot_time()),
            'process_count': len(psutil.pids())
        }
