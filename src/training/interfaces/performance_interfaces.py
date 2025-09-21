"""
Performance Monitoring Interfaces

Defines interfaces for performance monitoring components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from .base_interfaces import PerformanceInterface, ComponentInterface


class PerformanceMonitorInterface(PerformanceInterface, ComponentInterface):
    """Interface for performance monitors."""
    
    @abstractmethod
    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start monitoring an operation and return operation ID."""
        pass
    
    @abstractmethod
    def end_operation(self, operation_id: str) -> Dict[str, Any]:
        """End monitoring and return operation metrics."""
        pass
    
    @abstractmethod
    def get_active_operations(self) -> List[str]:
        """Get list of currently active operations."""
        pass
    
    @abstractmethod
    def get_operation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent operation history."""
        pass
    
    @abstractmethod
    def set_alert_threshold(self, metric_name: str, threshold: float) -> None:
        """Set alert threshold for a metric."""
        pass
    
    @abstractmethod
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get current alerts."""
        pass


class MetricsCollectorInterface(ComponentInterface):
    """Interface for metrics collection."""
    
    @abstractmethod
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-wide metrics."""
        pass
    
    @abstractmethod
    def collect_component_metrics(self, component_name: str) -> Dict[str, Any]:
        """Collect metrics for a specific component."""
        pass
    
    @abstractmethod
    def get_metric_trends(self, metric_name: str, time_range: int) -> List[Dict[str, Any]]:
        """Get trend data for a metric over time."""
        pass
    
    @abstractmethod
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format."""
        pass
    
    @abstractmethod
    def clear_old_metrics(self, older_than: datetime) -> int:
        """Clear metrics older than specified time."""
        pass
