"""
Performance Monitor Module

This module provides performance monitoring capabilities for the training system.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data class."""
    timestamp: datetime
    operation: str
    duration: float
    memory_usage: float
    cpu_usage: float
    success: bool
    error_message: Optional[str] = None

class PerformanceMonitor:
    """Performance monitoring system."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.start_times: Dict[str, float] = {}
        self.logger = logger
        
    def start_operation(self, operation: str) -> str:
        """Start monitoring an operation."""
        operation_id = f"{operation}_{int(time.time() * 1000)}"
        self.start_times[operation_id] = time.time()
        return operation_id
        
    def end_operation(self, operation_id: str, success: bool = True, 
                     error_message: Optional[str] = None) -> PerformanceMetrics:
        """End monitoring an operation and record metrics."""
        if operation_id not in self.start_times:
            raise ValueError(f"Operation {operation_id} not found")
            
        start_time = self.start_times.pop(operation_id)
        duration = time.time() - start_time
        
        # Get basic system metrics (simplified)
        memory_usage = 0.0  # Would use psutil in real implementation
        cpu_usage = 0.0     # Would use psutil in real implementation
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            operation=operation_id.split('_')[0],
            duration=duration,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            success=success,
            error_message=error_message
        )
        
        self.metrics.append(metrics)
        return metrics
        
    def get_metrics(self, operation: Optional[str] = None) -> List[PerformanceMetrics]:
        """Get performance metrics, optionally filtered by operation."""
        if operation:
            return [m for m in self.metrics if m.operation == operation]
        return self.metrics.copy()
        
    def get_average_duration(self, operation: Optional[str] = None) -> float:
        """Get average duration for operations."""
        metrics = self.get_metrics(operation)
        if not metrics:
            return 0.0
        return sum(m.duration for m in metrics) / len(metrics)
        
    def get_success_rate(self, operation: Optional[str] = None) -> float:
        """Get success rate for operations."""
        metrics = self.get_metrics(operation)
        if not metrics:
            return 0.0
        successful = sum(1 for m in metrics if m.success)
        return successful / len(metrics)
        
    def clear_metrics(self):
        """Clear all stored metrics."""
        self.metrics.clear()
        self.start_times.clear()
