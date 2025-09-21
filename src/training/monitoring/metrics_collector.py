"""
Metrics Collector Module

This module provides metrics collection capabilities for the training system.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class Metric:
    """Individual metric data class."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    tags: Optional[Dict[str, str]] = None

class MetricsCollector:
    """Metrics collection system."""
    
    def __init__(self):
        self.metrics: List[Metric] = []
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}
        self.timers: Dict[str, List[float]] = {}
        self.logger = logger
        
    def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        if name not in self.counters:
            self.counters[name] = 0.0
        self.counters[name] += value
        
        metric = Metric(
            name=name,
            value=self.counters[name],
            metric_type=MetricType.COUNTER,
            timestamp=datetime.now(),
            tags=tags
        )
        self.metrics.append(metric)
        
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        self.gauges[name] = value
        
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            timestamp=datetime.now(),
            tags=tags
        )
        self.metrics.append(metric)
        
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram metric value."""
        if name not in self.histograms:
            self.histograms[name] = []
        self.histograms[name].append(value)
        
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            timestamp=datetime.now(),
            tags=tags
        )
        self.metrics.append(metric)
        
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timer metric value."""
        if name not in self.timers:
            self.timers[name] = []
        self.timers[name].append(duration)
        
        metric = Metric(
            name=name,
            value=duration,
            metric_type=MetricType.TIMER,
            timestamp=datetime.now(),
            tags=tags
        )
        self.metrics.append(metric)
        
    def get_counter(self, name: str) -> float:
        """Get current counter value."""
        return self.counters.get(name, 0.0)
        
    def get_gauge(self, name: str) -> float:
        """Get current gauge value."""
        return self.gauges.get(name, 0.0)
        
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics."""
        values = self.histograms.get(name, [])
        if not values:
            return {"count": 0, "min": 0, "max": 0, "mean": 0, "median": 0}
            
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "median": sorted(values)[len(values) // 2]
        }
        
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """Get timer statistics."""
        return self.get_histogram_stats(name)
        
    def get_all_metrics(self) -> List[Metric]:
        """Get all collected metrics."""
        return self.metrics.copy()
        
    def clear_metrics(self):
        """Clear all collected metrics."""
        self.metrics.clear()
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.timers.clear()
