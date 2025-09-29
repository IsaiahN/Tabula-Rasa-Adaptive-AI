"""
Metrics Collector

Collects and aggregates various performance and training metrics.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# Global singleton instance
_metrics_collector_instance = None

class MetricsCollector:
    """Collects and aggregates training metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.current_metrics: Dict[str, Any] = {}
        self.metric_timestamps: Dict[str, datetime] = {}
        self.aggregated_metrics: Dict[str, Dict[str, float]] = {}
    
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None) -> None:
        """Record a metric value with timestamp."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.metrics_history[metric_name].append({
            'value': value,
            'timestamp': timestamp
        })
        
        self.current_metrics[metric_name] = value
        self.metric_timestamps[metric_name] = timestamp
        
        # Update aggregated metrics
        self._update_aggregated_metrics(metric_name)
    
    def get_metric_history(self, metric_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get history for a specific metric."""
        history = list(self.metrics_history.get(metric_name, []))
        if limit is not None:
            history = history[-limit:]
        return history
    
    def get_current_metric(self, metric_name: str, default: Any = None) -> Any:
        """Get current value of a metric."""
        return self.current_metrics.get(metric_name, default)
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        history = self.get_metric_history(metric_name)
        if not history:
            return {}
        
        values = [entry['value'] for entry in history]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'latest': values[-1] if values else 0,
            'latest_timestamp': history[-1]['timestamp'].isoformat() if history else None
        }
    
    def get_all_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary for all metrics."""
        summary = {}
        for metric_name in self.metrics_history.keys():
            summary[metric_name] = self.get_metric_summary(metric_name)
        return summary
    
    def get_metrics_trend(self, metric_name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """Get trend analysis for a metric over a time window."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        history = self.get_metric_history(metric_name)
        
        # Filter to window
        window_data = [
            entry for entry in history 
            if entry['timestamp'] >= cutoff_time
        ]
        
        if len(window_data) < 2:
            return {'trend': 'insufficient_data', 'change': 0.0}
        
        values = [entry['value'] for entry in window_data]
        
        # Calculate trend
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half) if first_half else 0
        second_avg = sum(second_half) / len(second_half) if second_half else 0
        
        change = second_avg - first_avg
        change_percent = (change / first_avg * 100) if first_avg != 0 else 0
        
        if abs(change_percent) < 5:
            trend = 'stable'
        elif change_percent > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return {
            'trend': trend,
            'change': change,
            'change_percent': change_percent,
            'data_points': len(window_data),
            'window_minutes': window_minutes
        }
    
    def get_correlation(self, metric1: str, metric2: str) -> Optional[float]:
        """Calculate correlation between two metrics."""
        history1 = self.get_metric_history(metric1)
        history2 = self.get_metric_history(metric2)
        
        if len(history1) != len(history2) or len(history1) < 2:
            return None
        
        # Align timestamps and calculate correlation
        values1 = [entry['value'] for entry in history1]
        values2 = [entry['value'] for entry in history2]
        
        try:
            # Simple correlation calculation
            n = len(values1)
            sum1 = sum(values1)
            sum2 = sum(values2)
            sum1_sq = sum(x*x for x in values1)
            sum2_sq = sum(x*x for x in values2)
            sum12 = sum(x*y for x, y in zip(values1, values2))
            
            numerator = n * sum12 - sum1 * sum2
            denominator = ((n * sum1_sq - sum1**2) * (n * sum2_sq - sum2**2))**0.5
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return None
    
    def _update_aggregated_metrics(self, metric_name: str) -> None:
        """Update aggregated metrics for a given metric."""
        summary = self.get_metric_summary(metric_name)
        self.aggregated_metrics[metric_name] = summary
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format."""
        if format == 'json':
            import json
            return json.dumps({
                'current_metrics': self.current_metrics,
                'aggregated_metrics': self.aggregated_metrics,
                'export_timestamp': datetime.now().isoformat()
            }, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        self.metrics_history.clear()
        self.current_metrics.clear()
        self.metric_timestamps.clear()
        self.aggregated_metrics.clear()
        logger.info("Metrics collector reset")


def create_metrics_collector(max_history: int = 1000) -> MetricsCollector:
    """Create or get the singleton MetricsCollector instance."""
    global _metrics_collector_instance
    if _metrics_collector_instance is None:
        print("  LEARNING MANAGER: Creating singleton MetricsCollector instance")
        _metrics_collector_instance = MetricsCollector(max_history=max_history)
    return _metrics_collector_instance


def get_metrics_collector() -> Optional[MetricsCollector]:
    """Get the singleton MetricsCollector instance if it exists."""
    return _metrics_collector_instance
