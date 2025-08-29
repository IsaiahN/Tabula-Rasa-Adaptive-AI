"""
Metrics collection and monitoring system for agent introspection.
"""

import time
from typing import Dict, List, Any, Optional
from collections import deque
import torch
import numpy as np
import logging

from core.data_models import MetricsSnapshot, AgentState

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects and manages agent metrics for monitoring and debugging.
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        logging_mode: str = "minimal"  # minimal, debug, full
    ):
        self.buffer_size = buffer_size
        self.logging_mode = logging_mode
        
        # Metrics storage
        self.metrics_history = deque(maxlen=buffer_size)
        self.step_count = 0
        
        # Performance tracking
        self.last_log_time = time.time()
        self.log_overhead_ms = 0.0
        
    def log_step(
        self,
        agent_state: AgentState,
        prediction_error: float,
        lp_signal: float,
        memory_usage: Optional[float] = None,
        additional_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Log metrics for a single step.
        
        Args:
            agent_state: Current agent state
            prediction_error: Current prediction error
            lp_signal: Learning progress signal
            memory_usage: Memory utilization (0-1)
            additional_metrics: Additional metrics to log
        """
        start_time = time.time()
        
        # Create metrics snapshot
        snapshot = MetricsSnapshot(
            timestamp=self.step_count,
            energy=agent_state.energy,
            learning_progress=lp_signal,
            prediction_error=prediction_error,
            memory_usage=memory_usage or 0.0,
            active_goals_count=len(agent_state.active_goals),
            position=agent_state.position
        )
        
        # Add to history
        self.metrics_history.append(snapshot)
        
        # Log additional metrics based on mode
        if self.logging_mode in ["debug", "full"] and additional_metrics:
            # Store additional metrics (simplified for now)
            pass
            
        self.step_count += 1
        
        # Track logging overhead
        end_time = time.time()
        self.log_overhead_ms = (end_time - start_time) * 1000
        
    def get_recent_metrics(self, num_steps: int = 100) -> List[MetricsSnapshot]:
        """Get recent metrics snapshots."""
        if len(self.metrics_history) <= num_steps:
            return list(self.metrics_history)
        else:
            return list(self.metrics_history)[-num_steps:]
            
    def compute_summary_statistics(self, window_size: int = 1000) -> Dict[str, float]:
        """
        Compute summary statistics over recent window.
        
        Args:
            window_size: Number of recent steps to analyze
            
        Returns:
            stats: Dictionary of summary statistics
        """
        recent_metrics = self.get_recent_metrics(window_size)
        
        if not recent_metrics:
            return {}
            
        # Extract time series
        energies = [m.energy for m in recent_metrics]
        lp_signals = [m.learning_progress for m in recent_metrics]
        pred_errors = [m.prediction_error for m in recent_metrics]
        memory_usage = [m.memory_usage for m in recent_metrics]
        
        # Compute statistics
        stats = {
            # Energy statistics
            'energy_mean': np.mean(energies),
            'energy_std': np.std(energies),
            'energy_min': np.min(energies),
            'energy_max': np.max(energies),
            'energy_trend': self._compute_trend(energies),
            
            # Learning progress statistics
            'lp_mean': np.mean(lp_signals),
            'lp_std': np.std(lp_signals),
            'lp_positive_ratio': np.mean([lp > 0 for lp in lp_signals]),
            
            # Prediction error statistics
            'pred_error_mean': np.mean(pred_errors),
            'pred_error_std': np.std(pred_errors),
            'pred_error_trend': self._compute_trend(pred_errors),
            
            # Memory statistics
            'memory_usage_mean': np.mean(memory_usage),
            'memory_usage_max': np.max(memory_usage),
            
            # Performance
            'log_overhead_ms': self.log_overhead_ms,
            'steps_logged': len(recent_metrics)
        }
        
        return stats
        
    def _compute_trend(self, values: List[float]) -> float:
        """Compute linear trend (slope) of values."""
        if len(values) < 2:
            return 0.0
            
        x = np.arange(len(values))
        try:
            slope, _ = np.polyfit(x, values, 1)
            return float(slope)
        except:
            return 0.0
            
    def detect_anomalies(self, window_size: int = 100) -> Dict[str, bool]:
        """
        Detect anomalies in recent metrics.
        
        Args:
            window_size: Window for anomaly detection
            
        Returns:
            anomalies: Dictionary of detected anomalies
        """
        recent_metrics = self.get_recent_metrics(window_size)
        
        if len(recent_metrics) < 10:
            return {}
            
        # Extract values
        energies = [m.energy for m in recent_metrics]
        lp_signals = [m.learning_progress for m in recent_metrics]
        pred_errors = [m.prediction_error for m in recent_metrics]
        
        anomalies = {}
        
        # Energy anomalies
        anomalies['energy_crash'] = any(e < 5.0 for e in energies[-10:])  # Very low energy
        anomalies['energy_stuck'] = np.std(energies[-20:]) < 0.1 if len(energies) >= 20 else False
        
        # Learning progress anomalies
        anomalies['lp_stuck'] = all(abs(lp) < 0.001 for lp in lp_signals[-50:]) if len(lp_signals) >= 50 else False
        anomalies['lp_oscillating'] = np.std(lp_signals[-20:]) > 0.5 if len(lp_signals) >= 20 else False
        
        # Prediction error anomalies
        anomalies['pred_error_exploding'] = any(pe > 10.0 for pe in pred_errors[-10:])
        
        return anomalies
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance and health report."""
        stats = self.compute_summary_statistics()
        anomalies = self.detect_anomalies()
        
        # Overall health score (0-1)
        health_score = 1.0
        
        # Penalize anomalies
        for anomaly_name, detected in anomalies.items():
            if detected:
                health_score -= 0.2
                
        # Penalize high logging overhead
        if stats.get('log_overhead_ms', 0) > 5.0:  # More than 5ms overhead
            health_score -= 0.1
            
        health_score = max(0.0, health_score)
        
        return {
            'health_score': health_score,
            'statistics': stats,
            'anomalies': anomalies,
            'total_steps': self.step_count,
            'buffer_utilization': len(self.metrics_history) / self.buffer_size
        }
        
    def export_metrics(self, format: str = "dict") -> Any:
        """
        Export metrics in specified format.
        
        Args:
            format: Export format ("dict", "numpy", "pandas")
            
        Returns:
            exported_data: Metrics in requested format
        """
        if format == "dict":
            return [m.to_dict() for m in self.metrics_history]
        elif format == "numpy":
            if not self.metrics_history:
                return {}
                
            return {
                'timestamps': np.array([m.timestamp for m in self.metrics_history]),
                'energy': np.array([m.energy for m in self.metrics_history]),
                'learning_progress': np.array([m.learning_progress for m in self.metrics_history]),
                'prediction_error': np.array([m.prediction_error for m in self.metrics_history]),
                'memory_usage': np.array([m.memory_usage for m in self.metrics_history]),
                'active_goals_count': np.array([m.active_goals_count for m in self.metrics_history])
            }
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
    def reset(self):
        """Reset metrics collector."""
        self.metrics_history.clear()
        self.step_count = 0
        self.last_log_time = time.time()
        
    def set_logging_mode(self, mode: str):
        """Set logging mode (minimal, debug, full)."""
        if mode not in ["minimal", "debug", "full"]:
            raise ValueError(f"Invalid logging mode: {mode}")
        self.logging_mode = mode
        logger.info(f"Logging mode set to: {mode}")


class AnomalyDetector:
    """Specialized anomaly detection for agent metrics."""
    
    def __init__(self, sensitivity: float = 0.1):
        self.sensitivity = sensitivity
        self.baseline_stats = {}
        
    def update_baseline(self, metrics: List[MetricsSnapshot]):
        """Update baseline statistics from normal operation."""
        if not metrics:
            return
            
        energies = [m.energy for m in metrics]
        lp_signals = [m.learning_progress for m in metrics]
        
        self.baseline_stats = {
            'energy_mean': np.mean(energies),
            'energy_std': np.std(energies),
            'lp_mean': np.mean(lp_signals),
            'lp_std': np.std(lp_signals)
        }
        
    def detect_anomaly(self, current_metrics: Dict[str, float]) -> bool:
        """Detect if current metrics are anomalous."""
        if not self.baseline_stats:
            return False
            
        # Z-score based detection
        energy_z = abs(current_metrics.get('energy', 0) - self.baseline_stats['energy_mean']) / max(self.baseline_stats['energy_std'], 0.1)
        lp_z = abs(current_metrics.get('learning_progress', 0) - self.baseline_stats['lp_mean']) / max(self.baseline_stats['lp_std'], 0.1)
        
        # Anomaly if any z-score exceeds threshold
        threshold = 3.0 / self.sensitivity
        return energy_z > threshold or lp_z > threshold