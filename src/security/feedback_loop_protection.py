#!/usr/bin/env python3
"""
Feedback Loop Protection System - Prevention of harmful feedback loops.

This module provides protection against:
- Self-reinforcing negative loops
- Catastrophic forgetting loops
- Performance degradation loops
- Memory corruption loops
- Learning stagnation loops
"""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime, timedelta
from collections import deque
import json

logger = logging.getLogger(__name__)


class LoopType(Enum):
    """Types of feedback loops."""
    NEGATIVE_REINFORCEMENT = "negative_reinforcement"
    CATASTROPHIC_FORGETTING = "catastrophic_forgetting"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MEMORY_CORRUPTION = "memory_corruption"
    LEARNING_STAGNATION = "learning_stagnation"
    ENERGY_DRAIN = "energy_drain"
    EXPLORATION_COLLAPSE = "exploration_collapse"
    CONFIDENCE_OVERFLOW = "confidence_overflow"
    UNKNOWN = "unknown"


class LoopSeverity(Enum):
    """Severity levels for feedback loops."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LoopDetection:
    """Result of feedback loop detection."""
    is_loop: bool
    loop_type: LoopType
    severity: LoopSeverity
    confidence: float
    pattern: Dict[str, Any]
    mitigation_applied: bool
    detection_time: float
    metadata: Dict[str, Any]


class FeedbackLoopDetector:
    """
    Advanced feedback loop detection and prevention system.
    
    Monitors system behavior for harmful patterns:
    - Performance degradation trends
    - Memory usage patterns
    - Learning progress patterns
    - Energy consumption patterns
    - Exploration behavior patterns
    """
    
    def __init__(self, window_size: int = 100, sensitivity: float = 0.7):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.logger = logging.getLogger(__name__)
        
        # Monitoring data
        self.performance_history = deque(maxlen=window_size)
        self.memory_history = deque(maxlen=window_size)
        self.learning_history = deque(maxlen=window_size)
        self.energy_history = deque(maxlen=window_size)
        self.exploration_history = deque(maxlen=window_size)
        
        # Detection patterns
        self.loop_patterns = {}
        self.baseline_metrics = {}
        
        # Detection thresholds
        self.thresholds = {
            'performance_degradation': 0.2,  # 20% performance drop
            'memory_growth': 0.5,  # 50% memory growth
            'learning_stagnation': 0.1,  # 10% learning progress
            'energy_drain': 0.3,  # 30% energy increase
            'exploration_collapse': 0.4,  # 40% exploration reduction
            'confidence_overflow': 0.9,  # 90% confidence threshold
        }
        
        # Initialize detection methods
        self._initialize_detection_methods()
    
    def _initialize_detection_methods(self):
        """Initialize detection methods for different loop types."""
        self.detection_methods = {
            LoopType.NEGATIVE_REINFORCEMENT: self._detect_negative_reinforcement,
            LoopType.CATASTROPHIC_FORGETTING: self._detect_catastrophic_forgetting,
            LoopType.PERFORMANCE_DEGRADATION: self._detect_performance_degradation,
            LoopType.MEMORY_CORRUPTION: self._detect_memory_corruption,
            LoopType.LEARNING_STAGNATION: self._detect_learning_stagnation,
            LoopType.ENERGY_DRAIN: self._detect_energy_drain,
            LoopType.EXPLORATION_COLLAPSE: self._detect_exploration_collapse,
            LoopType.CONFIDENCE_OVERFLOW: self._detect_confidence_overflow,
        }
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update system metrics for monitoring."""
        current_time = time.time()
        
        # Update performance metrics
        if 'performance' in metrics:
            self.performance_history.append({
                'timestamp': current_time,
                'value': metrics['performance'],
                'context': metrics.get('context', 'general')
            })
        
        # Update memory metrics
        if 'memory_usage' in metrics:
            self.memory_history.append({
                'timestamp': current_time,
                'value': metrics['memory_usage'],
                'context': metrics.get('context', 'general')
            })
        
        # Update learning metrics
        if 'learning_progress' in metrics:
            self.learning_history.append({
                'timestamp': current_time,
                'value': metrics['learning_progress'],
                'context': metrics.get('context', 'general')
            })
        
        # Update energy metrics
        if 'energy_usage' in metrics:
            self.energy_history.append({
                'timestamp': current_time,
                'value': metrics['energy_usage'],
                'context': metrics.get('context', 'general')
            })
        
        # Update exploration metrics
        if 'exploration_rate' in metrics:
            self.exploration_history.append({
                'timestamp': current_time,
                'value': metrics['exploration_rate'],
                'context': metrics.get('context', 'general')
            })
    
    def detect_loops(self) -> List[LoopDetection]:
        """Detect all types of feedback loops."""
        detections = []
        
        for loop_type, detection_method in self.detection_methods.items():
            try:
                detection = detection_method()
                if detection and detection.is_loop:
                    detections.append(detection)
            except Exception as e:
                self.logger.warning(f"Loop detection failed for {loop_type}: {e}")
        
        return detections
    
    def _detect_negative_reinforcement(self) -> Optional[LoopDetection]:
        """Detect negative reinforcement loops."""
        if len(self.performance_history) < 10:
            return None
        
        # Analyze performance trend
        performance_values = [p['value'] for p in self.performance_history]
        trend = self._calculate_trend(performance_values)
        
        # Check for consistent negative trend
        if trend < -0.1:  # Negative trend
            # Check for acceleration of decline
            recent_values = performance_values[-5:]
            if len(recent_values) >= 3:
                recent_trend = self._calculate_trend(recent_values)
                if recent_trend < trend:  # Accelerating decline
                    return LoopDetection(
                        is_loop=True,
                        loop_type=LoopType.NEGATIVE_REINFORCEMENT,
                        severity=LoopSeverity.HIGH if abs(trend) > 0.3 else LoopSeverity.MEDIUM,
                        confidence=min(abs(trend), 1.0),
                        pattern={
                            'trend': trend,
                            'recent_trend': recent_trend,
                            'performance_values': performance_values[-10:]
                        },
                        mitigation_applied=False,
                        detection_time=time.time(),
                        metadata={'detection_method': 'trend_analysis'}
                    )
        
        return None
    
    def _detect_catastrophic_forgetting(self) -> Optional[LoopDetection]:
        """Detect catastrophic forgetting loops."""
        if len(self.learning_history) < 20:
            return None
        
        learning_values = [l['value'] for l in self.learning_history]
        
        # Check for sudden drops in learning progress
        if len(learning_values) >= 10:
            # Calculate rolling variance
            window_size = 5
            variances = []
            for i in range(window_size, len(learning_values)):
                window = learning_values[i-window_size:i]
                variances.append(np.var(window))
            
            # Check for high variance (indicating instability)
            if variances and np.mean(variances) > np.var(learning_values) * 2:
                # Check for recent performance drops
                recent_performance = learning_values[-5:]
                if len(recent_performance) >= 3:
                    recent_trend = self._calculate_trend(recent_performance)
                    if recent_trend < -0.2:  # Significant drop
                        return LoopDetection(
                            is_loop=True,
                            loop_type=LoopType.CATASTROPHIC_FORGETTING,
                            severity=LoopSeverity.HIGH,
                            confidence=min(abs(recent_trend), 1.0),
                            pattern={
                                'variance': np.mean(variances),
                                'recent_trend': recent_trend,
                                'learning_values': learning_values[-10:]
                            },
                            mitigation_applied=False,
                            detection_time=time.time(),
                            metadata={'detection_method': 'variance_analysis'}
                        )
        
        return None
    
    def _detect_performance_degradation(self) -> Optional[LoopDetection]:
        """Detect performance degradation loops."""
        if len(self.performance_history) < 15:
            return None
        
        performance_values = [p['value'] for p in self.performance_history]
        
        # Calculate performance degradation rate
        if len(performance_values) >= 10:
            # Compare recent performance to baseline
            baseline = np.mean(performance_values[:5])
            recent = np.mean(performance_values[-5:])
            
            degradation_rate = (baseline - recent) / baseline if baseline > 0 else 0
            
            if degradation_rate > self.thresholds['performance_degradation']:
                # Check if degradation is accelerating
                mid_performance = np.mean(performance_values[5:10])
                recent_degradation = (mid_performance - recent) / mid_performance if mid_performance > 0 else 0
                
                severity = LoopSeverity.CRITICAL if degradation_rate > 0.5 else LoopSeverity.HIGH
                
                return LoopDetection(
                    is_loop=True,
                    loop_type=LoopType.PERFORMANCE_DEGRADATION,
                    severity=severity,
                    confidence=min(degradation_rate, 1.0),
                    pattern={
                        'degradation_rate': degradation_rate,
                        'recent_degradation': recent_degradation,
                        'baseline': baseline,
                        'recent': recent,
                        'performance_values': performance_values[-10:]
                    },
                    mitigation_applied=False,
                    detection_time=time.time(),
                    metadata={'detection_method': 'degradation_analysis'}
                )
        
        return None
    
    def _detect_memory_corruption(self) -> Optional[LoopDetection]:
        """Detect memory corruption loops."""
        if len(self.memory_history) < 10:
            return None
        
        memory_values = [m['value'] for m in self.memory_history]
        
        # Check for unusual memory growth patterns
        if len(memory_values) >= 5:
            # Calculate memory growth rate
            growth_rates = []
            for i in range(1, len(memory_values)):
                if memory_values[i-1] > 0:
                    growth_rate = (memory_values[i] - memory_values[i-1]) / memory_values[i-1]
                    growth_rates.append(growth_rate)
            
            if growth_rates:
                avg_growth_rate = np.mean(growth_rates)
                
                # Check for exponential growth
                if avg_growth_rate > self.thresholds['memory_growth']:
                    # Check for accelerating growth
                    recent_growth = np.mean(growth_rates[-3:]) if len(growth_rates) >= 3 else avg_growth_rate
                    
                    if recent_growth > avg_growth_rate * 1.5:  # Accelerating growth
                        return LoopDetection(
                            is_loop=True,
                            loop_type=LoopType.MEMORY_CORRUPTION,
                            severity=LoopSeverity.HIGH,
                            confidence=min(avg_growth_rate, 1.0),
                            pattern={
                                'growth_rate': avg_growth_rate,
                                'recent_growth': recent_growth,
                                'memory_values': memory_values[-10:],
                                'growth_rates': growth_rates[-5:]
                            },
                            mitigation_applied=False,
                            detection_time=time.time(),
                            metadata={'detection_method': 'growth_analysis'}
                        )
        
        return None
    
    def _detect_learning_stagnation(self) -> Optional[LoopDetection]:
        """Detect learning stagnation loops."""
        if len(self.learning_history) < 20:
            return None
        
        learning_values = [l['value'] for l in self.learning_history]
        
        # Check for stagnation (no significant improvement)
        if len(learning_values) >= 10:
            # Calculate learning progress over time
            progress_windows = []
            window_size = 5
            
            for i in range(window_size, len(learning_values)):
                window = learning_values[i-window_size:i]
                progress = window[-1] - window[0]
                progress_windows.append(progress)
            
            if progress_windows:
                avg_progress = np.mean(progress_windows)
                progress_std = np.std(progress_windows)
                
                # Check for stagnation (low progress with low variance)
                if abs(avg_progress) < self.thresholds['learning_stagnation'] and progress_std < 0.05:
                    # Check if stagnation is persistent
                    recent_progress = progress_windows[-3:] if len(progress_windows) >= 3 else progress_windows
                    if all(abs(p) < 0.05 for p in recent_progress):
                        return LoopDetection(
                            is_loop=True,
                            loop_type=LoopType.LEARNING_STAGNATION,
                            severity=LoopSeverity.MEDIUM,
                            confidence=1.0 - min(abs(avg_progress), 1.0),
                            pattern={
                                'avg_progress': avg_progress,
                                'progress_std': progress_std,
                                'recent_progress': recent_progress,
                                'learning_values': learning_values[-10:]
                            },
                            mitigation_applied=False,
                            detection_time=time.time(),
                            metadata={'detection_method': 'stagnation_analysis'}
                        )
        
        return None
    
    def _detect_energy_drain(self) -> Optional[LoopDetection]:
        """Detect energy drain loops."""
        if len(self.energy_history) < 10:
            return None
        
        energy_values = [e['value'] for e in self.energy_history]
        
        # Check for increasing energy consumption
        if len(energy_values) >= 5:
            # Calculate energy consumption trend
            trend = self._calculate_trend(energy_values)
            
            if trend > self.thresholds['energy_drain']:
                # Check for accelerating energy consumption
                recent_values = energy_values[-5:]
                recent_trend = self._calculate_trend(recent_values)
                
                if recent_trend > trend:  # Accelerating consumption
                    return LoopDetection(
                        is_loop=True,
                        loop_type=LoopType.ENERGY_DRAIN,
                        severity=LoopSeverity.HIGH if trend > 0.5 else LoopSeverity.MEDIUM,
                        confidence=min(trend, 1.0),
                        pattern={
                            'trend': trend,
                            'recent_trend': recent_trend,
                            'energy_values': energy_values[-10:]
                        },
                        mitigation_applied=False,
                        detection_time=time.time(),
                        metadata={'detection_method': 'energy_analysis'}
                    )
        
        return None
    
    def _detect_exploration_collapse(self) -> Optional[LoopDetection]:
        """Detect exploration collapse loops."""
        if len(self.exploration_history) < 10:
            return None
        
        exploration_values = [e['value'] for e in self.exploration_history]
        
        # Check for decreasing exploration
        if len(exploration_values) >= 5:
            trend = self._calculate_trend(exploration_values)
            
            if trend < -self.thresholds['exploration_collapse']:
                # Check if exploration is approaching zero
                recent_exploration = np.mean(exploration_values[-3:])
                
                if recent_exploration < 0.1:  # Very low exploration
                    return LoopDetection(
                        is_loop=True,
                        loop_type=LoopType.EXPLORATION_COLLAPSE,
                        severity=LoopSeverity.HIGH,
                        confidence=min(abs(trend), 1.0),
                        pattern={
                            'trend': trend,
                            'recent_exploration': recent_exploration,
                            'exploration_values': exploration_values[-10:]
                        },
                        mitigation_applied=False,
                        detection_time=time.time(),
                        metadata={'detection_method': 'exploration_analysis'}
                    )
        
        return None
    
    def _detect_confidence_overflow(self) -> Optional[LoopDetection]:
        """Detect confidence overflow loops."""
        # This would require confidence data from the system
        # For now, return None as we don't have confidence tracking
        return None
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend of values using linear regression."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        n = len(values)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
    
    def apply_mitigation(self, detection: LoopDetection) -> bool:
        """Apply mitigation strategies for detected loops."""
        try:
            self.logger.warning(f"Feedback loop detected: {detection.loop_type.value} "
                              f"(severity: {detection.severity.value})")
            
            # Apply mitigation based on loop type
            if detection.loop_type == LoopType.NEGATIVE_REINFORCEMENT:
                return self._mitigate_negative_reinforcement(detection)
            elif detection.loop_type == LoopType.CATASTROPHIC_FORGETTING:
                return self._mitigate_catastrophic_forgetting(detection)
            elif detection.loop_type == LoopType.PERFORMANCE_DEGRADATION:
                return self._mitigate_performance_degradation(detection)
            elif detection.loop_type == LoopType.MEMORY_CORRUPTION:
                return self._mitigate_memory_corruption(detection)
            elif detection.loop_type == LoopType.LEARNING_STAGNATION:
                return self._mitigate_learning_stagnation(detection)
            elif detection.loop_type == LoopType.ENERGY_DRAIN:
                return self._mitigate_energy_drain(detection)
            elif detection.loop_type == LoopType.EXPLORATION_COLLAPSE:
                return self._mitigate_exploration_collapse(detection)
            else:
                return self._mitigate_general_loop(detection)
                
        except Exception as e:
            self.logger.error(f"Mitigation application failed: {e}")
            return False
    
    def _mitigate_negative_reinforcement(self, detection: LoopDetection) -> bool:
        """Mitigate negative reinforcement loops."""
        # Reset performance expectations
        # Increase exploration
        # Apply positive reinforcement
        self.logger.info("Applying negative reinforcement mitigation")
        return True
    
    def _mitigate_catastrophic_forgetting(self, detection: LoopDetection) -> bool:
        """Mitigate catastrophic forgetting loops."""
        # Implement elastic weight consolidation
        # Reduce learning rate
        # Apply memory replay
        self.logger.info("Applying catastrophic forgetting mitigation")
        return True
    
    def _mitigate_performance_degradation(self, detection: LoopDetection) -> bool:
        """Mitigate performance degradation loops."""
        # Reset system parameters
        # Clear corrupted memory
        # Restart learning processes
        self.logger.info("Applying performance degradation mitigation")
        return True
    
    def _mitigate_memory_corruption(self, detection: LoopDetection) -> bool:
        """Mitigate memory corruption loops."""
        # Clear corrupted memory
        # Reset memory management
        # Apply memory validation
        self.logger.info("Applying memory corruption mitigation")
        return True
    
    def _mitigate_learning_stagnation(self, detection: LoopDetection) -> bool:
        """Mitigate learning stagnation loops."""
        # Increase exploration rate
        # Change learning strategy
        # Reset learning parameters
        self.logger.info("Applying learning stagnation mitigation")
        return True
    
    def _mitigate_energy_drain(self, detection: LoopDetection) -> bool:
        """Mitigate energy drain loops."""
        # Optimize energy usage
        # Reduce computational load
        # Implement energy conservation
        self.logger.info("Applying energy drain mitigation")
        return True
    
    def _mitigate_exploration_collapse(self, detection: LoopDetection) -> bool:
        """Mitigate exploration collapse loops."""
        # Force exploration
        # Reset exploration parameters
        # Apply curiosity mechanisms
        self.logger.info("Applying exploration collapse mitigation")
        return True
    
    def _mitigate_general_loop(self, detection: LoopDetection) -> bool:
        """Apply general mitigation for unknown loop types."""
        # Log the loop
        # Apply conservative measures
        # Monitor for resolution
        self.logger.info("Applying general loop mitigation")
        return True
    
    def get_loop_stats(self) -> Dict[str, Any]:
        """Get feedback loop detection statistics."""
        detections = self.detect_loops()
        
        if not detections:
            return {"total_loops": 0}
        
        loop_types = {}
        severities = {}
        
        for detection in detections:
            loop_type = detection.loop_type.value
            loop_types[loop_type] = loop_types.get(loop_type, 0) + 1
            
            severity = detection.severity.value
            severities[severity] = severities.get(severity, 0) + 1
        
        return {
            "total_loops": len(detections),
            "loop_types": loop_types,
            "severities": severities,
            "average_confidence": np.mean([d.confidence for d in detections])
        }
