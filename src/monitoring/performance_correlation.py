#!/usr/bin/env python3
"""
Performance Correlation Analysis System

Analyzes correlations between different performance metrics, system events,
and learning outcomes to identify patterns and optimization opportunities.
"""

import logging
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum
import json
import time

from ..core.caching_system import UnifiedCachingSystem, CacheConfig


class CorrelationType(Enum):
    """Types of correlations."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    COMPLEX = "complex"


class MetricType(Enum):
    """Types of metrics for correlation analysis."""
    PERFORMANCE = "performance"
    LEARNING = "learning"
    SYSTEM = "system"
    RESOURCE = "resource"
    ERROR = "error"


@dataclass
class MetricDataPoint:
    """A single metric data point."""
    metric_name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrelationResult:
    """Result of correlation analysis."""
    metric1: str
    metric2: str
    correlation_coefficient: float
    correlation_type: CorrelationType
    confidence: float
    p_value: float
    sample_size: int
    timestamp: datetime
    analysis_window_hours: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceInsight:
    """Insight derived from correlation analysis."""
    insight_id: str
    insight_type: str
    description: str
    confidence: float
    metrics_involved: List[str]
    recommendations: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class CorrelationAnalyzer:
    """Analyzes correlations between metrics."""
    
    def __init__(self, window_size: int = 1000, min_samples: int = 50):
        self.window_size = window_size
        self.min_samples = min_samples
        self.metric_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.correlation_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
    def add_metric_data(self, metric_name: str, metric_type: MetricType, 
                       value: float, timestamp: datetime,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add metric data point."""
        data_point = MetricDataPoint(
            metric_name=metric_name,
            metric_type=metric_type,
            value=value,
            timestamp=timestamp,
            metadata=metadata or {}
        )
        self.metric_data[metric_name].append(data_point)
    
    def calculate_correlation(self, metric1: str, metric2: str, 
                            window_hours: int = 24) -> Optional[CorrelationResult]:
        """Calculate correlation between two metrics."""
        # Check cache first
        cache_key = f"{metric1}_{metric2}_{window_hours}"
        if cache_key in self.correlation_cache:
            cached_result, timestamp = self.correlation_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_result
        
        # Get data for both metrics
        data1 = self._get_metric_data(metric1, window_hours)
        data2 = self._get_metric_data(metric2, window_hours)
        
        if not data1 or not data2:
            return None
        
        # Align data by timestamp
        aligned_data = self._align_metric_data(data1, data2)
        
        if len(aligned_data) < self.min_samples:
            return None
        
        # Calculate correlation
        values1 = [dp.value for dp in aligned_data[metric1]]
        values2 = [dp.value for dp in aligned_data[metric2]]
        
        correlation_coeff = np.corrcoef(values1, values2)[0, 1]
        
        if np.isnan(correlation_coeff):
            return None
        
        # Determine correlation type
        if correlation_coeff > 0.3:
            corr_type = CorrelationType.POSITIVE
        elif correlation_coeff < -0.3:
            corr_type = CorrelationType.NEGATIVE
        elif abs(correlation_coeff) < 0.1:
            corr_type = CorrelationType.NEUTRAL
        else:
            corr_type = CorrelationType.COMPLEX
        
        # Calculate confidence (simplified)
        confidence = min(1.0, abs(correlation_coeff) * (len(aligned_data) / 100))
        
        # Calculate p-value (simplified)
        p_value = self._calculate_p_value(correlation_coeff, len(aligned_data))
        
        result = CorrelationResult(
            metric1=metric1,
            metric2=metric2,
            correlation_coefficient=correlation_coeff,
            correlation_type=corr_type,
            confidence=confidence,
            p_value=p_value,
            sample_size=len(aligned_data),
            timestamp=datetime.now(),
            analysis_window_hours=window_hours
        )
        
        # Cache result
        self.correlation_cache[cache_key] = (result, time.time())
        
        return result
    
    def _get_metric_data(self, metric_name: str, window_hours: int) -> List[MetricDataPoint]:
        """Get metric data within time window."""
        if metric_name not in self.metric_data:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        return [dp for dp in self.metric_data[metric_name] if dp.timestamp > cutoff_time]
    
    def _align_metric_data(self, data1: List[MetricDataPoint], 
                          data2: List[MetricDataPoint]) -> Dict[str, List[MetricDataPoint]]:
        """Align two metric datasets by timestamp."""
        # Create time windows for alignment
        time_tolerance = timedelta(minutes=5)  # 5-minute tolerance
        
        aligned_data = {data1[0].metric_name: [], data2[0].metric_name: []}
        
        for dp1 in data1:
            # Find closest data point in data2
            closest_dp2 = None
            min_time_diff = float('inf')
            
            for dp2 in data2:
                time_diff = abs((dp1.timestamp - dp2.timestamp).total_seconds())
                if time_diff < min_time_diff and time_diff <= time_tolerance.total_seconds():
                    min_time_diff = time_diff
                    closest_dp2 = dp2
            
            if closest_dp2:
                aligned_data[dp1.metric_name].append(dp1)
                aligned_data[closest_dp2.metric_name].append(closest_dp2)
        
        return aligned_data
    
    def _calculate_p_value(self, correlation_coeff: float, sample_size: int) -> float:
        """Calculate p-value for correlation coefficient."""
        if sample_size < 3:
            return 1.0
        
        # Simplified p-value calculation
        # In practice, you'd use proper statistical tests
        t_stat = correlation_coeff * np.sqrt((sample_size - 2) / (1 - correlation_coeff**2 + 1e-10))
        
        # Approximate p-value (simplified)
        if abs(t_stat) > 2.576:  # 99% confidence
            return 0.01
        elif abs(t_stat) > 1.96:  # 95% confidence
            return 0.05
        elif abs(t_stat) > 1.645:  # 90% confidence
            return 0.10
        else:
            return 0.50
    
    def find_strong_correlations(self, threshold: float = 0.7, 
                               window_hours: int = 24) -> List[CorrelationResult]:
        """Find strong correlations between all metric pairs."""
        strong_correlations = []
        metric_names = list(self.metric_data.keys())
        
        for i, metric1 in enumerate(metric_names):
            for metric2 in metric_names[i+1:]:
                correlation = self.calculate_correlation(metric1, metric2, window_hours)
                if correlation and abs(correlation.correlation_coefficient) >= threshold:
                    strong_correlations.append(correlation)
        
        return strong_correlations
    
    def analyze_metric_relationships(self, target_metric: str, 
                                   window_hours: int = 24) -> Dict[str, CorrelationResult]:
        """Analyze relationships between a target metric and all others."""
        relationships = {}
        metric_names = list(self.metric_data.keys())
        
        for metric_name in metric_names:
            if metric_name != target_metric:
                correlation = self.calculate_correlation(target_metric, metric_name, window_hours)
                if correlation:
                    relationships[metric_name] = correlation
        
        return relationships


class PerformanceInsightGenerator:
    """Generates insights from correlation analysis."""
    
    def __init__(self):
        self.insight_templates = self._initialize_insight_templates()
    
    def _initialize_insight_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize insight generation templates."""
        return {
            'performance_bottleneck': {
                'description_template': "High correlation between {metric1} and {metric2} suggests {metric1} may be a bottleneck for {metric2}",
                'recommendations': [
                    "Optimize {metric1} performance",
                    "Consider scaling {metric1} resources",
                    "Monitor {metric1} more closely"
                ]
            },
            'learning_efficiency': {
                'description_template': "Strong positive correlation between {metric1} and {metric2} indicates efficient learning",
                'recommendations': [
                    "Maintain current learning parameters",
                    "Consider increasing {metric1} if {metric2} needs improvement"
                ]
            },
            'resource_optimization': {
                'description_template': "Negative correlation between {metric1} and {metric2} suggests resource trade-offs",
                'recommendations': [
                    "Balance resource allocation between {metric1} and {metric2}",
                    "Monitor both metrics for optimal performance"
                ]
            },
            'error_propagation': {
                'description_template': "High correlation between {metric1} and error rates suggests error propagation",
                'recommendations': [
                    "Investigate error sources in {metric1}",
                    "Implement error handling for {metric1}",
                    "Add monitoring for {metric1} error conditions"
                ]
            }
        }
    
    def generate_insights(self, correlations: List[CorrelationResult]) -> List[PerformanceInsight]:
        """Generate insights from correlation results."""
        insights = []
        
        for correlation in correlations:
            insight = self._generate_insight_for_correlation(correlation)
            if insight:
                insights.append(insight)
        
        return insights
    
    def _generate_insight_for_correlation(self, correlation: CorrelationResult) -> Optional[PerformanceInsight]:
        """Generate insight for a specific correlation."""
        # Determine insight type based on correlation characteristics
        insight_type = self._determine_insight_type(correlation)
        
        if insight_type not in self.insight_templates:
            return None
        
        template = self.insight_templates[insight_type]
        
        # Generate description
        description = template['description_template'].format(
            metric1=correlation.metric1,
            metric2=correlation.metric2
        )
        
        # Generate recommendations
        recommendations = []
        for rec_template in template['recommendations']:
            recommendation = rec_template.format(
                metric1=correlation.metric1,
                metric2=correlation.metric2
            )
            recommendations.append(recommendation)
        
        return PerformanceInsight(
            insight_id=f"insight_{int(time.time())}_{correlation.metric1}_{correlation.metric2}",
            insight_type=insight_type,
            description=description,
            confidence=correlation.confidence,
            metrics_involved=[correlation.metric1, correlation.metric2],
            recommendations=recommendations,
            timestamp=datetime.now(),
            metadata={
                'correlation_coefficient': correlation.correlation_coefficient,
                'correlation_type': correlation.correlation_type.value,
                'sample_size': correlation.sample_size
            }
        )
    
    def _determine_insight_type(self, correlation: CorrelationResult) -> str:
        """Determine insight type based on correlation characteristics."""
        metric1 = correlation.metric1.lower()
        metric2 = correlation.metric2.lower()
        coeff = correlation.correlation_coefficient
        
        # Performance bottleneck detection
        if (('cpu' in metric1 or 'memory' in metric1) and 
            ('performance' in metric2 or 'throughput' in metric2) and 
            coeff > 0.7):
            return 'performance_bottleneck'
        
        # Learning efficiency
        if (('learning' in metric1 or 'efficiency' in metric1) and 
            ('score' in metric2 or 'accuracy' in metric2) and 
            coeff > 0.6):
            return 'learning_efficiency'
        
        # Resource optimization
        if (('cpu' in metric1 and 'memory' in metric2) or 
            ('memory' in metric1 and 'cpu' in metric2)) and coeff < -0.5:
            return 'resource_optimization'
        
        # Error propagation
        if (('error' in metric1 or 'error' in metric2) and 
            abs(coeff) > 0.6):
            return 'error_propagation'
        
        # Default to performance bottleneck for strong correlations
        if abs(coeff) > 0.8:
            return 'performance_bottleneck'
        
        return None


class PerformanceCorrelationSystem:
    """Main performance correlation analysis system."""
    
    def __init__(self, cache_config: Optional[CacheConfig] = None):
        self.cache_config = cache_config or CacheConfig()
        self.cache = UnifiedCachingSystem(self.cache_config)
        
        # Core components
        self.correlation_analyzer = CorrelationAnalyzer()
        self.insight_generator = PerformanceInsightGenerator()
        
        # Results storage
        self.correlations: List[CorrelationResult] = []
        self.insights: List[PerformanceInsight] = []
        
        # Callbacks
        self.correlation_callbacks: List[Callable[[CorrelationResult], None]] = []
        self.insight_callbacks: List[Callable[[PerformanceInsight], None]] = []
        
        # Configuration
        self.analysis_intervals = [1, 6, 24]  # Hours
        self.correlation_thresholds = [0.5, 0.7, 0.9]  # Different thresholds
        
        self._initialized = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize the performance correlation system."""
        try:
            self._initialized = True
            self.logger.info("Performance correlation system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize correlation system: {e}")
            raise
    
    def add_metric_data(self, metric_name: str, metric_type: MetricType, 
                       value: float, timestamp: Optional[datetime] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add metric data for correlation analysis."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.correlation_analyzer.add_metric_data(
            metric_name, metric_type, value, timestamp, metadata
        )
        
        # Trigger analysis if enough data
        self._trigger_analysis()
    
    def _trigger_analysis(self) -> None:
        """Trigger correlation analysis."""
        # Find strong correlations
        for threshold in self.correlation_thresholds:
            correlations = self.correlation_analyzer.find_strong_correlations(
                threshold=threshold, window_hours=24
            )
            
            for correlation in correlations:
                if not self._correlation_exists(correlation):
                    self.correlations.append(correlation)
                    
                    # Trigger callbacks
                    for callback in self.correlation_callbacks:
                        try:
                            callback(correlation)
                        except Exception as e:
                            self.logger.error(f"Error in correlation callback: {e}")
        
        # Generate insights
        recent_correlations = self._get_recent_correlations(hours=24)
        if recent_correlations:
            insights = self.insight_generator.generate_insights(recent_correlations)
            
            for insight in insights:
                if not self._insight_exists(insight):
                    self.insights.append(insight)
                    
                    # Trigger callbacks
                    for callback in self.insight_callbacks:
                        try:
                            callback(insight)
                        except Exception as e:
                            self.logger.error(f"Error in insight callback: {e}")
    
    def _correlation_exists(self, correlation: CorrelationResult) -> bool:
        """Check if correlation already exists."""
        for existing in self.correlations:
            if (existing.metric1 == correlation.metric1 and 
                existing.metric2 == correlation.metric2 and
                abs(existing.timestamp - correlation.timestamp).total_seconds() < 3600):
                return True
        return False
    
    def _insight_exists(self, insight: PerformanceInsight) -> bool:
        """Check if insight already exists."""
        for existing in self.insights:
            if (existing.insight_type == insight.insight_type and
                set(existing.metrics_involved) == set(insight.metrics_involved) and
                abs(existing.timestamp - insight.timestamp).total_seconds() < 3600):
                return True
        return False
    
    def _get_recent_correlations(self, hours: int = 24) -> List[CorrelationResult]:
        """Get recent correlations."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [c for c in self.correlations if c.timestamp > cutoff_time]
    
    def add_correlation_callback(self, callback: Callable[[CorrelationResult], None]) -> None:
        """Add a callback for correlation results."""
        self.correlation_callbacks.append(callback)
    
    def add_insight_callback(self, callback: Callable[[PerformanceInsight], None]) -> None:
        """Add a callback for insights."""
        self.insight_callbacks.append(callback)
    
    def get_correlation_summary(self) -> Dict[str, Any]:
        """Get summary of correlation analysis."""
        recent_correlations = self._get_recent_correlations(24)
        
        # Count by type
        type_counts = defaultdict(int)
        for correlation in recent_correlations:
            type_counts[correlation.correlation_type.value] += 1
        
        # Count by strength
        strong_correlations = [c for c in recent_correlations if abs(c.correlation_coefficient) > 0.7]
        moderate_correlations = [c for c in recent_correlations if 0.5 <= abs(c.correlation_coefficient) <= 0.7]
        weak_correlations = [c for c in recent_correlations if abs(c.correlation_coefficient) < 0.5]
        
        return {
            'total_correlations': len(recent_correlations),
            'strong_correlations': len(strong_correlations),
            'moderate_correlations': len(moderate_correlations),
            'weak_correlations': len(weak_correlations),
            'type_distribution': dict(type_counts),
            'recent_insights': len([i for i in self.insights 
                                  if i.timestamp > datetime.now() - timedelta(hours=24)]),
            'metrics_analyzed': len(self.correlation_analyzer.metric_data)
        }
    
    def get_metric_relationships(self, metric_name: str) -> Dict[str, CorrelationResult]:
        """Get relationships for a specific metric."""
        return self.correlation_analyzer.analyze_metric_relationships(metric_name)
    
    def get_recent_insights(self, limit: int = 20) -> List[PerformanceInsight]:
        """Get recent insights."""
        insights = sorted(self.insights, key=lambda x: x.timestamp, reverse=True)
        return insights[:limit]
    
    def cleanup_old_data(self, max_age_hours: int = 168) -> None:  # 1 week default
        """Clean up old correlation data."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Clean up correlations
        self.correlations = [c for c in self.correlations if c.timestamp > cutoff_time]
        
        # Clean up insights
        self.insights = [i for i in self.insights if i.timestamp > cutoff_time]
        
        self.logger.info(f"Cleaned up correlation data older than {max_age_hours} hours")


# Global instance
_correlation_system = None

def get_correlation_system() -> PerformanceCorrelationSystem:
    """Get the global correlation system instance."""
    global _correlation_system
    if _correlation_system is None:
        _correlation_system = PerformanceCorrelationSystem()
        _correlation_system.initialize()
    return _correlation_system
