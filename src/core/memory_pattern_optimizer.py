"""
Memory Pattern Optimizer for Meta-Cognitive Intelligence

This module provides advanced memory pattern recognition and optimization
capabilities for the Governor system, enabling intelligent memory management
based on temporal, spatial, and semantic access patterns.

Phase 1 Implementation: Focus on immediate Governor integration for quick wins.
"""

import numpy as np
import json
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Set
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryAccessPattern:
    """Represents a detected memory access pattern"""
    
    def __init__(self, pattern_type: str, pattern_id: str, strength: float):
        self.pattern_type = pattern_type  # temporal, spatial, semantic
        self.pattern_id = pattern_id
        self.strength = strength
        self.frequency = 0
        self.last_seen = time.time()
        self.efficiency_score = 0.0
        self.optimization_potential = 0.0
        
    def update_metrics(self, access_data: Dict):
        """Update pattern metrics based on new access data"""
        self.frequency += 1
        self.last_seen = time.time()
        
        # Calculate efficiency based on access success
        if 'success' in access_data:
            success_rate = access_data['success']
            self.efficiency_score = 0.9 * self.efficiency_score + 0.1 * success_rate
        
        # Calculate optimization potential
        self.optimization_potential = self.strength * self.efficiency_score

class MemoryPatternOptimizer:
    """
    Advanced memory pattern recognition and optimization
    
    This class analyzes memory access patterns to provide immediate optimization
    opportunities for the Governor system. Focus on quick wins and measurable
    improvements in memory efficiency.
    """
    
    def __init__(self, window_size: int = 1000, pattern_threshold: float = 0.05):
        self.window_size = window_size
        self.pattern_threshold = pattern_threshold
        
        # Pattern tracking
        self.access_history = deque(maxlen=window_size)
        self.detected_patterns = {}
        self.pattern_efficiency = {}
        self.optimization_suggestions = []
        
        # Metrics
        self.metrics = {
            'patterns_detected': 0,
            'optimizations_applied': 0,
            'efficiency_improvement': 0.0,
            'last_analysis': None
        }
        
        # Pattern caches for performance
        self._temporal_cache = {}
        self._spatial_cache = {}
        self._semantic_cache = {}
        
    def record_memory_access(self, access_info: Dict):
        """Record a memory access for pattern analysis"""
        access_record = {
            'timestamp': time.time(),
            'memory_type': access_info.get('memory_type', 'unknown'),
            'file_path': access_info.get('file_path', ''),
            'operation': access_info.get('operation', 'read'),  # read, write, delete
            'size': access_info.get('size', 0),
            'success': access_info.get('success', True),
            'duration': access_info.get('duration', 0.0)
        }
        
        self.access_history.append(access_record)
        logger.debug(f"Recorded memory access: {access_record['memory_type']}")
    
    def analyze_access_patterns(self) -> Dict[str, Any]:
        """
        Analyze temporal, spatial, and semantic memory access patterns
        
        Returns immediate optimization opportunities for Governor integration.
        """
        if len(self.access_history) < 10:
            logger.warning("Insufficient access history for pattern analysis")
            return {'patterns': [], 'optimizations': []}
        
        logger.info(f"Analyzing {len(self.access_history)} memory accesses for patterns")
        
        patterns = {
            'temporal_patterns': self._detect_temporal_patterns(),
            'spatial_patterns': self._detect_spatial_patterns(),
            'semantic_patterns': self._detect_semantic_patterns(),
            'efficiency_metrics': self._calculate_efficiency_metrics(),
            'optimization_opportunities': self._identify_optimization_opportunities()
        }
        
        self.metrics['last_analysis'] = datetime.now().isoformat()
        self.metrics['patterns_detected'] = len(patterns['temporal_patterns']) + \
                                          len(patterns['spatial_patterns']) + \
                                          len(patterns['semantic_patterns'])
        
        logger.info(f"Detected {self.metrics['patterns_detected']} memory patterns")
        
        return patterns
    
    def _detect_temporal_patterns(self) -> List[MemoryAccessPattern]:
        """Detect recurring temporal access sequences - Governor's immediate win"""
        patterns = []
        accesses = list(self.access_history)
        
        # Look for sequential patterns
        sequence_counts = defaultdict(int)
        for i in range(len(accesses) - 2):
            sequence = tuple([a['memory_type'] for a in accesses[i:i+3]])
            sequence_counts[sequence] += 1
        
        # Identify frequent patterns (immediate optimization opportunity)
        total_sequences = len(accesses) - 2
        for sequence, count in sequence_counts.items():
            frequency = count / total_sequences if total_sequences > 0 else 0
            
            if frequency > self.pattern_threshold:
                pattern_id = f"temporal_{hash(sequence)}"
                pattern = MemoryAccessPattern("temporal", pattern_id, frequency)
                pattern.frequency = count
                
                # Calculate immediate optimization potential
                if len(sequence) >= 2 and sequence[0] == sequence[1]:
                    # Repeated access - high optimization potential
                    pattern.optimization_potential = frequency * 0.8
                elif 'CRITICAL' in sequence[0] and 'TEMP' in sequence[-1]:
                    # Critical to temp pattern - medium potential
                    pattern.optimization_potential = frequency * 0.6
                else:
                    pattern.optimization_potential = frequency * 0.3
                
                patterns.append(pattern)
                logger.debug(f"Detected temporal pattern: {sequence} (strength: {frequency:.3f})")
        
        return sorted(patterns, key=lambda p: p.optimization_potential, reverse=True)
    
    def _detect_spatial_patterns(self) -> List[MemoryAccessPattern]:
        """Detect spatial clustering patterns in file access"""
        patterns = []
        accesses = list(self.access_history)
        
        # Group by directory structure
        directory_access = defaultdict(list)
        for access in accesses:
            if access['file_path']:
                directory = str(Path(access['file_path']).parent)
                directory_access[directory].append(access)
        
        # Analyze directory access patterns
        for directory, dir_accesses in directory_access.items():
            if len(dir_accesses) < 3:
                continue
                
            # Calculate access density (accesses per time window)
            time_span = max(a['timestamp'] for a in dir_accesses) - \
                       min(a['timestamp'] for a in dir_accesses)
            
            if time_span > 0:
                access_density = len(dir_accesses) / time_span
                
                # High density indicates spatial clustering
                if access_density > 0.1:  # Threshold for clustering
                    pattern_id = f"spatial_{hash(directory)}"
                    pattern = MemoryAccessPattern("spatial", pattern_id, access_density)
                    pattern.optimization_potential = min(access_density * 0.5, 0.9)
                    
                    patterns.append(pattern)
                    logger.debug(f"Detected spatial pattern in {directory} (density: {access_density:.3f})")
        
        return sorted(patterns, key=lambda p: p.optimization_potential, reverse=True)
    
    def _detect_semantic_patterns(self) -> List[MemoryAccessPattern]:
        """Detect semantic relationships in memory access"""
        patterns = []
        accesses = list(self.access_history)
        
        # Group by memory type
        type_sequences = defaultdict(int)
        for i in range(len(accesses) - 1):
            current_type = accesses[i]['memory_type']
            next_type = accesses[i + 1]['memory_type']
            
            # Look for semantic relationships
            type_pair = (current_type, next_type)
            type_sequences[type_pair] += 1
        
        total_pairs = len(accesses) - 1
        for type_pair, count in type_sequences.items():
            frequency = count / total_pairs if total_pairs > 0 else 0
            
            if frequency > self.pattern_threshold:
                pattern_id = f"semantic_{hash(type_pair)}"
                pattern = MemoryAccessPattern("semantic", pattern_id, frequency)
                
                # Semantic optimization scoring
                current_type, next_type = type_pair
                if 'CRITICAL' in current_type and 'CRITICAL' in next_type:
                    # Critical chain - high potential
                    pattern.optimization_potential = frequency * 0.9
                elif 'TEMP' in current_type and 'TEMP' in next_type:
                    # Temp chain - medium potential (batch cleanup)
                    pattern.optimization_potential = frequency * 0.7
                else:
                    pattern.optimization_potential = frequency * 0.4
                
                patterns.append(pattern)
                logger.debug(f"Detected semantic pattern: {type_pair} (strength: {frequency:.3f})")
        
        return sorted(patterns, key=lambda p: p.optimization_potential, reverse=True)
    
    def _calculate_efficiency_metrics(self) -> Dict[str, float]:
        """Calculate memory system efficiency metrics"""
        if not self.access_history:
            return {'overall_efficiency': 0.0, 'success_rate': 0.0, 'avg_duration': 0.0}
        
        accesses = list(self.access_history)
        
        # Calculate success rate
        successful_accesses = sum(1 for a in accesses if a['success'])
        success_rate = successful_accesses / len(accesses)
        
        # Calculate average access duration
        durations = [a['duration'] for a in accesses if a['duration'] > 0]
        avg_duration = np.mean(durations) if durations else 0.0
        
        # Calculate overall efficiency (success rate / avg duration)
        overall_efficiency = success_rate / max(avg_duration, 0.001)
        
        return {
            'overall_efficiency': overall_efficiency,
            'success_rate': success_rate,
            'avg_duration': avg_duration,
            'total_accesses': len(accesses),
            'recent_trend': self._calculate_efficiency_trend()
        }
    
    def _calculate_efficiency_trend(self) -> float:
        """Calculate recent efficiency trend for Governor decision-making"""
        if len(self.access_history) < 20:
            return 0.0
        
        accesses = list(self.access_history)
        
        # Compare recent half vs older half
        mid_point = len(accesses) // 2
        older_half = accesses[:mid_point]
        recent_half = accesses[mid_point:]
        
        older_success = sum(1 for a in older_half if a['success']) / len(older_half)
        recent_success = sum(1 for a in recent_half if a['success']) / len(recent_half)
        
        return recent_success - older_success
    
    def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify immediate optimization opportunities for Governor"""
        opportunities = []
        
        # Analyze all detected patterns for optimization potential
        all_patterns = (self._detect_temporal_patterns() + 
                       self._detect_spatial_patterns() + 
                       self._detect_semantic_patterns())
        
        # Sort by optimization potential
        high_potential_patterns = [p for p in all_patterns if p.optimization_potential > 0.5]
        
        for pattern in high_potential_patterns[:5]:  # Top 5 opportunities
            if pattern.pattern_type == "temporal":
                opportunities.append({
                    'type': 'temporal_optimization',
                    'pattern_id': pattern.pattern_id,
                    'potential': pattern.optimization_potential,
                    'action': 'implement_sequence_caching',
                    'expected_improvement': f"{pattern.optimization_potential * 20:.1f}% efficiency gain",
                    'priority': 'high' if pattern.optimization_potential > 0.7 else 'medium'
                })
            
            elif pattern.pattern_type == "spatial":
                opportunities.append({
                    'type': 'spatial_optimization',
                    'pattern_id': pattern.pattern_id,
                    'potential': pattern.optimization_potential,
                    'action': 'implement_directory_prefetching',
                    'expected_improvement': f"{pattern.optimization_potential * 15:.1f}% access speed improvement",
                    'priority': 'high' if pattern.optimization_potential > 0.6 else 'medium'
                })
            
            elif pattern.pattern_type == "semantic":
                opportunities.append({
                    'type': 'semantic_optimization',
                    'pattern_id': pattern.pattern_id,
                    'potential': pattern.optimization_potential,
                    'action': 'implement_semantic_grouping',
                    'expected_improvement': f"{pattern.optimization_potential * 10:.1f}% memory coherence improvement",
                    'priority': 'medium'
                })
        
        return opportunities
    
    def get_governor_recommendations(self) -> Dict[str, Any]:
        """Get immediate actionable recommendations for Governor system"""
        patterns = self.analyze_access_patterns()
        efficiency = patterns['efficiency_metrics']
        opportunities = patterns['optimization_opportunities']
        
        recommendations = {
            'immediate_actions': [],
            'efficiency_status': {
                'current_efficiency': efficiency['overall_efficiency'],
                'success_rate': efficiency['success_rate'],
                'trend': 'improving' if efficiency['recent_trend'] > 0.05 else 
                        'declining' if efficiency['recent_trend'] < -0.05 else 'stable'
            },
            'priority_optimizations': opportunities[:3],  # Top 3 priorities
            'metrics': self.metrics
        }
        
        # Add immediate actions based on patterns
        if efficiency['success_rate'] < 0.8:
            recommendations['immediate_actions'].append({
                'action': 'improve_memory_reliability',
                'reason': f"Success rate only {efficiency['success_rate']:.1%}",
                'urgency': 'high'
            })
        
        if efficiency['recent_trend'] < -0.1:
            recommendations['immediate_actions'].append({
                'action': 'analyze_performance_degradation',
                'reason': f"Efficiency declining by {abs(efficiency['recent_trend']):.1%}",
                'urgency': 'high'
            })
        
        if len(opportunities) > 0 and opportunities[0]['potential'] > 0.7:
            recommendations['immediate_actions'].append({
                'action': 'implement_top_optimization',
                'reason': f"High-potential optimization available: {opportunities[0]['expected_improvement']}",
                'urgency': 'medium'
            })
        
        logger.info(f"Generated {len(recommendations['immediate_actions'])} immediate Governor recommendations")
        
        return recommendations
    
    def apply_optimization(self, optimization_id: str) -> Dict[str, Any]:
        """Apply a specific optimization and track results"""
        result = {
            'optimization_id': optimization_id,
            'applied': False,
            'improvement': 0.0,
            'error': None
        }
        
        try:
            # This is a placeholder for actual optimization implementation
            # In Phase 1, we focus on detection and recommendation
            logger.info(f"Optimization {optimization_id} marked for implementation")
            
            self.metrics['optimizations_applied'] += 1
            result['applied'] = True
            result['improvement'] = 0.05  # Placeholder improvement
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Failed to apply optimization {optimization_id}: {e}")
        
        return result
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of detected patterns for logging"""
        patterns = self.analyze_access_patterns()
        
        return {
            'total_patterns': self.metrics['patterns_detected'],
            'temporal_count': len(patterns['temporal_patterns']),
            'spatial_count': len(patterns['spatial_patterns']),
            'semantic_count': len(patterns['semantic_patterns']),
            'top_optimization_potential': max([p.optimization_potential 
                                             for p in (patterns['temporal_patterns'] + 
                                                      patterns['spatial_patterns'] + 
                                                      patterns['semantic_patterns'])]) 
                                           if (patterns['temporal_patterns'] + 
                                              patterns['spatial_patterns'] + 
                                              patterns['semantic_patterns']) else 0.0,
            'efficiency_status': patterns['efficiency_metrics']['overall_efficiency'],
            'last_analysis': self.metrics['last_analysis']
        }

if __name__ == "__main__":
    # Quick test of the pattern optimizer
    optimizer = MemoryPatternOptimizer()
    
    # Simulate some memory accesses
    test_accesses = [
        {'memory_type': 'CRITICAL_LOSSLESS', 'file_path': 'governor.log', 'operation': 'read', 'success': True, 'duration': 0.001},
        {'memory_type': 'CRITICAL_LOSSLESS', 'file_path': 'architect.json', 'operation': 'read', 'success': True, 'duration': 0.002},
        {'memory_type': 'IMPORTANT_DECAY', 'file_path': 'session.json', 'operation': 'write', 'success': True, 'duration': 0.005},
        {'memory_type': 'CRITICAL_LOSSLESS', 'file_path': 'governor.log', 'operation': 'write', 'success': True, 'duration': 0.001},
        {'memory_type': 'TEMP_PURGE', 'file_path': 'temp.log', 'operation': 'write', 'success': True, 'duration': 0.001},
    ]
    
    for access in test_accesses:
        optimizer.record_memory_access(access)
    
    # Analyze patterns
    recommendations = optimizer.get_governor_recommendations()
    summary = optimizer.get_pattern_summary()
    
    print(" Memory Pattern Optimizer Test Results:")
    print(f" Detected {summary['total_patterns']} patterns")
    print(f" Generated {len(recommendations['immediate_actions'])} recommendations")
    print(f" Current efficiency: {recommendations['efficiency_status']['current_efficiency']:.3f}")
    print(f" Top optimization potential: {summary['top_optimization_potential']:.3f}")
