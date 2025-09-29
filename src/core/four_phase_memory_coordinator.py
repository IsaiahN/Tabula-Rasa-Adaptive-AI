#!/usr/bin/env python3
"""
4-Phase Memory Optimization Coordinator

This module implements the unified coordinator for the complete 4-Phase Memory Optimization
system, integrating all phases with the Governor orchestration for seamless meta-cognitive
memory management.

The 4 phases are:
1. Pattern Recognition Engine - Memory access pattern detection and optimization
2. Hierarchical Memory Clustering - Intelligent memory grouping and hierarchy  
3. Architect Evolution Engine - Autonomous architectural evolution
4. Performance Optimization Engine - Real-time performance maximization

Key Features:
- Unified coordination of all 4 phases
- Cross-phase intelligence sharing
- Governor integration for meta-cognitive supervision
- Database storage for all phase data
- Real-time monitoring and optimization
- Autonomous operation with minimal human intervention
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from collections import deque, defaultdict

# Import phase implementations
from .memory_pattern_optimizer import MemoryPatternOptimizer
from .hierarchical_memory_clusterer import HierarchicalMemoryClusterer
from .architect_evolution_engine import ArchitectEvolutionEngine
from .performance_optimization_engine import PerformanceOptimizationEngine

# Import database integration
from ..database.system_integration import get_system_integration

logger = logging.getLogger(__name__)

@dataclass
class PhaseStatus:
    """Status information for a single phase."""
    phase_id: str
    phase_name: str
    status: str  # 'active', 'inactive', 'error', 'initializing'
    last_update: float
    performance_metrics: Dict[str, Any]
    optimization_count: int
    error_count: int
    last_error: Optional[str] = None

@dataclass
class CrossPhaseIntelligence:
    """Intelligence shared between phases."""
    source_phase: str
    target_phases: List[str]
    intelligence_type: str  # 'pattern_insight', 'cluster_insight', 'architect_insight', 'performance_insight'
    data: Dict[str, Any]
    confidence: float
    timestamp: float
    priority: int  # 1-5, higher is more important

@dataclass
class SystemOptimizationResult:
    """Result of a system-wide optimization cycle."""
    cycle_id: str
    timestamp: float
    phases_optimized: List[str]
    performance_improvements: Dict[str, float]
    new_insights: List[CrossPhaseIntelligence]
    optimization_duration: float
    success: bool
    error_message: Optional[str] = None

class FourPhaseMemoryCoordinator:
    """
    Unified coordinator for the complete 4-Phase Memory Optimization system.
    
    This class orchestrates all four phases of memory optimization, enabling
    cross-phase intelligence sharing and coordinated optimization under Governor
    supervision.
    """
    
    def __init__(self, persistence_dir: Optional[Path] = None):
        self.persistence_dir = persistence_dir or Path("continuous_learning_data")
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database integration
        self.integration = get_system_integration()
        
        # Phase implementations
        self.phases = {}
        self.phase_status = {}
        self.cross_phase_intelligence = deque(maxlen=1000)
        self.optimization_history = deque(maxlen=100)
        
        # Coordination state
        self.is_initialized = False
        self.last_optimization_cycle = None
        self.optimization_interval = 300  # 5 minutes
        self.last_cycle_time = 0
        
        # Performance tracking
        self.performance_metrics = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'cross_phase_insights_generated': 0,
            'system_performance_improvement': 0.0,
            'last_cycle_duration': 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize all four phases of the memory optimization system."""
        try:
            logger.info("Initializing 4-Phase Memory Optimization System...")
            
            # Phase 1: Pattern Recognition Engine
            self.phases['pattern_recognition'] = MemoryPatternOptimizer(
                window_size=1000,
                pattern_threshold=0.05
            )
            self.phase_status['pattern_recognition'] = PhaseStatus(
                phase_id='pattern_recognition',
                phase_name='Pattern Recognition Engine',
                status='initializing',
                last_update=time.time(),
                performance_metrics={},
                optimization_count=0,
                error_count=0
            )
            
            # Phase 2: Hierarchical Memory Clustering
            self.phases['memory_clustering'] = HierarchicalMemoryClusterer(
                max_clusters=100
            )
            self.phase_status['memory_clustering'] = PhaseStatus(
                phase_id='memory_clustering',
                phase_name='Hierarchical Memory Clustering',
                status='initializing',
                last_update=time.time(),
                performance_metrics={},
                optimization_count=0,
                error_count=0
            )
            
            # Phase 3: Architect Evolution Engine
            self.phases['architect_evolution'] = ArchitectEvolutionEngine(
                persistence_dir=self.persistence_dir
            )
            self.phase_status['architect_evolution'] = PhaseStatus(
                phase_id='architect_evolution',
                phase_name='Architect Evolution Engine',
                status='initializing',
                last_update=time.time(),
                performance_metrics={},
                optimization_count=0,
                error_count=0
            )
            
            # Phase 4: Performance Optimization Engine
            self.phases['performance_optimization'] = PerformanceOptimizationEngine(
                persistence_dir=self.persistence_dir
            )
            self.phase_status['performance_optimization'] = PhaseStatus(
                phase_id='performance_optimization',
                phase_name='Performance Optimization Engine',
                status='initializing',
                last_update=time.time(),
                performance_metrics={},
                optimization_count=0,
                error_count=0
            )
            
            # Initialize all phases
            for phase_id, phase in self.phases.items():
                try:
                    if hasattr(phase, 'initialize'):
                        await phase.initialize()
                    self.phase_status[phase_id].status = 'active'
                    self.phase_status[phase_id].last_update = time.time()
                    logger.info(f"Phase {phase_id} initialized successfully")
                except Exception as e:
                    self.phase_status[phase_id].status = 'error'
                    self.phase_status[phase_id].last_error = str(e)
                    self.phase_status[phase_id].error_count += 1
                    logger.error(f"Failed to initialize phase {phase_id}: {e}")
            
            self.is_initialized = True
            logger.info("4-Phase Memory Optimization System initialized successfully")
            
            # Log to database
            await self.integration.log_system_event(
                level="INFO",
                component="four_phase_memory_coordinator",
                message="4-Phase Memory Optimization System initialized",
                data={
                    "phases_initialized": len([p for p in self.phase_status.values() if p.status == 'active']),
                    "total_phases": len(self.phase_status)
                },
                session_id="system_init"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize 4-Phase Memory Coordinator: {e}")
            return False
    
    async def process_memory_data(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process memory data through all four phases.
        
        Args:
            memory_data: Memory access data to process
            
        Returns:
            Dictionary containing processing results from all phases
        """
        if not self.is_initialized:
            logger.warning("4-Phase Memory Coordinator not initialized")
            return {'error': 'System not initialized'}
        
        try:
            results = {}
            
            # Phase 1: Pattern Recognition
            if self.phase_status['pattern_recognition'].status == 'active':
                try:
                    pattern_result = await self._process_pattern_recognition(memory_data)
                    results['pattern_recognition'] = pattern_result
                    self.phase_status['pattern_recognition'].optimization_count += 1
                except Exception as e:
                    logger.error(f"Pattern recognition phase failed: {e}")
                    self.phase_status['pattern_recognition'].error_count += 1
                    self.phase_status['pattern_recognition'].last_error = str(e)
            
            # Phase 2: Memory Clustering
            if self.phase_status['memory_clustering'].status == 'active':
                try:
                    clustering_result = await self._process_memory_clustering(memory_data, results.get('pattern_recognition', {}))
                    results['memory_clustering'] = clustering_result
                    self.phase_status['memory_clustering'].optimization_count += 1
                except Exception as e:
                    logger.error(f"Memory clustering phase failed: {e}")
                    self.phase_status['memory_clustering'].error_count += 1
                    self.phase_status['memory_clustering'].last_error = str(e)
            
            # Phase 3: Architect Evolution
            if self.phase_status['architect_evolution'].status == 'active':
                try:
                    architect_result = await self._process_architect_evolution(memory_data, results)
                    results['architect_evolution'] = architect_result
                    self.phase_status['architect_evolution'].optimization_count += 1
                except Exception as e:
                    logger.error(f"Architect evolution phase failed: {e}")
                    self.phase_status['architect_evolution'].error_count += 1
                    self.phase_status['architect_evolution'].last_error = str(e)
            
            # Phase 4: Performance Optimization
            if self.phase_status['performance_optimization'].status == 'active':
                try:
                    performance_result = await self._process_performance_optimization(memory_data, results)
                    results['performance_optimization'] = performance_result
                    self.phase_status['performance_optimization'].optimization_count += 1
                except Exception as e:
                    logger.error(f"Performance optimization phase failed: {e}")
                    self.phase_status['performance_optimization'].error_count += 1
                    self.phase_status['performance_optimization'].last_error = str(e)
            
            # Update phase status
            for phase_id in self.phase_status:
                self.phase_status[phase_id].last_update = time.time()
            
            # Generate cross-phase intelligence
            await self._generate_cross_phase_intelligence(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to process memory data: {e}")
            return {'error': str(e)}
    
    async def _process_pattern_recognition(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through Phase 1: Pattern Recognition Engine."""
        pattern_engine = self.phases['pattern_recognition']
        
        # Record memory access for pattern analysis
        pattern_engine.record_memory_access(memory_data)
        
        # Analyze patterns
        pattern_analysis = pattern_engine.analyze_access_patterns()
        
        # Get optimization suggestions
        optimization_suggestions = pattern_engine.get_optimization_suggestions()
        
        return {
            'patterns_detected': pattern_analysis.get('patterns_detected', 0),
            'optimization_suggestions': optimization_suggestions,
            'efficiency_improvement': pattern_analysis.get('efficiency_improvement', 0.0),
            'pattern_metrics': pattern_analysis
        }
    
    async def _process_memory_clustering(self, memory_data: Dict[str, Any], pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through Phase 2: Hierarchical Memory Clustering."""
        clusterer = self.phases['memory_clustering']
        
        # Create intelligent clusters
        memories = [memory_data] if isinstance(memory_data, dict) else memory_data
        clusters = clusterer.create_intelligent_clusters(memories)
        
        # Analyze cluster health
        cluster_health = {}
        for cluster_id, cluster in clusters.items():
            cluster_health[cluster_id] = cluster.get_cluster_health()
        
        # Get clustering insights
        clustering_insights = clusterer.get_clustering_insights()
        
        return {
            'clusters_created': len(clusters),
            'cluster_health': cluster_health,
            'clustering_insights': clustering_insights,
            'pattern_integration': pattern_data
        }
    
    async def _process_architect_evolution(self, memory_data: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through Phase 3: Architect Evolution Engine."""
        architect = self.phases['architect_evolution']
        
        # Generate architectural insights
        insights = architect.generate_architectural_insights(previous_results)
        
        # Execute evolution strategies
        evolution_results = architect.execute_evolution_strategies(insights)
        
        # Get evolution status
        evolution_status = architect.get_evolution_status()
        
        return {
            'insights_generated': len(insights),
            'evolution_results': evolution_results,
            'evolution_status': evolution_status,
            'cross_phase_integration': previous_results
        }
    
    async def _process_performance_optimization(self, memory_data: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through Phase 4: Performance Optimization Engine."""
        performance_engine = self.phases['performance_optimization']
        
        # Update performance metrics
        performance_engine.update_performance_metrics(memory_data)
        
        # Generate optimization recommendations
        optimization_recommendations = performance_engine.generate_optimization_recommendations()
        
        # Apply performance optimizations
        optimization_results = performance_engine.apply_optimizations(optimization_recommendations)
        
        # Get performance status
        performance_status = performance_engine.get_performance_status()
        
        return {
            'optimization_recommendations': optimization_recommendations,
            'optimization_results': optimization_results,
            'performance_status': performance_status,
            'cross_phase_optimization': previous_results
        }
    
    async def _generate_cross_phase_intelligence(self, results: Dict[str, Any]) -> None:
        """Generate cross-phase intelligence from processing results."""
        try:
            # Pattern Recognition insights for other phases
            if 'pattern_recognition' in results:
                pattern_insight = CrossPhaseIntelligence(
                    source_phase='pattern_recognition',
                    target_phases=['memory_clustering', 'architect_evolution', 'performance_optimization'],
                    intelligence_type='pattern_insight',
                    data=results['pattern_recognition'],
                    confidence=0.8,
                    timestamp=time.time(),
                    priority=3
                )
                self.cross_phase_intelligence.append(pattern_insight)
            
            # Memory Clustering insights for other phases
            if 'memory_clustering' in results:
                cluster_insight = CrossPhaseIntelligence(
                    source_phase='memory_clustering',
                    target_phases=['architect_evolution', 'performance_optimization'],
                    intelligence_type='cluster_insight',
                    data=results['memory_clustering'],
                    confidence=0.7,
                    timestamp=time.time(),
                    priority=4
                )
                self.cross_phase_intelligence.append(cluster_insight)
            
            # Architect Evolution insights for other phases
            if 'architect_evolution' in results:
                architect_insight = CrossPhaseIntelligence(
                    source_phase='architect_evolution',
                    target_phases=['performance_optimization'],
                    intelligence_type='architect_insight',
                    data=results['architect_evolution'],
                    confidence=0.9,
                    timestamp=time.time(),
                    priority=5
                )
                self.cross_phase_intelligence.append(architect_insight)
            
            self.performance_metrics['cross_phase_insights_generated'] += 3
            
        except Exception as e:
            logger.error(f"Failed to generate cross-phase intelligence: {e}")
    
    async def run_optimization_cycle(self) -> SystemOptimizationResult:
        """
        Run a complete optimization cycle across all phases.
        
        Returns:
            SystemOptimizationResult containing cycle results
        """
        cycle_start_time = time.time()
        cycle_id = f"cycle_{int(cycle_start_time)}"
        
        try:
            logger.info(f"Starting optimization cycle {cycle_id}")
            
            # Collect current system state
            current_state = await self._collect_system_state()
            
            # Run optimization for each phase
            optimization_results = {}
            phases_optimized = []
            
            for phase_id, phase in self.phases.items():
                if self.phase_status[phase_id].status == 'active':
                    try:
                        if hasattr(phase, 'optimize'):
                            result = await phase.optimize(current_state)
                            optimization_results[phase_id] = result
                            phases_optimized.append(phase_id)
                            self.phase_status[phase_id].optimization_count += 1
                    except Exception as e:
                        logger.error(f"Phase {phase_id} optimization failed: {e}")
                        self.phase_status[phase_id].error_count += 1
                        self.phase_status[phase_id].last_error = str(e)
            
            # Calculate performance improvements
            performance_improvements = await self._calculate_performance_improvements(optimization_results)
            
            # Generate new insights
            new_insights = await self._generate_optimization_insights(optimization_results)
            
            # Update performance metrics
            self.performance_metrics['total_optimizations'] += 1
            self.performance_metrics['successful_optimizations'] += len(phases_optimized)
            self.performance_metrics['system_performance_improvement'] += sum(performance_improvements.values())
            
            cycle_duration = time.time() - cycle_start_time
            self.performance_metrics['last_cycle_duration'] = cycle_duration
            
            result = SystemOptimizationResult(
                cycle_id=cycle_id,
                timestamp=cycle_start_time,
                phases_optimized=phases_optimized,
                performance_improvements=performance_improvements,
                new_insights=new_insights,
                optimization_duration=cycle_duration,
                success=True
            )
            
            self.optimization_history.append(result)
            self.last_optimization_cycle = result
            self.last_cycle_time = cycle_start_time
            
            logger.info(f"Optimization cycle {cycle_id} completed successfully in {cycle_duration:.2f}s")
            
            # Log to database
            await self.integration.log_system_event(
                level="INFO",
                component="four_phase_memory_coordinator",
                message=f"Optimization cycle {cycle_id} completed",
                data=asdict(result),
                session_id=cycle_id
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization cycle {cycle_id} failed: {e}")
            
            result = SystemOptimizationResult(
                cycle_id=cycle_id,
                timestamp=cycle_start_time,
                phases_optimized=[],
                performance_improvements={},
                new_insights=[],
                optimization_duration=time.time() - cycle_start_time,
                success=False,
                error_message=str(e)
            )
            
            return result
    
    async def _collect_system_state(self) -> Dict[str, Any]:
        """Collect current system state for optimization."""
        return {
            'phase_status': {pid: asdict(status) for pid, status in self.phase_status.items()},
            'performance_metrics': self.performance_metrics.copy(),
            'cross_phase_intelligence_count': len(self.cross_phase_intelligence),
            'last_optimization_cycle': asdict(self.last_optimization_cycle) if self.last_optimization_cycle else None
        }
    
    async def _calculate_performance_improvements(self, optimization_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance improvements from optimization results."""
        improvements = {}
        
        for phase_id, result in optimization_results.items():
            if isinstance(result, dict) and 'performance_improvement' in result:
                improvements[phase_id] = result['performance_improvement']
            else:
                improvements[phase_id] = 0.0
        
        return improvements
    
    async def _generate_optimization_insights(self, optimization_results: Dict[str, Any]) -> List[CrossPhaseIntelligence]:
        """Generate new insights from optimization results."""
        insights = []
        
        for phase_id, result in optimization_results.items():
            if isinstance(result, dict) and 'insights' in result:
                insight = CrossPhaseIntelligence(
                    source_phase=phase_id,
                    target_phases=[pid for pid in self.phases.keys() if pid != phase_id],
                    intelligence_type='optimization_insight',
                    data=result['insights'],
                    confidence=0.8,
                    timestamp=time.time(),
                    priority=4
                )
                insights.append(insight)
        
        return insights
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'is_initialized': self.is_initialized,
            'phase_status': {pid: asdict(status) for pid, status in self.phase_status.items()},
            'performance_metrics': self.performance_metrics.copy(),
            'cross_phase_intelligence_count': len(self.cross_phase_intelligence),
            'last_optimization_cycle': asdict(self.last_optimization_cycle) if self.last_optimization_cycle else None,
            'next_optimization_due': self.last_cycle_time + self.optimization_interval - time.time()
        }
    
    async def cleanup(self) -> None:
        """Clean up all phases and resources."""
        try:
            for phase_id, phase in self.phases.items():
                if hasattr(phase, 'cleanup'):
                    await phase.cleanup()
            
            self.phases.clear()
            self.phase_status.clear()
            self.cross_phase_intelligence.clear()
            self.optimization_history.clear()
            
            self.is_initialized = False
            
            logger.info("4-Phase Memory Coordinator cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Singleton instance storage
_four_phase_coordinator_instance = None

# Factory function for easy integration with singleton pattern
def create_four_phase_memory_coordinator(persistence_dir: Optional[Path] = None) -> FourPhaseMemoryCoordinator:
    """Create a 4-Phase Memory Coordinator instance.

    Uses singleton pattern to prevent duplicate instances and redundant initializations.
    """
    global _four_phase_coordinator_instance

    if _four_phase_coordinator_instance is None:
        logger.info("4-Phase Memory Coordinator initialized")
        _four_phase_coordinator_instance = FourPhaseMemoryCoordinator(persistence_dir)
    else:
        # Instance already exists, don't log duplicate initialization
        pass

    return _four_phase_coordinator_instance

def get_four_phase_memory_coordinator() -> Optional[FourPhaseMemoryCoordinator]:
    """Get existing singleton instance without creating a new one."""
    return _four_phase_coordinator_instance
