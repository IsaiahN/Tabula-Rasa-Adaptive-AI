"""
Cognitive Coordinator

Coordinates all 37 cognitive subsystems and provides unified monitoring and management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import json

from .base_subsystem import SubsystemStatus, SubsystemHealth
from .memory_subsystems import (
    MemoryAccessMonitor, MemoryConsolidationMonitor, MemoryFragmentationMonitor,
    SalientMemoryRetrievalMonitor, MemoryRegularizationMonitor, TemporalMemoryMonitor
)
from .learning_subsystems import (
    LearningProgressMonitor, MetaLearningMonitor, KnowledgeTransferMonitor,
    PatternRecognitionMonitor, CurriculumLearningMonitor, CrossSessionLearningMonitor
)
from .action_subsystems import (
    ActionIntelligenceMonitor, ActionExperimentationMonitor, EmergencyMovementMonitor,
    PredictiveCoordinatesMonitor, CoordinateSuccessMonitor
)
from .exploration_subsystems import (
    ExplorationStrategyMonitor, BoredomDetectionMonitor, StagnationDetectionMonitor,
    ContrarianStrategyMonitor, GoalInventionMonitor
)
from .energy_subsystems import (
    EnergySystemMonitor, SleepCycleMonitor, MidGameSleepMonitor, DeathManagerMonitor
)
from .visual_subsystems import (
    FrameAnalysisMonitor, BoundaryDetectionMonitor, MultiModalInputMonitor, VisualPatternMonitor
)
from .system_subsystems import (
    ResourceUtilizationMonitor, GradientFlowMonitor, UsageTrackingMonitor,
    AntiBiasWeightingMonitor, ClusterFormationMonitor, DangerZoneAvoidanceMonitor,
    SwarmIntelligenceMonitor, HebbianBonusesMonitor
)

logger = logging.getLogger(__name__)

class CognitiveCoordinator:
    """
    Coordinates all 37 cognitive subsystems and provides unified monitoring and management.
    
    This class serves as the central hub for all cognitive subsystems, providing:
    - Unified initialization and management
    - Cross-subsystem communication and coordination
    - System-wide health monitoring and alerting
    - Performance analytics and reporting
    - Database integration for persistent storage
    """
    
    def __init__(self):
        self.subsystems: Dict[str, Any] = {}
        self.coordination_status = "initializing"
        self.last_coordination_update = datetime.now()
        self.system_health = SubsystemHealth.EXCELLENT
        self.coordination_metrics = {}
        
        # Initialize all 37 subsystems
        self._initialize_subsystems()
        
        logger.info("Cognitive Coordinator initialized with 37 subsystems")
    
    def _initialize_subsystems(self) -> None:
        """Initialize all 37 cognitive subsystems."""
        # Memory subsystems (6)
        self.subsystems["memory_access"] = MemoryAccessMonitor()
        self.subsystems["memory_consolidation"] = MemoryConsolidationMonitor()
        self.subsystems["memory_fragmentation"] = MemoryFragmentationMonitor()
        self.subsystems["salient_memory_retrieval"] = SalientMemoryRetrievalMonitor()
        self.subsystems["memory_regularization"] = MemoryRegularizationMonitor()
        self.subsystems["temporal_memory"] = TemporalMemoryMonitor()
        
        # Learning subsystems (6)
        self.subsystems["learning_progress"] = LearningProgressMonitor()
        self.subsystems["meta_learning"] = MetaLearningMonitor()
        self.subsystems["knowledge_transfer"] = KnowledgeTransferMonitor()
        self.subsystems["pattern_recognition"] = PatternRecognitionMonitor()
        self.subsystems["curriculum_learning"] = CurriculumLearningMonitor()
        self.subsystems["cross_session_learning"] = CrossSessionLearningMonitor()
        
        # Action subsystems (5)
        self.subsystems["action_intelligence"] = ActionIntelligenceMonitor()
        self.subsystems["action_experimentation"] = ActionExperimentationMonitor()
        self.subsystems["emergency_movement"] = EmergencyMovementMonitor()
        self.subsystems["predictive_coordinates"] = PredictiveCoordinatesMonitor()
        self.subsystems["coordinate_success"] = CoordinateSuccessMonitor()
        
        # Exploration subsystems (5)
        self.subsystems["exploration_strategy"] = ExplorationStrategyMonitor()
        self.subsystems["boredom_detection"] = BoredomDetectionMonitor()
        self.subsystems["stagnation_detection"] = StagnationDetectionMonitor()
        self.subsystems["contrarian_strategy"] = ContrarianStrategyMonitor()
        self.subsystems["goal_invention"] = GoalInventionMonitor()
        
        # Energy subsystems (4)
        self.subsystems["energy_system"] = EnergySystemMonitor()
        self.subsystems["sleep_cycle"] = SleepCycleMonitor()
        self.subsystems["mid_game_sleep"] = MidGameSleepMonitor()
        self.subsystems["death_manager"] = DeathManagerMonitor()
        
        # Visual subsystems (4)
        self.subsystems["frame_analysis"] = FrameAnalysisMonitor()
        self.subsystems["boundary_detection"] = BoundaryDetectionMonitor()
        self.subsystems["multi_modal_input"] = MultiModalInputMonitor()
        self.subsystems["visual_pattern"] = VisualPatternMonitor()
        
        # System subsystems (8)
        self.subsystems["resource_utilization"] = ResourceUtilizationMonitor()
        self.subsystems["gradient_flow"] = GradientFlowMonitor()
        self.subsystems["usage_tracking"] = UsageTrackingMonitor()
        self.subsystems["anti_bias_weighting"] = AntiBiasWeightingMonitor()
        self.subsystems["cluster_formation"] = ClusterFormationMonitor()
        self.subsystems["danger_zone_avoidance"] = DangerZoneAvoidanceMonitor()
        self.subsystems["swarm_intelligence"] = SwarmIntelligenceMonitor()
        self.subsystems["hebbian_bonuses"] = HebbianBonusesMonitor()
        
        logger.info(f"Initialized {len(self.subsystems)} cognitive subsystems")
    
    async def initialize_all_subsystems(self) -> None:
        """Initialize all subsystems asynchronously."""
        try:
            self.coordination_status = "initializing"
            
            # Initialize all subsystems in parallel
            initialization_tasks = []
            for subsystem_id, subsystem in self.subsystems.items():
                task = asyncio.create_task(subsystem.initialize())
                initialization_tasks.append((subsystem_id, task))
            
            # Wait for all initializations to complete
            initialization_results = []
            for subsystem_id, task in initialization_tasks:
                try:
                    await task
                    initialization_results.append((subsystem_id, "success"))
                    logger.info(f"Subsystem {subsystem_id} initialized successfully")
                except Exception as e:
                    initialization_results.append((subsystem_id, f"error: {e}"))
                    logger.error(f"Failed to initialize subsystem {subsystem_id}: {e}")
            
            # Update coordination status
            successful_initializations = len([r for r in initialization_results if r[1] == "success"])
            total_subsystems = len(self.subsystems)
            
            if successful_initializations == total_subsystems:
                self.coordination_status = "active"
                self.system_health = SubsystemHealth.EXCELLENT
            elif successful_initializations >= total_subsystems * 0.8:
                self.coordination_status = "active"
                self.system_health = SubsystemHealth.GOOD
            elif successful_initializations >= total_subsystems * 0.5:
                self.coordination_status = "active"
                self.system_health = SubsystemHealth.WARNING
            else:
                self.coordination_status = "error"
                self.system_health = SubsystemHealth.CRITICAL
            
            self.last_coordination_update = datetime.now()
            logger.info(f"Coordination initialization complete: {successful_initializations}/{total_subsystems} subsystems active")
            
        except Exception as e:
            self.coordination_status = "error"
            self.system_health = SubsystemHealth.FAILED
            logger.error(f"Failed to initialize cognitive coordination: {e}")
            raise
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview."""
        try:
            # Get status from all subsystems
            subsystem_statuses = {}
            for subsystem_id, subsystem in self.subsystems.items():
                try:
                    status = await subsystem.get_status()
                    subsystem_statuses[subsystem_id] = status
                except Exception as e:
                    logger.warning(f"Failed to get status for {subsystem_id}: {e}")
                    subsystem_statuses[subsystem_id] = {"error": str(e)}
            
            # Calculate system-wide metrics
            total_subsystems = len(self.subsystems)
            active_subsystems = len([s for s in subsystem_statuses.values() if s.get("status") == "active"])
            healthy_subsystems = len([s for s in subsystem_statuses.values() if s.get("health") == "excellent"])
            
            # Calculate overall system health
            health_scores = []
            for status in subsystem_statuses.values():
                if "health" in status:
                    health_mapping = {
                        "excellent": 1.0,
                        "good": 0.8,
                        "warning": 0.6,
                        "critical": 0.3,
                        "failed": 0.0
                    }
                    health_scores.append(health_mapping.get(status["health"], 0.0))
            
            overall_health_score = np.mean(health_scores) if health_scores else 0.0
            
            # Determine overall system health
            if overall_health_score >= 0.9:
                overall_health = "excellent"
            elif overall_health_score >= 0.7:
                overall_health = "good"
            elif overall_health_score >= 0.5:
                overall_health = "warning"
            elif overall_health_score >= 0.3:
                overall_health = "critical"
            else:
                overall_health = "failed"
            
            return {
                "coordination_status": self.coordination_status,
                "system_health": overall_health,
                "overall_health_score": overall_health_score,
                "total_subsystems": total_subsystems,
                "active_subsystems": active_subsystems,
                "healthy_subsystems": healthy_subsystems,
                "subsystem_statuses": subsystem_statuses,
                "last_update": self.last_coordination_update.isoformat(),
                "coordination_metrics": self.coordination_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to get system overview: {e}")
            return {
                "coordination_status": "error",
                "system_health": "failed",
                "error": str(e)
            }
    
    async def get_subsystem_health_summary(self) -> Dict[str, Any]:
        """Get health summary for all subsystems."""
        try:
            health_summary = {}
            
            for subsystem_id, subsystem in self.subsystems.items():
                try:
                    status = await subsystem.get_status()
                    health_summary[subsystem_id] = {
                        "health": status.get("health", "unknown"),
                        "status": status.get("status", "unknown"),
                        "last_update": status.get("last_update", "unknown"),
                        "success_rate": status.get("success_rate", 0.0),
                        "is_monitoring": status.get("is_monitoring", False)
                    }
                except Exception as e:
                    health_summary[subsystem_id] = {
                        "health": "error",
                        "status": "error",
                        "error": str(e)
                    }
            
            return health_summary
            
        except Exception as e:
            logger.error(f"Failed to get subsystem health summary: {e}")
            return {"error": str(e)}
    
    async def get_performance_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance analytics for all subsystems."""
        try:
            analytics = {}
            
            for subsystem_id, subsystem in self.subsystems.items():
                try:
                    # Get metrics history
                    metrics_history = await subsystem.get_metrics_history(hours)
                    
                    if metrics_history:
                        # Calculate performance metrics
                        health_scores = [m.get("health_score", 0) for m in metrics_history]
                        performance_scores = [m.get("performance_score", 0) for m in metrics_history]
                        efficiency_scores = [m.get("efficiency_score", 0) for m in metrics_history]
                        
                        analytics[subsystem_id] = {
                            "avg_health_score": np.mean(health_scores) if health_scores else 0.0,
                            "avg_performance_score": np.mean(performance_scores) if performance_scores else 0.0,
                            "avg_efficiency_score": np.mean(efficiency_scores) if efficiency_scores else 0.0,
                            "total_metrics": len(metrics_history),
                            "health_trend": np.polyfit(range(len(health_scores)), health_scores, 1)[0] if len(health_scores) > 1 else 0.0,
                            "performance_trend": np.polyfit(range(len(performance_scores)), performance_scores, 1)[0] if len(performance_scores) > 1 else 0.0
                        }
                    else:
                        analytics[subsystem_id] = {
                            "avg_health_score": 0.0,
                            "avg_performance_score": 0.0,
                            "avg_efficiency_score": 0.0,
                            "total_metrics": 0,
                            "health_trend": 0.0,
                            "performance_trend": 0.0
                        }
                        
                except Exception as e:
                    logger.warning(f"Failed to get analytics for {subsystem_id}: {e}")
                    analytics[subsystem_id] = {"error": str(e)}
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get performance analytics: {e}")
            return {"error": str(e)}
    
    async def get_cross_subsystem_insights(self) -> Dict[str, Any]:
        """Get insights from cross-subsystem analysis."""
        try:
            insights = {
                "memory_insights": {},
                "learning_insights": {},
                "action_insights": {},
                "exploration_insights": {},
                "energy_insights": {},
                "visual_insights": {},
                "system_insights": {}
            }
            
            # Memory subsystem insights
            memory_subsystems = ["memory_access", "memory_consolidation", "memory_fragmentation", 
                               "salient_memory_retrieval", "memory_regularization", "temporal_memory"]
            memory_health_scores = []
            for subsystem_id in memory_subsystems:
                if subsystem_id in self.subsystems:
                    try:
                        status = await self.subsystems[subsystem_id].get_status()
                        memory_health_scores.append(status.get("success_rate", 0.0))
                    except:
                        pass
            
            if memory_health_scores:
                insights["memory_insights"] = {
                    "avg_health": np.mean(memory_health_scores),
                    "health_consistency": 1 - np.std(memory_health_scores) if len(memory_health_scores) > 1 else 1.0,
                    "total_subsystems": len(memory_subsystems)
                }
            
            # Learning subsystem insights
            learning_subsystems = ["learning_progress", "meta_learning", "knowledge_transfer",
                                 "pattern_recognition", "curriculum_learning", "cross_session_learning"]
            learning_health_scores = []
            for subsystem_id in learning_subsystems:
                if subsystem_id in self.subsystems:
                    try:
                        status = await self.subsystems[subsystem_id].get_status()
                        learning_health_scores.append(status.get("success_rate", 0.0))
                    except:
                        pass
            
            if learning_health_scores:
                insights["learning_insights"] = {
                    "avg_health": np.mean(learning_health_scores),
                    "health_consistency": 1 - np.std(learning_health_scores) if len(learning_health_scores) > 1 else 1.0,
                    "total_subsystems": len(learning_subsystems)
                }
            
            # Action subsystem insights
            action_subsystems = ["action_intelligence", "action_experimentation", "emergency_movement",
                               "predictive_coordinates", "coordinate_success"]
            action_health_scores = []
            for subsystem_id in action_subsystems:
                if subsystem_id in self.subsystems:
                    try:
                        status = await self.subsystems[subsystem_id].get_status()
                        action_health_scores.append(status.get("success_rate", 0.0))
                    except:
                        pass
            
            if action_health_scores:
                insights["action_insights"] = {
                    "avg_health": np.mean(action_health_scores),
                    "health_consistency": 1 - np.std(action_health_scores) if len(action_health_scores) > 1 else 1.0,
                    "total_subsystems": len(action_subsystems)
                }
            
            # Exploration subsystem insights
            exploration_subsystems = ["exploration_strategy", "boredom_detection", "stagnation_detection",
                                    "contrarian_strategy", "goal_invention"]
            exploration_health_scores = []
            for subsystem_id in exploration_subsystems:
                if subsystem_id in self.subsystems:
                    try:
                        status = await self.subsystems[subsystem_id].get_status()
                        exploration_health_scores.append(status.get("success_rate", 0.0))
                    except:
                        pass
            
            if exploration_health_scores:
                insights["exploration_insights"] = {
                    "avg_health": np.mean(exploration_health_scores),
                    "health_consistency": 1 - np.std(exploration_health_scores) if len(exploration_health_scores) > 1 else 1.0,
                    "total_subsystems": len(exploration_subsystems)
                }
            
            # Energy subsystem insights
            energy_subsystems = ["energy_system", "sleep_cycle", "mid_game_sleep", "death_manager"]
            energy_health_scores = []
            for subsystem_id in energy_subsystems:
                if subsystem_id in self.subsystems:
                    try:
                        status = await self.subsystems[subsystem_id].get_status()
                        energy_health_scores.append(status.get("success_rate", 0.0))
                    except:
                        pass
            
            if energy_health_scores:
                insights["energy_insights"] = {
                    "avg_health": np.mean(energy_health_scores),
                    "health_consistency": 1 - np.std(energy_health_scores) if len(energy_health_scores) > 1 else 1.0,
                    "total_subsystems": len(energy_subsystems)
                }
            
            # Visual subsystem insights
            visual_subsystems = ["frame_analysis", "boundary_detection", "multi_modal_input", "visual_pattern"]
            visual_health_scores = []
            for subsystem_id in visual_subsystems:
                if subsystem_id in self.subsystems:
                    try:
                        status = await self.subsystems[subsystem_id].get_status()
                        visual_health_scores.append(status.get("success_rate", 0.0))
                    except:
                        pass
            
            if visual_health_scores:
                insights["visual_insights"] = {
                    "avg_health": np.mean(visual_health_scores),
                    "health_consistency": 1 - np.std(visual_health_scores) if len(visual_health_scores) > 1 else 1.0,
                    "total_subsystems": len(visual_subsystems)
                }
            
            # System subsystem insights
            system_subsystems = ["resource_utilization", "gradient_flow", "usage_tracking", "anti_bias_weighting",
                               "cluster_formation", "danger_zone_avoidance", "swarm_intelligence", "hebbian_bonuses"]
            system_health_scores = []
            for subsystem_id in system_subsystems:
                if subsystem_id in self.subsystems:
                    try:
                        status = await self.subsystems[subsystem_id].get_status()
                        system_health_scores.append(status.get("success_rate", 0.0))
                    except:
                        pass
            
            if system_health_scores:
                insights["system_insights"] = {
                    "avg_health": np.mean(system_health_scores),
                    "health_consistency": 1 - np.std(system_health_scores) if len(system_health_scores) > 1 else 1.0,
                    "total_subsystems": len(system_subsystems)
                }
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get cross-subsystem insights: {e}")
            return {"error": str(e)}
    
    async def update_subsystem_configuration(self, subsystem_id: str, new_config: Dict[str, Any]) -> bool:
        """Update configuration for a specific subsystem."""
        try:
            if subsystem_id not in self.subsystems:
                logger.error(f"Subsystem {subsystem_id} not found")
                return False
            
            subsystem = self.subsystems[subsystem_id]
            await subsystem.update_configuration(new_config)
            logger.info(f"Updated configuration for subsystem {subsystem_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration for {subsystem_id}: {e}")
            return False
    
    async def stop_all_monitoring(self) -> None:
        """Stop monitoring for all subsystems."""
        try:
            stop_tasks = []
            for subsystem_id, subsystem in self.subsystems.items():
                task = asyncio.create_task(subsystem.stop_monitoring())
                stop_tasks.append((subsystem_id, task))
            
            for subsystem_id, task in stop_tasks:
                try:
                    await task
                    logger.info(f"Stopped monitoring for {subsystem_id}")
                except Exception as e:
                    logger.warning(f"Failed to stop monitoring for {subsystem_id}: {e}")
            
            self.coordination_status = "stopped"
            logger.info("Stopped monitoring for all subsystems")
            
        except Exception as e:
            logger.error(f"Failed to stop all monitoring: {e}")
    
    async def cleanup_all_subsystems(self) -> None:
        """Cleanup all subsystems."""
        try:
            cleanup_tasks = []
            for subsystem_id, subsystem in self.subsystems.items():
                task = asyncio.create_task(subsystem.cleanup())
                cleanup_tasks.append((subsystem_id, task))
            
            for subsystem_id, task in cleanup_tasks:
                try:
                    await task
                    logger.info(f"Cleaned up {subsystem_id}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {subsystem_id}: {e}")
            
            self.coordination_status = "disabled"
            logger.info("Cleaned up all subsystems")
            
        except Exception as e:
            logger.error(f"Failed to cleanup all subsystems: {e}")
    
    def get_subsystem_count(self) -> int:
        """Get total number of subsystems."""
        return len(self.subsystems)
    
    def get_subsystem_ids(self) -> List[str]:
        """Get list of all subsystem IDs."""
        return list(self.subsystems.keys())
    
    def get_subsystem_by_id(self, subsystem_id: str) -> Optional[Any]:
        """Get subsystem by ID."""
        return self.subsystems.get(subsystem_id)
    
    def __str__(self) -> str:
        return f"CognitiveCoordinator({len(self.subsystems)} subsystems, {self.coordination_status})"
    
    def __repr__(self) -> str:
        return f"<CognitiveCoordinator(subsystems={len(self.subsystems)}, status='{self.coordination_status}')>"
