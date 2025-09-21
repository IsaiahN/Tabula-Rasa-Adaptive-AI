"""
Cognitive Subsystem API

Provides a unified API interface for accessing and managing all 37 cognitive subsystems.
This API is designed to be used by the Director, Governor, and Architect for system monitoring and control.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import json

from .cognitive_coordinator import CognitiveCoordinator
from .base_subsystem import SubsystemStatus, SubsystemHealth

logger = logging.getLogger(__name__)

class CognitiveSubsystemAPI:
    """
    Unified API interface for all 37 cognitive subsystems.
    
    This API provides:
    - High-level system monitoring and control
    - Subsystem-specific operations
    - Cross-subsystem analytics and insights
    - Database integration for persistent storage
    - Real-time status and health monitoring
    """
    
    def __init__(self):
        self.coordinator = CognitiveCoordinator()
        self.api_status = "initializing"
        self.last_api_update = datetime.now()
        
        logger.info("Cognitive Subsystem API initialized")
    
    async def initialize(self) -> None:
        """Initialize the API and all subsystems."""
        try:
            self.api_status = "initializing"
            await self.coordinator.initialize_all_subsystems()
            self.api_status = "active"
            self.last_api_update = datetime.now()
            logger.info("Cognitive Subsystem API initialized successfully")
        except Exception as e:
            self.api_status = "error"
            logger.error(f"Failed to initialize Cognitive Subsystem API: {e}")
            raise
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and health."""
        try:
            overview = await self.coordinator.get_system_overview()
            return {
                "api_status": self.api_status,
                "last_update": self.last_api_update.isoformat(),
                "system_overview": overview
            }
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                "api_status": "error",
                "error": str(e)
            }
    
    async def get_subsystem_status(self, subsystem_id: str) -> Dict[str, Any]:
        """Get status for a specific subsystem."""
        try:
            if subsystem_id not in self.coordinator.subsystems:
                return {"error": f"Subsystem {subsystem_id} not found"}
            
            subsystem = self.coordinator.subsystems[subsystem_id]
            status = await subsystem.get_status()
            return {
                "subsystem_id": subsystem_id,
                "status": status,
                "api_status": self.api_status
            }
        except Exception as e:
            logger.error(f"Failed to get status for {subsystem_id}: {e}")
            return {"error": str(e)}
    
    async def get_all_subsystem_statuses(self) -> Dict[str, Any]:
        """Get status for all subsystems."""
        try:
            health_summary = await self.coordinator.get_subsystem_health_summary()
            return {
                "total_subsystems": len(self.coordinator.subsystems),
                "subsystem_statuses": health_summary,
                "api_status": self.api_status,
                "last_update": self.last_api_update.isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get all subsystem statuses: {e}")
            return {"error": str(e)}
    
    async def get_performance_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance analytics for all subsystems."""
        try:
            analytics = await self.coordinator.get_performance_analytics(hours)
            return {
                "analytics": analytics,
                "time_period_hours": hours,
                "api_status": self.api_status,
                "last_update": self.last_api_update.isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get performance analytics: {e}")
            return {"error": str(e)}
    
    async def get_cross_subsystem_insights(self) -> Dict[str, Any]:
        """Get insights from cross-subsystem analysis."""
        try:
            insights = await self.coordinator.get_cross_subsystem_insights()
            return {
                "insights": insights,
                "api_status": self.api_status,
                "last_update": self.last_api_update.isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get cross-subsystem insights: {e}")
            return {"error": str(e)}
    
    async def get_memory_subsystem_insights(self) -> Dict[str, Any]:
        """Get insights specifically from memory subsystems."""
        try:
            memory_subsystems = ["memory_access", "memory_consolidation", "memory_fragmentation",
                               "salient_memory_retrieval", "memory_regularization", "temporal_memory"]
            
            memory_insights = {}
            for subsystem_id in memory_subsystems:
                if subsystem_id in self.coordinator.subsystems:
                    try:
                        status = await self.coordinator.subsystems[subsystem_id].get_status()
                        memory_insights[subsystem_id] = {
                            "health": status.get("health", "unknown"),
                            "status": status.get("status", "unknown"),
                            "success_rate": status.get("success_rate", 0.0),
                            "is_monitoring": status.get("is_monitoring", False)
                        }
                    except Exception as e:
                        memory_insights[subsystem_id] = {"error": str(e)}
            
            return {
                "memory_insights": memory_insights,
                "total_memory_subsystems": len(memory_subsystems),
                "api_status": self.api_status
            }
        except Exception as e:
            logger.error(f"Failed to get memory subsystem insights: {e}")
            return {"error": str(e)}
    
    async def get_learning_subsystem_insights(self) -> Dict[str, Any]:
        """Get insights specifically from learning subsystems."""
        try:
            learning_subsystems = ["learning_progress", "meta_learning", "knowledge_transfer",
                                 "pattern_recognition", "curriculum_learning", "cross_session_learning"]
            
            learning_insights = {}
            for subsystem_id in learning_subsystems:
                if subsystem_id in self.coordinator.subsystems:
                    try:
                        status = await self.coordinator.subsystems[subsystem_id].get_status()
                        learning_insights[subsystem_id] = {
                            "health": status.get("health", "unknown"),
                            "status": status.get("status", "unknown"),
                            "success_rate": status.get("success_rate", 0.0),
                            "is_monitoring": status.get("is_monitoring", False)
                        }
                    except Exception as e:
                        learning_insights[subsystem_id] = {"error": str(e)}
            
            return {
                "learning_insights": learning_insights,
                "total_learning_subsystems": len(learning_subsystems),
                "api_status": self.api_status
            }
        except Exception as e:
            logger.error(f"Failed to get learning subsystem insights: {e}")
            return {"error": str(e)}
    
    async def get_action_subsystem_insights(self) -> Dict[str, Any]:
        """Get insights specifically from action subsystems."""
        try:
            action_subsystems = ["action_intelligence", "action_experimentation", "emergency_movement",
                               "predictive_coordinates", "coordinate_success"]
            
            action_insights = {}
            for subsystem_id in action_subsystems:
                if subsystem_id in self.coordinator.subsystems:
                    try:
                        status = await self.coordinator.subsystems[subsystem_id].get_status()
                        action_insights[subsystem_id] = {
                            "health": status.get("health", "unknown"),
                            "status": status.get("status", "unknown"),
                            "success_rate": status.get("success_rate", 0.0),
                            "is_monitoring": status.get("is_monitoring", False)
                        }
                    except Exception as e:
                        action_insights[subsystem_id] = {"error": str(e)}
            
            return {
                "action_insights": action_insights,
                "total_action_subsystems": len(action_subsystems),
                "api_status": self.api_status
            }
        except Exception as e:
            logger.error(f"Failed to get action subsystem insights: {e}")
            return {"error": str(e)}
    
    async def get_energy_subsystem_insights(self) -> Dict[str, Any]:
        """Get insights specifically from energy subsystems."""
        try:
            energy_subsystems = ["energy_system", "sleep_cycle", "mid_game_sleep", "death_manager"]
            
            energy_insights = {}
            for subsystem_id in energy_subsystems:
                if subsystem_id in self.coordinator.subsystems:
                    try:
                        status = await self.coordinator.subsystems[subsystem_id].get_status()
                        energy_insights[subsystem_id] = {
                            "health": status.get("health", "unknown"),
                            "status": status.get("status", "unknown"),
                            "success_rate": status.get("success_rate", 0.0),
                            "is_monitoring": status.get("is_monitoring", False)
                        }
                    except Exception as e:
                        energy_insights[subsystem_id] = {"error": str(e)}
            
            return {
                "energy_insights": energy_insights,
                "total_energy_subsystems": len(energy_subsystems),
                "api_status": self.api_status
            }
        except Exception as e:
            logger.error(f"Failed to get energy subsystem insights: {e}")
            return {"error": str(e)}
    
    async def get_visual_subsystem_insights(self) -> Dict[str, Any]:
        """Get insights specifically from visual subsystems."""
        try:
            visual_subsystems = ["frame_analysis", "boundary_detection", "multi_modal_input", "visual_pattern"]
            
            visual_insights = {}
            for subsystem_id in visual_subsystems:
                if subsystem_id in self.coordinator.subsystems:
                    try:
                        status = await self.coordinator.subsystems[subsystem_id].get_status()
                        visual_insights[subsystem_id] = {
                            "health": status.get("health", "unknown"),
                            "status": status.get("status", "unknown"),
                            "success_rate": status.get("success_rate", 0.0),
                            "is_monitoring": status.get("is_monitoring", False)
                        }
                    except Exception as e:
                        visual_insights[subsystem_id] = {"error": str(e)}
            
            return {
                "visual_insights": visual_insights,
                "total_visual_subsystems": len(visual_subsystems),
                "api_status": self.api_status
            }
        except Exception as e:
            logger.error(f"Failed to get visual subsystem insights: {e}")
            return {"error": str(e)}
    
    async def get_system_subsystem_insights(self) -> Dict[str, Any]:
        """Get insights specifically from system subsystems."""
        try:
            system_subsystems = ["resource_utilization", "gradient_flow", "usage_tracking", "anti_bias_weighting",
                               "cluster_formation", "danger_zone_avoidance", "swarm_intelligence", "hebbian_bonuses"]
            
            system_insights = {}
            for subsystem_id in system_subsystems:
                if subsystem_id in self.coordinator.subsystems:
                    try:
                        status = await self.coordinator.subsystems[subsystem_id].get_status()
                        system_insights[subsystem_id] = {
                            "health": status.get("health", "unknown"),
                            "status": status.get("status", "unknown"),
                            "success_rate": status.get("success_rate", 0.0),
                            "is_monitoring": status.get("is_monitoring", False)
                        }
                    except Exception as e:
                        system_insights[subsystem_id] = {"error": str(e)}
            
            return {
                "system_insights": system_insights,
                "total_system_subsystems": len(system_subsystems),
                "api_status": self.api_status
            }
        except Exception as e:
            logger.error(f"Failed to get system subsystem insights: {e}")
            return {"error": str(e)}
    
    async def update_subsystem_configuration(self, subsystem_id: str, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration for a specific subsystem."""
        try:
            success = await self.coordinator.update_subsystem_configuration(subsystem_id, new_config)
            return {
                "subsystem_id": subsystem_id,
                "success": success,
                "api_status": self.api_status
            }
        except Exception as e:
            logger.error(f"Failed to update configuration for {subsystem_id}: {e}")
            return {"error": str(e)}
    
    async def get_subsystem_metrics_history(self, subsystem_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get metrics history for a specific subsystem."""
        try:
            if subsystem_id not in self.coordinator.subsystems:
                return {"error": f"Subsystem {subsystem_id} not found"}
            
            subsystem = self.coordinator.subsystems[subsystem_id]
            metrics_history = await subsystem.get_metrics_history(hours)
            
            return {
                "subsystem_id": subsystem_id,
                "metrics_history": metrics_history,
                "time_period_hours": hours,
                "api_status": self.api_status
            }
        except Exception as e:
            logger.error(f"Failed to get metrics history for {subsystem_id}: {e}")
            return {"error": str(e)}
    
    async def get_health_alerts(self) -> Dict[str, Any]:
        """Get health alerts for all subsystems."""
        try:
            alerts = []
            for subsystem_id, subsystem in self.coordinator.subsystems.items():
                try:
                    status = await subsystem.get_status()
                    health = status.get("health", "unknown")
                    
                    if health in ["critical", "failed"]:
                        alerts.append({
                            "subsystem_id": subsystem_id,
                            "health": health,
                            "status": status.get("status", "unknown"),
                            "severity": "critical" if health == "failed" else "warning",
                            "timestamp": status.get("last_update", datetime.now().isoformat())
                        })
                    elif health == "warning":
                        alerts.append({
                            "subsystem_id": subsystem_id,
                            "health": health,
                            "status": status.get("status", "unknown"),
                            "severity": "warning",
                            "timestamp": status.get("last_update", datetime.now().isoformat())
                        })
                except Exception as e:
                    alerts.append({
                        "subsystem_id": subsystem_id,
                        "health": "error",
                        "status": "error",
                        "severity": "critical",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
            
            return {
                "alerts": alerts,
                "total_alerts": len(alerts),
                "critical_alerts": len([a for a in alerts if a.get("severity") == "critical"]),
                "warning_alerts": len([a for a in alerts if a.get("severity") == "warning"]),
                "api_status": self.api_status
            }
        except Exception as e:
            logger.error(f"Failed to get health alerts: {e}")
            return {"error": str(e)}
    
    async def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary."""
        try:
            # Get all the key insights
            system_status = await self.get_system_status()
            performance_analytics = await self.get_performance_analytics()
            cross_subsystem_insights = await self.get_cross_subsystem_insights()
            health_alerts = await self.get_health_alerts()
            
            return {
                "system_status": system_status,
                "performance_analytics": performance_analytics,
                "cross_subsystem_insights": cross_subsystem_insights,
                "health_alerts": health_alerts,
                "api_status": self.api_status,
                "last_update": self.last_api_update.isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get system summary: {e}")
            return {"error": str(e)}
    
    async def stop_all_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring for all subsystems."""
        try:
            await self.coordinator.stop_all_monitoring()
            self.api_status = "stopped"
            self.last_api_update = datetime.now()
            
            return {
                "success": True,
                "api_status": self.api_status,
                "message": "Stopped monitoring for all subsystems"
            }
        except Exception as e:
            logger.error(f"Failed to stop all monitoring: {e}")
            return {"error": str(e)}
    
    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup all subsystems and API resources."""
        try:
            await self.coordinator.cleanup_all_subsystems()
            self.api_status = "disabled"
            self.last_api_update = datetime.now()
            
            return {
                "success": True,
                "api_status": self.api_status,
                "message": "Cleaned up all subsystems and API resources"
            }
        except Exception as e:
            logger.error(f"Failed to cleanup: {e}")
            return {"error": str(e)}
    
    def get_api_info(self) -> Dict[str, Any]:
        """Get API information and capabilities."""
        return {
            "api_name": "Cognitive Subsystem API",
            "version": "1.0.0",
            "total_subsystems": self.coordinator.get_subsystem_count(),
            "subsystem_ids": self.coordinator.get_subsystem_ids(),
            "api_status": self.api_status,
            "capabilities": [
                "System status monitoring",
                "Subsystem health tracking",
                "Performance analytics",
                "Cross-subsystem insights",
                "Configuration management",
                "Health alerting",
                "Metrics history retrieval"
            ]
        }
    
    def __str__(self) -> str:
        return f"CognitiveSubsystemAPI({self.coordinator.get_subsystem_count()} subsystems, {self.api_status})"
    
    def __repr__(self) -> str:
        return f"<CognitiveSubsystemAPI(subsystems={self.coordinator.get_subsystem_count()}, status='{self.api_status}')>"
