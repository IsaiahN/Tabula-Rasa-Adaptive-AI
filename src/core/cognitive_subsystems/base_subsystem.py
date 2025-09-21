"""
Base Cognitive Subsystem

Provides the foundation for all 37 cognitive subsystems with database integration,
health monitoring, and API access capabilities.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np

# Import database integration
from src.database.system_integration import get_system_integration

logger = logging.getLogger(__name__)

class SubsystemStatus(Enum):
    """Status of a cognitive subsystem."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    DISABLED = "disabled"

class SubsystemHealth(Enum):
    """Health level of a cognitive subsystem."""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"

@dataclass
class SubsystemMetrics:
    """Metrics collected by a cognitive subsystem."""
    timestamp: datetime
    subsystem_id: str
    metrics: Dict[str, Any]
    health_score: float
    performance_score: float
    efficiency_score: float
    error_count: int = 0
    warning_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'subsystem_id': self.subsystem_id,
            'metrics': self.metrics,
            'health_score': self.health_score,
            'performance_score': self.performance_score,
            'efficiency_score': self.efficiency_score,
            'error_count': self.error_count,
            'warning_count': self.warning_count
        }

@dataclass
class SubsystemConfiguration:
    """Configuration for a cognitive subsystem."""
    subsystem_id: str
    enabled: bool = True
    monitoring_interval: float = 1.0  # seconds
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'subsystem_id': self.subsystem_id,
            'enabled': self.enabled,
            'monitoring_interval': self.monitoring_interval,
            'alert_thresholds': self.alert_thresholds,
            'parameters': self.parameters,
            'dependencies': self.dependencies
        }

class BaseCognitiveSubsystem(ABC):
    """
    Base class for all 37 cognitive subsystems.
    
    Provides common functionality for monitoring, health tracking, and database integration.
    Each subsystem should inherit from this class and implement the abstract methods.
    """
    
    def __init__(self, subsystem_id: str, name: str, description: str):
        self.subsystem_id = subsystem_id
        self.name = name
        self.description = description
        
        # Database integration
        self.integration = get_system_integration()
        
        # State management
        self.status = SubsystemStatus.INITIALIZING
        self.health = SubsystemHealth.EXCELLENT
        self.last_update = datetime.now()
        self.metrics_history: List[SubsystemMetrics] = []
        
        # Configuration
        self.config = SubsystemConfiguration(subsystem_id=subsystem_id)
        self.alert_thresholds = {
            'health_score': 0.7,
            'performance_score': 0.6,
            'efficiency_score': 0.5,
            'error_rate': 0.1
        }
        
        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # Performance tracking
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.average_response_time = 0.0
        
        logger.info(f"Initialized {self.name} subsystem (ID: {subsystem_id})")
    
    async def initialize(self) -> None:
        """Initialize the subsystem."""
        try:
            self.status = SubsystemStatus.INITIALIZING
            await self._load_configuration()
            await self._setup_monitoring()
            await self._initialize_subsystem()
            self.status = SubsystemStatus.ACTIVE
            self.health = SubsystemHealth.EXCELLENT
            logger.info(f"{self.name} subsystem initialized successfully")
        except Exception as e:
            self.status = SubsystemStatus.ERROR
            self.health = SubsystemHealth.FAILED
            logger.error(f"Failed to initialize {self.name} subsystem: {e}")
            raise
    
    async def _load_configuration(self) -> None:
        """Load configuration from database."""
        try:
            config_data = await self.integration.get_subsystem_config(self.subsystem_id)
            if config_data:
                self.config = SubsystemConfiguration(**config_data)
                self.alert_thresholds.update(self.config.alert_thresholds)
        except Exception as e:
            logger.warning(f"Could not load configuration for {self.subsystem_id}: {e}")
    
    async def _setup_monitoring(self) -> None:
        """Setup monitoring if enabled."""
        if self.config.enabled and self.config.monitoring_interval > 0:
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    @abstractmethod
    async def _initialize_subsystem(self) -> None:
        """Initialize the specific subsystem implementation."""
        pass
    
    @abstractmethod
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics specific to this subsystem."""
        pass
    
    @abstractmethod
    async def analyze_health(self, metrics: Dict[str, Any]) -> SubsystemHealth:
        """Analyze health based on collected metrics."""
        pass
    
    @abstractmethod
    async def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance score (0.0 to 1.0)."""
        pass
    
    @abstractmethod
    async def get_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score (0.0 to 1.0)."""
        pass
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring and self.status != SubsystemStatus.DISABLED:
            try:
                await self._collect_and_store_metrics()
                await asyncio.sleep(self.config.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop for {self.subsystem_id}: {e}")
                await asyncio.sleep(5.0)  # Wait before retrying
    
    async def _collect_and_store_metrics(self) -> None:
        """Collect metrics and store in database."""
        try:
            # Collect subsystem-specific metrics
            metrics = await self.collect_metrics()
            
            # Calculate scores
            health = await self.analyze_health(metrics)
            performance_score = await self.get_performance_score(metrics)
            efficiency_score = await self.get_efficiency_score(metrics)
            
            # Count errors and warnings
            error_count = metrics.get('error_count', 0)
            warning_count = metrics.get('warning_count', 0)
            
            # Create metrics object
            subsystem_metrics = SubsystemMetrics(
                timestamp=datetime.now(),
                subsystem_id=self.subsystem_id,
                metrics=metrics,
                health_score=self._health_to_score(health),
                performance_score=performance_score,
                efficiency_score=efficiency_score,
                error_count=error_count,
                warning_count=warning_count
            )
            
            # Store in database
            await self.integration.store_subsystem_metrics(subsystem_metrics.to_dict())
            
            # Update local state
            self.health = health
            self.metrics_history.append(subsystem_metrics)
            self.last_update = datetime.now()
            
            # Keep only recent metrics in memory
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-500:]
            
            # Check for alerts
            await self._check_alerts(subsystem_metrics)
            
        except Exception as e:
            logger.error(f"Error collecting metrics for {self.subsystem_id}: {e}")
            self.failed_operations += 1
    
    def _health_to_score(self, health: SubsystemHealth) -> float:
        """Convert health enum to numeric score."""
        health_scores = {
            SubsystemHealth.EXCELLENT: 1.0,
            SubsystemHealth.GOOD: 0.8,
            SubsystemHealth.WARNING: 0.6,
            SubsystemHealth.CRITICAL: 0.3,
            SubsystemHealth.FAILED: 0.0
        }
        return health_scores.get(health, 0.0)
    
    async def _check_alerts(self, metrics: SubsystemMetrics) -> None:
        """Check for alert conditions."""
        alerts = []
        
        # Check health score
        if metrics.health_score < self.alert_thresholds['health_score']:
            alerts.append({
                'type': 'health_warning',
                'message': f"Health score {metrics.health_score:.2f} below threshold",
                'severity': 'warning' if metrics.health_score > 0.3 else 'critical'
            })
        
        # Check performance score
        if metrics.performance_score < self.alert_thresholds['performance_score']:
            alerts.append({
                'type': 'performance_warning',
                'message': f"Performance score {metrics.performance_score:.2f} below threshold",
                'severity': 'warning'
            })
        
        # Check efficiency score
        if metrics.efficiency_score < self.alert_thresholds['efficiency_score']:
            alerts.append({
                'type': 'efficiency_warning',
                'message': f"Efficiency score {metrics.efficiency_score:.2f} below threshold",
                'severity': 'warning'
            })
        
        # Check error rate
        if self.total_operations > 0:
            error_rate = self.failed_operations / self.total_operations
            if error_rate > self.alert_thresholds['error_rate']:
                alerts.append({
                    'type': 'error_rate_warning',
                    'message': f"Error rate {error_rate:.2f} above threshold",
                    'severity': 'critical'
                })
        
        # Store alerts in database
        if alerts:
            await self.integration.store_subsystem_alerts(self.subsystem_id, alerts)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current subsystem status."""
        return {
            'subsystem_id': self.subsystem_id,
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'health': self.health.value,
            'last_update': self.last_update.isoformat(),
            'total_operations': self.total_operations,
            'successful_operations': self.successful_operations,
            'failed_operations': self.failed_operations,
            'success_rate': self.successful_operations / max(self.total_operations, 1),
            'average_response_time': self.average_response_time,
            'is_monitoring': self.is_monitoring,
            'config': self.config.to_dict()
        }
    
    async def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics history for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        return [m.to_dict() for m in recent_metrics]
    
    async def update_configuration(self, new_config: Dict[str, Any]) -> None:
        """Update subsystem configuration."""
        try:
            # Update local config
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            # Store in database
            await self.integration.store_subsystem_config(self.subsystem_id, self.config.to_dict())
            
            # Restart monitoring if interval changed
            if 'monitoring_interval' in new_config:
                await self.stop_monitoring()
                await self._setup_monitoring()
            
            logger.info(f"Updated configuration for {self.subsystem_id}")
        except Exception as e:
            logger.error(f"Failed to update configuration for {self.subsystem_id}: {e}")
            raise
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring the subsystem."""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped monitoring for {self.subsystem_id}")
    
    async def cleanup(self) -> None:
        """Cleanup subsystem resources."""
        await self.stop_monitoring()
        self.status = SubsystemStatus.DISABLED
        logger.info(f"Cleaned up {self.subsystem_id}")
    
    def __str__(self) -> str:
        return f"{self.name} ({self.subsystem_id}) - {self.status.value}"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id='{self.subsystem_id}', status='{self.status.value}')>"
