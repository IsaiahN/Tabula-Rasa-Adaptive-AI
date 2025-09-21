"""
Intelligent Training Scheduler

Advanced scheduling system that optimizes training sessions based on
system resources, performance patterns, and learning objectives.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from ..interfaces import ComponentInterface
from ..caching import CacheManager, CacheConfig
from ..monitoring import SystemMonitor, TrainingMonitor


class SchedulePriority(Enum):
    """Schedule priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ScheduleType(Enum):
    """Types of training schedules."""
    CONTINUOUS = "continuous"
    BATCH = "batch"
    ADAPTIVE = "adaptive"
    MAINTENANCE = "maintenance"


@dataclass
class ScheduleConfig:
    """Configuration for training scheduler."""
    max_concurrent_sessions: int = 3
    resource_threshold: float = 0.8
    performance_window: int = 24  # hours
    adaptive_scheduling: bool = True
    maintenance_interval: int = 168  # hours (1 week)
    priority_weights: Dict[SchedulePriority, float] = None
    
    def __post_init__(self):
        if self.priority_weights is None:
            self.priority_weights = {
                SchedulePriority.CRITICAL: 1.0,
                SchedulePriority.HIGH: 0.8,
                SchedulePriority.MEDIUM: 0.6,
                SchedulePriority.LOW: 0.4
            }


@dataclass
class TrainingTask:
    """Represents a training task to be scheduled."""
    task_id: str
    task_type: ScheduleType
    priority: SchedulePriority
    estimated_duration: timedelta
    resource_requirements: Dict[str, float]
    dependencies: List[str]
    created_at: datetime
    scheduled_for: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed, cancelled
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TrainingScheduler(ComponentInterface):
    """
    Intelligent training scheduler that optimizes resource allocation
    and training session timing based on system conditions.
    """
    
    def __init__(self, config: ScheduleConfig, cache_config: Optional[CacheConfig] = None):
        """Initialize the training scheduler."""
        self.config = config
        self.cache_config = cache_config or CacheConfig()
        self.cache = CacheManager(self.cache_config)
        self.system_monitor = SystemMonitor()
        self.training_monitor = TrainingMonitor("scheduler")
        
        # Task management
        self.tasks: Dict[str, TrainingTask] = {}
        self.running_tasks: Dict[str, TrainingTask] = {}
        self.completed_tasks: List[TrainingTask] = []
        
        # Scheduling state
        self.schedule_queue: List[TrainingTask] = []
        self.resource_usage: Dict[str, float] = {
            'cpu': 0.0,
            'memory': 0.0,
            'disk': 0.0
        }
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.optimal_times: List[datetime] = []
        
        self._initialized = False
        self._scheduler_running = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize the training scheduler."""
        try:
            self.cache.initialize()
            self.system_monitor.start_monitoring()
            self.training_monitor.start_monitoring()
            self._initialized = True
            self.logger.info("Training scheduler initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize training scheduler: {e}")
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get current component state."""
        return {
            'name': 'TrainingScheduler',
            'status': 'running' if self._initialized else 'stopped',
            'last_updated': datetime.now(),
            'metadata': {
                'total_tasks': len(self.tasks),
                'running_tasks': len(self.running_tasks),
                'queued_tasks': len(self.schedule_queue),
                'completed_tasks': len(self.completed_tasks),
                'resource_usage': self.resource_usage
            }
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._scheduler_running = False
            self.system_monitor.stop_monitoring()
            self.training_monitor.stop_monitoring()
            self.cache.clear()
            self._initialized = False
            self.logger.info("Training scheduler cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self._initialized and self.cache.is_healthy()
    
    def add_task(self, task: TrainingTask) -> bool:
        """Add a new training task to the scheduler."""
        try:
            # Validate task
            if not self._validate_task(task):
                return False
            
            # Add to task registry
            self.tasks[task.task_id] = task
            
            # Add to schedule queue
            self.schedule_queue.append(task)
            
            # Sort queue by priority and creation time
            self.schedule_queue.sort(
                key=lambda t: (
                    -self.config.priority_weights[t.priority],
                    t.created_at
                )
            )
            
            # Cache task
            self.cache.set(f"task_{task.task_id}", task, ttl=86400)
            
            self.logger.info(f"Added task {task.task_id} with priority {task.priority.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding task {task.task_id}: {e}")
            return False
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the scheduler."""
        try:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                
                # Remove from queue if not running
                if task in self.schedule_queue:
                    self.schedule_queue.remove(task)
                
                # Cancel if running
                if task_id in self.running_tasks:
                    self._cancel_running_task(task_id)
                
                # Remove from registry
                del self.tasks[task_id]
                
                # Remove from cache
                self.cache.delete(f"task_{task_id}")
                
                self.logger.info(f"Removed task {task_id}")
                return True
            else:
                self.logger.warning(f"Task {task_id} not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Error removing task {task_id}: {e}")
            return False
    
    def get_schedule(self, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """Get the current schedule for the next N hours."""
        try:
            end_time = datetime.now() + timedelta(hours=hours_ahead)
            schedule = []
            
            # Add running tasks
            for task in self.running_tasks.values():
                if task.started_at and task.started_at < end_time:
                    schedule.append({
                        'task_id': task.task_id,
                        'type': task.task_type.value,
                        'priority': task.priority.value,
                        'start_time': task.started_at,
                        'estimated_end': task.started_at + task.estimated_duration,
                        'status': 'running'
                    })
            
            # Add scheduled tasks
            current_time = datetime.now()
            for task in self.schedule_queue:
                if task.scheduled_for and task.scheduled_for < end_time:
                    schedule.append({
                        'task_id': task.task_id,
                        'type': task.task_type.value,
                        'priority': task.priority.value,
                        'start_time': task.scheduled_for,
                        'estimated_end': task.scheduled_for + task.estimated_duration,
                        'status': 'scheduled'
                    })
            
            # Sort by start time
            schedule.sort(key=lambda x: x['start_time'])
            return schedule
            
        except Exception as e:
            self.logger.error(f"Error getting schedule: {e}")
            return []
    
    def optimize_schedule(self) -> Dict[str, Any]:
        """Optimize the current schedule based on system conditions."""
        try:
            # Get current system state
            system_health = self.system_monitor.get_current_health()
            training_progress = self.training_monitor.get_current_progress()
            
            # Calculate optimal scheduling
            optimization_results = {
                'tasks_rescheduled': 0,
                'resource_optimization': 0.0,
                'time_optimization': 0.0,
                'recommendations': []
            }
            
            # Reschedule tasks based on system conditions
            if system_health.status == 'critical':
                # Delay low priority tasks
                for task in self.schedule_queue:
                    if task.priority in [SchedulePriority.LOW, SchedulePriority.MEDIUM]:
                        task.scheduled_for = datetime.now() + timedelta(hours=2)
                        optimization_results['tasks_rescheduled'] += 1
                        optimization_results['recommendations'].append(
                            f"Delayed {task.task_id} due to critical system state"
                        )
            
            elif system_health.status == 'healthy' and training_progress.win_rate > 0.8:
                # Accelerate high priority tasks
                for task in self.schedule_queue:
                    if task.priority == SchedulePriority.HIGH and not task.scheduled_for:
                        task.scheduled_for = datetime.now() + timedelta(minutes=30)
                        optimization_results['tasks_rescheduled'] += 1
                        optimization_results['recommendations'].append(
                            f"Accelerated {task.task_id} due to high performance"
                        )
            
            # Calculate resource optimization
            current_usage = (system_health.cpu_usage + system_health.memory_usage) / 200.0
            optimization_results['resource_optimization'] = 1.0 - current_usage
            
            # Cache optimization results
            self.cache.set(
                f"optimization_{datetime.now().timestamp()}",
                optimization_results,
                ttl=3600
            )
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error optimizing schedule: {e}")
            return {'error': str(e)}
    
    def start_scheduler(self) -> None:
        """Start the background scheduler."""
        if not self._scheduler_running:
            self._scheduler_running = True
            asyncio.create_task(self._scheduler_loop())
            self.logger.info("Background scheduler started")
    
    def stop_scheduler(self) -> None:
        """Stop the background scheduler."""
        self._scheduler_running = False
        self.logger.info("Background scheduler stopped")
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._scheduler_running:
            try:
                # Check for tasks to start
                await self._check_and_start_tasks()
                
                # Check for completed tasks
                await self._check_completed_tasks()
                
                # Update resource usage
                await self._update_resource_usage()
                
                # Sleep for a short interval
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(30)
    
    async def _check_and_start_tasks(self) -> None:
        """Check for tasks that can be started."""
        if len(self.running_tasks) >= self.config.max_concurrent_sessions:
            return
        
        # Check system resources
        system_health = self.system_monitor.get_current_health()
        if system_health.status == 'critical':
            return
        
        # Find next task to start
        for task in self.schedule_queue[:]:
            if self._can_start_task(task):
                await self._start_task(task)
                self.schedule_queue.remove(task)
                break
    
    async def _start_task(self, task: TrainingTask) -> None:
        """Start a training task."""
        try:
            task.status = "running"
            task.started_at = datetime.now()
            self.running_tasks[task.task_id] = task
            
            # Update resource usage
            self.resource_usage['cpu'] += task.resource_requirements.get('cpu', 0.1)
            self.resource_usage['memory'] += task.resource_requirements.get('memory', 0.1)
            self.resource_usage['disk'] += task.resource_requirements.get('disk', 0.1)
            
            self.logger.info(f"Started task {task.task_id}")
            
        except Exception as e:
            self.logger.error(f"Error starting task {task.task_id}: {e}")
            task.status = "failed"
    
    async def _check_completed_tasks(self) -> None:
        """Check for completed tasks."""
        completed_tasks = []
        
        for task_id, task in self.running_tasks.items():
            if task.started_at:
                elapsed = datetime.now() - task.started_at
                if elapsed >= task.estimated_duration:
                    await self._complete_task(task)
                    completed_tasks.append(task_id)
        
        # Remove completed tasks from running tasks
        for task_id in completed_tasks:
            del self.running_tasks[task_id]
    
    async def _complete_task(self, task: TrainingTask) -> None:
        """Mark a task as completed."""
        try:
            task.status = "completed"
            task.completed_at = datetime.now()
            self.completed_tasks.append(task)
            
            # Update resource usage
            self.resource_usage['cpu'] -= task.resource_requirements.get('cpu', 0.1)
            self.resource_usage['memory'] -= task.resource_requirements.get('memory', 0.1)
            self.resource_usage['disk'] -= task.resource_requirements.get('disk', 0.1)
            
            # Record performance
            duration = (task.completed_at - task.started_at).total_seconds()
            self.performance_history.append({
                'task_id': task.task_id,
                'task_type': task.task_type.value,
                'priority': task.priority.value,
                'duration': duration,
                'estimated_duration': task.estimated_duration.total_seconds(),
                'completed_at': task.completed_at
            })
            
            self.logger.info(f"Completed task {task.task_id}")
            
        except Exception as e:
            self.logger.error(f"Error completing task {task.task_id}: {e}")
    
    def _validate_task(self, task: TrainingTask) -> bool:
        """Validate a training task."""
        if not task.task_id or task.task_id in self.tasks:
            return False
        
        if task.estimated_duration.total_seconds() <= 0:
            return False
        
        if not task.resource_requirements:
            return False
        
        return True
    
    def _can_start_task(self, task: TrainingTask) -> bool:
        """Check if a task can be started."""
        # Check dependencies
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        
        # Check resource availability
        system_health = self.system_monitor.get_current_health()
        required_cpu = task.resource_requirements.get('cpu', 0.1)
        required_memory = task.resource_requirements.get('memory', 0.1)
        
        if (system_health.cpu_usage + required_cpu * 100 > self.config.resource_threshold * 100):
            return False
        
        if (system_health.memory_usage + required_memory * 100 > self.config.resource_threshold * 100):
            return False
        
        return True
    
    async def _update_resource_usage(self) -> None:
        """Update current resource usage."""
        system_health = self.system_monitor.get_current_health()
        self.resource_usage = {
            'cpu': system_health.cpu_usage / 100.0,
            'memory': system_health.memory_usage / 100.0,
            'disk': system_health.disk_usage / 100.0
        }
    
    def _cancel_running_task(self, task_id: str) -> None:
        """Cancel a running task."""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.status = "cancelled"
            task.completed_at = datetime.now()
            
            # Update resource usage
            self.resource_usage['cpu'] -= task.resource_requirements.get('cpu', 0.1)
            self.resource_usage['memory'] -= task.resource_requirements.get('memory', 0.1)
            self.resource_usage['disk'] -= task.resource_requirements.get('disk', 0.1)
            
            self.logger.info(f"Cancelled task {task_id}")
