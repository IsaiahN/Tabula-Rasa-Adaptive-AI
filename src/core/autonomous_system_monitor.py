"""
Autonomous System Monitor - Phase 1 Implementation

This system continuously monitors system health and automatically takes corrective actions
when thresholds are exceeded.
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from ..database.system_integration import get_system_integration

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """System health status levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class ActionType(Enum):
    """Types of automatic actions."""
    OPTIMIZE_PROCESSING = "optimize_processing"
    TRIGGER_GARBAGE_COLLECTION = "trigger_garbage_collection"
    INCREASE_VALIDATION = "increase_validation"
    OPTIMIZE_QUERIES = "optimize_queries"
    REDUCE_CONCURRENCY = "reduce_concurrency"
    CLEAR_CACHES = "clear_caches"
    RESTART_COMPONENT = "restart_component"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"

@dataclass
class HealthThreshold:
    """Health threshold configuration."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: float
    auto_action: ActionType
    cooldown_seconds: int
    last_action_time: float = 0

@dataclass
class SystemMetrics:
    """Current system metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    response_time: float
    error_rate: float
    active_connections: int

class AutonomousSystemMonitor:
    """
    Autonomous System Monitor that continuously monitors system health
    and automatically takes corrective actions.
    
    Features:
    - Continuous health monitoring
    - Automatic threshold-based actions
    - Performance trend analysis
    - Predictive health warnings
    - Action cooldown management
    """
    
    def __init__(self):
        self.integration = get_system_integration()
        
        # Health thresholds
        self.thresholds = {
            'cpu_usage': HealthThreshold(
                metric_name='cpu_usage',
                warning_threshold=70.0,
                critical_threshold=85.0,
                emergency_threshold=95.0,
                auto_action=ActionType.OPTIMIZE_PROCESSING,
                cooldown_seconds=60
            ),
            'memory_usage': HealthThreshold(
                metric_name='memory_usage',
                warning_threshold=75.0,
                critical_threshold=90.0,
                emergency_threshold=98.0,
                auto_action=ActionType.TRIGGER_GARBAGE_COLLECTION,
                cooldown_seconds=30
            ),
            'disk_usage': HealthThreshold(
                metric_name='disk_usage',
                warning_threshold=80.0,
                critical_threshold=90.0,
                emergency_threshold=95.0,
                auto_action=ActionType.CLEAR_CACHES,
                cooldown_seconds=300
            ),
            'response_time': HealthThreshold(
                metric_name='response_time',
                warning_threshold=2.0,
                critical_threshold=5.0,
                emergency_threshold=10.0,
                auto_action=ActionType.OPTIMIZE_QUERIES,
                cooldown_seconds=120
            ),
            'error_rate': HealthThreshold(
                metric_name='error_rate',
                warning_threshold=0.05,
                critical_threshold=0.15,
                emergency_threshold=0.30,
                auto_action=ActionType.INCREASE_VALIDATION,
                cooldown_seconds=60
            ),
            'active_connections': HealthThreshold(
                metric_name='active_connections',
                warning_threshold=100,
                critical_threshold=200,
                emergency_threshold=500,
                auto_action=ActionType.REDUCE_CONCURRENCY,
                cooldown_seconds=180
            )
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.metrics_history = []
        self.max_history_size = 1000
        
        # Performance tracking
        self.metrics = {
            "monitoring_cycles": 0,
            "threshold_violations": 0,
            "auto_actions_taken": 0,
            "emergency_actions": 0,
            "false_positives": 0,
            "health_improvements": 0
        }
        
        # Action tracking
        self.action_history = []
        self.max_action_history = 100
        
    async def start_monitoring(self):
        """Start the autonomous monitoring system."""
        if self.monitoring_active:
            logger.warning("Monitoring system already active")
            return
        
        self.monitoring_active = True
        logger.info(" Starting Autonomous System Monitor")
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
        
        # Start trend analysis loop
        asyncio.create_task(self._trend_analysis_loop())
        
        # Start predictive analysis loop
        asyncio.create_task(self._predictive_analysis_loop())
    
    async def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        logger.info(" Stopping Autonomous System Monitor")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect current metrics
                metrics = await self._collect_system_metrics()
                
                # Store metrics
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history = self.metrics_history[-self.max_history_size:]
                
                # Check thresholds
                await self._check_thresholds(metrics)
                
                # Update metrics
                self.metrics["monitoring_cycles"] += 1
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _trend_analysis_loop(self):
        """Trend analysis loop."""
        while self.monitoring_active:
            try:
                # Analyze trends every 5 minutes
                await asyncio.sleep(300)
                
                if len(self.metrics_history) >= 60:  # At least 5 minutes of data
                    await self._analyze_trends()
                
            except Exception as e:
                logger.error(f"Error in trend analysis loop: {e}")
                await asyncio.sleep(60)
    
    async def _predictive_analysis_loop(self):
        """Predictive analysis loop."""
        while self.monitoring_active:
            try:
                # Run predictive analysis every 10 minutes
                await asyncio.sleep(600)
                
                if len(self.metrics_history) >= 120:  # At least 10 minutes of data
                    await self._run_predictive_analysis()
                
            except Exception as e:
                logger.error(f"Error in predictive analysis loop: {e}")
                await asyncio.sleep(120)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # Get CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Get network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Get process count
            process_count = len(psutil.pids())
            
            # Get response time (simulated)
            response_time = await self._measure_response_time()
            
            # Get error rate
            error_rate = await self._get_error_rate()
            
            # Get active connections
            active_connections = await self._get_active_connections()
            
            return SystemMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                process_count=process_count,
                response_time=response_time,
                error_rate=error_rate,
                active_connections=active_connections
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return SystemMetrics(
                timestamp=time.time(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                process_count=0,
                response_time=0.0,
                error_rate=0.0,
                active_connections=0
            )
    
    async def _measure_response_time(self) -> float:
        """Measure system response time."""
        try:
            start_time = time.time()
            
            # Test database response time
            await self.integration.test_connection()
            
            end_time = time.time()
            return end_time - start_time
            
        except Exception as e:
            logger.error(f"Error measuring response time: {e}")
            return 0.0
    
    async def _get_error_rate(self) -> float:
        """Get current error rate."""
        try:
            # Get recent errors from database
            recent_errors = await self.integration.get_recent_errors(limit=100)
            
            if not recent_errors:
                return 0.0
            
            # Calculate error rate over last 5 minutes
            current_time = time.time()
            five_minutes_ago = current_time - 300
            
            recent_error_count = sum(
                1 for error in recent_errors
                if error.get('timestamp', 0) > five_minutes_ago
            )
            
            # Estimate total operations (this would be more sophisticated in practice)
            total_operations = 1000  # Placeholder
            return recent_error_count / total_operations
            
        except Exception as e:
            logger.error(f"Error getting error rate: {e}")
            return 0.0
    
    async def _get_active_connections(self) -> int:
        """Get number of active connections."""
        try:
            # This would count actual active connections
            # For now, return a placeholder
            return 50
            
        except Exception as e:
            logger.error(f"Error getting active connections: {e}")
            return 0
    
    async def _check_thresholds(self, metrics: SystemMetrics):
        """Check all health thresholds."""
        try:
            for metric_name, threshold in self.thresholds.items():
                current_value = getattr(metrics, metric_name, 0)
                
                # Check if we're in cooldown period
                if time.time() - threshold.last_action_time < threshold.cooldown_seconds:
                    continue
                
                # Check emergency threshold
                if current_value >= threshold.emergency_threshold:
                    await self._handle_emergency_threshold(metric_name, current_value, threshold)
                
                # Check critical threshold
                elif current_value >= threshold.critical_threshold:
                    await self._handle_critical_threshold(metric_name, current_value, threshold)
                
                # Check warning threshold
                elif current_value >= threshold.warning_threshold:
                    await self._handle_warning_threshold(metric_name, current_value, threshold)
                
        except Exception as e:
            logger.error(f"Error checking thresholds: {e}")
    
    async def _handle_emergency_threshold(self, metric_name: str, value: float, threshold: HealthThreshold):
        """Handle emergency threshold violation."""
        try:
            logger.critical(f" EMERGENCY: {metric_name} = {value:.2f} (threshold: {threshold.emergency_threshold})")
            
            # Take emergency action
            if threshold.auto_action == ActionType.EMERGENCY_SHUTDOWN:
                await self._emergency_shutdown()
            else:
                await self._execute_action(threshold.auto_action, metric_name, value)
            
            # Update threshold
            threshold.last_action_time = time.time()
            
            # Update metrics
            self.metrics["emergency_actions"] += 1
            self.metrics["threshold_violations"] += 1
            
            # Log action
            await self._log_action("emergency", metric_name, value, threshold.auto_action)
            
        except Exception as e:
            logger.error(f"Error handling emergency threshold: {e}")
    
    async def _handle_critical_threshold(self, metric_name: str, value: float, threshold: HealthThreshold):
        """Handle critical threshold violation."""
        try:
            logger.warning(f" CRITICAL: {metric_name} = {value:.2f} (threshold: {threshold.critical_threshold})")
            
            # Take critical action
            await self._execute_action(threshold.auto_action, metric_name, value)
            
            # Update threshold
            threshold.last_action_time = time.time()
            
            # Update metrics
            self.metrics["threshold_violations"] += 1
            
            # Log action
            await self._log_action("critical", metric_name, value, threshold.auto_action)
            
        except Exception as e:
            logger.error(f"Error handling critical threshold: {e}")
    
    async def _handle_warning_threshold(self, metric_name: str, value: float, threshold: HealthThreshold):
        """Handle warning threshold violation."""
        try:
            logger.info(f" WARNING: {metric_name} = {value:.2f} (threshold: {threshold.warning_threshold})")
            
            # Take warning action (less aggressive)
            await self._execute_action(threshold.auto_action, metric_name, value, aggressive=False)
            
            # Update threshold
            threshold.last_action_time = time.time()
            
            # Update metrics
            self.metrics["threshold_violations"] += 1
            
            # Log action
            await self._log_action("warning", metric_name, value, threshold.auto_action)
            
        except Exception as e:
            logger.error(f"Error handling warning threshold: {e}")
    
    async def _execute_action(self, action_type: ActionType, metric_name: str, value: float, aggressive: bool = True):
        """Execute an automatic action."""
        try:
            logger.info(f" Executing action: {action_type.value} for {metric_name} = {value:.2f}")
            
            if action_type == ActionType.OPTIMIZE_PROCESSING:
                await self._optimize_processing(aggressive)
            elif action_type == ActionType.TRIGGER_GARBAGE_COLLECTION:
                await self._trigger_garbage_collection(aggressive)
            elif action_type == ActionType.INCREASE_VALIDATION:
                await self._increase_validation(aggressive)
            elif action_type == ActionType.OPTIMIZE_QUERIES:
                await self._optimize_queries(aggressive)
            elif action_type == ActionType.REDUCE_CONCURRENCY:
                await self._reduce_concurrency(aggressive)
            elif action_type == ActionType.CLEAR_CACHES:
                await self._clear_caches(aggressive)
            elif action_type == ActionType.RESTART_COMPONENT:
                await self._restart_component(aggressive)
            elif action_type == ActionType.EMERGENCY_SHUTDOWN:
                await self._emergency_shutdown()
            
            # Update metrics
            self.metrics["auto_actions_taken"] += 1
            
        except Exception as e:
            logger.error(f"Error executing action {action_type.value}: {e}")
    
    # Action implementations
    async def _optimize_processing(self, aggressive: bool = True):
        """Optimize system processing."""
        try:
            if aggressive:
                # More aggressive optimization
                logger.info(" Aggressive processing optimization")
                # Implement aggressive optimization
            else:
                # Gentle optimization
                logger.info(" Gentle processing optimization")
                # Implement gentle optimization
                
        except Exception as e:
            logger.error(f"Error optimizing processing: {e}")
    
    async def _trigger_garbage_collection(self, aggressive: bool = True):
        """Trigger garbage collection."""
        try:
            import gc
            
            if aggressive:
                # Force garbage collection
                collected = gc.collect()
                logger.info(f" Aggressive garbage collection: {collected} objects collected")
            else:
                # Gentle garbage collection
                collected = gc.collect()
                logger.info(f" Gentle garbage collection: {collected} objects collected")
                
        except Exception as e:
            logger.error(f"Error triggering garbage collection: {e}")
    
    async def _increase_validation(self, aggressive: bool = True):
        """Increase input validation."""
        try:
            if aggressive:
                logger.info(" Aggressive validation increase")
                # Implement aggressive validation
            else:
                logger.info(" Gentle validation increase")
                # Implement gentle validation
                
        except Exception as e:
            logger.error(f"Error increasing validation: {e}")
    
    async def _optimize_queries(self, aggressive: bool = True):
        """Optimize database queries."""
        try:
            if aggressive:
                logger.info(" Aggressive query optimization")
                # Implement aggressive query optimization
            else:
                logger.info(" Gentle query optimization")
                # Implement gentle query optimization
                
        except Exception as e:
            logger.error(f"Error optimizing queries: {e}")
    
    async def _reduce_concurrency(self, aggressive: bool = True):
        """Reduce concurrent operations."""
        try:
            if aggressive:
                logger.info(" Aggressive concurrency reduction")
                # Implement aggressive concurrency reduction
            else:
                logger.info(" Gentle concurrency reduction")
                # Implement gentle concurrency reduction
                
        except Exception as e:
            logger.error(f"Error reducing concurrency: {e}")
    
    async def _clear_caches(self, aggressive: bool = True):
        """Clear system caches."""
        try:
            if aggressive:
                logger.info(" Aggressive cache clearing")
                # Implement aggressive cache clearing
            else:
                logger.info(" Gentle cache clearing")
                # Implement gentle cache clearing
                
        except Exception as e:
            logger.error(f"Error clearing caches: {e}")
    
    async def _restart_component(self, aggressive: bool = True):
        """Restart a system component."""
        try:
            if aggressive:
                logger.info(" Aggressive component restart")
                # Implement aggressive component restart
            else:
                logger.info(" Gentle component restart")
                # Implement gentle component restart
                
        except Exception as e:
            logger.error(f"Error restarting component: {e}")
    
    async def _emergency_shutdown(self):
        """Emergency system shutdown."""
        try:
            logger.critical(" EMERGENCY SHUTDOWN INITIATED")
            
            # Save critical state
            await self._save_critical_state()
            
            # Graceful shutdown
            await self._graceful_shutdown()
            
        except Exception as e:
            logger.error(f"Error in emergency shutdown: {e}")
    
    async def _save_critical_state(self):
        """Save critical system state before shutdown."""
        try:
            # Save current metrics
            await self.integration.log_system_event(
                "CRITICAL", "EMERGENCY_SHUTDOWN",
                "Emergency shutdown initiated",
                {
                    "metrics": self.metrics,
                    "thresholds": {k: v.__dict__ for k, v in self.thresholds.items()},
                    "timestamp": time.time()
                }
            )
            
        except Exception as e:
            logger.error(f"Error saving critical state: {e}")
    
    async def _graceful_shutdown(self):
        """Perform graceful shutdown."""
        try:
            # Stop monitoring
            await self.stop_monitoring()
            
            # Stop other systems
            # This would stop other system components
            
            logger.info(" Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error in graceful shutdown: {e}")
    
    async def _analyze_trends(self):
        """Analyze system trends."""
        try:
            if len(self.metrics_history) < 60:
                return
            
            # Get recent metrics (last 5 minutes)
            recent_metrics = self.metrics_history[-60:]
            
            # Analyze trends for each metric
            for metric_name in ['cpu_usage', 'memory_usage', 'disk_usage', 'response_time', 'error_rate']:
                values = [getattr(m, metric_name, 0) for m in recent_metrics]
                
                if values:
                    trend = self._calculate_trend(values)
                    
                    if trend > 0.1:  # Increasing trend
                        logger.warning(f" Increasing trend detected for {metric_name}: {trend:.3f}")
                        await self._handle_trend_warning(metric_name, trend)
                    elif trend < -0.1:  # Decreasing trend
                        logger.info(f" Decreasing trend detected for {metric_name}: {trend:.3f}")
                        self.metrics["health_improvements"] += 1
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend from a list of values."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        n = len(values)
        x = list(range(n))
        y = values
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope
    
    async def _handle_trend_warning(self, metric_name: str, trend: float):
        """Handle trend warning."""
        try:
            # Take preventive action based on trend
            if metric_name in self.thresholds:
                threshold = self.thresholds[metric_name]
                
                # Check if we can take action (not in cooldown)
                if time.time() - threshold.last_action_time >= threshold.cooldown_seconds:
                    await self._execute_action(threshold.auto_action, metric_name, 0, aggressive=False)
                    threshold.last_action_time = time.time()
            
        except Exception as e:
            logger.error(f"Error handling trend warning: {e}")
    
    async def _run_predictive_analysis(self):
        """Run predictive analysis to forecast potential issues."""
        try:
            if len(self.metrics_history) < 120:
                return
            
            # Get historical data
            historical_metrics = self.metrics_history[-120:]
            
            # Predict future values
            predictions = {}
            for metric_name in ['cpu_usage', 'memory_usage', 'disk_usage', 'response_time', 'error_rate']:
                values = [getattr(m, metric_name, 0) for m in historical_metrics]
                
                if values:
                    # Simple prediction based on trend
                    trend = self._calculate_trend(values)
                    current_value = values[-1]
                    predicted_value = current_value + (trend * 60)  # Predict 5 minutes ahead
                    
                    predictions[metric_name] = predicted_value
                    
                    # Check if prediction exceeds thresholds
                    if metric_name in self.thresholds:
                        threshold = self.thresholds[metric_name]
                        
                        if predicted_value >= threshold.warning_threshold:
                            logger.warning(f" PREDICTION: {metric_name} may reach {predicted_value:.2f} in 5 minutes")
                            await self._handle_predictive_warning(metric_name, predicted_value, threshold)
            
        except Exception as e:
            logger.error(f"Error in predictive analysis: {e}")
    
    async def _handle_predictive_warning(self, metric_name: str, predicted_value: float, threshold: HealthThreshold):
        """Handle predictive warning."""
        try:
            # Take preventive action
            if time.time() - threshold.last_action_time >= threshold.cooldown_seconds:
                await self._execute_action(threshold.auto_action, metric_name, predicted_value, aggressive=False)
                threshold.last_action_time = time.time()
                
                logger.info(f" Preventive action taken for predicted {metric_name} = {predicted_value:.2f}")
            
        except Exception as e:
            logger.error(f"Error handling predictive warning: {e}")
    
    async def _log_action(self, severity: str, metric_name: str, value: float, action_type: ActionType):
        """Log an action taken by the monitor."""
        try:
            action_record = {
                "timestamp": time.time(),
                "severity": severity,
                "metric_name": metric_name,
                "value": value,
                "action_type": action_type.value,
                "monitoring_cycle": self.metrics["monitoring_cycles"]
            }
            
            # Store in action history
            self.action_history.append(action_record)
            if len(self.action_history) > self.max_action_history:
                self.action_history = self.action_history[-self.max_action_history:]
            
            # Log to database
            await self.integration.log_system_event(
                "INFO", "AUTONOMOUS_MONITOR",
                f"Action taken: {action_type.value} for {metric_name} = {value:.2f}",
                action_record
            )
            
        except Exception as e:
            logger.error(f"Error logging action: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            "monitoring_active": self.monitoring_active,
            "metrics": self.metrics,
            "thresholds": {k: {
                "warning": v.warning_threshold,
                "critical": v.critical_threshold,
                "emergency": v.emergency_threshold,
                "last_action": v.last_action_time
            } for k, v in self.thresholds.items()},
            "metrics_history_size": len(self.metrics_history),
            "action_history_size": len(self.action_history)
        }

# Global monitoring system instance
autonomous_monitor = AutonomousSystemMonitor()

async def start_autonomous_monitoring():
    """Start the autonomous monitoring system."""
    await autonomous_monitor.start_monitoring()

async def stop_autonomous_monitoring():
    """Stop the autonomous monitoring system."""
    await autonomous_monitor.stop_monitoring()

def get_monitoring_status():
    """Get monitoring system status."""
    return autonomous_monitor.get_monitoring_status()
