"""
Self-Configuring System - Phase 1 Implementation

This system automatically configures itself based on environment, performance,
and requirements without human intervention.
"""

import asyncio
import logging
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..database.system_integration import get_system_integration

logger = logging.getLogger(__name__)

class ConfigurationType(Enum):
    """Types of configuration changes."""
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    SECURITY = "security"
    FEATURE = "feature"
    OPTIMIZATION = "optimization"
    ADAPTATION = "adaptation"

class ConfigurationPriority(Enum):
    """Configuration change priorities."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ConfigurationItem:
    """A configuration item with its current and target values."""
    key: str
    current_value: Any
    target_value: Any
    config_type: ConfigurationType
    priority: ConfigurationPriority
    confidence: float
    reasoning: str
    last_updated: float

@dataclass
class ConfigurationChange:
    """A proposed configuration change."""
    change_id: str
    config_type: ConfigurationType
    priority: ConfigurationPriority
    changes: List[ConfigurationItem]
    expected_improvement: float
    confidence: float
    reasoning: str
    timestamp: float
    applied: bool = False
    result: Optional[Dict[str, Any]] = None

class SelfConfiguringSystem:
    """
    Self-Configuring System that automatically optimizes system configuration
    based on environment, performance, and requirements.
    
    Features:
    - Automatic configuration optimization
    - Environment adaptation
    - Performance-based tuning
    - A/B testing of configurations
    - Rollback capabilities
    - Learning from configuration changes
    """
    
    def __init__(self):
        self.integration = get_system_integration()
        
        # Configuration state
        self.current_config = {}
        self.config_history = []
        self.pending_changes = []
        self.applied_changes = []
        
        # Performance tracking
        self.performance_baseline = {}
        self.config_performance = {}
        self.optimization_cycles = 0
        
        # Learning parameters
        self.learning_threshold = 5  # Learn after 5 configuration cycles
        self.confidence_threshold = 0.7  # Minimum confidence for auto-config
        self.rollback_threshold = 0.3  # Rollback if performance drops below this
        
        # System state
        self.configuring_active = False
        self.last_config_cycle = 0
        self.config_cycle_interval = 300  # 5 minutes
        
        # Metrics
        self.metrics = {
            "config_cycles": 0,
            "config_changes_applied": 0,
            "config_changes_rolled_back": 0,
            "performance_improvements": 0,
            "performance_degradations": 0,
            "learning_cycles": 0
        }
        
    async def start_configuring(self):
        """Start the self-configuring system."""
        if self.configuring_active:
            logger.warning("Self-configuring system already active")
            return
        
        self.configuring_active = True
        logger.info(" Starting Self-Configuring System")
        
        # Load current configuration
        await self._load_current_configuration()
        
        # Start configuration loop
        asyncio.create_task(self._configuration_loop())
        
        # Start performance monitoring
        asyncio.create_task(self._performance_monitoring_loop())
        
        # Start learning loop
        asyncio.create_task(self._learning_loop())
    
    async def stop_configuring(self):
        """Stop the self-configuring system."""
        self.configuring_active = False
        logger.info(" Stopping Self-Configuring System")
    
    async def _configuration_loop(self):
        """Main configuration loop."""
        while self.configuring_active:
            try:
                current_time = time.time()
                
                if current_time - self.last_config_cycle >= self.config_cycle_interval:
                    await self._run_configuration_cycle()
                    self.last_config_cycle = current_time
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in configuration loop: {e}")
                await asyncio.sleep(30)
    
    async def _performance_monitoring_loop(self):
        """Performance monitoring loop."""
        while self.configuring_active:
            try:
                # Monitor performance of current configuration
                await self._monitor_configuration_performance()
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _learning_loop(self):
        """Learning loop for configuration optimization."""
        while self.configuring_active:
            try:
                # Learn from configuration changes
                await self._learn_from_configuration_changes()
                
                await asyncio.sleep(300)  # Learn every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                await asyncio.sleep(60)
    
    async def _run_configuration_cycle(self):
        """Run a complete configuration cycle."""
        try:
            logger.info(" Running configuration cycle")
            
            # 1. Analyze current performance
            current_performance = await self._analyze_current_performance()
            
            # 2. Identify optimization opportunities
            opportunities = await self._identify_optimization_opportunities(current_performance)
            
            # 3. Generate configuration changes
            changes = await self._generate_configuration_changes(opportunities)
            
            # 4. Test and apply changes
            for change in changes:
                if await self._test_configuration_change(change):
                    await self._apply_configuration_change(change)
                else:
                    logger.warning(f"Configuration change failed testing: {change.change_id}")
            
            # 5. Update metrics
            self.metrics["config_cycles"] += 1
            
            logger.info(f" Configuration cycle completed: {len(changes)} changes processed")
            
        except Exception as e:
            logger.error(f"Error in configuration cycle: {e}")
    
    async def _load_current_configuration(self):
        """Load current system configuration."""
        try:
            # Load configuration from various sources
            self.current_config = {
                # Database configuration
                'database_pool_size': 10,
                'database_timeout': 30,
                'database_retry_attempts': 3,
                
                # Memory configuration
                'memory_cache_size': 1000,
                'memory_cleanup_interval': 300,
                'memory_max_usage': 0.8,
                
                # Performance configuration
                'max_concurrent_operations': 50,
                'response_timeout': 5.0,
                'batch_size': 100,
                
                # Learning configuration
                'learning_rate': 0.1,
                'exploration_rate': 0.2,
                'adaptation_speed': 0.5,
                
                # Monitoring configuration
                'monitoring_interval': 5,
                'health_check_interval': 30,
                'log_level': 'INFO'
            }
            
            logger.info(" Current configuration loaded")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    async def _analyze_current_performance(self) -> Dict[str, Any]:
        """Analyze current system performance."""
        try:
            # Get system metrics
            performance = {
                'response_time': await self._measure_response_time(),
                'throughput': await self._measure_throughput(),
                'error_rate': await self._measure_error_rate(),
                'resource_usage': await self._measure_resource_usage(),
                'memory_usage': await self._measure_memory_usage(),
                'cpu_usage': await self._measure_cpu_usage()
            }
            
            # Calculate performance score
            performance['score'] = self._calculate_performance_score(performance)
            
            return performance
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return {}
    
    async def _measure_response_time(self) -> float:
        """Measure average response time."""
        try:
            # Simulate response time measurement
            start_time = time.time()
            await asyncio.sleep(0.01)  # Simulate work
            return time.time() - start_time
            
        except Exception as e:
            logger.error(f"Error measuring response time: {e}")
            return 0.0
    
    async def _measure_throughput(self) -> float:
        """Measure system throughput."""
        try:
            # This would measure actual throughput
            return 100.0  # Placeholder
            
        except Exception as e:
            logger.error(f"Error measuring throughput: {e}")
            return 0.0
    
    async def _measure_error_rate(self) -> float:
        """Measure system error rate."""
        try:
            # Get recent errors
            recent_errors = await self.integration.get_recent_errors(limit=100)
            total_operations = 1000  # Placeholder
            
            if total_operations > 0:
                return len(recent_errors) / total_operations
            return 0.0
            
        except Exception as e:
            logger.error(f"Error measuring error rate: {e}")
            return 0.0
    
    async def _measure_resource_usage(self) -> Dict[str, float]:
        """Measure resource usage."""
        try:
            import psutil
            
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            }
            
        except Exception as e:
            logger.error(f"Error measuring resource usage: {e}")
            return {}
    
    async def _measure_memory_usage(self) -> float:
        """Measure memory usage."""
        try:
            import psutil
            return psutil.virtual_memory().percent
            
        except Exception as e:
            logger.error(f"Error measuring memory usage: {e}")
            return 0.0
    
    async def _measure_cpu_usage(self) -> float:
        """Measure CPU usage."""
        try:
            import psutil
            return psutil.cpu_percent()
            
        except Exception as e:
            logger.error(f"Error measuring CPU usage: {e}")
            return 0.0
    
    def _calculate_performance_score(self, performance: Dict[str, Any]) -> float:
        """Calculate overall performance score."""
        try:
            # Weighted performance score
            weights = {
                'response_time': 0.3,
                'throughput': 0.2,
                'error_rate': 0.3,
                'memory_usage': 0.1,
                'cpu_usage': 0.1
            }
            
            score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in performance:
                    value = performance[metric]
                    
                    # Normalize value (higher is better for throughput, lower is better for others)
                    if metric == 'throughput':
                        normalized = min(1.0, value / 1000.0)  # Normalize to 0-1
                    else:
                        normalized = max(0.0, 1.0 - value)  # Invert for error rate, response time, etc.
                    
                    score += normalized * weight
                    total_weight += weight
            
            return score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.0
    
    async def _identify_optimization_opportunities(self, performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify optimization opportunities based on performance."""
        opportunities = []
        
        try:
            # Check response time
            if performance.get('response_time', 0) > 2.0:
                opportunities.append({
                    'type': ConfigurationType.PERFORMANCE,
                    'priority': ConfigurationPriority.HIGH,
                    'metric': 'response_time',
                    'current_value': performance.get('response_time'),
                    'target_value': 1.0,
                    'reasoning': 'Response time is too high'
                })
            
            # Check error rate
            if performance.get('error_rate', 0) > 0.05:
                opportunities.append({
                    'type': ConfigurationType.OPTIMIZATION,
                    'priority': ConfigurationPriority.HIGH,
                    'metric': 'error_rate',
                    'current_value': performance.get('error_rate'),
                    'target_value': 0.01,
                    'reasoning': 'Error rate is too high'
                })
            
            # Check memory usage
            if performance.get('memory_usage', 0) > 80.0:
                opportunities.append({
                    'type': ConfigurationType.RESOURCE,
                    'priority': ConfigurationPriority.MEDIUM,
                    'metric': 'memory_usage',
                    'current_value': performance.get('memory_usage'),
                    'target_value': 70.0,
                    'reasoning': 'Memory usage is too high'
                })
            
            # Check CPU usage
            if performance.get('cpu_usage', 0) > 80.0:
                opportunities.append({
                    'type': ConfigurationType.RESOURCE,
                    'priority': ConfigurationPriority.MEDIUM,
                    'metric': 'cpu_usage',
                    'current_value': performance.get('cpu_usage'),
                    'target_value': 70.0,
                    'reasoning': 'CPU usage is too high'
                })
            
            # Check throughput
            if performance.get('throughput', 0) < 50.0:
                opportunities.append({
                    'type': ConfigurationType.PERFORMANCE,
                    'priority': ConfigurationPriority.MEDIUM,
                    'metric': 'throughput',
                    'current_value': performance.get('throughput'),
                    'target_value': 100.0,
                    'reasoning': 'Throughput is too low'
                })
            
        except Exception as e:
            logger.error(f"Error identifying opportunities: {e}")
        
        return opportunities
    
    async def _generate_configuration_changes(self, opportunities: List[Dict[str, Any]]) -> List[ConfigurationChange]:
        """Generate configuration changes based on opportunities."""
        changes = []
        
        try:
            for opportunity in opportunities:
                change = await self._create_configuration_change(opportunity)
                if change:
                    changes.append(change)
            
        except Exception as e:
            logger.error(f"Error generating configuration changes: {e}")
        
        return changes
    
    async def _create_configuration_change(self, opportunity: Dict[str, Any]) -> Optional[ConfigurationChange]:
        """Create a configuration change for an opportunity."""
        try:
            change_id = f"config_change_{int(time.time() * 1000)}"
            config_type = opportunity['type']
            priority = opportunity['priority']
            metric = opportunity['metric']
            
            # Generate configuration items based on the metric
            config_items = await self._generate_config_items(metric, opportunity)
            
            if not config_items:
                return None
            
            # Calculate expected improvement
            expected_improvement = self._calculate_expected_improvement(opportunity, config_items)
            
            # Calculate confidence
            confidence = self._calculate_confidence(opportunity, config_items)
            
            return ConfigurationChange(
                change_id=change_id,
                config_type=config_type,
                priority=priority,
                changes=config_items,
                expected_improvement=expected_improvement,
                confidence=confidence,
                reasoning=opportunity['reasoning'],
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error creating configuration change: {e}")
            return None
    
    async def _generate_config_items(self, metric: str, opportunity: Dict[str, Any]) -> List[ConfigurationItem]:
        """Generate configuration items for a metric."""
        config_items = []
        
        try:
            if metric == 'response_time':
                config_items.extend([
                    ConfigurationItem(
                        key='database_timeout',
                        current_value=self.current_config.get('database_timeout', 30),
                        target_value=15,
                        config_type=ConfigurationType.PERFORMANCE,
                        priority=ConfigurationPriority.HIGH,
                        confidence=0.8,
                        reasoning='Reduce database timeout to improve response time',
                        last_updated=time.time()
                    ),
                    ConfigurationItem(
                        key='response_timeout',
                        current_value=self.current_config.get('response_timeout', 5.0),
                        target_value=3.0,
                        config_type=ConfigurationType.PERFORMANCE,
                        priority=ConfigurationPriority.HIGH,
                        confidence=0.7,
                        reasoning='Reduce response timeout to improve responsiveness',
                        last_updated=time.time()
                    )
                ])
            
            elif metric == 'error_rate':
                config_items.extend([
                    ConfigurationItem(
                        key='database_retry_attempts',
                        current_value=self.current_config.get('database_retry_attempts', 3),
                        target_value=5,
                        config_type=ConfigurationType.OPTIMIZATION,
                        priority=ConfigurationPriority.HIGH,
                        confidence=0.8,
                        reasoning='Increase retry attempts to reduce errors',
                        last_updated=time.time()
                    ),
                    ConfigurationItem(
                        key='learning_rate',
                        current_value=self.current_config.get('learning_rate', 0.1),
                        target_value=0.05,
                        config_type=ConfigurationType.OPTIMIZATION,
                        priority=ConfigurationPriority.MEDIUM,
                        confidence=0.6,
                        reasoning='Reduce learning rate to improve stability',
                        last_updated=time.time()
                    )
                ])
            
            elif metric == 'memory_usage':
                config_items.extend([
                    ConfigurationItem(
                        key='memory_cache_size',
                        current_value=self.current_config.get('memory_cache_size', 1000),
                        target_value=500,
                        config_type=ConfigurationType.RESOURCE,
                        priority=ConfigurationPriority.MEDIUM,
                        confidence=0.7,
                        reasoning='Reduce cache size to lower memory usage',
                        last_updated=time.time()
                    ),
                    ConfigurationItem(
                        key='memory_cleanup_interval',
                        current_value=self.current_config.get('memory_cleanup_interval', 300),
                        target_value=180,
                        config_type=ConfigurationType.RESOURCE,
                        priority=ConfigurationPriority.MEDIUM,
                        confidence=0.6,
                        reasoning='Increase cleanup frequency to reduce memory usage',
                        last_updated=time.time()
                    )
                ])
            
            elif metric == 'cpu_usage':
                config_items.extend([
                    ConfigurationItem(
                        key='max_concurrent_operations',
                        current_value=self.current_config.get('max_concurrent_operations', 50),
                        target_value=30,
                        config_type=ConfigurationType.RESOURCE,
                        priority=ConfigurationPriority.MEDIUM,
                        confidence=0.7,
                        reasoning='Reduce concurrent operations to lower CPU usage',
                        last_updated=time.time()
                    ),
                    ConfigurationItem(
                        key='batch_size',
                        current_value=self.current_config.get('batch_size', 100),
                        target_value=50,
                        config_type=ConfigurationType.RESOURCE,
                        priority=ConfigurationPriority.MEDIUM,
                        confidence=0.6,
                        reasoning='Reduce batch size to lower CPU usage',
                        last_updated=time.time()
                    )
                ])
            
            elif metric == 'throughput':
                config_items.extend([
                    ConfigurationItem(
                        key='database_pool_size',
                        current_value=self.current_config.get('database_pool_size', 10),
                        target_value=20,
                        config_type=ConfigurationType.PERFORMANCE,
                        priority=ConfigurationPriority.MEDIUM,
                        confidence=0.7,
                        reasoning='Increase database pool size to improve throughput',
                        last_updated=time.time()
                    ),
                    ConfigurationItem(
                        key='batch_size',
                        current_value=self.current_config.get('batch_size', 100),
                        target_value=200,
                        config_type=ConfigurationType.PERFORMANCE,
                        priority=ConfigurationPriority.MEDIUM,
                        confidence=0.6,
                        reasoning='Increase batch size to improve throughput',
                        last_updated=time.time()
                    )
                ])
            
        except Exception as e:
            logger.error(f"Error generating config items: {e}")
        
        return config_items
    
    def _calculate_expected_improvement(self, opportunity: Dict[str, Any], config_items: List[ConfigurationItem]) -> float:
        """Calculate expected improvement from configuration changes."""
        try:
            # Simple calculation based on the difference between current and target values
            current_value = opportunity['current_value']
            target_value = opportunity['target_value']
            
            if current_value == 0:
                return 0.0
            
            # Calculate improvement percentage
            if opportunity['metric'] in ['response_time', 'error_rate', 'memory_usage', 'cpu_usage']:
                # Lower is better
                improvement = (current_value - target_value) / current_value
            else:
                # Higher is better (throughput)
                improvement = (target_value - current_value) / current_value
            
            return max(0.0, min(1.0, improvement))  # Clamp to 0-1
            
        except Exception as e:
            logger.error(f"Error calculating expected improvement: {e}")
            return 0.0
    
    def _calculate_confidence(self, opportunity: Dict[str, Any], config_items: List[ConfigurationItem]) -> float:
        """Calculate confidence in the configuration change."""
        try:
            # Base confidence on the number of config items and their individual confidence
            if not config_items:
                return 0.0
            
            total_confidence = sum(item.confidence for item in config_items)
            avg_confidence = total_confidence / len(config_items)
            
            # Adjust based on priority
            priority_multiplier = {
                ConfigurationPriority.LOW: 0.8,
                ConfigurationPriority.MEDIUM: 0.9,
                ConfigurationPriority.HIGH: 1.0,
                ConfigurationPriority.CRITICAL: 1.1
            }.get(opportunity['priority'], 1.0)
            
            return min(1.0, avg_confidence * priority_multiplier)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    async def _test_configuration_change(self, change: ConfigurationChange) -> bool:
        """Test a configuration change in a safe environment."""
        try:
            logger.info(f" Testing configuration change: {change.change_id}")
            
            # Create rollback point
            rollback_config = dict(self.current_config)
            
            # Apply changes temporarily
            for item in change.changes:
                self.current_config[item.key] = item.target_value
            
            # Test the configuration
            test_result = await self._test_configuration()
            
            # Restore original configuration
            self.current_config = rollback_config
            
            return test_result
            
        except Exception as e:
            logger.error(f"Error testing configuration change: {e}")
            return False
    
    async def _test_configuration(self) -> bool:
        """Test current configuration."""
        try:
            # Test database connection
            await self.integration.test_connection()
            
            # Test basic operations
            # This would test various system operations
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration test failed: {e}")
            return False
    
    async def _apply_configuration_change(self, change: ConfigurationChange):
        """Apply a configuration change."""
        try:
            logger.info(f" Applying configuration change: {change.change_id}")
            
            # Apply each configuration item
            for item in change.changes:
                self.current_config[item.key] = item.target_value
                item.last_updated = time.time()
            
            # Mark change as applied
            change.applied = True
            change.result = {"status": "applied", "timestamp": time.time()}
            
            # Store in history
            self.config_history.append(change)
            self.applied_changes.append(change)
            
            # Update metrics
            self.metrics["config_changes_applied"] += 1
            
            # Log the change
            await self._log_configuration_change(change)
            
            logger.info(f" Configuration change applied: {change.change_id}")
            
        except Exception as e:
            logger.error(f"Error applying configuration change: {e}")
            change.result = {"status": "failed", "error": str(e), "timestamp": time.time()}
    
    async def _log_configuration_change(self, change: ConfigurationChange):
        """Log a configuration change."""
        try:
            await self.integration.log_system_event(
                "INFO", "SELF_CONFIGURING_SYSTEM",
                f"Configuration change applied: {change.change_id}",
                {
                    "change_id": change.change_id,
                    "config_type": change.config_type.value,
                    "priority": change.priority.value,
                    "expected_improvement": change.expected_improvement,
                    "confidence": change.confidence,
                    "changes": [
                        {
                            "key": item.key,
                            "old_value": item.current_value,
                            "new_value": item.target_value,
                            "reasoning": item.reasoning
                        }
                        for item in change.changes
                    ],
                    "timestamp": change.timestamp
                }
            )
            
        except Exception as e:
            logger.error(f"Error logging configuration change: {e}")
    
    async def _monitor_configuration_performance(self):
        """Monitor performance of current configuration."""
        try:
            # Measure current performance
            current_performance = await self._analyze_current_performance()
            
            # Compare with baseline
            if self.performance_baseline:
                improvement = self._calculate_performance_improvement(
                    self.performance_baseline, current_performance
                )
                
                if improvement > 0.1:  # 10% improvement
                    self.metrics["performance_improvements"] += 1
                    logger.info(f" Performance improved by {improvement:.2%}")
                elif improvement < -0.1:  # 10% degradation
                    self.metrics["performance_degradations"] += 1
                    logger.warning(f" Performance degraded by {abs(improvement):.2%}")
                    
                    # Consider rollback
                    if improvement < -self.rollback_threshold:
                        await self._consider_rollback()
            else:
                # Set baseline
                self.performance_baseline = current_performance
            
        except Exception as e:
            logger.error(f"Error monitoring configuration performance: {e}")
    
    def _calculate_performance_improvement(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> float:
        """Calculate performance improvement percentage."""
        try:
            baseline_score = baseline.get('score', 0)
            current_score = current.get('score', 0)
            
            if baseline_score == 0:
                return 0.0
            
            return (current_score - baseline_score) / baseline_score
            
        except Exception as e:
            logger.error(f"Error calculating performance improvement: {e}")
            return 0.0
    
    async def _consider_rollback(self):
        """Consider rolling back recent configuration changes."""
        try:
            if not self.applied_changes:
                return
            
            # Get the most recent change
            recent_change = self.applied_changes[-1]
            
            logger.warning(f" Considering rollback for change: {recent_change.change_id}")
            
            # Rollback the change
            await self._rollback_configuration_change(recent_change)
            
        except Exception as e:
            logger.error(f"Error considering rollback: {e}")
    
    async def _rollback_configuration_change(self, change: ConfigurationChange):
        """Rollback a configuration change."""
        try:
            logger.info(f" Rolling back configuration change: {change.change_id}")
            
            # Restore original values
            for item in change.changes:
                # Find original value (this would be stored in the change)
                original_value = item.current_value  # This should be the original value
                self.current_config[item.key] = original_value
            
            # Mark change as rolled back
            change.applied = False
            change.result = {"status": "rolled_back", "timestamp": time.time()}
            
            # Update metrics
            self.metrics["config_changes_rolled_back"] += 1
            
            logger.info(f" Configuration change rolled back: {change.change_id}")
            
        except Exception as e:
            logger.error(f"Error rolling back configuration change: {e}")
    
    async def _learn_from_configuration_changes(self):
        """Learn from configuration changes to improve future decisions."""
        try:
            if len(self.applied_changes) < self.learning_threshold:
                return
            
            # Analyze successful changes
            successful_changes = [
                change for change in self.applied_changes
                if change.applied and change.result and change.result.get('status') == 'applied'
            ]
            
            # Learn patterns from successful changes
            await self._learn_successful_patterns(successful_changes)
            
            # Update learning metrics
            self.metrics["learning_cycles"] += 1
            
        except Exception as e:
            logger.error(f"Error learning from configuration changes: {e}")
    
    async def _learn_successful_patterns(self, successful_changes: List[ConfigurationChange]):
        """Learn patterns from successful configuration changes."""
        try:
            # Group changes by type
            changes_by_type = {}
            for change in successful_changes:
                config_type = change.config_type
                if config_type not in changes_by_type:
                    changes_by_type[config_type] = []
                changes_by_type[config_type].append(change)
            
            # Learn from each type
            for config_type, changes in changes_by_type.items():
                await self._learn_type_patterns(config_type, changes)
            
        except Exception as e:
            logger.error(f"Error learning successful patterns: {e}")
    
    async def _learn_type_patterns(self, config_type: ConfigurationType, changes: List[ConfigurationChange]):
        """Learn patterns for a specific configuration type."""
        try:
            # Analyze common configuration items
            common_items = {}
            for change in changes:
                for item in change.changes:
                    key = item.key
                    if key not in common_items:
                        common_items[key] = []
                    common_items[key].append(item.target_value)
            
            # Calculate average values for common items
            for key, values in common_items.items():
                if len(values) > 1:  # Only if we have multiple examples
                    avg_value = sum(values) / len(values)
                    logger.info(f" Learned pattern for {key}: average value = {avg_value}")
            
        except Exception as e:
            logger.error(f"Error learning type patterns: {e}")
    
    def get_configuring_status(self) -> Dict[str, Any]:
        """Get current configuring system status."""
        return {
            "configuring_active": self.configuring_active,
            "current_config": self.current_config,
            "metrics": self.metrics,
            "config_history_size": len(self.config_history),
            "applied_changes_size": len(self.applied_changes),
            "performance_baseline": self.performance_baseline
        }

# Global self-configuring system instance
self_configuring_system = SelfConfiguringSystem()

async def start_self_configuring():
    """Start the self-configuring system."""
    await self_configuring_system.start_configuring()

async def stop_self_configuring():
    """Stop the self-configuring system."""
    await self_configuring_system.stop_configuring()

def get_configuring_status():
    """Get configuring system status."""
    return self_configuring_system.get_configuring_status()
