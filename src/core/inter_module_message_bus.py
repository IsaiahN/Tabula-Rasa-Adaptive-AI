#!/usr/bin/env python3
"""
Inter-Module Message Bus - Enhanced Interconnectivity System

Implements a high-performance publish-subscribe system for real-time communication
between Tabula Rasa modules (Governor, Director, Architect, Memory Manager).

Features:
- Priority-based message routing
- Sub-millisecond latency for critical messages
- Message filtering and transformation
- Health monitoring and recovery
- Asynchronous processing with backpressure control
"""

import asyncio
import time
import logging
import heapq
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)

class MessagePriority(Enum):
    """Message priority levels for routing optimization."""
    CRITICAL = 0    # <1ms latency required
    HIGH = 1        # <5ms latency required
    NORMAL = 2      # <10ms latency required
    LOW = 3         # <50ms latency required

class MessageType(Enum):
    """Types of messages in the system."""
    SYSTEM_STATUS = "system_status"
    MEMORY_UPDATE = "memory_update"
    LEARNING_PROGRESS = "learning_progress"
    ACTION_SELECTION = "action_selection"
    PREDICTION_ERROR = "prediction_error"
    GOVERNOR_DECISION = "governor_decision"
    ARCHITECT_REQUEST = "architect_request"
    MEMORY_CLEANUP = "memory_cleanup"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR_REPORT = "error_report"

@dataclass
class Message:
    """Structured message for inter-module communication."""
    topic: str
    message_type: MessageType
    payload: Dict[str, Any]
    priority: MessagePriority
    timestamp: float = field(default_factory=time.time)
    source_module: str = ""
    target_module: Optional[str] = None
    correlation_id: Optional[str] = None
    ttl: float = 5.0  # Time to live in seconds
    
    def __lt__(self, other):
        """Enable comparison for heapq priority queue."""
        if not isinstance(other, Message):
            return NotImplemented
        # Compare by priority first, then timestamp
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.timestamp < other.timestamp
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        return time.time() - self.timestamp > self.ttl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            'topic': self.topic,
            'message_type': self.message_type.value,
            'payload': self.payload,
            'priority': self.priority.value,
            'timestamp': self.timestamp,
            'source_module': self.source_module,
            'target_module': self.target_module,
            'correlation_id': self.correlation_id,
            'ttl': self.ttl
        }

@dataclass
class Subscriber:
    """Subscriber configuration for message filtering."""
    callback: Callable[[Message], None]
    filter_func: Optional[Callable[[Message], bool]] = None
    priority_threshold: MessagePriority = MessagePriority.LOW
    max_processing_time: float = 0.001  # 1ms max processing time
    error_handler: Optional[Callable[[Exception], None]] = None

class InterModuleMessageBus:
    """
    High-performance publish-subscribe message bus for Tabula Rasa modules.
    
    Provides sub-millisecond latency for critical messages and intelligent
    routing based on message priority and module health.
    """
    
    def __init__(self, max_queue_size: int = 10000, worker_threads: int = 4):
        self.max_queue_size = max_queue_size
        self.worker_threads = worker_threads
        
        # Message routing
        self.subscribers: Dict[str, List[Subscriber]] = defaultdict(list)
        self.priority_queues: Dict[MessagePriority, List[Message]] = {
            priority: [] for priority in MessagePriority
        }
        
        # Performance tracking
        self.latency_tracker: Dict[str, List[float]] = defaultdict(list)
        self.message_counts: Dict[MessageType, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Health monitoring
        self.module_health: Dict[str, Dict[str, Any]] = {}
        self.last_heartbeat: Dict[str, float] = {}
        
        # Threading and async
        self.executor = ThreadPoolExecutor(max_workers=worker_threads)
        self.message_lock = threading.RLock()
        self.running = False
        self.processing_tasks: List[asyncio.Task] = []
        
        # Backpressure control
        self.queue_sizes: Dict[MessagePriority, int] = defaultdict(int)
        self.backpressure_threshold = max_queue_size * 0.8
        
        logger.info(f"InterModuleMessageBus initialized with {worker_threads} worker threads")
    
    async def start(self):
        """Start the message bus processing."""
        if self.running:
            return
        
        self.running = True
        
        # Start priority-based message processors
        for priority in MessagePriority:
            task = asyncio.create_task(self._process_priority_queue(priority))
            self.processing_tasks.append(task)
        
        # Start health monitoring
        health_task = asyncio.create_task(self._monitor_module_health())
        self.processing_tasks.append(health_task)
        
        logger.info("InterModuleMessageBus started successfully")
    
    async def stop(self):
        """Stop the message bus processing."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel all processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("InterModuleMessageBus stopped")
    
    async def publish(self, 
                     topic: str, 
                     message_type: MessageType,
                     payload: Dict[str, Any],
                     priority: MessagePriority = MessagePriority.NORMAL,
                     source_module: str = "",
                     target_module: Optional[str] = None,
                     correlation_id: Optional[str] = None) -> bool:
        """
        Publish a message to the bus.
        
        Args:
            topic: Message topic
            message_type: Type of message
            payload: Message data
            priority: Message priority
            source_module: Source module identifier
            target_module: Optional target module
            correlation_id: Optional correlation ID for tracking
            
        Returns:
            bool: True if message was queued successfully
        """
        try:
            # Check backpressure
            if self._is_backpressure_active(priority):
                logger.warning(f"Backpressure active for priority {priority.name}, dropping message")
                return False
            
            # Create message
            message = Message(
                topic=topic,
                message_type=message_type,
                payload=payload,
                priority=priority,
                source_module=source_module,
                target_module=target_module,
                correlation_id=correlation_id
            )
            
            # Queue message
            with self.message_lock:
                if len(self.priority_queues[priority]) >= self.max_queue_size:
                    # Remove oldest message of same priority
                    self.priority_queues[priority].pop(0)
                
                heapq.heappush(self.priority_queues[priority], message)
                self.queue_sizes[priority] += 1
                self.message_counts[message_type] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            self.error_counts["publish_error"] += 1
            return False
    
    def subscribe(self, 
                 topic: str, 
                 callback: Callable[[Message], None],
                 filter_func: Optional[Callable[[Message], bool]] = None,
                 priority_threshold: MessagePriority = MessagePriority.LOW,
                 max_processing_time: float = 0.001,
                 error_handler: Optional[Callable[[Exception], None]] = None):
        """
        Subscribe to messages on a topic.
        
        Args:
            topic: Topic to subscribe to
            callback: Function to call when message received
            filter_func: Optional function to filter messages
            priority_threshold: Minimum priority to process
            max_processing_time: Maximum processing time per message
            error_handler: Optional error handler
        """
        subscriber = Subscriber(
            callback=callback,
            filter_func=filter_func,
            priority_threshold=priority_threshold,
            max_processing_time=max_processing_time,
            error_handler=error_handler
        )
        
        self.subscribers[topic].append(subscriber)
        logger.info(f"Subscribed to topic '{topic}' with priority threshold {priority_threshold.name}")
    
    def unsubscribe(self, topic: str, callback: Callable[[Message], None]):
        """Unsubscribe from a topic."""
        if topic in self.subscribers:
            self.subscribers[topic] = [
                sub for sub in self.subscribers[topic] 
                if sub.callback != callback
            ]
            logger.info(f"Unsubscribed from topic '{topic}'")
    
    async def _process_priority_queue(self, priority: MessagePriority):
        """Process messages for a specific priority level."""
        while self.running:
            try:
                message = None
                
                with self.message_lock:
                    if self.priority_queues[priority]:
                        message = heapq.heappop(self.priority_queues[priority])
                        self.queue_sizes[priority] -= 1
                
                if message is None:
                    await asyncio.sleep(0.001)  # 1ms sleep
                    continue
                
                # Check if message expired
                if message.is_expired():
                    continue
                
                # Process message
                await self._deliver_message(message)
                
            except Exception as e:
                logger.error(f"Error processing priority queue {priority.name}: {e}")
                await asyncio.sleep(0.001)
    
    async def _deliver_message(self, message: Message):
        """Deliver message to subscribers."""
        start_time = time.time()
        
        # Find subscribers for topic
        subscribers = self.subscribers.get(message.topic, [])
        
        # Filter subscribers based on priority and target
        active_subscribers = []
        for subscriber in subscribers:
            if (message.priority.value <= subscriber.priority_threshold.value and
                (message.target_module is None or 
                 message.target_module == subscriber.callback.__self__.__class__.__name__)):
                active_subscribers.append(subscriber)
        
        # Deliver to subscribers
        for subscriber in active_subscribers:
            try:
                # Apply filter if present
                if subscriber.filter_func and not subscriber.filter_func(message):
                    continue
                
                # Check processing time limit
                if subscriber.max_processing_time > 0:
                    # Run in executor with timeout
                    future = self.executor.submit(subscriber.callback, message)
                    try:
                        future.result(timeout=subscriber.max_processing_time)
                    except TimeoutError:
                        logger.warning(f"Subscriber {subscriber.callback} exceeded processing time limit")
                        continue
                else:
                    subscriber.callback(message)
                
            except Exception as e:
                logger.error(f"Error delivering message to subscriber: {e}")
                if subscriber.error_handler:
                    try:
                        subscriber.error_handler(e)
                    except Exception as handler_error:
                        logger.error(f"Error in error handler: {handler_error}")
        
        # Track latency
        latency = time.time() - start_time
        self.latency_tracker[message.topic].append(latency)
        
        # Keep only recent latency measurements
        if len(self.latency_tracker[message.topic]) > 1000:
            self.latency_tracker[message.topic] = self.latency_tracker[message.topic][-500:]
    
    def _is_backpressure_active(self, priority: MessagePriority) -> bool:
        """Check if backpressure is active for a priority level."""
        return self.queue_sizes[priority] > self.backpressure_threshold
    
    async def _monitor_module_health(self):
        """Monitor module health and performance."""
        while self.running:
            try:
                current_time = time.time()
                
                # Check for stale heartbeats
                for module, last_heartbeat in self.last_heartbeat.items():
                    if current_time - last_heartbeat > 30:  # 30 seconds timeout
                        self.module_health[module] = {
                            'status': 'stale',
                            'last_heartbeat': last_heartbeat,
                            'latency': float('inf')
                        }
                
                # Update health metrics
                for topic, latencies in self.latency_tracker.items():
                    if latencies:
                        avg_latency = sum(latencies) / len(latencies)
                        max_latency = max(latencies)
                        
                        # Find module for topic (simplified)
                        module = topic.split('.')[0] if '.' in topic else topic
                        self.module_health[module] = {
                            'status': 'healthy',
                            'avg_latency': avg_latency,
                            'max_latency': max_latency,
                            'message_count': len(latencies)
                        }
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(5)
    
    def register_module_heartbeat(self, module_name: str):
        """Register a heartbeat for a module."""
        self.last_heartbeat[module_name] = time.time()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the message bus."""
        metrics = {
            'queue_sizes': dict(self.queue_sizes),
            'message_counts': dict(self.message_counts),
            'error_counts': dict(self.error_counts),
            'module_health': dict(self.module_health),
            'avg_latencies': {}
        }
        
        # Calculate average latencies
        for topic, latencies in self.latency_tracker.items():
            if latencies:
                metrics['avg_latencies'][topic] = sum(latencies) / len(latencies)
        
        return metrics
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of the message bus."""
        total_messages = sum(self.message_counts.values())
        total_errors = sum(self.error_counts.values())
        
        health_score = 1.0
        if total_messages > 0:
            health_score = 1.0 - (total_errors / total_messages)
        
        return {
            'health_score': health_score,
            'total_messages': total_messages,
            'total_errors': total_errors,
            'queue_utilization': sum(self.queue_sizes.values()) / (self.max_queue_size * len(MessagePriority)),
            'modules_healthy': sum(1 for h in self.module_health.values() if h.get('status') == 'healthy'),
            'total_modules': len(self.module_health)
        }

# Global message bus instance
_message_bus_instance: Optional[InterModuleMessageBus] = None

def get_message_bus() -> InterModuleMessageBus:
    """Get the global message bus instance."""
    global _message_bus_instance
    if _message_bus_instance is None:
        _message_bus_instance = InterModuleMessageBus()
    return _message_bus_instance

async def initialize_message_bus():
    """Initialize the global message bus."""
    global _message_bus_instance
    if _message_bus_instance is None:
        _message_bus_instance = InterModuleMessageBus()
        await _message_bus_instance.start()
    return _message_bus_instance

async def shutdown_message_bus():
    """Shutdown the global message bus."""
    global _message_bus_instance
    if _message_bus_instance is not None:
        await _message_bus_instance.stop()
        _message_bus_instance = None
