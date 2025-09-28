"""
Weighted Communication System for Enhanced Coordination (TIER 1)

Manages prioritized message routing between subsystems with dynamic weight
adjustment and "myelination" speed queuing for critical communication paths.
Implements intelligent inter-subsystem coordination for AGI-like efficiency.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class PathwayStatus(Enum):
    """Communication pathway status."""
    ACTIVE = "active"
    MYELINATED = "myelinated"  # Speed lane
    DEGRADED = "degraded"
    BLOCKED = "blocked"

@dataclass
class CommunicationMessage:
    """A message between subsystems."""
    message_id: str
    sender_system: str
    receiver_system: str
    message_type: str
    message_data: Dict[str, Any]
    priority: MessagePriority
    sent_timestamp: float
    expected_response: bool = False
    correlation_id: Optional[str] = None
    routing_metadata: Optional[Dict[str, Any]] = None

@dataclass
class CommunicationPathway:
    """A communication pathway between two subsystems."""
    pathway_id: str
    sender_system: str
    receiver_system: str
    message_type: str
    pathway_weight: float  # 0.0 to 2.0 (higher = more important)
    speed_multiplier: float  # 1.0 = normal, higher = faster (myelination)
    priority_level: int
    status: PathwayStatus
    message_count: int
    success_count: int
    failure_count: int
    average_latency: float
    last_message_timestamp: float
    pathway_effectiveness: float
    auto_adjust_enabled: bool

@dataclass
class CommunicationStats:
    """Statistics for communication system performance."""
    total_messages: int
    successful_messages: int
    failed_messages: int
    average_latency: float
    pathway_count: int
    myelinated_pathways: int
    bottleneck_pathways: List[str]

class WeightedCommunicationSystem:
    """
    Weighted Communication System for intelligent inter-subsystem messaging.

    Manages message routing, priority queuing, dynamic weight adjustment,
    and implements "myelination" for frequently used high-priority pathways.
    """

    def __init__(self, db_manager):
        self.db = db_manager

        # Communication state by game_id
        self.game_states: Dict[str, Dict[str, Any]] = {}

        # Registered pathways by game_id
        self.pathways: Dict[str, Dict[str, CommunicationPathway]] = {}

        # Message queues by game_id and priority
        self.message_queues: Dict[str, Dict[MessagePriority, deque]] = {}

        # Communication configuration
        self.config = {
            "max_queue_size": 100,
            "weight_adjustment_threshold": 0.1,
            "myelination_threshold": 0.8,  # Effectiveness threshold for myelination
            "demyelination_threshold": 0.3,  # Effectiveness threshold for removing myelination
            "pathway_cleanup_interval": 60.0,  # Seconds
            "default_pathway_weight": 1.0,
            "max_pathway_weight": 2.0,
            "min_pathway_weight": 0.1,
            "latency_tracking_window": 50,
            "effectiveness_decay_rate": 0.05
        }

        # Known subsystem communication types
        self.message_type_configs = {
            "pattern_detected": {
                "default_priority": MessagePriority.MEDIUM,
                "expected_latency": 0.1,
                "importance_multiplier": 1.2
            },
            "strategy_adjustment": {
                "default_priority": MessagePriority.HIGH,
                "expected_latency": 0.05,
                "importance_multiplier": 1.5
            },
            "losing_streak_warning": {
                "default_priority": MessagePriority.CRITICAL,
                "expected_latency": 0.02,
                "importance_multiplier": 2.0
            },
            "attention_request": {
                "default_priority": MessagePriority.HIGH,
                "expected_latency": 0.03,
                "importance_multiplier": 1.8
            },
            "outcome_feedback": {
                "default_priority": MessagePriority.MEDIUM,
                "expected_latency": 0.08,
                "importance_multiplier": 1.0
            },
            "resource_status": {
                "default_priority": MessagePriority.LOW,
                "expected_latency": 0.2,
                "importance_multiplier": 0.8
            },
            "performance_metrics": {
                "default_priority": MessagePriority.LOW,
                "expected_latency": 0.15,
                "importance_multiplier": 0.9
            }
        }

        # Performance metrics
        self.metrics = {
            "messages_routed": 0,
            "successful_deliveries": 0,
            "failed_deliveries": 0,
            "pathways_created": 0,
            "pathways_myelinated": 0,
            "weight_adjustments": 0,
            "average_system_latency": 0.0
        }

        # Message processing queues
        self._processing_tasks: Dict[str, asyncio.Task] = {}

    async def initialize_communication_system(self, game_id: str, session_id: str) -> Dict[str, Any]:
        """Initialize communication system for a new game."""
        try:
            self.game_states[game_id] = {
                "session_id": session_id,
                "message_routing_enabled": True,
                "priority_queuing_enabled": True,
                "myelination_enabled": True,
                "total_messages_processed": 0,
                "system_performance": {
                    "average_latency": 0.0,
                    "success_rate": 1.0,
                    "throughput": 0.0
                }
            }

            self.pathways[game_id] = {}
            self.message_queues[game_id] = {
                MessagePriority.LOW: deque(maxlen=self.config["max_queue_size"]),
                MessagePriority.MEDIUM: deque(maxlen=self.config["max_queue_size"]),
                MessagePriority.HIGH: deque(maxlen=self.config["max_queue_size"]),
                MessagePriority.CRITICAL: deque(maxlen=self.config["max_queue_size"])
            }

            # Start message processing task
            self._processing_tasks[game_id] = asyncio.create_task(
                self._process_message_queue(game_id)
            )

            # Initialize common pathways
            await self._initialize_common_pathways(game_id, session_id)

            logger.info(f"Initialized weighted communication system for game {game_id}")
            return {"status": "initialized", "pathways": len(self.pathways[game_id])}

        except Exception as e:
            logger.error(f"Failed to initialize communication system: {e}")
            return {"status": "error", "error": str(e)}

    async def route_message(self,
                           sender: str,
                           receiver: str,
                           message_type: str,
                           data: Dict[str, Any],
                           priority: Optional[MessagePriority] = None,
                           game_id: Optional[str] = None) -> str:
        """
        Route a message between subsystems with appropriate priority and weighting.

        Returns message_id for tracking.
        """
        if not game_id or game_id not in self.game_states:
            logger.warning(f"No communication system active for game {game_id}")
            return ""

        try:
            # Determine priority if not specified
            if priority is None:
                priority = self._determine_message_priority(message_type, data)

            # Create message
            message = CommunicationMessage(
                message_id=f"msg_{game_id}_{int(time.time() * 1000000)}",
                sender_system=sender,
                receiver_system=receiver,
                message_type=message_type,
                message_data=data,
                priority=priority,
                sent_timestamp=time.time(),
                routing_metadata={"game_id": game_id}
            )

            # Get or create pathway
            pathway = await self._get_or_create_pathway(game_id, sender, receiver, message_type)

            # Apply pathway weighting to priority
            effective_priority = self._apply_pathway_weighting(priority, pathway)

            # Add to appropriate queue
            queue = self.message_queues[game_id][effective_priority]
            if len(queue) >= self.config["max_queue_size"]:
                # Remove oldest low-priority message if queue is full
                if effective_priority in [MessagePriority.HIGH, MessagePriority.CRITICAL]:
                    self._make_room_in_queue(game_id, effective_priority)
                else:
                    logger.warning(f"Message queue full for {effective_priority}, dropping message")
                    return ""

            queue.append((message, pathway))

            # Log message for analysis
            await self._log_message(message, pathway)

            self.metrics["messages_routed"] += 1

            return message.message_id

        except Exception as e:
            logger.error(f"Error routing message: {e}")
            return ""

    async def adjust_pathway_weights(self,
                                   game_id: str,
                                   pathway_id: str,
                                   effectiveness_feedback: float) -> bool:
        """Adjust pathway weights based on effectiveness feedback."""
        if game_id not in self.pathways or pathway_id not in self.pathways[game_id]:
            return False

        try:
            pathway = self.pathways[game_id][pathway_id]

            # Calculate current effectiveness
            current_effectiveness = pathway.pathway_effectiveness

            # Determine weight adjustment
            effectiveness_change = effectiveness_feedback - current_effectiveness
            weight_adjustment = effectiveness_change * self.config["weight_adjustment_threshold"]

            # Apply adjustment
            old_weight = pathway.pathway_weight
            new_weight = max(self.config["min_pathway_weight"],
                           min(self.config["max_pathway_weight"],
                               old_weight + weight_adjustment))

            pathway.pathway_weight = new_weight
            pathway.pathway_effectiveness = effectiveness_feedback

            # Check for myelination/demyelination
            await self._check_pathway_myelination(game_id, pathway)

            # Store weight adjustment in database
            await self._store_pathway_update(game_id, pathway)

            self.metrics["weight_adjustments"] += 1

            logger.debug(f"Adjusted pathway {pathway_id} weight: {old_weight:.3f} -> {new_weight:.3f}")
            return True

        except Exception as e:
            logger.error(f"Error adjusting pathway weights: {e}")
            return False

    async def create_speed_lane(self, game_id: str, pathway_id: str, duration: float) -> bool:
        """Create a temporary speed lane (myelination) for a pathway."""
        if game_id not in self.pathways or pathway_id not in self.pathways[game_id]:
            return False

        try:
            pathway = self.pathways[game_id][pathway_id]

            # Apply temporary myelination
            pathway.speed_multiplier = min(3.0, pathway.speed_multiplier * 2.0)
            pathway.status = PathwayStatus.MYELINATED

            # Schedule demyelination
            asyncio.create_task(self._schedule_demyelination(game_id, pathway_id, duration))

            logger.info(f"Created speed lane for pathway {pathway_id} for {duration} seconds")
            return True

        except Exception as e:
            logger.error(f"Error creating speed lane: {e}")
            return False

    async def get_communication_recommendations(self, game_id: str, subsystem_id: str) -> List[Dict[str, Any]]:
        """Get communication recommendations for a subsystem."""
        if game_id not in self.pathways:
            return []

        try:
            recommendations = []

            # Analyze pathways for this subsystem
            for pathway_id, pathway in self.pathways[game_id].items():
                if pathway.sender_system == subsystem_id or pathway.receiver_system == subsystem_id:

                    # Check for performance issues
                    if pathway.pathway_effectiveness < 0.5:
                        recommendations.append({
                            "type": "pathway_optimization",
                            "pathway_id": pathway_id,
                            "issue": "Low effectiveness",
                            "suggestion": f"Consider optimizing {pathway.message_type} communication",
                            "priority": "medium"
                        })

                    # Check for high latency
                    if pathway.average_latency > self.message_type_configs.get(
                        pathway.message_type, {}
                    ).get("expected_latency", 0.1) * 2:
                        recommendations.append({
                            "type": "latency_improvement",
                            "pathway_id": pathway_id,
                            "issue": "High latency",
                            "suggestion": f"Optimize processing for {pathway.message_type} messages",
                            "priority": "high"
                        })

                    # Suggest myelination for high-performing pathways
                    if (pathway.pathway_effectiveness > self.config["myelination_threshold"] and
                        pathway.status != PathwayStatus.MYELINATED):
                        recommendations.append({
                            "type": "myelination_candidate",
                            "pathway_id": pathway_id,
                            "issue": "High performance pathway",
                            "suggestion": f"Consider myelinating {pathway.message_type} pathway",
                            "priority": "low"
                        })

            return recommendations

        except Exception as e:
            logger.error(f"Error generating communication recommendations: {e}")
            return []

    async def _process_message_queue(self, game_id: str):
        """Process messages from the queue with priority ordering."""
        while game_id in self.game_states:
            try:
                # Process messages in priority order
                for priority in [MessagePriority.CRITICAL, MessagePriority.HIGH,
                               MessagePriority.MEDIUM, MessagePriority.LOW]:

                    queue = self.message_queues[game_id][priority]
                    if queue:
                        message, pathway = queue.popleft()
                        await self._deliver_message(game_id, message, pathway)

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error processing message queue for game {game_id}: {e}")
                await asyncio.sleep(0.1)

    async def _deliver_message(self, game_id: str, message: CommunicationMessage, pathway: CommunicationPathway):
        """Deliver a message to its destination."""
        try:
            delivery_start = time.time()

            # Apply speed multiplier (myelination effect)
            if pathway.status == PathwayStatus.MYELINATED:
                # Simulate faster processing
                await asyncio.sleep(max(0.001, 0.01 / pathway.speed_multiplier))
            else:
                await asyncio.sleep(0.01)  # Normal processing delay

            # Simulate message delivery (in real implementation, this would route to actual subsystem)
            delivery_success = await self._simulate_message_delivery(message, pathway)

            delivery_time = time.time() - delivery_start

            # Update pathway statistics
            pathway.message_count += 1
            if delivery_success:
                pathway.success_count += 1
                self.metrics["successful_deliveries"] += 1
            else:
                pathway.failure_count += 1
                self.metrics["failed_deliveries"] += 1

            # Update latency
            pathway.average_latency = (
                (pathway.average_latency * (pathway.message_count - 1) + delivery_time) /
                pathway.message_count
            )

            # Update effectiveness
            success_rate = pathway.success_count / pathway.message_count if pathway.message_count > 0 else 0
            latency_factor = max(0.1, 1.0 - pathway.average_latency)
            pathway.pathway_effectiveness = success_rate * latency_factor

            pathway.last_message_timestamp = time.time()

            # Log delivery
            await self._log_message_delivery(message, pathway, delivery_success, delivery_time)

        except Exception as e:
            logger.error(f"Error delivering message: {e}")

    async def _simulate_message_delivery(self, message: CommunicationMessage, pathway: CommunicationPathway) -> bool:
        """Simulate message delivery (placeholder for actual delivery logic)."""
        # In a real implementation, this would route to the actual subsystem
        # For now, simulate based on pathway effectiveness
        success_probability = max(0.1, pathway.pathway_effectiveness)
        return True  # Always succeed for simulation

    def _determine_message_priority(self, message_type: str, data: Dict[str, Any]) -> MessagePriority:
        """Determine message priority based on type and content."""
        # Check message type configuration
        if message_type in self.message_type_configs:
            base_priority = self.message_type_configs[message_type]["default_priority"]
        else:
            base_priority = MessagePriority.MEDIUM

        # Adjust based on message content
        if "urgent" in data or "critical" in data:
            return MessagePriority.CRITICAL
        elif "high_priority" in data:
            return MessagePriority.HIGH

        return base_priority

    def _apply_pathway_weighting(self, priority: MessagePriority, pathway: CommunicationPathway) -> MessagePriority:
        """Apply pathway weighting to determine effective priority."""
        # Higher weight pathways get priority boost
        if pathway.pathway_weight > 1.5 and priority == MessagePriority.MEDIUM:
            return MessagePriority.HIGH
        elif pathway.pathway_weight > 1.8 and priority == MessagePriority.HIGH:
            return MessagePriority.CRITICAL

        return priority

    def _make_room_in_queue(self, game_id: str, target_priority: MessagePriority):
        """Make room in queue by removing lower priority messages."""
        if target_priority in [MessagePriority.HIGH, MessagePriority.CRITICAL]:
            # Remove from lower priority queues
            for lower_priority in [MessagePriority.LOW, MessagePriority.MEDIUM]:
                queue = self.message_queues[game_id][lower_priority]
                if queue:
                    queue.popleft()  # Remove oldest lower priority message
                    break

    async def _get_or_create_pathway(self,
                                   game_id: str,
                                   sender: str,
                                   receiver: str,
                                   message_type: str) -> CommunicationPathway:
        """Get existing pathway or create new one."""
        pathway_id = f"{sender}_{receiver}_{message_type}"

        if pathway_id in self.pathways[game_id]:
            return self.pathways[game_id][pathway_id]

        # Create new pathway
        pathway = CommunicationPathway(
            pathway_id=pathway_id,
            sender_system=sender,
            receiver_system=receiver,
            message_type=message_type,
            pathway_weight=self.config["default_pathway_weight"],
            speed_multiplier=1.0,
            priority_level=2,  # Default medium priority
            status=PathwayStatus.ACTIVE,
            message_count=0,
            success_count=0,
            failure_count=0,
            average_latency=0.0,
            last_message_timestamp=time.time(),
            pathway_effectiveness=0.5,  # Start at neutral effectiveness
            auto_adjust_enabled=True
        )

        self.pathways[game_id][pathway_id] = pathway
        self.metrics["pathways_created"] += 1

        # Store in database
        await self._store_pathway(game_id, pathway)

        return pathway

    async def _check_pathway_myelination(self, game_id: str, pathway: CommunicationPathway):
        """Check if pathway should be myelinated or demyelinated."""
        try:
            # Myelination conditions
            if (pathway.pathway_effectiveness >= self.config["myelination_threshold"] and
                pathway.message_count >= 10 and
                pathway.status != PathwayStatus.MYELINATED):

                pathway.status = PathwayStatus.MYELINATED
                pathway.speed_multiplier = min(2.5, pathway.speed_multiplier * 1.5)
                self.metrics["pathways_myelinated"] += 1

                logger.info(f"Myelinated pathway {pathway.pathway_id} due to high effectiveness")

            # Demyelination conditions
            elif (pathway.pathway_effectiveness <= self.config["demyelination_threshold"] and
                  pathway.status == PathwayStatus.MYELINATED):

                pathway.status = PathwayStatus.ACTIVE
                pathway.speed_multiplier = 1.0

                logger.info(f"Demyelinated pathway {pathway.pathway_id} due to low effectiveness")

        except Exception as e:
            logger.error(f"Error checking pathway myelination: {e}")

    async def _schedule_demyelination(self, game_id: str, pathway_id: str, duration: float):
        """Schedule automatic demyelination after duration."""
        await asyncio.sleep(duration)

        if (game_id in self.pathways and
            pathway_id in self.pathways[game_id]):
            pathway = self.pathways[game_id][pathway_id]
            pathway.speed_multiplier = 1.0
            pathway.status = PathwayStatus.ACTIVE

    async def _initialize_common_pathways(self, game_id: str, session_id: str):
        """Initialize common communication pathways."""
        common_pathways = [
            ("real_time_learning", "strategy_discovery", "pattern_detected"),
            ("real_time_learning", "attention_controller", "attention_request"),
            ("strategy_discovery", "action_selection", "strategy_adjustment"),
            ("losing_streak_detection", "attention_controller", "losing_streak_warning"),
            ("attention_controller", "all_systems", "resource_allocation"),
            ("action_selection", "real_time_learning", "outcome_feedback")
        ]

        for sender, receiver, message_type in common_pathways:
            await self._get_or_create_pathway(game_id, sender, receiver, message_type)

    async def _store_pathway(self, game_id: str, pathway: CommunicationPathway):
        """Store pathway in database."""
        try:
            state = self.game_states[game_id]

            await self.db.execute_query(
                """INSERT INTO communication_pathways
                   (pathway_id, sender_system, receiver_system, message_type, pathway_weight,
                    speed_multiplier, priority_level, message_count, success_count, failure_count,
                    average_latency, pathway_effectiveness, auto_adjust_enabled)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (pathway.pathway_id, pathway.sender_system, pathway.receiver_system,
                 pathway.message_type, pathway.pathway_weight, pathway.speed_multiplier,
                 pathway.priority_level, pathway.message_count, pathway.success_count,
                 pathway.failure_count, pathway.average_latency, pathway.pathway_effectiveness,
                 pathway.auto_adjust_enabled)
            )

        except Exception as e:
            logger.error(f"Error storing pathway: {e}")

    async def _store_pathway_update(self, game_id: str, pathway: CommunicationPathway):
        """Update pathway in database."""
        try:
            await self.db.execute_query(
                """UPDATE communication_pathways
                   SET pathway_weight = ?, speed_multiplier = ?, message_count = ?,
                       success_count = ?, failure_count = ?, average_latency = ?,
                       pathway_effectiveness = ?, last_message_timestamp = ?, updated_at = ?
                   WHERE pathway_id = ?""",
                (pathway.pathway_weight, pathway.speed_multiplier, pathway.message_count,
                 pathway.success_count, pathway.failure_count, pathway.average_latency,
                 pathway.pathway_effectiveness, pathway.last_message_timestamp,
                 time.time(), pathway.pathway_id)
            )

        except Exception as e:
            logger.error(f"Error updating pathway: {e}")

    async def _log_message(self, message: CommunicationMessage, pathway: CommunicationPathway):
        """Log message for analysis."""
        try:
            game_id = message.routing_metadata.get("game_id")
            if not game_id:
                return

            state = self.game_states[game_id]

            await self.db.execute_query(
                """INSERT INTO communication_message_logs
                   (message_id, pathway_id, game_id, session_id, sender_system, receiver_system,
                    message_type, message_data, message_priority, sent_timestamp, message_size_bytes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (message.message_id, pathway.pathway_id, game_id, state["session_id"],
                 message.sender_system, message.receiver_system, message.message_type,
                 json.dumps(message.message_data), message.priority.value,
                 message.sent_timestamp, len(json.dumps(message.message_data)))
            )

        except Exception as e:
            logger.error(f"Error logging message: {e}")

    async def _log_message_delivery(self,
                                  message: CommunicationMessage,
                                  pathway: CommunicationPathway,
                                  success: bool,
                                  latency: float):
        """Log message delivery results."""
        try:
            current_time = time.time()

            await self.db.execute_query(
                """UPDATE communication_message_logs
                   SET received_timestamp = ?, processed_timestamp = ?,
                       processing_success = ?, latency_ms = ?
                   WHERE message_id = ?""",
                (current_time, current_time, success, latency * 1000, message.message_id)
            )

        except Exception as e:
            logger.error(f"Error logging message delivery: {e}")

    async def get_communication_stats(self, game_id: str) -> CommunicationStats:
        """Get communication statistics for a game."""
        if game_id not in self.pathways:
            return CommunicationStats(0, 0, 0, 0.0, 0, 0, [])

        try:
            pathways = self.pathways[game_id]
            total_messages = sum(p.message_count for p in pathways.values())
            successful_messages = sum(p.success_count for p in pathways.values())
            failed_messages = sum(p.failure_count for p in pathways.values())

            if total_messages > 0:
                avg_latency = sum(p.average_latency * p.message_count for p in pathways.values()) / total_messages
            else:
                avg_latency = 0.0

            myelinated_count = sum(1 for p in pathways.values() if p.status == PathwayStatus.MYELINATED)

            bottleneck_pathways = [
                p.pathway_id for p in pathways.values()
                if p.average_latency > 0.2 and p.pathway_effectiveness < 0.5
            ]

            return CommunicationStats(
                total_messages=total_messages,
                successful_messages=successful_messages,
                failed_messages=failed_messages,
                average_latency=avg_latency,
                pathway_count=len(pathways),
                myelinated_pathways=myelinated_count,
                bottleneck_pathways=bottleneck_pathways
            )

        except Exception as e:
            logger.error(f"Error getting communication stats: {e}")
            return CommunicationStats(0, 0, 0, 0.0, 0, 0, [])

    async def finalize_communication_system(self, game_id: str) -> Dict[str, Any]:
        """Finalize communication system for a game."""
        if game_id not in self.game_states:
            return {}

        try:
            # Cancel processing task
            if game_id in self._processing_tasks:
                self._processing_tasks[game_id].cancel()
                del self._processing_tasks[game_id]

            # Get final statistics
            final_stats = await self.get_communication_stats(game_id)

            # Clean up state
            del self.game_states[game_id]
            del self.pathways[game_id]
            del self.message_queues[game_id]

            logger.info(f"Finalized communication system for game {game_id}")
            return asdict(final_stats)

        except Exception as e:
            logger.error(f"Error finalizing communication system: {e}")
            return {}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall performance metrics for the communication system."""
        return {
            "total_messages_routed": self.metrics["messages_routed"],
            "successful_deliveries": self.metrics["successful_deliveries"],
            "failed_deliveries": self.metrics["failed_deliveries"],
            "success_rate": (self.metrics["successful_deliveries"] /
                           max(1, self.metrics["successful_deliveries"] + self.metrics["failed_deliveries"])),
            "pathways_created": self.metrics["pathways_created"],
            "pathways_myelinated": self.metrics["pathways_myelinated"],
            "weight_adjustments": self.metrics["weight_adjustments"],
            "active_games": len(self.game_states),
            "total_active_pathways": sum(len(pathways) for pathways in self.pathways.values())
        }