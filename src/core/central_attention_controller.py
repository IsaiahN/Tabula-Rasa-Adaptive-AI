"""
Central Attention Controller for Enhanced Coordination (TIER 1)

Monitors computational loads across all subsystems and dynamically allocates
processing priority to optimize overall system performance. Implements
intelligent resource allocation based on game context and subsystem demands.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class SubsystemDemand:
    """A demand for attention from a subsystem."""
    subsystem_name: str
    requested_priority: float  # 0.0 to 1.0
    current_load: float  # 0.0 to 1.0
    processing_complexity: float  # 0.0 to 1.0
    urgency_level: int  # 1=low, 2=medium, 3=high, 4=critical
    justification: str  # Why this subsystem needs attention
    context_data: Dict[str, Any]

@dataclass
class AttentionAllocation:
    """An allocation of attention to subsystems."""
    allocation_id: str
    allocations: Dict[str, float]  # subsystem_name -> priority (0.0 to 1.0)
    allocation_reasoning: str
    total_demand: float
    allocation_timestamp: float
    expected_duration: float
    context_hash: str

@dataclass
class ResourceUsage:
    """Resource usage data for a subsystem."""
    subsystem_name: str
    cpu_usage: float
    memory_usage: float
    processing_time: float
    queue_depth: int
    throughput_rate: float
    error_rate: float
    bottleneck_detected: bool

class CentralAttentionController:
    """
    Central Attention Controller for coordinating computational resources.

    Monitors all subsystems and dynamically allocates processing priority
    based on current demands, historical performance, and game context.
    """

    def __init__(self, db_manager):
        self.db = db_manager

        # Attention allocation state by game_id
        self.game_states: Dict[str, Dict[str, Any]] = {}

        # Known subsystems and their characteristics
        self.registered_subsystems = {
            "real_time_learning": {
                "base_priority": 0.3,
                "max_priority": 0.8,
                "processing_cost": 0.4,
                "response_time_importance": 0.8
            },
            "strategy_discovery": {
                "base_priority": 0.2,
                "max_priority": 0.7,
                "processing_cost": 0.6,
                "response_time_importance": 0.6
            },
            "losing_streak_detection": {
                "base_priority": 0.15,
                "max_priority": 0.9,  # High max because it prevents failures
                "processing_cost": 0.2,
                "response_time_importance": 0.7
            },
            "action_selection": {
                "base_priority": 0.4,
                "max_priority": 0.9,
                "processing_cost": 0.3,
                "response_time_importance": 0.9  # Critical for real-time decisions
            },
            "pattern_detection": {
                "base_priority": 0.25,
                "max_priority": 0.6,
                "processing_cost": 0.5,
                "response_time_importance": 0.5
            },
            "communication_system": {
                "base_priority": 0.1,
                "max_priority": 0.3,
                "processing_cost": 0.1,
                "response_time_importance": 0.9
            }
        }

        # Configuration for attention allocation
        self.config = {
            "allocation_interval": 2.0,  # Seconds between allocation decisions
            "max_total_allocation": 1.0,  # Total priority cannot exceed this
            "priority_decay_rate": 0.1,  # How quickly unused priority decays
            "demand_history_size": 20,  # How many demand records to keep
            "effectiveness_weight": 0.4,  # How much to weight historical effectiveness
            "urgency_multiplier": 1.5,  # How much urgency affects allocation
            "load_balancing_threshold": 0.8,  # When to trigger load balancing
            "bottleneck_detection_threshold": 0.9  # When to flag bottlenecks
        }

        # Performance tracking
        self.metrics = {
            "allocations_made": 0,
            "load_balancing_events": 0,
            "bottlenecks_detected": 0,
            "allocation_changes": 0,
            "average_allocation_effectiveness": 0.0
        }

        # Attention allocation history by game_id
        self.allocation_history: Dict[str, deque] = {}
        self.demand_history: Dict[str, deque] = {}
        self.resource_usage_history: Dict[str, deque] = {}

    async def initialize_attention_monitoring(self, game_id: str, session_id: str) -> Dict[str, Any]:
        """Initialize attention monitoring for a new game."""
        try:
            self.game_states[game_id] = {
                "session_id": session_id,
                "current_allocation": {},
                "last_allocation_time": time.time(),
                "active_demands": {},
                "resource_usage": {},
                "allocation_strategy": "balanced",  # "balanced", "focused", "distributed"
                "context_awareness": 0.5,
                "game_phase": "early"  # "early", "middle", "late", "critical"
            }

            self.allocation_history[game_id] = deque(maxlen=self.config["demand_history_size"])
            self.demand_history[game_id] = deque(maxlen=self.config["demand_history_size"])
            self.resource_usage_history[game_id] = deque(maxlen=self.config["demand_history_size"])

            # Initialize baseline allocation
            baseline_allocation = await self._create_baseline_allocation(game_id, session_id)

            logger.info(f"Initialized attention monitoring for game {game_id}")
            return baseline_allocation

        except Exception as e:
            logger.error(f"Failed to initialize attention monitoring: {e}")
            return {}

    async def allocate_attention_resources(self,
                                         game_id: str,
                                         subsystem_demands: List[SubsystemDemand],
                                         game_context: Optional[Dict[str, Any]] = None) -> AttentionAllocation:
        """
        Allocate attention resources based on subsystem demands and context.

        Returns the attention allocation strategy for this moment.
        """
        if game_id not in self.game_states:
            logger.warning(f"No attention monitoring initialized for game {game_id}")
            return await self._create_emergency_allocation(subsystem_demands)

        state = self.game_states[game_id]

        try:
            # Update game context awareness
            if game_context:
                await self._update_context_awareness(game_id, game_context)

            # Process subsystem demands
            demand_analysis = await self._analyze_subsystem_demands(game_id, subsystem_demands)

            # Calculate optimal allocation
            allocation_strategy = await self._calculate_optimal_allocation(
                game_id, demand_analysis, game_context
            )

            # Create allocation object
            allocation = AttentionAllocation(
                allocation_id=f"attn_{game_id}_{int(time.time() * 1000)}",
                allocations=allocation_strategy["allocations"],
                allocation_reasoning=allocation_strategy["reasoning"],
                total_demand=allocation_strategy["total_demand"],
                allocation_timestamp=time.time(),
                expected_duration=allocation_strategy["duration"],
                context_hash=allocation_strategy["context_hash"]
            )

            # Store allocation
            await self._store_allocation(game_id, allocation, subsystem_demands)

            # Update state
            state["current_allocation"] = allocation.allocations
            state["last_allocation_time"] = allocation.allocation_timestamp
            self.allocation_history[game_id].append(allocation)

            self.metrics["allocations_made"] += 1

            return allocation

        except Exception as e:
            logger.error(f"Error allocating attention resources: {e}")
            return await self._create_emergency_allocation(subsystem_demands)

    async def monitor_subsystem_loads(self, game_id: str, resource_usage_data: List[ResourceUsage]) -> Dict[str, Any]:
        """Monitor current subsystem loads and detect bottlenecks."""
        if game_id not in self.game_states:
            return {}

        try:
            state = self.game_states[game_id]
            monitoring_results = {
                "monitoring_timestamp": time.time(),
                "subsystem_loads": {},
                "bottlenecks_detected": [],
                "load_balancing_needed": False,
                "optimization_suggestions": []
            }

            total_load = 0.0
            max_load = 0.0
            bottleneck_count = 0

            for usage in resource_usage_data:
                subsystem_name = usage.subsystem_name

                # Calculate composite load score
                load_score = self._calculate_load_score(usage)
                monitoring_results["subsystem_loads"][subsystem_name] = load_score

                total_load += load_score
                max_load = max(max_load, load_score)

                # Detect bottlenecks
                if (usage.bottleneck_detected or
                    load_score >= self.config["bottleneck_detection_threshold"]):
                    monitoring_results["bottlenecks_detected"].append({
                        "subsystem": subsystem_name,
                        "load_score": load_score,
                        "issues": self._identify_performance_issues(usage)
                    })
                    bottleneck_count += 1

                # Store resource usage
                await self._store_resource_usage(game_id, usage)

            # Determine if load balancing is needed
            if (max_load >= self.config["load_balancing_threshold"] or
                bottleneck_count >= 2):
                monitoring_results["load_balancing_needed"] = True
                self.metrics["load_balancing_events"] += 1

            # Generate optimization suggestions
            monitoring_results["optimization_suggestions"] = await self._generate_optimization_suggestions(
                game_id, resource_usage_data, monitoring_results
            )

            # Update metrics
            self.metrics["bottlenecks_detected"] += bottleneck_count

            # Store resource usage history
            self.resource_usage_history[game_id].append({
                "timestamp": monitoring_results["monitoring_timestamp"],
                "total_load": total_load,
                "max_load": max_load,
                "bottleneck_count": bottleneck_count
            })

            return monitoring_results

        except Exception as e:
            logger.error(f"Error monitoring subsystem loads: {e}")
            return {}

    async def adjust_processing_priorities(self,
                                         game_id: str,
                                         performance_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust processing priorities based on performance feedback."""
        if game_id not in self.game_states:
            return {}

        try:
            state = self.game_states[game_id]
            adjustment_results = {
                "adjustment_timestamp": time.time(),
                "priority_changes": {},
                "reasoning": "",
                "effectiveness_prediction": 0.0
            }

            # Analyze performance feedback
            feedback_analysis = await self._analyze_performance_feedback(game_id, performance_feedback)

            # Calculate priority adjustments
            priority_adjustments = await self._calculate_priority_adjustments(
                game_id, feedback_analysis
            )

            # Apply adjustments
            for subsystem, adjustment in priority_adjustments.items():
                if subsystem in state["current_allocation"]:
                    old_priority = state["current_allocation"][subsystem]
                    new_priority = max(0.0, min(1.0, old_priority + adjustment))

                    state["current_allocation"][subsystem] = new_priority
                    adjustment_results["priority_changes"][subsystem] = {
                        "old_priority": old_priority,
                        "new_priority": new_priority,
                        "adjustment": adjustment
                    }

            # Generate reasoning
            adjustment_results["reasoning"] = await self._generate_adjustment_reasoning(
                feedback_analysis, priority_adjustments
            )

            # Predict effectiveness
            adjustment_results["effectiveness_prediction"] = await self._predict_adjustment_effectiveness(
                game_id, priority_adjustments
            )

            # Store adjustment decision
            await self._store_adjustment_decision(game_id, adjustment_results, performance_feedback)

            self.metrics["allocation_changes"] += len(priority_adjustments)

            return adjustment_results

        except Exception as e:
            logger.error(f"Error adjusting processing priorities: {e}")
            return {}

    async def get_attention_allocation_strategy(self, game_id: str, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get current attention allocation strategy for a game."""
        if game_id not in self.game_states:
            return {"allocations": {}, "strategy": "none", "reasoning": "No game state found"}

        try:
            state = self.game_states[game_id]

            # Get current allocation
            current_allocation = state.get("current_allocation", {})

            # Analyze current context
            context_analysis = await self._analyze_current_context(game_id, current_context)

            # Determine strategy
            strategy = await self._determine_allocation_strategy(game_id, context_analysis)

            return {
                "allocations": current_allocation,
                "strategy": strategy["type"],
                "reasoning": strategy["reasoning"],
                "context_analysis": context_analysis,
                "recommendations": strategy.get("recommendations", []),
                "expected_effectiveness": strategy.get("effectiveness", 0.5)
            }

        except Exception as e:
            logger.error(f"Error getting attention allocation strategy: {e}")
            return {"allocations": {}, "strategy": "error", "reasoning": str(e)}

    def _calculate_load_score(self, usage: ResourceUsage) -> float:
        """Calculate composite load score for a subsystem."""
        # Weighted combination of different load factors
        cpu_weight = 0.3
        memory_weight = 0.2
        queue_weight = 0.3
        error_weight = 0.2

        load_score = (
            usage.cpu_usage * cpu_weight +
            min(1.0, usage.memory_usage / 1000.0) * memory_weight +  # Normalize memory
            min(1.0, usage.queue_depth / 100.0) * queue_weight +  # Normalize queue depth
            usage.error_rate * error_weight
        )

        return min(1.0, load_score)

    def _identify_performance_issues(self, usage: ResourceUsage) -> List[str]:
        """Identify specific performance issues from resource usage."""
        issues = []

        if usage.cpu_usage > 0.8:
            issues.append("high_cpu_usage")
        if usage.memory_usage > 500:  # MB
            issues.append("high_memory_usage")
        if usage.queue_depth > 50:
            issues.append("processing_backlog")
        if usage.error_rate > 0.1:
            issues.append("high_error_rate")
        if usage.throughput_rate < 1.0:
            issues.append("low_throughput")

        return issues

    async def _create_baseline_allocation(self, game_id: str, session_id: str) -> Dict[str, Any]:
        """Create baseline attention allocation for a new game."""
        baseline_allocations = {}
        total_allocated = 0.0

        # Allocate based on base priorities
        for subsystem, config in self.registered_subsystems.items():
            baseline_allocations[subsystem] = config["base_priority"]
            total_allocated += config["base_priority"]

        # Normalize to ensure total doesn't exceed max
        if total_allocated > self.config["max_total_allocation"]:
            normalization_factor = self.config["max_total_allocation"] / total_allocated
            for subsystem in baseline_allocations:
                baseline_allocations[subsystem] *= normalization_factor

        return {
            "allocations": baseline_allocations,
            "strategy": "baseline",
            "reasoning": "Initial baseline allocation based on subsystem characteristics"
        }

    async def _create_emergency_allocation(self, demands: List[SubsystemDemand]) -> AttentionAllocation:
        """Create emergency allocation when normal processing fails."""
        emergency_allocations = {}

        # Give equal priority to all demanding subsystems
        if demands:
            equal_priority = 1.0 / len(demands)
            for demand in demands:
                emergency_allocations[demand.subsystem_name] = equal_priority

        return AttentionAllocation(
            allocation_id=f"emergency_{int(time.time() * 1000)}",
            allocations=emergency_allocations,
            allocation_reasoning="Emergency equal allocation due to processing error",
            total_demand=len(demands),
            allocation_timestamp=time.time(),
            expected_duration=30.0,  # 30 seconds
            context_hash="emergency"
        )

    async def _analyze_subsystem_demands(self,
                                       game_id: str,
                                       demands: List[SubsystemDemand]) -> Dict[str, Any]:
        """Analyze current subsystem demands."""
        total_requested_priority = sum(d.requested_priority for d in demands)
        max_urgency = max(d.urgency_level for d in demands) if demands else 1

        demand_analysis = {
            "total_demands": len(demands),
            "total_requested_priority": total_requested_priority,
            "max_urgency_level": max_urgency,
            "demand_distribution": {},
            "conflict_detection": [],
            "priority_competition": total_requested_priority > 1.0
        }

        # Analyze individual demands
        for demand in demands:
            demand_analysis["demand_distribution"][demand.subsystem_name] = {
                "requested_priority": demand.requested_priority,
                "current_load": demand.current_load,
                "urgency": demand.urgency_level,
                "complexity": demand.processing_complexity,
                "justification": demand.justification
            }

        # Store demand history
        self.demand_history[game_id].append({
            "timestamp": time.time(),
            "demands": [asdict(d) for d in demands],
            "analysis": demand_analysis
        })

        return demand_analysis

    async def _calculate_optimal_allocation(self,
                                          game_id: str,
                                          demand_analysis: Dict[str, Any],
                                          game_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate optimal attention allocation strategy."""
        state = self.game_states[game_id]

        # Start with current allocation as baseline
        new_allocations = state.get("current_allocation", {}).copy()

        # Adjust based on demands
        for subsystem, demand_info in demand_analysis["demand_distribution"].items():
            if subsystem in self.registered_subsystems:
                subsystem_config = self.registered_subsystems[subsystem]

                # Calculate desired priority based on demand and urgency
                base_priority = subsystem_config["base_priority"]
                max_priority = subsystem_config["max_priority"]

                urgency_multiplier = 1.0 + (demand_info["urgency"] - 1) * 0.3
                complexity_multiplier = 1.0 + demand_info["complexity"] * 0.2

                desired_priority = min(max_priority,
                                     base_priority * urgency_multiplier * complexity_multiplier)

                new_allocations[subsystem] = desired_priority

        # Normalize allocations
        total_allocation = sum(new_allocations.values())
        if total_allocation > self.config["max_total_allocation"]:
            normalization_factor = self.config["max_total_allocation"] / total_allocation
            for subsystem in new_allocations:
                new_allocations[subsystem] *= normalization_factor

        # Generate reasoning
        reasoning = f"Allocated resources to {len(new_allocations)} subsystems based on urgency and complexity"
        if demand_analysis["priority_competition"]:
            reasoning += " (priority competition detected - normalized allocations)"

        return {
            "allocations": new_allocations,
            "reasoning": reasoning,
            "total_demand": demand_analysis["total_requested_priority"],
            "duration": 10.0,  # Expected duration in seconds
            "context_hash": str(hash(str(game_context)))
        }

    async def _store_allocation(self,
                              game_id: str,
                              allocation: AttentionAllocation,
                              demands: List[SubsystemDemand]):
        """Store attention allocation in database."""
        try:
            state = self.game_states[game_id]

            for subsystem, priority in allocation.allocations.items():
                await self.db.execute_query(
                    """INSERT INTO attention_allocations
                       (allocation_id, game_id, session_id, subsystem_name, allocated_priority,
                        processing_load, requested_priority, allocation_timestamp, context_data)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (allocation.allocation_id, game_id, state["session_id"], subsystem,
                     priority, 0.0,  # Will be updated with actual load
                     next((d.requested_priority for d in demands if d.subsystem_name == subsystem), 0.5),
                     allocation.allocation_timestamp, json.dumps({"reasoning": allocation.allocation_reasoning}))
                )

        except Exception as e:
            logger.error(f"Error storing attention allocation: {e}")

    async def _store_resource_usage(self, game_id: str, usage: ResourceUsage):
        """Store resource usage data in database."""
        try:
            state = self.game_states[game_id]
            usage_id = f"usage_{game_id}_{usage.subsystem_name}_{int(time.time() * 1000)}"

            await self.db.execute_query(
                """INSERT INTO resource_usage_monitoring
                   (usage_id, game_id, session_id, subsystem_name, cpu_usage_estimate,
                    memory_usage_estimate, processing_time, queue_depth, throughput_rate,
                    error_rate, performance_impact, bottleneck_detected, monitoring_timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (usage_id, game_id, state["session_id"], usage.subsystem_name,
                 usage.cpu_usage, usage.memory_usage, usage.processing_time,
                 usage.queue_depth, usage.throughput_rate, usage.error_rate,
                 0.0, usage.bottleneck_detected, time.time())
            )

        except Exception as e:
            logger.error(f"Error storing resource usage: {e}")

    async def _update_context_awareness(self, game_id: str, game_context: Dict[str, Any]):
        """Update context awareness for more intelligent allocation."""
        state = self.game_states[game_id]

        # Analyze game context to determine game phase
        actions_taken = game_context.get("actions_taken", 0)
        current_score = game_context.get("current_score", 0)

        if actions_taken < 50:
            state["game_phase"] = "early"
        elif actions_taken < 200:
            state["game_phase"] = "middle"
        elif actions_taken < 400:
            state["game_phase"] = "late"
        else:
            state["game_phase"] = "critical"

        # Adjust context awareness based on score progression
        score_changes = game_context.get("recent_score_changes", [])
        if score_changes:
            recent_improvement = sum(1 for change in score_changes[-5:] if change > 0)
            state["context_awareness"] = min(1.0, recent_improvement / 5.0)

    async def _generate_optimization_suggestions(self,
                                               game_id: str,
                                               resource_usage_data: List[ResourceUsage],
                                               monitoring_results: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on resource monitoring."""
        suggestions = []

        # Analyze bottlenecks
        for bottleneck in monitoring_results["bottlenecks_detected"]:
            subsystem = bottleneck["subsystem"]
            issues = bottleneck["issues"]

            if "high_cpu_usage" in issues:
                suggestions.append(f"Reduce {subsystem} processing frequency or optimize algorithms")
            if "high_memory_usage" in issues:
                suggestions.append(f"Implement memory optimization for {subsystem}")
            if "processing_backlog" in issues:
                suggestions.append(f"Increase processing priority for {subsystem} or add parallel processing")

        # Suggest load balancing
        if monitoring_results["load_balancing_needed"]:
            suggestions.append("Consider redistributing processing load across subsystems")

        return suggestions

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the attention controller."""
        return {
            "total_allocations": self.metrics["allocations_made"],
            "load_balancing_events": self.metrics["load_balancing_events"],
            "bottlenecks_detected": self.metrics["bottlenecks_detected"],
            "allocation_changes": self.metrics["allocation_changes"],
            "average_effectiveness": self.metrics["average_allocation_effectiveness"],
            "active_games": len(self.game_states),
            "registered_subsystems": len(self.registered_subsystems)
        }

    # Additional helper methods would continue here...
    async def _analyze_performance_feedback(self, game_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance feedback for adjustment decisions."""
        return {
            "feedback_quality": feedback.get("quality", 0.5),
            "subsystem_performance": feedback.get("subsystem_scores", {}),
            "overall_effectiveness": feedback.get("overall_effectiveness", 0.5)
        }

    async def _calculate_priority_adjustments(self, game_id: str, feedback_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate how to adjust priorities based on feedback."""
        adjustments = {}

        for subsystem, performance in feedback_analysis.get("subsystem_performance", {}).items():
            if performance < 0.3:  # Poor performance
                adjustments[subsystem] = 0.1  # Increase priority
            elif performance > 0.8:  # Excellent performance
                adjustments[subsystem] = -0.05  # Slightly decrease priority

        return adjustments

    async def _generate_adjustment_reasoning(self, feedback_analysis: Dict[str, Any], adjustments: Dict[str, float]) -> str:
        """Generate human-readable reasoning for priority adjustments."""
        if not adjustments:
            return "No priority adjustments needed based on current performance"

        reasoning_parts = []
        for subsystem, adjustment in adjustments.items():
            if adjustment > 0:
                reasoning_parts.append(f"Increased {subsystem} priority due to suboptimal performance")
            else:
                reasoning_parts.append(f"Reduced {subsystem} priority due to good performance")

        return "; ".join(reasoning_parts)

    async def _predict_adjustment_effectiveness(self, game_id: str, adjustments: Dict[str, float]) -> float:
        """Predict how effective the adjustments will be."""
        if not adjustments:
            return 0.0

        # Simple heuristic based on historical data
        return min(0.9, sum(abs(adj) for adj in adjustments.values()) * 0.3)

    async def _store_adjustment_decision(self, game_id: str, adjustment_results: Dict[str, Any], feedback: Dict[str, Any]):
        """Store adjustment decision in database."""
        try:
            state = self.game_states[game_id]
            decision_id = f"decision_{game_id}_{int(time.time() * 1000)}"

            await self.db.execute_query(
                """INSERT INTO attention_controller_decisions
                   (decision_id, game_id, session_id, decision_type, decision_timestamp,
                    current_context, subsystem_demands, allocation_strategy, reasoning_data)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (decision_id, game_id, state["session_id"], "priority_adjustment",
                 adjustment_results["adjustment_timestamp"], json.dumps(feedback),
                 json.dumps({}), json.dumps(adjustment_results["priority_changes"]),
                 json.dumps({"reasoning": adjustment_results["reasoning"]}))
            )

        except Exception as e:
            logger.error(f"Error storing adjustment decision: {e}")

    async def _analyze_current_context(self, game_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current game context for allocation decisions."""
        return {
            "context_complexity": len(context.get("available_actions", [])) / 10.0,
            "urgency_level": 1 if context.get("actions_taken", 0) < 100 else 2,
            "score_trend": "improving" if context.get("score_change", 0) > 0 else "declining"
        }

    async def _determine_allocation_strategy(self, game_id: str, context_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal allocation strategy based on context."""
        if context_analysis.get("urgency_level", 1) >= 3:
            strategy_type = "focused"
            reasoning = "High urgency situation - focusing resources on critical subsystems"
        elif context_analysis.get("context_complexity", 0.5) > 0.7:
            strategy_type = "distributed"
            reasoning = "Complex situation - distributing resources across multiple subsystems"
        else:
            strategy_type = "balanced"
            reasoning = "Standard situation - maintaining balanced resource allocation"

        return {
            "type": strategy_type,
            "reasoning": reasoning,
            "effectiveness": 0.7,
            "recommendations": []
        }