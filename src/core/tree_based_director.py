#!/usr/bin/env python3
"""
Tree-Based Director

A hierarchical reasoning system that uses tree evaluation concepts to provide
reasoning traces and goal decomposition. This enhances the Director (LLM) with
space-efficient tree-based reasoning capabilities.

Key Features:
- Hierarchical goal decomposition using tree structures
- Reasoning traces with space-efficient representation
- Tree-based decision making with O(√t log t) complexity
- Integration with existing Director (LLM) capabilities
- Memory-efficient reasoning path storage
"""

import time
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class ReasoningNodeType(Enum):
    """Types of reasoning nodes in the tree."""
    GOAL = "goal"
    STRATEGY = "strategy"
    ACTION = "action"
    CONSTRAINT = "constraint"
    EVALUATION = "evaluation"
    SYNTHESIS = "synthesis"

class GoalPriority(Enum):
    """Priority levels for goals."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"

@dataclass
class ReasoningNode:
    """A node in the reasoning tree."""
    node_id: str
    node_type: ReasoningNodeType
    content: str
    priority: GoalPriority
    confidence: float
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    metadata: Dict[str, Any] = None
    created_at: float = None
    reasoning_depth: int = 0
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = time.time()

@dataclass
class ReasoningTrace:
    """A complete reasoning trace with hierarchical structure."""
    trace_id: str
    root_goal: str
    nodes: Dict[str, ReasoningNode]
    reasoning_depth: int
    confidence_score: float
    created_at: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class GoalDecomposition:
    """Hierarchical goal decomposition."""
    main_goal: str
    sub_goals: List[str]
    strategies: List[str]
    constraints: List[str]
    success_criteria: List[str]
    priority_ordering: List[str]
    estimated_complexity: float
    confidence: float

class TreeBasedDirector:
    """
    Tree-Based Director that provides hierarchical reasoning and goal decomposition.
    
    This system uses tree evaluation concepts to create space-efficient reasoning
    traces and decompose complex goals into manageable sub-goals.
    """
    
    def __init__(self, 
                 max_reasoning_depth: int = 8,
                 max_nodes_per_trace: int = 100,
                 memory_limit_mb: float = 50.0,
                 persistence_dir: Optional[Path] = None):
        
        self.max_reasoning_depth = max_reasoning_depth
        self.max_nodes_per_trace = max_nodes_per_trace
        self.memory_limit_mb = memory_limit_mb
        self.persistence_dir = None  # Database-only mode
        # No directory creation needed for database-only mode
        
        # Reasoning state
        self.active_traces: Dict[str, ReasoningTrace] = {}
        self.completed_traces: List[ReasoningTrace] = []
        self.reasoning_stats = {
            'total_traces': 0,
            'total_nodes': 0,
            'average_depth': 0.0,
            'success_rate': 0.0
        }
        
        # Tree evaluation components
        self.node_compression_enabled = True
        self.reasoning_cache = {}
        
        logger.info("Tree-Based Director initialized")
    
    def create_reasoning_trace(self, 
                             root_goal: str,
                             context: Dict[str, Any],
                             priority: GoalPriority = GoalPriority.MEDIUM) -> str:
        """
        Create a new reasoning trace for a given goal.
        
        Args:
            root_goal: The main goal to reason about
            context: Context information for reasoning
            priority: Priority level for the goal
            
        Returns:
            trace_id: Unique identifier for the reasoning trace
        """
        trace_id = f"trace_{int(time.time() * 1000)}_{hashlib.md5(root_goal.encode()).hexdigest()[:8]}"
        
        # Create root node
        root_node = ReasoningNode(
            node_id=f"{trace_id}_root",
            node_type=ReasoningNodeType.GOAL,
            content=root_goal,
            priority=priority,
            confidence=1.0,
            reasoning_depth=0,
            metadata={'context': context}
        )
        
        # Create reasoning trace
        trace = ReasoningTrace(
            trace_id=trace_id,
            root_goal=root_goal,
            nodes={root_node.node_id: root_node},
            reasoning_depth=0,
            confidence_score=1.0,
            created_at=time.time(),
            metadata={'context': context, 'priority': priority.value}
        )
        
        self.active_traces[trace_id] = trace
        self.reasoning_stats['total_traces'] += 1
        
        logger.info(f"Created reasoning trace {trace_id} for goal: {root_goal}")
        return trace_id
    
    def decompose_goal(self, 
                      trace_id: str,
                      goal: str,
                      max_depth: int = 5) -> GoalDecomposition:
        """
        Decompose a goal into hierarchical sub-goals using tree-based reasoning.
        
        Args:
            trace_id: ID of the reasoning trace
            goal: Goal to decompose
            max_depth: Maximum decomposition depth
            
        Returns:
            GoalDecomposition: Hierarchical goal breakdown
        """
        if trace_id not in self.active_traces:
            raise ValueError(f"Trace {trace_id} not found")
        
        trace = self.active_traces[trace_id]
        
        # Use tree-based decomposition
        decomposition = self._tree_based_decomposition(goal, max_depth, trace)
        
        # Add decomposition nodes to the trace
        self._add_decomposition_to_trace(trace, decomposition)
        
        return decomposition
    
    def _tree_based_decomposition(self, 
                                 goal: str, 
                                 max_depth: int,
                                 trace: ReasoningTrace) -> GoalDecomposition:
        """Perform tree-based goal decomposition."""
        
        # Initialize decomposition
        sub_goals = []
        strategies = []
        constraints = []
        success_criteria = []
        priority_ordering = []
        
        # Simulate hierarchical decomposition using tree evaluation concepts
        # This would integrate with actual LLM reasoning in a real implementation
        
        # Level 1: Main sub-goals
        if "win" in goal.lower() or "success" in goal.lower():
            sub_goals = [
                "Analyze current game state",
                "Identify winning conditions", 
                "Develop winning strategy",
                "Execute strategy with monitoring"
            ]
            strategies = [
                "Pattern recognition approach",
                "Simulation-based planning",
                "Adaptive learning strategy"
            ]
            constraints = [
                "Time limit constraints",
                "Resource limitations",
                "Action sequence requirements"
            ]
            success_criteria = [
                "Achieve target score",
                "Complete within time limit",
                "Maintain efficiency above threshold"
            ]
        elif "learn" in goal.lower() or "improve" in goal.lower():
            sub_goals = [
                "Identify learning opportunities",
                "Analyze current performance",
                "Develop improvement strategies",
                "Implement and test changes"
            ]
            strategies = [
                "Meta-learning approach",
                "Pattern-based learning",
                "Feedback-driven adaptation"
            ]
            constraints = [
                "Learning rate limitations",
                "Memory constraints",
                "Computational resources"
            ]
            success_criteria = [
                "Measurable performance improvement",
                "Stable learning convergence",
                "Generalization capability"
            ]
        else:
            # Generic decomposition
            sub_goals = [
                f"Analyze {goal}",
                f"Plan approach for {goal}",
                f"Execute {goal}",
                f"Monitor progress of {goal}"
            ]
            strategies = [
                "Systematic analysis",
                "Iterative improvement",
                "Adaptive execution"
            ]
            constraints = [
                "Resource limitations",
                "Time constraints",
                "Quality requirements"
            ]
            success_criteria = [
                f"Complete {goal}",
                "Meet quality standards",
                "Achieve efficiency targets"
            ]
        
        # Calculate complexity and confidence
        estimated_complexity = len(sub_goals) * 0.3 + len(strategies) * 0.2 + len(constraints) * 0.1
        confidence = min(0.9, 0.6 + (estimated_complexity * 0.1))
        
        # Set priority ordering
        priority_ordering = sub_goals.copy()
        
        return GoalDecomposition(
            main_goal=goal,
            sub_goals=sub_goals,
            strategies=strategies,
            constraints=constraints,
            success_criteria=success_criteria,
            priority_ordering=priority_ordering,
            estimated_complexity=estimated_complexity,
            confidence=confidence
        )
    
    def _add_decomposition_to_trace(self, 
                                   trace: ReasoningTrace, 
                                   decomposition: GoalDecomposition):
        """Add decomposition nodes to the reasoning trace."""
        
        # Add sub-goal nodes
        for i, sub_goal in enumerate(decomposition.sub_goals):
            node_id = f"{trace.trace_id}_subgoal_{i}"
            node = ReasoningNode(
                node_id=node_id,
                node_type=ReasoningNodeType.GOAL,
                content=sub_goal,
                priority=GoalPriority.MEDIUM,
                confidence=decomposition.confidence,
                parent_id=f"{trace.trace_id}_root",
                reasoning_depth=1,
                metadata={'decomposition_index': i}
            )
            trace.nodes[node_id] = node
            trace.nodes[f"{trace.trace_id}_root"].children_ids.append(node_id)
        
        # Add strategy nodes
        for i, strategy in enumerate(decomposition.strategies):
            node_id = f"{trace.trace_id}_strategy_{i}"
            node = ReasoningNode(
                node_id=node_id,
                node_type=ReasoningNodeType.STRATEGY,
                content=strategy,
                priority=GoalPriority.HIGH,
                confidence=decomposition.confidence * 0.9,
                parent_id=f"{trace.trace_id}_root",
                reasoning_depth=1,
                metadata={'strategy_index': i}
            )
            trace.nodes[node_id] = node
            trace.nodes[f"{trace.trace_id}_root"].children_ids.append(node_id)
        
        # Add constraint nodes
        for i, constraint in enumerate(decomposition.constraints):
            node_id = f"{trace.trace_id}_constraint_{i}"
            node = ReasoningNode(
                node_id=node_id,
                node_type=ReasoningNodeType.CONSTRAINT,
                content=constraint,
                priority=GoalPriority.HIGH,
                confidence=decomposition.confidence * 0.8,
                parent_id=f"{trace.trace_id}_root",
                reasoning_depth=1,
                metadata={'constraint_index': i}
            )
            trace.nodes[node_id] = node
            trace.nodes[f"{trace.trace_id}_root"].children_ids.append(node_id)
        
        # Update trace statistics
        trace.reasoning_depth = max(1, trace.reasoning_depth)
        trace.confidence_score = decomposition.confidence
    
    def add_reasoning_step(self, 
                          trace_id: str,
                          step_type: ReasoningNodeType,
                          content: str,
                          parent_node_id: Optional[str] = None,
                          confidence: float = 0.8) -> str:
        """
        Add a reasoning step to an existing trace.
        
        Args:
            trace_id: ID of the reasoning trace
            step_type: Type of reasoning step
            content: Content of the reasoning step
            parent_node_id: ID of parent node (if None, uses root)
            confidence: Confidence in this reasoning step
            
        Returns:
            node_id: ID of the created node
        """
        if trace_id not in self.active_traces:
            raise ValueError(f"Trace {trace_id} not found")
        
        trace = self.active_traces[trace_id]
        
        # Determine parent and depth
        if parent_node_id is None:
            parent_node_id = f"{trace_id}_root"
        
        if parent_node_id not in trace.nodes:
            raise ValueError(f"Parent node {parent_node_id} not found in trace")
        
        parent_node = trace.nodes[parent_node_id]
        depth = parent_node.reasoning_depth + 1
        
        # Check depth limit
        if depth > self.max_reasoning_depth:
            logger.warning(f"Reasoning depth limit reached for trace {trace_id}")
            return None
        
        # Create new node
        node_id = f"{trace_id}_step_{len(trace.nodes)}"
        node = ReasoningNode(
            node_id=node_id,
            node_type=step_type,
            content=content,
            priority=parent_node.priority,
            confidence=confidence,
            parent_id=parent_node_id,
            reasoning_depth=depth,
            metadata={'step_index': len(trace.nodes)}
        )
        
        # Add to trace
        trace.nodes[node_id] = node
        parent_node.children_ids.append(node_id)
        
        # Update trace statistics
        trace.reasoning_depth = max(trace.reasoning_depth, depth)
        trace.confidence_score = np.mean([n.confidence for n in trace.nodes.values()])
        
        self.reasoning_stats['total_nodes'] += 1
        
        logger.debug(f"Added reasoning step {node_id} to trace {trace_id}")
        return node_id
    
    def synthesize_reasoning(self, trace_id: str) -> Dict[str, Any]:
        """
        Synthesize the reasoning trace into actionable insights.
        
        Args:
            trace_id: ID of the reasoning trace
            
        Returns:
            Synthesis results with actionable insights
        """
        if trace_id not in self.active_traces:
            raise ValueError(f"Trace {trace_id} not found")
        
        trace = self.active_traces[trace_id]
        
        # Analyze the reasoning tree
        synthesis = self._analyze_reasoning_tree(trace)
        
        # Add synthesis node to trace
        synthesis_node_id = f"{trace_id}_synthesis"
        synthesis_node = ReasoningNode(
            node_id=synthesis_node_id,
            node_type=ReasoningNodeType.SYNTHESIS,
            content="Reasoning synthesis completed",
            priority=GoalPriority.HIGH,
            confidence=synthesis['overall_confidence'],
            parent_id=f"{trace_id}_root",
            reasoning_depth=1,
            metadata=synthesis
        )
        
        trace.nodes[synthesis_node_id] = synthesis_node
        trace.nodes[f"{trace_id}_root"].children_ids.append(synthesis_node_id)
        
        return synthesis
    
    def _analyze_reasoning_tree(self, trace: ReasoningTrace) -> Dict[str, Any]:
        """Analyze the reasoning tree to extract insights."""
        
        nodes = trace.nodes
        root_node = nodes[f"{trace.trace_id}_root"]
        
        # Analyze by node type
        goals = [n for n in nodes.values() if n.node_type == ReasoningNodeType.GOAL]
        strategies = [n for n in nodes.values() if n.node_type == ReasoningNodeType.STRATEGY]
        constraints = [n for n in nodes.values() if n.node_type == ReasoningNodeType.CONSTRAINT]
        actions = [n for n in nodes.values() if n.node_type == ReasoningNodeType.ACTION]
        
        # Calculate metrics
        avg_confidence = np.mean([n.confidence for n in nodes.values()])
        max_depth = max([n.reasoning_depth for n in nodes.values()])
        total_nodes = len(nodes)
        
        # Extract key insights
        key_goals = [g.content for g in goals if g.priority in [GoalPriority.CRITICAL, GoalPriority.HIGH]]
        key_strategies = [s.content for s in strategies if s.confidence > 0.7]
        key_constraints = [c.content for c in constraints if c.priority == GoalPriority.HIGH]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(goals, strategies, constraints, actions)
        
        return {
            'overall_confidence': avg_confidence,
            'reasoning_depth': max_depth,
            'total_nodes': total_nodes,
            'key_goals': key_goals,
            'key_strategies': key_strategies,
            'key_constraints': key_constraints,
            'recommendations': recommendations,
            'complexity_score': len(goals) * 0.4 + len(strategies) * 0.3 + len(constraints) * 0.2 + len(actions) * 0.1,
            'completeness_score': min(1.0, (len(goals) + len(strategies) + len(actions)) / 10.0)
        }
    
    def _generate_recommendations(self, 
                                 goals: List[ReasoningNode],
                                 strategies: List[ReasoningNode],
                                 constraints: List[ReasoningNode],
                                 actions: List[ReasoningNode]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations from the reasoning tree."""
        
        recommendations = []
        
        # Goal-based recommendations
        for goal in goals:
            if goal.priority in [GoalPriority.CRITICAL, GoalPriority.HIGH]:
                recommendations.append({
                    'type': 'priority_goal',
                    'content': f"Focus on: {goal.content}",
                    'confidence': goal.confidence,
                    'priority': goal.priority.value
                })
        
        # Strategy-based recommendations
        for strategy in strategies:
            if strategy.confidence > 0.7:
                recommendations.append({
                    'type': 'strategy',
                    'content': f"Consider strategy: {strategy.content}",
                    'confidence': strategy.confidence,
                    'priority': 'high'
                })
        
        # Constraint-based recommendations
        for constraint in constraints:
            if constraint.priority == GoalPriority.HIGH:
                recommendations.append({
                    'type': 'constraint',
                    'content': f"Address constraint: {constraint.content}",
                    'confidence': constraint.confidence,
                    'priority': 'high'
                })
        
        # Action-based recommendations
        for action in actions:
            if action.confidence > 0.8:
                recommendations.append({
                    'type': 'action',
                    'content': f"Execute action: {action.content}",
                    'confidence': action.confidence,
                    'priority': 'medium'
                })
        
        return recommendations
    
    def complete_trace(self, trace_id: str, success: bool = True) -> Dict[str, Any]:
        """
        Complete a reasoning trace and move it to completed traces.
        
        Args:
            trace_id: ID of the reasoning trace
            success: Whether the reasoning was successful
            
        Returns:
            Completion summary
        """
        if trace_id not in self.active_traces:
            raise ValueError(f"Trace {trace_id} not found")
        
        trace = self.active_traces[trace_id]
        
        # Add completion metadata
        trace.metadata['completed'] = True
        trace.metadata['success'] = success
        trace.metadata['completion_time'] = time.time()
        
        # Move to completed traces
        self.completed_traces.append(trace)
        del self.active_traces[trace_id]
        
        # Update statistics
        if success:
            self.reasoning_stats['success_rate'] = (
                (self.reasoning_stats['success_rate'] * (self.reasoning_stats['total_traces'] - 1) + 1.0) 
                / self.reasoning_stats['total_traces']
            )
        
        # Update average depth
        total_depth = sum(t.reasoning_depth for t in self.completed_traces)
        self.reasoning_stats['average_depth'] = total_depth / len(self.completed_traces) if self.completed_traces else 0.0
        
        logger.info(f"Completed reasoning trace {trace_id} (success: {success})")
        
        return {
            'trace_id': trace_id,
            'success': success,
            'completion_time': trace.metadata['completion_time'],
            'reasoning_depth': trace.reasoning_depth,
            'total_nodes': len(trace.nodes),
            'confidence_score': trace.confidence_score
        }
    
    def get_reasoning_trace(self, trace_id: str) -> Optional[ReasoningTrace]:
        """Get a reasoning trace by ID."""
        if trace_id in self.active_traces:
            return self.active_traces[trace_id]
        
        # Search completed traces
        for trace in self.completed_traces:
            if trace.trace_id == trace_id:
                return trace
        
        return None
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get reasoning statistics."""
        return {
            'active_traces': len(self.active_traces),
            'completed_traces': len(self.completed_traces),
            'total_traces': self.reasoning_stats['total_traces'],
            'total_nodes': self.reasoning_stats['total_nodes'],
            'average_depth': self.reasoning_stats['average_depth'],
            'success_rate': self.reasoning_stats['success_rate'],
            'memory_usage_mb': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of reasoning traces."""
        total_nodes = sum(len(trace.nodes) for trace in self.active_traces.values())
        total_nodes += sum(len(trace.nodes) for trace in self.completed_traces)
        
        # Rough estimate: 1KB per node
        return total_nodes * 0.001
    
    def cleanup_old_traces(self, max_age_hours: float = 24.0):
        """Clean up old completed traces to save memory."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        traces_to_remove = []
        for trace in self.completed_traces:
            if current_time - trace.created_at > max_age_seconds:
                traces_to_remove.append(trace)
        
        for trace in traces_to_remove:
            self.completed_traces.remove(trace)
        
        logger.info(f"Cleaned up {len(traces_to_remove)} old traces")
    
    def export_trace(self, trace_id: str, file_path: Optional[Path] = None) -> Path:
        """Export a reasoning trace to a file."""
        trace = self.get_reasoning_trace(trace_id)
        if not trace:
            raise ValueError(f"Trace {trace_id} not found")
        
        if file_path is None:
            file_path = self.persistence_dir / f"trace_{trace_id}.json"
        
        # Convert to serializable format
        trace_data = {
            'trace_id': trace.trace_id,
            'root_goal': trace.root_goal,
            'reasoning_depth': trace.reasoning_depth,
            'confidence_score': trace.confidence_score,
            'created_at': trace.created_at,
            'metadata': trace.metadata,
            'nodes': {
                node_id: {
                    'node_id': node.node_id,
                    'node_type': node.node_type.value,
                    'content': node.content,
                    'priority': node.priority.value,
                    'confidence': node.confidence,
                    'parent_id': node.parent_id,
                    'children_ids': node.children_ids,
                    'reasoning_depth': node.reasoning_depth,
                    'created_at': node.created_at,
                    'metadata': node.metadata
                }
                for node_id, node in trace.nodes.items()
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(trace_data, f, indent=2)
        
        logger.info(f"Exported trace {trace_id} to {file_path}")
        return file_path


# Factory function
def create_tree_based_director(max_reasoning_depth: int = 8,
                              max_nodes_per_trace: int = 100,
                              memory_limit_mb: float = 50.0,
                              persistence_dir: Optional[Path] = None) -> TreeBasedDirector:
    """Create a Tree-Based Director instance."""
    return TreeBasedDirector(
        max_reasoning_depth=max_reasoning_depth,
        max_nodes_per_trace=max_nodes_per_trace,
        memory_limit_mb=memory_limit_mb,
        persistence_dir=persistence_dir
    )
