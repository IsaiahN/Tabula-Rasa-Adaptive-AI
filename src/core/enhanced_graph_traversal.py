"""
Enhanced Graph Traversal System - Tier 3 System

Provides advanced graph-based navigation through complex pattern spaces,
decision trees, and game state representations with optimal path finding.

Key Features:
- Advanced traversal algorithms (A*, Dijkstra, DFS, BFS, custom algorithms)
- Dynamic graph construction and updating
- Optimal path finding through pattern spaces
- Decision tree traversal for strategic planning
- Backtracking and alternative path exploration
- Pattern space navigation and optimization
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import heapq
import math
from collections import defaultdict, deque
import sqlite3

logger = logging.getLogger(__name__)


class GraphType(Enum):
    """Types of graphs the system can manage."""
    GAME_STATE_GRAPH = "game_state_graph"  # Game states and transitions
    DECISION_TREE = "decision_tree"  # Decision trees for strategic planning
    PATTERN_SPACE = "pattern_space"  # Pattern discovery and relationships
    ACTION_SEQUENCE = "action_sequence"  # Action sequences and outcomes
    COORDINATE_SPACE = "coordinate_space"  # Spatial coordinate relationships
    STRATEGY_NETWORK = "strategy_network"  # Strategy relationships and dependencies
    LEARNING_PATHWAY = "learning_pathway"  # Learning progression paths


class TraversalAlgorithm(Enum):
    """Available traversal algorithms."""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    DIJKSTRA = "dijkstra"
    A_STAR = "a_star"
    BEST_FIRST = "best_first"
    BIDIRECTIONAL = "bidirectional"
    CUSTOM_HEURISTIC = "custom_heuristic"


class NodeType(Enum):
    """Types of nodes in the graph."""
    STATE_NODE = "state_node"  # Game state
    ACTION_NODE = "action_node"  # Action or decision
    PATTERN_NODE = "pattern_node"  # Discovered pattern
    OUTCOME_NODE = "outcome_node"  # Result or outcome
    STRATEGY_NODE = "strategy_node"  # Strategic choice point
    LEARNING_NODE = "learning_node"  # Learning milestone


@dataclass
class GraphNode:
    """Represents a node in the graph with properties and metadata."""
    node_id: str
    node_type: NodeType
    properties: Dict[str, Any]
    coordinates: Optional[Tuple[float, float]] = None  # For spatial graphs
    created_at: datetime = None
    last_visited: Optional[datetime] = None
    visit_count: int = 0
    success_rate: float = 0.0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class GraphEdge:
    """Represents an edge between nodes with weight and properties."""
    edge_id: str
    from_node: str
    to_node: str
    weight: float
    properties: Dict[str, Any]
    edge_type: str = "default"
    created_at: datetime = None
    traversal_count: int = 0
    success_rate: float = 0.0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class TraversalPath:
    """Represents a path through the graph."""
    path_id: str
    nodes: List[str]
    edges: List[str]
    total_weight: float
    algorithm_used: TraversalAlgorithm
    heuristic_score: float
    success_probability: float
    alternative_paths: List['TraversalPath']
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class TraversalResult:
    """Results from a graph traversal operation."""
    success: bool
    primary_path: Optional[TraversalPath]
    alternative_paths: List[TraversalPath]
    nodes_explored: int
    computation_time: float
    algorithm_performance: Dict[str, float]
    backtracking_points: List[str]
    optimization_suggestions: List[str]


class EnhancedGraph:
    """
    Enhanced graph structure with dynamic updates and advanced operations.
    """

    def __init__(self, graph_id: str, graph_type: GraphType):
        """Initialize the enhanced graph."""
        self.graph_id = graph_id
        self.graph_type = graph_type
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.adjacency_list: Dict[str, List[str]] = defaultdict(list)
        self.reverse_adjacency: Dict[str, List[str]] = defaultdict(list)
        self.node_index: Dict[str, Set[str]] = defaultdict(set)  # For quick lookups
        self.created_at = datetime.now()
        self.last_updated = datetime.now()

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node
        self.node_index[node.node_type.value].add(node.node_id)
        self.last_updated = datetime.now()

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        if edge.from_node not in self.nodes or edge.to_node not in self.nodes:
            raise ValueError("Both nodes must exist before adding edge")

        self.edges[edge.edge_id] = edge
        self.adjacency_list[edge.from_node].append(edge.to_node)
        self.reverse_adjacency[edge.to_node].append(edge.from_node)
        self.last_updated = datetime.now()

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all associated edges."""
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]

        # Remove from node index
        self.node_index[node.node_type.value].discard(node_id)

        # Remove all edges involving this node
        edges_to_remove = []
        for edge_id, edge in self.edges.items():
            if edge.from_node == node_id or edge.to_node == node_id:
                edges_to_remove.append(edge_id)

        for edge_id in edges_to_remove:
            self.remove_edge(edge_id)

        # Remove node
        del self.nodes[node_id]
        self.last_updated = datetime.now()

    def remove_edge(self, edge_id: str) -> None:
        """Remove an edge from the graph."""
        if edge_id not in self.edges:
            return

        edge = self.edges[edge_id]

        # Update adjacency lists
        if edge.to_node in self.adjacency_list[edge.from_node]:
            self.adjacency_list[edge.from_node].remove(edge.to_node)
        if edge.from_node in self.reverse_adjacency[edge.to_node]:
            self.reverse_adjacency[edge.to_node].remove(edge.from_node)

        del self.edges[edge_id]
        self.last_updated = datetime.now()

    def get_neighbors(self, node_id: str) -> List[str]:
        """Get all neighboring nodes."""
        return self.adjacency_list.get(node_id, [])

    def get_edge_weight(self, from_node: str, to_node: str) -> Optional[float]:
        """Get the weight of an edge between two nodes."""
        for edge in self.edges.values():
            if edge.from_node == from_node and edge.to_node == to_node:
                return edge.weight
        return None

    def update_node_statistics(self, node_id: str, visited: bool = True, success: bool = False) -> None:
        """Update node visit statistics."""
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]
        if visited:
            node.visit_count += 1
            node.last_visited = datetime.now()

        if success:
            # Update success rate using moving average
            old_rate = node.success_rate
            new_rate = (old_rate * (node.visit_count - 1) + 1.0) / node.visit_count
            node.success_rate = new_rate

    def update_edge_statistics(self, edge_id: str, traversed: bool = True, success: bool = False) -> None:
        """Update edge traversal statistics."""
        if edge_id not in self.edges:
            return

        edge = self.edges[edge_id]
        if traversed:
            edge.traversal_count += 1

        if success:
            # Update success rate using moving average
            old_rate = edge.success_rate
            new_rate = (old_rate * (edge.traversal_count - 1) + 1.0) / edge.traversal_count
            edge.success_rate = new_rate


class EnhancedGraphTraversal:
    """
    Advanced graph traversal system with multiple algorithms and optimization.

    Provides sophisticated graph navigation capabilities for pattern discovery,
    decision tree traversal, and optimal path finding.
    """

    def __init__(self, db_connection: sqlite3.Connection):
        """Initialize the Enhanced Graph Traversal system."""
        self.db_connection = db_connection
        self.graphs: Dict[str, EnhancedGraph] = {}
        self.traversal_cache: Dict[str, TraversalResult] = {}
        self.heuristic_functions: Dict[str, Callable] = {}

        # Configuration parameters
        self.max_graphs = 50
        self.max_nodes_per_graph = 1000
        self.cache_size = 100
        self.default_edge_weight = 1.0

        # Initialize database schema
        self._init_database_schema()

        # Load existing graphs (will be loaded lazily when needed)
        self._data_loaded = False

        # Register default heuristic functions
        self._register_default_heuristics()

        logger.info("Enhanced Graph Traversal system initialized")

    def _init_database_schema(self) -> None:
        """Initialize database schema for graph traversal."""
        try:
            cursor = self.db_connection.cursor()

            # Graphs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS graph_structures (
                    graph_id TEXT PRIMARY KEY,
                    graph_type TEXT NOT NULL,
                    properties TEXT, -- JSON
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    node_count INTEGER DEFAULT 0,
                    edge_count INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1
                )
            """)

            # Nodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    node_id TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    properties TEXT, -- JSON
                    coordinates_x REAL,
                    coordinates_y REAL,
                    created_at TEXT NOT NULL,
                    last_visited TEXT,
                    visit_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    FOREIGN KEY (graph_id) REFERENCES graph_structures (graph_id)
                )
            """)

            # Edges table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS graph_edges (
                    edge_id TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    from_node TEXT NOT NULL,
                    to_node TEXT NOT NULL,
                    weight REAL NOT NULL,
                    edge_type TEXT DEFAULT 'default',
                    properties TEXT, -- JSON
                    created_at TEXT NOT NULL,
                    traversal_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    FOREIGN KEY (graph_id) REFERENCES graph_structures (graph_id),
                    FOREIGN KEY (from_node) REFERENCES graph_nodes (node_id),
                    FOREIGN KEY (to_node) REFERENCES graph_nodes (node_id)
                )
            """)

            # Traversal results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS graph_traversals (
                    traversal_id TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    start_node TEXT NOT NULL,
                    end_node TEXT NOT NULL,
                    path_nodes TEXT, -- JSON array
                    path_edges TEXT, -- JSON array
                    total_weight REAL,
                    success BOOLEAN NOT NULL,
                    computation_time REAL,
                    nodes_explored INTEGER,
                    heuristic_score REAL,
                    created_at TEXT NOT NULL,
                    game_id TEXT,
                    session_id TEXT,
                    FOREIGN KEY (graph_id) REFERENCES graph_structures (graph_id)
                )
            """)

            # Path optimization table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS path_optimizations (
                    optimization_id TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    original_path TEXT, -- JSON
                    optimized_path TEXT, -- JSON
                    improvement_factor REAL,
                    optimization_strategy TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (graph_id) REFERENCES graph_structures (graph_id)
                )
            """)

            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_graphs_type ON graph_structures (graph_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_graph ON graph_nodes (graph_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON graph_nodes (node_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_graph ON graph_edges (graph_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_nodes ON graph_edges (from_node, to_node)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_traversals_graph ON graph_traversals (graph_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_traversals_algorithm ON graph_traversals (algorithm)")

            self.db_connection.commit()
            logger.info("Graph traversal database schema initialized")

        except Exception as e:
            logger.error(f"Error initializing graph traversal database schema: {e}")
            raise

    async def _load_existing_graphs(self) -> None:
        """Load existing graphs from database."""
        try:
            cursor = self.db_connection.cursor()

            # Load graphs
            cursor.execute("""
                SELECT graph_id, graph_type, properties, created_at, last_updated
                FROM graph_structures WHERE is_active = 1
                ORDER BY last_updated DESC
            """)

            for row in cursor.fetchall():
                graph_id, graph_type, properties, created_at, last_updated = row

                graph = EnhancedGraph(graph_id, GraphType(graph_type))
                graph.created_at = datetime.fromisoformat(created_at)
                graph.last_updated = datetime.fromisoformat(last_updated)

                # Load nodes for this graph
                cursor.execute("""
                    SELECT node_id, node_type, properties, coordinates_x, coordinates_y,
                           created_at, last_visited, visit_count, success_rate
                    FROM graph_nodes WHERE graph_id = ?
                """, (graph_id,))

                for node_row in cursor.fetchall():
                    node_id, node_type, props, coord_x, coord_y, created, visited, visits, success = node_row

                    coordinates = (coord_x, coord_y) if coord_x is not None and coord_y is not None else None
                    last_visited = datetime.fromisoformat(visited) if visited else None

                    node = GraphNode(
                        node_id=node_id,
                        node_type=NodeType(node_type),
                        properties=json.loads(props) if props else {},
                        coordinates=coordinates,
                        created_at=datetime.fromisoformat(created),
                        last_visited=last_visited,
                        visit_count=visits,
                        success_rate=success
                    )
                    graph.add_node(node)

                # Load edges for this graph
                cursor.execute("""
                    SELECT edge_id, from_node, to_node, weight, edge_type, properties,
                           created_at, traversal_count, success_rate
                    FROM graph_edges WHERE graph_id = ?
                """, (graph_id,))

                for edge_row in cursor.fetchall():
                    edge_id, from_node, to_node, weight, edge_type, props, created, traversals, success = edge_row

                    edge = GraphEdge(
                        edge_id=edge_id,
                        from_node=from_node,
                        to_node=to_node,
                        weight=weight,
                        properties=json.loads(props) if props else {},
                        edge_type=edge_type,
                        created_at=datetime.fromisoformat(created),
                        traversal_count=traversals,
                        success_rate=success
                    )
                    graph.add_edge(edge)

                self.graphs[graph_id] = graph

            logger.info(f"Loaded {len(self.graphs)} graphs from database")

        except Exception as e:
            logger.error(f"Error loading existing graphs: {e}")

    def _register_default_heuristics(self) -> None:
        """Register default heuristic functions for different graph types."""

        def euclidean_distance(node1: GraphNode, node2: GraphNode) -> float:
            """Euclidean distance heuristic for spatial graphs."""
            if node1.coordinates and node2.coordinates:
                return math.sqrt(
                    (node1.coordinates[0] - node2.coordinates[0])**2 +
                    (node1.coordinates[1] - node2.coordinates[1])**2
                )
            return 0.0

        def success_rate_heuristic(node1: GraphNode, node2: GraphNode) -> float:
            """Success rate based heuristic."""
            return 1.0 - node2.success_rate  # Lower values are better for A*

        def visit_count_heuristic(node1: GraphNode, node2: GraphNode) -> float:
            """Visit count based heuristic (prefer less visited nodes)."""
            return 1.0 / (1.0 + node2.visit_count)

        self.heuristic_functions['euclidean'] = euclidean_distance
        self.heuristic_functions['success_rate'] = success_rate_heuristic
        self.heuristic_functions['visit_count'] = visit_count_heuristic

    async def _ensure_data_loaded(self):
        """Ensure existing data is loaded from database."""
        if not self._data_loaded:
            await self._load_existing_graphs()
            self._data_loaded = True

    async def create_graph(self,
                          graph_type: GraphType,
                          initial_nodes: List[GraphNode] = None,
                          initial_edges: List[GraphEdge] = None,
                          game_id: str = None) -> str:
        """Create a new graph structure."""
        try:
            await self._ensure_data_loaded()
            import uuid
            graph_id = f"{graph_type.value}_{game_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            graph = EnhancedGraph(graph_id, graph_type)

            # Add initial nodes and edges if provided
            if initial_nodes:
                for node in initial_nodes:
                    graph.add_node(node)

            if initial_edges:
                for edge in initial_edges:
                    graph.add_edge(edge)

            self.graphs[graph_id] = graph

            # Store in database
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO graph_structures (
                    graph_id, graph_type, properties, created_at, last_updated,
                    node_count, edge_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                graph_id, graph_type.value, json.dumps({}),
                graph.created_at.isoformat(), graph.last_updated.isoformat(),
                len(graph.nodes), len(graph.edges)
            ))

            # Store nodes and edges
            for node in graph.nodes.values():
                await self._store_node(node, graph_id)

            for edge in graph.edges.values():
                await self._store_edge(edge, graph_id)

            self.db_connection.commit()

            logger.info(f"Created graph {graph_id} with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            return graph_id

        except Exception as e:
            logger.error(f"Error creating graph: {e}")
            raise

    async def _store_node(self, node: GraphNode, graph_id: str) -> None:
        """Store a node in the database."""
        cursor = self.db_connection.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO graph_nodes (
                node_id, graph_id, node_type, properties, coordinates_x, coordinates_y,
                created_at, last_visited, visit_count, success_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            node.node_id, graph_id, node.node_type.value, json.dumps(node.properties),
            node.coordinates[0] if node.coordinates else None,
            node.coordinates[1] if node.coordinates else None,
            node.created_at.isoformat(),
            node.last_visited.isoformat() if node.last_visited else None,
            node.visit_count, node.success_rate
        ))

    async def _store_edge(self, edge: GraphEdge, graph_id: str) -> None:
        """Store an edge in the database."""
        cursor = self.db_connection.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO graph_edges (
                edge_id, graph_id, from_node, to_node, weight, edge_type,
                properties, created_at, traversal_count, success_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            edge.edge_id, graph_id, edge.from_node, edge.to_node, edge.weight,
            edge.edge_type, json.dumps(edge.properties), edge.created_at.isoformat(),
            edge.traversal_count, edge.success_rate
        ))

    async def traverse_graph(self,
                           graph_id: str,
                           start_node: str,
                           end_node: str,
                           algorithm: TraversalAlgorithm,
                           heuristic: str = None,
                           max_iterations: int = 1000) -> TraversalResult:
        """Traverse a graph using the specified algorithm."""
        try:
            if graph_id not in self.graphs:
                raise ValueError(f"Graph {graph_id} not found")

            graph = self.graphs[graph_id]

            if start_node not in graph.nodes or end_node not in graph.nodes:
                raise ValueError("Start or end node not found in graph")

            start_time = datetime.now()

            # Select traversal algorithm
            if algorithm == TraversalAlgorithm.BREADTH_FIRST:
                result = await self._breadth_first_search(graph, start_node, end_node)
            elif algorithm == TraversalAlgorithm.DEPTH_FIRST:
                result = await self._depth_first_search(graph, start_node, end_node, max_iterations)
            elif algorithm == TraversalAlgorithm.DIJKSTRA:
                result = await self._dijkstra_search(graph, start_node, end_node)
            elif algorithm == TraversalAlgorithm.A_STAR:
                heuristic_func = self.heuristic_functions.get(heuristic, self.heuristic_functions['euclidean'])
                result = await self._a_star_search(graph, start_node, end_node, heuristic_func)
            elif algorithm == TraversalAlgorithm.BEST_FIRST:
                heuristic_func = self.heuristic_functions.get(heuristic, self.heuristic_functions['success_rate'])
                result = await self._best_first_search(graph, start_node, end_node, heuristic_func)
            elif algorithm == TraversalAlgorithm.BIDIRECTIONAL:
                result = await self._bidirectional_search(graph, start_node, end_node)
            else:
                raise ValueError(f"Algorithm {algorithm} not implemented")

            computation_time = (datetime.now() - start_time).total_seconds()

            # Update result with timing information
            result.computation_time = computation_time
            result.algorithm_performance = {
                'algorithm': algorithm.value,
                'computation_time': computation_time,
                'nodes_per_second': result.nodes_explored / max(computation_time, 0.001)
            }

            # Store traversal result in database
            await self._store_traversal_result(graph_id, algorithm, start_node, end_node, result)

            return result

        except Exception as e:
            logger.error(f"Error traversing graph: {e}")
            raise

    async def _breadth_first_search(self, graph: EnhancedGraph, start: str, end: str) -> TraversalResult:
        """Breadth-first search implementation."""
        queue = deque([(start, [start], [])])
        visited = {start}
        nodes_explored = 0

        while queue:
            current, path, edges = queue.popleft()
            nodes_explored += 1

            if current == end:
                # Found path
                path_obj = TraversalPath(
                    path_id=f"bfs_path_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    nodes=path,
                    edges=edges,
                    total_weight=len(path) - 1,  # Simple path length
                    algorithm_used=TraversalAlgorithm.BREADTH_FIRST,
                    heuristic_score=0.0,
                    success_probability=1.0,
                    alternative_paths=[],
                    created_at=datetime.now(),
                    metadata={'search_type': 'breadth_first'}
                )

                return TraversalResult(
                    success=True,
                    primary_path=path_obj,
                    alternative_paths=[],
                    nodes_explored=nodes_explored,
                    computation_time=0.0,  # Will be set by caller
                    algorithm_performance={},
                    backtracking_points=[],
                    optimization_suggestions=[]
                )

            # Explore neighbors
            for neighbor in graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)

                    # Find edge ID
                    edge_id = None
                    for eid, edge in graph.edges.items():
                        if edge.from_node == current and edge.to_node == neighbor:
                            edge_id = eid
                            break

                    new_path = path + [neighbor]
                    new_edges = edges + ([edge_id] if edge_id else [])
                    queue.append((neighbor, new_path, new_edges))

        # No path found
        return TraversalResult(
            success=False,
            primary_path=None,
            alternative_paths=[],
            nodes_explored=nodes_explored,
            computation_time=0.0,
            algorithm_performance={},
            backtracking_points=[],
            optimization_suggestions=['Try a different algorithm', 'Check graph connectivity']
        )

    async def _depth_first_search(self, graph: EnhancedGraph, start: str, end: str, max_iterations: int) -> TraversalResult:
        """Depth-first search implementation with backtracking."""
        stack = [(start, [start], [], set([start]))]
        nodes_explored = 0
        backtracking_points = []

        while stack and nodes_explored < max_iterations:
            current, path, edges, visited_in_path = stack.pop()
            nodes_explored += 1

            if current == end:
                # Found path
                path_obj = TraversalPath(
                    path_id=f"dfs_path_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    nodes=path,
                    edges=edges,
                    total_weight=len(path) - 1,
                    algorithm_used=TraversalAlgorithm.DEPTH_FIRST,
                    heuristic_score=0.0,
                    success_probability=1.0,
                    alternative_paths=[],
                    created_at=datetime.now(),
                    metadata={'search_type': 'depth_first', 'backtracking_points': len(backtracking_points)}
                )

                return TraversalResult(
                    success=True,
                    primary_path=path_obj,
                    alternative_paths=[],
                    nodes_explored=nodes_explored,
                    computation_time=0.0,
                    algorithm_performance={},
                    backtracking_points=backtracking_points,
                    optimization_suggestions=[]
                )

            # Check if we need to backtrack
            neighbors = graph.get_neighbors(current)
            unvisited_neighbors = [n for n in neighbors if n not in visited_in_path]

            if not unvisited_neighbors:
                backtracking_points.append(current)

            # Explore neighbors in reverse order (DFS characteristic)
            for neighbor in reversed(unvisited_neighbors):
                # Find edge ID
                edge_id = None
                for eid, edge in graph.edges.items():
                    if edge.from_node == current and edge.to_node == neighbor:
                        edge_id = eid
                        break

                new_path = path + [neighbor]
                new_edges = edges + ([edge_id] if edge_id else [])
                new_visited = visited_in_path | {neighbor}
                stack.append((neighbor, new_path, new_edges, new_visited))

        # No path found or max iterations reached
        suggestions = ['Increase max_iterations', 'Try breadth-first search', 'Check for cycles']
        if nodes_explored >= max_iterations:
            suggestions.append('Search space too large for DFS')

        return TraversalResult(
            success=False,
            primary_path=None,
            alternative_paths=[],
            nodes_explored=nodes_explored,
            computation_time=0.0,
            algorithm_performance={},
            backtracking_points=backtracking_points,
            optimization_suggestions=suggestions
        )

    async def _dijkstra_search(self, graph: EnhancedGraph, start: str, end: str) -> TraversalResult:
        """Dijkstra's shortest path algorithm."""
        distances = defaultdict(lambda: float('inf'))
        distances[start] = 0
        previous = {}
        priority_queue = [(0, start)]
        visited = set()
        nodes_explored = 0

        while priority_queue:
            current_distance, current = heapq.heappop(priority_queue)

            if current in visited:
                continue

            visited.add(current)
            nodes_explored += 1

            if current == end:
                # Reconstruct path
                path = []
                edges = []
                node = end

                while node is not None:
                    path.append(node)
                    if node in previous:
                        prev_node = previous[node]
                        # Find edge ID
                        for eid, edge in graph.edges.items():
                            if edge.from_node == prev_node and edge.to_node == node:
                                edges.append(eid)
                                break
                        node = prev_node
                    else:
                        node = None

                path.reverse()
                edges.reverse()

                path_obj = TraversalPath(
                    path_id=f"dijkstra_path_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    nodes=path,
                    edges=edges,
                    total_weight=current_distance,
                    algorithm_used=TraversalAlgorithm.DIJKSTRA,
                    heuristic_score=0.0,
                    success_probability=1.0,
                    alternative_paths=[],
                    created_at=datetime.now(),
                    metadata={'search_type': 'dijkstra', 'optimal_weight': current_distance}
                )

                return TraversalResult(
                    success=True,
                    primary_path=path_obj,
                    alternative_paths=[],
                    nodes_explored=nodes_explored,
                    computation_time=0.0,
                    algorithm_performance={},
                    backtracking_points=[],
                    optimization_suggestions=[]
                )

            # Explore neighbors
            for neighbor in graph.get_neighbors(current):
                if neighbor in visited:
                    continue

                weight = graph.get_edge_weight(current, neighbor) or self.default_edge_weight
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(priority_queue, (distance, neighbor))

        # No path found
        return TraversalResult(
            success=False,
            primary_path=None,
            alternative_paths=[],
            nodes_explored=nodes_explored,
            computation_time=0.0,
            algorithm_performance={},
            backtracking_points=[],
            optimization_suggestions=['Check graph connectivity', 'Verify edge weights']
        )

    async def _a_star_search(self, graph: EnhancedGraph, start: str, end: str, heuristic_func: Callable) -> TraversalResult:
        """A* search algorithm with heuristic."""
        open_set = [(0, start)]
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start] = 0
        f_score = defaultdict(lambda: float('inf'))

        end_node = graph.nodes[end]
        start_node = graph.nodes[start]
        f_score[start] = heuristic_func(start_node, end_node)

        visited = set()
        nodes_explored = 0

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current in visited:
                continue

            visited.add(current)
            nodes_explored += 1

            if current == end:
                # Reconstruct path
                path = []
                edges = []
                node = end

                while node in came_from:
                    path.append(node)
                    prev_node = came_from[node]
                    # Find edge ID
                    for eid, edge in graph.edges.items():
                        if edge.from_node == prev_node and edge.to_node == node:
                            edges.append(eid)
                            break
                    node = prev_node

                path.append(start)
                path.reverse()
                edges.reverse()

                path_obj = TraversalPath(
                    path_id=f"astar_path_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    nodes=path,
                    edges=edges,
                    total_weight=g_score[end],
                    algorithm_used=TraversalAlgorithm.A_STAR,
                    heuristic_score=f_score[end],
                    success_probability=1.0,
                    alternative_paths=[],
                    created_at=datetime.now(),
                    metadata={'search_type': 'a_star', 'heuristic_used': True}
                )

                return TraversalResult(
                    success=True,
                    primary_path=path_obj,
                    alternative_paths=[],
                    nodes_explored=nodes_explored,
                    computation_time=0.0,
                    algorithm_performance={},
                    backtracking_points=[],
                    optimization_suggestions=[]
                )

            # Explore neighbors
            for neighbor in graph.get_neighbors(current):
                if neighbor in visited:
                    continue

                weight = graph.get_edge_weight(current, neighbor) or self.default_edge_weight
                tentative_g_score = g_score[current] + weight

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    neighbor_node = graph.nodes[neighbor]
                    f_score[neighbor] = tentative_g_score + heuristic_func(neighbor_node, end_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # No path found
        return TraversalResult(
            success=False,
            primary_path=None,
            alternative_paths=[],
            nodes_explored=nodes_explored,
            computation_time=0.0,
            algorithm_performance={},
            backtracking_points=[],
            optimization_suggestions=['Check heuristic function', 'Verify graph connectivity']
        )

    async def _best_first_search(self, graph: EnhancedGraph, start: str, end: str, heuristic_func: Callable) -> TraversalResult:
        """Best-first search using only heuristic."""
        end_node = graph.nodes[end]
        open_set = [(heuristic_func(graph.nodes[start], end_node), start)]
        visited = set()
        came_from = {}
        nodes_explored = 0

        while open_set:
            current_h, current = heapq.heappop(open_set)

            if current in visited:
                continue

            visited.add(current)
            nodes_explored += 1

            if current == end:
                # Reconstruct path
                path = []
                edges = []
                node = end

                while node in came_from:
                    path.append(node)
                    prev_node = came_from[node]
                    # Find edge ID
                    for eid, edge in graph.edges.items():
                        if edge.from_node == prev_node and edge.to_node == node:
                            edges.append(eid)
                            break
                    node = prev_node

                path.append(start)
                path.reverse()
                edges.reverse()

                path_obj = TraversalPath(
                    path_id=f"bestfirst_path_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    nodes=path,
                    edges=edges,
                    total_weight=len(path) - 1,
                    algorithm_used=TraversalAlgorithm.BEST_FIRST,
                    heuristic_score=current_h,
                    success_probability=1.0,
                    alternative_paths=[],
                    created_at=datetime.now(),
                    metadata={'search_type': 'best_first', 'final_heuristic': current_h}
                )

                return TraversalResult(
                    success=True,
                    primary_path=path_obj,
                    alternative_paths=[],
                    nodes_explored=nodes_explored,
                    computation_time=0.0,
                    algorithm_performance={},
                    backtracking_points=[],
                    optimization_suggestions=[]
                )

            # Explore neighbors
            for neighbor in graph.get_neighbors(current):
                if neighbor not in visited:
                    came_from[neighbor] = current
                    neighbor_node = graph.nodes[neighbor]
                    heuristic_value = heuristic_func(neighbor_node, end_node)
                    heapq.heappush(open_set, (heuristic_value, neighbor))

        # No path found
        return TraversalResult(
            success=False,
            primary_path=None,
            alternative_paths=[],
            nodes_explored=nodes_explored,
            computation_time=0.0,
            algorithm_performance={},
            backtracking_points=[],
            optimization_suggestions=['Try A* instead of best-first', 'Check heuristic function']
        )

    async def _bidirectional_search(self, graph: EnhancedGraph, start: str, end: str) -> TraversalResult:
        """Bidirectional search from both start and end."""
        # Forward search from start
        forward_queue = deque([start])
        forward_visited = {start: None}

        # Backward search from end
        backward_queue = deque([end])
        backward_visited = {end: None}

        nodes_explored = 0

        while forward_queue or backward_queue:
            # Forward step
            if forward_queue:
                current = forward_queue.popleft()
                nodes_explored += 1

                if current in backward_visited:
                    # Found intersection, reconstruct path
                    path = self._reconstruct_bidirectional_path(
                        start, end, current, forward_visited, backward_visited, graph
                    )

                    path_obj = TraversalPath(
                        path_id=f"bidirectional_path_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        nodes=path['nodes'],
                        edges=path['edges'],
                        total_weight=len(path['nodes']) - 1,
                        algorithm_used=TraversalAlgorithm.BIDIRECTIONAL,
                        heuristic_score=0.0,
                        success_probability=1.0,
                        alternative_paths=[],
                        created_at=datetime.now(),
                        metadata={'search_type': 'bidirectional', 'meeting_point': current}
                    )

                    return TraversalResult(
                        success=True,
                        primary_path=path_obj,
                        alternative_paths=[],
                        nodes_explored=nodes_explored,
                        computation_time=0.0,
                        algorithm_performance={},
                        backtracking_points=[],
                        optimization_suggestions=[]
                    )

                # Explore forward neighbors
                for neighbor in graph.get_neighbors(current):
                    if neighbor not in forward_visited:
                        forward_visited[neighbor] = current
                        forward_queue.append(neighbor)

            # Backward step
            if backward_queue:
                current = backward_queue.popleft()
                nodes_explored += 1

                if current in forward_visited:
                    # Found intersection, reconstruct path
                    path = self._reconstruct_bidirectional_path(
                        start, end, current, forward_visited, backward_visited, graph
                    )

                    path_obj = TraversalPath(
                        path_id=f"bidirectional_path_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        nodes=path['nodes'],
                        edges=path['edges'],
                        total_weight=len(path['nodes']) - 1,
                        algorithm_used=TraversalAlgorithm.BIDIRECTIONAL,
                        heuristic_score=0.0,
                        success_probability=1.0,
                        alternative_paths=[],
                        created_at=datetime.now(),
                        metadata={'search_type': 'bidirectional', 'meeting_point': current}
                    )

                    return TraversalResult(
                        success=True,
                        primary_path=path_obj,
                        alternative_paths=[],
                        nodes_explored=nodes_explored,
                        computation_time=0.0,
                        algorithm_performance={},
                        backtracking_points=[],
                        optimization_suggestions=[]
                    )

                # Explore backward neighbors (using reverse adjacency)
                for neighbor in graph.reverse_adjacency.get(current, []):
                    if neighbor not in backward_visited:
                        backward_visited[neighbor] = current
                        backward_queue.append(neighbor)

        # No path found
        return TraversalResult(
            success=False,
            primary_path=None,
            alternative_paths=[],
            nodes_explored=nodes_explored,
            computation_time=0.0,
            algorithm_performance={},
            backtracking_points=[],
            optimization_suggestions=['Check graph connectivity', 'Try unidirectional search']
        )

    def _reconstruct_bidirectional_path(self, start: str, end: str, meeting_point: str,
                                      forward_visited: Dict, backward_visited: Dict,
                                      graph: EnhancedGraph) -> Dict[str, List[str]]:
        """Reconstruct path from bidirectional search."""
        # Forward path from start to meeting point
        forward_path = []
        current = meeting_point
        while current is not None:
            forward_path.append(current)
            current = forward_visited[current]
        forward_path.reverse()

        # Backward path from meeting point to end
        backward_path = []
        current = backward_visited[meeting_point]
        while current is not None:
            backward_path.append(current)
            current = backward_visited[current]

        # Combine paths
        full_path = forward_path + backward_path

        # Find edges
        edges = []
        for i in range(len(full_path) - 1):
            from_node = full_path[i]
            to_node = full_path[i + 1]

            # Find edge ID
            for eid, edge in graph.edges.items():
                if edge.from_node == from_node and edge.to_node == to_node:
                    edges.append(eid)
                    break

        return {'nodes': full_path, 'edges': edges}

    async def _store_traversal_result(self, graph_id: str, algorithm: TraversalAlgorithm,
                                    start_node: str, end_node: str, result: TraversalResult) -> None:
        """Store traversal result in database."""
        try:
            cursor = self.db_connection.cursor()

            traversal_id = f"traversal_{graph_id}_{algorithm.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            path_nodes = result.primary_path.nodes if result.primary_path else []
            path_edges = result.primary_path.edges if result.primary_path else []
            total_weight = result.primary_path.total_weight if result.primary_path else 0.0
            heuristic_score = result.primary_path.heuristic_score if result.primary_path else 0.0

            cursor.execute("""
                INSERT INTO graph_traversals (
                    traversal_id, graph_id, algorithm, start_node, end_node,
                    path_nodes, path_edges, total_weight, success, computation_time,
                    nodes_explored, heuristic_score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                traversal_id, graph_id, algorithm.value, start_node, end_node,
                json.dumps(path_nodes), json.dumps(path_edges), total_weight,
                result.success, result.computation_time, result.nodes_explored,
                heuristic_score, datetime.now().isoformat()
            ))

            self.db_connection.commit()

        except Exception as e:
            logger.error(f"Error storing traversal result: {e}")

    async def find_optimal_paths(self, graph_id: str, start_node: str, end_node: str,
                               max_alternatives: int = 3) -> List[TraversalPath]:
        """Find multiple optimal paths using different algorithms."""
        try:
            if graph_id not in self.graphs:
                return []

            algorithms_to_try = [
                TraversalAlgorithm.DIJKSTRA,
                TraversalAlgorithm.A_STAR,
                TraversalAlgorithm.BEST_FIRST,
                TraversalAlgorithm.BIDIRECTIONAL
            ]

            paths = []

            for algorithm in algorithms_to_try:
                try:
                    result = await self.traverse_graph(graph_id, start_node, end_node, algorithm)
                    if result.success and result.primary_path:
                        paths.append(result.primary_path)

                        if len(paths) >= max_alternatives:
                            break

                except Exception as e:
                    logger.warning(f"Algorithm {algorithm} failed: {e}")
                    continue

            # Sort paths by weight (ascending)
            paths.sort(key=lambda p: p.total_weight)

            return paths[:max_alternatives]

        except Exception as e:
            logger.error(f"Error finding optimal paths: {e}")
            return []

    def set_attention_coordination(self, attention_controller, communication_system) -> None:
        """Set attention controller and communication system for coordination."""
        self.attention_controller = attention_controller
        self.communication_system = communication_system
        logger.info("Enhanced Graph Traversal linked with attention coordination")

    def set_fitness_evolution_coordination(self, fitness_evolution_system) -> None:
        """Set fitness evolution system for coordination."""
        self.fitness_evolution_system = fitness_evolution_system
        logger.info("Enhanced Graph Traversal linked with fitness evolution")

    async def request_attention_allocation(self,
                                         game_id: str,
                                         session_id: str,
                                         traversal_complexity: float) -> Optional[Dict[str, Any]]:
        """Request attention allocation for complex graph traversal operations."""
        try:
            if not hasattr(self, 'attention_controller') or not self.attention_controller:
                return None

            from src.core.central_attention_controller import SubsystemDemand

            traversal_demand = SubsystemDemand(
                subsystem_name="graph_traversal",
                requested_priority=min(0.7, 0.2 + traversal_complexity),
                current_load=traversal_complexity,
                processing_complexity=traversal_complexity,
                urgency_level=3 if traversal_complexity > 0.6 else 2,
                justification="Complex graph traversal and path optimization required",
                context_data={
                    "active_graphs": len(self.graphs),
                    "traversal_complexity": traversal_complexity,
                    "session_id": session_id
                }
            )

            allocation = await self.attention_controller.allocate_attention_resources(
                game_id, [traversal_demand], {"graph_traversal": True}
            )

            return allocation

        except Exception as e:
            logger.error(f"Error requesting attention allocation: {e}")
            return None