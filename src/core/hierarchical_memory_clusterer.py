"""
Hierarchical Memory Clusterer for Meta-Cognitive Intelligence

This module implements advanced memory clustering that goes beyond the static 4-tier
system to create intelligent, dynamic clusters based on causal relationships, temporal
patterns, semantic similarities, performance impact, and cross-session patterns.

Phase 2 Implementation: Enhances Governor decision-making with cluster intelligence.
"""

import numpy as np
import json
import time
import logging
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Set
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryClusterType(Enum):
    """Types of intelligent memory clusters"""
    CAUSAL_CHAIN = "causal_chain"              # Cause-effect relationships
    TEMPORAL_SEQUENCE = "temporal_sequence"     # Time-based sequences
    SEMANTIC_GROUP = "semantic_group"           # Conceptually related
    PERFORMANCE_CLUSTER = "performance_cluster" # Performance-critical
    CROSS_SESSION = "cross_session"            # Cross-session patterns

@dataclass
class MemoryCluster:
    """Represents an intelligent memory cluster"""
    cluster_id: str
    cluster_type: MemoryClusterType
    members: Set[str]
    relationships: Dict[str, float]  # member -> relationship strength
    access_frequency: float
    retention_priority: float
    last_accessed: float
    performance_impact: float
    creation_time: float
    evolution_history: List[Dict] = None
    
    def __post_init__(self):
        if self.evolution_history is None:
            self.evolution_history = []
    
    def add_member(self, memory_id: str, relationship_strength: float = 0.5):
        """Add a new member to the cluster"""
        self.members.add(memory_id)
        self.relationships[memory_id] = relationship_strength
        
        # Update cluster metrics
        self._update_cluster_metrics()
    
    def remove_member(self, memory_id: str):
        """Remove a member from the cluster"""
        self.members.discard(memory_id)
        self.relationships.pop(memory_id, None)
        
        # Update cluster metrics
        self._update_cluster_metrics()
    
    def _update_cluster_metrics(self):
        """Update cluster-level metrics based on members"""
        if not self.members:
            return
        
        # Calculate average relationship strength
        avg_strength = np.mean(list(self.relationships.values()))
        
        # Update retention priority based on cluster type and relationships
        if self.cluster_type == MemoryClusterType.CAUSAL_CHAIN:
            self.retention_priority = min(0.95, avg_strength * 1.2)
        elif self.cluster_type == MemoryClusterType.PERFORMANCE_CLUSTER:
            self.retention_priority = min(0.98, avg_strength * 1.3)
        elif self.cluster_type == MemoryClusterType.CROSS_SESSION:
            self.retention_priority = min(0.99, avg_strength * 1.4)
        else:
            self.retention_priority = min(0.8, avg_strength)
    
    def get_cluster_health(self) -> Dict[str, Any]:
        """Get cluster health metrics"""
        age_hours = (time.time() - self.creation_time) / 3600
        last_access_hours = (time.time() - self.last_accessed) / 3600
        
        health_score = 1.0
        
        # Age penalty (but not for cross-session clusters)
        if self.cluster_type != MemoryClusterType.CROSS_SESSION:
            if age_hours > 168:  # 1 week
                health_score *= 0.8
        
        # Access recency bonus
        if last_access_hours < 1:  # Recently accessed
            health_score *= 1.2
        elif last_access_hours > 72:  # Not accessed in 3 days
            health_score *= 0.7
        
        # Member count consideration
        optimal_size = {
            MemoryClusterType.CAUSAL_CHAIN: 5,
            MemoryClusterType.TEMPORAL_SEQUENCE: 8,
            MemoryClusterType.SEMANTIC_GROUP: 10,
            MemoryClusterType.PERFORMANCE_CLUSTER: 3,
            MemoryClusterType.CROSS_SESSION: 15
        }
        
        size_ratio = len(self.members) / optimal_size.get(self.cluster_type, 5)
        if 0.5 <= size_ratio <= 2.0:  # Good size range
            health_score *= 1.1
        elif size_ratio > 3.0:  # Too large
            health_score *= 0.8
        
        return {
            "health_score": min(health_score, 1.0),
            "cluster_size": len(self.members),
            "age_hours": age_hours,
            "last_access_hours": last_access_hours,
            "retention_priority": self.retention_priority,
            "relationship_strength": np.mean(list(self.relationships.values())) if self.relationships else 0.0
        }

class HierarchicalMemoryClusterer:
    """
    Advanced memory clustering with hierarchical organization
    
    This class creates intelligent memory clusters that go beyond the static 4-tier
    system, enabling dynamic cluster-based optimization that adapts to actual usage
    patterns and relationships.
    """
    
    def __init__(self, max_clusters: int = 100):
        self.max_clusters = max_clusters
        self.clusters = {}  # cluster_id -> MemoryCluster
        self.cluster_graph = nx.DiGraph()  # Relationships between clusters
        self.memory_to_clusters = defaultdict(set)  # memory_id -> set of cluster_ids
        
        # Pattern analysis integration
        self.access_patterns = defaultdict(list)
        self.performance_metrics = {}
        self.cluster_evolution_log = []
        
        # Clustering parameters (will be evolved by Architect in Phase 3)
        self.clustering_params = {
            'causal_threshold': 0.7,
            'temporal_window_hours': 24,
            'semantic_similarity_threshold': 0.6,
            'performance_impact_threshold': 0.8,
            'cross_session_persistence_threshold': 0.5
        }
        
        logger.info(" HierarchicalMemoryClusterer initialized for Phase 2")
    
    def create_intelligent_clusters(self, memories: List[Dict], 
                                   access_patterns: List[Dict] = None) -> Dict[str, MemoryCluster]:
        """
        Create intelligent memory clusters based on multiple dimensions
        
        This replaces static 4-tier classification with dynamic, relationship-based clustering.
        """
        logger.info(f"Creating intelligent clusters for {len(memories)} memories")
        
        # Clear existing clusters for fresh analysis
        new_clusters = {}
        
        # 1. Causal relationship clustering - Critical for Governor decisions
        causal_clusters = self._cluster_by_causality(memories, access_patterns or [])
        new_clusters.update(causal_clusters)
        logger.info(f"Created {len(causal_clusters)} causal chain clusters")
        
        # 2. Temporal sequence clustering - Important for optimization  
        temporal_clusters = self._cluster_by_temporal_patterns(memories, access_patterns or [])
        new_clusters.update(temporal_clusters)
        logger.info(f"Created {len(temporal_clusters)} temporal sequence clusters")
        
        # 3. Semantic similarity clustering - Enhances coherence
        semantic_clusters = self._cluster_by_semantic_similarity(memories)
        new_clusters.update(semantic_clusters)
        logger.info(f"Created {len(semantic_clusters)} semantic group clusters")
        
        # 4. Performance impact clustering - Critical for Governor efficiency
        performance_clusters = self._cluster_by_performance_impact(memories, access_patterns or [])
        new_clusters.update(performance_clusters)
        logger.info(f"Created {len(performance_clusters)} performance clusters")
        
        # 5. Cross-session pattern clustering - Architect evolution foundation
        cross_session_clusters = self._cluster_by_cross_session_patterns(memories)
        new_clusters.update(cross_session_clusters)
        logger.info(f"Created {len(cross_session_clusters)} cross-session clusters")
        
        # Update cluster relationships and hierarchy
        self._build_cluster_hierarchy(new_clusters)
        
        # Replace current clusters with new intelligent clusters
        self.clusters = new_clusters
        self._rebuild_memory_to_cluster_mapping()
        
        logger.info(f" Intelligent clustering complete: {len(new_clusters)} total clusters created")
        
        return new_clusters
    
    def _cluster_by_causality(self, memories: List[Dict], access_patterns: List[Dict]) -> Dict[str, MemoryCluster]:
        """Detect causal relationships - Governor decisions leading to actions"""
        clusters = {}
        
        # Build causal graph from access patterns
        causal_graph = nx.DiGraph()
        
        # Analyze access sequences for causal relationships
        for i in range(len(access_patterns) - 1):
            current = access_patterns[i]
            next_access = access_patterns[i + 1]
            
            # Time-based causality (actions within reasonable time window)
            time_diff = next_access.get('timestamp', 0) - current.get('timestamp', 0)
            
            if 0.1 <= time_diff <= 10.0:  # 100ms to 10 seconds - reasonable causal window
                current_file = current.get('file_path', '')
                next_file = next_access.get('file_path', '')
                
                if current_file and next_file and current_file != next_file:
                    # Add causal edge
                    if causal_graph.has_edge(current_file, next_file):
                        causal_graph[current_file][next_file]['weight'] += 1
                    else:
                        causal_graph.add_edge(current_file, next_file, weight=1)
        
        # Identify strongly connected components as causal chains
        try:
            components = list(nx.weakly_connected_components(causal_graph))
            
            for i, component in enumerate(components):
                if len(component) >= 2:  # Need at least 2 files for a causal chain
                    cluster_id = f"causal_chain_{i}"
                    
                    # Calculate relationship strengths based on edge weights
                    relationships = {}
                    total_weight = 0
                    
                    for node in component:
                        # Get total weight of edges involving this node
                        in_weight = sum(causal_graph[pred][node].get('weight', 0) 
                                      for pred in causal_graph.predecessors(node))
                        out_weight = sum(causal_graph[node][succ].get('weight', 0) 
                                       for succ in causal_graph.successors(node))
                        
                        node_weight = in_weight + out_weight
                        relationships[node] = min(node_weight / 10.0, 1.0)  # Normalize
                        total_weight += node_weight
                    
                    # Create causal chain cluster
                    cluster = MemoryCluster(
                        cluster_id=cluster_id,
                        cluster_type=MemoryClusterType.CAUSAL_CHAIN,
                        members=component,
                        relationships=relationships,
                        access_frequency=total_weight / len(component) if component else 0,
                        retention_priority=0.9,  # High priority for causal chains
                        last_accessed=time.time(),
                        performance_impact=0.8,
                        creation_time=time.time()
                    )
                    
                    clusters[cluster_id] = cluster
                    
                    logger.debug(f"Causal chain cluster {cluster_id}: {len(component)} members")
        
        except Exception as e:
            logger.warning(f"Causal clustering failed: {e}")
        
        return clusters
    
    def _cluster_by_temporal_patterns(self, memories: List[Dict], access_patterns: List[Dict]) -> Dict[str, MemoryCluster]:
        """Cluster memories based on temporal access patterns"""
        clusters = {}
        
        # Group accesses by time windows
        time_windows = defaultdict(list)
        current_time = time.time()
        
        for pattern in access_patterns:
            timestamp = pattern.get('timestamp', current_time)
            # Group into hour buckets
            hour_bucket = int(timestamp // 3600)
            time_windows[hour_bucket].append(pattern)
        
        # Find windows with high activity (temporal clusters)
        for window_id, window_patterns in time_windows.items():
            if len(window_patterns) >= 3:  # Need sufficient activity
                
                # Extract unique files from this time window
                files_in_window = set()
                file_access_counts = defaultdict(int)
                
                for pattern in window_patterns:
                    file_path = pattern.get('file_path', '')
                    if file_path:
                        files_in_window.add(file_path)
                        file_access_counts[file_path] += 1
                
                if len(files_in_window) >= 2:
                    cluster_id = f"temporal_seq_{window_id}"
                    
                    # Calculate relationships based on co-occurrence in time window
                    total_accesses = sum(file_access_counts.values())
                    relationships = {
                        file_path: count / total_accesses 
                        for file_path, count in file_access_counts.items()
                    }
                    
                    cluster = MemoryCluster(
                        cluster_id=cluster_id,
                        cluster_type=MemoryClusterType.TEMPORAL_SEQUENCE,
                        members=files_in_window,
                        relationships=relationships,
                        access_frequency=total_accesses / len(files_in_window),
                        retention_priority=0.7,
                        last_accessed=max(p.get('timestamp', 0) for p in window_patterns),
                        performance_impact=0.6,
                        creation_time=time.time()
                    )
                    
                    clusters[cluster_id] = cluster
                    
                    logger.debug(f"Temporal cluster {cluster_id}: {len(files_in_window)} members")
        
        return clusters
    
    def _cluster_by_semantic_similarity(self, memories: List[Dict]) -> Dict[str, MemoryCluster]:
        """Cluster memories based on semantic content similarity"""
        clusters = {}
        
        try:
            # Extract text content for similarity analysis
            memory_texts = []
            memory_paths = []
            
            for memory in memories:
                # Use file path and type as semantic content
                file_path = memory.get('file_path', '')
                memory_type = memory.get('memory_type', '')
                classification = memory.get('classification', '')
                
                # Create semantic signature
                semantic_text = f"{file_path} {memory_type} {classification}"
                memory_texts.append(semantic_text)
                memory_paths.append(file_path)
            
            if len(memory_texts) < 2:
                return clusters
            
            # Vectorize content using TF-IDF
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            try:
                vectors = vectorizer.fit_transform(memory_texts)
                
                # Use DBSCAN clustering for semantic groups
                clustering = DBSCAN(eps=0.4, min_samples=2, metric='cosine')
                cluster_labels = clustering.fit_predict(vectors.toarray())
                
                # Group memories by cluster labels
                semantic_groups = defaultdict(list)
                for i, label in enumerate(cluster_labels):
                    if label != -1:  # Ignore noise cluster
                        semantic_groups[label].append((memory_paths[i], vectors[i]))
                
                # Create semantic clusters
                for cluster_label, group_data in semantic_groups.items():
                    if len(group_data) >= 2:
                        cluster_id = f"semantic_group_{cluster_label}"
                        
                        # Calculate pairwise similarities within cluster
                        files = [item[0] for item in group_data]
                        file_vectors = [item[1].toarray().flatten() for item in group_data]
                        
                        relationships = {}
                        for i, file_path in enumerate(files):
                            # Calculate average similarity to other members
                            similarities = []
                            for j, other_vector in enumerate(file_vectors):
                                if i != j:
                                    sim = cosine_similarity([file_vectors[i]], [other_vector])[0][0]
                                    similarities.append(sim)
                            
                            relationships[file_path] = np.mean(similarities) if similarities else 0.5
                        
                        cluster = MemoryCluster(
                            cluster_id=cluster_id,
                            cluster_type=MemoryClusterType.SEMANTIC_GROUP,
                            members=set(files),
                            relationships=relationships,
                            access_frequency=0.5,  # Default frequency
                            retention_priority=0.6,
                            last_accessed=time.time(),
                            performance_impact=0.4,
                            creation_time=time.time()
                        )
                        
                        clusters[cluster_id] = cluster
                        
                        logger.debug(f"Semantic cluster {cluster_id}: {len(files)} members")
            
            except Exception as e:
                logger.warning(f"TF-IDF vectorization failed: {e}")
        
        except Exception as e:
            logger.warning(f"Semantic clustering failed: {e}")
        
        return clusters
    
    def _cluster_by_performance_impact(self, memories: List[Dict], access_patterns: List[Dict]) -> Dict[str, MemoryCluster]:
        """Cluster memories based on performance impact"""
        clusters = {}
        
        # Analyze performance metrics from access patterns
        performance_data = defaultdict(list)
        
        for pattern in access_patterns:
            file_path = pattern.get('file_path', '')
            duration = pattern.get('duration', 0)
            success = pattern.get('success', True)
            
            if file_path:
                performance_data[file_path].append({
                    'duration': duration,
                    'success': 1 if success else 0
                })
        
        # Calculate performance metrics per file
        high_performance_files = []
        low_performance_files = []
        
        for file_path, perf_records in performance_data.items():
            if len(perf_records) >= 2:  # Need multiple accesses for reliable metrics
                avg_duration = np.mean([r['duration'] for r in perf_records])
                success_rate = np.mean([r['success'] for r in perf_records])
                
                # Performance score (lower duration + higher success = better performance)
                performance_score = success_rate / max(avg_duration, 0.001)
                
                if performance_score > self.clustering_params['performance_impact_threshold']:
                    high_performance_files.append((file_path, performance_score))
                elif avg_duration > 0.1 or success_rate < 0.5:  # Problematic files
                    low_performance_files.append((file_path, performance_score))
        
        # Create high-performance cluster
        if high_performance_files:
            cluster_id = "high_performance_cluster"
            files = [item[0] for item in high_performance_files]
            scores = [item[1] for item in high_performance_files]
            
            # Normalize scores as relationships
            max_score = max(scores)
            relationships = {files[i]: scores[i] / max_score for i in range(len(files))}
            
            cluster = MemoryCluster(
                cluster_id=cluster_id,
                cluster_type=MemoryClusterType.PERFORMANCE_CLUSTER,
                members=set(files),
                relationships=relationships,
                access_frequency=len(high_performance_files) / len(access_patterns) if access_patterns else 0,
                retention_priority=0.95,  # Very high priority
                last_accessed=time.time(),
                performance_impact=0.9,
                creation_time=time.time()
            )
            
            clusters[cluster_id] = cluster
            logger.debug(f"High-performance cluster: {len(files)} members")
        
        # Create low-performance cluster for optimization attention
        if low_performance_files:
            cluster_id = "low_performance_cluster" 
            files = [item[0] for item in low_performance_files]
            scores = [item[1] for item in low_performance_files]
            
            # Inverse scoring for low performance (lower is worse)
            min_score = min(scores) if scores else 0.1
            relationships = {files[i]: min_score / max(scores[i], 0.01) for i in range(len(files))}
            
            cluster = MemoryCluster(
                cluster_id=cluster_id,
                cluster_type=MemoryClusterType.PERFORMANCE_CLUSTER,
                members=set(files),
                relationships=relationships,
                access_frequency=len(low_performance_files) / len(access_patterns) if access_patterns else 0,
                retention_priority=0.3,  # Low priority for cleanup
                last_accessed=time.time(),
                performance_impact=0.1,  # Negative performance impact
                creation_time=time.time()
            )
            
            clusters[cluster_id] = cluster
            logger.debug(f"Low-performance cluster: {len(files)} members")
        
        return clusters
    
    def _cluster_by_cross_session_patterns(self, memories: List[Dict]) -> Dict[str, MemoryCluster]:
        """Cluster memories that show cross-session persistence patterns"""
        clusters = {}
        
        # Look for files that appear across multiple sessions/time periods
        # This is a simplified version - full implementation would analyze historical data
        
        # Group by file type and classification for potential cross-session patterns
        pattern_groups = defaultdict(list)
        
        for memory in memories:
            memory_type = memory.get('memory_type', 'unknown')
            classification = memory.get('classification', 'unknown')
            file_path = memory.get('file_path', '')
            
            # Create pattern key
            pattern_key = f"{memory_type}_{classification}"
            pattern_groups[pattern_key].append({
                'file_path': file_path,
                'memory': memory
            })
        
        # Create clusters for patterns with multiple instances
        for pattern_key, group_memories in pattern_groups.items():
            if len(group_memories) >= 3:  # Need multiple instances for cross-session pattern
                cluster_id = f"cross_session_{pattern_key}"
                
                files = [item['file_path'] for item in group_memories if item['file_path']]
                
                if files:
                    # Equal relationships for cross-session patterns
                    relationships = {file_path: 0.8 for file_path in files}
                    
                    cluster = MemoryCluster(
                        cluster_id=cluster_id,
                        cluster_type=MemoryClusterType.CROSS_SESSION,
                        members=set(files),
                        relationships=relationships,
                        access_frequency=0.3,  # Lower frequency but high persistence
                        retention_priority=0.99,  # Highest priority - cross-session learning
                        last_accessed=time.time(),
                        performance_impact=0.7,
                        creation_time=time.time()
                    )
                    
                    clusters[cluster_id] = cluster
                    logger.debug(f"Cross-session cluster {cluster_id}: {len(files)} members")
        
        return clusters
    
    def _build_cluster_hierarchy(self, clusters: Dict[str, MemoryCluster]):
        """Build hierarchical relationships between clusters"""
        # Clear existing graph
        self.cluster_graph.clear()
        
        # Add all clusters as nodes
        for cluster_id, cluster in clusters.items():
            self.cluster_graph.add_node(cluster_id, cluster_data=cluster)
        
        # Build relationships between clusters based on member overlap
        cluster_ids = list(clusters.keys())
        
        for i, cluster_id_1 in enumerate(cluster_ids):
            for cluster_id_2 in cluster_ids[i+1:]:
                cluster_1 = clusters[cluster_id_1]
                cluster_2 = clusters[cluster_id_2]
                
                # Calculate member overlap
                overlap = cluster_1.members.intersection(cluster_2.members)
                overlap_ratio = len(overlap) / min(len(cluster_1.members), len(cluster_2.members))
                
                # Create relationships for significant overlaps
                if overlap_ratio > 0.2:  # 20% overlap threshold
                    self.cluster_graph.add_edge(cluster_id_1, cluster_id_2, 
                                              weight=overlap_ratio, overlap_size=len(overlap))
                    
                    logger.debug(f"Cluster relationship: {cluster_id_1} <-> {cluster_id_2} "
                               f"(overlap: {overlap_ratio:.2f})")
    
    def _rebuild_memory_to_cluster_mapping(self):
        """Rebuild the mapping from memory IDs to cluster IDs"""
        self.memory_to_clusters.clear()
        
        for cluster_id, cluster in self.clusters.items():
            for memory_id in cluster.members:
                self.memory_to_clusters[memory_id].add(cluster_id)
    
    def get_memory_cluster_info(self, memory_id: str) -> Dict[str, Any]:
        """Get cluster information for a specific memory"""
        cluster_ids = self.memory_to_clusters.get(memory_id, set())
        
        if not cluster_ids:
            return {"status": "unclustered", "cluster_count": 0}
        
        cluster_info = {
            "status": "clustered",
            "cluster_count": len(cluster_ids),
            "clusters": []
        }
        
        for cluster_id in cluster_ids:
            cluster = self.clusters.get(cluster_id)
            if cluster:
                health = cluster.get_cluster_health()
                cluster_info["clusters"].append({
                    "cluster_id": cluster_id,
                    "cluster_type": cluster.cluster_type.value,
                    "retention_priority": cluster.retention_priority,
                    "relationship_strength": cluster.relationships.get(memory_id, 0.0),
                    "health_score": health["health_score"],
                    "cluster_size": health["cluster_size"]
                })
        
        # Calculate overall retention priority (max across clusters)
        if cluster_info["clusters"]:
            cluster_info["max_retention_priority"] = max(
                c["retention_priority"] for c in cluster_info["clusters"]
            )
            cluster_info["avg_relationship_strength"] = np.mean([
                c["relationship_strength"] for c in cluster_info["clusters"]
            ])
        
        return cluster_info
    
    def get_cluster_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get Governor-ready optimization recommendations based on clusters"""
        recommendations = []
        
        # Analyze cluster health and generate recommendations
        for cluster_id, cluster in self.clusters.items():
            health = cluster.get_cluster_health()
            
            # Low health clusters need attention
            if health["health_score"] < 0.5:
                if cluster.cluster_type == MemoryClusterType.PERFORMANCE_CLUSTER:
                    recommendations.append({
                        "type": "performance_optimization",
                        "priority": "high",
                        "action": f"optimize_performance_cluster_{cluster_id}",
                        "reason": f"Performance cluster {cluster_id} has low health score: {health['health_score']:.2f}",
                        "cluster_id": cluster_id,
                        "expected_improvement": f"Up to {(1-health['health_score'])*100:.0f}% performance gain",
                        "source": "hierarchical_clustering_phase2"
                    })
                
                elif health["last_access_hours"] > 72:  # Not accessed recently
                    recommendations.append({
                        "type": "cluster_cleanup",
                        "priority": "medium",
                        "action": f"cleanup_stale_cluster_{cluster_id}",
                        "reason": f"Cluster {cluster_id} not accessed for {health['last_access_hours']:.1f} hours",
                        "cluster_id": cluster_id,
                        "expected_improvement": "Memory usage reduction",
                        "source": "hierarchical_clustering_phase2"
                    })
            
            # Oversized clusters need splitting
            if health["cluster_size"] > 20:
                recommendations.append({
                    "type": "cluster_management",
                    "priority": "medium", 
                    "action": f"split_oversized_cluster_{cluster_id}",
                    "reason": f"Cluster {cluster_id} has {health['cluster_size']} members (oversized)",
                    "cluster_id": cluster_id,
                    "expected_improvement": "Better cluster organization and access efficiency",
                    "source": "hierarchical_clustering_phase2"
                })
        
        # Cross-cluster optimization opportunities
        high_overlap_pairs = []
        for edge in self.cluster_graph.edges(data=True):
            cluster_1, cluster_2, edge_data = edge
            if edge_data.get('weight', 0) > 0.5:  # High overlap
                high_overlap_pairs.append((cluster_1, cluster_2, edge_data['weight']))
        
        for cluster_1, cluster_2, overlap_weight in high_overlap_pairs:
            recommendations.append({
                "type": "cluster_merge_opportunity",
                "priority": "low",
                "action": f"consider_merging_{cluster_1}_{cluster_2}",
                "reason": f"High overlap ({overlap_weight:.2f}) between clusters",
                "cluster_ids": [cluster_1, cluster_2],
                "expected_improvement": "Reduced cluster overhead and improved coherence",
                "source": "hierarchical_clustering_phase2"
            })
        
        # Sort recommendations by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "medium"), 1))
        
        logger.info(f"Generated {len(recommendations)} cluster-based optimization recommendations")
        
        return recommendations
    
    def get_clustering_summary(self) -> Dict[str, Any]:
        """Get summary of current clustering state"""
        cluster_type_counts = defaultdict(int)
        total_members = 0
        
        for cluster in self.clusters.values():
            cluster_type_counts[cluster.cluster_type.value] += 1
            total_members += len(cluster.members)
        
        # Calculate cluster health distribution
        health_scores = []
        for cluster in self.clusters.values():
            health = cluster.get_cluster_health()
            health_scores.append(health["health_score"])
        
        return {
            "total_clusters": len(self.clusters),
            "cluster_types": dict(cluster_type_counts),
            "total_clustered_memories": total_members,
            "avg_cluster_size": total_members / len(self.clusters) if self.clusters else 0,
            "cluster_health": {
                "avg_health_score": np.mean(health_scores) if health_scores else 0,
                "healthy_clusters": sum(1 for h in health_scores if h > 0.7),
                "unhealthy_clusters": sum(1 for h in health_scores if h < 0.5)
            },
            "cluster_relationships": self.cluster_graph.number_of_edges(),
            "clustering_timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    # Quick test of the hierarchical clusterer
    clusterer = HierarchicalMemoryClusterer()
    
    # Simulate some memory data
    test_memories = [
        {"file_path": "governor_decisions.log", "memory_type": "CRITICAL_LOSSLESS", "classification": "critical"},
        {"file_path": "architect_evolution.json", "memory_type": "CRITICAL_LOSSLESS", "classification": "critical"},
        {"file_path": "session_data_001.json", "memory_type": "IMPORTANT_DECAY", "classification": "important"},
        {"file_path": "session_data_002.json", "memory_type": "IMPORTANT_DECAY", "classification": "important"},
        {"file_path": "temp_debug.log", "memory_type": "TEMPORARY_PURGE", "classification": "temporary"},
        {"file_path": "temp_cache.json", "memory_type": "TEMPORARY_PURGE", "classification": "temporary"},
    ]
    
    test_patterns = [
        {"file_path": "governor_decisions.log", "timestamp": time.time() - 100, "duration": 0.001, "success": True},
        {"file_path": "architect_evolution.json", "timestamp": time.time() - 95, "duration": 0.002, "success": True},
        {"file_path": "session_data_001.json", "timestamp": time.time() - 50, "duration": 0.005, "success": True},
        {"file_path": "session_data_002.json", "timestamp": time.time() - 45, "duration": 0.004, "success": True},
    ]
    
    # Create clusters
    clusters = clusterer.create_intelligent_clusters(test_memories, test_patterns)
    
    # Get summary
    summary = clusterer.get_clustering_summary()
    recommendations = clusterer.get_cluster_optimization_recommendations()
    
    print(" Hierarchical Memory Clustering Test Results:")
    print(f" Created {summary['total_clusters']} intelligent clusters")
    print(f" Cluster types: {summary['cluster_types']}")
    print(f" Average cluster health: {summary['cluster_health']['avg_health_score']:.3f}")
    print(f" Optimization recommendations: {len(recommendations)}")
    print(" Phase 2 hierarchical clustering operational!")
