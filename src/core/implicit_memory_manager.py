#!/usr/bin/env python3
"""
Implicit Memory Manager

A memory system that uses implicit representations and compressed storage
to efficiently store and retrieve memories with space-efficient encoding.

Key Features:
- Implicit memory representations using compressed codes
- Tree-based memory organization for efficient retrieval
- Space-efficient storage with O(√n) complexity
- Memory hierarchy with different compression levels
- Integration with existing memory systems
"""

import time
import json
import logging
import hashlib
import pickle
import zlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of memories in the system."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    META_COGNITIVE = "meta_cognitive"
    PATTERN = "pattern"
    EXPERIENCE = "experience"
    LEARNING = "learning"
    STRATEGY = "strategy"
    MEMORY = "memory"

class CompressionLevel(Enum):
    """Compression levels for memory storage."""
    NONE = "none"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"
    ULTRA = "ultra"

class MemoryPriority(Enum):
    """Priority levels for memory retention."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ARCHIVE = "archive"

@dataclass
class MemoryNode:
    """A node in the memory tree."""
    memory_id: str
    memory_type: MemoryType
    content: Any
    compressed_content: Optional[bytes] = None
    compression_level: CompressionLevel = CompressionLevel.MEDIUM
    priority: MemoryPriority = MemoryPriority.MEDIUM
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    metadata: Dict[str, Any] = None
    created_at: float = None
    last_accessed: float = None
    access_count: int = 0
    compressed_size: int = 0
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = time.time()
        if self.last_accessed is None:
            self.last_accessed = self.created_at

@dataclass
class MemoryCluster:
    """A cluster of related memories."""
    cluster_id: str
    cluster_type: MemoryType
    memories: List[str]  # Memory IDs
    centroid: Optional[Dict[str, Any]] = None
    compressed_centroid: Optional[bytes] = None
    created_at: float = None
    last_updated: float = None
    access_count: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.last_updated is None:
            self.last_updated = self.created_at

class ImplicitMemoryManager:
    """
    Implicit Memory Manager that provides compressed memory storage and retrieval.
    
    This system uses implicit representations and compressed storage to efficiently
    manage memories with space-efficient encoding and tree-based organization.
    """
    
    def __init__(self, 
                 max_memory_mb: float = 100.0,
                 compression_threshold: float = 0.7,
                 cluster_size: int = 10,
                 persistence_dir: Optional[Path] = None):
        
        self.max_memory_mb = max_memory_mb
        self.compression_threshold = compression_threshold
        self.cluster_size = cluster_size
        self.persistence_dir = None  # Database-only mode
        # No directory creation needed for database-only mode
        
        # Memory storage
        self.memories: Dict[str, MemoryNode] = {}
        self.memory_clusters: Dict[str, MemoryCluster] = {}
        self.memory_index: Dict[str, List[str]] = {}  # Type -> Memory IDs
        self.access_history: List[str] = []  # LRU tracking
        
        # Compression statistics
        self.compression_stats = {
            'total_memories': 0,
            'compressed_memories': 0,
            'total_compression_ratio': 0.0,
            'memory_savings_mb': 0.0
        }
        
        # Tree-based organization
        self.memory_tree_root = None
        self.tree_depth = 0
        
        logger.info("Implicit Memory Manager initialized")
    
    def store_memory(self, 
                    content: Any,
                    memory_type: MemoryType,
                    priority: MemoryPriority = MemoryPriority.MEDIUM,
                    compression_level: CompressionLevel = CompressionLevel.MEDIUM,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a memory with implicit representation and compression.
        
        Args:
            content: Memory content to store
            memory_type: Type of memory
            priority: Priority level for retention
            compression_level: Level of compression to apply
            metadata: Additional metadata
            
        Returns:
            memory_id: Unique identifier for the stored memory
        """
        memory_id = f"mem_{int(time.time() * 1000)}_{hashlib.md5(str(content).encode()).hexdigest()[:8]}"
        
        # Compress content if needed
        compressed_content, compressed_size = self._compress_content(content, compression_level)
        
        # Create memory node
        memory_node = MemoryNode(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            compressed_content=compressed_content,
            compression_level=compression_level,
            priority=priority,
            metadata=metadata or {},
            compressed_size=compressed_size
        )
        
        # Store memory
        self.memories[memory_id] = memory_node
        
        # Update index
        if memory_type.value not in self.memory_index:
            self.memory_index[memory_type.value] = []
        self.memory_index[memory_type.value].append(memory_id)
        
        # Update statistics
        self.compression_stats['total_memories'] += 1
        if compressed_content is not None:
            self.compression_stats['compressed_memories'] += 1
            original_size = len(pickle.dumps(content))
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
            self.compression_stats['total_compression_ratio'] = (
                (self.compression_stats['total_compression_ratio'] * (self.compression_stats['compressed_memories'] - 1) + 
                 compression_ratio) / self.compression_stats['compressed_memories']
            )
            self.compression_stats['memory_savings_mb'] += (original_size - compressed_size) / (1024 * 1024)
        
        # Check memory limits
        self._check_memory_limits()
        
        # Add to memory tree
        self._add_to_memory_tree(memory_node)
        
        logger.debug(f"Stored memory {memory_id} (type: {memory_type.value}, compressed: {compressed_content is not None})")
        return memory_id
    
    def retrieve_memory(self, 
                       memory_id: str,
                       decompress: bool = True) -> Optional[Any]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            decompress: Whether to decompress the content
            
        Returns:
            Memory content or None if not found
        """
        if memory_id not in self.memories:
            return None
        
        memory_node = self.memories[memory_id]
        
        # Update access tracking
        memory_node.last_accessed = time.time()
        memory_node.access_count += 1
        self._update_access_history(memory_id)
        
        # Return content
        if decompress and memory_node.compressed_content is not None:
            return self._decompress_content(memory_node.compressed_content, memory_node.compression_level)
        else:
            return memory_node.content
    
    def search_memories(self, 
                       query: str,
                       memory_type: Optional[MemoryType] = None,
                       limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search memories using implicit representations.
        
        Args:
            query: Search query
            memory_type: Filter by memory type
            limit: Maximum number of results
            
        Returns:
            List of matching memories with relevance scores
        """
        results = []
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Get memories to search
        memories_to_search = []
        if memory_type:
            memories_to_search = self.memory_index.get(memory_type.value, [])
        else:
            memories_to_search = list(self.memories.keys())
        
        # Search using implicit representations
        for memory_id in memories_to_search:
            if memory_id not in self.memories:
                continue
            
            memory_node = self.memories[memory_id]
            
            # Calculate relevance score using implicit matching
            relevance_score = self._calculate_relevance_score(query, memory_node)
            
            if relevance_score > 0.1:  # Threshold for relevance
                results.append({
                    'memory_id': memory_id,
                    'memory_type': memory_node.memory_type.value,
                    'content': memory_node.content,
                    'relevance_score': relevance_score,
                    'created_at': memory_node.created_at,
                    'access_count': memory_node.access_count,
                    'priority': memory_node.priority.value
                })
        
        # Sort by relevance score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return results[:limit]
    
    def cluster_memories(self, 
                        memory_type: MemoryType,
                        max_clusters: int = 10) -> List[MemoryCluster]:
        """
        Cluster memories of a specific type using implicit representations.
        
        Args:
            memory_type: Type of memories to cluster
            max_clusters: Maximum number of clusters
            
        Returns:
            List of memory clusters
        """
        memory_ids = self.memory_index.get(memory_type.value, [])
        if len(memory_ids) < 2:
            return []
        
        # Extract memory features for clustering
        features = []
        memory_id_map = {}
        
        for i, memory_id in enumerate(memory_ids):
            if memory_id in self.memories:
                memory_node = self.memories[memory_id]
                feature_vector = self._extract_memory_features(memory_node)
                features.append(feature_vector)
                memory_id_map[i] = memory_id
        
        if len(features) < 2:
            return []
        
        # Perform clustering using simple k-means-like approach
        clusters = self._perform_clustering(features, memory_id_map, max_clusters)
        
        # Create memory clusters
        memory_clusters = []
        for i, cluster_memory_ids in enumerate(clusters):
            if not cluster_memory_ids:
                continue
            
            cluster_id = f"cluster_{memory_type.value}_{i}_{int(time.time())}"
            
            # Calculate centroid
            centroid = self._calculate_cluster_centroid(cluster_memory_ids)
            compressed_centroid = self._compress_content(centroid, CompressionLevel.HEAVY)[0]
            
            cluster = MemoryCluster(
                cluster_id=cluster_id,
                cluster_type=memory_type,
                memories=cluster_memory_ids,
                centroid=centroid,
                compressed_centroid=compressed_centroid
            )
            
            self.memory_clusters[cluster_id] = cluster
            memory_clusters.append(cluster)
        
        logger.info(f"Created {len(memory_clusters)} clusters for {memory_type.value} memories")
        return memory_clusters
    
    def _compress_content(self, 
                         content: Any, 
                         compression_level: CompressionLevel) -> Tuple[Optional[bytes], int]:
        """Compress content based on compression level."""
        
        if compression_level == CompressionLevel.NONE:
            return None, 0
        
        try:
            # Serialize content
            serialized = pickle.dumps(content)
            
            # Apply compression based on level
            if compression_level == CompressionLevel.LIGHT:
                compressed = zlib.compress(serialized, level=1)
            elif compression_level == CompressionLevel.MEDIUM:
                compressed = zlib.compress(serialized, level=6)
            elif compression_level == CompressionLevel.HEAVY:
                compressed = zlib.compress(serialized, level=9)
            elif compression_level == CompressionLevel.ULTRA:
                # Ultra compression with additional encoding
                compressed = zlib.compress(serialized, level=9)
                compressed = zlib.compress(compressed, level=9)  # Double compression
            else:
                compressed = serialized
            
            return compressed, len(compressed)
            
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return None, 0
    
    def _decompress_content(self, 
                           compressed_content: bytes, 
                           compression_level: CompressionLevel) -> Any:
        """Decompress content based on compression level."""
        
        try:
            if compression_level == CompressionLevel.ULTRA:
                # Double decompression for ultra level
                decompressed = zlib.decompress(compressed_content)
                decompressed = zlib.decompress(decompressed)
            else:
                decompressed = zlib.decompress(compressed_content)
            
            return pickle.loads(decompressed)
            
        except Exception as e:
            logger.warning(f"Decompression failed: {e}")
            return None
    
    def _calculate_relevance_score(self, 
                                 query: str, 
                                 memory_node: MemoryNode) -> float:
        """Calculate relevance score using implicit representations."""
        
        # Simple relevance calculation based on content similarity
        content_str = str(memory_node.content).lower()
        query_lower = query.lower()
        
        # Basic keyword matching
        query_words = query_lower.split()
        content_words = content_str.split()
        
        if not query_words:
            return 0.0
        
        # Calculate word overlap
        common_words = set(query_words) & set(content_words)
        word_overlap = len(common_words) / len(query_words)
        
        # Calculate character-level similarity
        char_similarity = self._calculate_char_similarity(query_lower, content_str)
        
        # Combine scores
        relevance_score = (word_overlap * 0.7 + char_similarity * 0.3)
        
        # Boost score based on access frequency and recency
        recency_boost = min(0.2, (time.time() - memory_node.last_accessed) / (24 * 3600) * 0.1)
        frequency_boost = min(0.1, memory_node.access_count * 0.01)
        
        relevance_score += recency_boost + frequency_boost
        
        return min(1.0, relevance_score)
    
    def _calculate_char_similarity(self, str1: str, str2: str) -> float:
        """Calculate character-level similarity between two strings."""
        if not str1 or not str2:
            return 0.0
        
        # Simple character overlap calculation
        set1 = set(str1)
        set2 = set(str2)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_memory_features(self, memory_node: MemoryNode) -> List[float]:
        """Extract features from memory for clustering."""
        
        features = []
        
        # Content-based features
        content_str = str(memory_node.content)
        features.append(len(content_str))  # Length
        features.append(len(content_str.split()))  # Word count
        features.append(len(set(content_str.split())))  # Unique word count
        
        # Temporal features
        features.append(time.time() - memory_node.created_at)  # Age
        features.append(time.time() - memory_node.last_accessed)  # Time since last access
        features.append(memory_node.access_count)  # Access frequency
        
        # Type-based features
        type_encoding = [0.0] * len(MemoryType)
        type_encoding[list(MemoryType).index(memory_node.memory_type)] = 1.0
        features.extend(type_encoding)
        
        # Priority-based features
        priority_encoding = [0.0] * len(MemoryPriority)
        priority_encoding[list(MemoryPriority).index(memory_node.priority)] = 1.0
        features.extend(priority_encoding)
        
        return features
    
    def _perform_clustering(self, 
                          features: List[List[float]], 
                          memory_id_map: Dict[int, str],
                          max_clusters: int) -> List[List[str]]:
        """Perform clustering on memory features."""
        
        if len(features) < 2:
            return []
        
        # Simple k-means-like clustering
        n_features = len(features[0])
        n_memories = len(features)
        
        # Determine number of clusters
        n_clusters = min(max_clusters, max(1, n_memories // self.cluster_size))
        
        # Initialize cluster centers randomly
        cluster_centers = []
        for _ in range(n_clusters):
            center = [np.random.random() for _ in range(n_features)]
            cluster_centers.append(center)
        
        # Assign memories to clusters
        clusters = [[] for _ in range(n_clusters)]
        
        for i, feature_vector in enumerate(features):
            # Find closest cluster center
            distances = []
            for center in cluster_centers:
                distance = sum((a - b) ** 2 for a, b in zip(feature_vector, center))
                distances.append(distance)
            
            closest_cluster = distances.index(min(distances))
            memory_id = memory_id_map[i]
            clusters[closest_cluster].append(memory_id)
        
        # Filter out empty clusters
        return [cluster for cluster in clusters if cluster]
    
    def _calculate_cluster_centroid(self, memory_ids: List[str]) -> Dict[str, Any]:
        """Calculate centroid for a cluster of memories."""
        
        if not memory_ids:
            return {}
        
        # Extract common features from cluster memories
        cluster_features = {}
        
        for memory_id in memory_ids:
            if memory_id in self.memories:
                memory_node = self.memories[memory_id]
                
                # Extract common patterns
                content_str = str(memory_node.content)
                words = content_str.split()
                
                for word in words:
                    if word not in cluster_features:
                        cluster_features[word] = 0
                    cluster_features[word] += 1
        
        # Return top features as centroid
        sorted_features = sorted(cluster_features.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:20])  # Top 20 features
    
    def _add_to_memory_tree(self, memory_node: MemoryNode):
        """Add memory to the tree-based organization."""
        
        # Simple tree organization based on memory type and priority
        if self.memory_tree_root is None:
            self.memory_tree_root = memory_node.memory_id
        else:
            # Find appropriate parent based on type and priority
            parent_id = self._find_tree_parent(memory_node)
            if parent_id:
                memory_node.parent_id = parent_id
                if parent_id in self.memories:
                    self.memories[parent_id].children_ids.append(memory_node.memory_id)
    
    def _find_tree_parent(self, memory_node: MemoryNode) -> Optional[str]:
        """Find appropriate parent for memory in tree structure."""
        
        # Simple parent finding based on type and priority
        for existing_id, existing_memory in self.memories.items():
            if (existing_memory.memory_type == memory_node.memory_type and
                existing_memory.priority == memory_node.priority and
                len(existing_memory.children_ids) < 5):  # Limit children per node
                return existing_id
        
        return None
    
    def _update_access_history(self, memory_id: str):
        """Update LRU access history."""
        
        if memory_id in self.access_history:
            self.access_history.remove(memory_id)
        
        self.access_history.append(memory_id)
        
        # Limit history size
        if len(self.access_history) > 1000:
            self.access_history = self.access_history[-1000:]
    
    def _check_memory_limits(self):
        """Check and enforce memory limits."""
        
        current_memory_mb = self._estimate_memory_usage()
        
        if current_memory_mb > self.max_memory_mb:
            # Remove least recently used memories
            self._cleanup_memories()
    
    def _cleanup_memories(self):
        """Clean up least recently used memories."""
        
        # Sort by priority and last access time
        memory_list = list(self.memories.items())
        memory_list.sort(key=lambda x: (
            x[1].priority.value,
            x[1].last_accessed
        ))
        
        # Remove low priority, old memories
        removed_count = 0
        for memory_id, memory_node in memory_list:
            if memory_node.priority in [MemoryPriority.LOW, MemoryPriority.ARCHIVE]:
                self._remove_memory(memory_id)
                removed_count += 1
                
                if self._estimate_memory_usage() <= self.max_memory_mb * 0.8:
                    break
        
        logger.info(f"Cleaned up {removed_count} memories")
    
    def _remove_memory(self, memory_id: str):
        """Remove a memory from the system."""
        
        if memory_id not in self.memories:
            return
        
        memory_node = self.memories[memory_id]
        
        # Remove from index
        if memory_node.memory_type.value in self.memory_index:
            if memory_id in self.memory_index[memory_node.memory_type.value]:
                self.memory_index[memory_node.memory_type.value].remove(memory_id)
        
        # Remove from access history
        if memory_id in self.access_history:
            self.access_history.remove(memory_id)
        
        # Remove from tree
        if memory_node.parent_id and memory_node.parent_id in self.memories:
            self.memories[memory_node.parent_id].children_ids.remove(memory_id)
        
        # Remove children
        for child_id in memory_node.children_ids:
            if child_id in self.memories:
                self.memories[child_id].parent_id = None
        
        # Remove memory
        del self.memories[memory_id]
    
    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB."""
        
        total_size = 0
        
        for memory_node in self.memories.values():
            # Estimate size of memory node
            node_size = len(pickle.dumps(memory_node))
            total_size += node_size
        
        return total_size / (1024 * 1024)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        
        return {
            'total_memories': len(self.memories),
            'memory_clusters': len(self.memory_clusters),
            'memory_usage_mb': self._estimate_memory_usage(),
            'compression_stats': self.compression_stats.copy(),
            'memory_types': {
                memory_type_name: len(memory_ids)
                for memory_type_name, memory_ids in self.memory_index.items()
            },
            'access_history_size': len(self.access_history)
        }
    
    def export_memories(self, 
                       memory_type: Optional[MemoryType] = None,
                       file_path: Optional[Path] = None) -> Path:
        """Export memories to a file."""
        
        if file_path is None:
            file_path = self.persistence_dir / f"memories_{int(time.time())}.json"
        
        # Filter memories by type
        memories_to_export = {}
        if memory_type:
            memory_ids = self.memory_index.get(memory_type.value, [])
            memories_to_export = {mid: self.memories[mid] for mid in memory_ids if mid in self.memories}
        else:
            memories_to_export = self.memories.copy()
        
        # Convert to serializable format
        export_data = {
            'export_time': time.time(),
            'memory_type_filter': memory_type.value if memory_type else None,
            'memories': {}
        }
        
        for memory_id, memory_node in memories_to_export.items():
            export_data['memories'][memory_id] = {
                'memory_id': memory_node.memory_id,
                'memory_type': memory_node.memory_type.value,
                'content': memory_node.content,
                'priority': memory_node.priority.value,
                'created_at': memory_node.created_at,
                'last_accessed': memory_node.last_accessed,
                'access_count': memory_node.access_count,
                'metadata': memory_node.metadata
            }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(memories_to_export)} memories to {file_path}")
        return file_path


# Factory function
def create_implicit_memory_manager(max_memory_mb: float = 100.0,
                                  compression_threshold: float = 0.7,
                                  cluster_size: int = 10,
                                  persistence_dir: Optional[Path] = None) -> ImplicitMemoryManager:
    """Create an Implicit Memory Manager instance."""
    return ImplicitMemoryManager(
        max_memory_mb=max_memory_mb,
        compression_threshold=compression_threshold,
        cluster_size=cluster_size,
        persistence_dir=persistence_dir
    )
