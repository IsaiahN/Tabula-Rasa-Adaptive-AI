"""
Enhanced Knowledge Transfer System

This module provides advanced knowledge transfer capabilities with comprehensive
cross-task learning persistence, semantic similarity analysis, and adaptive
transfer strategies.
"""

import logging
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import numpy as np
from pathlib import Path

from ..database.system_integration import get_system_integration
from ..database.api import Component, LogLevel

logger = logging.getLogger(__name__)

class TransferType(Enum):
    """Types of knowledge transfer."""
    PATTERN = "pattern"
    STRATEGY = "strategy"
    COORDINATE = "coordinate"
    ACTION_SEQUENCE = "action_sequence"
    VISUAL_PATTERN = "visual_pattern"
    SPATIAL_REASONING = "spatial_reasoning"
    LOGICAL_REASONING = "logical_reasoning"

class TransferConfidence(Enum):
    """Confidence levels for knowledge transfer."""
    LOW = "low"           # 0.0 - 0.4
    MEDIUM = "medium"     # 0.4 - 0.7
    HIGH = "high"         # 0.7 - 0.9
    VERY_HIGH = "very_high"  # 0.9 - 1.0

@dataclass
class TransferableKnowledge:
    """Represents knowledge that can be transferred between tasks."""
    knowledge_id: str
    source_game: str
    knowledge_type: TransferType
    content: Dict[str, Any]
    confidence: float
    success_rate: float
    usage_count: int
    last_used: float
    created_at: float
    tags: List[str]
    context_features: Dict[str, Any]
    transfer_history: List[Dict[str, Any]]

@dataclass
class TransferResult:
    """Result of a knowledge transfer operation."""
    transfer_id: str
    source_game: str
    target_game: str
    knowledge_type: TransferType
    transferred_items: List[str]
    confidence: float
    effectiveness: float
    adaptation_notes: List[str]
    timestamp: float
    success: bool
    performance_improvement: float

@dataclass
class GameSimilarityProfile:
    """Profile of a game for similarity analysis."""
    game_id: str
    visual_features: Dict[str, Any]
    spatial_features: Dict[str, Any]
    action_patterns: List[str]
    success_patterns: List[str]
    coordinate_zones: List[Tuple[int, int]]
    complexity_score: float
    pattern_types: Set[TransferType]
    last_updated: float

class EnhancedKnowledgeTransfer:
    """Enhanced knowledge transfer system with comprehensive persistence."""
    
    def __init__(self, persistence_dir: Optional[Path] = None, 
                 transfer_threshold: float = 0.6,
                 enable_database_storage: bool = True):
        self.persistence_dir = persistence_dir
        self.transfer_threshold = transfer_threshold
        self.enable_database_storage = enable_database_storage
        
        # Initialize database integration
        if self.enable_database_storage:
            self.integration = get_system_integration()
            self.Component = Component
            self.LogLevel = LogLevel
        
        # Knowledge storage
        self.transferable_knowledge: Dict[str, TransferableKnowledge] = {}
        self.game_profiles: Dict[str, GameSimilarityProfile] = {}
        self.transfer_history: List[TransferResult] = []
        self.transfer_results: Dict[str, TransferResult] = {}
        
        # Transfer statistics
        self.stats = {
            'total_transfers': 0,
            'successful_transfers': 0,
            'failed_transfers': 0,
            'transfer_effectiveness': 0.0,
            'knowledge_items': 0,
            'games_tracked': 0,
            'last_cleanup': time.time()
        }
        
        # Similarity analysis
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        self.similarity_threshold = 0.5
        
        # Adaptive learning
        self.adaptation_learning = defaultdict(list)
        self.transfer_strategies = defaultdict(list)
        
        # Load existing data
        self._load_persistent_data()
        
        logger.info("Enhanced Knowledge Transfer system initialized")
    
    def add_knowledge(self, source_game: str, knowledge_type: TransferType,
                     content: Dict[str, Any], confidence: float, 
                     success_rate: float, tags: List[str] = None,
                     context_features: Dict[str, Any] = None) -> str:
        """Add new transferable knowledge."""
        try:
            knowledge_id = self._generate_knowledge_id(source_game, knowledge_type, content)
            
            knowledge = TransferableKnowledge(
                knowledge_id=knowledge_id,
                source_game=source_game,
                knowledge_type=knowledge_type,
                content=content,
                confidence=confidence,
                success_rate=success_rate,
                usage_count=0,
                last_used=time.time(),
                created_at=time.time(),
                tags=tags or [],
                context_features=context_features or {},
                transfer_history=[]
            )
            
            self.transferable_knowledge[knowledge_id] = knowledge
            self.stats['knowledge_items'] += 1
            
            # Update game profile
            self._update_game_profile(source_game, knowledge)
            
            # Store in database
            if self.enable_database_storage:
                self._store_knowledge_in_database(knowledge)
            
            logger.info(f"Added knowledge {knowledge_id} from {source_game}")
            return knowledge_id
            
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return ""
    
    def transfer_knowledge(self, source_game: str, target_game: str,
                          knowledge_types: List[TransferType] = None,
                          min_confidence: float = None) -> TransferResult:
        """Transfer knowledge from source game to target game."""
        try:
            transfer_id = self._generate_transfer_id(source_game, target_game)
            
            # Calculate game similarity
            similarity = self._calculate_game_similarity(source_game, target_game)
            
            if similarity < self.similarity_threshold:
                return TransferResult(
                    transfer_id=transfer_id,
                    source_game=source_game,
                    target_game=target_game,
                    knowledge_type=TransferType.PATTERN,
                    transferred_items=[],
                    confidence=0.0,
                    effectiveness=0.0,
                    adaptation_notes=[f"Low similarity: {similarity:.2f}"],
                    timestamp=time.time(),
                    success=False,
                    performance_improvement=0.0
                )
            
            # Find applicable knowledge
            applicable_knowledge = self._find_applicable_knowledge(
                source_game, target_game, knowledge_types, min_confidence
            )
            
            if not applicable_knowledge:
                transfer_result = TransferResult(
                    transfer_id=transfer_id,
                    source_game=source_game,
                    target_game=target_game,
                    knowledge_type=TransferType.PATTERN,
                    transferred_items=[],
                    confidence=0.0,
                    effectiveness=0.0,
                    adaptation_notes=["No applicable knowledge found"],
                    timestamp=time.time(),
                    success=False,
                    performance_improvement=0.0
                )
                self.transfer_results[transfer_id] = transfer_result
                return transfer_result
            
            # Transfer knowledge
            transferred_items = []
            adaptation_notes = []
            total_confidence = 0.0
            
            for knowledge in applicable_knowledge:
                # Adapt knowledge to target context
                adapted_content = self._adapt_knowledge_to_context(
                    knowledge, target_game
                )
                
                # Create transfer record
                transfer_record = {
                    'target_game': target_game,
                    'confidence': knowledge.confidence,
                    'adaptation': adapted_content,
                    'timestamp': time.time()
                }
                
                knowledge.transfer_history.append(transfer_record)
                knowledge.usage_count += 1
                knowledge.last_used = time.time()
                
                transferred_items.append(knowledge.knowledge_id)
                total_confidence += knowledge.confidence
                
                # Generate adaptation notes
                notes = self._generate_adaptation_notes(knowledge, target_game)
                adaptation_notes.extend(notes)
            
            # Calculate overall confidence and effectiveness
            avg_confidence = total_confidence / len(applicable_knowledge)
            effectiveness = self._calculate_transfer_effectiveness(
                applicable_knowledge, target_game
            )
            
            # Create transfer result
            transfer_result = TransferResult(
                transfer_id=transfer_id,
                source_game=source_game,
                target_game=target_game,
                knowledge_type=knowledge_types[0] if knowledge_types else TransferType.PATTERN,
                transferred_items=transferred_items,
                confidence=avg_confidence,
                effectiveness=effectiveness,
                adaptation_notes=adaptation_notes,
                timestamp=time.time(),
                success=True,
                performance_improvement=0.0  # Will be updated after evaluation
            )
            
            # Record transfer
            self.transfer_history.append(transfer_result)
            self.transfer_results[transfer_id] = transfer_result
            self.stats['total_transfers'] += 1
            
            # Store in database
            if self.enable_database_storage:
                self._store_transfer_in_database(transfer_result)
            
            logger.info(f"Transferred {len(transferred_items)} knowledge items from {source_game} to {target_game}")
            return transfer_result
            
        except Exception as e:
            logger.error(f"Error transferring knowledge: {e}")
            transfer_result = TransferResult(
                transfer_id=transfer_id,
                source_game=source_game,
                target_game=target_game,
                knowledge_type=TransferType.PATTERN,
                transferred_items=[],
                confidence=0.0,
                effectiveness=0.0,
                adaptation_notes=[f"Transfer failed: {str(e)}"],
                timestamp=time.time(),
                success=False,
                performance_improvement=0.0
            )
            self.transfer_results[transfer_id] = transfer_result
            return transfer_result
    
    def evaluate_transfer_success(self, transfer_id: str, success: bool,
                                 performance_improvement: float = 0.0):
        """Evaluate the success of a knowledge transfer."""
        try:
            # Find transfer result
            transfer_result = None
            for result in self.transfer_history:
                if result.transfer_id == transfer_id:
                    transfer_result = result
                    break
            
            if not transfer_result:
                logger.warning(f"Transfer result {transfer_id} not found")
                return
            
            # Update transfer result
            transfer_result.success = success
            transfer_result.performance_improvement = performance_improvement
            
            # Update statistics
            if success:
                self.stats['successful_transfers'] += 1
            else:
                self.stats['failed_transfers'] += 1
            
            # Update transfer effectiveness
            total_transfers = self.stats['total_transfers']
            if total_transfers > 0:
                self.stats['transfer_effectiveness'] = (
                    self.stats['successful_transfers'] / total_transfers
                )
            
            # Update knowledge usage statistics
            for knowledge_id in transfer_result.transferred_items:
                if knowledge_id in self.transferable_knowledge:
                    knowledge = self.transferable_knowledge[knowledge_id]
                    if success:
                        knowledge.success_rate = min(1.0, knowledge.success_rate + 0.1)
                    else:
                        knowledge.success_rate = max(0.0, knowledge.success_rate - 0.05)
            
            # Store updated data
            if self.enable_database_storage:
                self._update_transfer_in_database(transfer_result)
            
            logger.info(f"Transfer {transfer_id} evaluated: success={success}, improvement={performance_improvement:.2f}")
            
        except Exception as e:
            logger.error(f"Error evaluating transfer success: {e}")
    
    def get_transfer_recommendations(self, target_game: str, 
                                   context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get knowledge transfer recommendations for a target game."""
        try:
            recommendations = []
            
            # Find similar games
            similar_games = self._find_similar_games(target_game, limit=5)
            
            for source_game, similarity in similar_games:
                # Get knowledge from source game
                source_knowledge = [
                    k for k in self.transferable_knowledge.values()
                    if k.source_game == source_game
                ]
                
                if not source_knowledge:
                    continue
                
                # Calculate transfer potential
                transfer_potential = self._calculate_transfer_potential(
                    source_knowledge, target_game, context
                )
                
                if transfer_potential > self.transfer_threshold:
                    recommendation = {
                        'source_game': source_game,
                        'similarity': similarity,
                        'transfer_potential': transfer_potential,
                        'knowledge_types': list(set(k.knowledge_type for k in source_knowledge)),
                        'recommended_items': len(source_knowledge),
                        'confidence': np.mean([k.confidence for k in source_knowledge]),
                        'success_rate': np.mean([k.success_rate for k in source_knowledge])
                    }
                    recommendations.append(recommendation)
            
            # Sort by transfer potential
            recommendations.sort(key=lambda x: x['transfer_potential'], reverse=True)
            
            return recommendations[:10]  # Return top 10 recommendations
            
        except Exception as e:
            logger.error(f"Error getting transfer recommendations: {e}")
            return []
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get comprehensive knowledge transfer statistics."""
        try:
            # Calculate knowledge type distribution
            type_distribution = defaultdict(int)
            for knowledge in self.transferable_knowledge.values():
                type_distribution[knowledge.knowledge_type.value] += 1
            
            # Calculate confidence distribution
            confidences = [k.confidence for k in self.transferable_knowledge.values()]
            confidence_stats = {
                'mean': np.mean(confidences) if confidences else 0.0,
                'std': np.std(confidences) if confidences else 0.0,
                'min': np.min(confidences) if confidences else 0.0,
                'max': np.max(confidences) if confidences else 0.0
            }
            
            # Calculate success rate distribution
            success_rates = [k.success_rate for k in self.transferable_knowledge.values()]
            success_rate_stats = {
                'mean': np.mean(success_rates) if success_rates else 0.0,
                'std': np.std(success_rates) if success_rates else 0.0,
                'min': np.min(success_rates) if success_rates else 0.0,
                'max': np.max(success_rates) if success_rates else 0.0
            }
            
            # Calculate transfer effectiveness by type
            transfer_effectiveness_by_type = defaultdict(list)
            for result in self.transfer_history:
                if result.success:
                    transfer_effectiveness_by_type[result.knowledge_type.value].append(
                        result.effectiveness
                    )
            
            type_effectiveness = {}
            for k_type, effectiveness_values in transfer_effectiveness_by_type.items():
                type_effectiveness[k_type] = {
                    'mean': np.mean(effectiveness_values),
                    'count': len(effectiveness_values)
                }
            
            return {
                'total_knowledge_items': len(self.transferable_knowledge),
                'total_transfers': self.stats['total_transfers'],
                'successful_transfers': self.stats['successful_transfers'],
                'failed_transfers': self.stats['failed_transfers'],
                'transfer_effectiveness': self.stats['transfer_effectiveness'],
                'games_tracked': len(self.game_profiles),
                'type_distribution': dict(type_distribution),
                'confidence_stats': confidence_stats,
                'success_rate_stats': success_rate_stats,
                'type_effectiveness': type_effectiveness,
                'recent_transfers': [
                    {
                        'transfer_id': r.transfer_id,
                        'source_game': r.source_game,
                        'target_game': r.target_game,
                        'success': r.success,
                        'confidence': r.confidence,
                        'effectiveness': r.effectiveness,
                        'timestamp': r.timestamp
                    }
                    for r in self.transfer_history[-10:]
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting knowledge statistics: {e}")
            return {}
    
    def cleanup_old_knowledge(self, max_age_days: int = 30, 
                            min_usage_count: int = 1):
        """Clean up old and unused knowledge."""
        try:
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60
            
            items_to_remove = []
            
            for knowledge_id, knowledge in self.transferable_knowledge.items():
                # Check age
                age = current_time - knowledge.created_at
                if age > max_age_seconds:
                    # Check usage
                    if knowledge.usage_count < min_usage_count:
                        items_to_remove.append(knowledge_id)
                        continue
                
                # Check success rate
                if knowledge.success_rate < 0.2 and knowledge.usage_count > 5:
                    items_to_remove.append(knowledge_id)
            
            # Remove items
            for knowledge_id in items_to_remove:
                del self.transferable_knowledge[knowledge_id]
                self.stats['knowledge_items'] -= 1
            
            # Update last cleanup time
            self.stats['last_cleanup'] = current_time
            
            logger.info(f"Cleaned up {len(items_to_remove)} old knowledge items")
            
            # Store updated data
            if self.enable_database_storage:
                self._cleanup_database_knowledge(items_to_remove)
            
        except Exception as e:
            logger.error(f"Error cleaning up old knowledge: {e}")
    
    def _generate_knowledge_id(self, source_game: str, knowledge_type: TransferType,
                              content: Dict[str, Any]) -> str:
        """Generate unique knowledge ID."""
        content_str = json.dumps(content, sort_keys=True)
        hash_input = f"{source_game}:{knowledge_type.value}:{content_str}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _generate_transfer_id(self, source_game: str, target_game: str) -> str:
        """Generate unique transfer ID."""
        timestamp = str(int(time.time() * 1000))
        hash_input = f"{source_game}:{target_game}:{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _update_game_profile(self, game_id: str, knowledge: TransferableKnowledge):
        """Update game similarity profile."""
        if game_id not in self.game_profiles:
            self.game_profiles[game_id] = GameSimilarityProfile(
                game_id=game_id,
                visual_features={},
                spatial_features={},
                action_patterns=[],
                success_patterns=[],
                coordinate_zones=[],
                complexity_score=0.0,
                pattern_types=set(),
                last_updated=time.time()
            )
        
        profile = self.game_profiles[game_id]
        profile.pattern_types.add(knowledge.knowledge_type)
        profile.last_updated = time.time()
        
        # Update features based on knowledge type
        if knowledge.knowledge_type == TransferType.VISUAL_PATTERN:
            profile.visual_features.update(knowledge.context_features.get('visual', {}))
        elif knowledge.knowledge_type == TransferType.SPATIAL_REASONING:
            profile.spatial_features.update(knowledge.context_features.get('spatial', {}))
        elif knowledge.knowledge_type == TransferType.ACTION_SEQUENCE:
            profile.action_patterns.extend(knowledge.content.get('actions', []))
        elif knowledge.knowledge_type == TransferType.COORDINATE:
            profile.coordinate_zones.extend(knowledge.content.get('zones', []))
        elif knowledge.knowledge_type == TransferType.PATTERN:
            # For general patterns, update both visual and spatial features if available
            profile.visual_features.update(knowledge.context_features.get('visual', {}))
            profile.spatial_features.update(knowledge.context_features.get('spatial', {}))
            # Also add to action patterns if actions are present
            if 'actions' in knowledge.content:
                profile.action_patterns.extend(knowledge.content.get('actions', []))
    
    def _calculate_game_similarity(self, game1: str, game2: str) -> float:
        """Calculate similarity between two games."""
        try:
            # Check cache
            cache_key = tuple(sorted([game1, game2]))
            if cache_key in self.similarity_cache:
                return self.similarity_cache[cache_key]
            
            if game1 not in self.game_profiles or game2 not in self.game_profiles:
                # If one game doesn't have a profile, return a default similarity
                # This allows transfers to work even when target game is new
                return 0.5  # Default moderate similarity
            
            profile1 = self.game_profiles[game1]
            profile2 = self.game_profiles[game2]
            
            # Calculate pattern type similarity
            type_similarity = len(profile1.pattern_types & profile2.pattern_types) / max(
                len(profile1.pattern_types | profile2.pattern_types), 1
            )
            
            # Calculate visual feature similarity
            visual_similarity = self._calculate_feature_similarity(
                profile1.visual_features, profile2.visual_features
            )
            
            # Calculate spatial feature similarity
            spatial_similarity = self._calculate_feature_similarity(
                profile1.spatial_features, profile2.spatial_features
            )
            
            # Calculate action pattern similarity
            action_similarity = self._calculate_list_similarity(
                profile1.action_patterns, profile2.action_patterns
            )
            
            # Calculate coordinate zone similarity
            coordinate_similarity = self._calculate_coordinate_similarity(
                profile1.coordinate_zones, profile2.coordinate_zones
            )
            
            # Weighted average
            similarity = (
                type_similarity * 0.3 +
                visual_similarity * 0.25 +
                spatial_similarity * 0.25 +
                action_similarity * 0.1 +
                coordinate_similarity * 0.1
            )
            
            # Cache result
            self.similarity_cache[cache_key] = similarity
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating game similarity: {e}")
            return 0.0
    
    def _calculate_feature_similarity(self, features1: Dict[str, Any], 
                                    features2: Dict[str, Any]) -> float:
        """Calculate similarity between feature dictionaries."""
        if not features1 or not features2:
            return 0.0
        
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1 = features1[key]
            val2 = features2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    similarity = 1.0 if val1 == val2 else 0.0
                else:
                    similarity = 1.0 - abs(val1 - val2) / max_val
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity (simple)
                similarity = 1.0 if val1 == val2 else 0.0
            else:
                # Default similarity
                similarity = 1.0 if val1 == val2 else 0.0
            
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_list_similarity(self, list1: List[str], list2: List[str]) -> float:
        """Calculate similarity between two lists."""
        if not list1 or not list2:
            return 0.0
        
        set1 = set(list1)
        set2 = set(list2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_coordinate_similarity(self, zones1: List[Tuple[int, int]], 
                                       zones2: List[Tuple[int, int]]) -> float:
        """Calculate similarity between coordinate zones."""
        if not zones1 or not zones2:
            return 0.0
        
        # Convert to sets for comparison
        set1 = set(zones1)
        set2 = set(zones2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _find_applicable_knowledge(self, source_game: str, target_game: str,
                                 knowledge_types: List[TransferType] = None,
                                 min_confidence: float = None) -> List[TransferableKnowledge]:
        """Find knowledge applicable for transfer."""
        try:
            applicable = []
            
            for knowledge in self.transferable_knowledge.values():
                if knowledge.source_game != source_game:
                    continue
                
                # Check knowledge type filter
                if knowledge_types and knowledge.knowledge_type not in knowledge_types:
                    continue
                
                # Check confidence filter
                if min_confidence and knowledge.confidence < min_confidence:
                    continue
                
                # Check success rate
                if knowledge.success_rate < 0.3:
                    continue
                
                # Check if knowledge is recent enough
                age_days = (time.time() - knowledge.created_at) / (24 * 60 * 60)
                if age_days > 30:  # Skip very old knowledge
                    continue
                
                applicable.append(knowledge)
            
            # Sort by confidence and success rate
            applicable.sort(key=lambda k: k.confidence * k.success_rate, reverse=True)
            
            return applicable[:20]  # Limit to top 20 items
            
        except Exception as e:
            logger.error(f"Error finding applicable knowledge: {e}")
            return []
    
    def _adapt_knowledge_to_context(self, knowledge: TransferableKnowledge,
                                  target_game: str) -> Dict[str, Any]:
        """Adapt knowledge to target game context."""
        try:
            adapted_content = knowledge.content.copy()
            
            # Get target game profile
            target_profile = self.game_profiles.get(target_game)
            if not target_profile:
                return adapted_content
            
            # Adapt based on knowledge type
            if knowledge.knowledge_type == TransferType.COORDINATE:
                # Adapt coordinate zones to target game
                adapted_content = self._adapt_coordinate_zones(
                    adapted_content, target_profile
                )
            elif knowledge.knowledge_type == TransferType.ACTION_SEQUENCE:
                # Adapt action sequences to target game
                adapted_content = self._adapt_action_sequences(
                    adapted_content, target_profile
                )
            elif knowledge.knowledge_type == TransferType.VISUAL_PATTERN:
                # Adapt visual patterns to target game
                adapted_content = self._adapt_visual_patterns(
                    adapted_content, target_profile
                )
            
            return adapted_content
            
        except Exception as e:
            logger.error(f"Error adapting knowledge to context: {e}")
            return knowledge.content
    
    def _adapt_coordinate_zones(self, content: Dict[str, Any], 
                               target_profile: GameSimilarityProfile) -> Dict[str, Any]:
        """Adapt coordinate zones to target game."""
        adapted = content.copy()
        
        if 'zones' in adapted and target_profile.coordinate_zones:
            # Scale zones based on target game's coordinate patterns
            target_zones = target_profile.coordinate_zones
            if target_zones:
                # Calculate scaling factor
                target_avg = np.mean([z[0] + z[1] for z in target_zones])
                source_zones = adapted['zones']
                if source_zones:
                    source_avg = np.mean([z[0] + z[1] for z in source_zones])
                    if source_avg > 0:
                        scale_factor = target_avg / source_avg
                        adapted['zones'] = [
                            (int(z[0] * scale_factor), int(z[1] * scale_factor))
                            for z in source_zones
                        ]
        
        return adapted
    
    def _adapt_action_sequences(self, content: Dict[str, Any],
                              target_profile: GameSimilarityProfile) -> Dict[str, Any]:
        """Adapt action sequences to target game."""
        adapted = content.copy()
        
        if 'actions' in adapted and target_profile.action_patterns:
            # Filter actions to those available in target game
            target_actions = set(target_profile.action_patterns)
            adapted['actions'] = [
                action for action in adapted['actions']
                if action in target_actions
            ]
        
        return adapted
    
    def _adapt_visual_patterns(self, content: Dict[str, Any],
                             target_profile: GameSimilarityProfile) -> Dict[str, Any]:
        """Adapt visual patterns to target game."""
        adapted = content.copy()
        
        # Visual patterns are generally more transferable
        # Just add adaptation metadata
        adapted['adapted_for'] = target_profile.game_id
        adapted['adaptation_timestamp'] = time.time()
        
        return adapted
    
    def _generate_adaptation_notes(self, knowledge: TransferableKnowledge,
                                 target_game: str) -> List[str]:
        """Generate adaptation notes for knowledge transfer."""
        notes = []
        
        # Check confidence level
        if knowledge.confidence < 0.5:
            notes.append(f"Low confidence knowledge ({knowledge.confidence:.2f}) - test carefully")
        
        # Check success rate
        if knowledge.success_rate < 0.6:
            notes.append(f"Moderate success rate ({knowledge.success_rate:.2f}) - monitor effectiveness")
        
        # Check usage count
        if knowledge.usage_count < 3:
            notes.append("Limited usage history - may need validation")
        
        # Check age
        age_days = (time.time() - knowledge.created_at) / (24 * 60 * 60)
        if age_days > 7:
            notes.append(f"Knowledge is {age_days:.1f} days old - verify relevance")
        
        return notes
    
    def _calculate_transfer_effectiveness(self, knowledge_items: List[TransferableKnowledge],
                                        target_game: str) -> float:
        """Calculate expected effectiveness of knowledge transfer."""
        try:
            if not knowledge_items:
                return 0.0
            
            # Base effectiveness on knowledge quality
            avg_confidence = np.mean([k.confidence for k in knowledge_items])
            avg_success_rate = np.mean([k.success_rate for k in knowledge_items])
            
            # Weight by knowledge type relevance
            type_weights = {
                TransferType.PATTERN: 0.9,
                TransferType.STRATEGY: 0.8,
                TransferType.ACTION_SEQUENCE: 0.7,
                TransferType.COORDINATE: 0.6,
                TransferType.VISUAL_PATTERN: 0.8,
                TransferType.SPATIAL_REASONING: 0.7,
                TransferType.LOGICAL_REASONING: 0.8
            }
            
            type_weight = np.mean([
                type_weights.get(k.knowledge_type, 0.5) for k in knowledge_items
            ])
            
            # Calculate final effectiveness
            effectiveness = (avg_confidence * 0.4 + avg_success_rate * 0.4 + type_weight * 0.2)
            
            return min(1.0, effectiveness)
            
        except Exception as e:
            logger.error(f"Error calculating transfer effectiveness: {e}")
            return 0.0
    
    def _find_similar_games(self, target_game: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Find games similar to target game."""
        try:
            similarities = []
            
            for game_id in self.game_profiles:
                if game_id == target_game:
                    continue
                
                similarity = self._calculate_game_similarity(target_game, game_id)
                similarities.append((game_id, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar games: {e}")
            return []
    
    def _calculate_transfer_potential(self, knowledge_items: List[TransferableKnowledge],
                                    target_game: str, context: Dict[str, Any] = None) -> float:
        """Calculate transfer potential for knowledge items."""
        try:
            if not knowledge_items:
                return 0.0
            
            # Base potential on knowledge quality
            avg_confidence = np.mean([k.confidence for k in knowledge_items])
            avg_success_rate = np.mean([k.success_rate for k in knowledge_items])
            
            # Factor in usage count (more used = more reliable)
            avg_usage = np.mean([k.usage_count for k in knowledge_items])
            usage_factor = min(1.0, avg_usage / 10.0)  # Normalize to 0-1
            
            # Factor in recency
            current_time = time.time()
            avg_age = np.mean([current_time - k.last_used for k in knowledge_items])
            recency_factor = max(0.0, 1.0 - (avg_age / (7 * 24 * 60 * 60)))  # Decay over 7 days
            
            # Calculate transfer potential
            potential = (
                avg_confidence * 0.4 +
                avg_success_rate * 0.3 +
                usage_factor * 0.2 +
                recency_factor * 0.1
            )
            
            return min(1.0, potential)
            
        except Exception as e:
            logger.error(f"Error calculating transfer potential: {e}")
            return 0.0
    
    def _load_persistent_data(self):
        """Load persistent data from database."""
        try:
            if not self.enable_database_storage:
                return
            
            # Load knowledge items
            # This would typically query the database for stored knowledge
            # For now, we'll start with empty state
            logger.info("Loading persistent knowledge transfer data from database")
            
        except Exception as e:
            logger.error(f"Error loading persistent data: {e}")
    
    def _store_knowledge_in_database(self, knowledge: TransferableKnowledge):
        """Store knowledge in database."""
        try:
            if not self.enable_database_storage:
                return
            
            # Store knowledge data
            knowledge_data = {
                'knowledge_id': knowledge.knowledge_id,
                'source_game': knowledge.source_game,
                'knowledge_type': knowledge.knowledge_type.value,
                'content': knowledge.content,
                'confidence': knowledge.confidence,
                'success_rate': knowledge.success_rate,
                'usage_count': knowledge.usage_count,
                'last_used': knowledge.last_used,
                'created_at': knowledge.created_at,
                'tags': knowledge.tags,
                'context_features': knowledge.context_features,
                'transfer_history': knowledge.transfer_history
            }
            
            # Log to database
            self.integration.log_system_event(
                self.LogLevel.INFO,
                self.Component.LEARNING_LOOP,
                f"Stored knowledge transfer item: {knowledge.knowledge_id}",
                knowledge_data,
                knowledge.source_game
            )
            
        except Exception as e:
            logger.error(f"Error storing knowledge in database: {e}")
    
    def _store_transfer_in_database(self, transfer_result: TransferResult):
        """Store transfer result in database."""
        try:
            if not self.enable_database_storage:
                return
            
            # Store transfer data
            transfer_data = {
                'transfer_id': transfer_result.transfer_id,
                'source_game': transfer_result.source_game,
                'target_game': transfer_result.target_game,
                'knowledge_type': transfer_result.knowledge_type.value,
                'transferred_items': transfer_result.transferred_items,
                'confidence': transfer_result.confidence,
                'effectiveness': transfer_result.effectiveness,
                'adaptation_notes': transfer_result.adaptation_notes,
                'timestamp': transfer_result.timestamp,
                'success': transfer_result.success,
                'performance_improvement': transfer_result.performance_improvement
            }
            
            # Log to database
            self.integration.log_system_event(
                self.LogLevel.INFO,
                self.Component.LEARNING_LOOP,
                f"Stored knowledge transfer: {transfer_result.transfer_id}",
                transfer_data,
                transfer_result.target_game
            )
            
        except Exception as e:
            logger.error(f"Error storing transfer in database: {e}")
    
    def _update_transfer_in_database(self, transfer_result: TransferResult):
        """Update transfer result in database."""
        try:
            if not self.enable_database_storage:
                return
            
            # Update transfer data
            transfer_data = {
                'transfer_id': transfer_result.transfer_id,
                'success': transfer_result.success,
                'performance_improvement': transfer_result.performance_improvement,
                'updated_at': time.time()
            }
            
            # Log to database
            self.integration.log_system_event(
                self.LogLevel.INFO,
                self.Component.LEARNING_LOOP,
                f"Updated knowledge transfer: {transfer_result.transfer_id}",
                transfer_data,
                transfer_result.target_game
            )
            
        except Exception as e:
            logger.error(f"Error updating transfer in database: {e}")
    
    def _cleanup_database_knowledge(self, knowledge_ids: List[str]):
        """Clean up knowledge items from database."""
        try:
            if not self.enable_database_storage:
                return
            
            # Log cleanup
            self.integration.log_system_event(
                self.LogLevel.INFO,
                self.Component.LEARNING_LOOP,
                f"Cleaned up {len(knowledge_ids)} knowledge items",
                {'removed_items': knowledge_ids},
                'knowledge_cleanup'
            )
            
        except Exception as e:
            logger.error(f"Error cleaning up database knowledge: {e}")


def create_enhanced_knowledge_transfer(persistence_dir: Optional[Path] = None,
                                     transfer_threshold: float = 0.6,
                                     enable_database_storage: bool = True) -> EnhancedKnowledgeTransfer:
    """Create an enhanced knowledge transfer system."""
    return EnhancedKnowledgeTransfer(
        persistence_dir=persistence_dir,
        transfer_threshold=transfer_threshold,
        enable_database_storage=enable_database_storage
    )
