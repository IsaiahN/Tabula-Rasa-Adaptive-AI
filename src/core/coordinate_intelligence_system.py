"""
Enhanced Coordinate Intelligence System with Success Zone Mapping.

This system provides advanced coordinate intelligence capabilities including:
1. Success zone mapping and clustering
2. Cross-game coordinate learning
3. Dynamic success zone expansion
4. Intelligent coordinate recommendation
5. Failure pattern avoidance
6. Real-time success zone updates
"""

import time
import math
import logging
import asyncio
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque

from src.core.penalty_decay_system import get_penalty_decay_system
import json

logger = logging.getLogger(__name__)


class ZoneType(Enum):
    """Types of coordinate zones."""
    SUCCESS = "success"
    FAILURE = "failure"
    NEUTRAL = "neutral"
    BOUNDARY = "boundary"
    UNKNOWN = "unknown"


class ZoneConfidence(Enum):
    """Confidence levels for zone classification."""
    HIGH = "high"      # >80% success rate with >10 attempts
    MEDIUM = "medium"  # 60-80% success rate with >5 attempts
    LOW = "low"        # 40-60% success rate with >3 attempts
    UNKNOWN = "unknown" # <3 attempts or <40% success rate


@dataclass
class CoordinateZone:
    """Represents a coordinate zone with success/failure patterns."""
    zone_id: str
    center: Tuple[int, int]
    coordinates: Set[Tuple[int, int]]
    zone_type: ZoneType
    confidence: ZoneConfidence
    success_rate: float
    total_attempts: int
    successful_actions: Set[int]
    failure_actions: Set[int]
    last_updated: float
    expansion_radius: int = 1
    stability_score: float = 0.0
    cross_game_success: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinateIntelligence:
    """Enhanced coordinate intelligence data."""
    game_id: str
    x: int
    y: int
    attempts: int = 0
    successes: int = 0
    success_rate: float = 0.0
    frame_changes: int = 0
    last_used: float = 0.0
    action_success_rates: Dict[int, float] = field(default_factory=dict)
    zone_id: Optional[str] = None
    confidence_score: float = 0.0
    cross_game_references: int = 0


class SuccessZoneMapper:
    """
    Maps and manages success zones for coordinate intelligence.
    """
    
    def __init__(self, min_zone_size: int = 3, max_zone_radius: int = 5):
        self.min_zone_size = min_zone_size
        self.max_zone_radius = max_zone_radius
        
        # Zone storage: game_id -> {zone_id: CoordinateZone}
        self.zones: Dict[str, Dict[str, CoordinateZone]] = defaultdict(dict)
        
        # Coordinate to zone mapping: game_id -> {(x,y): zone_id}
        self.coordinate_to_zone: Dict[str, Dict[Tuple[int, int], str]] = defaultdict(dict)
        
        # Cross-game zone learning
        self.universal_zones: Dict[str, CoordinateZone] = {}
        self.cross_game_patterns: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(dict)
        
        # Zone clustering and merging
        self.zone_clusters: Dict[str, List[str]] = defaultdict(list)
        self.merged_zones: Dict[str, str] = {}
        
        # Performance tracking
        self.zone_creation_count = 0
        self.zone_merge_count = 0
        self.zone_expansion_count = 0
    
    def update_coordinate_intelligence(
        self, 
        game_id: str, 
        x: int, 
        y: int, 
        action_id: int,
        success: bool, 
        frame_changes: int = 0
    ) -> Optional[str]:
        """
        Update coordinate intelligence and manage success zones.
        
        Args:
            game_id: Game identifier
            x, y: Coordinate position
            action_id: Action performed
            success: Whether the action was successful
            frame_changes: Number of frame changes
            
        Returns:
            Zone ID if coordinate was assigned to a zone, None otherwise
        """
        # Update coordinate intelligence
        coord_key = (x, y)
        
        # Get or create coordinate intelligence
        if coord_key not in self.coordinate_to_zone[game_id]:
            # New coordinate - check if it should be added to existing zone
            zone_id = self._find_nearby_zone(game_id, x, y, action_id, success)
            if zone_id:
                self._add_coordinate_to_zone(game_id, x, y, zone_id)
                return zone_id
            else:
                # Create new zone if conditions are met
                zone_id = self._create_new_zone(game_id, x, y, action_id, success)
                if zone_id:
                    return zone_id
        else:
            # Update existing coordinate
            zone_id = self.coordinate_to_zone[game_id][coord_key]
            self._update_zone_with_result(game_id, zone_id, action_id, success, frame_changes)
            return zone_id
        
        return None
    
    def _find_nearby_zone(
        self, 
        game_id: str, 
        x: int, 
        y: int, 
        action_id: int, 
        success: bool
    ) -> Optional[str]:
        """Find a nearby zone that this coordinate could belong to."""
        zones = self.zones[game_id]
        
        for zone_id, zone in zones.items():
            # Check if coordinate is within expansion radius
            distance = math.sqrt((x - zone.center[0])**2 + (y - zone.center[1])**2)
            
            if distance <= zone.expansion_radius:
                # Check if action compatibility
                if success and action_id in zone.successful_actions:
                    return zone_id
                elif not success and action_id in zone.failure_actions:
                    return zone_id
                elif zone.zone_type == ZoneType.NEUTRAL:
                    return zone_id
        
        return None
    
    def _create_new_zone(
        self, 
        game_id: str, 
        x: int, 
        y: int, 
        action_id: int, 
        success: bool
    ) -> Optional[str]:
        """Create a new zone for the coordinate."""
        # Only create zones for successful actions or after multiple attempts
        if not success:
            return None
        
        zone_id = f"{game_id}_zone_{self.zone_creation_count}"
        self.zone_creation_count += 1
        
        # Determine zone type and confidence
        zone_type = ZoneType.SUCCESS if success else ZoneType.FAILURE
        confidence = ZoneConfidence.LOW  # New zones start with low confidence
        
        # Create zone
        zone = CoordinateZone(
            zone_id=zone_id,
            center=(x, y),
            coordinates={(x, y)},
            zone_type=zone_type,
            confidence=confidence,
            success_rate=1.0 if success else 0.0,
            total_attempts=1,
            successful_actions={action_id} if success else set(),
            failure_actions=set() if success else {action_id},
            last_updated=time.time(),
            expansion_radius=1
        )
        
        # Add to storage
        self.zones[game_id][zone_id] = zone
        self.coordinate_to_zone[game_id][(x, y)] = zone_id
        
        logger.info(f"Created new {zone_type.value} zone {zone_id} at ({x}, {y})")
        
        return zone_id
    
    def _add_coordinate_to_zone(
        self, 
        game_id: str, 
        x: int, 
        y: int, 
        zone_id: str
    ):
        """Add a coordinate to an existing zone."""
        zone = self.zones[game_id][zone_id]
        zone.coordinates.add((x, y))
        self.coordinate_to_zone[game_id][(x, y)] = zone_id
        
        # Update zone center (weighted average)
        if len(zone.coordinates) > 1:
            center_x = sum(coord[0] for coord in zone.coordinates) / len(zone.coordinates)
            center_y = sum(coord[1] for coord in zone.coordinates) / len(zone.coordinates)
            zone.center = (int(center_x), int(center_y))
        
        # Expand zone if needed
        self._expand_zone_if_needed(game_id, zone_id)
    
    def _update_zone_with_result(
        self, 
        game_id: str, 
        zone_id: str, 
        action_id: int, 
        success: bool, 
        frame_changes: int
    ):
        """Update a zone with new action result."""
        zone = self.zones[game_id][zone_id]
        
        # Update statistics
        zone.total_attempts += 1
        if success:
            zone.successes += 1
            zone.successful_actions.add(action_id)
        else:
            zone.failure_actions.add(action_id)
        
        # Safe division to prevent ZeroDivisionError
        if zone.total_attempts > 0:
            zone.success_rate = zone.successes / zone.total_attempts
        else:
            # Handle edge case where total_attempts is 0
            zone.success_rate = 0.0
            zone.total_attempts = 1  # Fix inconsistent state
            logger.warning(f"Fixed zone {zone_id} with zero total_attempts")

        zone.last_updated = time.time()
        
        # Update confidence based on success rate and attempts
        if zone.total_attempts >= 10 and zone.success_rate >= 0.8:
            zone.confidence = ZoneConfidence.HIGH
        elif zone.total_attempts >= 5 and zone.success_rate >= 0.6:
            zone.confidence = ZoneConfidence.MEDIUM
        elif zone.total_attempts >= 3 and zone.success_rate >= 0.4:
            zone.confidence = ZoneConfidence.LOW
        else:
            zone.confidence = ZoneConfidence.UNKNOWN
        
        # Update zone type based on success rate
        if zone.success_rate >= 0.7:
            zone.zone_type = ZoneType.SUCCESS
        elif zone.success_rate <= 0.3:
            zone.zone_type = ZoneType.FAILURE
        else:
            zone.zone_type = ZoneType.NEUTRAL
        
        # Update stability score
        zone.stability_score = self._calculate_stability_score(zone)
        
        # Check for zone merging opportunities
        self._check_zone_merging(game_id, zone_id)
    
    def _expand_zone_if_needed(self, game_id: str, zone_id: str):
        """Expand zone if it meets expansion criteria."""
        zone = self.zones[game_id][zone_id]
        
        # Only expand successful zones with high confidence
        if (zone.zone_type == ZoneType.SUCCESS and 
            zone.confidence in [ZoneConfidence.HIGH, ZoneConfidence.MEDIUM] and
            zone.expansion_radius < self.max_zone_radius):
            
            zone.expansion_radius += 1
            self.zone_expansion_count += 1
            
            logger.debug(f"Expanded zone {zone_id} to radius {zone.expansion_radius}")
    
    def _calculate_stability_score(self, zone: CoordinateZone) -> float:
        """Calculate stability score for a zone."""
        # Base score from success rate
        base_score = zone.success_rate
        
        # Bonus for more attempts (more data = more stable)
        attempt_bonus = min(0.2, zone.total_attempts / 50.0)
        
        # Bonus for consistent action success
        action_consistency = len(zone.successful_actions) / max(1, len(zone.successful_actions | zone.failure_actions))
        consistency_bonus = action_consistency * 0.3
        
        # Penalty for recent changes
        time_since_update = time.time() - zone.last_updated
        recency_penalty = 0.1 if time_since_update < 60 else 0.0  # Penalty for very recent updates
        
        return min(1.0, base_score + attempt_bonus + consistency_bonus - recency_penalty)
    
    def _check_zone_merging(self, game_id: str, zone_id: str):
        """Check if zone should be merged with nearby zones."""
        zone = self.zones[game_id][zone_id]
        
        # Look for nearby zones of the same type
        for other_zone_id, other_zone in self.zones[game_id].items():
            if (other_zone_id != zone_id and 
                other_zone.zone_type == zone.zone_type and
                other_zone.confidence == zone.confidence):
                
                # Check distance between zone centers
                distance = math.sqrt(
                    (zone.center[0] - other_zone.center[0])**2 + 
                    (zone.center[1] - other_zone.center[1])**2
                )
                
                # Merge if zones are close enough
                if distance <= (zone.expansion_radius + other_zone.expansion_radius):
                    self._merge_zones(game_id, zone_id, other_zone_id)
                    break
    
    def _merge_zones(self, game_id: str, zone1_id: str, zone2_id: str):
        """Merge two zones into one."""
        zone1 = self.zones[game_id][zone1_id]
        zone2 = self.zones[game_id][zone2_id]
        
        # Create merged zone
        merged_coordinates = zone1.coordinates | zone2.coordinates
        if len(merged_coordinates) > 0:
            merged_center_x = sum(coord[0] for coord in merged_coordinates) / len(merged_coordinates)
            merged_center_y = sum(coord[1] for coord in merged_coordinates) / len(merged_coordinates)
        else:
            # Fallback to zone1's center if no coordinates (should not happen)
            merged_center_x, merged_center_y = zone1.center
            logger.warning(f"Merged zone has no coordinates, using zone1 center: {zone1.center}")
        
        # Calculate merged statistics
        merged_attempts = zone1.total_attempts + zone2.total_attempts
        merged_successes = zone1.successes + zone2.successes
        merged_success_rate = merged_successes / merged_attempts if merged_attempts > 0 else 0.0
        
        # Update zone1 with merged data
        zone1.coordinates = merged_coordinates
        zone1.center = (int(merged_center_x), int(merged_center_y))
        zone1.total_attempts = merged_attempts
        zone1.successes = merged_successes
        zone1.success_rate = merged_success_rate
        zone1.successful_actions = zone1.successful_actions | zone2.successful_actions
        zone1.failure_actions = zone1.failure_actions | zone2.failure_actions
        zone1.last_updated = time.time()
        
        # Update coordinate mappings
        for coord in zone2.coordinates:
            self.coordinate_to_zone[game_id][coord] = zone1_id
        
        # Remove zone2
        del self.zones[game_id][zone2_id]
        self.merged_zones[zone2_id] = zone1_id
        self.zone_merge_count += 1
        
        logger.info(f"Merged zones {zone1_id} and {zone2_id} into {zone1_id}")
    
    def get_recommended_coordinates(
        self, 
        game_id: str, 
        action_id: int, 
        grid_dims: Tuple[int, int],
        preference: str = "success"
    ) -> List[Tuple[int, int]]:
        """
        Get recommended coordinates based on success zone mapping.
        
        Args:
            game_id: Game identifier
            action_id: Action to perform
            grid_dims: Grid dimensions (width, height)
            preference: "success", "exploration", or "mixed"
            
        Returns:
            List of recommended coordinates ordered by preference
        """
        recommendations = []
        zones = self.zones[game_id]
        
        if preference == "success":
            # Prioritize high-confidence success zones
            success_zones = [
                zone for zone in zones.values() 
                if zone.zone_type == ZoneType.SUCCESS and 
                   action_id in zone.successful_actions
            ]
            success_zones.sort(key=lambda z: (z.confidence.value, z.success_rate), reverse=True)
            
            for zone in success_zones:
                # Get coordinates from this zone
                zone_coords = list(zone.coordinates)
                zone_coords.sort(key=lambda c: math.sqrt((c[0] - zone.center[0])**2 + (c[1] - zone.center[1])**2))
                recommendations.extend(zone_coords[:5])  # Top 5 from each zone
        
        elif preference == "exploration":
            # Prioritize unexplored areas
            explored_coords = set()
            for zone in zones.values():
                explored_coords.update(zone.coordinates)
            
            # Find unexplored coordinates
            grid_width, grid_height = grid_dims
            unexplored = []
            for x in range(grid_width):
                for y in range(grid_height):
                    if (x, y) not in explored_coords:
                        unexplored.append((x, y))
            
            # Sort by distance from center (explore from center outward)
            center_x, center_y = grid_width // 2, grid_height // 2
            unexplored.sort(key=lambda c: math.sqrt((c[0] - center_x)**2 + (c[1] - center_y)**2))
            recommendations.extend(unexplored[:20])  # Top 20 unexplored
        
        else:  # mixed
            # Combine success zones and exploration
            success_recs = self.get_recommended_coordinates(game_id, action_id, grid_dims, "success")
            exploration_recs = self.get_recommended_coordinates(game_id, action_id, grid_dims, "exploration")
            
            # Interleave recommendations
            max_len = max(len(success_recs), len(exploration_recs))
            for i in range(max_len):
                if i < len(success_recs):
                    recommendations.append(success_recs[i])
                if i < len(exploration_recs):
                    recommendations.append(exploration_recs[i])
        
        return recommendations[:50]  # Limit to 50 recommendations
    
    def get_zone_statistics(self, game_id: str) -> Dict[str, Any]:
        """Get statistics about zones for a game."""
        zones = self.zones[game_id]
        
        if not zones:
            return {
                'total_zones': 0,
                'success_zones': 0,
                'failure_zones': 0,
                'neutral_zones': 0,
                'total_coordinates': 0,
                'avg_success_rate': 0.0,
                'high_confidence_zones': 0
            }
        
        success_zones = sum(1 for z in zones.values() if z.zone_type == ZoneType.SUCCESS)
        failure_zones = sum(1 for z in zones.values() if z.zone_type == ZoneType.FAILURE)
        neutral_zones = sum(1 for z in zones.values() if z.zone_type == ZoneType.NEUTRAL)
        total_coordinates = sum(len(z.coordinates) for z in zones.values())
        avg_success_rate = sum(z.success_rate for z in zones.values()) / len(zones) if len(zones) > 0 else 0.0
        high_confidence_zones = sum(1 for z in zones.values() if z.confidence == ZoneConfidence.HIGH)
        
        return {
            'total_zones': len(zones),
            'success_zones': success_zones,
            'failure_zones': failure_zones,
            'neutral_zones': neutral_zones,
            'total_coordinates': total_coordinates,
            'avg_success_rate': avg_success_rate,
            'high_confidence_zones': high_confidence_zones,
            'zone_creation_count': self.zone_creation_count,
            'zone_merge_count': self.zone_merge_count,
            'zone_expansion_count': self.zone_expansion_count
        }
    
    def export_zones(self, game_id: str) -> Dict[str, Any]:
        """Export zone data for persistence."""
        zones_data = {}
        for zone_id, zone in self.zones[game_id].items():
            zones_data[zone_id] = {
                'center': zone.center,
                'coordinates': list(zone.coordinates),
                'zone_type': zone.zone_type.value,
                'confidence': zone.confidence.value,
                'success_rate': zone.success_rate,
                'total_attempts': zone.total_attempts,
                'successful_actions': list(zone.successful_actions),
                'failure_actions': list(zone.failure_actions),
                'last_updated': zone.last_updated,
                'expansion_radius': zone.expansion_radius,
                'stability_score': zone.stability_score,
                'metadata': zone.metadata
            }
        
        return {
            'game_id': game_id,
            'zones': zones_data,
            'coordinate_mappings': dict(self.coordinate_to_zone[game_id]),
            'statistics': self.get_zone_statistics(game_id)
        }
    
    def import_zones(self, zones_data: Dict[str, Any]):
        """Import zone data from persistence."""
        game_id = zones_data['game_id']
        
        # Clear existing data
        self.zones[game_id] = {}
        self.coordinate_to_zone[game_id] = {}
        
        # Import zones
        for zone_id, zone_data in zones_data['zones'].items():
            zone = CoordinateZone(
                zone_id=zone_id,
                center=tuple(zone_data['center']),
                coordinates=set(tuple(coord) for coord in zone_data['coordinates']),
                zone_type=ZoneType(zone_data['zone_type']),
                confidence=ZoneConfidence(zone_data['confidence']),
                success_rate=zone_data['success_rate'],
                total_attempts=zone_data['total_attempts'],
                successful_actions=set(zone_data['successful_actions']),
                failure_actions=set(zone_data['failure_actions']),
                last_updated=zone_data['last_updated'],
                expansion_radius=zone_data['expansion_radius'],
                stability_score=zone_data['stability_score'],
                metadata=zone_data['metadata']
            )
            
            self.zones[game_id][zone_id] = zone
            
            # Update coordinate mappings
            for coord in zone.coordinates:
                self.coordinate_to_zone[game_id][coord] = zone_id
        
        logger.info(f"Imported {len(self.zones[game_id])} zones for game {game_id}")


class CoordinateIntelligenceSystem:
    """
    Main coordinate intelligence system that integrates with the success zone mapper.
    """
    
    def __init__(self):
        self.zone_mapper = SuccessZoneMapper()
        self.coordinate_intelligence: Dict[Tuple[str, int, int], CoordinateIntelligence] = {}
        self.cross_game_learning = True
        self.learning_rate = 0.1
        
        # Performance tracking
        self.total_updates = 0
        self.successful_recommendations = 0
        self.failed_recommendations = 0
        
        # Initialize penalty decay system
        self.penalty_system = get_penalty_decay_system()
    
    async def update_coordinate_intelligence(
        self, 
        game_id: str, 
        x: int, 
        y: int, 
        action_id: int,
        success: bool, 
        frame_changes: int = 0,
        score_change: float = 0.0,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Update coordinate intelligence and success zone mapping with penalty integration.
        
        Returns:
            Dictionary with update results and recommendations
        """
        # Update zone mapping
        zone_id = self.zone_mapper.update_coordinate_intelligence(
            game_id, x, y, action_id, success, frame_changes
        )
        
        # Update coordinate intelligence
        coord_key = (game_id, x, y)
        if coord_key not in self.coordinate_intelligence:
            self.coordinate_intelligence[coord_key] = CoordinateIntelligence(
                game_id=game_id,
                x=x,
                y=y,
                last_used=time.time()
            )
        
        coord_intel = self.coordinate_intelligence[coord_key]
        coord_intel.attempts += 1
        if success:
            coord_intel.successes += 1
        # Safe division to prevent ZeroDivisionError
        coord_intel.success_rate = coord_intel.successes / coord_intel.attempts if coord_intel.attempts > 0 else 0.0
        coord_intel.last_used = time.time()
        coord_intel.frame_changes += frame_changes
        coord_intel.zone_id = zone_id
        
        # Update action-specific success rates
        if action_id not in coord_intel.action_success_rates:
            coord_intel.action_success_rates[action_id] = 0.0
        
        # Update with exponential moving average
        current_rate = coord_intel.action_success_rates[action_id]
        new_rate = 1.0 if success else 0.0
        coord_intel.action_success_rates[action_id] = (
            (1 - self.learning_rate) * current_rate + 
            self.learning_rate * new_rate
        )
        
        # Integrate with penalty decay system
        penalty_info = await self.penalty_system.record_coordinate_attempt(
            game_id=game_id,
            x=x,
            y=y,
            success=success,
            score_change=score_change,
            action_type=f"ACTION{action_id}",
            context=context or {}
        )
        
        # Update confidence score
        coord_intel.confidence_score = self._calculate_confidence_score(coord_intel)
        
        # Cross-game learning
        if self.cross_game_learning:
            self._update_cross_game_learning(game_id, x, y, action_id, success)
        
        # Save to database
        try:
            from src.database.system_integration import get_system_integration
            integration = get_system_integration()
            asyncio.create_task(integration.update_coordinate_intelligence(
                game_id=game_id,
                x=x,
                y=y,
                attempts=coord_intel.attempts,
                successes=coord_intel.successes,
                success_rate=coord_intel.success_rate,
                frame_changes=coord_intel.frame_changes
            ))
        except Exception as e:
            logger.warning(f"Failed to save coordinate intelligence to database: {e}")
        
        self.total_updates += 1
        
        return {
            'zone_id': zone_id,
            'success_rate': coord_intel.success_rate,
            'confidence_score': coord_intel.confidence_score,
            'action_success_rates': coord_intel.action_success_rates.copy(),
            'total_attempts': coord_intel.attempts
        }
    
    def get_recommended_coordinates(
        self, 
        game_id: str, 
        action_id: int, 
        grid_dims: Tuple[int, int],
        preference: str = "success",
        limit: int = 10
    ) -> List[Tuple[int, int]]:
        """Get recommended coordinates using success zone mapping."""
        recommendations = self.zone_mapper.get_recommended_coordinates(
            game_id, action_id, grid_dims, preference
        )
        
        # Limit recommendations
        return recommendations[:limit]
    
    def get_coordinate_intelligence(
        self, 
        game_id: str = None, 
        min_success_rate: float = 0.0
    ) -> List[CoordinateIntelligence]:
        """Get coordinate intelligence data."""
        results = []
        
        for coord_key, coord_intel in self.coordinate_intelligence.items():
            if game_id and coord_intel.game_id != game_id:
                continue
            
            if coord_intel.success_rate >= min_success_rate:
                results.append(coord_intel)
        
        # Sort by success rate and attempts
        results.sort(key=lambda c: (c.success_rate, c.attempts), reverse=True)
        
        return results
    
    def get_best_coordinates(
        self, 
        game_id: str = None, 
        action_id: int = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get best coordinates based on success rates."""
        intelligence = self.get_coordinate_intelligence(game_id)
        
        if action_id is not None:
            # Filter by action-specific success rate
            intelligence = [
                c for c in intelligence 
                if action_id in c.action_success_rates and 
                   c.action_success_rates[action_id] > 0.5
            ]
        
        results = []
        for coord_intel in intelligence[:limit]:
            results.append({
                'game_id': coord_intel.game_id,
                'x': coord_intel.x,
                'y': coord_intel.y,
                'success_rate': coord_intel.success_rate,
                'attempts': coord_intel.attempts,
                'confidence_score': coord_intel.confidence_score,
                'action_success_rates': coord_intel.action_success_rates,
                'zone_id': coord_intel.zone_id,
                'last_used': coord_intel.last_used
            })
        
        return results
    
    def _calculate_confidence_score(self, coord_intel: CoordinateIntelligence) -> float:
        """Calculate confidence score for coordinate intelligence."""
        # Base confidence from success rate
        base_confidence = coord_intel.success_rate
        
        # Bonus for more attempts (more data = more confident)
        attempt_bonus = min(0.3, coord_intel.attempts / 20.0)
        
        # Bonus for consistent action success
        if coord_intel.action_success_rates:
            avg_action_success = sum(coord_intel.action_success_rates.values()) / len(coord_intel.action_success_rates) if len(coord_intel.action_success_rates) > 0 else 0.0
            consistency_bonus = avg_action_success * 0.2
        else:
            consistency_bonus = 0.0
        
        # Cross-game bonus
        cross_game_bonus = min(0.2, coord_intel.cross_game_references / 5.0)
        
        return min(1.0, base_confidence + attempt_bonus + consistency_bonus + cross_game_bonus)
    
    def _update_cross_game_learning(self, game_id: str, x: int, y: int, action_id: int, success: bool):
        """Update cross-game learning patterns."""
        coord_key = (x, y)
        
        if coord_key not in self.zone_mapper.cross_game_patterns:
            self.zone_mapper.cross_game_patterns[coord_key] = {}
        
        # Update success rate for this coordinate across games
        if game_id not in self.zone_mapper.cross_game_patterns[coord_key]:
            self.zone_mapper.cross_game_patterns[coord_key][game_id] = 0.0
        
        current_rate = self.zone_mapper.cross_game_patterns[coord_key][game_id]
        new_rate = 1.0 if success else 0.0
        self.zone_mapper.cross_game_patterns[coord_key][game_id] = (
            (1 - self.learning_rate) * current_rate + 
            self.learning_rate * new_rate
        )
        
        # Update cross-game references
        coord_intel = self.coordinate_intelligence.get((game_id, x, y))
        if coord_intel:
            coord_intel.cross_game_references = len(self.zone_mapper.cross_game_patterns[coord_key])
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        zone_stats = {}
        for game_id in self.zone_mapper.zones:
            zone_stats[game_id] = self.zone_mapper.get_zone_statistics(game_id)
        
        return {
            'total_coordinates_tracked': len(self.coordinate_intelligence),
            'total_updates': self.total_updates,
            'successful_recommendations': self.successful_recommendations,
            'failed_recommendations': self.failed_recommendations,
            'zone_statistics': zone_stats,
            'cross_game_patterns': len(self.zone_mapper.cross_game_patterns),
            'learning_rate': self.learning_rate,
            'cross_game_learning_enabled': self.cross_game_learning
        }
    
    async def get_coordinate_avoidance_scores(
        self, 
        game_id: str, 
        candidate_coordinates: List[Tuple[int, int]]
    ) -> Dict[Tuple[int, int], float]:
        """
        Get avoidance scores for candidate coordinates based on penalties and diversity.
        
        Args:
            game_id: Game identifier
            candidate_coordinates: List of (x, y) coordinate tuples
            
        Returns:
            Dictionary mapping coordinates to avoidance scores (higher = more avoidable)
        """
        try:
            # Get penalty-based avoidance scores
            penalty_scores = await self.penalty_system.get_avoidance_recommendations(
                game_id, candidate_coordinates
            )
            
            # Get diversity-based avoidance scores
            diversity_scores = {}
            for x, y in candidate_coordinates:
                coord_key = (game_id, x, y)
                
                # Check recent usage frequency
                recent_usage = 0
                if coord_key in self.coordinate_intelligence:
                    coord_intel = self.coordinate_intelligence[coord_key]
                    time_since_use = time.time() - coord_intel.last_used
                    if time_since_use < 300:  # 5 minutes
                        recent_usage = 1.0 - (time_since_use / 300)
                
                # Check if coordinate is in a recently used zone
                zone_penalty = 0.0
                if coord_key in self.coordinate_intelligence:
                    zone_id = self.coordinate_intelligence[coord_key].zone_id
                    if zone_id and zone_id in self.zone_mapper.zones.get(game_id, {}):
                        zone = self.zone_mapper.zones[game_id][zone_id]
                        if zone.last_used and (time.time() - zone.last_used) < 180:  # 3 minutes
                            zone_penalty = 0.5
                
                diversity_scores[(x, y)] = recent_usage + zone_penalty
            
            # Combine penalty and diversity scores
            combined_scores = {}
            for coord in candidate_coordinates:
                penalty_score = penalty_scores.get(coord, 0.0)
                diversity_score = diversity_scores.get(coord, 0.0)
                
                # Weighted combination: penalties are more important than diversity
                combined_scores[coord] = (penalty_score * 0.7) + (diversity_score * 0.3)
            
            return combined_scores
            
        except Exception as e:
            logger.error(f"Failed to get coordinate avoidance scores: {e}")
            return {coord: 0.0 for coord in candidate_coordinates}
    
    async def get_diverse_coordinate_recommendations(
        self, 
        game_id: str, 
        action_id: int, 
        grid_size: Tuple[int, int], 
        strategy: str = "balanced",
        max_recommendations: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get diverse coordinate recommendations that avoid recently used and penalized coordinates.
        
        Args:
            game_id: Game identifier
            action_id: Action type
            grid_size: (width, height) of the grid
            strategy: Recommendation strategy ('success', 'exploration', 'balanced')
            max_recommendations: Maximum number of recommendations
            
        Returns:
            List of coordinate recommendations with diversity and penalty information
        """
        try:
            # Get base recommendations from success zones
            base_recommendations = self.get_recommended_coordinates(
                game_id, action_id, grid_size, strategy
            )
            
            if not base_recommendations:
                return []
            
            # Extract coordinates
            candidate_coords = [(rec['x'], rec['y']) for rec in base_recommendations]
            
            # Get avoidance scores
            avoidance_scores = await self.get_coordinate_avoidance_scores(
                game_id, candidate_coords
            )
            
            # Combine recommendations with avoidance scores
            enhanced_recommendations = []
            for rec in base_recommendations:
                coord = (rec['x'], rec['y'])
                avoidance_score = avoidance_scores.get(coord, 0.0)
                
                # Calculate diversity-adjusted score
                diversity_factor = 1.0 - avoidance_score
                adjusted_score = rec['confidence_score'] * diversity_factor
                
                enhanced_rec = rec.copy()
                enhanced_rec.update({
                    'avoidance_score': avoidance_score,
                    'diversity_factor': diversity_factor,
                    'adjusted_score': adjusted_score,
                    'penalty_info': await self.penalty_system.get_coordinate_penalty(
                        game_id, rec['x'], rec['y']
                    )
                })
                
                enhanced_recommendations.append(enhanced_rec)
            
            # Sort by adjusted score (higher is better)
            enhanced_recommendations.sort(key=lambda x: x['adjusted_score'], reverse=True)
            
            return enhanced_recommendations[:max_recommendations]
            
        except Exception as e:
            logger.error(f"Failed to get diverse coordinate recommendations: {e}")
            return []
    
    async def decay_penalties(self, game_id: str = None) -> Dict[str, Any]:
        """Apply penalty decay to allow recovery of previously penalized coordinates."""
        try:
            return await self.penalty_system.decay_penalties(game_id)
        except Exception as e:
            logger.error(f"Failed to decay penalties: {e}")
            return {'error': str(e)}
    
    async def get_penalty_system_status(self) -> Dict[str, Any]:
        """Get status of the penalty decay system."""
        try:
            return await self.penalty_system.get_system_status()
        except Exception as e:
            logger.error(f"Failed to get penalty system status: {e}")
            return {'error': str(e)}
    
    def export_system_data(self) -> Dict[str, Any]:
        """Export all system data for persistence."""
        zones_data = {}
        for game_id in self.zone_mapper.zones:
            zones_data[game_id] = self.zone_mapper.export_zones(game_id)
        
        coordinate_data = {}
        for coord_key, coord_intel in self.coordinate_intelligence.items():
            coordinate_data[f"{coord_key[0]}_{coord_key[1]}_{coord_key[2]}"] = {
                'game_id': coord_intel.game_id,
                'x': coord_intel.x,
                'y': coord_intel.y,
                'attempts': coord_intel.attempts,
                'successes': coord_intel.successes,
                'success_rate': coord_intel.success_rate,
                'frame_changes': coord_intel.frame_changes,
                'last_used': coord_intel.last_used,
                'action_success_rates': coord_intel.action_success_rates,
                'zone_id': coord_intel.zone_id,
                'confidence_score': coord_intel.confidence_score,
                'cross_game_references': coord_intel.cross_game_references
            }
        
        return {
            'zones': zones_data,
            'coordinates': coordinate_data,
            'cross_game_patterns': dict(self.zone_mapper.cross_game_patterns),
            'statistics': self.get_system_statistics()
        }
    
    def import_system_data(self, data: Dict[str, Any]):
        """Import system data from persistence."""
        # Import zones
        for game_id, zones_data in data['zones'].items():
            self.zone_mapper.import_zones(zones_data)
        
        # Import coordinate intelligence
        self.coordinate_intelligence.clear()
        for coord_key, coord_data in data['coordinates'].items():
            coord_intel = CoordinateIntelligence(
                game_id=coord_data['game_id'],
                x=coord_data['x'],
                y=coord_data['y'],
                attempts=coord_data['attempts'],
                successes=coord_data['successes'],
                success_rate=coord_data['success_rate'],
                frame_changes=coord_data['frame_changes'],
                last_used=coord_data['last_used'],
                action_success_rates=coord_data['action_success_rates'],
                zone_id=coord_data['zone_id'],
                confidence_score=coord_data['confidence_score'],
                cross_game_references=coord_data['cross_game_references']
            )
            
            coord_key_tuple = (coord_data['game_id'], coord_data['x'], coord_data['y'])
            self.coordinate_intelligence[coord_key_tuple] = coord_intel
        
        # Import cross-game patterns
        self.zone_mapper.cross_game_patterns = defaultdict(dict, data['cross_game_patterns'])
        
        logger.info("Imported coordinate intelligence system data")


# Factory function for easy integration
def create_coordinate_intelligence_system() -> CoordinateIntelligenceSystem:
    """Create a new coordinate intelligence system."""
    return CoordinateIntelligenceSystem()


if __name__ == "__main__":
    # Example usage
    system = create_coordinate_intelligence_system()
    
    # Simulate some coordinate updates
    system.update_coordinate_intelligence("game1", 10, 10, 6, True, 1)
    system.update_coordinate_intelligence("game1", 10, 11, 6, True, 1)
    system.update_coordinate_intelligence("game1", 11, 10, 6, True, 1)
    
    # Get recommendations
    recommendations = system.get_recommended_coordinates("game1", 6, (64, 64), "success")
    print(f"Recommended coordinates: {recommendations[:5]}")
    
    # Get statistics
    stats = system.get_system_statistics()
    print(f"System statistics: {stats}")
    
    print("Coordinate Intelligence System test completed successfully!")
