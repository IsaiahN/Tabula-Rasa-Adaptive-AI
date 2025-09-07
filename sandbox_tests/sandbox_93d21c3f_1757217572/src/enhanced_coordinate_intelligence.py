#!/usr/bin/env python3
"""
Enhanced Coordinate Intelligence for ARC Training

This module provides intelligent coordinate selection that:
1. Leverages memory to avoid repeated attempts
2. Expands clusters systematically (left, right, up, down)
3. Tracks coordinate effectiveness across sessions
4. Implements smart exploration strategies
"""

from typing import Tuple, Dict, Set, List, Any, Optional
import random
import time
import numpy as np

class EnhancedCoordinateIntelligence:
    """
    Advanced coordinate selection system that learns from past attempts
    and systematically expands successful clusters.
    """
    
    def __init__(self, continuous_loop):
        self.continuous_loop = continuous_loop
        self.cluster_expansion_directions = [
            (0, 1),   # Right
            (0, -1),  # Left  
            (1, 0),   # Down
            (-1, 0),  # Up
            (1, 1),   # Down-Right
            (-1, -1), # Up-Left
            (1, -1),  # Down-Left
            (-1, 1),  # Up-Right
        ]
    
    def get_intelligent_coordinates(self, action: int, grid_dims: Tuple[int, int], game_id: str) -> Tuple[int, int]:
        """
        Get intelligently selected coordinates that leverage memory and cluster expansion.
        
        Priority Order:
        1. Expand existing successful clusters
        2. Avoid recently attempted coordinates
        3. Use strategic exploration patterns
        4. Fall back to smart random selection
        """
        grid_width, grid_height = grid_dims
        boundary_system = self.continuous_loop.available_actions_memory['universal_boundary_detection']
        
        # Ensure boundary system is initialized
        self._ensure_boundary_system_initialized(game_id)
        
        # Priority 1: Expand successful clusters
        cluster_coord = self._get_cluster_expansion_coordinate(action, grid_dims, game_id)
        if cluster_coord:
            print(f"🔗 ACTION {action} CLUSTER EXPANSION: Expanding to {cluster_coord}")
            return cluster_coord
        
        # Priority 2: Avoid recently attempted coordinates
        memory_guided_coord = self._get_memory_guided_coordinate(action, grid_dims, game_id)
        if memory_guided_coord:
            print(f"🧠 ACTION {action} MEMORY GUIDED: Avoiding known coordinates, trying {memory_guided_coord}")
            return memory_guided_coord
        
        # Priority 3: Strategic exploration
        strategic_coord = self._get_strategic_exploration_coordinate(action, grid_dims, game_id)
        if strategic_coord:
            print(f"🎯 ACTION {action} STRATEGIC: Exploring {strategic_coord}")
            return strategic_coord
        
        # Priority 4: Smart random with grid analysis
        smart_random_coord = self._get_smart_random_coordinate(action, grid_dims, game_id)
        print(f"🎲 ACTION {action} SMART RANDOM: Trying {smart_random_coord}")
        return smart_random_coord
    
    def _get_cluster_expansion_coordinate(self, action: int, grid_dims: Tuple[int, int], game_id: str) -> Optional[Tuple[int, int]]:
        """Find coordinates to expand existing successful clusters."""
        boundary_system = self.continuous_loop.available_actions_memory['universal_boundary_detection']
        success_zones = boundary_system['success_zone_mapping'].get(game_id, {})
        clusters = boundary_system['coordinate_clusters'].get(game_id, {})
        
        if not success_zones:
            return None
        
        # Find the most promising cluster to expand
        best_cluster = None
        best_score = 0
        
        for cluster_id, cluster_data in clusters.items():
            # Score based on success rate and recency
            avg_success_rate = cluster_data.get('avg_success_rate', 0)
            member_count = len(cluster_data.get('members', set()))
            recency_score = 1.0 / (time.time() - cluster_data.get('created_time', time.time()) + 1)
            
            cluster_score = avg_success_rate * member_count * recency_score
            if cluster_score > best_score:
                best_score = cluster_score
                best_cluster = cluster_data
        
        if not best_cluster:
            # No clusters yet, try to create one from individual success zones
            return self._find_cluster_seed_coordinate(action, grid_dims, game_id)
        
        # Find unexplored neighbors of the best cluster
        cluster_members = best_cluster['members']
        grid_width, grid_height = grid_dims
        
        candidate_expansions = []
        
        for member_coord in cluster_members:
            x, y = member_coord
            
            # Try all 8 directions for expansion
            for dx, dy in self.cluster_expansion_directions:
                new_x, new_y = x + dx, y + dy
                
                # Check bounds
                if 1 <= new_x <= grid_width - 2 and 1 <= new_y <= grid_height - 2:
                    new_coord = (new_x, new_y)
                    
                    # Check if we've already tried this coordinate recently
                    if not self._has_been_attempted_recently(new_coord, game_id, max_recent_attempts=3):
                        # Check if it's not in a known danger zone
                        if not self._is_dangerous_coordinate(new_coord, game_id):
                            candidate_expansions.append((new_coord, dx, dy, member_coord))
        
        if candidate_expansions:
            # Prioritize cardinal directions over diagonals
            cardinal_expansions = [(coord, dx, dy, parent) for coord, dx, dy, parent in candidate_expansions 
                                 if abs(dx) + abs(dy) == 1]
            
            if cardinal_expansions:
                # Choose a cardinal direction expansion
                chosen = random.choice(cardinal_expansions)
                expansion_coord, dx, dy, parent_coord = chosen
                direction_name = {(0,1): "RIGHT", (0,-1): "LEFT", (1,0): "DOWN", (-1,0): "UP"}
                print(f"   📍 Expanding {direction_name.get((dx,dy), 'DIAGONAL')} from successful {parent_coord}")
                return expansion_coord
            else:
                # Use diagonal expansion if no cardinal available
                chosen = random.choice(candidate_expansions)
                expansion_coord, dx, dy, parent_coord = chosen
                print(f"   📍 Expanding diagonally from successful {parent_coord}")
                return expansion_coord
        
        return None
    
    def _find_cluster_seed_coordinate(self, action: int, grid_dims: Tuple[int, int], game_id: str) -> Optional[Tuple[int, int]]:
        """Find a coordinate near a successful individual point to start a cluster."""
        boundary_system = self.continuous_loop.available_actions_memory['universal_boundary_detection']
        success_zones = boundary_system['success_zone_mapping'].get(game_id, {})
        
        if not success_zones:
            return None
        
        # Find the most successful individual coordinate
        best_coord = None
        best_success_rate = 0
        
        for coord, zone_data in success_zones.items():
            success_rate = zone_data['success_count'] / max(1, zone_data['total_attempts'])
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_coord = coord
        
        if best_coord and best_success_rate > 0.3:  # At least 30% success rate
            # Try to expand around this successful coordinate
            x, y = best_coord
            
            for dx, dy in self.cluster_expansion_directions:
                new_x, new_y = x + dx, y + dy
                grid_width, grid_height = grid_dims
                
                if 1 <= new_x <= grid_width - 2 and 1 <= new_y <= grid_height - 2:
                    new_coord = (new_x, new_y)
                    
                    if not self._has_been_attempted_recently(new_coord, game_id, max_recent_attempts=2):
                        print(f"   🌱 Starting new cluster from successful {best_coord} (success rate: {best_success_rate:.1%})")
                        return new_coord
        
        return None
    
    def _get_memory_guided_coordinate(self, action: int, grid_dims: Tuple[int, int], game_id: str) -> Optional[Tuple[int, int]]:
        """Get coordinates that avoid recently attempted locations."""
        boundary_system = self.continuous_loop.available_actions_memory['universal_boundary_detection']
        coordinate_attempts = boundary_system['coordinate_attempts'].get(game_id, {})
        
        grid_width, grid_height = grid_dims
        max_attempts = 50
        
        for attempt in range(max_attempts):
            # Generate strategic coordinate based on action type
            if action == 6:
                # For action 6, use broader exploration
                x = random.randint(2, grid_width - 3)
                y = random.randint(2, grid_height - 3)
            else:
                # For other actions, use action-specific regions
                x, y = self._get_action_specific_strategic_coordinate(action, grid_dims)
            
            coord_str = str((x, y))
            
            # Check if this coordinate has been attempted recently
            if coord_str not in coordinate_attempts:
                return (x, y)
            
            # If it has been attempted, check if it was long ago or unsuccessful
            coord_data = coordinate_attempts[coord_str]
            attempts = coord_data.get('attempts', 0)
            success_rate = coord_data.get('success_rate', 0)
            
            # Avoid coordinates that have been attempted many times without success
            if attempts >= 3 and success_rate < 0.2:
                continue
            
            # Allow re-trying coordinates that were successful but not recently
            if success_rate > 0.5 and attempts < 5:
                return (x, y)
        
        return None
    
    def _get_strategic_exploration_coordinate(self, action: int, grid_dims: Tuple[int, int], game_id: str) -> Optional[Tuple[int, int]]:
        """Get coordinates using strategic exploration patterns."""
        grid_width, grid_height = grid_dims
        
        # Use grid subdivision strategy
        # Divide grid into quadrants and systematically explore
        quadrants = [
            (grid_width // 4, grid_height // 4),           # Upper left
            (3 * grid_width // 4, grid_height // 4),       # Upper right
            (grid_width // 4, 3 * grid_height // 4),       # Lower left
            (3 * grid_width // 4, 3 * grid_height // 4),   # Lower right
            (grid_width // 2, grid_height // 2),           # Center
        ]
        
        # Check exploration coverage for each quadrant
        boundary_system = self.continuous_loop.available_actions_memory['universal_boundary_detection']
        coordinate_attempts = boundary_system['coordinate_attempts'].get(game_id, {})
        
        quadrant_coverage = []
        for qx, qy in quadrants:
            # Count attempts in this quadrant (within 5 units)
            attempts_in_quadrant = 0
            for coord_str, coord_data in coordinate_attempts.items():
                try:
                    coord = eval(coord_str)  # Convert string back to tuple
                    if isinstance(coord, tuple) and len(coord) == 2:
                        cx, cy = coord
                        distance = abs(cx - qx) + abs(cy - qy)  # Manhattan distance
                        if distance <= 5:
                            attempts_in_quadrant += coord_data.get('attempts', 0)
                except:
                    continue
            
            quadrant_coverage.append((attempts_in_quadrant, qx, qy))
        
        # Choose the least explored quadrant
        quadrant_coverage.sort()  # Sort by attempts (ascending)
        least_explored_attempts, qx, qy = quadrant_coverage[0]
        
        # Add some randomization around the chosen quadrant center
        x = max(1, min(grid_width - 2, qx + random.randint(-3, 3)))
        y = max(1, min(grid_height - 2, qy + random.randint(-3, 3)))
        
        print(f"   🗺️ Quadrant at ({qx},{qy}) has {least_explored_attempts} attempts, exploring near ({x},{y})")
        return (x, y)
    
    def _get_smart_random_coordinate(self, action: int, grid_dims: Tuple[int, int], game_id: str) -> Tuple[int, int]:
        """Get a smart random coordinate with action-specific patterns."""
        return self._get_action_specific_strategic_coordinate(action, grid_dims)
    
    def _get_action_specific_strategic_coordinate(self, action: int, grid_dims: Tuple[int, int]) -> Tuple[int, int]:
        """Get action-specific strategic coordinates with variation."""
        grid_width, grid_height = grid_dims
        
        # Action-specific strategic patterns with larger variation
        if action == 1:  # Drawing/placing
            base_x, base_y = grid_width // 4, grid_height // 4
            x = max(1, min(grid_width - 2, base_x + random.randint(-5, 5)))
            y = max(1, min(grid_height - 2, base_y + random.randint(-5, 5)))
        elif action == 2:  # Modifying
            base_x, base_y = grid_width // 2, grid_height // 2
            x = max(1, min(grid_width - 2, base_x + random.randint(-7, 7)))
            y = max(1, min(grid_height - 2, base_y + random.randint(-7, 7)))
        elif action == 3:  # Erasing/removing
            base_x, base_y = 3 * grid_width // 4, 3 * grid_height // 4
            x = max(1, min(grid_width - 2, base_x + random.randint(-5, 5)))
            y = max(1, min(grid_height - 2, base_y + random.randint(-5, 5)))
        elif action == 4:  # Pattern-related
            base_x, base_y = grid_width // 4, 3 * grid_height // 4
            x = max(1, min(grid_width - 2, base_x + random.randint(-5, 5)))
            y = max(1, min(grid_height - 2, base_y + random.randint(-5, 5)))
        elif action == 5:  # Transformation
            base_x, base_y = 3 * grid_width // 4, grid_height // 4
            x = max(1, min(grid_width - 2, base_x + random.randint(-5, 5)))
            y = max(1, min(grid_height - 2, base_y + random.randint(-5, 5)))
        elif action == 6:  # Special coordinate action - much broader exploration
            # Use the full grid with bias towards unexplored areas
            x = random.randint(3, grid_width - 4)
            y = random.randint(3, grid_height - 4)
        else:
            # Broad exploration for unknown actions
            x = random.randint(2, grid_width - 3)
            y = random.randint(2, grid_height - 3)
        
        return (x, y)
    
    def _has_been_attempted_recently(self, coord: Tuple[int, int], game_id: str, max_recent_attempts: int = 2) -> bool:
        """Check if a coordinate has been attempted recently."""
        boundary_system = self.continuous_loop.available_actions_memory['universal_boundary_detection']
        coordinate_attempts = boundary_system['coordinate_attempts'].get(game_id, {})
        
        coord_str = str(coord)
        if coord_str in coordinate_attempts:
            attempts = coordinate_attempts[coord_str].get('attempts', 0)
            return attempts >= max_recent_attempts
        
        return False
    
    def _is_dangerous_coordinate(self, coord: Tuple[int, int], game_id: str) -> bool:
        """Check if a coordinate is in a known danger zone."""
        boundary_system = self.continuous_loop.available_actions_memory['universal_boundary_detection']
        boundary_data = boundary_system['boundary_data'].get(game_id, {})
        
        x, y = coord
        
        # Check if coordinate is directly on a known boundary
        for boundary_coord in boundary_data.keys():
            try:
                if isinstance(boundary_coord, str):
                    boundary_coord = eval(boundary_coord)
                if boundary_coord == coord:
                    return True
                
                # Check if too close to boundary (within 1 unit)
                bx, by = boundary_coord
                if abs(x - bx) <= 1 and abs(y - by) <= 1:
                    return True
            except:
                continue
        
        return False
    
    def _ensure_boundary_system_initialized(self, game_id: str):
        """Ensure the boundary system is properly initialized for the game."""
        boundary_system = self.continuous_loop.available_actions_memory['universal_boundary_detection']
        
        for key in ['boundary_data', 'coordinate_attempts', 'action_coordinate_history', 'success_zone_mapping', 'coordinate_clusters']:
            if key not in boundary_system:
                boundary_system[key] = {}
            if game_id not in boundary_system[key]:
                boundary_system[key][game_id] = {}
