#!/usr/bin/env python3
"""
Systematic Exploration Phases System

This module implements systematic exploration phases (corners → center → edges → random)
with database tracking and intelligent coordinate selection.
"""

import logging
import json
import time
import random
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.database.system_integration import get_system_integration

logger = logging.getLogger(__name__)

class ExplorationPhase(Enum):
    """Exploration phases for systematic coordinate discovery."""
    CORNERS = "corners"
    CENTER = "center"
    EDGES = "edges"
    RANDOM = "random"

@dataclass
class ExplorationPhaseData:
    """Data for an exploration phase."""
    game_id: str
    session_id: str
    phase_name: ExplorationPhase
    phase_attempts: int
    successful_attempts: int
    coordinates_tried: List[Tuple[int, int]]
    phase_start_time: float
    phase_end_time: Optional[float] = None
    phase_success_rate: float = 0.0

class SystematicExplorationSystem:
    """
    Systematic Exploration Phases System
    
    Implements intelligent exploration phases for discovering effective coordinates
    through systematic progression from corners to center to edges to random.
    """
    
    def __init__(self):
        self.integration = get_system_integration()
        self.active_phases: Dict[str, ExplorationPhaseData] = {}
        self.exploration_history: Dict[str, List[ExplorationPhaseData]] = {}
        
    async def get_exploration_coordinates(self, 
                                        game_id: str,
                                        session_id: str,
                                        grid_dimensions: Tuple[int, int],
                                        available_actions: List[int]) -> Tuple[int, int, str]:
        """
        Get coordinates for systematic exploration.
        
        Args:
            game_id: Game identifier
            session_id: Session identifier
            grid_dimensions: (width, height) of the grid
            available_actions: Available actions for the game
            
        Returns:
            Tuple of (x, y, phase_name) for exploration
        """
        try:
            grid_width, grid_height = grid_dimensions
            
            # Get or create current exploration phase
            current_phase = await self._get_current_exploration_phase(game_id, session_id)
            
            # Generate coordinates based on current phase
            if current_phase.phase_name == ExplorationPhase.CORNERS:
                x, y = self._generate_corner_coordinates(grid_width, grid_height, current_phase)
                phase_name = "corners"
            elif current_phase.phase_name == ExplorationPhase.CENTER:
                x, y = self._generate_center_coordinates(grid_width, grid_height, current_phase)
                phase_name = "center"
            elif current_phase.phase_name == ExplorationPhase.EDGES:
                x, y = self._generate_edge_coordinates(grid_width, grid_height, current_phase)
                phase_name = "edges"
            else:  # RANDOM
                x, y = self._generate_random_coordinates(grid_width, grid_height, current_phase)
                phase_name = "random"
            
            # Record coordinate attempt
            await self._record_coordinate_attempt(current_phase, x, y)
            
            # Check if phase should advance
            await self._check_phase_advancement(current_phase, grid_width, grid_height)
            
            logger.debug(f"Exploration coordinates: ({x}, {y}) - Phase: {phase_name} "
                        f"(attempt {current_phase.phase_attempts})")
            
            return x, y, phase_name
            
        except Exception as e:
            logger.error(f"Error getting exploration coordinates: {e}")
            # Fallback to center coordinates
            return grid_width // 2, grid_height // 2, "fallback"
    
    async def record_exploration_result(self, 
                                      game_id: str,
                                      coordinates: Tuple[int, int],
                                      success: bool,
                                      frame_changes: bool,
                                      score_impact: float = 0.0):
        """Record the result of an exploration attempt."""
        try:
            if game_id not in self.active_phases:
                return
            
            current_phase = self.active_phases[game_id]
            
            # Update phase success tracking
            if success or frame_changes or score_impact > 0:
                current_phase.successful_attempts += 1
            
            # Update success rate
            if current_phase.phase_attempts > 0:
                current_phase.phase_success_rate = current_phase.successful_attempts / current_phase.phase_attempts
            
            # Store result in database
            await self._store_exploration_result(
                current_phase, coordinates, success, frame_changes, score_impact
            )
            
            logger.debug(f"Exploration result recorded: ({coordinates[0]}, {coordinates[1]}) - "
                        f"Success: {success}, Frame changes: {frame_changes}, "
                        f"Score impact: {score_impact}")
            
        except Exception as e:
            logger.error(f"Error recording exploration result: {e}")
    
    async def _get_current_exploration_phase(self, 
                                           game_id: str, 
                                           session_id: str) -> ExplorationPhaseData:
        """Get or create current exploration phase for a game."""
        try:
            if game_id in self.active_phases:
                return self.active_phases[game_id]
            
            # Load from database or create new phase
            query = """
                SELECT phase_name, phase_attempts, successful_attempts, coordinates_tried,
                       phase_start_time, phase_end_time, phase_success_rate
                FROM exploration_phases
                WHERE game_id = ? AND phase_end_time IS NULL
                ORDER BY phase_start_time DESC
                LIMIT 1
            """
            
            result = await self.integration.db.fetch_one(query, (game_id,))
            
            if result:
                # Resume existing phase
                phase = ExplorationPhaseData(
                    game_id=game_id,
                    session_id=session_id,
                    phase_name=ExplorationPhase(result['phase_name']),
                    phase_attempts=result['phase_attempts'],
                    successful_attempts=result['successful_attempts'],
                    coordinates_tried=json.loads(result['coordinates_tried']),
                    phase_start_time=result['phase_start_time'],
                    phase_end_time=result['phase_end_time'],
                    phase_success_rate=result['phase_success_rate']
                )
            else:
                # Create new exploration phase starting with corners
                phase = ExplorationPhaseData(
                    game_id=game_id,
                    session_id=session_id,
                    phase_name=ExplorationPhase.CORNERS,
                    phase_attempts=0,
                    successful_attempts=0,
                    coordinates_tried=[],
                    phase_start_time=time.time()
                )
                
                # Store new phase in database
                await self._store_exploration_phase(phase)
            
            self.active_phases[game_id] = phase
            return phase
            
        except Exception as e:
            logger.error(f"Error getting current exploration phase: {e}")
            # Return default phase
            return ExplorationPhaseData(
                game_id=game_id,
                session_id=session_id,
                phase_name=ExplorationPhase.CORNERS,
                phase_attempts=0,
                successful_attempts=0,
                coordinates_tried=[],
                phase_start_time=time.time()
            )
    
    def _generate_corner_coordinates(self, 
                                   grid_width: int, 
                                   grid_height: int,
                                   phase: ExplorationPhaseData) -> Tuple[int, int]:
        """Generate coordinates for corner exploration phase."""
        try:
            # Define corner positions
            corners = [
                (5, 5),  # Top-left
                (grid_width - 6, 5),  # Top-right
                (5, grid_height - 6),  # Bottom-left
                (grid_width - 6, grid_height - 6)  # Bottom-right
            ]
            
            # Try each corner in sequence
            attempt = phase.phase_attempts % len(corners)
            x, y = corners[attempt]
            
            # Add small random offset to avoid exact repetition
            offset_x = random.randint(-2, 2)
            offset_y = random.randint(-2, 2)
            
            x = max(0, min(grid_width - 1, x + offset_x))
            y = max(0, min(grid_height - 1, y + offset_y))
            
            return x, y
            
        except Exception as e:
            logger.error(f"Error generating corner coordinates: {e}")
            return grid_width // 2, grid_height // 2
    
    def _generate_center_coordinates(self, 
                                   grid_width: int, 
                                   grid_height: int,
                                   phase: ExplorationPhaseData) -> Tuple[int, int]:
        """Generate coordinates for center exploration phase."""
        try:
            center_x = grid_width // 2
            center_y = grid_height // 2
            
            # Generate coordinates around center with increasing radius
            attempt = phase.phase_attempts
            radius = min(10, attempt // 2 + 1)  # Increase radius with attempts
            
            # Generate points in a spiral pattern around center
            angle = (attempt * 0.5) % (2 * np.pi)
            offset_x = int(radius * np.cos(angle))
            offset_y = int(radius * np.sin(angle))
            
            x = max(0, min(grid_width - 1, center_x + offset_x))
            y = max(0, min(grid_height - 1, center_y + offset_y))
            
            return x, y
            
        except Exception as e:
            logger.error(f"Error generating center coordinates: {e}")
            return grid_width // 2, grid_height // 2
    
    def _generate_edge_coordinates(self, 
                                 grid_width: int, 
                                 grid_height: int,
                                 phase: ExplorationPhaseData) -> Tuple[int, int]:
        """Generate coordinates for edge exploration phase."""
        try:
            # Define edge positions
            edges = [
                (grid_width // 2, 3),  # Top edge
                (grid_width - 3, grid_height // 2),  # Right edge
                (grid_width // 2, grid_height - 3),  # Bottom edge
                (3, grid_height // 2)  # Left edge
            ]
            
            # Try each edge in sequence
            attempt = phase.phase_attempts % len(edges)
            x, y = edges[attempt]
            
            # Add small random offset
            offset_x = random.randint(-3, 3)
            offset_y = random.randint(-3, 3)
            
            x = max(0, min(grid_width - 1, x + offset_x))
            y = max(0, min(grid_height - 1, y + offset_y))
            
            return x, y
            
        except Exception as e:
            logger.error(f"Error generating edge coordinates: {e}")
            return grid_width // 2, grid_height // 2
    
    def _generate_random_coordinates(self, 
                                   grid_width: int, 
                                   grid_height: int,
                                   phase: ExplorationPhaseData) -> Tuple[int, int]:
        """Generate coordinates for random exploration phase."""
        try:
            # Avoid recently tried coordinates
            recent_coords = phase.coordinates_tried[-10:]  # Last 10 coordinates
            
            max_attempts = 20
            for _ in range(max_attempts):
                x = random.randint(3, grid_width - 4)
                y = random.randint(3, grid_height - 4)
                
                # Check if this coordinate is too close to recent attempts
                too_close = False
                for prev_x, prev_y in recent_coords:
                    if abs(x - prev_x) < 5 and abs(y - prev_y) < 5:
                        too_close = True
                        break
                
                if not too_close:
                    return x, y
            
            # If all attempts failed, return random coordinate
            return random.randint(3, grid_width - 4), random.randint(3, grid_height - 4)
            
        except Exception as e:
            logger.error(f"Error generating random coordinates: {e}")
            return grid_width // 2, grid_height // 2
    
    async def _record_coordinate_attempt(self, 
                                       phase: ExplorationPhaseData,
                                       x: int, 
                                       y: int):
        """Record a coordinate attempt for the current phase."""
        try:
            phase.phase_attempts += 1
            phase.coordinates_tried.append((x, y))
            
            # Keep only recent coordinates
            if len(phase.coordinates_tried) > 20:
                phase.coordinates_tried = phase.coordinates_tried[-20:]
            
            # Update phase in database
            await self._update_exploration_phase(phase)
            
        except Exception as e:
            logger.error(f"Error recording coordinate attempt: {e}")
    
    async def _check_phase_advancement(self, 
                                     phase: ExplorationPhaseData,
                                     grid_width: int, 
                                     grid_height: int):
        """Check if the current phase should advance to the next phase."""
        try:
            should_advance = False
            next_phase = None
            
            if phase.phase_name == ExplorationPhase.CORNERS:
                # Advance after trying all corners
                if phase.phase_attempts >= 4:
                    should_advance = True
                    next_phase = ExplorationPhase.CENTER
            elif phase.phase_name == ExplorationPhase.CENTER:
                # Advance after sufficient center exploration
                if phase.phase_attempts >= 6:
                    should_advance = True
                    next_phase = ExplorationPhase.EDGES
            elif phase.phase_name == ExplorationPhase.EDGES:
                # Advance after trying all edges
                if phase.phase_attempts >= 4:
                    should_advance = True
                    next_phase = ExplorationPhase.RANDOM
            # RANDOM phase continues indefinitely
            
            if should_advance and next_phase:
                # End current phase
                phase.phase_end_time = time.time()
                await self._update_exploration_phase(phase)
                
                # Start new phase
                new_phase = ExplorationPhaseData(
                    game_id=phase.game_id,
                    session_id=phase.session_id,
                    phase_name=next_phase,
                    phase_attempts=0,
                    successful_attempts=0,
                    coordinates_tried=[],
                    phase_start_time=time.time()
                )
                
                await self._store_exploration_phase(new_phase)
                self.active_phases[phase.game_id] = new_phase
                
                logger.info(f"Exploration phase advanced: {phase.phase_name.value} → {next_phase.value}")
            
        except Exception as e:
            logger.error(f"Error checking phase advancement: {e}")
    
    async def _store_exploration_phase(self, phase: ExplorationPhaseData):
        """Store exploration phase in database."""
        try:
            await self.integration.db.execute("""
                INSERT INTO exploration_phases
                (game_id, session_id, phase_name, phase_attempts, successful_attempts,
                 coordinates_tried, phase_start_time, phase_end_time, phase_success_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                phase.game_id, phase.session_id, phase.phase_name.value,
                phase.phase_attempts, phase.successful_attempts,
                json.dumps(phase.coordinates_tried), phase.phase_start_time,
                phase.phase_end_time, phase.phase_success_rate
            ))
            
        except Exception as e:
            logger.error(f"Error storing exploration phase: {e}")
    
    async def _update_exploration_phase(self, phase: ExplorationPhaseData):
        """Update exploration phase in database."""
        try:
            await self.integration.db.execute("""
                UPDATE exploration_phases
                SET phase_attempts = ?, successful_attempts = ?, coordinates_tried = ?,
                    phase_end_time = ?, phase_success_rate = ?
                WHERE game_id = ? AND phase_start_time = ?
            """, (
                phase.phase_attempts, phase.successful_attempts,
                json.dumps(phase.coordinates_tried), phase.phase_end_time,
                phase.phase_success_rate, phase.game_id, phase.phase_start_time
            ))
            
        except Exception as e:
            logger.error(f"Error updating exploration phase: {e}")
    
    async def _store_exploration_result(self, 
                                      phase: ExplorationPhaseData,
                                      coordinates: Tuple[int, int],
                                      success: bool,
                                      frame_changes: bool,
                                      score_impact: float):
        """Store exploration result in database."""
        try:
            # This could be stored in a separate exploration_results table
            # For now, we'll just log the result
            logger.debug(f"Exploration result: Phase {phase.phase_name.value}, "
                        f"Coordinates ({coordinates[0]}, {coordinates[1]}), "
                        f"Success: {success}, Frame changes: {frame_changes}")
            
        except Exception as e:
            logger.error(f"Error storing exploration result: {e}")
    
    async def get_exploration_statistics(self, game_id: str) -> Dict[str, Any]:
        """Get exploration statistics for a game."""
        try:
            query = """
                SELECT phase_name, phase_attempts, successful_attempts, phase_success_rate,
                       phase_start_time, phase_end_time
                FROM exploration_phases
                WHERE game_id = ?
                ORDER BY phase_start_time ASC
            """
            
            results = await self.integration.db.fetch_all(query, (game_id,))
            
            if not results:
                return {'total_phases': 0}
            
            phases = []
            total_attempts = 0
            total_successes = 0
            
            for row in results:
                phase_data = {
                    'phase_name': row['phase_name'],
                    'attempts': row['phase_attempts'],
                    'successes': row['successful_attempts'],
                    'success_rate': row['phase_success_rate'],
                    'duration': (row['phase_end_time'] or time.time()) - row['phase_start_time']
                }
                phases.append(phase_data)
                
                total_attempts += row['phase_attempts']
                total_successes += row['successful_attempts']
            
            overall_success_rate = (total_successes / total_attempts) if total_attempts > 0 else 0.0
            
            return {
                'total_phases': len(phases),
                'total_attempts': total_attempts,
                'total_successes': total_successes,
                'overall_success_rate': overall_success_rate,
                'phases': phases,
                'current_phase': phases[-1]['phase_name'] if phases else None
            }
            
        except Exception as e:
            logger.error(f"Error getting exploration statistics: {e}")
            return {}
    
    async def get_effective_coordinates(self, game_id: str) -> List[Tuple[int, int]]:
        """Get coordinates that have been effective in exploration."""
        try:
            # This would require tracking successful coordinates
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Error getting effective coordinates: {e}")
            return []
    
    async def reset_exploration_for_game(self, game_id: str):
        """Reset exploration phases for a new game."""
        try:
            # End any active phases
            if game_id in self.active_phases:
                phase = self.active_phases[game_id]
                phase.phase_end_time = time.time()
                await self._update_exploration_phase(phase)
                del self.active_phases[game_id]
            
            logger.info(f"Exploration reset for game: {game_id}")
            
        except Exception as e:
            logger.error(f"Error resetting exploration for game: {e}")
