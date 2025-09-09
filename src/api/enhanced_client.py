"""
Enhanced ARC-AGI-3 API Client with proper coordinate handling
Implements correct ACTION6 usage and comprehensive action management.
"""
import requests
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import time


class ArcAgiApiClient:
    """
    Enhanced API client for ARC-AGI-3 with proper coordinate handling
    and comprehensive action support.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://three.arcprize.org"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': api_key,
            'Content-Type': 'application/json'
        })
        
        # Current session state
        self.current_game_id = None
        self.current_guid = None
        self.current_scorecard_id = None
        self.current_score = 0
        self.win_score = 0
        
        # Action tracking
        self.action_history = []
        self.last_response = None
        
    def start_game(self, game_id: str, scorecard_id: str, reset_existing: bool = False) -> Dict[str, Any]:
        """
        Start a new game or reset existing session.
        
        Args:
            game_id: Game identifier (e.g., 'ls20-016295f7601e')
            scorecard_id: Scorecard ID for tracking results
            reset_existing: If True and we have a current session, reset it
            
        Returns:
            API response with initial game state
        """
        url = f"{self.base_url}/api/cmd/RESET"
        
        payload = {
            "game_id": game_id,
            "card_id": scorecard_id
        }
        
        # Include GUID for reset if we have an existing session
        if reset_existing and self.current_guid:
            payload["guid"] = self.current_guid
        
        try:
            response = self.session.post(url, json=payload)
            
            # Handle HTTP errors with JSON error messages
            if response.status_code == 400:
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        return {'error': f'API Error: {error_data["error"]}', 'details': error_data}
                except:
                    pass
                return {'error': f'Bad Request (400): {response.text}'}
            
            response.raise_for_status()
            
            result = response.json()
            
            # Update session state
            self.current_game_id = result.get('game_id')
            self.current_guid = result.get('guid')
            self.current_scorecard_id = scorecard_id
            self.current_score = result.get('score', 0)
            self.win_score = result.get('win_score', 0)
            
            # Clear action history for new game
            if not reset_existing:
                self.action_history.clear()
            
            self.last_response = result
            
            return result
            
        except requests.exceptions.RequestException as e:
            return {'error': f'Failed to start game: {str(e)}'}
    
    def execute_action(self, action_number: int, coordinates: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Execute an action with proper parameter handling.
        
        Args:
            action_number: Action to execute (1-7)
            coordinates: Required for ACTION6, optional for others
            
        Returns:
            API response with updated game state
        """
        if not self.current_game_id or not self.current_guid:
            return {'error': 'No active game session. Call start_game() first.'}
        
        # Validate action number
        if action_number not in range(1, 8):
            return {'error': f'Invalid action number: {action_number}. Must be 1-7.'}
        
        # Handle ACTION6 coordinate requirements
        if action_number == 6:
            if coordinates is None:
                return {'error': 'ACTION6 requires coordinates (x, y)'}
            
            x, y = coordinates
            # Validate coordinate bounds
            if not (0 <= x <= 63 and 0 <= y <= 63):
                return {'error': f'Coordinates ({x}, {y}) out of bounds. Must be 0-63.'}
            
            return self._execute_action6(x, y)
        else:
            return self._execute_simple_action(action_number)
    
    def _execute_action6(self, x: int, y: int) -> Dict[str, Any]:
        """
        Execute ACTION6 with coordinates.
        """
        url = f"{self.base_url}/api/cmd/ACTION6"
        
        payload = {
            "game_id": self.current_game_id,
            "guid": self.current_guid,
            "x": int(x),
            "y": int(y)
        }
        
        try:
            response = self.session.post(url, json=payload)
            
            # Handle HTTP errors with JSON error messages
            if response.status_code == 400:
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        return {'error': f'API Error: {error_data["error"]}', 'details': error_data}
                except:
                    pass
                return {'error': f'Bad Request (400): {response.text}'}
            
            response.raise_for_status()
            
            result = response.json()
            self._update_session_state(result)
            
            # Track this action
            action_record = {
                'action': 6,
                'coordinates': (x, y),
                'score_before': self.current_score,
                'score_after': result.get('score', self.current_score),
                'timestamp': datetime.now().isoformat(),
                'response': result
            }
            self.action_history.append(action_record)
            
            return result
            
        except requests.exceptions.RequestException as e:
            return {'error': f'Failed to execute ACTION6: {str(e)}'}
    
    def _execute_simple_action(self, action_number: int) -> Dict[str, Any]:
        """
        Execute simple actions (ACTION1-5, ACTION7).
        """
        url = f"{self.base_url}/api/cmd/ACTION{action_number}"
        
        payload = {
            "game_id": self.current_game_id,
            "guid": self.current_guid
        }
        
        try:
            response = self.session.post(url, json=payload)
            
            # Handle HTTP errors with JSON error messages
            if response.status_code == 400:
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        return {'error': f'API Error: {error_data["error"]}', 'details': error_data}
                except:
                    pass
                return {'error': f'Bad Request (400): {response.text}'}
            
            response.raise_for_status()
            
            result = response.json()
            self._update_session_state(result)
            
            # Track this action
            action_record = {
                'action': action_number,
                'score_before': self.current_score,
                'score_after': result.get('score', self.current_score),
                'timestamp': datetime.now().isoformat(),
                'response': result
            }
            self.action_history.append(action_record)
            
            return result
            
        except requests.exceptions.RequestException as e:
            return {'error': f'Failed to execute ACTION{action_number}: {str(e)}'}
    
    def _update_session_state(self, response: Dict[str, Any]):
        """
        Update internal state based on API response.
        """
        self.current_score = response.get('score', self.current_score)
        self.win_score = response.get('win_score', self.win_score)
        self.last_response = response
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current game state information.
        """
        return {
            'game_id': self.current_game_id,
            'guid': self.current_guid,
            'scorecard_id': self.current_scorecard_id,
            'current_score': self.current_score,
            'win_score': self.win_score,
            'actions_taken': len(self.action_history),
            'last_response': self.last_response
        }
    
    def get_available_actions(self) -> List[int]:
        """
        Get currently available actions from last response.
        """
        if self.last_response and 'available_actions' in self.last_response:
            return self.last_response['available_actions']
        return []
    
    def get_current_frame(self) -> Optional[List[List[List[int]]]]:
        """
        Get current game frame data.
        """
        if self.last_response and 'frame' in self.last_response:
            return self.last_response['frame']
        return None
    
    def get_score_progress(self) -> Dict[str, float]:
        """
        Get score progress information.
        """
        if self.win_score == 0:
            return {'progress_ratio': 0.0, 'remaining_score': 0}
        
        progress_ratio = self.current_score / self.win_score
        remaining_score = self.win_score - self.current_score
        
        return {
            'progress_ratio': progress_ratio,
            'remaining_score': remaining_score,
            'current_score': self.current_score,
            'win_score': self.win_score
        }
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        """
        Get history of actions taken in current session.
        """
        return self.action_history.copy()
    
    def is_game_finished(self) -> bool:
        """
        Check if current game has ended.
        """
        if not self.last_response:
            return False
        
        state = self.last_response.get('state', 'NOT_STARTED')
        return state in ['WIN', 'GAME_OVER', 'NOT_STARTED']
    
    def get_game_result(self) -> Optional[str]:
        """
        Get game result if finished.
        """
        if not self.last_response:
            return None
        
        state = self.last_response.get('state')
        if state in ['WIN', 'GAME_OVER']:
            return state
        return None


class CoordinateManager:
    """
    Helper class for managing coordinates with bounds checking and strategic positioning.
    """
    
    GRID_SIZE = 64
    MIN_COORD = 0
    MAX_COORD = 63
    
    @staticmethod
    def clamp_coordinates(x: int, y: int) -> Tuple[int, int]:
        """
        Ensure coordinates are within valid bounds.
        """
        clamped_x = max(CoordinateManager.MIN_COORD, min(CoordinateManager.MAX_COORD, x))
        clamped_y = max(CoordinateManager.MIN_COORD, min(CoordinateManager.MAX_COORD, y))
        return (clamped_x, clamped_y)
    
    @staticmethod
    def get_center_coordinates() -> Tuple[int, int]:
        """
        Get center coordinates for initial positioning.
        """
        return (32, 32)
    
    @staticmethod
    def get_corner_coordinates() -> List[Tuple[int, int]]:
        """
        Get corner coordinates for exploration.
        """
        return [
            (0, 0),                                                    # Top-left
            (CoordinateManager.MAX_COORD, 0),                         # Top-right
            (0, CoordinateManager.MAX_COORD),                         # Bottom-left
            (CoordinateManager.MAX_COORD, CoordinateManager.MAX_COORD) # Bottom-right
        ]
    
    @staticmethod
    def calculate_movement(current_x: int, current_y: int, direction: str, distance: int = 1) -> Tuple[int, int]:
        """
        Calculate new coordinates based on movement direction.
        
        Args:
            current_x, current_y: Current position
            direction: 'up', 'down', 'left', 'right', or diagonal combinations
            distance: How far to move
            
        Returns:
            New coordinates (clamped to bounds)
        """
        new_x, new_y = current_x, current_y
        
        if 'up' in direction:
            new_y -= distance
        if 'down' in direction:
            new_y += distance
        if 'left' in direction:
            new_x -= distance  
        if 'right' in direction:
            new_x += distance
        
        return CoordinateManager.clamp_coordinates(new_x, new_y)
    
    @staticmethod
    def get_strategic_positions(current_pos: Optional[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        """
        Get strategic coordinate positions for exploration.
        """
        positions = [
            CoordinateManager.get_center_coordinates(),  # Center
            (16, 16),                                    # Upper-left quadrant center
            (48, 16),                                    # Upper-right quadrant center  
            (16, 48),                                    # Lower-left quadrant center
            (48, 48),                                    # Lower-right quadrant center
        ]
        
        # Add corner exploration
        positions.extend(CoordinateManager.get_corner_coordinates())
        
        # Add edge midpoints
        positions.extend([
            (32, 0),   # Top edge center
            (32, 63),  # Bottom edge center
            (0, 32),   # Left edge center
            (63, 32),  # Right edge center
        ])
        
        return positions
    
    @staticmethod
    def distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Calculate distance between two positions.
        """
        x1, y1 = pos1
        x2, y2 = pos2
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
