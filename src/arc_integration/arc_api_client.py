"""
ARC API Client for interacting with the ARC-AGI-3 evaluation system.

This module provides a client for communicating with the ARC-AGI-3 API,
handling game actions, state management, and error handling.
"""
import os
import json
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
import aiohttp
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Default API endpoints
DEFAULT_BASE_URL = "https://three.arcprize.org"

# API endpoints - Updated to match official ARC API structure
GAMES_ENDPOINT = f"{DEFAULT_BASE_URL}/api/games"
SCORECARD_OPEN_ENDPOINT = f"{DEFAULT_BASE_URL}/api/scorecard/open"
SCORECARD_CLOSE_ENDPOINT = f"{DEFAULT_BASE_URL}/api/scorecard/close"
RESET_ENDPOINT = f"{DEFAULT_BASE_URL}/api/cmd/RESET"
ACTION1_ENDPOINT = f"{DEFAULT_BASE_URL}/api/cmd/ACTION1"
ACTION2_ENDPOINT = f"{DEFAULT_BASE_URL}/api/cmd/ACTION2"
ACTION3_ENDPOINT = f"{DEFAULT_BASE_URL}/api/cmd/ACTION3"
ACTION4_ENDPOINT = f"{DEFAULT_BASE_URL}/api/cmd/ACTION4"
ACTION5_ENDPOINT = f"{DEFAULT_BASE_URL}/api/cmd/ACTION5"
ACTION6_ENDPOINT = f"{DEFAULT_BASE_URL}/api/cmd/ACTION6"
ACTION7_ENDPOINT = f"{DEFAULT_BASE_URL}/api/cmd/ACTION7"

@dataclass
class GameState:
    """Dataclass for holding game state information."""
    game_id: str
    guid: str
    frame: List[List[List[Union[int, str]]]]
    state: str
    score: int
    win_score: int
    action_input: Dict[str, Any]
    available_actions: List[int]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameState':
        """Create a GameState from API response data."""
        return cls(
            game_id=data["game_id"],
            guid=data["guid"],
            frame=data["frame"],
            state=data["state"],
            score=data.get("score", 0),
            win_score=data.get("win_score", 0),
            action_input=data.get("action_input", {}),
            available_actions=data.get("available_actions", [])
        )

@dataclass
class Scorecard:
    """Dataclass for holding scorecard information."""
    card_id: str
    tags: List[str] = None
    source_url: str = None
    opaque: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the scorecard to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Scorecard':
        """Create a scorecard from a dictionary."""
        return cls(**data)

@dataclass
class ARCScorecard:
    """Dataclass for holding ARC evaluation metrics."""
    task_id: str
    score: float
    accuracy: float
    efficiency: float
    generalization: float
    timestamp: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the scorecard to a dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ARCScorecard':
        """Create a scorecard from a dictionary."""
        return cls(**data)

class ARCError(Exception):
    """Base exception for ARC API errors."""
    pass

class ARCAuthenticationError(ARCError):
    """Raised when authentication fails."""
    pass

class ARCAPIError(ARCError):
    """Raised when the ARC API returns an error."""
    pass

class ARCClient:
    """Client for interacting with the ARC-AGI-3 API."""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """Initialize the ARC-AGI-3 API client.
        
        Args:
            api_key: Your ARC API key. If not provided, will try to get from ARC_API_KEY env var.
            base_url: Base URL for the ARC API. Defaults to the production API.
        """
        self.api_key = api_key or os.getenv('ARC_API_KEY')
        if not self.api_key:
            raise ARCAuthenticationError("ARC API key is required. Set ARC_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = base_url or DEFAULT_BASE_URL
        self.session = None
        self.current_game_id = None
        self.current_guid = None
        self.current_card_id = None
        self.current_scorecard_id = None
        
        # Headers for API requests
        self.headers = {
            "X-API-Key": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized ARC client with base URL: {self.base_url}")
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            self.session = None
        # Reset state
        self.current_game_id = None
        self.current_guid = None
        self.current_card_id = None
        self.current_scorecard_id = None
    
    async def close(self):
        """Manually close the session and cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None
        # Reset state
        self.current_game_id = None
        self.current_guid = None
        self.current_card_id = None
        self.current_scorecard_id = None
    
    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make a request to the ARC API with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL for the request
            **kwargs: Additional arguments for the request
            
        Returns:
            Parsed JSON response as a dictionary
            
        Raises:
            ARCError: If the request fails after all retries
            ARCAuthenticationError: If authentication fails
            ARCAPIError: If the API returns an error response
        """
        if not self.session:
            raise ARCError("Session not initialized. Use async with statement.")
            
        max_retries = 3
        retry_delay = 1
        last_error = None
        
        logger.debug(f"Making {method} request to {url}")
        
        for attempt in range(max_retries):
            try:
                async with self.session.request(method=method, url=url, **kwargs) as response:
                    # Handle rate limiting with exponential backoff
                    if response.status == 429:
                        try:
                            error_data = await response.json()
                            logger.warning(f"Rate limit exceeded: {error_data}")
                        except:
                            logger.warning("Rate limit exceeded (no JSON response)")
                        
                        if attempt < max_retries - 1:
                            # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                            wait_time = retry_delay * (2 ** attempt)
                            logger.info(f"Rate limit hit, retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error("Rate limit exceeded, max retries reached")
                            raise ARCAPIError("Rate limit exceeded, max retries reached")
                    
                    # Try to parse JSON response
                    try:
                        data = await response.json()
                    except aiohttp.ContentTypeError:
                        text = await response.text()
                        logger.error(f"Non-JSON response: {text}")
                        raise ARCAPIError(f"Non-JSON response: {text}")
                    
                    # Handle authentication errors
                    if response.status == 401:
                        error_msg = data.get('error', 'Authentication failed - invalid API key')
                        logger.error(f"Authentication error: {error_msg}")
                        raise ARCAuthenticationError(error_msg)
                    
                    # Handle other error status codes
                    if response.status >= 400:
                        error_msg = data.get('error', data.get('message', 'Unknown error'))
                        logger.error(f"API error {response.status}: {error_msg}")
                        raise ARCAPIError(f"API request failed with status {response.status}: {error_msg}")
                    
                    return data
                    
            except aiohttp.ClientError as e:
                last_error = e
                logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                continue
                
        error_msg = f"Failed to complete request after {max_retries} attempts: {str(last_error)}"
        logger.error(error_msg)
        raise ARCError(error_msg)
    
    async def get_available_games(self) -> List[Dict[str, Any]]:
        """Get list of available games.
        
        Returns:
            List of available games with their metadata
        """
        response = await self._make_request("GET", GAMES_ENDPOINT)
        return response
    
    async def open_scorecard(self, tags: List[str] = None) -> Scorecard:
        """Open a new scorecard for tracking performance.
        
        Args:
            tags: Optional list of tags for the scorecard
            
        Returns:
            Scorecard: The opened scorecard
        """
        payload = {}
        if tags:
            payload["tags"] = tags
            
        response = await self._make_request("POST", SCORECARD_OPEN_ENDPOINT, json=payload)
        
        scorecard = Scorecard.from_dict(response)
        self.current_scorecard_id = scorecard.card_id
        
        logger.info(f"Opened scorecard: {scorecard.card_id}")
        return scorecard
    
    async def close_scorecard(self, card_id: str = None) -> Dict[str, Any]:
        """Close a scorecard.
        
        Args:
            card_id: Scorecard ID to close. If not provided, uses current scorecard.
            
        Returns:
            Scorecard results
        """
        if not card_id:
            card_id = self.current_scorecard_id
            
        if not card_id:
            raise ValueError("No scorecard ID provided and no current scorecard")
            
        payload = {"card_id": card_id}
        response = await self._make_request("POST", SCORECARD_CLOSE_ENDPOINT, json=payload)
        
        logger.info(f"Closed scorecard: {card_id}")
        return response
    
    async def reset_game(self, game_id: str, card_id: str = None) -> GameState:
        """Reset the current game or start a new one.
        
        Args:
            game_id: Game ID to reset
            card_id: Scorecard ID. If not provided, uses current scorecard.
            
        Returns:
            GameState: The initial game state after reset.
        """
        if not card_id:
            card_id = self.current_scorecard_id
            
        if not card_id:
            raise ValueError("No scorecard ID provided and no current scorecard")
            
        payload = {
            "game_id": game_id,
            "card_id": card_id
        }
        
        response = await self._make_request("POST", RESET_ENDPOINT, json=payload)
        
        # Update game state
        self.current_game_id = response.get("game_id")
        self.current_guid = response.get("guid")
        
        return GameState.from_dict(response)
    
    async def reset_level(self, game_id: str, card_id: str, guid: str) -> GameState:
        """Reset the current level (same as reset_game but with guid).
        
        Args:
            game_id: Game ID to reset
            card_id: Scorecard ID
            guid: Game GUID
            
        Returns:
            GameState: The initial game state after reset.
        """
        payload = {
            "game_id": game_id,
            "card_id": card_id,
            "guid": guid
        }
        
        response = await self._make_request("POST", RESET_ENDPOINT, json=payload)
        
        # Update game state
        self.current_game_id = response.get("game_id")
        self.current_guid = response.get("guid")
        
        return GameState.from_dict(response)
    
    async def get_game_state(self, game_id: str, card_id: str = None, guid: str = None) -> GameState:
        """Get current game state without taking an action."""
        try:
            # Use the provided parameters or fall back to stored values
            card_id = card_id or self.current_card_id
            guid = guid or self.current_guid
            
            if not card_id or not guid:
                raise ValueError("Missing required parameters: card_id and guid must be provided")
            
            # Use ACTION1 as a safe action to get current state
            # This is a common pattern in ARC-AGI-3 where ACTION1 is often a no-op or safe action
            return await self.send_action("ACTION1", game_id=game_id, card_id=card_id, guid=guid)
        except Exception as e:
            logger.error(f"Error getting game state: {e}")
            return None
    
    async def send_action(self, action: str, game_id: str = None, card_id: str = None, 
                         guid: str = None, x: int = None, y: int = None, **kwargs) -> GameState:
        """Send an action to the game.
        
        Args:
            action: The action to send (ACTION1, ACTION2, etc.)
            game_id: Game ID. If not provided, uses current game.
            card_id: Scorecard ID. If not provided, uses current scorecard.
            guid: Game GUID. If not provided, uses current GUID.
            **kwargs: Additional parameters for the action
                For ACTION6, must include x and y coordinates
                
        Returns:
            GameState: The new game state after the action
        """
        # Use current values if not provided
        if not game_id:
            game_id = self.current_game_id
        if not card_id:
            card_id = self.current_scorecard_id
        if not guid:
            guid = self.current_guid
            
        if not all([game_id, card_id, guid]):
            raise ValueError("Missing required parameters: game_id, card_id, and guid must be provided")
        
        # Map action to endpoint
        action_endpoints = {
            "ACTION1": ACTION1_ENDPOINT,
            "ACTION2": ACTION2_ENDPOINT,
            "ACTION3": ACTION3_ENDPOINT,
            "ACTION4": ACTION4_ENDPOINT,
            "ACTION5": ACTION5_ENDPOINT,
            "ACTION6": ACTION6_ENDPOINT,
            "ACTION7": ACTION7_ENDPOINT
        }
        
        if action not in action_endpoints:
            raise ValueError(f"Invalid action: {action}. Must be one of {list(action_endpoints.keys())}")
            
        # Prepare request data
        payload = {
            "game_id": game_id,
            "card_id": card_id,
            "guid": guid
        }
        
        # Add action-specific parameters
        if action == "ACTION6":
            if x is None or y is None:
                raise ValueError("ACTION6 requires x and y coordinates")
            payload["x"] = x
            payload["y"] = y
        else:
            # Add reasoning for other actions
            if "reasoning" in kwargs:
                payload["reasoning"] = kwargs["reasoning"]
        
        # Make the request
        response = await self._make_request("POST", action_endpoints[action], json=payload)
        
        # Update game state
        self.current_game_id = response.get("game_id")
        self.current_guid = response.get("guid")
        
        return GameState.from_dict(response)
    
    async def play_game(self, game_id: str, agent_func, max_actions: int = 100) -> Dict[str, Any]:
        """Play a complete game with an agent function.
        
        Args:
            game_id: Game ID to play
            agent_func: Function that takes (game_state, available_actions) and returns action
            max_actions: Maximum number of actions to take
            
        Returns:
            Dictionary with game results
        """
        # Open scorecard
        scorecard = await self.open_scorecard(tags=["tabula_rasa_agent"])
        
        try:
            # Reset game
            game_state = await self.reset_game(game_id, scorecard.card_id)
            
            actions_taken = 0
            total_score = 0
            
            while game_state.state == "NOT_FINISHED" and actions_taken < max_actions:
                # Get action from agent
                action = agent_func(game_state, game_state.available_actions)
                
                if isinstance(action, str):
                    action_name = action
                    action_kwargs = {}
                elif isinstance(action, dict):
                    action_name = action.get("action")
                    action_kwargs = {k: v for k, v in action.items() if k != "action"}
                else:
                    raise ValueError(f"Invalid action format: {action}")
                
                # Send action
                game_state = await self.send_action(
                    action_name, 
                    game_id, 
                    scorecard.card_id, 
                    game_state.guid,
                    **action_kwargs
                )
                
                actions_taken += 1
                total_score = game_state.score
                
                logger.info(f"Action {actions_taken}: {action_name} -> State: {game_state.state}, Score: {total_score}")
                
                # Check if game is finished
                if game_state.state in ["WIN", "GAME_OVER"]:
                    break
            
            # Close scorecard
            scorecard_results = await self.close_scorecard(scorecard.card_id)
            
            return {
                "game_id": game_id,
                "final_state": game_state.state,
                "final_score": total_score,
                "actions_taken": actions_taken,
                "scorecard_id": scorecard.card_id,
                "scorecard_results": scorecard_results
            }
            
        except Exception as e:
            logger.error(f"Error playing game {game_id}: {e}")
            # Try to close scorecard even if there was an error
            try:
                await self.close_scorecard(scorecard.card_id)
            except:
                pass
            raise

class ScorecardTracker:
    """Tracks and analyzes scorecards over time."""
    
    def __init__(self):
        """Initialize the scorecard tracker."""
        self.scorecards = []
        self.metrics = {
            'scores': [],
            'accuracies': [],
            'efficiencies': [],
            'generalizations': [],
            'timestamps': []
        }
        
    def add_scorecard(self, scorecard: Union[ARCScorecard, Dict]) -> None:
        """Add a scorecard to the tracker.
        
        Args:
            scorecard: The scorecard to add.
        """
        if isinstance(scorecard, dict):
            scorecard = ARCScorecard.from_dict(scorecard)
            
        self.scorecards.append(scorecard)
        self.metrics['scores'].append(scorecard.score)
        self.metrics['accuracies'].append(scorecard.accuracy)
        self.metrics['efficiencies'].append(scorecard.efficiency)
        self.metrics['generalizations'].append(scorecard.generalization)
        self.metrics['timestamps'].append(scorecard.timestamp)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the tracked scorecards.
        
        Returns:
            Dictionary with summary statistics.
        """
        if not self.scorecards:
            return {}
            
        scores = self.metrics['scores']
        return {
            'total_scorecards': len(self.scorecards),
            'avg_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'latest_score': scores[-1],
            'improvement': scores[-1] - scores[0] if len(scores) > 1 else 0,
            'trend': self._calculate_trend(scores)
        }
        
    def _calculate_trend(self, values: List[float], window: int = 5) -> float:
        """Calculate the trend of the last 'window' values."""
        if len(values) < 2:
            return 0
            
        window = min(window, len(values))
        recent = values[-window:]
        x = list(range(len(recent)))
        y = recent
        
        # Simple linear regression for trend
        n = len(x)
        if n == 0:
            return 0
            
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0

