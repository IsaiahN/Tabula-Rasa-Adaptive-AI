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
    frame: List[List[List[str]]]
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
            score=data["score"],
            win_score=data["win_score"],
            action_input=data["action_input"],
            available_actions=data["available_actions"]
        )

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
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.api_key = api_key or os.getenv("ARC_API_KEY")
        if not self.api_key:
            raise ARCAuthenticationError("API key not provided and ARC_API_KEY environment variable not set")
            
        self.base_url = base_url or DEFAULT_BASE_URL
        self.session = None
        self.game_id = None
        self.guid = None
        self.available_actions = []  # Will be populated on game start/reset
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Log client initialization (without exposing the full API key)
        self.logger.info(f"Initialized ARC client with base URL: {self.base_url}")
        
    async def __aenter__(self):
        """Async context manager entry."""
        # Create a custom connector with SSL verification disabled
        # WARNING: Disabling SSL verification is not recommended for production use
        conn = aiohttp.TCPConnector(ssl=False)
        
        # Ensure base_url ends with a single slash
        base_url = self.base_url.rstrip('/') + '/api/cmd/'
        
        self.session = aiohttp.ClientSession(
            connector=conn,
            headers=self.headers,
            base_url=base_url
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a request to the ARC API with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
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
            
        # Ensure endpoint starts with a single slash
        endpoint = '/' + endpoint.lstrip('/')
        
        # Prepare headers
        headers = kwargs.pop('headers', {})
        headers.update(self.headers)
        
        max_retries = 3
        retry_delay = 1
        last_error = None
        
        self.logger.debug(f"Making {method} request to {endpoint}")
        
        for attempt in range(max_retries):
            try:
                # Log request details (without sensitive data)
                log_data = {k: v for k, v in kwargs.items() if k != 'json'}
                log_data['endpoint'] = endpoint
                self.logger.debug(f"Attempt {attempt + 1}/{max_retries}: {method} {log_data}")
                
                async with self.session.request(
                    method=method,
                    url=endpoint,  # base_url is already set in the session
                    headers=headers,
                    **kwargs
                ) as response:
                    # Try to parse JSON response
                    try:
                        data = await response.json() if response.content_length else {}
                    except (json.JSONDecodeError, aiohttp.ContentTypeError) as e:
                        text = await response.text()
                        self.logger.warning(f"Failed to parse JSON response: {text}")
                        data = {}
                    
                    # Log response status
                    self.logger.debug(f"Response status: {response.status} - {response.reason}")
                    
                    # Handle authentication errors
                    if response.status == 401:
                        error_msg = data.get('detail', 'Authentication failed - invalid API key')
                        self.logger.error(f"Authentication error: {error_msg}")
                        raise ARCAuthenticationError(error_msg)
                    
                    # Handle other error status codes
                    if response.status >= 400:
                        error_msg = data.get('detail', 'Unknown error')
                        if isinstance(error_msg, dict):
                            error_msg = json.dumps(error_msg)
                        self.logger.error(f"API error {response.status}: {error_msg}")
                        raise ARCAPIError(
                            f"API request failed with status {response.status}: {error_msg}"
                        )
                    
                    return data
                    
            except aiohttp.ClientError as e:
                last_error = e
                self.logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                continue
                
        error_msg = f"Failed to complete request after {max_retries} attempts: {str(last_error)}"
        self.logger.error(error_msg)
        raise ARCError(error_msg)
    
    async def reset_game(self, game_id: str = None) -> GameState:
        """Reset the current game or start a new one.
        
        Args:
            game_id: Optional game ID to reset. If not provided, starts a new game.
            
        Returns:
            GameState: The initial game state after reset.
        """
        endpoint = RESET_ENDPOINT
        params = {}
        if game_id:
            params["game_id"] = game_id
            
        response = await self._make_request(
            "POST",
            endpoint,
            params=params
        )
        
        # Update game state
        self.game_id = response.get("game_id")
        self.guid = response.get("guid")
        self.available_actions = response.get("available_actions", [])
        
        return GameState.from_dict(response)
    
    async def send_action(self, action_id: int, **kwargs) -> GameState:
        """Send an action to the game.
        
        Args:
            action_id: The action ID (1-7)
            **kwargs: Additional parameters for the action
                For ACTION6, must include x and y coordinates
                
        Returns:
            GameState: The new game state after the action
            
        Raises:
            ValueError: If action_id is invalid or required parameters are missing
            ARCAPIError: If the API returns an error
        """
        # Map action ID to endpoint
        action_endpoints = {
            1: ACTION1_ENDPOINT,
            2: ACTION2_ENDPOINT,
            3: ACTION3_ENDPOINT,
            4: ACTION4_ENDPOINT,
            5: ACTION5_ENDPOINT,
            6: ACTION6_ENDPOINT,
            7: ACTION7_ENDPOINT
        }
        
        if action_id not in action_endpoints:
            raise ValueError(f"Invalid action_id: {action_id}. Must be between 1 and 7.")
            
        # Validate required parameters for specific actions
        if action_id == 6 and not all(k in kwargs for k in ('x', 'y')):
            raise ValueError("ACTION6 requires x and y coordinates")
            
        # Prepare request data
        data = {"id": action_id, "data": kwargs}
        
        # Make the request
        response = await self._make_request(
            "POST",
            action_endpoints[action_id],
            json=data
        )
        
        # Update internal state
        self.game_id = response.get("game_id", self.game_id)
        self.guid = response.get("guid", self.guid)
        self.available_actions = response.get("available_actions", getattr(self, 'available_actions', []))
        
        return GameState.from_dict(response)
    
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
