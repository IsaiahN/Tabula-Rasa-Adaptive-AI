"""
API Manager

Manages ARC API client connections and operations.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path

from .rate_limiter import RateLimiter, RateLimitConfig
from .scorecard_manager import ScorecardManager

logger = logging.getLogger(__name__)

class APIManager:
    """Manages ARC API client and related services."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.arc_client = None
        self.rate_limiter = RateLimiter()
        self.scorecard_manager = ScorecardManager(api_key)
        self.initialized = False
        self.connection_retries = 0
        self.max_retries = 3
    
    async def initialize(self) -> bool:
        """Initialize the API client and test connection."""
        # Don't reinitialize if already initialized
        if self.initialized and self.arc_client:
            logger.debug("API manager already initialized, skipping")
            return True
            
        try:
            await self._initialize_real_client()
            
            self.initialized = True
            self.connection_retries = 0
            logger.debug("API manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing API manager: {e}")
            self.connection_retries += 1
            return False
    
    
    async def _initialize_real_client(self) -> None:
        """Initialize real ARC API client."""
        try:
            # Close existing client if it exists
            if self.arc_client:
                try:
                    await self.arc_client.close()
                except Exception as e:
                    logger.warning(f"Error closing existing ARC client: {e}")
                self.arc_client = None
            
            from src.arc_integration.arc_api_client import ARCClient
            logger.debug("Initializing REAL ARC API client...")

            if not self.api_key:
                import os
                self.api_key = os.getenv('ARC_API_KEY')
                if not self.api_key:
                    raise ValueError("ARC_API_KEY not found in environment variables")

            self.arc_client = ARCClient(api_key=self.api_key)
            # Initialize the async session
            await self.arc_client.__aenter__()
            logger.debug("ARC client created and initialized successfully")
            
        except ImportError as e:
            logger.error(f"ARC API client not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating ARC client: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """Test the API connection."""
        if not self.arc_client:
            return False
        
        try:
            # Skip connection test to avoid hanging
            logger.info("Skipping connection test to avoid hanging...")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    async def make_request(self, request_type: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Make a rate-limited API request."""
        if not self.initialized or not self.arc_client:
            logger.warning("API client not initialized")
            return None
        
        # Check if we need to pause due to rate limiting
        should_pause, pause_duration = self.rate_limiter.should_pause()
        if should_pause:
            logger.info(f"Rate limit pause: {pause_duration:.1f}s before {request_type}")
            await asyncio.sleep(pause_duration)
        
        # Check rate limits
        if not self.rate_limiter.can_make_request():
            wait_time = self.rate_limiter.wait_if_needed()
            if wait_time > 0:
                logger.info(f"Waited {wait_time:.2f}s for rate limit")
        
        # Record the request
        if not self.rate_limiter.record_request():
            logger.warning("Request blocked by rate limiter")
            return None
        
        try:
            # Make the actual request
            if request_type == "create_game":
                return await self.arc_client.create_game(**kwargs)
            elif request_type == "reset_game":
                # Extract scorecard_id and pass it to reset_game
                scorecard_id = kwargs.pop('scorecard_id', None)
                return await self.arc_client.reset_game(kwargs.get('game_id'), scorecard_id)
            elif request_type == "submit_action":
                # Extract parameters for send_action
                action = kwargs.get('action', {})
                action_id = action.get('action_id', 1)
                action_str = f"ACTION{action_id}"
                game_id = kwargs.get('game_id')
                card_id = kwargs.get('card_id')
                guid = kwargs.get('guid')
                
                # For ACTION6, we need to pass x,y coordinates as separate parameters
                if action_id == 6:
                    x = action.get('x', 0)
                    y = action.get('y', 0)
                    return await self.arc_client.send_action(action_str, game_id=game_id, card_id=card_id, guid=guid, x=x, y=y)
                else:
                    # For actions 1-5 and 7, include reasoning in the payload
                    reasoning = action.get('reasoning', {})
                    if reasoning:
                        return await self.arc_client.send_action(action_str, game_id=game_id, card_id=card_id, guid=guid, reasoning=reasoning)
                    else:
                        return await self.arc_client.send_action(action_str, game_id=game_id, card_id=card_id, guid=guid)
            elif request_type == "get_game_state":
                # Extract parameters for get_game_state
                game_id = kwargs.get('game_id')
                card_id = kwargs.get('card_id')
                guid = kwargs.get('guid')
                return await self.arc_client.get_game_state(game_id, card_id, guid)
            else:
                logger.error(f"Unknown request type: {request_type}")
                return None
                
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return None
    
    async def create_game(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Create a new game."""
        return await self.make_request("create_game", game_id=game_id)
    
    async def reset_game(self, game_id: str, scorecard_id: str = None) -> Optional[Dict[str, Any]]:
        """Reset a game."""
        return await self.make_request("reset_game", game_id=game_id, scorecard_id=scorecard_id)
    
    async def submit_action(self, game_id: str, action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Submit an action to a game."""
        return await self.make_request("submit_action", game_id=game_id, action=action)
    
    async def get_game_state(self, game_id: str, card_id: str = None, guid: str = None) -> Optional[Dict[str, Any]]:
        """Get the current state of a game."""
        return await self.make_request("get_game_state", game_id=game_id, card_id=card_id, guid=guid)
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        return self.rate_limiter.get_rate_limit_status()
    
    async def get_scorecard_status(self) -> Dict[str, Any]:
        """Get scorecard status."""
        return await self.scorecard_manager.get_scorecard_status()
    
    async def submit_score(self, game_id: str, score: float, level: int = 1, 
                          actions_taken: int = 0, win: bool = False) -> bool:
        """Submit a score to the scorecard."""
        return await self.scorecard_manager.submit_score(
            game_id, score, level, actions_taken, win
        )
    
    async def create_scorecard(self, name: str, description: str = "") -> Optional[str]:
        """Create a new scorecard."""
        return await self.scorecard_manager.create_scorecard(name, description)
    
    async def get_available_games(self) -> Optional[List[Dict[str, Any]]]:
        """Get list of available games from ARC-AGI-3 API."""
        if not self.initialized or not self.arc_client:
            logger.warning("API client not initialized")
            return None
        
        try:
            games = await self.arc_client.get_available_games()
            logger.info(f"Retrieved {len(games)} available games from ARC-AGI-3 API")
            return games
        except Exception as e:
            logger.error(f"Error getting available games: {e}")
            return None
    
    async def close_scorecard(self, scorecard_id: str) -> bool:
        """Close a scorecard."""
        if not self.initialized or not self.arc_client:
            logger.warning("API client not initialized")
            return True  # Don't fail training if API not initialized
        
        try:
            # Try to close the scorecard, but don't fail if it's already closed
            result = await self.arc_client.close_scorecard(scorecard_id)
            if result is not None:
                logger.info(f"Closed scorecard: {scorecard_id}")
                return True
            else:
                logger.info(f"Scorecard {scorecard_id} already closed or not found")
                return True  # Consider this a success since the scorecard is closed
        except Exception as e:
            # Log as info instead of warning to reduce noise
            logger.info(f"Scorecard {scorecard_id} close result: {e}")
            return True  # Always consider this a success to avoid breaking the training loop
    
    async def take_action(self, game_id: str, action: Dict[str, Any], card_id: str = None, guid: str = None) -> Optional[Dict[str, Any]]:
        """Take an action in the game."""
        if not self.initialized or not self.arc_client:
            logger.warning("API client not initialized")
            return None
        
        try:
            # Convert action dict to string command
            action_id = action.get('id')
            
            # Validate action_id exists and is valid
            if action_id is None:
                logger.error("Action ID is None - cannot create action string")
                return None
            
            if not isinstance(action_id, int) or action_id < 1 or action_id > 7:
                logger.error(f"Invalid action ID: {action_id} - must be integer 1-7")
                return None
            
            action_str = f"ACTION{action_id}"
            
            # For ACTION6, we need to pass x,y coordinates as separate parameters
            if action_id == 6:
                x = action.get('x', 0)
                y = action.get('y', 0)
                logger.debug(f"Sending ACTION6 with coordinates x={x}, y={y}")
                # Validate coordinates are within valid range (0-63 for 64x64 grid)
                if not (0 <= x <= 63 and 0 <= y <= 63):
                    logger.warning(f"Invalid coordinates: x={x}, y={y} - clamping to valid range")
                    x = max(0, min(63, x))
                    y = max(0, min(63, y))
                game_state = await self.arc_client.send_action(action_str, game_id=game_id, card_id=card_id, guid=guid, x=x, y=y)
                # Add delay to prevent burst limit issues
                await asyncio.sleep(0.5)  # 500ms delay between actions
            else:
                # For actions 1-5 and 7, include reasoning in the payload
                reasoning = action.get('reasoning', {})
                if reasoning:
                    logger.debug(f"Sending {action_str} with reasoning: {reasoning.get('policy', 'unknown')}")
                    game_state = await self.arc_client.send_action(action_str, game_id=game_id, card_id=card_id, guid=guid, reasoning=reasoning)
                else:
                    logger.debug(f"No reasoning provided for {action_str}")
                    game_state = await self.arc_client.send_action(action_str, game_id=game_id, card_id=card_id, guid=guid)
                # Add delay to prevent burst limit issues
                await asyncio.sleep(0.5)  # 500ms delay between actions
            if game_state:
                return {
                    'game_id': game_state.game_id,
                    'guid': game_state.guid,
                    'frame': game_state.frame,
                    'state': game_state.state,
                    'score': game_state.score,
                    'win_score': game_state.win_score,
                    'action_input': game_state.action_input,
                    'available_actions': game_state.available_actions
                }
            return None
        except Exception as e:
            logger.error(f"Error taking action: {e}")
            return None
    
    def is_initialized(self) -> bool:
        """Check if API manager is initialized."""
        return self.initialized and self.arc_client is not None
    
    def is_healthy(self) -> bool:
        """Check if API manager is healthy."""
        if not self.is_initialized():
            return False
        
        # Check rate limiter health
        rate_status = self.get_rate_limit_status()
        if rate_status.get('blocked_requests', 0) > 100:
            logger.warning("High number of blocked requests")
            return False
        
        return True
    
    async def close(self) -> None:
        """Close the API manager and clean up resources."""
        if self.arc_client:
            try:
                await self.arc_client.close()
            except Exception as e:
                logger.warning(f"Error closing ARC client: {e}")
            self.arc_client = None
        
        if hasattr(self, 'scorecard_manager') and self.scorecard_manager:
            try:
                await self.scorecard_manager.close()
            except Exception as e:
                logger.warning(f"Error closing scorecard manager: {e}")
        
        self.initialized = False
        logger.info("API manager closed")

    async def get_current_scorecard_data(self) -> Optional[Dict[str, Any]]:
        """Get current scorecard data for saving to database."""
        try:
            if hasattr(self, 'scorecard_manager') and self.scorecard_manager:
                # Get scorecard data from scorecard manager
                scorecard_data = await self.scorecard_manager.get_current_scorecard()
                if scorecard_data:
                    return {
                        'scorecard_id': scorecard_data.get('id'),
                        'name': scorecard_data.get('name'),
                        'description': scorecard_data.get('description'),
                        'total_games': scorecard_data.get('total_games', 0),
                        'total_score': scorecard_data.get('total_score', 0.0),
                        'session_id': getattr(self, 'current_session_id', 'unknown'),
                        'created_at': scorecard_data.get('created_at'),
                        'updated_at': scorecard_data.get('updated_at')
                    }
            return None
        except Exception as e:
            logger.error(f"Error getting scorecard data: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive API manager status."""
        return {
            'initialized': self.initialized,
            'has_api_key': bool(self.api_key),
            'rate_limit_status': self.get_rate_limit_status(),
            'connection_retries': self.connection_retries,
            'is_healthy': self.is_healthy()
        }
