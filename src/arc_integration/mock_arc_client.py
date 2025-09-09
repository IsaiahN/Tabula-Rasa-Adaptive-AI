"""
Mock ARC API Client for local development and testing.

This module provides a mock implementation of the ARC API client that can be used
for local development and testing without requiring a connection to the real ARC API.
"""
import asyncio
import random
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from .arc_api_client import GameState, ARCClient

class MockARCClient(ARCClient):
    """Mock implementation of the ARC API client for local testing."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the mock client with default values."""
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.game_id = f"mock-game-{random.randint(1000, 9999)}"
        self.guid = f"mock-guid-{random.randint(1000, 9999)}"
        self.initialized = False
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.initialized = True
        self.logger.info("Initialized mock ARC client")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.initialized = False
        
    async def reset_game(self, game_id: str = None) -> GameState:
        """Reset the current game or start a new one with mock data."""
        self.logger.info(f"Resetting game with ID: {game_id or 'new'}")
        
        # Generate a mock game state
        game_state = GameState(
            game_id=self.game_id,
            guid=self.guid,
            frame=0,
            state="running",
            score=0.0,
            win_score=100.0,
            action_input=None,
            available_actions=list(range(1, 7))  # Actions 1-6 are available
        )
        
        return game_state
        
    async def send_action(self, action_id: int, **kwargs) -> GameState:
        """Simulate sending an action to the game."""
        if not self.initialized:
            raise RuntimeError("Client not initialized. Use async with statement.")
            
        self.logger.info(f"Sending action {action_id} with params: {kwargs}")
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        # Generate a mock response with a small score increase
        score_increase = random.uniform(0.1, 5.0)
        
        return GameState(
            game_id=self.game_id,
            guid=self.guid,
            frame=kwargs.get('frame', 0) + 1,
            state="running",
            score=min(100.0, score_increase * 5),  # Cap score at 100
            win_score=100.0,
            action_input={"action_id": action_id, **kwargs},
            available_actions=list(range(1, 7))  # All actions remain available
        )
        
    async def close(self):
        """Close the mock client."""
        self.initialized = False
        self.logger.info("Closed mock ARC client")
