"""
Scorecard Manager

Manages scorecard API integration for tracking training performance.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class ScorecardManager:
    """Manages scorecard API integration."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.active_scorecard_id: Optional[str] = None
        self.scorecard_stats = {
            'total_level_completions': 0,
            'total_games_completed': 0,
            'total_wins': 0,
            'total_played': 0,
            'total_actions': 0,
            'total_score': 0
        }
        self.scorecard_api_manager = None
        self._initialize_scorecard_api()
    
    def _initialize_scorecard_api(self) -> None:
        """Initialize scorecard API manager."""
        try:
            from src.arc_integration.scorecard_api import ScorecardAPIManager, get_api_key_from_config
            if not self.api_key:
                self.api_key = get_api_key_from_config()
            self.scorecard_api_manager = ScorecardAPIManager(self.api_key)
            logger.info("Scorecard API manager initialized")
        except ImportError as e:
            logger.warning(f"Scorecard API not available: {e}")
            self.scorecard_api_manager = None
        except Exception as e:
            logger.error(f"Error initializing scorecard API: {e}")
            self.scorecard_api_manager = None
    
    async def create_scorecard(self, name: str, description: str = "") -> Optional[str]:
        """Create a new scorecard."""
        if not self.scorecard_api_manager:
            logger.warning("Scorecard API not available")
            return None
        
        try:
            # create_scorecard is synchronous, not async
            scorecard_id = self.scorecard_api_manager.create_scorecard(name, description)
            self.active_scorecard_id = scorecard_id
            logger.info(f"Created scorecard: {scorecard_id}")
            return scorecard_id
        except Exception as e:
            logger.error(f"Error creating scorecard: {e}")
            return None
    
    async def submit_score(self, game_id: str, score: float, level: int = 1, 
                          actions_taken: int = 0, win: bool = False) -> bool:
        """Submit a score to the active scorecard."""
        if not self.scorecard_api_manager or not self.active_scorecard_id:
            logger.warning("No active scorecard for score submission")
            return False
        
        try:
            success = await self.scorecard_api_manager.submit_score(
                self.active_scorecard_id,
                game_id,
                score,
                level,
                actions_taken,
                win
            )
            
            if success:
                self._update_stats(score, actions_taken, win)
                logger.info(f"Score submitted: game={game_id}, score={score}, win={win}")
            
            return success
        except Exception as e:
            logger.error(f"Error submitting score: {e}")
            return False
    
    def _update_stats(self, score: float, actions_taken: int, win: bool) -> None:
        """Update internal scorecard statistics."""
        self.scorecard_stats['total_played'] += 1
        self.scorecard_stats['total_actions'] += actions_taken
        self.scorecard_stats['total_score'] += score
        
        if win:
            self.scorecard_stats['total_wins'] += 1
            self.scorecard_stats['total_games_completed'] += 1
    
    async def get_scorecard_status(self) -> Dict[str, Any]:
        """Get current scorecard status."""
        if not self.scorecard_api_manager or not self.active_scorecard_id:
            return {
                'active_scorecard_id': None,
                'api_available': False,
                'stats': self.scorecard_stats
            }
        
        try:
            status = await self.scorecard_api_manager.get_scorecard_status(self.active_scorecard_id)
            status.update({
                'active_scorecard_id': self.active_scorecard_id,
                'api_available': True,
                'stats': self.scorecard_stats
            })
            return status
        except Exception as e:
            logger.error(f"Error getting scorecard status: {e}")
            return {
                'active_scorecard_id': self.active_scorecard_id,
                'api_available': False,
                'error': str(e),
                'stats': self.scorecard_stats
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current scorecard statistics."""
        return self.scorecard_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset scorecard statistics."""
        self.scorecard_stats = {
            'total_level_completions': 0,
            'total_games_completed': 0,
            'total_wins': 0,
            'total_played': 0,
            'total_actions': 0,
            'total_score': 0
        }
        logger.info("Scorecard stats reset")
    
    def set_active_scorecard(self, scorecard_id: str) -> None:
        """Set the active scorecard ID."""
        self.active_scorecard_id = scorecard_id
        logger.info(f"Set active scorecard: {scorecard_id}")
    
    def get_active_scorecard_id(self) -> Optional[str]:
        """Get the active scorecard ID."""
        return self.active_scorecard_id
    
    async def close(self) -> None:
        """Close the scorecard manager."""
        if self.scorecard_api_manager:
            try:
                await self.scorecard_api_manager.close()
            except Exception as e:
                logger.warning(f"Error closing scorecard API manager: {e}")
        self.scorecard_api_manager = None
        logger.info("Scorecard manager closed")
