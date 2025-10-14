"""Game analyzer integration for continuous learning loop."""

from typing import Dict, Any, Optional
from src.analysis.game_analyzer import GameAnalyzer
from src.analysis.consolidation_manager import ConsolidationManager
from src.analysis.game_state_replay import GameStateReplay
import logging

logger = logging.getLogger(__name__)

class GameAnalyzerIntegration:
    """Provides functions for integrating game analysis into the training loop."""
    
    def __init__(self, db_path: str):
        """Initialize game analyzer integration.
        
        Args:
            db_path: Path to SQLite database
        """
        self.game_state_replay = GameStateReplay(db_path)
        self.consolidation_manager = ConsolidationManager(
            self.game_state_replay,
            db_path
        )
        self.game_analyzer = GameAnalyzer(self.consolidation_manager)
        
    def record_game_state(self, state: Any, action: str, score: float) -> None:
        """Record game state for analysis.
        
        Args:
            state: Current game state
            action: Action taken
            score: Current score
        """
        self.game_analyzer.record_game_state(state, action, score)
        
    async def do_pre_game_analysis(self) -> Dict[str, Any]:
        """Analyze past games to identify patterns and strategies before starting a new game.
        
        Returns:
            Dict containing insights from analysis
        """
        try:
            return await self.consolidation_manager.analyze_historical_patterns()
        except Exception as e:
            logger.error(f"Error in pre-game analysis: {e}")
            return {}
        
    async def do_mid_game_analysis(self, game_type: str, moves_made: int) -> Dict[str, Any]:
        """Perform mid-game analysis.
        
        Args:
            game_type: Type of game being played
            moves_made: Number of moves made so far
            
        Returns:
            Dict containing mid-game insights
        """
        try:
            return await self.game_analyzer.do_mid_game_analysis(game_type, moves_made)
        except Exception as e:
            logger.error(f"Error in mid-game analysis: {e}")
            return {}
        
    async def do_post_game_analysis(self, game_type: str, final_score: float, win: bool) -> Dict[str, Any]:
        """Perform post-game analysis.
        
        Args:
            game_type: Type of game played
            final_score: Final score achieved 
            win: Whether game was won
            
        Returns:
            Dict containing post-game insights
        """
        try:
            return await self.game_analyzer.do_post_game_analysis(game_type, final_score, win)
        except Exception as e:
            logger.error(f"Error in post-game analysis: {e}")
            return {}
        
    def reset(self) -> None:
        """Reset game analyzer state."""
        self.game_analyzer.reset()