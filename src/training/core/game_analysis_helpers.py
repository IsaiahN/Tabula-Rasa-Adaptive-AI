from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

async def _record_and_analyze_game_state(self, 
                                  state: Dict[str, Any],
                                  action: Dict[str, Any],
                                  game_type: str,
                                  moves_made: int) -> None:
    """Record and analyze the current game state.
    
    Args:
        state: Current game state
        action: Action taken
        game_type: Type of game being played
        moves_made: Number of moves made so far
    """
    try:
        # Record the state
        self.game_analyzer.record_game_state(
            state=state['frame'],
            action=str(action['id']),
            score=float(state.get('score', 0))
        )
        
        # Do mid-game analysis
        await self.game_analyzer.do_mid_game_analysis(
            game_type=game_type,
            moves_made=moves_made
        )
        
    except Exception as e:
        logger.error(f"Error in game state analysis: {e}")

async def _do_post_game_analysis(self,
                              state: Dict[str, Any],
                              game_type: str,
                              win: bool) -> None:
    """Do post-game analysis.
    
    Args:
        state: Final game state
        game_type: Type of game played
        win: Whether the game was won
    """
    try:
        await self.game_analyzer.do_post_game_analysis(
            game_type=game_type,
            final_score=float(state.get('score', 0)),
            win=win
        )
        
        # Reset analyzer for next game
        self.game_analyzer.reset()
        
    except Exception as e:
        logger.error(f"Error in post-game analysis: {e}")