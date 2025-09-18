"""
Database-based action and session trace logger.

All action traces and session data are now stored in the database
instead of creating files in the data/ directory.
"""

import json
import time
import threading
import logging
from typing import Optional

_logger = logging.getLogger(__name__)

# Thread safety
_lock = threading.Lock()


def log_action_trace(record: dict) -> None:
    """Log action trace to database instead of file.

    The record should be JSON-serializable. This function never raises on write
    errors (it logs them to stderr) to avoid disrupting the training loop.
    """
    try:
        # Use database integration instead of file I/O
        from src.database.system_integration import get_system_integration
        import asyncio
        
        integration = get_system_integration()
        
        # Log action trace to database
        coordinates = record.get('coordinates', None)
        if coordinates is None:
            # Try to get coordinates from x,y fields
            x = record.get('x')
            y = record.get('y')
            if x is not None and y is not None:
                coordinates = (x, y)
        
        if isinstance(coordinates, (tuple, list)):
            coordinates = json.dumps(coordinates)
        
        # Use create_task instead of run to avoid event loop conflicts
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule the coroutine to run in the background
                asyncio.create_task(integration.log_action_trace(
                    record.get('session_id', 'unknown'),
                    record.get('game_id', 'unknown'),
                    record.get('action_number', 0),
                    coordinates,
                    record.get('frame_before', None),
                    record.get('frame_after', None),
                    record.get('frame_changed', False),
                    record.get('score_before', 0),
                    record.get('score_after', 0),
                    record.get('response_data', None)
                ))
            else:
                # No event loop running, use run
                asyncio.run(integration.log_action_trace(
                    record.get('session_id', 'unknown'),
                    record.get('game_id', 'unknown'),
                    record.get('action_number', 0),
                    coordinates,
                    record.get('frame_before', None),
                    record.get('frame_after', None),
                    record.get('frame_changed', False),
                    record.get('score_before', 0),
                    record.get('score_after', 0),
                    record.get('response_data', None)
                ))
        except RuntimeError:
            # Fallback: run in new thread
            import threading
            def run_in_thread():
                asyncio.run(integration.log_action_trace(
                    record.get('session_id', 'unknown'),
                    record.get('game_id', 'unknown'),
                    record.get('action_number', 0),
                    coordinates,
                    record.get('frame_before', None),
                    record.get('frame_after', None),
                    record.get('frame_changed', False),
                    record.get('score_before', 0),
                    record.get('score_after', 0),
                    record.get('response_data', None)
                ))
            threading.Thread(target=run_in_thread, daemon=True).start()
        
    except Exception as e:
        try:
            print(f"⚠️ Failed to log action trace: {e}")
        except Exception:
            pass


def write_session_trace(game_id: str, session_result: dict, raw_output: str = None) -> None:
    """Write session trace to database instead of files.
    
    Args:
        game_id: Unique game identifier
        session_result: Dictionary containing session results
        raw_output: Raw output string (optional)
    """
    try:
        # Use database storage instead of file I/O
        from src.database.system_integration import get_system_integration
        import asyncio
        
        integration = get_system_integration()
        
        # Store session data in database
        asyncio.run(integration.update_session_metrics(
            session_id=game_id,
            metrics={
                "final_score": session_result.get("final_score") if isinstance(session_result, dict) else None,
                "total_actions": session_result.get("total_actions") if isinstance(session_result, dict) else None,
                "raw_output": raw_output,
                "session_result": session_result
            }
        ))
        
    except Exception as e:
        try:
            print(f"⚠️ Failed to write session trace for {game_id}: {e}")
        except Exception:
            pass


def cleanup_old_traces(max_age_days: int = 7) -> None:
    """Cleanup old traces from database.
    
    Args:
        max_age_days: Maximum age of traces to keep (in days)
    """
    try:
        # Database cleanup is handled by the database managers
        # This function is kept for compatibility but does nothing
        pass
    except Exception as e:
        try:
            print(f"⚠️ Failed to cleanup old traces: {e}")
        except Exception:
            pass