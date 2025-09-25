"""Convenience helpers to persist in-memory events to the database.

These helpers provide a small, safe API used by higher-level modules when
they append to in-memory lists so the same data is also saved to the DB.
"""

from typing import Any, Dict, List, Optional
import json
import logging
from .api import get_database


# Deep debug logging setup
logger = logging.getLogger("tabula_rasa.persistence_helpers")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(name)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # File handler for deep debug
    fh = logging.FileHandler("persistence_helpers_debug.log", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


async def persist_winning_sequence(game_id: str, sequence: List[int], frequency: int = 1, avg_score: float = 0.0, success_rate: float = 1.0):
    print("[persist_winning_sequence] CALLED", game_id, sequence)
    logger.info(f"[persist_winning_sequence] CALLED with game_id={game_id}, sequence={sequence}")
    db = get_database()
    sql = (
        """
        INSERT OR REPLACE INTO winning_sequences
        (game_id, sequence, frequency, avg_score, success_rate, last_used)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """
    )
    params = (game_id, json.dumps(sequence), frequency, avg_score, success_rate)
    try:
        logger.debug(f"[persist_winning_sequence] DB: {db} | SQL: {sql.strip()} | Params: {params}")
        result = await db.execute(sql, params, table_name='winning_sequences')
        logger.info(f"[persist_winning_sequence] Success for game_id={game_id}, sequence={sequence} | Result: {result}")
        logger.debug(f"[persist_winning_sequence] Commit attempted for game_id={game_id}")
        return result
    except Exception as e:
        import traceback
        logger.error(f"[persist_winning_sequence] Exception: {e} | SQL: {sql.strip()} | Params: {params}\nTraceback: {traceback.format_exc()}")
        logger.debug(f"[persist_winning_sequence] Rollback attempted for game_id={game_id}")
        return False


async def persist_button_priorities(game_type: str, x: int, y: int, button_type: str, confidence: float = 0.0):
    print("[persist_button_priorities] CALLED", game_type, x, y, button_type)
    logger.info(f"[persist_button_priorities] CALLED with game_type={game_type}, x={x}, y={y}, button_type={button_type}")
    db = get_database()
    sql = (
        """
        INSERT OR REPLACE INTO button_priorities
        (game_type, coordinate_x, coordinate_y, button_type, confidence, last_used)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """
    )
    params = (game_type, x, y, button_type, confidence)
    try:
        logger.debug(f"[persist_button_priorities] DB: {db} | SQL: {sql.strip()} | Params: {params}")
        result = await db.execute(sql, params, table_name='button_priorities')
        logger.info(f"[persist_button_priorities] Success for game_type={game_type}, x={x}, y={y}, button_type={button_type} | Result: {result}")
        logger.debug(f"[persist_button_priorities] Commit attempted for game_type={game_type}")
        return result
    except Exception as e:
        import traceback
        logger.error(f"[persist_button_priorities] Exception: {e} | SQL: {sql.strip()} | Params: {params}\nTraceback: {traceback.format_exc()}")
        logger.debug(f"[persist_button_priorities] Rollback attempted for game_type={game_type}")
        return False


async def persist_governor_decision(session_id: str, decision_type: str, context: Dict[str, Any], confidence: float, outcome: Dict[str, Any]):
    print("[persist_governor_decision] CALLED", session_id, decision_type)
    logger.info(f"[persist_governor_decision] CALLED with session_id={session_id}, decision_type={decision_type}")
    db = get_database()
    sql = (
        """
        INSERT INTO governor_decisions
        (session_id, decision_type, context_data, governor_confidence, decision_outcome, decision_timestamp)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """
    )
    params = (session_id, decision_type, json.dumps(context), confidence, json.dumps(outcome))
    try:
        logger.debug(f"[persist_governor_decision] DB: {db} | SQL: {sql.strip()} | Params: {params}")
        result = await db.execute(sql, params, table_name='governor_decisions')
        logger.info(f"[persist_governor_decision] Success for session_id={session_id}, decision_type={decision_type} | Result: {result}")
        logger.debug(f"[persist_governor_decision] Commit attempted for session_id={session_id}")
        return result
    except Exception as e:
        import traceback
        logger.error(f"[persist_governor_decision] Exception: {e} | SQL: {sql.strip()} | Params: {params}\nTraceback: {traceback.format_exc()}")
        logger.debug(f"[persist_governor_decision] Rollback attempted for session_id={session_id}")
        return False
