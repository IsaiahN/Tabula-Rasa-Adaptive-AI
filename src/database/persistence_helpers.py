"""Convenience helpers to persist in-memory events to the database.

These helpers provide a small, safe API used by higher-level modules when
they append to in-memory lists so the same data is also saved to the DB.
"""

from typing import Any, Dict, List, Optional
import json
import logging
from datetime import datetime
from .api import get_database


# Console logging setup (keep console logging but remove file logging)
logger = logging.getLogger("tabula_rasa.persistence_helpers")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    # Console handler only
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(name)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


async def log_to_database(function_name: str, operation_type: str, message: str,
                         log_level: str = "INFO", parameters: str = None,
                         db_info: str = None, sql_query: str = None,
                         result_info: str = None, session_id: str = None,
                         game_id: str = None):
    """Log persistence helper activity to database instead of file."""
    try:
        db = get_database()
        await db.execute("""
            INSERT INTO persistence_debug_logs
            (timestamp, log_level, logger_name, function_name, operation_type,
             message, parameters, db_info, sql_query, result_info, session_id, game_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(),
            log_level,
            "tabula_rasa.persistence_helpers",
            function_name,
            operation_type,
            message,
            parameters,
            db_info,
            sql_query,
            result_info,
            session_id,
            game_id
        ), table_name='persistence_debug_logs')
    except Exception as e:
        # Fallback to console logging if database logging fails
        logger.error(f"Failed to log to database: {e}")
        logger.debug(f"[{function_name}] {operation_type}: {message}")


def make_json_serializable(obj: Any) -> Any:
    """Convert objects to JSON-serializable format.

    Handles GameState objects by converting them to dictionaries.
    """
    if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        # Object has a to_dict method (like GameState)
        return obj.to_dict()
    elif isinstance(obj, dict):
        # Recursively handle dictionary values
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively handle list/tuple items
        return [make_json_serializable(item) for item in obj]
    else:
        # Return as-is for basic types (str, int, float, bool, None)
        return obj


async def persist_winning_sequence(game_id: str, sequence: List[int], frequency: int = 1, avg_score: float = 0.0, success_rate: float = 1.0):
    print("[persist_winning_sequence] CALLED", game_id, sequence)

    # Log to database instead of file
    await log_to_database(
        function_name="persist_winning_sequence",
        operation_type="CALLED",
        message=f"CALLED with game_id={game_id}, sequence={sequence}",
        log_level="INFO",
        parameters=f"game_id={game_id}, sequence={sequence}, frequency={frequency}, avg_score={avg_score}, success_rate={success_rate}",
        game_id=game_id
    )

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
        # Log database operation
        await log_to_database(
            function_name="persist_winning_sequence",
            operation_type="DB_DEBUG",
            message=f"DB: {db} | SQL: {sql.strip()} | Params: {params}",
            log_level="DEBUG",
            db_info=str(db),
            sql_query=sql.strip(),
            parameters=str(params),
            game_id=game_id
        )

        result = await db.execute(sql, params, table_name='winning_sequences')

        # Log success
        await log_to_database(
            function_name="persist_winning_sequence",
            operation_type="Success",
            message=f"Success for game_id={game_id}, sequence={sequence} | Result: {result}",
            log_level="INFO",
            result_info=str(result),
            game_id=game_id
        )

        # Log commit
        await log_to_database(
            function_name="persist_winning_sequence",
            operation_type="Commit attempted",
            message=f"Commit attempted for game_id={game_id}",
            log_level="DEBUG",
            game_id=game_id
        )

        return result
    except Exception as e:
        import traceback
        # Log exception
        await log_to_database(
            function_name="persist_winning_sequence",
            operation_type="Exception",
            message=f"Exception: {e} | SQL: {sql.strip()} | Params: {params}",
            log_level="ERROR",
            parameters=str(params),
            sql_query=sql.strip(),
            result_info=f"Exception: {e}\nTraceback: {traceback.format_exc()}",
            game_id=game_id
        )

        # Log rollback
        await log_to_database(
            function_name="persist_winning_sequence",
            operation_type="Rollback attempted",
            message=f"Rollback attempted for game_id={game_id}",
            log_level="DEBUG",
            game_id=game_id
        )

        return False


async def persist_button_priorities(game_type: str, x: int, y: int, button_type: str, confidence: float = 0.0):
    print("[persist_button_priorities] CALLED", game_type, x, y, button_type)

    # Log to database instead of file
    await log_to_database(
        function_name="persist_button_priorities",
        operation_type="CALLED",
        message=f"CALLED with game_type={game_type}, x={x}, y={y}, button_type={button_type}",
        log_level="INFO",
        parameters=f"game_type={game_type}, x={x}, y={y}, button_type={button_type}, confidence={confidence}"
    )

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
        # Log database operation
        await log_to_database(
            function_name="persist_button_priorities",
            operation_type="DB_DEBUG",
            message=f"DB: {db} | SQL: {sql.strip()} | Params: {params}",
            log_level="DEBUG",
            db_info=str(db),
            sql_query=sql.strip(),
            parameters=str(params)
        )

        result = await db.execute(sql, params, table_name='button_priorities')

        # Log success
        await log_to_database(
            function_name="persist_button_priorities",
            operation_type="Success",
            message=f"Success for game_type={game_type}, x={x}, y={y}, button_type={button_type} | Result: {result}",
            log_level="INFO",
            result_info=str(result)
        )

        # Log commit
        await log_to_database(
            function_name="persist_button_priorities",
            operation_type="Commit attempted",
            message=f"Commit attempted for game_type={game_type}",
            log_level="DEBUG"
        )

        return result
    except Exception as e:
        import traceback
        # Log exception
        await log_to_database(
            function_name="persist_button_priorities",
            operation_type="Exception",
            message=f"Exception: {e} | SQL: {sql.strip()} | Params: {params}",
            log_level="ERROR",
            parameters=str(params),
            sql_query=sql.strip(),
            result_info=f"Exception: {e}\nTraceback: {traceback.format_exc()}"
        )

        # Log rollback
        await log_to_database(
            function_name="persist_button_priorities",
            operation_type="Rollback attempted",
            message=f"Rollback attempted for game_type={game_type}",
            log_level="DEBUG"
        )

        return False


async def persist_governor_decision(session_id: str, decision_type: str, context: Dict[str, Any], confidence: float, outcome: Dict[str, Any]):
    print("[persist_governor_decision] CALLED", session_id, decision_type)

    # Log to database instead of file
    await log_to_database(
        function_name="persist_governor_decision",
        operation_type="CALLED",
        message=f"CALLED with session_id={session_id}, decision_type={decision_type}",
        log_level="INFO",
        parameters=f"session_id={session_id}, decision_type={decision_type}, confidence={confidence}",
        session_id=session_id
    )

    db = get_database()
    sql = (
        """
        INSERT INTO governor_decisions
        (session_id, decision_type, context_data, governor_confidence, decision_outcome, decision_timestamp)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """
    )
    # Convert objects to JSON-serializable format before dumping
    serializable_context = make_json_serializable(context)
    serializable_outcome = make_json_serializable(outcome)
    params = (session_id, decision_type, json.dumps(serializable_context), confidence, json.dumps(serializable_outcome))
    try:
        # Log database operation
        await log_to_database(
            function_name="persist_governor_decision",
            operation_type="DB_DEBUG",
            message=f"DB: {db} | SQL: {sql.strip()} | Params: {params}",
            log_level="DEBUG",
            db_info=str(db),
            sql_query=sql.strip(),
            parameters=str(params),
            session_id=session_id
        )

        result = await db.execute(sql, params, table_name='governor_decisions')

        # Log success
        await log_to_database(
            function_name="persist_governor_decision",
            operation_type="Success",
            message=f"Success for session_id={session_id}, decision_type={decision_type} | Result: {result}",
            log_level="INFO",
            result_info=str(result),
            session_id=session_id
        )

        # Log commit
        await log_to_database(
            function_name="persist_governor_decision",
            operation_type="Commit attempted",
            message=f"Commit attempted for session_id={session_id}",
            log_level="DEBUG",
            session_id=session_id
        )

        return result
    except Exception as e:
        import traceback
        # Log exception
        await log_to_database(
            function_name="persist_governor_decision",
            operation_type="Exception",
            message=f"Exception: {e} | SQL: {sql.strip()} | Params: {params}",
            log_level="ERROR",
            parameters=str(params),
            sql_query=sql.strip(),
            result_info=f"Exception: {e}\nTraceback: {traceback.format_exc()}",
            session_id=session_id
        )

        # Log rollback
        await log_to_database(
            function_name="persist_governor_decision",
            operation_type="Rollback attempted",
            message=f"Rollback attempted for session_id={session_id}",
            log_level="DEBUG",
            session_id=session_id
        )

        return False
