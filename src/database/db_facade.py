"""Database facade for TB-Seed baseline persistence.

Provides a simple SQLite-backed API to create and write to baseline tables:
- sessions
- games
- actions
- strategies
- metrics

This is intentionally minimal and synchronous; callers may call it from async code via a thread executor if needed.
"""
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
import json

DEFAULT_DB_PATH = Path.cwd() / "tabula_rasa.db"

CREATE_TABLES_SQL = [
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        start_time TEXT,
        end_time TEXT,
        status TEXT,
        metadata TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS games (
        game_id TEXT PRIMARY KEY,
        guid TEXT,
        scorecard_id TEXT,
        start_time TEXT,
        end_time TEXT,
        final_score REAL,
        final_state TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        game_id TEXT,
        action_payload TEXT,
        result_payload TEXT,
        timestamp TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS strategies (
        strategy_id TEXT PRIMARY KEY,
        game_id TEXT,
        action_sequence TEXT,
        efficiency REAL,
        metadata TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        metric_name TEXT,
        metric_value TEXT,
        timestamp TEXT,
        context TEXT
    )
    """,
]

class DBFacade:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._ensure_db()

    def _get_conn(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            for stmt in CREATE_TABLES_SQL:
                cur.execute(stmt)
            conn.commit()
        finally:
            conn.close()

    def upsert_session(self, session_id: str, start_time: str, status: str = 'running', metadata: Optional[Dict] = None):
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO sessions(session_id, start_time, end_time, status, metadata) VALUES (?, ?, ?, ?, ?)",
                (session_id, start_time, None, status, json.dumps(metadata or {}))
            )
            conn.commit()
        finally:
            conn.close()

    def end_session(self, session_id: str, end_time: str, status: str = 'completed'):
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE sessions SET end_time = ?, status = ? WHERE session_id = ?",
                (end_time, status, session_id)
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_game(self, game_id: str, guid: Optional[str], scorecard_id: Optional[str], start_time: str):
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO games(game_id, guid, scorecard_id, start_time, end_time, final_score, final_state) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (game_id, guid, scorecard_id, start_time, None, None, None)
            )
            conn.commit()
        finally:
            conn.close()

    def end_game(self, game_id: str, end_time: str, final_score: float, final_state: str):
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE games SET end_time = ?, final_score = ?, final_state = ? WHERE game_id = ?",
                (end_time, final_score, final_state, game_id)
            )
            conn.commit()
        finally:
            conn.close()

    def add_action(self, session_id: Optional[str], game_id: str, action_payload: Dict[str, Any], result_payload: Optional[Dict[str, Any]], timestamp: str):
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO actions(session_id, game_id, action_payload, result_payload, timestamp) VALUES (?, ?, ?, ?, ?)",
                (session_id, game_id, json.dumps(action_payload), json.dumps(result_payload or {}), timestamp)
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def add_strategy(self, strategy_id: str, game_id: str, action_sequence: list, efficiency: float = 0.0, metadata: Optional[Dict] = None):
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO strategies(strategy_id, game_id, action_sequence, efficiency, metadata) VALUES (?, ?, ?, ?, ?)",
                (strategy_id, game_id, json.dumps(action_sequence), efficiency, json.dumps(metadata or {}))
            )
            conn.commit()
        finally:
            conn.close()

    def add_metric(self, metric_name: str, metric_value: Any, timestamp: str, context: Optional[Dict] = None):
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO metrics(metric_name, metric_value, timestamp, context) VALUES (?, ?, ?, ?)",
                (metric_name, json.dumps(metric_value), timestamp, json.dumps(context or {}))
            )
            conn.commit()
        finally:
            conn.close()
