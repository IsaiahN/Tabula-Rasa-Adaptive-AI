"""Append-only action and session trace logger.

Writes NDJSON lines for individual actions to `data/action_traces.ndjson` and
writes per-session JSON files into `data/sessions/` for parsed external runs.

This is intentionally lightweight and tolerant of concurrent appends.
"""
from pathlib import Path
import json
import time
import threading


_lock = threading.Lock()

DATA_DIR = Path("data")
ACTION_TRACES = DATA_DIR / "action_traces.ndjson"
SESSION_DIR = DATA_DIR / "sessions"
SESSION_INDEX = DATA_DIR / "session_traces.ndjson"


def _ensure_paths():
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Best-effort; calling code should not fail if logging can't be written
        pass


def log_action_trace(record: dict) -> None:
    """Append a single action trace as a JSON line to `data/action_traces.ndjson`.

    The record should be JSON-serializable. This function never raises on write
    errors (it logs them to stderr) to avoid disrupting the training loop.
    """
    _ensure_paths()
    try:
        line = json.dumps(record, default=str, ensure_ascii=False)
        with _lock:
            with ACTION_TRACES.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        try:
            print(f"⚠️ Failed to write action trace: {e}")
        except Exception:
            pass


def write_session_trace(game_id: str, session_result: dict, raw_output: str = None) -> None:
    """Write a per-session JSON file under `data/sessions/` and append a short index line.

    `session_result` is expected to be a dict returned by the parser.
    """
    _ensure_paths()
    ts = int(time.time())
    safe_game = str(game_id).replace("/", "_")
    fname = SESSION_DIR / f"session_{safe_game}_{ts}.json"
    try:
        # Write full session record as pretty JSON
        out = {
            "game_id": game_id,
            "timestamp": ts,
            "session_result": session_result
        }
        if raw_output is not None:
            out["raw_output_snippet"] = (raw_output[:4000] + "...") if len(raw_output) > 4000 else raw_output

        with fname.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        # Also append an index line to session_traces.ndjson for quick scanning
        index_line = {
            "game_id": game_id,
            "timestamp": ts,
            "file": str(fname.name),
            "final_score": session_result.get("final_score") if isinstance(session_result, dict) else None,
            "total_actions": session_result.get("total_actions") if isinstance(session_result, dict) else None
        }
        with _lock:
            with SESSION_INDEX.open("a", encoding="utf-8") as f:
                f.write(json.dumps(index_line, default=str, ensure_ascii=False) + "\n")

    except Exception as e:
        try:
            print(f"⚠️ Failed to write session trace for {game_id}: {e}")
        except Exception:
            pass
