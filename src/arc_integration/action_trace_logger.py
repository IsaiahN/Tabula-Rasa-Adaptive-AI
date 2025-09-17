"""Append-only action and session trace logger with rotation.

Writes NDJSON lines for individual actions to `data/action_traces.ndjson` and
writes per-session JSON files into `data/sessions/` for parsed external runs.

This is intentionally lightweight and tolerant of concurrent appends.
Includes automatic rotation to prevent files from exceeding GitHub's 100MB limit.
"""
from pathlib import Path
import json
import time
import threading
import os
import shutil
import logging
from typing import Optional


_lock = threading.Lock()

DATA_DIR = Path("data")
ACTION_TRACES = DATA_DIR / "action_traces.ndjson"
ACTION_TRACES_BACKUP = DATA_DIR / "action_traces_backup.ndjson"
SESSION_DIR = DATA_DIR / "sessions"
SESSION_INDEX = DATA_DIR / "session_traces.ndjson"

# Configuration for action traces rotation
MAX_ACTION_TRACES_LINES = 75000  # Approximately 10MB (based on ~135 bytes per line)
ACTION_TRACES_BACKUP_LINES = 37500  # Keep half in backup


def _ensure_paths():
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Best-effort; calling code should not fail if logging can't be written
        pass


def _should_rotate_action_traces() -> bool:
    """Check if action traces file should be rotated."""
    if not ACTION_TRACES.exists():
        return False
    
    try:
        with ACTION_TRACES.open('r', encoding='utf-8', errors='replace') as f:
            line_count = sum(1 for _ in f)
        return line_count > MAX_ACTION_TRACES_LINES
    except Exception:
        return False


def _rotate_action_traces() -> bool:
    """Rotate action traces file to prevent it from exceeding size limits."""
    try:
        if not ACTION_TRACES.exists():
            return True
        
        # Get current line count
        with ACTION_TRACES.open('r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        if total_lines <= MAX_ACTION_TRACES_LINES:
            return True
        
        # Create backup of current file
        if ACTION_TRACES_BACKUP.exists():
            ACTION_TRACES_BACKUP.unlink()
        shutil.copy2(ACTION_TRACES, ACTION_TRACES_BACKUP)
        
        # Keep only the most recent lines
        lines_to_keep = ACTION_TRACES_BACKUP_LINES
        if len(lines) > lines_to_keep:
            lines_to_write = lines[-lines_to_keep:]
        else:
            lines_to_write = lines
        
        # Write trimmed version
        temp_file = f"{ACTION_TRACES}.tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.writelines(lines_to_write)
        
        # Replace original with trimmed version
        try:
            os.replace(temp_file, str(ACTION_TRACES))
        except (OSError, PermissionError) as e:
            # Fallback: copy method
            try:
                shutil.copy2(temp_file, str(ACTION_TRACES))
                os.remove(temp_file)
            except Exception as copy_error:
                print(f"⚠️ Failed to rotate action traces: {copy_error}")
                try:
                    os.remove(temp_file)
                except:
                    pass
                return False
        
        print(f"✅ Action traces rotated: {total_lines} → {len(lines_to_write)} lines")
        return True
        
    except Exception as e:
        print(f"⚠️ Failed to rotate action traces: {e}")
        return False


def log_action_trace(record: dict) -> None:
    """Append a single action trace as a JSON line to `data/action_traces.ndjson`.

    The record should be JSON-serializable. This function never raises on write
    errors (it logs them to stderr) to avoid disrupting the training loop.
    Includes automatic rotation to prevent file size issues.
    """
    _ensure_paths()
    
    try:
        # Use database integration instead of file I/O
        from src.database.system_integration import get_system_integration
        import asyncio
        
        integration = get_system_integration()
        
        # Log action trace to database
        asyncio.run(integration.log_action_trace(
            record.get('game_id', 'unknown'),
            record.get('action_number', 0),
            record.get('coordinates', None),
            record.get('timestamp', time.time()),
            record.get('frame_before', None),
            record.get('frame_after', None),
            record.get('frame_changed', False),
            record.get('score_before', 0),
            record.get('score_after', 0),
            record.get('response_data', None)
        ))
        
    except Exception as e:
        # Fallback to file I/O if database fails
        try:
            # Check if rotation is needed before writing
            if _should_rotate_action_traces():
                _rotate_action_traces()
            
            line = json.dumps(record, default=str, ensure_ascii=False)
            with _lock:
                with ACTION_TRACES.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e2:
            try:
                print(f"⚠️ Failed to write action trace: {e2}")
        except Exception:
            pass


def write_session_trace(game_id: str, session_result: dict, raw_output: str = None) -> None:
    """Write a per-session JSON file under `data/sessions/` and append a short index line.

    `session_result` is expected to be a dict returned by the parser.
    """
    _ensure_paths()
    ts = int(time.time())
    safe_game = str(game_id).replace("/", "_")
    
    try:
        # Use database integration instead of file I/O
        from src.database.system_integration import get_system_integration
        import asyncio
        
        integration = get_system_integration()
        
        # Prepare session data
        session_data = {
            "game_id": game_id,
            "timestamp": ts,
            "session_result": session_result
        }
        if raw_output is not None:
            session_data["raw_output_snippet"] = (raw_output[:4000] + "...") if len(raw_output) > 4000 else raw_output

        # Save session to database
        session_id = f"session_{safe_game}_{ts}"
        asyncio.run(integration.update_session_metrics(session_id, session_data))
        
    except Exception as e:
        # Fallback to file I/O if database fails
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


def rotate_action_traces_manual() -> bool:
    """Manually rotate action traces file. Returns True if successful."""
    return _rotate_action_traces()


def get_action_traces_stats() -> dict:
    """Get statistics about the action traces file."""
    stats = {
        "file_exists": False,
        "lines": 0,
        "size_mb": 0.0,
        "needs_rotation": False,
        "path": str(ACTION_TRACES)
    }
    
    try:
        if ACTION_TRACES.exists():
            stats["file_exists"] = True
            stats["size_mb"] = round(ACTION_TRACES.stat().st_size / (1024 * 1024), 2)
            
            with ACTION_TRACES.open('r', encoding='utf-8', errors='replace') as f:
                stats["lines"] = sum(1 for _ in f)
            
            stats["needs_rotation"] = stats["lines"] > MAX_ACTION_TRACES_LINES
    except Exception as e:
        stats["error"] = str(e)
    
    return stats


def cleanup_old_action_traces(days_to_keep: int = 7) -> bool:
    """Clean up old action traces backup files."""
    try:
        if not ACTION_TRACES_BACKUP.exists():
            return True
        
        # Check if backup is older than specified days
        backup_age_days = (time.time() - ACTION_TRACES_BACKUP.stat().st_mtime) / (24 * 3600)
        
        if backup_age_days > days_to_keep:
            ACTION_TRACES_BACKUP.unlink()
            print(f"✅ Cleaned up old action traces backup ({backup_age_days:.1f} days old)")
            return True
        
        return True
    except Exception as e:
        print(f"⚠️ Failed to cleanup old action traces: {e}")
        return False
