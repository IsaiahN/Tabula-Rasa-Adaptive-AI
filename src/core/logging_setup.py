import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_path: str = None, level=logging.INFO, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5):
    """Database-only logging setup - console only.

    This helper sets up console-only logging for database-only mode.
    All logging data is stored in the database instead of files.
    """

    logger = logging.getLogger()
    logger.setLevel(level)

    # Console handler (no encoding change) - let console handle replacement
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    console_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(console_fmt)

    # Database-only mode: Skip file-based logging
    # log_file = Path(log_path)
    # if not log_file.parent.exists():
    #     log_file.parent.mkdir(parents=True, exist_ok=True)

    # # File handler with UTF-8 encoding and rotation
    # fh = RotatingFileHandler(str(log_file), maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8')
    # fh.setLevel(level)
    # file_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    # fh.setFormatter(file_fmt)

    # Replace existing handlers with ours for deterministic behavior
    for h in list(logger.handlers):
        logger.removeHandler(h)

    logger.addHandler(ch)
    # logger.addHandler(fh)  # Database-only mode: No file handler

    # Try to reconfigure stdout/stderr to UTF-8 where supported (Windows/modern Python)
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        # Non-fatal: continue without breaking execution
        pass

    return logger
