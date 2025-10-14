"""Central configuration for Tabula-Rasa minimal run.

This is intentionally small: reads from environment with defaults.
"""
import os
from pathlib import Path

DB_PATH = os.environ.get('TABULA_RASA_DB', str(Path.cwd() / 'tabula_rasa.db'))
USE_STUB_API_FOR_CI = os.environ.get('USE_STUB_API_FOR_CI', '0') in ('1', 'true', 'True')
MINIMAL_MODE = os.environ.get('MINIMAL_MODE', '1') in ('1', 'true', 'True')
API_HOST = os.environ.get('ARC3_API_HOST', 'https://api.example.com')
