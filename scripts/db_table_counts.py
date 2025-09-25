import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / 'tabula_rasa.db'
print(f"Using DB: {DB}")
conn = sqlite3.connect(str(DB))
cur = conn.cursor()

tables = [
    'training_sessions',
    'game_results',
    'action_effectiveness',
    'coordinate_intelligence',
    'winning_sequences',
    'button_priorities',
    'strategy_refinements',
    'strategy_replications',
    'action_effectiveness_detailed',
    'db_error_debug'
]

for t in tables:
    try:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        n = cur.fetchone()[0]
        print(f"{t}: {n}")
    except Exception as e:
        print(f"{t}: ERROR ({e})")

conn.close()
