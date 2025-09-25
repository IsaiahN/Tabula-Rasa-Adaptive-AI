import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / 'tabula_rasa.db'
print(f"Using DB: {DB}")
conn = sqlite3.connect(str(DB))
cur = conn.cursor()

tables = [
    'action_effectiveness',
    'coordinate_intelligence',
    'winning_sequences',
    'button_priorities',
    'strategy_refinements',
    'strategy_replications',
    'action_effectiveness_detailed'
]

for t in tables:
    print(f"\n--- {t} ---")
    try:
        cur.execute(f"SELECT * FROM {t} LIMIT 5")
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        print("| ".join(colnames))
        for row in rows:
            print("| ".join(str(x) for x in row))
        if not rows:
            print("(no rows)")
    except Exception as e:
        print(f"ERROR: {e}")

conn.close()
