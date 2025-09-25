import sqlite3, os
root = r"C:\Users\Admin\Documents\GitHub\tabula-rasa"
db = os.path.join(root, 'tabula_rasa.db')
print('DB path:', db, 'exists:', os.path.exists(db))
conn = sqlite3.connect(db)
cur = conn.cursor()
tables = ['winning_sequences','button_priorities','strategy_refinements','strategy_replications','action_effectiveness_detailed','db_error_debug']
for t in tables:
    try:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        print(f"{t}: {cur.fetchone()[0]}")
    except Exception as e:
        print(f"{t}: ERROR - {e}")
conn.close()
