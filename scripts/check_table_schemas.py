import sqlite3
import json
from pathlib import Path

def get_table_schema(cursor, table_name):
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    result = cursor.fetchone()
    if result:
        return result[0]
    return None

def main():
    db_path = Path(__file__).parent.parent / "tabula_rasa.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print(f"\nDatabase file: {db_path}\n")
    print("Table Schemas:\n")
    
    for (table_name,) in tables:
        schema = get_table_schema(cursor, table_name)
        print(f"=== {table_name} ===")
        print(schema)
        print()

    conn.close()

if __name__ == "__main__":
    main()