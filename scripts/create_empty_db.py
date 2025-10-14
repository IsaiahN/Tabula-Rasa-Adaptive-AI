import sqlite3
from pathlib import Path

def create_empty_database():
    # Get the paths
    repo_root = Path(__file__).parent.parent
    schema_path = repo_root / 'src' / 'database' / 'schema.sql'
    db_path = repo_root / 'tabula_rasa.db'
    
    # Read the schema
    with open(schema_path, 'r') as f:
        schema = f.read()
    
    # Create new database and apply schema
    conn = sqlite3.connect(db_path)
    conn.executescript(schema)
    conn.commit()
    conn.close()
    
    print(f"Created new empty database at {db_path}")

if __name__ == '__main__':
    create_empty_database()