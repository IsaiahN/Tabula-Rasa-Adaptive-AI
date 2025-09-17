#!/usr/bin/env python3
"""
Create a template database with schema but no data
"""
import sqlite3
import os
from pathlib import Path

def create_template_database():
    """Create tabula_rasa_template.db with schema but no data"""
    
    # Read the schema file
    schema_file = Path("src/database/schema.sql")
    if not schema_file.exists():
        print(f"❌ Schema file not found: {schema_file}")
        return False
    
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema_sql = f.read()
    
    # Create the template database
    template_db_path = "tabula_rasa_template.db"
    
    # Remove existing template if it exists
    if os.path.exists(template_db_path):
        os.remove(template_db_path)
        print(f"🗑️  Removed existing {template_db_path}")
    
    try:
        # Create new database with schema
        conn = sqlite3.connect(template_db_path)
        cursor = conn.cursor()
        
        # Execute the schema
        cursor.executescript(schema_sql)
        
        # Verify tables were created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = cursor.fetchall()
        
        print(f"✅ Created {template_db_path} with {len(tables)} tables:")
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            print(f"   📋 {table_name}: {count} rows")
        
        conn.commit()
        conn.close()
        
        print(f"\n🎉 Template database created successfully!")
        print(f"   📁 File: {template_db_path}")
        print(f"   📊 Tables: {len(tables)}")
        print(f"   💾 Size: {os.path.getsize(template_db_path)} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating template database: {e}")
        return False

if __name__ == "__main__":
    success = create_template_database()
    exit(0 if success else 1)
