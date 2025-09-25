#!/usr/bin/env python3
"""
Database Initialization Utility

Handles database initialization by copying template database if main database doesn't exist.
"""

import os
import shutil
import sys
from pathlib import Path

def initialize_database() -> bool:
    """
    Initialize the database by copying template if main database doesn't exist.
    
    Returns:
        bool: True if database is ready, False if initialization failed
    """
    # Ensure we're looking in the project root, not relative to current working directory
    current_dir = Path(__file__).parent
    project_root = current_dir
    while project_root.parent != project_root:
        if (project_root / "README.md").exists() or (project_root / "requirements.txt").exists():
            break
        project_root = project_root.parent
    
    main_db = project_root / "tabula_rasa.db"
    template_db = project_root / "tabula_rasa_template.db"
    
    # Check if main database exists
    if os.path.exists(main_db):
        print(f" Database found: {main_db}")
        return True
    
    # Check if template database exists
    if not os.path.exists(template_db):
        print(f" Error: Neither {main_db} nor {template_db} found!")
        print("   Please ensure tabula_rasa_template.db exists in the project root.")
        return False
    
    # Ask user if they want to copy template
    print(f" Main database not found: {main_db}")
    print(f" Template database found: {template_db}")
    print()
    
    while True:
        response = input(" Would you like to copy the template to create a new database instance? (y/n): ").strip().lower()
        
        if response in ['y', 'yes']:
            try:
                # Copy template to main database
                shutil.copy2(template_db, main_db)
                print(f" Successfully copied {template_db} to {main_db}")
                print(" Database initialized and ready for training!")
                return True
                
            except Exception as e:
                print(f" Error copying template database: {e}")
                return False
                
        elif response in ['n', 'no']:
            print(" Database initialization cancelled by user.")
            print("   Training cannot proceed without a database.")
            return False
            
        else:
            print("   Please enter 'y' for yes or 'n' for no.")

def check_database_ready() -> bool:
    """
    Check if database is ready for training.
    
    Returns:
        bool: True if database is ready, False otherwise
    """
    # Ensure we're looking in the project root, not relative to current working directory
    current_dir = Path(__file__).parent
    project_root = current_dir
    while project_root.parent != project_root:
        if (project_root / "README.md").exists() or (project_root / "requirements.txt").exists():
            break
        project_root = project_root.parent
    
    main_db = project_root / "tabula_rasa.db"
    
    if os.path.exists(main_db):
        # Check if database is accessible
        try:
            from .api import get_database
            db = get_database()
            # Try to connect to verify database is working
            return True
        except Exception as e:
            print(f" Database exists but is not accessible: {e}")
            return False
    
    return False

def ensure_database_ready() -> bool:
    """
    Ensure database is ready for training, initializing if necessary.
    
    Returns:
        bool: True if database is ready, False otherwise
    """
    # First check if database is already ready
    if check_database_ready():
        return True
    
    # If not ready, try to initialize
    return initialize_database()

if __name__ == "__main__":
    # Test the database initialization
    print(" Testing database initialization...")
    
    if ensure_database_ready():
        print(" Database is ready!")
    else:
        print(" Database initialization failed!")
        sys.exit(1)
