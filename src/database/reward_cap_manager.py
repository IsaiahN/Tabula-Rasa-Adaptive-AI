"""
Reward Cap Configuration Manager
Replaces reward_cap_config.json with database storage.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import sqlite3
from pathlib import Path

class RewardCapManager:
    """
    Manages reward cap configuration in the database.
    Replaces the reward_cap_config.json file.
    """
    
    def __init__(self, db_path: str = "tabula_rasa.db"):
        self.db_path = db_path
        self._ensure_tables_exist()
    
    def _ensure_tables_exist(self):
        """Ensure the reward_cap_config table exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reward_cap_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_key TEXT NOT NULL UNIQUE,
                    config_value TEXT NOT NULL,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def set_config(self, key: str, value: Any) -> bool:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value (will be JSON serialized)
            
        Returns:
            True if successful
        """
        try:
            value_json = json.dumps(value)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO reward_cap_config 
                    (config_key, config_value, last_updated)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (key, value_json))
                conn.commit()
            
            return True
        except Exception as e:
            print(f"Error setting config {key}: {e}")
            return False
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT config_value FROM reward_cap_config 
                    WHERE config_key = ?
                """, (key,))
                
                result = cursor.fetchone()
                if result:
                    return json.loads(result[0])
                else:
                    return default
        except Exception as e:
            print(f"Error getting config {key}: {e}")
            return default
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration values."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT config_key, config_value FROM reward_cap_config
                """)
                
                config = {}
                for row in cursor.fetchall():
                    config[row[0]] = json.loads(row[1])
                
                return config
        except Exception as e:
            print(f"Error getting all config: {e}")
            return {}
    
    def update_current_caps(self, caps: Dict[str, float]) -> bool:
        """Update the current caps configuration."""
        return self.set_config("current_caps", caps)
    
    def get_current_caps(self) -> Dict[str, float]:
        """Get the current caps configuration."""
        return self.get_config("current_caps", {})
    
    def update_cap_history(self, history_entry: Dict[str, Any]) -> bool:
        """Add an entry to the cap history."""
        current_history = self.get_config("cap_history", [])
        current_history.append(history_entry)
        return self.set_config("cap_history", current_history)
    
    def get_cap_history(self) -> List[Dict[str, Any]]:
        """Get the cap history."""
        return self.get_config("cap_history", [])
    
    def set_learning_phase(self, phase: str) -> bool:
        """Set the learning phase."""
        return self.set_config("learning_phase", phase)
    
    def get_learning_phase(self) -> str:
        """Get the learning phase."""
        return self.get_config("learning_phase", "balanced")
    
    def set_last_adjustment(self, adjustment: float) -> bool:
        """Set the last adjustment value."""
        return self.set_config("last_adjustment", adjustment)
    
    def get_last_adjustment(self) -> float:
        """Get the last adjustment value."""
        return self.get_config("last_adjustment", 0.0)
    
    def set_action_count(self, count: int) -> bool:
        """Set the action count."""
        return self.set_config("action_count", count)
    
    def get_action_count(self) -> int:
        """Get the action count."""
        return self.get_config("action_count", 0)
    
    def migrate_from_file(self, file_path: str = "data/reward_cap_config.json") -> bool:
        """
        Migrate configuration from the old JSON file to database.
        
        Args:
            file_path: Path to the old JSON file
            
        Returns:
            True if migration successful
        """
        try:
            if not Path(file_path).exists():
                print(f"File {file_path} does not exist, skipping migration")
                return True
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Migrate all configuration
            for key, value in data.items():
                self.set_config(key, value)
            
            print(f"Successfully migrated reward cap config from {file_path}")
            return True
            
        except Exception as e:
            print(f"Error migrating reward cap config: {e}")
            return False
    
    def export_to_file(self, file_path: str = "data/reward_cap_config.json") -> bool:
        """
        Export current configuration to JSON file (for backup/debugging).
        
        Args:
            file_path: Path to export to
            
        Returns:
            True if export successful
        """
        try:
            config = self.get_all_config()
            
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Successfully exported reward cap config to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting reward cap config: {e}")
            return False

# Global instance
_reward_cap_manager = None

def get_reward_cap_manager() -> RewardCapManager:
    """Get the global reward cap manager instance."""
    global _reward_cap_manager
    if _reward_cap_manager is None:
        _reward_cap_manager = RewardCapManager()
    return _reward_cap_manager
