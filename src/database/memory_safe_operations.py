"""
Memory-safe database operations to replace deprecated JSON file operations.

This module provides database operations that are memory-safe and replace
the deprecated JSON file operations in continuous_learning_loop.py.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class MemorySafeDatabaseOperations:
    """Memory-safe database operations to replace JSON file operations."""
    
    def __init__(self):
        self.director_commands = None
        self.system_integration = None
        self._initialize_database_connections()
    
    def _initialize_database_connections(self):
        """Initialize database connections."""
        try:
            from src.database.director_commands import get_director_commands
            from src.database.system_integration import get_system_integration
            self.director_commands = get_director_commands()
            self.system_integration = get_system_integration()
        except ImportError as e:
            logger.warning(f"Database connections not available: {e}")
    
    async def save_global_counters(self, counters: Dict[str, Any]) -> bool:
        """Save global counters to database instead of JSON file."""
        if not self.system_integration:
            logger.warning("System integration not available, skipping global counters save")
            return False
        
        try:
            # Use database to store global counters
            await self.system_integration.update_global_counters(counters)
            logger.info("Global counters saved to database")
            return True
        except Exception as e:
            logger.error(f"Failed to save global counters to database: {e}")
            return False
    
    async def load_global_counters(self) -> Dict[str, Any]:
        """Load global counters from database instead of JSON file."""
        if not self.system_integration:
            logger.warning("System integration not available, returning empty counters")
            return {}
        
        try:
            counters = await self.system_integration.get_global_counters()
            logger.info("Global counters loaded from database")
            return counters
        except Exception as e:
            logger.error(f"Failed to load global counters from database: {e}")
            return {}
    
    async def save_action_intelligence(self, game_id: str, intelligence_data: Dict[str, Any]) -> bool:
        """Save action intelligence data to database instead of JSON file."""
        if not self.system_integration:
            logger.warning("System integration not available, skipping action intelligence save")
            return False
        
        try:
            # Store action intelligence data in database
            await self.system_integration.store_action_intelligence(game_id, intelligence_data)
            logger.info(f"Action intelligence for {game_id} saved to database")
            return True
        except Exception as e:
            logger.error(f"Failed to save action intelligence to database: {e}")
            return False
    
    async def load_action_intelligence(self, game_id: str) -> Dict[str, Any]:
        """Load action intelligence data from database instead of JSON file."""
        if not self.system_integration:
            logger.warning("System integration not available, returning empty intelligence data")
            return {}
        
        try:
            intelligence_data = await self.system_integration.get_action_intelligence(game_id)
            logger.info(f"Action intelligence for {game_id} loaded from database")
            return intelligence_data
        except Exception as e:
            logger.error(f"Failed to load action intelligence from database: {e}")
            return {}
    
    async def save_performance_data(self, performance_data: Dict[str, Any]) -> bool:
        """Save performance data to database instead of JSON file."""
        if not self.system_integration:
            logger.warning("System integration not available, skipping performance data save")
            return False
        
        try:
            # Store performance data in database
            await self.system_integration.store_performance_data(performance_data)
            logger.info("Performance data saved to database")
            return True
        except Exception as e:
            logger.error(f"Failed to save performance data to database: {e}")
            return False
    
    async def load_performance_data(self, session_id: str) -> Dict[str, Any]:
        """Load performance data from database instead of JSON file."""
        if not self.system_integration:
            logger.warning("System integration not available, returning empty performance data")
            return {}
        
        try:
            performance_data = await self.system_integration.get_performance_data(session_id)
            logger.info(f"Performance data for {session_id} loaded from database")
            return performance_data
        except Exception as e:
            logger.error(f"Failed to load performance data from database: {e}")
            return {}
    
    async def save_architect_evolution_data(self, evolution_data: Dict[str, Any]) -> bool:
        """Save architect evolution data to database instead of JSON file."""
        if not self.system_integration:
            logger.warning("System integration not available, skipping architect evolution data save")
            return False
        
        try:
            # Store architect evolution data in database
            await self.system_integration.store_architect_evolution_data(evolution_data)
            logger.info("Architect evolution data saved to database")
            return True
        except Exception as e:
            logger.error(f"Failed to save architect evolution data to database: {e}")
            return False
    
    async def load_architect_evolution_data(self) -> Dict[str, Any]:
        """Load architect evolution data from database instead of JSON file."""
        if not self.system_integration:
            logger.warning("System integration not available, returning empty architect evolution data")
            return {}
        
        try:
            evolution_data = await self.system_integration.get_architect_evolution_data()
            logger.info("Architect evolution data loaded from database")
            return evolution_data
        except Exception as e:
            logger.error(f"Failed to load architect evolution data from database: {e}")
            return {}
    
    async def save_learning_state(self, state_data: Dict[str, Any]) -> bool:
        """Save learning state to database instead of JSON file."""
        if not self.system_integration:
            logger.warning("System integration not available, skipping learning state save")
            return False
        
        try:
            # Store learning state in database
            await self.system_integration.store_learning_state(state_data)
            logger.info("Learning state saved to database")
            return True
        except Exception as e:
            logger.error(f"Failed to save learning state to database: {e}")
            return False
    
    async def load_learning_state(self) -> Dict[str, Any]:
        """Load learning state from database instead of JSON file."""
        if not self.system_integration:
            logger.warning("System integration not available, returning empty learning state")
            return {}
        
        try:
            state_data = await self.system_integration.get_learning_state()
            logger.info("Learning state loaded from database")
            return state_data
        except Exception as e:
            logger.error(f"Failed to load learning state from database: {e}")
            return {}
    
    async def cleanup_old_data(self, max_age_days: int = 30) -> bool:
        """Clean up old data from database."""
        if not self.system_integration:
            logger.warning("System integration not available, skipping cleanup")
            return False
        
        try:
            # Clean up old data
            await self.system_integration.cleanup_old_data(max_age_days)
            logger.info(f"Cleaned up data older than {max_age_days} days")
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return False


class FallbackJSONOperations:
    """Fallback JSON operations when database is not available."""
    
    def __init__(self, save_directory: Path):
        self.save_directory = save_directory
        self.save_directory.mkdir(parents=True, exist_ok=True)
    
    def save_global_counters(self, counters: Dict[str, Any]) -> bool:
        """Fallback: Save global counters to JSON file."""
        try:
            counter_file = self.save_directory / "global_counters.json"
            with open(counter_file, 'w', encoding='utf-8') as f:
                json.dump(counters, f, indent=2, ensure_ascii=False)
            logger.info("Global counters saved to JSON file (fallback)")
            return True
        except Exception as e:
            logger.error(f"Failed to save global counters to JSON file: {e}")
            return False
    
    def load_global_counters(self) -> Dict[str, Any]:
        """Fallback: Load global counters from JSON file."""
        try:
            counter_file = self.save_directory / "global_counters.json"
            if counter_file.exists():
                with open(counter_file, 'r', encoding='utf-8') as f:
                    counters = json.load(f)
                logger.info("Global counters loaded from JSON file (fallback)")
                return counters
            return {}
        except Exception as e:
            logger.error(f"Failed to load global counters from JSON file: {e}")
            return {}
    
    def save_action_intelligence(self, game_id: str, intelligence_data: Dict[str, Any]) -> bool:
        """Fallback: Save action intelligence data to JSON file."""
        try:
            intelligence_file = self.save_directory / f"action_intelligence_{game_id}.json"
            with open(intelligence_file, 'w', encoding='utf-8') as f:
                json.dump(intelligence_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Action intelligence for {game_id} saved to JSON file (fallback)")
            return True
        except Exception as e:
            logger.error(f"Failed to save action intelligence to JSON file: {e}")
            return False
    
    def load_action_intelligence(self, game_id: str) -> Dict[str, Any]:
        """Fallback: Load action intelligence data from JSON file."""
        try:
            intelligence_file = self.save_directory / f"action_intelligence_{game_id}.json"
            if intelligence_file.exists():
                with open(intelligence_file, 'r', encoding='utf-8') as f:
                    intelligence_data = json.load(f)
                logger.info(f"Action intelligence for {game_id} loaded from JSON file (fallback)")
                return intelligence_data
            return {}
        except Exception as e:
            logger.error(f"Failed to load action intelligence from JSON file: {e}")
            return {}


class HybridDataManager:
    """Hybrid data manager that uses database when available, falls back to JSON files."""
    
    def __init__(self, save_directory: Path):
        self.database_ops = MemorySafeDatabaseOperations()
        self.json_ops = FallbackJSONOperations(save_directory)
        self.use_database = self.database_ops.director_commands is not None
    
    async def save_global_counters(self, counters: Dict[str, Any]) -> bool:
        """Save global counters using database or JSON fallback."""
        if self.use_database:
            return await self.database_ops.save_global_counters(counters)
        else:
            return self.json_ops.save_global_counters(counters)
    
    async def load_global_counters(self) -> Dict[str, Any]:
        """Load global counters using database or JSON fallback."""
        if self.use_database:
            return await self.database_ops.load_global_counters()
        else:
            return self.json_ops.load_global_counters()
    
    async def save_action_intelligence(self, game_id: str, intelligence_data: Dict[str, Any]) -> bool:
        """Save action intelligence using database or JSON fallback."""
        if self.use_database:
            return await self.database_ops.save_action_intelligence(game_id, intelligence_data)
        else:
            return self.json_ops.save_action_intelligence(game_id, intelligence_data)
    
    async def load_action_intelligence(self, game_id: str) -> Dict[str, Any]:
        """Load action intelligence using database or JSON fallback."""
        if self.use_database:
            return await self.database_ops.load_action_intelligence(game_id)
        else:
            return self.json_ops.load_action_intelligence(game_id)
    
    async def save_performance_data(self, performance_data: Dict[str, Any]) -> bool:
        """Save performance data using database or JSON fallback."""
        if self.use_database:
            return await self.database_ops.save_performance_data(performance_data)
        else:
            # Fallback to JSON file
            try:
                performance_file = self.json_ops.save_directory / f"performance_{int(time.time())}.json"
                with open(performance_file, 'w', encoding='utf-8') as f:
                    json.dump(performance_data, f, indent=2, ensure_ascii=False)
                return True
            except Exception as e:
                logger.error(f"Failed to save performance data to JSON file: {e}")
                return False
    
    async def cleanup_old_data(self, max_age_days: int = 30) -> bool:
        """Clean up old data."""
        if self.use_database:
            return await self.database_ops.cleanup_old_data(max_age_days)
        else:
            # Fallback: Clean up old JSON files
            try:
                current_time = time.time()
                max_age_seconds = max_age_days * 24 * 3600
                
                for file_path in self.json_ops.save_directory.glob("*.json"):
                    if current_time - file_path.stat().st_mtime > max_age_seconds:
                        file_path.unlink()
                        logger.info(f"Deleted old file: {file_path}")
                
                return True
            except Exception as e:
                logger.error(f"Failed to cleanup old JSON files: {e}")
                return False


# Global instance for easy access
def get_data_manager(save_directory: Path) -> HybridDataManager:
    """Get a data manager instance."""
    return HybridDataManager(save_directory)
