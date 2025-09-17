"""
Database-backed bounded list that automatically stores data in the database.

This class provides the same interface as BoundedList but automatically
stores data in the database to prevent memory leaks and provide persistence.
"""

import asyncio
import json
import logging
from typing import Any, List, Dict, Optional
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


class DatabaseBoundedList:
    """A list that automatically stores data in the database and bounds memory usage."""
    
    def __init__(self, max_size: int = 100, table_name: str = "performance_history", 
                 session_id: Optional[str] = None, game_id: Optional[str] = None):
        self.max_size = max_size
        self.table_name = table_name
        self.session_id = session_id
        self.game_id = game_id
        self._data = deque(maxlen=max_size)
        self._performance_manager = None
        self._initialize_performance_manager()
    
    def _initialize_performance_manager(self):
        """Initialize the performance manager."""
        try:
            from src.database.performance_data_manager import get_performance_manager
            self._performance_manager = get_performance_manager()
        except ImportError as e:
            logger.warning(f"Performance manager not available: {e}")
            self._performance_manager = None
    
    def append(self, item: Any) -> None:
        """Append item, automatically storing in database and bounding memory."""
        self._data.append(item)
        
        # Store in database if available (deferred to avoid event loop issues)
        if self._performance_manager:
            # Store synchronously to avoid event loop issues
            self._store_item_sync(item)
    
    def _store_item_sync(self, item: Any) -> None:
        """Store item in database synchronously."""
        try:
            if self.table_name == "performance_history":
                # Use synchronous database operations
                import sqlite3
                import json
                
                conn = sqlite3.connect(self._performance_manager.db_path)
                conn.execute("""
                    INSERT INTO performance_history 
                    (session_id, game_id, score, win_rate, learning_efficiency, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    self.session_id or "unknown",
                    item.get('game_id') if isinstance(item, dict) else None,
                    item.get('score') if isinstance(item, dict) else None,
                    item.get('win_rate') if isinstance(item, dict) else None,
                    item.get('learning_efficiency') if isinstance(item, dict) else None,
                    json.dumps(item if isinstance(item, dict) else {"data": item})
                ))
                conn.commit()
                conn.close()
                
            elif self.table_name == "session_history":
                import sqlite3
                import json
                
                conn = sqlite3.connect(self._performance_manager.db_path)
                conn.execute("""
                    INSERT INTO session_history 
                    (session_id, game_id, status, duration_seconds, actions_taken, score, win, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.session_id or "unknown",
                    item.get('game_id') if isinstance(item, dict) else None,
                    item.get('status') if isinstance(item, dict) else None,
                    item.get('duration_seconds') if isinstance(item, dict) else None,
                    item.get('actions_taken') if isinstance(item, dict) else None,
                    item.get('score') if isinstance(item, dict) else None,
                    item.get('win') if isinstance(item, dict) else None,
                    json.dumps(item if isinstance(item, dict) else {"data": item})
                ))
                conn.commit()
                conn.close()
                
        except Exception as e:
            logger.error(f"Failed to store item synchronously: {e}")
    
    async def _store_item_async(self, item: Any) -> None:
        """Store item in database asynchronously."""
        try:
            if self.table_name == "performance_history":
                await self._performance_manager.store_performance_data(
                    self.session_id or "unknown", 
                    item if isinstance(item, dict) else {"data": item}
                )
            elif self.table_name == "session_history":
                await self._performance_manager.store_session_data(
                    self.session_id or "unknown",
                    item if isinstance(item, dict) else {"data": item}
                )
            elif self.table_name == "action_tracking":
                await self._performance_manager.store_action_tracking(
                    self.game_id or "unknown",
                    item if isinstance(item, dict) else {"data": item}
                )
            elif self.table_name == "score_history":
                await self._performance_manager.store_score_data(
                    self.game_id or "unknown",
                    item if isinstance(item, dict) else {"score": item}
                )
        except Exception as e:
            logger.error(f"Failed to store item in database: {e}")
    
    def extend(self, items: List[Any]) -> None:
        """Extend with items, automatically storing in database."""
        for item in items:
            self.append(item)
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, index):
        return self._data[index]
    
    def __setitem__(self, index, value):
        self._data[index] = value
    
    def __iter__(self):
        return iter(self._data)
    
    def __contains__(self, item):
        return item in self._data
    
    def clear(self):
        """Clear all data."""
        self._data.clear()
    
    def to_list(self) -> List[Any]:
        """Convert to regular list."""
        return list(self._data)
    
    def get_recent(self, count: int) -> List[Any]:
        """Get the most recent items."""
        return list(self._data)[-count:]
    
    async def load_from_database(self, limit: int = None) -> None:
        """Load data from database into memory."""
        if not self._performance_manager:
            return
        
        try:
            if self.table_name == "performance_history":
                data = await self._performance_manager.get_performance_history(
                    self.session_id, limit or self.max_size
                )
            elif self.table_name == "session_history":
                data = await self._performance_manager.get_session_history(
                    self.session_id, limit or self.max_size
                )
            elif self.table_name == "action_tracking":
                data = await self._performance_manager.get_action_tracking(
                    self.game_id or "unknown", limit or self.max_size
                )
            elif self.table_name == "score_history":
                data = await self._performance_manager.get_score_history(
                    self.game_id or "unknown", limit or self.max_size
                )
            else:
                return
            
            # Clear current data and load from database
            self._data.clear()
            for item in data:
                self._data.append(item)
                
        except Exception as e:
            logger.error(f"Failed to load data from database: {e}")
    
    async def sync_to_database(self) -> bool:
        """Sync all current data to database."""
        if not self._performance_manager:
            return False
        
        try:
            for item in self._data:
                await self._store_item_async(item)
            return True
        except Exception as e:
            logger.error(f"Failed to sync data to database: {e}")
            return False


class DatabaseBoundedDict:
    """A dictionary that automatically stores data in the database and bounds memory usage."""
    
    def __init__(self, max_size: int = 100, table_name: str = "coordinate_tracking",
                 game_id: Optional[str] = None):
        self.max_size = max_size
        self.table_name = table_name
        self.game_id = game_id
        self._data = {}
        self._access_order = deque(maxlen=max_size)
        self._performance_manager = None
        self._initialize_performance_manager()
    
    def _initialize_performance_manager(self):
        """Initialize the performance manager."""
        try:
            from src.database.performance_data_manager import get_performance_manager
            self._performance_manager = get_performance_manager()
        except ImportError as e:
            logger.warning(f"Performance manager not available: {e}")
            self._performance_manager = None
    
    def __setitem__(self, key, value):
        if key not in self._data and len(self._data) >= self.max_size:
            # Remove oldest item
            oldest_key = self._access_order.popleft()
            if oldest_key in self._data:
                del self._data[oldest_key]
        
        self._data[key] = value
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        # Store in database if available
        if self._performance_manager:
            asyncio.create_task(self._store_item_async(key, value))
    
    async def _store_item_async(self, key: str, value: Any) -> None:
        """Store item in database asynchronously."""
        try:
            if self.table_name == "coordinate_tracking":
                # Parse coordinate from key if possible
                if isinstance(key, str) and ',' in key:
                    try:
                        x, y = key.split(',')
                        coordinate_data = {
                            'coordinate_x': int(x),
                            'coordinate_y': int(y),
                            'context': value if isinstance(value, dict) else {"data": value}
                        }
                    except ValueError:
                        coordinate_data = {
                            'coordinate_x': 0,
                            'coordinate_y': 0,
                            'context': {"key": key, "value": value}
                        }
                else:
                    coordinate_data = {
                        'coordinate_x': 0,
                        'coordinate_y': 0,
                        'context': {"key": key, "value": value}
                    }
                
                await self._performance_manager.store_coordinate_tracking(
                    self.game_id or "unknown",
                    coordinate_data
                )
            elif self.table_name == "frame_tracking":
                frame_data = {
                    'frame_hash': key,
                    'frame_analysis': value if isinstance(value, dict) else {"data": value}
                }
                await self._performance_manager.store_frame_tracking(
                    self.game_id or "unknown",
                    frame_data
                )
        except Exception as e:
            logger.error(f"Failed to store item in database: {e}")
    
    def __getitem__(self, key):
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        return self._data[key]
    
    def __contains__(self, key):
        return key in self._data
    
    def __len__(self):
        return len(self._data)
    
    def __iter__(self):
        return iter(self._data)
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        return self._data.values()
    
    def items(self):
        return self._data.items()
    
    def get(self, key, default=None):
        if key in self._data:
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._data[key]
        return default
    
    def clear(self):
        """Clear all data."""
        self._data.clear()
        self._access_order.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to regular dictionary."""
        return dict(self._data)
    
    async def load_from_database(self, limit: int = None) -> None:
        """Load data from database into memory."""
        if not self._performance_manager:
            return
        
        try:
            if self.table_name == "coordinate_tracking":
                data = await self._performance_manager.get_coordinate_tracking(
                    self.game_id or "unknown", limit or self.max_size
                )
            elif self.table_name == "frame_tracking":
                data = await self._performance_manager.get_frame_tracking(
                    self.game_id or "unknown", limit or self.max_size
                )
            else:
                return
            
            # Clear current data and load from database
            self._data.clear()
            self._access_order.clear()
            
            for item in data:
                if self.table_name == "coordinate_tracking":
                    key = f"{item['coordinate_x']},{item['coordinate_y']}"
                    value = item.get('context', {})
                elif self.table_name == "frame_tracking":
                    key = item['frame_hash']
                    value = item.get('frame_analysis', {})
                else:
                    continue
                
                self._data[key] = value
                self._access_order.append(key)
                
        except Exception as e:
            logger.error(f"Failed to load data from database: {e}")
    
    async def sync_to_database(self) -> bool:
        """Sync all current data to database."""
        if not self._performance_manager:
            return False
        
        try:
            for key, value in self._data.items():
                await self._store_item_async(key, value)
            return True
        except Exception as e:
            logger.error(f"Failed to sync data to database: {e}")
            return False
