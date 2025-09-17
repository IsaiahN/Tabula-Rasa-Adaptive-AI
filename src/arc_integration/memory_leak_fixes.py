"""
Memory leak fixes for continuous_learning_loop.py

This module provides utilities to fix memory leaks by bounding data structures
and implementing proper cleanup mechanisms.
"""

import gc
import time
from typing import List, Dict, Any, Optional
from collections import deque
from datetime import datetime, timedelta


class MemoryLeakFixer:
    """Utility class to fix memory leaks in data structures."""
    
    def __init__(self, max_performance_history: int = 100, max_session_history: int = 100):
        self.max_performance_history = max_performance_history
        self.max_session_history = max_session_history
        self.cleanup_interval = 100  # Cleanup every 100 operations
        self.operation_count = 0
    
    def bound_list(self, data_list: List[Any], max_size: int) -> List[Any]:
        """Bound a list to maximum size, keeping most recent items."""
        if len(data_list) > max_size:
            return data_list[-max_size:]
        return data_list
    
    def bound_dict(self, data_dict: Dict[str, Any], max_size: int) -> Dict[str, Any]:
        """Bound a dictionary to maximum size, keeping most recent items."""
        if len(data_dict) > max_size:
            # Convert to list of items, sort by key (assuming keys are timestamps or similar)
            items = list(data_dict.items())
            items.sort(key=lambda x: str(x[0]), reverse=True)
            return dict(items[:max_size])
        return data_dict
    
    def cleanup_old_data(self, data_dict: Dict[str, Any], max_age_hours: int = 24) -> Dict[str, Any]:
        """Clean up old data based on timestamp."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        cleaned = {}
        for key, value in data_dict.items():
            # Check if value has timestamp
            if isinstance(value, dict) and 'timestamp' in value:
                timestamp = value.get('timestamp', 0)
                if isinstance(timestamp, str):
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        timestamp = dt.timestamp()
                    except:
                        continue
                
                if current_time - timestamp < max_age_seconds:
                    cleaned[key] = value
            else:
                cleaned[key] = value
        
        return cleaned
    
    def periodic_cleanup(self, obj: Any) -> None:
        """Perform periodic cleanup on an object."""
        self.operation_count += 1
        
        if self.operation_count % self.cleanup_interval == 0:
            self._cleanup_object(obj)
            gc.collect()
    
    def _cleanup_object(self, obj: Any) -> None:
        """Clean up specific object attributes."""
        if hasattr(obj, 'performance_history'):
            obj.performance_history = self.bound_list(
                obj.performance_history, 
                self.max_performance_history
            )
        
        if hasattr(obj, 'session_history'):
            obj.session_history = self.bound_list(
                obj.session_history, 
                self.max_session_history
            )
        
        if hasattr(obj, 'governor_decisions'):
            obj.governor_decisions = self.bound_list(
                obj.governor_decisions, 
                100  # Max governor decisions
            )
        
        if hasattr(obj, 'architect_evolutions'):
            obj.architect_evolutions = self.bound_list(
                obj.architect_evolutions, 
                100  # Max architect evolutions
            )
        
        # Clean up action tracking data
        if hasattr(obj, 'available_actions_memory'):
            memory = obj.available_actions_memory
            if 'action_history' in memory:
                memory['action_history'] = self.bound_list(
                    memory['action_history'], 
                    1000  # Max action history
                )
            
            if 'action_sequences' in memory:
                memory['action_sequences'] = self.bound_list(
                    memory['action_sequences'], 
                    500  # Max action sequences
                )
            
            if 'winning_action_sequences' in memory:
                memory['winning_action_sequences'] = self.bound_list(
                    memory['winning_action_sequences'], 
                    200  # Max winning sequences
                )
        
        # Clean up coordinate tracking data
        if hasattr(obj, '_coordinate_tracking'):
            obj._coordinate_tracking = self.bound_dict(
                obj._coordinate_tracking, 
                500  # Max coordinate tracking
            )
        
        if hasattr(obj, '_tried_coordinates'):
            obj._tried_coordinates = self.bound_dict(
                obj._tried_coordinates, 
                500  # Max tried coordinates
            )
        
        # Clean up frame tracking data
        if hasattr(obj, '_frame_history'):
            obj._frame_history = self.bound_dict(
                obj._frame_history, 
                50  # Max frame history
            )
        
        if hasattr(obj, 'frame_stagnation_tracker'):
            obj.frame_stagnation_tracker = self.bound_dict(
                obj.frame_stagnation_tracker, 
                100  # Max stagnation tracking
            )


class BoundedList:
    """A list that automatically bounds itself to prevent memory leaks."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._data = deque(maxlen=max_size)
    
    def append(self, item: Any) -> None:
        """Append item, automatically removing oldest if at capacity."""
        self._data.append(item)
    
    def extend(self, items: List[Any]) -> None:
        """Extend with items, automatically removing oldest if at capacity."""
        for item in items:
            self._data.append(item)
    
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


class BoundedDict:
    """A dictionary that automatically bounds itself to prevent memory leaks."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._data = {}
        self._access_order = deque(maxlen=max_size)
    
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


def apply_memory_leak_fixes(obj: Any) -> None:
    """Apply memory leak fixes to an object."""
    fixer = MemoryLeakFixer()
    
    # Replace lists with bounded lists
    if hasattr(obj, 'performance_history') and isinstance(obj.performance_history, list):
        bounded_list = BoundedList(max_size=100)
        bounded_list.extend(obj.performance_history)
        obj.performance_history = bounded_list
    
    if hasattr(obj, 'session_history') and isinstance(obj.session_history, list):
        bounded_list = BoundedList(max_size=100)
        bounded_list.extend(obj.session_history)
        obj.session_history = bounded_list
    
    if hasattr(obj, 'governor_decisions') and isinstance(obj.governor_decisions, list):
        bounded_list = BoundedList(max_size=100)
        bounded_list.extend(obj.governor_decisions)
        obj.governor_decisions = bounded_list
    
    if hasattr(obj, 'architect_evolutions') and isinstance(obj.architect_evolutions, list):
        bounded_list = BoundedList(max_size=100)
        bounded_list.extend(obj.architect_evolutions)
        obj.architect_evolutions = bounded_list
    
    # Replace dictionaries with bounded dictionaries
    if hasattr(obj, '_coordinate_tracking') and isinstance(obj._coordinate_tracking, dict):
        bounded_dict = BoundedDict(max_size=500)
        bounded_dict.update(obj._coordinate_tracking)
        obj._coordinate_tracking = bounded_dict
    
    if hasattr(obj, '_tried_coordinates') and isinstance(obj._tried_coordinates, dict):
        bounded_dict = BoundedDict(max_size=500)
        bounded_dict.update(obj._tried_coordinates)
        obj._tried_coordinates = bounded_dict
    
    if hasattr(obj, 'frame_stagnation_tracker') and isinstance(obj.frame_stagnation_tracker, dict):
        bounded_dict = BoundedDict(max_size=100)
        bounded_dict.update(obj.frame_stagnation_tracker)
        obj.frame_stagnation_tracker = bounded_dict


def monitor_memory_usage() -> Dict[str, Any]:
    """Monitor current memory usage."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss,  # Resident Set Size
        'vms': memory_info.vms,  # Virtual Memory Size
        'percent': process.memory_percent(),
        'available': psutil.virtual_memory().available,
        'total': psutil.virtual_memory().total
    }
