"""
Handles storing and retrieving subsystem metrics in the database
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from .api import get_database, LogLevel, Component

logger = logging.getLogger(__name__)

async def store_subsystem_metrics(db, subsystem_id: str, metrics_data: Dict[str, Any]) -> bool:
    """Store metrics for a subsystem in the database."""
    try:
        # Convert metrics data to JSON string for storage
        metrics_json = json.dumps(metrics_data)
        
        # Insert the metrics data
        await db.execute("""
            INSERT INTO subsystem_metrics
            (subsystem_id, metrics_data, collection_timestamp,
             health_score, performance_score, efficiency_score,
             error_count, warning_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            subsystem_id,
            metrics_json,
            datetime.now().isoformat(),
            metrics_data.get('health_score', 1.0),  # Default to 1.0 if not provided
            metrics_data.get('performance_score', 1.0),
            metrics_data.get('efficiency_score', 1.0),
            metrics_data.get('error_count', 0),
            metrics_data.get('warning_count', 0)
        ))
        
        # Log the event
        await db.log_system_event(
            LogLevel.INFO,
            Component.SUBSYSTEM_MONITOR,
            f"Stored metrics for subsystem {subsystem_id}",
            metrics_data,
            session_id=subsystem_id
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to store metrics for subsystem {subsystem_id}: {e}")
        return False

async def get_subsystem_metrics(db, subsystem_id: str) -> Optional[Dict[str, Any]]:
    """Get the latest metrics for a subsystem."""
    try:
        result = await db.fetch_one("""
            SELECT metrics_data
            FROM subsystem_metrics
            WHERE subsystem_id = ?
            ORDER BY collection_timestamp DESC
            LIMIT 1
        """, (subsystem_id,))
        
        if result:
            return json.loads(result["metrics_data"])
        return None
        
    except Exception as e:
        logger.error(f"Failed to get metrics for subsystem {subsystem_id}: {e}")
        return None

async def get_subsystem_metrics_history(db, subsystem_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Get historical metrics for a subsystem."""
    try:
        results = await db.fetch_all("""
            SELECT metrics_data, collection_timestamp
            FROM subsystem_metrics
            WHERE subsystem_id = ?
            ORDER BY collection_timestamp DESC
            LIMIT ?
        """, (subsystem_id, limit))
        
        return [
            {
                "timestamp": result["collection_timestamp"],
                "metrics": json.loads(result["metrics_data"])
            }
            for result in results
        ]
        
    except Exception as e:
        logger.error(f"Failed to get metrics history for subsystem {subsystem_id}: {e}")
        return []

async def store_subsystem_config(db, subsystem_id: str, config_data: Dict[str, Any]) -> bool:
    """Store configuration for a subsystem."""
    try:
        config_json = json.dumps(config_data)
        
        # Insert or update configuration
        await db.execute("""
            INSERT OR REPLACE INTO subsystem_configs
            (subsystem_id, config_data, last_updated)
            VALUES (?, ?, ?)
        """, (subsystem_id, config_json, datetime.now().isoformat()))
        
        await db.log_system_event(
            LogLevel.INFO,
            Component.SUBSYSTEM_MONITOR,
            f"Updated configuration for subsystem {subsystem_id}",
            config_data,
            session_id=subsystem_id
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to store config for subsystem {subsystem_id}: {e}")
        return False

async def get_subsystem_config(db, subsystem_id: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a subsystem."""
    try:
        result = await db.fetch_one("""
            SELECT config_data
            FROM subsystem_configs
            WHERE subsystem_id = ?
        """, (subsystem_id,))
        
        if result:
            return json.loads(result["config_data"])
        return None
        
    except Exception as e:
        logger.error(f"Failed to get config for subsystem {subsystem_id}: {e}")
        return None