"""
Shutdown Handler Utilities

Provides graceful shutdown handling for training processes.
"""

import signal
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

class ShutdownHandler:
    """Handles graceful shutdown of training processes."""
    
    def __init__(self):
        self._shutdown_requested = False
        self._shutdown_callbacks = []
    
    def request_shutdown(self):
        """Request shutdown from external code."""
        if self._shutdown_requested:
            return
            
        logger.info("Shutdown requested externally")
        self._shutdown_requested = True
        
        # Call all registered shutdown callbacks
        for callback in self._shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in shutdown callback: {e}")
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested
    
    def add_shutdown_callback(self, callback: Callable[[], None]):
        """Add a callback to be called during shutdown."""
        self._shutdown_callbacks.append(callback)
    
    def remove_shutdown_callback(self, callback: Callable[[], None]):
        """Remove a shutdown callback."""
        if callback in self._shutdown_callbacks:
            self._shutdown_callbacks.remove(callback)
