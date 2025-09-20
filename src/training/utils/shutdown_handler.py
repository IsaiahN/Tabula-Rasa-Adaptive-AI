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
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            logger.debug("Signal handlers setup successfully")
        except Exception as e:
            logger.warning(f"Could not setup signal handlers: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
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
