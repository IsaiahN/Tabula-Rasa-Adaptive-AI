"""API module exports for tb-minimal compatibility."""

# Import the actual APIManager from its current location
try:
    from src.training.api.api_manager import APIManager
except ImportError:
    APIManager = None

# Re-export for backward compatibility
__all__ = ['APIManager']