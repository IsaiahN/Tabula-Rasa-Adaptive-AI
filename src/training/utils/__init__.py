"""
Training Utilities

Contains utility functions for lazy imports, shutdown handling,
and compatibility.
"""

from .lazy_imports import LazyImports
from .shutdown_handler import ShutdownHandler
from .compatibility import CompatibilityShim

__all__ = [
    'LazyImports',
    'ShutdownHandler',
    'CompatibilityShim'
]
