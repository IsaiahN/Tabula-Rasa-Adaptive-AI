# src/core/component_coordination/__init__.py
from .coordinator import ComponentCoordinator
from .integration import SystemIntegration as ComponentSystemIntegration

__all__ = [
    'ComponentCoordinator',
    'ComponentSystemIntegration'
]
