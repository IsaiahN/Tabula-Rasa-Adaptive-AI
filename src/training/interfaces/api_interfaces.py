"""
API Management Interfaces

Defines interfaces for API management components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from .base_interfaces import APIManagerInterface, ComponentInterface


class APIClientInterface(ABC):
    """Interface for API clients."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to API service."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from API service."""
        pass
    
    @abstractmethod
    async def send_request(self, method: str, endpoint: str, 
                          data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a request to the API."""
        pass
    
    @abstractmethod
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status information."""
        pass


class APIManagerInterface(APIManagerInterface, ComponentInterface):
    """Enhanced interface for API managers."""
    
    @abstractmethod
    def register_client(self, client_name: str, client: APIClientInterface) -> None:
        """Register an API client."""
        pass
    
    @abstractmethod
    def unregister_client(self, client_name: str) -> None:
        """Unregister an API client."""
        pass
    
    @abstractmethod
    def get_client(self, client_name: str) -> Optional[APIClientInterface]:
        """Get a registered client."""
        pass
    
    @abstractmethod
    def list_clients(self) -> List[str]:
        """List all registered clients."""
        pass
    
    @abstractmethod
    def set_rate_limit(self, client_name: str, requests_per_second: float) -> None:
        """Set rate limit for a client."""
        pass
    
    @abstractmethod
    def get_rate_limit_status(self, client_name: str) -> Dict[str, Any]:
        """Get rate limit status for a client."""
        pass
