"""Frame provider interface and lightweight implementations for minimal mode."""
from typing import Any, Optional, Tuple, List

class FrameProvider:
    """Abstract interface for providing frames to vision components."""
    async def get_frame(self) -> Optional[Any]:
        """Return the latest frame representation. Can be a 2D array, bytes, or structured dict."""
        raise NotImplementedError()

class DummyFrameProvider(FrameProvider):
    """Simple dummy frame provider that returns a small synthetic frame."""
    def __init__(self, width: int = 32, height: int = 32):
        self.width = width
        self.height = height

    async def get_frame(self) -> List[List[int]]:
        # Return a tiny grid with zeros (no objects)
        return [[0 for _ in range(self.width)] for _ in range(self.height)]
