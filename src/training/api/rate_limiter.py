"""
Rate Limiter

Implements rate limiting for API calls to prevent overwhelming the server.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    max_requests_per_minute: int = 550  # Close to API limit (600 - 50 buffer)
    max_requests_per_hour: int = 30000  # Reasonable hourly limit
    burst_limit: int = 100  # Allow more burst requests for training
    window_size_seconds: int = 60
    # Dynamic rate limiting
    warning_threshold: float = 0.8  # Warn when 80% of limit reached
    pause_threshold: float = 0.9   # Pause when 90% of limit reached
    min_pause_seconds: float = 0.1  # Minimum pause time
    max_pause_seconds: float = 5.0  # Maximum pause time

class RateLimiter:
    """Implements rate limiting for API calls."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.request_times: deque = deque()
        self.hourly_requests: deque = deque()
        self.burst_requests: deque = deque()
        self.blocked_until: Optional[float] = None
        self.total_requests = 0
        self.blocked_requests = 0
    
    def can_make_request(self) -> bool:
        """Check if a request can be made without violating rate limits."""
        current_time = time.time()
        
        # Check if we're currently blocked
        if self.blocked_until and current_time < self.blocked_until:
            return False
        
        # Clean up old requests
        self._cleanup_old_requests(current_time)
        
        # Check burst limit
        if len(self.burst_requests) >= self.config.burst_limit:
            logger.warning("Burst limit exceeded")
            return False
        
        # Check per-minute limit
        if len(self.request_times) >= self.config.max_requests_per_minute:
            logger.warning("Per-minute rate limit exceeded")
            return False
        
        # Check per-hour limit
        if len(self.hourly_requests) >= self.config.max_requests_per_hour:
            logger.warning("Per-hour rate limit exceeded")
            return False
        
        return True
    
    def record_request(self) -> bool:
        """Record a request and return whether it was allowed."""
        current_time = time.time()
        
        if not self.can_make_request():
            self.blocked_requests += 1
            return False
        
        # Record the request
        self.request_times.append(current_time)
        self.hourly_requests.append(current_time)
        self.burst_requests.append(current_time)
        self.total_requests += 1
        
        return True
    
    def record_success(self) -> None:
        """Record a successful request to adjust rate limiting."""
        # Successful requests allow us to be slightly more aggressive
        pass
    
    def record_failure(self) -> None:
        """Record a failed request to adjust rate limiting."""
        # Failed requests should make us more conservative
        # Reduce burst limit temporarily
        if self.config.burst_limit > 5:
            self.config.burst_limit = max(5, self.config.burst_limit - 2)
    
    def should_pause(self) -> Tuple[bool, float]:
        """
        Check if we should pause and for how long.
        Returns (should_pause, pause_duration_seconds)
        """
        current_usage = self.get_current_usage()
        usage_ratio = current_usage / self.config.max_requests_per_minute
        
        if usage_ratio >= self.config.pause_threshold:
            # Calculate pause duration based on how close we are to the limit
            excess_ratio = usage_ratio - self.config.pause_threshold
            pause_duration = self.config.min_pause_seconds + (
                excess_ratio * (self.config.max_pause_seconds - self.config.min_pause_seconds)
            )
            return True, min(pause_duration, self.config.max_pause_seconds)
        
        return False, 0.0
    
    def get_usage_warning(self) -> Optional[str]:
        """Get usage warning if approaching limits."""
        current_usage = self.get_current_usage()
        usage_ratio = current_usage / self.config.max_requests_per_minute
        
        if usage_ratio >= self.config.warning_threshold:
            return f"Rate limit usage: {current_usage}/{self.config.max_requests_per_minute} ({usage_ratio:.1%})"
        
        return None
    
    def wait_if_needed(self) -> float:
        """Wait if necessary to respect rate limits. Returns wait time."""
        if self.can_make_request():
            return 0.0
        
        current_time = time.time()
        wait_time = 0.0
        
        # Calculate wait time based on the most restrictive limit
        if len(self.request_times) >= self.config.max_requests_per_minute:
            # Wait until the oldest request in the minute window expires
            oldest_request = self.request_times[0]
            wait_time = max(wait_time, oldest_request + 60 - current_time)
        
        if len(self.hourly_requests) >= self.config.max_requests_per_hour:
            # Wait until the oldest request in the hour window expires
            oldest_request = self.hourly_requests[0]
            wait_time = max(wait_time, oldest_request + 3600 - current_time)
        
        if wait_time > 0:
            logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)
            self._cleanup_old_requests(time.time())
        
        return wait_time
    
    def _cleanup_old_requests(self, current_time: float) -> None:
        """Remove old requests from tracking deques."""
        # Clean up minute window (60 seconds)
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()
        
        # Clean up hour window (3600 seconds)
        while self.hourly_requests and current_time - self.hourly_requests[0] > 3600:
            self.hourly_requests.popleft()
        
        # Clean up burst window (30 seconds)
        while self.burst_requests and current_time - self.burst_requests[0] > 30:
            self.burst_requests.popleft()
    
    def get_current_usage(self) -> int:
        """Get current usage count for the current window."""
        current_time = time.time()
        self._cleanup_old_requests(current_time)
        return len(self.request_times)
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        current_time = time.time()
        self._cleanup_old_requests(current_time)
        
        return {
            "requests_last_minute": len(self.request_times),
            "requests_last_hour": len(self.hourly_requests),
            "burst_requests": len(self.burst_requests),
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "blocked_until": self.blocked_until,
            "can_make_request": self.can_make_request(),
            "current_usage": len(self.request_times),
            "max_requests": self.config.max_requests_per_minute,
            "limits": {
                "max_per_minute": self.config.max_requests_per_minute,
                "max_per_hour": self.config.max_requests_per_hour,
                "burst_limit": self.config.burst_limit
            }
        }
    
    def get_usage_warning(self) -> Optional[str]:
        """Get a warning message if approaching rate limits."""
        current_time = time.time()
        self._cleanup_old_requests(current_time)
        
        # Check if we're approaching limits
        minute_usage = len(self.request_times) / self.config.max_requests_per_minute
        hour_usage = len(self.hourly_requests) / self.config.max_requests_per_hour
        
        if minute_usage > 0.9:  # 90% of minute limit
            return f"WARNING: Using {minute_usage:.1%} of minute rate limit ({len(self.request_times)}/{self.config.max_requests_per_minute})"
        elif hour_usage > 0.8:  # 80% of hour limit
            return f"WARNING: Using {hour_usage:.1%} of hour rate limit ({len(self.hourly_requests)}/{self.config.max_requests_per_hour})"
        
        return None
    
    def reset(self) -> None:
        """Reset rate limiter state."""
        self.request_times.clear()
        self.hourly_requests.clear()
        self.burst_requests.clear()
        self.blocked_until = None
        self.total_requests = 0
        self.blocked_requests = 0
        logger.info("Rate limiter reset")
    
    def set_blocked_until(self, timestamp: float) -> None:
        """Set a block until a specific timestamp."""
        self.blocked_until = timestamp
        logger.warning(f"Rate limiter blocked until {timestamp}")
    
    def is_blocked(self) -> bool:
        """Check if currently blocked."""
        if not self.blocked_until:
            return False
        return time.time() < self.blocked_until
