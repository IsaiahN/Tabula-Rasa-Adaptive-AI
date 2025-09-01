# ARC-AGI-3 API Rate Limiting Implementation

## Overview

This document describes the comprehensive rate limiting system implemented in the Continuous Learning Loop to ensure compliance with the ARC-AGI-3 API rate limits.

## Official Rate Limits

According to the ARC-AGI-3 API documentation:
- **Rate Limit**: 600 requests per minute (RPM)
- **Error Response**: `429` status code with JSON: `{"error":"RATE_LIMIT_EXCEEDED","message":"rate limit has been exceeded"}`
- **Backoff Mechanism**: Exponential backoff required for 429 responses

## Implementation Details

### 1. Rate Limiter Class

```python
class RateLimiter:
    """Rate limiter for ARC-AGI-3 API that respects the 600 RPM limit"""
```

**Key Features:**
- Tracks request timestamps in a sliding 60-second window
- Implements per-second (8 RPS) and per-minute (600 RPM) limits
- Uses conservative 8 RPS limit (20% safety buffer)
- Exponential backoff: 1s → 2s → 4s → 8s → 16s → 32s → 60s (max)
- Automatic retry on 429 responses
- Comprehensive statistics tracking

### 2. Configuration

```python
ARC3_RATE_LIMIT = {
    'requests_per_minute': 600,
    'requests_per_second': 10,      # Theoretical max
    'safe_requests_per_second': 8,  # Conservative limit (20% buffer)
    'backoff_base_delay': 1.0,      # Base delay for exponential backoff
    'backoff_max_delay': 60.0,      # Maximum backoff delay
    'request_timeout': 30.0         # Timeout per request
}
```

### 3. Integration Points

The rate limiter is integrated into all API methods:

#### API Discovery
- `get_available_games()` - Rate limited game list retrieval

#### Session Management
- `_open_scorecard()` - Rate limited scorecard creation
- `_close_scorecard()` - Rate limited scorecard closure
- `_start_game_session()` - Rate limited RESET commands

#### Game Actions (Critical)
- `_send_enhanced_action()` - Rate limited ACTION1-7 commands
  - **Most important**: These are the high-frequency calls during training
  - Handles 429 responses with exponential backoff and retry
  - Maintains action history even during rate limiting

### 4. Training Delays

To ensure rate limit compliance during training:

```python
# Between individual actions (most critical)
await asyncio.sleep(0.15)  # 6.67 RPS actual rate

# Between training episodes
await asyncio.sleep(3.0)   # Conservative episode spacing

# After API failures (progressive backoff)
failure_delay = min(10.0, 5.0 + (consecutive_failures * 2.0))

# After errors (even longer backoff)
error_delay = min(15.0, 5.0 + (consecutive_failures * 3.0))

# Swarm mode (concurrent games)
await asyncio.sleep(2.0)   # More conservative for concurrent operations
```

### 5. Error Handling

#### 429 Response Handling
```python
elif response.status == 429:
    self.rate_limiter.handle_429_response()
    print(f"🚫 Rate limit exceeded - backing off {self.rate_limiter.backoff_delay:.1f}s")
    await asyncio.sleep(self.rate_limiter.backoff_delay)
    return await self._send_enhanced_action(...)  # Recursive retry
```

#### Success Response Handling
```python
if response.status == 200:
    self.rate_limiter.handle_success_response()  # Reset backoff
```

### 6. Monitoring and Statistics

The system provides comprehensive monitoring:

```python
def get_rate_limit_stats(self) -> Dict[str, Any]:
    """Get comprehensive rate limiting statistics"""
    
def print_rate_limit_status(self):
    """Print current rate limiting status"""
```

**Statistics Tracked:**
- Total requests made
- Total 429 responses received
- Current backoff delay
- Consecutive 429s
- Requests in last minute
- Rate limit hit rate
- Success rate
- Efficiency metrics

### 7. Safety Measures

#### Conservative Limits
- Uses 8 RPS instead of theoretical 10 RPS maximum
- Provides 20% safety buffer below the limit
- Progressive delays prevent burst requests

#### Request Spacing
- **Action commands**: 0.15s minimum between requests (6.67 RPS)
- **Episode boundaries**: 3.0s pause for rate compliance
- **Failure recovery**: 5-15s progressive backoff
- **Error handling**: Up to 60s maximum backoff

#### Timeout Protection
- 30-second timeout per request prevents hanging
- Proper async/await usage prevents blocking

### 8. Usage in Training

The rate limiting system is transparent to the training logic:

```python
# Before each API call
await self.rate_limiter.acquire()

# Make the API request
async with aiohttp.ClientSession(timeout=...) as session:
    async with session.post(url, headers=headers, json=payload) as response:
        # Handle response with rate limiting logic
```

### 9. Benefits

1. **API Compliance**: Ensures all requests stay within 600 RPM limit
2. **Reliability**: Handles 429 responses gracefully with exponential backoff
3. **Efficiency**: Maintains maximum safe throughput with 8 RPS limit
4. **Monitoring**: Provides comprehensive statistics for optimization
5. **Robustness**: Progressive delays prevent rapid retry cycles
6. **Transparency**: Rate limiting is handled internally without affecting training logic

### 10. Testing

The rate limiter has been tested with:
- Exponential backoff progression (1s → 2s → 4s...)
- Statistics tracking accuracy
- Request timing compliance
- 429 response simulation
- Success response handling

## Conclusion

This implementation ensures full compliance with the ARC-AGI-3 API rate limits while maintaining optimal training performance. The conservative 8 RPS limit provides a 20% safety buffer, and the exponential backoff mechanism handles any rate limiting gracefully.

The system is production-ready and will prevent any API abuse while allowing for intensive training sessions with hundreds of thousands of actions across multiple concurrent games.
