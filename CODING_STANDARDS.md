# Coding Standards

This document defines the coding standards and best practices for the Tabula Rasa project.

## Overview

We follow Python best practices with a focus on:
- **Readability**: Code should be self-documenting and easy to understand
- **Maintainability**: Code should be easy to modify and extend
- **Consistency**: All code should follow the same patterns and conventions
- **Performance**: Code should be efficient and scalable

## Code Formatting

### Black Configuration
- **Line Length**: 120 characters maximum
- **Target Python**: 3.8+
- **String Quotes**: Double quotes preferred

### Import Organization (isort)
- **Profile**: Black-compatible
- **Line Length**: 120 characters
- **Multi-line Output**: Vertical hanging indent
- **Trailing Comma**: Always include
- **Parentheses**: Use for multi-line imports

### Example
```python
# Good
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from .base_interfaces import ComponentInterface
from .memory_interfaces import MemoryManagerInterface


@dataclass
class CacheConfig:
    """Configuration for cache manager."""
    backend: str = "memory"
    max_size: int = 1000
    ttl_seconds: int = 3600


class CacheManager(ComponentInterface):
    """Centralized cache manager supporting multiple backends."""
    
    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self._cache = self._create_backend()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self._cache.get(key)
```

## Naming Conventions

### Variables and Functions
- **snake_case** for variables, functions, and methods
- **Descriptive names** that clearly indicate purpose
- **Avoid abbreviations** unless they're widely understood

```python
# Good
user_session_data = get_user_session_data()
max_retry_attempts = 3
is_connection_healthy = check_connection()

# Bad
usd = get_usd()
mra = 3
ich = check_conn()
```

### Classes
- **PascalCase** for class names
- **Descriptive names** that indicate the class purpose
- **Interface suffix** for abstract base classes

```python
# Good
class CacheManager(ComponentInterface):
    pass

class MemoryManagerInterface(ABC):
    pass

# Bad
class cache_manager:
    pass

class MemoryManager:
    pass  # Not an interface
```

### Constants
- **UPPER_SNAKE_CASE** for module-level constants
- **Group related constants** in classes or modules

```python
# Good
MAX_CACHE_SIZE = 1000
DEFAULT_TTL_SECONDS = 3600
CACHE_BACKENDS = ["memory", "disk", "redis"]

# Bad
maxCacheSize = 1000
defaultTtlSeconds = 3600
```

## Documentation

### Docstrings
- **Google style** docstrings for all public functions, classes, and modules
- **Type hints** for all function parameters and return values
- **Examples** for complex functions

```python
def process_training_data(
    data: List[Dict[str, Any]], 
    config: TrainingConfig,
    validate: bool = True
) -> ProcessedData:
    """Process training data according to configuration.
    
    Args:
        data: Raw training data as list of dictionaries
        config: Training configuration parameters
        validate: Whether to validate data before processing
        
    Returns:
        ProcessedData object containing processed training data
        
    Raises:
        ValidationError: If data validation fails and validate=True
        ProcessingError: If data processing fails
        
    Example:
        >>> config = TrainingConfig(batch_size=32)
        >>> data = [{"input": [1, 2, 3], "output": [4, 5, 6]}]
        >>> processed = process_training_data(data, config)
        >>> print(processed.batch_count)
        1
    """
    pass
```

### Comments
- **Explain why, not what** - the code should be self-explanatory
- **Update comments** when code changes
- **Remove outdated comments**

```python
# Good
# Use exponential backoff to avoid overwhelming the API
delay = min(2 ** attempt, 60)

# Bad
# Increment attempt by 1
attempt += 1
```

## Type Hints

### Required Type Hints
- **All function parameters** and return values
- **Class attributes** that are not obvious
- **Module-level variables** that are not constants

```python
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass

@dataclass
class TrainingMetrics:
    """Training performance metrics."""
    accuracy: float
    loss: float
    epoch: int
    timestamp: datetime

class TrainingMonitor:
    """Monitors training progress and metrics."""
    
    def __init__(self, session_id: str) -> None:
        self.session_id: str = session_id
        self._metrics: List[TrainingMetrics] = []
        self._callbacks: List[Callable[[TrainingMetrics], None]] = []
    
    def add_metrics(
        self, 
        accuracy: float, 
        loss: float, 
        epoch: int
    ) -> None:
        """Add new training metrics."""
        metrics = TrainingMetrics(
            accuracy=accuracy,
            loss=loss,
            epoch=epoch,
            timestamp=datetime.now()
        )
        self._metrics.append(metrics)
        
        # Notify callbacks
        for callback in self._callbacks:
            callback(metrics)
    
    def get_average_accuracy(self, last_n: int = 10) -> Optional[float]:
        """Get average accuracy for last N epochs."""
        if not self._metrics:
            return None
        
        recent_metrics = self._metrics[-last_n:]
        return sum(m.accuracy for m in recent_metrics) / len(recent_metrics)
```

## Error Handling

### Exception Handling
- **Specific exceptions** rather than bare except clauses
- **Log errors** with appropriate level and context
- **Re-raise** with additional context when appropriate

```python
import logging

logger = logging.getLogger(__name__)

def load_training_data(file_path: str) -> List[Dict[str, Any]]:
    """Load training data from file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.error(f"Training data file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in training data file {file_path}: {e}")
        raise ValueError(f"Invalid training data format: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error loading training data: {e}")
        raise
```

### Validation
- **Validate inputs** at function boundaries
- **Use Pydantic** for complex data validation
- **Provide clear error messages**

```python
from pydantic import BaseModel, validator
from typing import List

class TrainingConfig(BaseModel):
    """Training configuration with validation."""
    batch_size: int
    learning_rate: float
    epochs: int
    validation_split: float
    
    @validator('batch_size')
    def batch_size_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('batch_size must be positive')
        return v
    
    @validator('learning_rate')
    def learning_rate_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('learning_rate must be positive')
        return v
    
    @validator('validation_split')
    def validation_split_must_be_between_0_and_1(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('validation_split must be between 0 and 1')
        return v
```

## Performance Guidelines

### Memory Management
- **Use generators** for large datasets
- **Close resources** properly with context managers
- **Avoid memory leaks** in long-running processes

```python
def process_large_dataset(file_path: str) -> Iterator[ProcessedItem]:
    """Process large dataset without loading everything into memory."""
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                yield process_item(data)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line: {line[:100]}...")
                continue
```

### Async/Await
- **Use async/await** for I/O operations
- **Don't mix** sync and async code unnecessarily
- **Handle exceptions** properly in async functions

```python
async def fetch_training_data(api_url: str) -> List[Dict[str, Any]]:
    """Fetch training data from API."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(api_url) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get('items', [])
        except aiohttp.ClientError as e:
            logger.error(f"API request failed: {e}")
            raise
```

## Testing Standards

### Test Organization
- **One test file** per module
- **Test classes** for related test methods
- **Descriptive test names** that explain what is being tested

```python
import pytest
from unittest.mock import Mock, patch

class TestCacheManager:
    """Test cases for CacheManager."""
    
    def test_get_existing_key_returns_value(self):
        """Test that get returns value for existing key."""
        cache = CacheManager(CacheConfig())
        cache.set("test_key", "test_value")
        
        result = cache.get("test_key")
        
        assert result == "test_value"
    
    def test_get_nonexistent_key_returns_none(self):
        """Test that get returns None for nonexistent key."""
        cache = CacheManager(CacheConfig())
        
        result = cache.get("nonexistent_key")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Test async operations."""
        cache = CacheManager(CacheConfig())
        
        await cache.async_set("key", "value")
        result = await cache.async_get("key")
        
        assert result == "value"
```

## Code Review Guidelines

### Before Submitting
1. **Run all tests** and ensure they pass
2. **Run linting** tools (black, flake8, mypy)
3. **Update documentation** if needed
4. **Check performance** for critical paths

### Review Checklist
- [ ] Code follows naming conventions
- [ ] Type hints are present and correct
- [ ] Docstrings are complete and accurate
- [ ] Error handling is appropriate
- [ ] Tests cover new functionality
- [ ] Performance is acceptable
- [ ] Code is readable and maintainable

## Tools and Automation

### Pre-commit Hooks
We use pre-commit hooks to automatically enforce coding standards:

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks on all files
pre-commit run --all-files
```

### IDE Configuration
Recommended VS Code settings:

```json
{
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=120"],
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.sortImports.args": ["--profile", "black"],
    "editor.rulers": [120],
    "editor.formatOnSave": true
}
```

## Enforcement

### Automated Checks
- **Pre-commit hooks** run on every commit
- **CI/CD pipeline** runs all checks on pull requests
- **Code coverage** must be above 80%

### Manual Review
- **All code** must be reviewed before merging
- **Standards violations** must be fixed before approval
- **Documentation** must be updated for API changes

## Questions?

If you have questions about these standards or need clarification, please:
1. Check existing code for examples
2. Ask in the team chat
3. Create an issue for discussion
