# Refactoring: Error Handling Unification

## Priority
**HIGH** - Estimated effort: 1-2 days

## Problem

The codebase has **inconsistent error handling patterns** scattered throughout, making it difficult to:
- Debug issues in production
- Understand error flow
- Provide consistent user feedback
- Recover from errors gracefully

### Current Issues

#### 1. Multiple Error Handling Patterns

**Pattern 1: Catch and log to stderr with return**
```python
# controller.py:112-114
try:
    # ... code ...
except Exception as exc:  # pragma: no cover
    print(f"{ERROR_LOG_LABEL} Error: {exc}", file=sys.stderr)
    return
```

**Pattern 2: Catch specific exception and re-raise**
```python
# controller.py:186-188
try:
    # ... code ...
except RuntimeError as exc:
    print(f"{ERROR_LOG_LABEL} {exc}", file=sys.stderr)
    raise
```

**Pattern 3: Silent exception swallowing**
```python
# llm.py:312
try:
    # ... code ...
except Exception:
    return None, None  # No logging, no context
```

**Pattern 4: Bare exception catching**
```python
# Found in 10+ locations
except Exception as exc:  # Too broad!
    # ... handle ...
```

#### 2. Lost Error Context

```python
# audio/capture.py:94-98
try:
    device = sd.query_devices(device_index)
except Exception:
    raise RuntimeError(f"Invalid device index: {device_index}")
    # Original exception info is lost!
```

#### 3. Duplicate Error Logging Code

Found in **10+ locations**:
```python
# controller.py:88, 113, 129, 148, etc.
except Exception as exc:  # pragma: no cover
    print(f"{LABEL} Error message: {exc}", file=sys.stderr)
```

#### 4. No Structured Error Hierarchy

All errors use generic exceptions:
- `RuntimeError` for everything
- `Exception` catching is too broad
- No domain-specific exceptions

## Proposed Solution

Create a **unified error handling system** with:
1. Custom exception hierarchy for domain errors
2. Centralized error handler utility
3. Consistent logging and recovery patterns
4. Structured error context

### Implementation

#### 1. Custom Exception Hierarchy (`errors.py`)

```python
"""Custom exceptions for pi-assistant."""

class PiAssistantError(Exception):
    """Base exception for all pi-assistant errors."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause
        self.message = message

    def __str__(self) -> str:
        if self.cause:
            return f"{self.message} (caused by: {self.cause})"
        return self.message


# Audio errors
class AudioError(PiAssistantError):
    """Base class for audio-related errors."""
    pass

class AudioDeviceError(AudioError):
    """Error with audio device configuration or access."""
    pass

class AudioCaptureError(AudioError):
    """Error capturing audio from input device."""
    pass

class AudioPlaybackError(AudioError):
    """Error playing audio to output device."""
    pass


# Wake word errors
class WakeWordError(PiAssistantError):
    """Base class for wake word detection errors."""
    pass

class WakeWordEngineError(WakeWordError):
    """Error initializing or using wake word engine."""
    pass


# LLM errors
class LLMError(PiAssistantError):
    """Base class for LLM interaction errors."""
    pass

class LLMConnectionError(LLMError):
    """Error connecting to LLM service."""
    pass

class LLMResponseError(LLMError):
    """Error in LLM response format or content."""
    pass

class LLMAuthenticationError(LLMError):
    """Authentication error with LLM service."""
    pass


# WebSocket errors
class WebSocketError(PiAssistantError):
    """Base class for WebSocket errors."""
    pass

class WebSocketConnectionError(WebSocketError):
    """Error connecting to WebSocket."""
    pass

class WebSocketMessageError(WebSocketError):
    """Error sending or receiving WebSocket message."""
    pass


# Configuration errors
class ConfigurationError(PiAssistantError):
    """Base class for configuration errors."""
    pass

class ConfigValidationError(ConfigurationError):
    """Configuration validation failed."""
    pass

class ConfigLoadError(ConfigurationError):
    """Error loading configuration file."""
    pass
```

#### 2. Error Handler Utility (`error_handler.py`)

```python
"""Centralized error handling utilities."""

import sys
import logging
import traceback
from typing import TypeVar, Callable, Optional
from functools import wraps
from .errors import PiAssistantError

T = TypeVar('T')

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Centralized error handler with consistent logging and recovery."""

    @staticmethod
    def log_error(
        error: Exception,
        context: str = "",
        level: int = logging.ERROR
    ) -> None:
        """Log an error with context and traceback."""
        if context:
            logger.log(level, f"{context}: {error}")
        else:
            logger.log(level, str(error))

        # Include traceback for unexpected errors
        if not isinstance(error, PiAssistantError):
            logger.debug("Traceback:", exc_info=error)

    @staticmethod
    def log_and_raise(
        error_class: type[Exception],
        message: str,
        cause: Exception | None = None
    ) -> None:
        """Log an error and raise a new exception with context."""
        if cause:
            logger.error(f"{message} (caused by: {cause})")
            raise error_class(message, cause=cause) from cause
        else:
            logger.error(message)
            raise error_class(message)

    @staticmethod
    def handle_task_error(task_name: str) -> Callable:
        """Decorator for async task error handling."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                try:
                    return await func(*args, **kwargs)
                except PiAssistantError as exc:
                    ErrorHandler.log_error(exc, context=f"Task '{task_name}' failed")
                    raise
                except Exception as exc:
                    ErrorHandler.log_error(
                        exc,
                        context=f"Unexpected error in task '{task_name}'"
                    )
                    raise
            return wrapper
        return decorator

    @staticmethod
    def safe_execute(
        func: Callable[..., T],
        *args,
        default: T | None = None,
        context: str = "",
        **kwargs
    ) -> T | None:
        """Execute a function and return default on error."""
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            ErrorHandler.log_error(exc, context=context)
            return default

    @staticmethod
    def add_task_error_handler(
        task: 'asyncio.Task',
        task_name: str
    ) -> None:
        """Add error handler callback to an asyncio task."""
        def _handle_task_exception(fut: 'asyncio.Task') -> None:
            try:
                fut.result()
            except Exception as exc:
                ErrorHandler.log_error(
                    exc,
                    context=f"Task '{task_name}' failed"
                )

        task.add_done_callback(_handle_task_exception)


# Convenient decorator aliases
def handle_audio_errors(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to wrap audio-related functions with error handling."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            ErrorHandler.log_and_raise(
                AudioError,
                f"Audio error in {func.__name__}",
                cause=exc
            )
    return wrapper


def handle_llm_errors(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to wrap LLM-related functions with error handling."""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            ErrorHandler.log_and_raise(
                LLMError,
                f"LLM error in {func.__name__}",
                cause=exc
            )
    return wrapper
```

#### 3. Retry Logic (`retry.py`)

```python
"""Retry utilities for transient errors."""

import asyncio
import logging
from typing import TypeVar, Callable, Type
from functools import wraps

T = TypeVar('T')
logger = logging.getLogger(__name__)


async def retry_on_error(
    func: Callable[..., T],
    *args,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    retry_on: tuple[Type[Exception], ...] = (Exception,),
    **kwargs
) -> T:
    """Retry a function with exponential backoff."""
    last_exception = None

    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except retry_on as exc:
            last_exception = exc
            if attempt < max_retries - 1:
                wait_time = backoff_factor ** attempt
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {exc}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed")

    raise last_exception


def retry_async(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    retry_on: tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """Decorator for async functions with retry logic."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await retry_on_error(
                func,
                *args,
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                retry_on=retry_on,
                **kwargs
            )
        return wrapper
    return decorator
```

## Usage Examples

### Before (Current)

```python
# controller.py - Inconsistent error handling
try:
    audio_capture = AudioCapture()
except Exception as exc:  # Too broad!
    print(f"{ERROR_LOG_LABEL} Error: {exc}", file=sys.stderr)
    return

try:
    ws_client.connect()
except RuntimeError as exc:  # Lost context
    print(f"{ERROR_LOG_LABEL} {exc}", file=sys.stderr)
    raise
```

### After (Proposed)

```python
from pi_assistant.errors import AudioDeviceError, WebSocketConnectionError
from pi_assistant.error_handler import ErrorHandler, retry_async

# Clear, specific exceptions
try:
    audio_capture = AudioCapture()
except AudioDeviceError as exc:
    ErrorHandler.log_error(exc, context="Failed to initialize audio capture")
    raise

# Retry logic for transient errors
@retry_async(max_retries=3, retry_on=(WebSocketConnectionError,))
async def connect_websocket():
    await ws_client.connect()

# Task error handling
task = asyncio.create_task(process_audio())
ErrorHandler.add_task_error_handler(task, "audio_processing")
```

### Converting Existing Code

**Before:**
```python
# audio/capture.py:94-98
try:
    device = sd.query_devices(device_index)
except Exception:
    raise RuntimeError(f"Invalid device index: {device_index}")
```

**After:**
```python
try:
    device = sd.query_devices(device_index)
except Exception as exc:
    raise AudioDeviceError(
        f"Invalid device index: {device_index}",
        cause=exc
    ) from exc
```

**Before:**
```python
# llm.py:312 - Silent failure
try:
    return extract_audio(response)
except Exception:
    return None, None
```

**After:**
```python
try:
    return extract_audio(response)
except Exception as exc:
    logger.warning(f"Failed to extract audio from response: {exc}")
    return None, None  # Now we know why it failed!
```

## Migration Plan

### Phase 1: Create Error Infrastructure (Day 1, Morning)
1. Create `src/pi_assistant/errors.py` with exception hierarchy
2. Create `src/pi_assistant/error_handler.py` with utilities
3. Create `src/pi_assistant/retry.py` with retry logic
4. Add unit tests for error utilities

### Phase 2: Update Critical Paths (Day 1, Afternoon)
1. Update `audio/capture.py` to use `AudioDeviceError`
2. Update `audio/playback.py` to use `AudioPlaybackError`
3. Update `network/websocket_client.py` to use `WebSocketError`
4. Update `assistant/llm.py` to use `LLMError`

### Phase 3: Update Controllers (Day 2, Morning)
1. Update `cli/controller.py` to use `ErrorHandler`
2. Replace duplicate error logging with `ErrorHandler.log_error`
3. Add retry logic for transient failures
4. Update task error handling to use `add_task_error_handler`

### Phase 4: Update Remaining Modules (Day 2, Afternoon)
1. Update `wake_word/` modules
2. Update `cli/app.py`
3. Search for all `except Exception` and replace with specific types
4. Add logging to silent exception handlers

## Testing Strategy

```python
import pytest
from pi_assistant.errors import AudioDeviceError
from pi_assistant.error_handler import ErrorHandler, retry_async

def test_custom_exception_with_cause():
    """Test custom exception preserves cause."""
    cause = ValueError("Original error")
    error = AudioDeviceError("Device not found", cause=cause)

    assert str(error) == "Device not found (caused by: Original error)"
    assert error.cause is cause

def test_error_handler_logging(caplog):
    """Test error handler logs with context."""
    error = AudioDeviceError("Test error")
    ErrorHandler.log_error(error, context="Audio setup")

    assert "Audio setup: Test error" in caplog.text

@pytest.mark.asyncio
async def test_retry_decorator():
    """Test retry decorator with exponential backoff."""
    attempt_count = 0

    @retry_async(max_retries=3, backoff_factor=0.1)
    async def failing_func():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError("Transient error")
        return "success"

    result = await failing_func()

    assert result == "success"
    assert attempt_count == 3
```

## Benefits

1. **Consistency**: All errors handled the same way
2. **Debuggability**: Error context is preserved and logged
3. **Maintainability**: Single place to update error handling logic
4. **Type safety**: Specific exception types for different error domains
5. **Resilience**: Retry logic for transient errors
6. **Code quality**: No more silent failures or bare `except Exception`

## Migration Checklist

- [ ] Create error infrastructure files
- [ ] Update all `audio/` modules to use custom exceptions
- [ ] Update all `network/` modules to use custom exceptions
- [ ] Update all `assistant/` modules to use custom exceptions
- [ ] Replace duplicate error logging in `controller.py`
- [ ] Add retry logic for WebSocket connections
- [ ] Search and replace all `except Exception` (47 occurrences)
- [ ] Add logging to all silent exception handlers
- [ ] Write unit tests for error handling (90%+ coverage)
- [ ] Update documentation with error handling guidelines

## Success Metrics

- [ ] Zero `except Exception` without re-raise
- [ ] Zero silent exception swallowing
- [ ] All errors have context in logs
- [ ] 90%+ test coverage on error paths
- [ ] Error handling code reduced by 50%

## Related Refactorings

- **Audio Controller State Machine**: States can have consistent error handling
- **Configuration Module**: Use `ConfigurationError` for validation
- **Type Safety**: Strong exception typing improves error handling
