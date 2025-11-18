# Refactoring: Quick Wins

## Priority
**LOW** - Estimated effort: 1-2 days total

## Overview

These are **small, focused refactorings** that can be done independently and provide immediate value with minimal risk. Each can be completed in 1-4 hours.

---

## Quick Win #1: Extract Duplicate Error Logging

### Effort
**1-2 hours**

### Problem

Error logging pattern duplicated **10+ times**:

```python
# controller.py:88, 113, 129, 148, etc.
except Exception as exc:  # pragma: no cover
    print(f"{LABEL} Error message: {exc}", file=sys.stderr)
```

### Solution

Create utility function:

```python
# logging_utils.py
def log_error(message: str, error: Exception, label: str = ERROR_LOG_LABEL) -> None:
    """Log an error with consistent formatting."""
    print(f"{label} {message}: {error}", file=sys.stderr)
```

### Usage

**Before:**
```python
try:
    audio_capture = AudioCapture()
except Exception as exc:
    print(f"{ERROR_LOG_LABEL} Error: {exc}", file=sys.stderr)
    return
```

**After:**
```python
try:
    audio_capture = AudioCapture()
except Exception as exc:
    log_error("Failed to initialize audio capture", exc)
    return
```

### Benefits
- Consistent error formatting
- Single place to update logging format
- Easier to add structured logging later

---

## Quick Win #2: Extract Task Error Handler

### Effort
**1-2 hours**

### Problem

Task error handling duplicated:

```python
# controller.py:85-91 and 143-150
def _log_task_error(fut: asyncio.Task):
    try:
        fut.result()
    except Exception as exc:
        verbose_print(f"{LABEL} Error: {exc}")

task.add_done_callback(_log_task_error)
```

### Solution

Create reusable decorator:

```python
# async_utils.py
from functools import wraps
import asyncio
from typing import Callable, TypeVar

T = TypeVar('T')

def log_task_errors(task_name: str) -> Callable:
    """Decorator to log errors from async tasks."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                log_error(f"Task '{task_name}' failed", exc)
                raise
        return wrapper
    return decorator

def add_error_logging(task: asyncio.Task, task_name: str) -> None:
    """Add error logging callback to a task."""
    def _log_error(fut: asyncio.Task) -> None:
        try:
            fut.result()
        except Exception as exc:
            log_error(f"Task '{task_name}' failed", exc)

    task.add_done_callback(_log_error)
```

### Usage

**Before:**
```python
task = asyncio.create_task(process_events())

def _log_task_error(fut: asyncio.Task):
    try:
        fut.result()
    except Exception as exc:
        verbose_print(f"{LABEL} Error: {exc}")

task.add_done_callback(_log_task_error)
```

**After:**
```python
task = asyncio.create_task(process_events())
add_error_logging(task, "event_processing")
```

Or with decorator:
```python
@log_task_errors("event_processing")
async def process_events():
    # ...
```

### Benefits
- No duplicate error handling code
- Consistent task error logging
- Easy to extend with more functionality

---

## Quick Win #3: Extract Audio Resampling Logic

### Effort
**2-3 hours**

### Problem

Resampling logic scattered across:
- `wake_word.py:103`
- `controller.py:191-207`
- `audio/playback.py:70-77`

### Solution

Create utility module:

```python
# audio/resampler.py
import numpy as np
from scipy import signal

class AudioResampler:
    """Handles audio resampling between different sample rates."""

    @staticmethod
    def resample(
        audio_data: bytes,
        source_rate: int,
        target_rate: int,
        source_width: int = 2
    ) -> bytes:
        """Resample audio from source rate to target rate."""
        if source_rate == target_rate:
            return audio_data

        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Calculate resampling ratio
        num_samples = int(len(audio_array) * target_rate / source_rate)

        # Resample
        resampled = signal.resample(audio_array, num_samples)

        # Convert back to bytes
        return resampled.astype(np.int16).tobytes()

    @staticmethod
    def resample_stream(
        source_rate: int,
        target_rate: int
    ) -> Callable[[bytes], bytes]:
        """Create a resampling function for streaming audio."""
        def resample_chunk(chunk: bytes) -> bytes:
            return AudioResampler.resample(chunk, source_rate, target_rate)
        return resample_chunk
```

### Usage

**Before:**
```python
# Duplicated in multiple files
audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
num_samples = int(len(audio_array) * TARGET_RATE / SOURCE_RATE)
resampled = signal.resample(audio_array, num_samples)
resampled_bytes = resampled.astype(np.int16).tobytes()
```

**After:**
```python
from pi_assistant.audio.resampler import AudioResampler

resampled_bytes = AudioResampler.resample(
    audio_chunk,
    source_rate=SAMPLE_RATE,
    target_rate=TARGET_RATE
)
```

### Benefits
- Single, tested resampling implementation
- Easier to optimize or replace algorithm
- Clearer intent in calling code

---

## Quick Win #4: Extract RMS Calculation

### Effort
**30 minutes**

### Problem

RMS (Root Mean Square) calculation in `controller.py:53-64` could be in audio utilities.

### Solution

Move to audio module:

```python
# audio/utils.py
import numpy as np

def calculate_rms(audio_bytes: bytes) -> float:
    """Calculate RMS (Root Mean Square) of audio data."""
    if not audio_bytes:
        return 0.0

    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))

    return rms / 32768.0  # Normalize to 0-1 range
```

### Usage

**Before:**
```python
# controller.py:53-64
def calculate_rms(audio_bytes: bytes) -> float:
    if not audio_bytes:
        return 0.0
    # ... calculation
```

**After:**
```python
from pi_assistant.audio.utils import calculate_rms

# Use anywhere
rms = calculate_rms(audio_chunk)
```

### Benefits
- Reusable across modules
- Testable in isolation
- Natural location (audio utilities)

---

## Quick Win #5: Create Audio Constants Module

### Effort
**1 hour**

### Problem

Audio constants scattered throughout codebase:
- Sample rates
- Bit depths
- Channel counts
- Chunk sizes

### Solution

Create constants module:

```python
# audio/constants.py
"""Audio processing constants."""

# Sample rates
SAMPLE_RATE_16KHZ = 16000
SAMPLE_RATE_24KHZ = 24000
SAMPLE_RATE_48KHZ = 48000

# Bit depths
BIT_DEPTH_16 = 16
BYTES_PER_SAMPLE_16BIT = 2

# Channels
MONO = 1
STEREO = 2

# Common calculations
def samples_to_bytes(samples: int, bit_depth: int = BIT_DEPTH_16) -> int:
    """Convert sample count to byte count."""
    return samples * (bit_depth // 8)

def bytes_to_samples(byte_count: int, bit_depth: int = BIT_DEPTH_16) -> int:
    """Convert byte count to sample count."""
    return byte_count // (bit_depth // 8)

def duration_to_samples(duration_seconds: float, sample_rate: int) -> int:
    """Convert duration to sample count."""
    return int(duration_seconds * sample_rate)
```

### Usage

```python
from pi_assistant.audio.constants import (
    SAMPLE_RATE_16KHZ,
    duration_to_samples,
    samples_to_bytes
)

chunk_samples = duration_to_samples(0.1, SAMPLE_RATE_16KHZ)
chunk_bytes = samples_to_bytes(chunk_samples)
```

### Benefits
- Clear, self-documenting constants
- Utility functions for common calculations
- Reduces magic numbers

---

## Quick Win #6: Extract Verbose Print Logic

### Effort
**30 minutes**

### Problem

Verbose printing scattered with module globals:

```python
# logging_utils.py:34-40
_VERBOSE_LOGGING = False
_VERBOSE_LOG_FILE: Optional[TextIO] = None

def verbose_print(message: str) -> None:
    if _VERBOSE_LOGGING:
        # ... implementation
```

### Solution

Create logger class:

```python
# logging_utils.py
from typing import Optional, TextIO
from pathlib import Path

class VerboseLogger:
    """Logger for verbose output."""

    def __init__(self, enabled: bool = False, log_file: Optional[Path] = None):
        self.enabled = enabled
        self.log_file_path = log_file
        self._log_file: Optional[TextIO] = None

        if self.enabled and log_file:
            self._log_file = open(log_file, 'a')

    def log(self, message: str) -> None:
        """Log a verbose message."""
        if not self.enabled:
            return

        if self._log_file:
            self._log_file.write(f"{message}\n")
            self._log_file.flush()
        else:
            print(message)

    def close(self) -> None:
        """Close log file if open."""
        if self._log_file:
            self._log_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```

### Usage

**Before:**
```python
from pi_assistant.cli.logging_utils import verbose_print, set_verbose_logging

set_verbose_logging(True, log_file_path)
verbose_print("Message")
```

**After:**
```python
from pi_assistant.cli.logging_utils import VerboseLogger

logger = VerboseLogger(enabled=True, log_file=Path("log.txt"))
logger.log("Message")
```

### Benefits
- No module globals
- Testable with different configs
- Context manager support

---

## Quick Win #7: Add __all__ to Public Modules

### Effort
**1 hour**

### Problem

No explicit exports, unclear public API.

### Solution

Add `__all__` to each module:

```python
# audio/__init__.py
"""Audio processing module."""

from .capture import AudioCapture
from .playback import SpeechPlayer
from .utils import calculate_rms
from .resampler import AudioResampler

__all__ = [
    "AudioCapture",
    "SpeechPlayer",
    "calculate_rms",
    "AudioResampler",
]
```

### Benefits
- Clear public API
- Better IDE autocomplete
- Helps with `from module import *`

---

## Quick Win #8: Extract Magic Numbers

### Effort
**1-2 hours**

### Problem

Magic numbers throughout code:

```python
# controller.py
if rms < 0.02:  # What is 0.02?
    silence_duration += 0.1  # What is 0.1?
```

### Solution

Use named constants:

```python
# constants.py
DEFAULT_SILENCE_THRESHOLD = 0.02
CHUNK_DURATION_SECONDS = 0.1

# controller.py
from pi_assistant.constants import DEFAULT_SILENCE_THRESHOLD, CHUNK_DURATION_SECONDS

if rms < DEFAULT_SILENCE_THRESHOLD:
    silence_duration += CHUNK_DURATION_SECONDS
```

### Benefits
- Self-documenting code
- Easy to adjust values
- Prevents typos

---

## Implementation Priority

| Quick Win | Effort | Impact | Priority |
|-----------|--------|--------|----------|
| #1: Error Logging | 1-2h | High | **1** |
| #2: Task Error Handler | 1-2h | High | **2** |
| #4: RMS Calculation | 30m | Medium | **3** |
| #7: Add __all__ | 1h | Medium | **4** |
| #8: Extract Magic Numbers | 1-2h | Medium | **5** |
| #6: Verbose Print | 30m | Low | **6** |
| #3: Audio Resampling | 2-3h | Medium | **7** |
| #5: Audio Constants | 1h | Low | **8** |

## Combined Effort

- **Total**: 8-13 hours
- **High priority (1-2)**: 2-4 hours
- **Quick wins (30m-1h)**: 2-3 hours

## Benefits Summary

1. **Reduced duplication**: ~100+ lines of duplicate code removed
2. **Better organization**: Utilities in appropriate modules
3. **Easier testing**: Smaller, focused functions
4. **Clearer code**: Named constants, explicit APIs
5. **Immediate value**: Each provides standalone benefit

## Migration Strategy

These can be done **incrementally** without blocking other work:

1. Start with error logging (highest impact, 1-2h)
2. Do RMS and magic numbers (easy wins, 1-2h)
3. Tackle task error handling (2h)
4. Add remaining as time permits

## Success Metrics

- [ ] Zero duplicate error logging patterns
- [ ] Zero duplicate task error handling
- [ ] All magic numbers extracted to constants
- [ ] All public modules have `__all__`
- [ ] Audio utilities in `audio/` module
- [ ] Test coverage added for extracted utilities
