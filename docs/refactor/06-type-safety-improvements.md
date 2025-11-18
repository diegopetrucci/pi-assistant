# Refactoring: Type Safety Improvements

## Priority
**MEDIUM** - Estimated effort: 2 days

## Problem

The codebase has **inconsistent type annotations** and missing type definitions, leading to:

1. **Runtime errors** that could be caught at development time
2. **Poor IDE support** - no autocomplete or type hints
3. **Hidden bugs** - type mismatches discovered late
4. **Unclear interfaces** - hard to know what functions expect/return

### Current Issues

#### 1. Missing Type Annotations

**websocket_client.py:24-26**
```python
def __init__(self):
    self.websocket = None  # Type unknown!
    self.connected = False
```

What type is `websocket`? `WebSocketClientProtocol`? Something else?

#### 2. Untyped Dictionary Parameters

**events.py:24**
```python
def handle_transcription_event(event: dict) -> None:
    # What keys does event have?
    # What are the value types?
    event_type = event["type"]  # Could KeyError
    transcript = event.get("transcript", "")  # Type unknown
```

**llm.py:82, 121**
```python
async def generate_reply(
    self,
    transcript: str,
    conversation: list[dict],  # What structure?
    audio_fallback: bool = False
) -> LLMReply | None:
```

What's in the `conversation` list? What keys/values in each dict?

#### 3. Implicit Optional Returns

**controller.py:53-64**
```python
def calculate_rms(audio_bytes: bytes) -> float:
    if not audio_bytes:
        return 0.0  # OK, returns float
    # ... calculation
```

Good! But compare to:

**llm.py:75**
```python
async def generate_reply(...) -> LLMReply | None:
    # Return type is clear, but implementation doesn't make it obvious
    # when None is returned
```

#### 4. No Protocol Definitions

```python
# Multiple places use websocket client
# But what interface does it need to implement?
# No Protocol definition exists
```

#### 5. Lack of Type Checking

No `mypy` or `pyright` configuration means:
- Type errors not caught in CI/CD
- Easy to introduce type bugs
- No enforcement of type consistency

## Proposed Solution

Implement **comprehensive type safety** with:

1. TypedDict for structured dictionaries
2. Protocol classes for interfaces
3. Complete type annotations
4. Type checking in CI/CD
5. Strict type checker configuration

### Implementation

#### 1. Event Type Definitions (`types/events.py`)

```python
"""Type definitions for WebSocket events."""

from typing import TypedDict, Literal, NotRequired

class BaseEvent(TypedDict):
    """Base event structure."""
    type: str
    event_id: str

class TranscriptionEvent(TypedDict):
    """Transcription completion event."""
    type: Literal["conversation.item.input_audio_transcription.completed"]
    event_id: str
    item_id: str
    transcript: str
    content_index: int

class AudioDeltaEvent(TypedDict):
    """Audio delta event."""
    type: Literal["response.audio.delta"]
    event_id: str
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str  # Base64 encoded audio

class AudioDoneEvent(TypedDict):
    """Audio completion event."""
    type: Literal["response.audio.done"]
    event_id: str
    response_id: str
    item_id: str
    output_index: int
    content_index: int

class TextDeltaEvent(TypedDict):
    """Text delta event."""
    type: Literal["response.text.delta"]
    event_id: str
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str

class ResponseDoneEvent(TypedDict):
    """Response completion event."""
    type: Literal["response.done"]
    event_id: str
    response_id: str

# Union of all event types
Event = (
    TranscriptionEvent |
    AudioDeltaEvent |
    AudioDoneEvent |
    TextDeltaEvent |
    ResponseDoneEvent
)
```

#### 2. Message Type Definitions (`types/messages.py`)

```python
"""Type definitions for LLM messages."""

from typing import TypedDict, Literal, NotRequired

class TextContent(TypedDict):
    """Text content block."""
    type: Literal["text"]
    text: str

class AudioContent(TypedDict):
    """Audio content block."""
    type: Literal["audio"]
    source: dict  # Could be further typed
    format: str

Content = TextContent | AudioContent

class Message(TypedDict):
    """LLM conversation message."""
    role: Literal["user", "assistant"]
    content: str | list[Content]

class MessageWithId(Message):
    """Message with unique identifier."""
    id: NotRequired[str]
    timestamp: NotRequired[int]
```

#### 3. WebSocket Protocol (`types/protocols.py`)

```python
"""Protocol definitions for dependency injection."""

from typing import Protocol, Any, AsyncIterator

class WebSocketProtocol(Protocol):
    """Protocol for WebSocket client."""

    async def connect(self) -> None:
        """Connect to WebSocket server."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        ...

    async def send(self, message: dict[str, Any]) -> None:
        """Send a message to the server."""
        ...

    async def receive(self) -> dict[str, Any]:
        """Receive a message from the server."""
        ...

    @property
    def connected(self) -> bool:
        """Check if WebSocket is connected."""
        ...

class AudioCaptureProtocol(Protocol):
    """Protocol for audio capture."""

    async def start(self) -> None:
        """Start audio capture."""
        ...

    async def stop(self) -> None:
        """Stop audio capture."""
        ...

    def stream(self) -> AsyncIterator[bytes]:
        """Stream audio chunks."""
        ...

class AudioPlaybackProtocol(Protocol):
    """Protocol for audio playback."""

    async def play(self, audio_data: bytes) -> None:
        """Play audio data."""
        ...

    async def stop(self) -> None:
        """Stop playback."""
        ...
```

#### 4. Configuration Types (`types/config.py`)

```python
"""Type definitions for configuration."""

from typing import TypedDict, NotRequired
from pathlib import Path

class AnthropicConfigDict(TypedDict):
    """Anthropic API configuration dictionary."""
    api_key: str
    model: str
    base_url: NotRequired[str]

class AudioConfigDict(TypedDict):
    """Audio configuration dictionary."""
    sample_rate: NotRequired[int]
    chunk_length_seconds: NotRequired[float]
    silence_threshold: NotRequired[float]
    silence_duration_threshold: NotRequired[float]
    input_device_index: NotRequired[int]
    output_device_index: NotRequired[int]

class WakeWordConfigDict(TypedDict):
    """Wake word configuration dictionary."""
    enabled: NotRequired[bool]
    engine: NotRequired[str]
    model_path: NotRequired[str]
    sensitivity: NotRequired[float]
    access_key: NotRequired[str]

class AppConfigDict(TypedDict):
    """Complete application configuration dictionary."""
    anthropic: AnthropicConfigDict
    audio: NotRequired[AudioConfigDict]
    wake_word: NotRequired[WakeWordConfigDict]
```

#### 5. Updated Event Handler with Types

**Before:**
```python
# events.py:24
def handle_transcription_event(event: dict) -> None:
    event_type = event["type"]
    transcript = event.get("transcript", "")
```

**After:**
```python
from pi_assistant.types.events import TranscriptionEvent, Event

def handle_transcription_event(event: TranscriptionEvent) -> None:
    """Handle transcription completion event."""
    event_type = event["type"]  # Type checker knows this exists!
    transcript = event["transcript"]  # Type checker knows this is str!
    item_id = event["item_id"]  # Autocomplete works!
```

#### 6. Updated WebSocket Client

**Before:**
```python
# websocket_client.py:24-26
def __init__(self):
    self.websocket = None
    self.connected = False
```

**After:**
```python
from websockets.client import WebSocketClientProtocol
from typing import Optional

def __init__(self):
    self.websocket: Optional[WebSocketClientProtocol] = None
    self.connected: bool = False
```

#### 7. Updated LLM Responder

**Before:**
```python
async def generate_reply(
    self,
    transcript: str,
    conversation: list[dict],
    audio_fallback: bool = False
) -> LLMReply | None:
```

**After:**
```python
from pi_assistant.types.messages import Message

async def generate_reply(
    self,
    transcript: str,
    conversation: list[Message],
    audio_fallback: bool = False
) -> LLMReply | None:
```

#### 8. Type Checker Configuration

**pyproject.toml**
```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_unimported = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
check_untyped_defs = true
strict_equality = true

[[tool.mypy.overrides]]
module = "sounddevice.*"
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "pvporcupine.*"
ignore_missing_imports = true

[tool.pyright]
include = ["src"]
exclude = ["**/node_modules", "**/__pycache__"]
typeCheckingMode = "strict"
reportMissingTypeStubs = false
pythonVersion = "3.11"
```

**Type checking in CI (.github/workflows/type-check.yml)**
```yaml
name: Type Check

on: [push, pull_request]

jobs:
  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run mypy
        run: mypy src
      - name: Run pyright
        run: pyright src
```

## Benefits

1. **Catch errors early**: Type errors caught before runtime
2. **Better IDE support**: Autocomplete, type hints, refactoring
3. **Self-documenting**: Types show what functions expect
4. **Safer refactoring**: Type checker validates changes
5. **Fewer bugs**: Many bugs are type errors in disguise

## Migration Plan

### Phase 1: Create Type Definitions (Day 1, Morning)
1. Create `src/pi_assistant/types/` package
2. Define `events.py` with all event types
3. Define `messages.py` with message types
4. Define `protocols.py` with interface protocols
5. Define `config.py` with configuration types

### Phase 2: Add Type Annotations (Day 1, Afternoon)
1. Update `websocket_client.py` with proper types
2. Update `events.py` to use `TypedDict`
3. Update `llm.py` to use message types
4. Update `controller.py` with complete annotations

### Phase 3: Configure Type Checkers (Day 2, Morning)
1. Add `mypy` configuration to `pyproject.toml`
2. Add `pyright` configuration
3. Set up CI/CD type checking
4. Fix initial type errors

### Phase 4: Gradual Typing (Day 2, Afternoon)
1. Add type: ignore comments where needed
2. Create typing TODO list for future work
3. Document typing standards in CONTRIBUTING.md

## Testing Strategy

```python
# No runtime tests needed for types!
# But we can validate TypedDict usage:

from pi_assistant.types.events import TranscriptionEvent

def test_transcription_event_structure():
    """Validate TranscriptionEvent structure."""
    event: TranscriptionEvent = {
        "type": "conversation.item.input_audio_transcription.completed",
        "event_id": "evt_123",
        "item_id": "item_456",
        "transcript": "Hello world",
        "content_index": 0
    }

    assert event["type"] == "conversation.item.input_audio_transcription.completed"
    assert event["transcript"] == "Hello world"

    # Type checker ensures all required fields present
```

## Example Type Errors Caught

### 1. Missing Dictionary Key
```python
# Before: Runtime KeyError
def process_event(event: dict):
    transcript = event["transcript"]  # Might not exist!

# After: Type error at development time
def process_event(event: TranscriptionEvent):
    transcript = event["transcript"]  # Type checker guarantees this exists
```

### 2. Wrong Parameter Type
```python
# Before: Runtime error deep in code
websocket.send("hello")  # Expects dict, not str

# After: Type error immediately
websocket.send("hello")  # Type checker: Expected dict, got str
```

### 3. Undefined Attribute
```python
# Before: Runtime AttributeError
client.webscoket.send(msg)  # Typo!

# After: Type error in IDE
client.webscoket.send(msg)  # Type checker: No attribute 'webscoket'
```

## IDE Benefits

### Before (No Types)
```python
def process(event):  # IDE doesn't know what 'event' is
    event.  # No autocomplete suggestions
```

### After (With Types)
```python
from pi_assistant.types.events import Event

def process(event: Event):
    event.  # IDE suggests: type, event_id, transcript, etc.
```

## Success Metrics

- [ ] All public functions have type annotations
- [ ] All event handlers use `TypedDict`
- [ ] WebSocket client uses `Protocol`
- [ ] `mypy` passes with strict config
- [ ] `pyright` passes in strict mode
- [ ] CI/CD runs type checking
- [ ] Zero `type: ignore` comments (or documented exceptions)

## Related Refactorings

- **Error Handling**: Exception types are well-typed
- **Configuration**: Config dataclasses are typed
- **LLM Responder**: Message/response types clearly defined
- **Testing**: Type-safe mocks and stubs

## Resources

- [PEP 589 - TypedDict](https://peps.python.org/pep-0589/)
- [PEP 544 - Protocols](https://peps.python.org/pep-0544/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [Pyright Documentation](https://github.com/microsoft/pyright)
