# Refactoring: Configuration Coupling Reduction

## Priority
**MEDIUM** - Estimated effort: 2-3 days

## Problem

The codebase has **tight coupling to configuration** through direct imports of config constants:

### Scope of the Problem

**23 modules** import configuration constants directly:

**controller.py:23-43** (20+ imports!)
```python
from pi_assistant.config import (
    ANTHROPIC_API_KEY,
    ASSISTANT_MODEL,
    CHUNK_LENGTH_SECONDS,
    ERROR_LOG_LABEL,
    INPUT_DEVICE_INDEX,
    LLM_SYSTEM_PROMPT,
    MAX_RETRIGGER_BUDGET,
    OUTPUT_DEVICE_INDEX,
    PORCUPINE_ACCESS_KEY,
    PORCUPINE_MODEL_PATH,
    SAMPLE_RATE,
    SILENCE_DURATION_THRESHOLD,
    SILENCE_THRESHOLD,
    USE_CONVERSATION_API,
    VERBOSE,
    VERBOSE_LOG_FILE,
    WAKE_WORD_ENABLED,
    # ... more constants
)
```

### Issues

1. **Hard to test**: Can't override config for tests
2. **Hidden dependencies**: No clear dependency chain
3. **Tight coupling**: Changes to config affect all modules
4. **No flexibility**: Can't run multiple instances with different configs
5. **Module-level imports**: Config loaded at import time

## Current Import Locations

```
controller.py    → 20+ config constants
app.py           → 15+ config constants
capture.py       → 5 config constants
playback.py      → 4 config constants
llm.py           → 3 config constants
wake_word.py     → 4 config constants
logging_utils.py → 3 config constants
... 16 more files
```

## Proposed Solution

Implement **Dependency Injection** pattern:

1. Components receive config through constructors
2. No direct config imports in most modules
3. Config passed down from entry point (app.py)
4. Easy to override config for testing

### Implementation

#### 1. Component Configuration Classes

Instead of importing individual constants, components receive config objects:

**Before:**
```python
# llm.py
from pi_assistant.config import (
    ANTHROPIC_API_KEY,
    ASSISTANT_MODEL,
    LLM_SYSTEM_PROMPT
)

class LLMResponder:
    def __init__(self):
        self.api_key = ANTHROPIC_API_KEY
        self.model = ASSISTANT_MODEL
        self.system_prompt = LLM_SYSTEM_PROMPT
```

**After:**
```python
# llm.py - No config imports!
from dataclasses import dataclass

@dataclass
class LLMConfig:
    """Configuration for LLM responder."""
    api_key: str
    model: str
    system_prompt: str

class LLMResponder:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = config.api_key
        self.model = config.model
        self.system_prompt = config.system_prompt
```

#### 2. Audio Configuration

**Before:**
```python
# capture.py
from pi_assistant.config import (
    SAMPLE_RATE,
    CHUNK_LENGTH_SECONDS,
    INPUT_DEVICE_INDEX
)

class AudioCapture:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.chunk_size = int(SAMPLE_RATE * CHUNK_LENGTH_SECONDS)
        self.device_index = INPUT_DEVICE_INDEX
```

**After:**
```python
# capture.py - No config imports!
from dataclasses import dataclass
from typing import Optional

@dataclass
class AudioCaptureConfig:
    """Configuration for audio capture."""
    sample_rate: int
    chunk_length_seconds: float
    device_index: Optional[int] = None

    @property
    def chunk_size(self) -> int:
        """Calculate chunk size in bytes."""
        return int(self.sample_rate * self.chunk_length_seconds)

class AudioCapture:
    def __init__(self, config: AudioCaptureConfig):
        self.config = config
        self.sample_rate = config.sample_rate
        self.chunk_size = config.chunk_size
        self.device_index = config.device_index
```

#### 3. Controller Configuration

**Before:**
```python
# controller.py - 20+ imports from config!
from pi_assistant.config import (
    ANTHROPIC_API_KEY,
    ASSISTANT_MODEL,
    CHUNK_LENGTH_SECONDS,
    # ... 17 more
)

async def run_audio_controller():
    # Uses global config constants
    if WAKE_WORD_ENABLED:
        # ...
```

**After:**
```python
# controller.py - Just one import!
from dataclasses import dataclass

@dataclass
class ControllerConfig:
    """Configuration for audio controller."""
    chunk_length_seconds: float
    silence_threshold: float
    silence_duration_threshold: float
    max_retrigger_budget: int
    wake_word_enabled: bool
    sample_rate: int

class AudioController:
    def __init__(
        self,
        config: ControllerConfig,
        websocket_client: WebSocketClient,
        audio_capture: AudioCapture,
        speech_player: SpeechPlayer,
        llm_responder: LLMResponder
    ):
        self.config = config
        self.websocket = websocket_client
        self.capture = audio_capture
        self.player = speech_player
        self.llm = llm_responder
```

#### 4. Entry Point with Dependency Injection

**app.py** - Only place that imports main config:

```python
"""Application entry point - constructs all dependencies."""

from pi_assistant.config import get_config
from pi_assistant.audio import AudioCapture, AudioCaptureConfig
from pi_assistant.audio import SpeechPlayer, SpeechPlayerConfig
from pi_assistant.assistant import LLMResponder, LLMConfig
from pi_assistant.network import WebSocketClient, WebSocketConfig
from pi_assistant.cli import AudioController, ControllerConfig
from pi_assistant.wake_word import WakeWordEngine, WakeWordConfig

async def main():
    # 1. Load configuration (only done here!)
    app_config = get_config()

    # 2. Create component configs
    llm_config = LLMConfig(
        api_key=app_config.anthropic.api_key,
        model=app_config.anthropic.model,
        system_prompt=app_config.llm.system_prompt
    )

    audio_capture_config = AudioCaptureConfig(
        sample_rate=app_config.audio.sample_rate,
        chunk_length_seconds=app_config.audio.chunk_length_seconds,
        device_index=app_config.audio.input_device_index
    )

    speech_player_config = SpeechPlayerConfig(
        sample_rate=app_config.audio.sample_rate,
        device_index=app_config.audio.output_device_index
    )

    websocket_config = WebSocketConfig(
        url=app_config.websocket.url,
        api_key=app_config.anthropic.api_key
    )

    controller_config = ControllerConfig(
        chunk_length_seconds=app_config.audio.chunk_length_seconds,
        silence_threshold=app_config.audio.silence_threshold,
        silence_duration_threshold=app_config.audio.silence_duration_threshold,
        max_retrigger_budget=app_config.controller.max_retrigger_budget,
        wake_word_enabled=app_config.wake_word.enabled,
        sample_rate=app_config.audio.sample_rate
    )

    # 3. Construct dependencies (dependency injection!)
    llm_responder = LLMResponder(llm_config)
    audio_capture = AudioCapture(audio_capture_config)
    speech_player = SpeechPlayer(speech_player_config)
    websocket_client = WebSocketClient(websocket_config)

    # 4. Create controller with all dependencies
    controller = AudioController(
        config=controller_config,
        websocket_client=websocket_client,
        audio_capture=audio_capture,
        speech_player=speech_player,
        llm_responder=llm_responder
    )

    # 5. Run application
    await controller.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Benefits

### Testability

**Before:** Can't test with different config
```python
# test_controller.py - Can't override global config!
def test_controller():
    controller = run_audio_controller()  # Uses global config
```

**After:** Easy to test with custom config
```python
# test_controller.py - Easy to inject test config!
def test_controller():
    test_config = ControllerConfig(
        chunk_length_seconds=0.05,  # Smaller for faster tests
        silence_threshold=0.01,
        silence_duration_threshold=0.1,
        max_retrigger_budget=5,
        wake_word_enabled=False,  # Disable for unit tests
        sample_rate=16000
    )

    mock_websocket = MockWebSocketClient()
    mock_capture = MockAudioCapture()
    # ... more mocks

    controller = AudioController(
        config=test_config,
        websocket_client=mock_websocket,
        audio_capture=mock_capture,
        speech_player=mock_player,
        llm_responder=mock_llm
    )

    # Now we can test controller in isolation!
```

### Multiple Instances

**Before:** Can't run multiple instances
```python
# Only one instance possible with global config
controller1 = run_audio_controller()  # Uses global config
controller2 = run_audio_controller()  # Same config! Can't differ!
```

**After:** Multiple instances with different configs
```python
# Different configs for different use cases
production_config = ControllerConfig(...)
test_config = ControllerConfig(...)

prod_controller = AudioController(production_config, ...)
test_controller = AudioController(test_config, ...)
```

### Clear Dependencies

**Before:** Hidden dependencies
```python
class LLMResponder:
    def __init__(self):
        # What does this depend on? You have to read the imports!
        pass
```

**After:** Explicit dependencies
```python
class LLMResponder:
    def __init__(self, config: LLMConfig):
        # Clear! Depends on LLMConfig
        pass
```

## Migration Plan

### Phase 1: Create Config Classes (Day 1)
1. Create `LLMConfig` in `assistant/llm.py`
2. Create `AudioCaptureConfig` in `audio/capture.py`
3. Create `SpeechPlayerConfig` in `audio/playback.py`
4. Create `WebSocketConfig` in `network/websocket_client.py`
5. Create `ControllerConfig` in `cli/controller.py`
6. Create `WakeWordConfig` in `wake_word/engine.py`

### Phase 2: Update Component Constructors (Day 2)
1. Update `LLMResponder.__init__` to accept `LLMConfig`
2. Update `AudioCapture.__init__` to accept `AudioCaptureConfig`
3. Update `SpeechPlayer.__init__` to accept `SpeechPlayerConfig`
4. Update `WebSocketClient.__init__` to accept `WebSocketConfig`
5. Update `AudioController.__init__` to accept all dependencies
6. Keep backward compatibility with default config

### Phase 3: Update Entry Point (Day 2-3)
1. Refactor `app.py` to use dependency injection
2. Create component configs from main config
3. Pass configs to component constructors
4. Remove direct config imports from components

### Phase 4: Update Tests (Day 3)
1. Create test fixtures for component configs
2. Update tests to use dependency injection
3. Add tests for different config scenarios
4. Verify all tests pass

## Testing Strategy

```python
import pytest
from pi_assistant.assistant.llm import LLMResponder, LLMConfig

@pytest.fixture
def llm_config():
    """Fixture for LLM configuration."""
    return LLMConfig(
        api_key="test-key",
        model="claude-3-5-sonnet-20241022",
        system_prompt="Test prompt"
    )

def test_llm_responder_with_custom_config(llm_config):
    """Test LLM responder with custom config."""
    responder = LLMResponder(llm_config)

    assert responder.api_key == "test-key"
    assert responder.model == "claude-3-5-sonnet-20241022"

def test_llm_responder_with_different_models():
    """Test LLM responder works with different models."""
    configs = [
        LLMConfig(api_key="key1", model="model1", system_prompt="prompt1"),
        LLMConfig(api_key="key2", model="model2", system_prompt="prompt2"),
    ]

    responders = [LLMResponder(config) for config in configs]

    assert responders[0].model == "model1"
    assert responders[1].model == "model2"
```

## Configuration Organization

```
src/pi_assistant/
├── config/
│   ├── __init__.py        # Main app config (AppConfig)
│   └── ...
├── audio/
│   ├── capture.py         # AudioCaptureConfig
│   └── playback.py        # SpeechPlayerConfig
├── assistant/
│   └── llm.py             # LLMConfig
├── network/
│   └── websocket_client.py # WebSocketConfig
├── cli/
│   ├── controller.py      # ControllerConfig
│   └── app.py             # Dependency injection wiring
└── wake_word/
    └── engine.py          # WakeWordConfig
```

## Backward Compatibility

During migration, support both patterns:

```python
class AudioCapture:
    def __init__(self, config: Optional[AudioCaptureConfig] = None):
        if config is None:
            # Backward compatibility - use global config
            from pi_assistant.config import SAMPLE_RATE, CHUNK_LENGTH_SECONDS
            config = AudioCaptureConfig(
                sample_rate=SAMPLE_RATE,
                chunk_length_seconds=CHUNK_LENGTH_SECONDS
            )

        self.config = config
```

After migration complete, remove backward compatibility.

## Success Metrics

- [ ] All components accept config through constructors
- [ ] No direct config imports in component modules
- [ ] `app.py` is only place that imports main config
- [ ] Tests can inject custom configs
- [ ] Can create multiple instances with different configs
- [ ] Configuration dependencies are explicit

## Related Refactorings

- **Configuration Module Refactor**: Provides `get_config()` API
- **Testing**: Makes all components testable with mock configs
- **Type Safety**: Config classes are strongly typed
- **Audio Controller State Machine**: States can receive config objects
