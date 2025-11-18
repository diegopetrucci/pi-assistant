# Refactoring: Audio Controller State Machine

## Priority
**HIGH** - Estimated effort: 3-4 days

## Problem

The `run_audio_controller` function in `src/pi_assistant/cli/controller.py:155-424` is a monolithic 270-line function that:

- Manages 13+ state variables manually
- Contains 3 nested function definitions
- Has deeply nested conditional logic
- Exhibits high cyclomatic complexity
- Is difficult to test individual behaviors
- Makes debugging state transitions challenging

### Current Issues

**Complex State Management**
```python
# Lines 247-271, 283-323: Manual state tracking
is_streaming = False
chunk_count = 0
silence_duration = 0.0
heard_speech = False
retrigger_budget = MAX_RETRIGGER_BUDGET
# ... and 8 more state variables
```

**Nested Function Definitions**
```python
def _prepare_for_stream(): ...
def _schedule_response_task(): ...
def _transition_stream_to_listening(): ...
```

**Scattered State Transition Logic**
- Lines 247-271: Wake word detection transitions
- Lines 283-323: Streaming state transitions
- Logic duplicated across multiple conditional blocks

## Proposed Solution

Implement the **State Pattern** to formalize state transitions and encapsulate state-specific behavior.

### Architecture

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class StateTransition:
    """Result of handling an event in a state."""
    new_state: Optional['AudioControllerState']
    action: Optional[callable] = None

class AudioControllerState(ABC):
    """Base class for audio controller states."""

    @abstractmethod
    async def handle_audio_chunk(
        self,
        chunk: bytes,
        context: 'AudioControllerContext'
    ) -> StateTransition:
        """Process an audio chunk in this state."""
        pass

    @abstractmethod
    async def handle_wake_word(
        self,
        context: 'AudioControllerContext'
    ) -> StateTransition:
        """Handle wake word detection in this state."""
        pass

    @abstractmethod
    async def on_enter(self, context: 'AudioControllerContext') -> None:
        """Called when entering this state."""
        pass

    @abstractmethod
    async def on_exit(self, context: 'AudioControllerContext') -> None:
        """Called when exiting this state."""
        pass

class ListeningState(AudioControllerState):
    """State when listening for wake word."""

    async def handle_audio_chunk(
        self,
        chunk: bytes,
        context: 'AudioControllerContext'
    ) -> StateTransition:
        # Detect wake word
        if await context.wake_word_engine.detect(chunk):
            return StateTransition(
                new_state=StreamingState(),
                action=lambda: context.start_streaming()
            )
        return StateTransition(new_state=None)

    async def handle_wake_word(
        self,
        context: 'AudioControllerContext'
    ) -> StateTransition:
        return StateTransition(new_state=StreamingState())

    async def on_enter(self, context: 'AudioControllerContext') -> None:
        context.reset_wake_engine()
        context.clear_audio_buffers()

    async def on_exit(self, context: 'AudioControllerContext') -> None:
        pass

class StreamingState(AudioControllerState):
    """State when streaming audio to LLM."""

    def __init__(self):
        self.chunk_count = 0
        self.silence_duration = 0.0
        self.heard_speech = False

    async def handle_audio_chunk(
        self,
        chunk: bytes,
        context: 'AudioControllerContext'
    ) -> StateTransition:
        self.chunk_count += 1

        # Check for silence
        if self._is_silence(chunk, context):
            self.silence_duration += context.chunk_duration

            if self._should_end_stream(context):
                return StateTransition(
                    new_state=ListeningState(),
                    action=lambda: context.commit_audio()
                )
        else:
            self.silence_duration = 0.0
            self.heard_speech = True

        # Send chunk to LLM
        await context.websocket.send_audio(chunk)
        return StateTransition(new_state=None)

    async def handle_wake_word(
        self,
        context: 'AudioControllerContext'
    ) -> StateTransition:
        # Handle re-trigger within conversation
        if context.retrigger_budget > 0:
            context.retrigger_budget -= 1
            return StateTransition(
                new_state=StreamingState(),
                action=lambda: context.restart_streaming()
            )
        return StateTransition(new_state=None)

    def _is_silence(self, chunk: bytes, context: 'AudioControllerContext') -> bool:
        rms = calculate_rms(chunk)
        return rms < context.config.silence_threshold

    def _should_end_stream(self, context: 'AudioControllerContext') -> bool:
        return (
            self.heard_speech and
            self.silence_duration >= context.config.silence_duration_threshold
        )

    async def on_enter(self, context: 'AudioControllerContext') -> None:
        await context.websocket.create_response()
        self.chunk_count = 0
        self.silence_duration = 0.0
        self.heard_speech = False

    async def on_exit(self, context: 'AudioControllerContext') -> None:
        pass

@dataclass
class AudioControllerContext:
    """Shared context for audio controller states."""
    websocket: WebSocketClient
    wake_word_engine: WakeWordEngine
    speech_player: SpeechPlayer
    config: AudioConfig
    retrigger_budget: int

    def reset_wake_engine(self) -> None:
        """Reset wake word detection engine."""
        self.wake_word_engine.reset()

    def clear_audio_buffers(self) -> None:
        """Clear any buffered audio."""
        # Implementation
        pass

    async def start_streaming(self) -> None:
        """Prepare to start streaming audio."""
        # Implementation
        pass

    async def commit_audio(self) -> None:
        """Commit audio input to server."""
        await self.websocket.commit_audio_buffer()

    async def restart_streaming(self) -> None:
        """Restart streaming (re-trigger case)."""
        # Implementation
        pass

    @property
    def chunk_duration(self) -> float:
        """Duration of one audio chunk in seconds."""
        return self.config.chunk_size / self.config.sample_rate

class AudioController:
    """Main audio controller using state pattern."""

    def __init__(
        self,
        websocket: WebSocketClient,
        wake_word_engine: WakeWordEngine,
        speech_player: SpeechPlayer,
        config: AudioConfig
    ):
        self.context = AudioControllerContext(
            websocket=websocket,
            wake_word_engine=wake_word_engine,
            speech_player=speech_player,
            config=config,
            retrigger_budget=config.max_retrigger_budget
        )
        self.state: AudioControllerState = ListeningState()

    async def run(self, audio_stream: AsyncIterator[bytes]) -> None:
        """Main run loop."""
        await self.state.on_enter(self.context)

        async for chunk in audio_stream:
            transition = await self.state.handle_audio_chunk(chunk, self.context)
            await self._apply_transition(transition)

    async def handle_wake_word_detection(self) -> None:
        """Handle external wake word detection."""
        transition = await self.state.handle_wake_word(self.context)
        await self._apply_transition(transition)

    async def _apply_transition(self, transition: StateTransition) -> None:
        """Apply a state transition."""
        if transition.new_state is not None:
            await self.state.on_exit(self.context)
            self.state = transition.new_state
            await self.state.on_enter(self.context)

        if transition.action is not None:
            await transition.action()
```

## Benefits

1. **Testability**: Each state can be tested independently
2. **Clarity**: State-specific logic is encapsulated in state classes
3. **Extensibility**: Easy to add new states (e.g., PausedState, ErrorState)
4. **Debugging**: State transitions are explicit and traceable
5. **Reduced complexity**: Main controller becomes simple orchestrator

## Migration Plan

### Phase 1: Extract State Classes (Day 1-2)
1. Create `src/pi_assistant/cli/states.py`
2. Implement base `AudioControllerState` class
3. Extract `ListeningState` logic from lines 155-246
4. Extract `StreamingState` logic from lines 247-424

### Phase 2: Create Context Object (Day 2)
1. Create `AudioControllerContext` dataclass
2. Move shared state and dependencies into context
3. Extract helper methods from nested functions

### Phase 3: Refactor Controller (Day 3)
1. Create `AudioController` class
2. Replace `run_audio_controller` function with `AudioController.run`
3. Update call sites in `app.py`

### Phase 4: Testing (Day 3-4)
1. Add unit tests for each state class
2. Add integration tests for state transitions
3. Test edge cases (re-trigger, silence detection, etc.)

## Testing Strategy

```python
import pytest
from pi_assistant.cli.states import ListeningState, StreamingState, AudioControllerContext

@pytest.mark.asyncio
async def test_listening_state_wake_word_detection():
    """Test transition from listening to streaming on wake word."""
    state = ListeningState()
    context = create_mock_context()

    # Simulate wake word detection
    transition = await state.handle_wake_word(context)

    assert isinstance(transition.new_state, StreamingState)
    assert transition.action is not None

@pytest.mark.asyncio
async def test_streaming_state_silence_detection():
    """Test transition from streaming to listening on silence."""
    state = StreamingState()
    context = create_mock_context()

    # Simulate speech followed by silence
    await state.handle_audio_chunk(loud_audio, context)

    for _ in range(10):  # Simulate silence duration
        transition = await state.handle_audio_chunk(silent_audio, context)

    assert isinstance(transition.new_state, ListeningState)
```

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing behavior | Extensive integration tests before refactoring |
| Performance overhead from state objects | Profile before/after, optimize hot paths |
| Complexity in state transitions | Clear state diagram documentation |
| Incomplete state extraction | Gradual migration, keep old code until validated |

## Success Metrics

- [ ] `run_audio_controller` reduced to < 50 lines
- [ ] Each state class < 100 lines
- [ ] 90%+ test coverage on state classes
- [ ] No regression in audio processing performance
- [ ] State transitions fully documented

## Related Refactorings

- **Error Handling**: States can have consistent error handling
- **Event System**: States can emit events for monitoring
- **Configuration**: States can validate required config at construction
