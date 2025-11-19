# Refactor: Audio Controller

## Priority: CRITICAL

## Location
`src/pi_assistant/cli/controller.py:155-417`

## Problem Summary
The `run_audio_controller` function is a 262-line monolithic function that violates multiple SOLID principles and has excessive complexity. It currently has lint suppressions (`noqa: PLR0913, PLR0912, PLR0915`) indicating too many arguments (8), branches, and statements (exceeds the max-statements=50 lint rule).

## Specific Issues

### 1. Single Responsibility Principle Violations
The function handles multiple distinct responsibilities:
- Wake word detection
- Audio chunk processing
- State management and transitions
- Transcript handling
- Silence detection and auto-stop logic
- Server communication
- Event coordination

### 2. Excessive State Management
13+ local variables tracking different aspects of state:
- `state` - current stream state
- `chunk_count` - audio chunks processed
- `silence_duration` - accumulated silence time
- `heard_speech` - speech detection flag
- `retrigger_budget` - wake word retrigger count
- `awaiting_server_stop_event` - server stop coordination
- `pending_finalize_reason` - deferred finalization state
- `suppress_next_server_stop_event` - event suppression flag
- And more...

### 3. Complex Nested Logic
- 4 internal helper functions that modify non-local state
- Complex conditional logic for state transitions (lines 240-316)
- Auto-stop logic embedded in main event loop (lines 373-401)
- Difficult to reason about execution flow

### 4. Testing Challenges
- Cannot test individual components in isolation
- State transitions are implicit and scattered
- Mock requirements are extensive due to tight coupling
- No clear interfaces for testing state machine behavior

## Proposed Refactoring Strategy

### Phase 1: Extract State Management
Create a `StreamStateManager` class to encapsulate state transitions:

```python
class StreamStateManager:
    """Manages audio streaming state transitions and validation."""

    def __init__(self):
        self.state = StreamState.LISTENING
        self.chunk_count = 0
        self.silence_duration = 0.0
        self.heard_speech = False
        self.retrigger_budget = 0
        # ... other state variables

    def transition_to_streaming(self) -> None:
        """Transition from LISTENING to STREAMING."""
        if self.state != StreamState.LISTENING:
            raise InvalidStateTransitionError(...)
        self.state = StreamState.STREAMING
        self._reset_streaming_state()

    def transition_to_listening(self) -> None:
        """Transition from STREAMING to LISTENING."""
        # validation and state reset
        ...

    def increment_chunk_count(self) -> int:
        """Increment and return chunk count."""
        self.chunk_count += 1
        return self.chunk_count
```

### Phase 2: Extract Audio Processing
Create an `AudioProcessor` class for chunk handling:

```python
class AudioProcessor:
    """Processes audio chunks and detects speech/silence."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.silence_detector = SilenceDetector(config)

    def process_chunk(
        self,
        chunk: bytes,
        heard_speech: bool
    ) -> AudioChunkResult:
        """
        Process an audio chunk and return analysis results.

        Returns:
            AudioChunkResult with silence duration, speech detected, etc.
        """
        ...
```

### Phase 3: Extract Silence Detection
Create a `SilenceDetector` class for auto-stop logic:

```python
class SilenceDetector:
    """Detects silence periods and determines auto-stop conditions."""

    def __init__(self, auto_stop_seconds: float, chunk_duration: float):
        self.auto_stop_threshold = auto_stop_seconds
        self.chunk_duration = chunk_duration

    def update(self, chunk: bytes, heard_speech: bool) -> SilenceUpdate:
        """
        Update silence tracking with new chunk.

        Returns:
            SilenceUpdate indicating if auto-stop threshold reached.
        """
        ...

    def should_auto_stop(self) -> bool:
        """Check if auto-stop threshold has been reached."""
        ...
```

### Phase 4: Extract Wake Word Handler
Create dedicated wake word handling:

```python
class WakeWordHandler:
    """Handles wake word detection and retrigger logic."""

    def __init__(self, allow_retrigger: bool):
        self.allow_retrigger = allow_retrigger
        self.retrigger_budget = 0

    def handle_wake_word(self, current_state: StreamState) -> WakeWordAction:
        """
        Process wake word event and determine appropriate action.

        Returns:
            WakeWordAction indicating what should happen.
        """
        ...
```

### Phase 5: Reassemble with Coordination
The main controller function becomes a coordination layer:

```python
async def run_audio_controller(
    config: AudioControllerConfig,
    audio_queue: AudioQueue,
    websocket_client: WebSocketClient,
    # ... other dependencies
) -> None:
    """
    Coordinate audio processing, state management, and event handling.

    This function now acts as a lightweight coordinator between
    specialized components rather than implementing all logic directly.
    """
    state_manager = StreamStateManager()
    audio_processor = AudioProcessor(config.audio_config)
    silence_detector = SilenceDetector(
        config.auto_stop_seconds,
        config.chunk_duration
    )
    wake_word_handler = WakeWordHandler(config.allow_retrigger)

    async for event in event_stream:
        if isinstance(event, AudioChunkEvent):
            result = audio_processor.process_chunk(
                event.chunk,
                state_manager.heard_speech
            )

            if state_manager.state == StreamState.STREAMING:
                silence_update = silence_detector.update(
                    event.chunk,
                    result.speech_detected
                )

                if silence_update.should_stop:
                    await handle_auto_stop(...)

        elif isinstance(event, WakeWordEvent):
            action = wake_word_handler.handle_wake_word(
                state_manager.state
            )

            if action.should_start_streaming:
                state_manager.transition_to_streaming()
                ...
```

## Benefits of Refactoring

### Improved Testability
- Each component can be unit tested independently
- State transitions can be tested in isolation
- Mock requirements reduced significantly
- Clear interfaces for testing

### Better Maintainability
- Each class has a single, clear responsibility
- State management is explicit and centralized
- Easier to understand control flow
- Modifications are localized to specific components

### Enhanced Extensibility
- Easy to add new audio processing features
- Simple to modify state transition rules
- Can swap implementations (e.g., different silence detection algorithms)
- Clear extension points for new functionality

### Reduced Complexity
- Each function/method is shorter and focused
- Fewer branches and nested conditionals per unit
- Lint suppressions can be removed
- Cognitive load significantly reduced

## Migration Strategy

1. **Write comprehensive tests for existing behavior** - Capture current functionality before refactoring
2. **Extract classes one at a time** - Start with `StreamStateManager`, then others
3. **Run tests after each extraction** - Ensure behavior remains unchanged
4. **Gradually migrate controller logic** - Move responsibility to new classes incrementally
5. **Remove old code once migrated** - Clean up redundant logic
6. **Remove lint suppressions** - Verify complexity is now acceptable

## Estimated Effort
- **Time**: 2-3 days for complete refactoring
- **Risk**: Medium (complex logic, but well-tested functionality)
- **Impact**: High (affects core audio streaming behavior)

## Success Metrics
- [ ] `run_audio_controller` function reduced to < 100 lines
- [ ] No lint suppressions needed for complexity
- [ ] All state transitions tested independently
- [ ] Unit test coverage > 85% for new classes
- [ ] All existing integration tests pass
- [ ] No regression in audio streaming behavior

## References
- Original issue: Lint rule violation (max-statements=50)
- Related: State machine pattern documentation
- Similar refactor: Consider how other event-driven systems handle state
