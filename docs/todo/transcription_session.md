# Refactor: Transcription Session Management

## Priority: CRITICAL

## Location
`src/pi_assistant/cli/app.py:113-265`

## Problem Summary
The `run_transcription` function is a 152-line "god function" that handles too many responsibilities. It has lint suppressions (`noqa: PLR0912, PLR0915`) indicating excessive branches and statements. The function mixes initialization, configuration, connection management, task orchestration, and cleanup in a way that makes it difficult to test and maintain.

## Specific Issues

### 1. Multiple Responsibilities
The function handles:
- Argument validation and preprocessing
- Configuration object creation
- Component initialization (audio capture, playback, wake word detector, etc.)
- WebSocket connection management
- Task creation and orchestration
- Event stream coordination
- Resource cleanup
- Error handling and propagation

### 2. Complex Object Construction (Lines 126-174)
Object creation logic is mixed with validation:
```python
# Configuration and validation intermixed
if not config.api_key:
    _prompt_for_api_key()  # Side effect during initialization

# Object creation scattered throughout
audio_capture = AudioCapture(...)
audio_playback = AudioPlayback(...)
wake_word_detector = WakeWordDetector(...)
# ... many more objects
```

### 3. Manual Task Orchestration (Lines 201-245)
Complex task dependency management that is error-prone:
```python
async with asyncio.TaskGroup() as task_group:
    task_group.create_task(capture_task())
    task_group.create_task(controller_task())
    if wake_word_detector:
        task_group.create_task(wake_word_task())
    # ... more conditional task creation
```

### 4. Cleanup Logic Issues (Lines 254-264)
Resource cleanup is manual and potentially incomplete:
- No guaranteed cleanup on error paths
- Resource lifetimes not clearly scoped
- Difficult to ensure proper shutdown order

### 5. Testing Challenges
- Cannot test component initialization separately
- Cannot test task orchestration without full system
- Cannot test cleanup logic in isolation
- Mock requirements are extensive
- Integration tests are slow and brittle

## Proposed Refactoring Strategy

### Phase 1: Create Configuration Validator
Extract validation logic into dedicated class:

```python
class TranscriptionConfigValidator:
    """Validates and prepares configuration for transcription session."""

    @staticmethod
    def validate(config: Config) -> None:
        """
        Validate configuration and prompt for missing required values.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        if not config.api_key:
            raise ConfigurationError("API key is required")

        if config.audio.input_device_id is not None:
            TranscriptionConfigValidator._validate_input_device(
                config.audio.input_device_id
            )

        # ... other validation

    @staticmethod
    def _validate_input_device(device_id: int) -> None:
        """Validate input device exists and is accessible."""
        # Validation logic
        ...
```

### Phase 2: Create Session Component Builder
Extract object creation into builder pattern:

```python
@dataclass
class TranscriptionComponents:
    """Container for all transcription session components."""
    audio_capture: AudioCapture
    audio_playback: AudioPlayback
    audio_queue: AudioQueue
    websocket_client: WebSocketClient
    wake_word_detector: WakeWordDetector | None
    llm_responder: LLMResponder | None
    transcript_collector: TranscriptCollector | None


class TranscriptionComponentBuilder:
    """Builds components needed for transcription session."""

    def __init__(self, config: Config):
        self.config = config

    def build_audio_capture(self) -> AudioCapture:
        """Create and configure audio capture component."""
        return AudioCapture(
            device_id=self.config.audio.input_device_id,
            sample_rate=self.config.audio.input_sample_rate,
            # ...
        )

    def build_audio_playback(self) -> AudioPlayback:
        """Create and configure audio playback component."""
        # ...

    def build_all(self) -> TranscriptionComponents:
        """
        Build all components needed for transcription session.

        Returns:
            TranscriptionComponents with all initialized components.
        """
        return TranscriptionComponents(
            audio_capture=self.build_audio_capture(),
            audio_playback=self.build_audio_playback(),
            audio_queue=self.build_audio_queue(),
            websocket_client=self.build_websocket_client(),
            wake_word_detector=self.build_wake_word_detector(),
            llm_responder=self.build_llm_responder(),
            transcript_collector=self.build_transcript_collector(),
        )
```

### Phase 3: Create Session Context Manager
Implement proper resource management with context manager:

```python
class TranscriptionSession:
    """
    Manages a complete transcription session with proper resource lifecycle.

    Usage:
        async with TranscriptionSession(config) as session:
            await session.run()
    """

    def __init__(self, config: Config):
        self.config = config
        self.components: TranscriptionComponents | None = None
        self._connection_established = False

    async def __aenter__(self) -> "TranscriptionSession":
        """Initialize session and establish connections."""
        # Validate configuration
        TranscriptionConfigValidator.validate(self.config)

        # Build components
        builder = TranscriptionComponentBuilder(self.config)
        self.components = builder.build_all()

        # Establish connection
        await self._establish_connection()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up resources in proper order."""
        if self._connection_established:
            await self._close_connection()

        if self.components:
            # Stop components in reverse dependency order
            if self.components.audio_capture:
                self.components.audio_capture.stop()

            if self.components.audio_playback:
                self.components.audio_playback.stop()

        # No exception suppression
        return False

    async def _establish_connection(self) -> None:
        """Establish WebSocket connection to server."""
        await self.components.websocket_client.connect()
        self._connection_established = True

    async def _close_connection(self) -> None:
        """Close WebSocket connection gracefully."""
        await self.components.websocket_client.close()
        self._connection_established = False

    async def run(self) -> None:
        """Run the transcription session until completion or interruption."""
        coordinator = TaskCoordinator(self.components, self.config)
        await coordinator.run()
```

### Phase 4: Create Task Coordinator
Extract task orchestration into dedicated class:

```python
class TaskCoordinator:
    """Coordinates concurrent tasks for transcription session."""

    def __init__(
        self,
        components: TranscriptionComponents,
        config: Config
    ):
        self.components = components
        self.config = config

    async def run(self) -> None:
        """
        Start and coordinate all session tasks.

        Tasks are started in appropriate order with proper dependency handling.
        """
        async with asyncio.TaskGroup() as task_group:
            # Create core tasks
            task_group.create_task(
                self._run_audio_capture(),
                name="audio_capture"
            )

            task_group.create_task(
                self._run_controller(),
                name="controller"
            )

            # Create optional tasks
            if self.components.wake_word_detector:
                task_group.create_task(
                    self._run_wake_word_detector(),
                    name="wake_word"
                )

            if self.components.llm_responder:
                task_group.create_task(
                    self._run_llm_responder(),
                    name="llm_responder"
                )

            # TaskGroup handles cancellation and error propagation

    async def _run_audio_capture(self) -> None:
        """Run audio capture task."""
        await run_audio_capture(
            self.components.audio_capture,
            self.components.audio_queue,
            # ...
        )

    async def _run_controller(self) -> None:
        """Run controller task."""
        await run_audio_controller(
            # ...
        )

    # ... other task runner methods
```

### Phase 5: Simplified Main Function
The `run_transcription` function becomes a simple coordinator:

```python
async def run_transcription(
    args: argparse.Namespace,
    config: Config
) -> None:
    """
    Run transcription session with provided configuration.

    Args:
        args: Parsed command-line arguments
        config: Application configuration

    Raises:
        ConfigurationError: If configuration is invalid
        ConnectionError: If unable to connect to server
        TranscriptionError: If session fails
    """
    # Apply argument overrides to config
    apply_args_to_config(config, args)

    # Run session with automatic resource management
    async with TranscriptionSession(config) as session:
        await session.run()
```

## Benefits of Refactoring

### Improved Testability
- Each component can be unit tested independently
- Configuration validation can be tested without I/O
- Task coordination can be tested with mock components
- Resource cleanup can be verified in isolation

### Better Maintainability
- Clear separation of concerns
- Each class has single, well-defined responsibility
- Easier to understand initialization flow
- Modifications are localized

### Enhanced Reliability
- Guaranteed resource cleanup via context managers
- Proper exception handling and propagation
- Clear error boundaries
- Easier to reason about resource lifetimes

### Cleaner Code
- No more 150-line functions
- No lint suppressions needed
- Self-documenting structure
- Clear dependency relationships

## Migration Strategy

1. **Create new classes alongside existing code**
   - Implement `TranscriptionConfigValidator`
   - Implement `TranscriptionComponentBuilder`
   - Implement `TranscriptionSession`
   - Implement `TaskCoordinator`

2. **Add comprehensive tests for new code**
   - Unit tests for validator
   - Unit tests for builder
   - Integration tests for session
   - Mock tests for coordinator

3. **Create adapter function**
   - New `run_transcription_v2` that uses new architecture
   - Run both in parallel during migration

4. **Switch over when confident**
   - Replace old implementation
   - Remove deprecated code
   - Remove lint suppressions

5. **Monitor for regressions**
   - Ensure all integration tests pass
   - Verify no behavior changes
   - Check for resource leaks

## Migration Risks & Mitigation

### Risk: Breaking existing functionality
**Mitigation**: Run comprehensive integration tests, use feature flag to toggle implementations

### Risk: Performance degradation
**Mitigation**: Benchmark before/after, profile session initialization

### Risk: Incomplete resource cleanup
**Mitigation**: Add explicit resource lifecycle tests, use context manager protocol

## Estimated Effort
- **Time**: 2-3 days for complete refactoring
- **Risk**: Medium (core functionality, but clear boundaries)
- **Impact**: High (affects all transcription sessions)

## Success Metrics
- [ ] `run_transcription` function reduced to < 30 lines
- [ ] All components have dedicated builder methods
- [ ] Resource cleanup handled by context manager
- [ ] No lint suppressions needed
- [ ] Unit test coverage > 90% for new classes
- [ ] All existing integration tests pass
- [ ] No resource leaks in long-running sessions
- [ ] Initialization time remains within 5% of original

## Dependencies
- Should be done after or in parallel with `audio_controller` refactor
- May benefit from `llm_configuration` refactor being done first

## References
- Python context manager protocol: https://docs.python.org/3/reference/datamodel.html#context-managers
- Builder pattern: https://refactoring.guru/design-patterns/builder
- asyncio TaskGroup: https://docs.python.org/3/library/asyncio-task.html#asyncio.TaskGroup
