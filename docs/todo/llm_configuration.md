# Refactor: LLM Configuration and Initialization

## Priority: HIGH

## Location
`src/pi_assistant/assistant/llm.py:47-92`

## Problem Summary
The `LLMResponder.__init__` method has 13 parameters, which violates the parameter count guidelines and has a lint suppression (`noqa: PLR0913`). This makes the class difficult to construct, maintain, and extend. Many parameters have default values derived from configuration, suggesting a configuration object pattern would be more appropriate.

## Specific Issues

### 1. Parameter Explosion
Current constructor signature:
```python
def __init__(
    self,
    client: anthropic.AsyncAnthropic,
    model: str,
    max_tokens: int,
    console_print: ConsolePrint,
    thinking_budget: int,
    location: str | None,
    system_prompt: str | None,
    tts_client: openai.AsyncOpenAI | None,
    tts_voice: str | None,
    tts_speed: float,
    tts_model: str,
    playback: AudioPlayback | None,
    verbose: bool,
) -> None:  # noqa: PLR0913
```

13 parameters with mixed concerns:
- Core dependencies (client, playback)
- Model configuration (model, max_tokens, thinking_budget)
- TTS configuration (tts_client, tts_voice, tts_speed, tts_model)
- Context configuration (location, system_prompt)
- Utilities (console_print, verbose)

### 2. Construction Complexity
Creating an instance requires gathering 13 separate values:
```python
# Current usage (hypothetical)
llm_responder = LLMResponder(
    client=anthropic_client,
    model="claude-3-5-sonnet-20241022",
    max_tokens=4096,
    console_print=console_print_func,
    thinking_budget=10000,
    location="San Francisco",
    system_prompt=None,
    tts_client=openai_client,
    tts_voice="alloy",
    tts_speed=1.0,
    tts_model="tts-1",
    playback=audio_playback,
    verbose=True,
)
```

### 3. Unclear Parameter Relationships
Related parameters are not grouped:
- TTS parameters (4 params) could be a group
- Model parameters (3 params) could be a group
- Context parameters (2 params) could be a group

### 4. Difficult to Extend
Adding new configuration requires:
- Modifying constructor signature
- Updating all call sites
- Managing additional instance variables
- No clear organization for new parameters

### 5. Testing Challenges
- Tests must provide 13 parameters even if most aren't relevant
- Difficult to create test fixtures
- Hard to test different configurations
- Many parameters make mocking complex

## Proposed Refactoring Strategy

### Phase 1: Create Configuration Objects
Group related parameters into typed configuration objects:

```python
@dataclass
class LLMModelConfig:
    """Configuration for LLM model behavior."""
    model: str
    max_tokens: int
    thinking_budget: int

    @staticmethod
    def from_config(config: Config) -> "LLMModelConfig":
        """Create from application configuration."""
        return LLMModelConfig(
            model=config.llm.model,
            max_tokens=config.llm.max_tokens,
            thinking_budget=config.llm.thinking_budget,
        )

    @staticmethod
    def default() -> "LLMModelConfig":
        """Create with sensible defaults."""
        return LLMModelConfig(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            thinking_budget=10000,
        )


@dataclass
class TTSConfig:
    """Configuration for text-to-speech synthesis."""
    client: openai.AsyncOpenAI | None
    voice: str
    speed: float
    model: str

    @staticmethod
    def from_config(config: Config) -> "TTSConfig | None":
        """Create from application configuration, or None if TTS disabled."""
        if not config.tts.enabled:
            return None

        return TTSConfig(
            client=None,  # Created separately
            voice=config.tts.voice,
            speed=config.tts.speed,
            model=config.tts.model,
        )

    @staticmethod
    def default() -> "TTSConfig":
        """Create with sensible defaults."""
        return TTSConfig(
            client=None,
            voice="alloy",
            speed=1.0,
            model="tts-1",
        )

    @property
    def enabled(self) -> bool:
        """Check if TTS is enabled (has client)."""
        return self.client is not None


@dataclass
class LLMContextConfig:
    """Configuration for LLM context and prompts."""
    location: str | None
    system_prompt: str | None

    @staticmethod
    def from_config(config: Config) -> "LLMContextConfig":
        """Create from application configuration."""
        return LLMContextConfig(
            location=config.location.name,
            system_prompt=config.llm.system_prompt,
        )

    @staticmethod
    def default() -> "LLMContextConfig":
        """Create with minimal defaults."""
        return LLMContextConfig(
            location=None,
            system_prompt=None,
        )
```

### Phase 2: Refactor Constructor
Simplify constructor using configuration objects:

```python
class LLMResponder:
    """
    Handles LLM interactions with streaming responses and optional TTS.

    Uses configuration objects for cleaner initialization and easier testing.
    """

    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        model_config: LLMModelConfig,
        context_config: LLMContextConfig,
        tts_config: TTSConfig | None = None,
        playback: AudioPlayback | None = None,
        console_print: ConsolePrint | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize LLM responder with configuration.

        Args:
            client: Anthropic API client
            model_config: Model behavior configuration
            context_config: Context and prompt configuration
            tts_config: Optional TTS configuration
            playback: Optional audio playback for TTS
            console_print: Optional console output handler
            verbose: Enable verbose logging
        """
        self.client = client
        self.model_config = model_config
        self.context_config = context_config
        self.tts_config = tts_config
        self.playback = playback
        self.console_print = console_print or self._default_console_print
        self.verbose = verbose

        # Validate configuration
        if tts_config and tts_config.enabled and not playback:
            raise ValueError("TTS enabled but no playback device provided")

    @staticmethod
    def _default_console_print(text: str) -> None:
        """Default console print implementation."""
        print(text, end="", flush=True)
```

Now only 7 parameters (down from 13), with clearer organization.

### Phase 3: Create Builder Pattern (Alternative Approach)
For even more flexibility, implement builder pattern:

```python
class LLMResponderBuilder:
    """Builder for constructing LLMResponder with fluent interface."""

    def __init__(self, client: anthropic.AsyncAnthropic):
        """Initialize builder with required client."""
        self._client = client
        self._model_config = LLMModelConfig.default()
        self._context_config = LLMContextConfig.default()
        self._tts_config: TTSConfig | None = None
        self._playback: AudioPlayback | None = None
        self._console_print: ConsolePrint | None = None
        self._verbose = False

    def with_model_config(
        self,
        config: LLMModelConfig
    ) -> "LLMResponderBuilder":
        """Set model configuration."""
        self._model_config = config
        return self

    def with_model(self, model: str) -> "LLMResponderBuilder":
        """Set model name (convenience method)."""
        self._model_config.model = model
        return self

    def with_tts(
        self,
        config: TTSConfig,
        playback: AudioPlayback
    ) -> "LLMResponderBuilder":
        """Enable TTS with configuration and playback."""
        self._tts_config = config
        self._playback = playback
        return self

    def with_location(self, location: str) -> "LLMResponderBuilder":
        """Set location context."""
        self._context_config.location = location
        return self

    def with_system_prompt(self, prompt: str) -> "LLMResponderBuilder":
        """Set custom system prompt."""
        self._context_config.system_prompt = prompt
        return self

    def with_console_print(
        self,
        console_print: ConsolePrint
    ) -> "LLMResponderBuilder":
        """Set console print handler."""
        self._console_print = console_print
        return self

    def with_verbose(self, verbose: bool = True) -> "LLMResponderBuilder":
        """Enable verbose logging."""
        self._verbose = verbose
        return self

    def build(self) -> LLMResponder:
        """
        Build LLMResponder instance.

        Returns:
            Configured LLMResponder instance

        Raises:
            ValueError: If configuration is invalid
        """
        return LLMResponder(
            client=self._client,
            model_config=self._model_config,
            context_config=self._context_config,
            tts_config=self._tts_config,
            playback=self._playback,
            console_print=self._console_print,
            verbose=self._verbose,
        )


# Usage with builder:
llm_responder = (
    LLMResponderBuilder(anthropic_client)
    .with_model("claude-3-5-sonnet-20241022")
    .with_location("San Francisco")
    .with_tts(tts_config, audio_playback)
    .with_verbose()
    .build()
)
```

### Phase 4: Create Factory Method
Add factory method for common construction patterns:

```python
class LLMResponder:
    # ... existing code ...

    @classmethod
    def from_config(
        cls,
        config: Config,
        anthropic_client: anthropic.AsyncAnthropic,
        openai_client: openai.AsyncOpenAI | None = None,
        playback: AudioPlayback | None = None,
        console_print: ConsolePrint | None = None,
    ) -> "LLMResponder":
        """
        Create LLMResponder from application configuration.

        Args:
            config: Application configuration
            anthropic_client: Anthropic API client
            openai_client: Optional OpenAI client for TTS
            playback: Optional audio playback device
            console_print: Optional console output handler

        Returns:
            Configured LLMResponder instance
        """
        model_config = LLMModelConfig.from_config(config)
        context_config = LLMContextConfig.from_config(config)

        # Create TTS config if enabled
        tts_config = None
        if config.tts.enabled and openai_client:
            tts_config = TTSConfig.from_config(config)
            tts_config.client = openai_client

        return cls(
            client=anthropic_client,
            model_config=model_config,
            context_config=context_config,
            tts_config=tts_config,
            playback=playback,
            console_print=console_print,
            verbose=config.verbose,
        )


# Simple usage:
llm_responder = LLMResponder.from_config(
    config=app_config,
    anthropic_client=anthropic_client,
    openai_client=openai_client,
    playback=audio_playback,
)
```

## Benefits of Refactoring

### Improved Usability
- Clearer parameter organization
- Self-documenting configuration groups
- Easier to construct in common cases
- Optional builder for complex configurations

### Better Maintainability
- Adding new TTS parameters only affects `TTSConfig`
- Adding new model parameters only affects `LLMModelConfig`
- Changes are localized to specific config classes
- No need to modify constructor signature frequently

### Enhanced Testability
- Can test with minimal configuration objects
- Easy to create test fixtures for specific scenarios
- Mock requirements reduced
- Configuration validation can be tested separately

### Type Safety
- Configuration objects are type-checked
- Related parameters are grouped
- Invalid configurations caught at construction time
- Better IDE autocomplete support

### Extensibility
- New configuration groups easy to add
- Builder pattern allows optional features
- Factory methods for common patterns
- Clear extension points

## Migration Strategy

1. **Create configuration classes**
   - Implement `LLMModelConfig`
   - Implement `TTSConfig`
   - Implement `LLMContextConfig`
   - Add tests for each

2. **Add new constructor alongside old**
   - Create `__init_v2__` or similar
   - Keep old constructor temporarily
   - Add factory methods

3. **Update call sites incrementally**
   - Start with test code
   - Update application code
   - Verify no behavior changes

4. **Remove old constructor**
   - Once all call sites updated
   - Remove lint suppression
   - Clean up deprecated code

5. **Add builder if needed**
   - Implement builder pattern
   - Update documentation
   - Add builder examples

## Estimated Effort
- **Time**: 1-2 days for complete refactoring
- **Risk**: Low (straightforward refactoring, clear interfaces)
- **Impact**: Medium (affects LLM responder initialization)

## Success Metrics
- [ ] Constructor has ≤ 7 parameters
- [ ] No lint suppressions needed
- [ ] All parameters logically grouped
- [ ] Factory method from Config works
- [ ] All tests pass with new constructor
- [ ] No behavior changes in LLM interaction
- [ ] Documentation updated with examples

## Additional Improvements

### Consider These Related Refactors
1. **Extract TTS functionality**: Move TTS synthesis to separate class
2. **Separate response parsing**: Extract `_extract_modalities` into `ResponseParser`
3. **Add configuration validation**: Validate config objects at construction
4. **Create configuration presets**: Common configurations as named presets

### Future Enhancements
- Configuration serialization/deserialization
- Configuration inheritance/composition
- Runtime configuration updates
- Configuration validation schemas

## References
- Builder Pattern: https://refactoring.guru/design-patterns/builder
- Python dataclasses: https://docs.python.org/3/library/dataclasses.html
- Too Many Parameters refactoring: https://refactoring.guru/smells/long-parameter-list
