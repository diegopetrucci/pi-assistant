# Refactoring: Configuration Module

## Priority
**HIGH** - Estimated effort: 2-3 days

## Problem

The configuration module `src/pi_assistant/config/__init__.py` (493 lines) has severe architectural issues:

### 1. Module-Level Side Effects
- Configuration is loaded and validated at **import time** (lines 223-494)
- Interactive prompts can trigger during module import (lines 95-220)
- Makes testing nearly impossible
- Creates hidden dependencies throughout the codebase

### 2. Mixed Responsibilities
The module handles multiple concerns:
- **File I/O**: Reading TOML config files (lines 38-75)
- **User interaction**: Interactive prompts for missing config (lines 95-220)
- **Validation**: Wake word configuration validation (lines 286-453)
- **Environment variable resolution**: Overriding config from env vars (lines 48-54)
- **Default value management**: Scattered throughout the file

### 3. Untestable Code
```python
# Lines 223-494: Executed on import
config = _load_config()
ANTHROPIC_API_KEY = _prompt_for_api_key(config) if not config.api_key else config.api_key
# ... 200+ more lines of module-level initialization
```

This means:
- Cannot test configuration loading in isolation
- Cannot mock user input for tests
- Cannot test error handling without side effects
- Every import triggers file I/O and validation

### 4. Tight Coupling
23 different modules import configuration constants directly:
```python
# controller.py:23-43
from pi_assistant.config import (
    ANTHROPIC_API_KEY,
    ASSISTANT_MODEL,
    CHUNK_LENGTH_SECONDS,
    # ... 20+ more constants
)
```

Changes to config structure ripple across the entire codebase.

## Proposed Solution

Refactor into a **lazy-loaded configuration system** with clear separation of concerns.

### New Structure

```
src/pi_assistant/config/
├── __init__.py          # Public API only
├── models.py            # Configuration data classes
├── loader.py            # File loading logic
├── validator.py         # Validation logic
├── wizard.py            # Interactive configuration setup
├── defaults.py          # Default values
└── environment.py       # Environment variable handling
```

### Implementation

#### 1. Configuration Models (`models.py`)

```python
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

@dataclass(frozen=True)
class AnthropicConfig:
    """Anthropic API configuration."""
    api_key: str
    model: str = "claude-3-5-sonnet-20241022"
    base_url: str = "https://api.anthropic.com"

@dataclass(frozen=True)
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    chunk_length_seconds: float = 0.1
    silence_threshold: float = 0.02
    silence_duration_threshold: float = 0.5
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None

    @property
    def chunk_size(self) -> int:
        """Calculate chunk size in bytes."""
        return int(self.sample_rate * self.chunk_length_seconds)

@dataclass(frozen=True)
class WakeWordConfig:
    """Wake word detection configuration."""
    enabled: bool = True
    engine: str = "porcupine"
    model_path: Optional[Path] = None
    sensitivity: float = 0.5
    access_key: Optional[str] = None

    def validate(self) -> None:
        """Validate wake word configuration."""
        if self.enabled and not self.access_key:
            raise ValueError("Wake word access_key is required when enabled")
        if self.sensitivity < 0.0 or self.sensitivity > 1.0:
            raise ValueError(f"Invalid sensitivity: {self.sensitivity}")

@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""
    verbose: bool = False
    log_file: Optional[Path] = None
    log_level: str = "INFO"

@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration."""
    anthropic: AnthropicConfig
    audio: AudioConfig
    wake_word: WakeWordConfig
    logging: LoggingConfig

    def validate(self) -> None:
        """Validate entire configuration."""
        self.wake_word.validate()
        # Add other validations

    @classmethod
    def from_dict(cls, data: dict) -> 'AppConfig':
        """Create configuration from dictionary."""
        return cls(
            anthropic=AnthropicConfig(**data.get('anthropic', {})),
            audio=AudioConfig(**data.get('audio', {})),
            wake_word=WakeWordConfig(**data.get('wake_word', {})),
            logging=LoggingConfig(**data.get('logging', {}))
        )
```

#### 2. Configuration Loader (`loader.py`)

```python
from pathlib import Path
from typing import Optional
import tomllib
from .models import AppConfig
from .defaults import DEFAULT_CONFIG

class ConfigLoader:
    """Handles loading configuration from files."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or self._default_config_path()

    @staticmethod
    def _default_config_path() -> Path:
        """Get default configuration file path."""
        return Path.home() / ".config" / "pi-assistant" / "config.toml"

    def load(self) -> dict:
        """Load configuration from file."""
        if not self.config_path.exists():
            return {}

        with open(self.config_path, 'rb') as f:
            return tomllib.load(f)

    def save(self, config: dict) -> None:
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        import tomli_w
        with open(self.config_path, 'wb') as f:
            tomli_w.dump(config, f)
```

#### 3. Environment Variable Resolver (`environment.py`)

```python
import os
from typing import Any, Dict

class EnvironmentResolver:
    """Resolves configuration from environment variables."""

    PREFIX = "PI_ASSISTANT_"

    @classmethod
    def resolve(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Override config values from environment variables."""
        env_overrides = {
            'ANTHROPIC_API_KEY': ('anthropic', 'api_key'),
            'ASSISTANT_MODEL': ('anthropic', 'model'),
            'SAMPLE_RATE': ('audio', 'sample_rate'),
            'WAKE_WORD_ENABLED': ('wake_word', 'enabled'),
            # ... more mappings
        }

        for env_var, (section, key) in env_overrides.items():
            full_env_var = f"{cls.PREFIX}{env_var}"
            if full_env_var in os.environ:
                value = os.environ[full_env_var]
                config.setdefault(section, {})[key] = cls._cast_value(key, value)

        return config

    @staticmethod
    def _cast_value(key: str, value: str) -> Any:
        """Cast environment variable to appropriate type."""
        # Type inference based on key or schema
        if key.endswith('_enabled'):
            return value.lower() in ('true', '1', 'yes')
        if key.endswith('_rate') or key.endswith('_index'):
            return int(value)
        if key.endswith('_threshold') or key.endswith('_sensitivity'):
            return float(value)
        return value
```

#### 4. Interactive Configuration Wizard (`wizard.py`)

```python
from typing import Optional
from .models import AppConfig, AnthropicConfig, WakeWordConfig

class ConfigurationWizard:
    """Interactive configuration setup."""

    def run(self, existing_config: Optional[dict] = None) -> AppConfig:
        """Run interactive configuration wizard."""
        config = existing_config or {}

        # Anthropic API setup
        if 'anthropic' not in config:
            config['anthropic'] = self._setup_anthropic()

        # Wake word setup
        if 'wake_word' not in config:
            config['wake_word'] = self._setup_wake_word()

        return AppConfig.from_dict(config)

    def _setup_anthropic(self) -> dict:
        """Interactive Anthropic API setup."""
        print("Anthropic API Configuration")
        print("-" * 40)

        api_key = input("Enter your Anthropic API key: ").strip()
        if not api_key:
            raise ValueError("API key is required")

        model = input(
            "Enter assistant model [claude-3-5-sonnet-20241022]: "
        ).strip() or "claude-3-5-sonnet-20241022"

        return {
            'api_key': api_key,
            'model': model
        }

    def _setup_wake_word(self) -> dict:
        """Interactive wake word setup."""
        print("\nWake Word Configuration")
        print("-" * 40)

        enabled = input("Enable wake word detection? [Y/n]: ").strip().lower()
        if enabled in ('n', 'no'):
            return {'enabled': False}

        access_key = input("Enter Porcupine access key: ").strip()
        sensitivity = input("Enter sensitivity [0.5]: ").strip() or "0.5"

        return {
            'enabled': True,
            'engine': 'porcupine',
            'access_key': access_key,
            'sensitivity': float(sensitivity)
        }
```

#### 5. Configuration Builder (`__init__.py`)

```python
from pathlib import Path
from typing import Optional
from .models import AppConfig
from .loader import ConfigLoader
from .environment import EnvironmentResolver
from .wizard import ConfigurationWizard
from .defaults import DEFAULT_CONFIG

class ConfigBuilder:
    """Builds application configuration from multiple sources."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        interactive: bool = True
    ):
        self.loader = ConfigLoader(config_path)
        self.interactive = interactive
        self._cached_config: Optional[AppConfig] = None

    def build(self) -> AppConfig:
        """Build configuration from all sources."""
        if self._cached_config is not None:
            return self._cached_config

        # 1. Start with defaults
        config_dict = DEFAULT_CONFIG.copy()

        # 2. Load from file
        file_config = self.loader.load()
        self._merge_config(config_dict, file_config)

        # 3. Override with environment variables
        config_dict = EnvironmentResolver.resolve(config_dict)

        # 4. Run interactive wizard if needed
        if self.interactive and self._needs_setup(config_dict):
            wizard = ConfigurationWizard()
            app_config = wizard.run(config_dict)
            self.loader.save(self._config_to_dict(app_config))
            self._cached_config = app_config
            return app_config

        # 5. Create and validate
        app_config = AppConfig.from_dict(config_dict)
        app_config.validate()

        self._cached_config = app_config
        return app_config

    def _needs_setup(self, config: dict) -> bool:
        """Check if interactive setup is needed."""
        return (
            not config.get('anthropic', {}).get('api_key') or
            not config.get('wake_word', {}).get('access_key')
        )

    @staticmethod
    def _merge_config(base: dict, override: dict) -> None:
        """Deep merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                ConfigBuilder._merge_config(base[key], value)
            else:
                base[key] = value

    @staticmethod
    def _config_to_dict(config: AppConfig) -> dict:
        """Convert AppConfig to dictionary."""
        from dataclasses import asdict
        return asdict(config)

# Public API
_builder: Optional[ConfigBuilder] = None

def get_config(
    config_path: Optional[Path] = None,
    interactive: bool = True,
    force_reload: bool = False
) -> AppConfig:
    """Get application configuration (cached)."""
    global _builder

    if _builder is None or force_reload:
        _builder = ConfigBuilder(config_path, interactive)

    return _builder.build()

# For backward compatibility (deprecated)
def load_config() -> AppConfig:
    """Load configuration (deprecated, use get_config instead)."""
    return get_config()
```

## Usage

### Before (Current)
```python
# Automatically loads on import - can't control it!
from pi_assistant.config import (
    ANTHROPIC_API_KEY,
    ASSISTANT_MODEL,
    SAMPLE_RATE
)
```

### After (Proposed)
```python
from pi_assistant.config import get_config

# Lazy load - only when needed
config = get_config()

# Access nested configuration
api_key = config.anthropic.api_key
model = config.anthropic.model
sample_rate = config.audio.sample_rate

# Non-interactive loading (for tests)
config = get_config(interactive=False)

# Custom config path
config = get_config(config_path=Path("/custom/config.toml"))
```

## Benefits

1. **Testability**: Configuration can be mocked and tested in isolation
2. **No side effects**: Import doesn't trigger I/O or user prompts
3. **Lazy loading**: Configuration only loaded when requested
4. **Type safety**: Strongly typed configuration with dataclasses
5. **Separation of concerns**: Each module has single responsibility
6. **Explicit dependencies**: Config is passed to components, not imported globally

## Migration Plan

### Phase 1: Create New Structure (Day 1)
1. Create new `config/` package structure
2. Implement `models.py` with all configuration dataclasses
3. Implement `loader.py` for file I/O
4. Implement `environment.py` for env var resolution

### Phase 2: Extract Interactive Logic (Day 1-2)
1. Implement `wizard.py` with interactive setup
2. Move all user prompting logic from `__init__.py`
3. Add tests for wizard in isolation

### Phase 3: Implement Builder (Day 2)
1. Implement `ConfigBuilder` in `__init__.py`
2. Add `get_config()` public API
3. Maintain backward compatibility with deprecation warnings

### Phase 4: Update Consumers (Day 2-3)
1. Update `app.py` to use `get_config()`
2. Update all modules to accept config in constructors instead of importing
3. Remove direct imports of config constants

### Phase 5: Cleanup (Day 3)
1. Remove old module-level initialization code
2. Remove deprecated backward compatibility shims
3. Update tests to use new configuration system

## Testing Strategy

```python
import pytest
from pi_assistant.config import ConfigBuilder, AppConfig
from pi_assistant.config.models import AnthropicConfig

def test_config_from_file(tmp_path):
    """Test loading configuration from file."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
[anthropic]
api_key = "test-key"
model = "claude-3-5-sonnet-20241022"
""")

    builder = ConfigBuilder(config_path=config_file, interactive=False)
    config = builder.build()

    assert config.anthropic.api_key == "test-key"
    assert config.anthropic.model == "claude-3-5-sonnet-20241022"

def test_config_env_override(tmp_path, monkeypatch):
    """Test environment variable overrides."""
    monkeypatch.setenv("PI_ASSISTANT_ANTHROPIC_API_KEY", "env-key")

    builder = ConfigBuilder(interactive=False)
    config = builder.build()

    assert config.anthropic.api_key == "env-key"

def test_config_validation():
    """Test configuration validation."""
    with pytest.raises(ValueError, match="access_key is required"):
        AppConfig(
            anthropic=AnthropicConfig(api_key="test"),
            wake_word=WakeWordConfig(enabled=True, access_key=None)
        ).validate()
```

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking changes for existing users | Maintain backward compatibility layer during transition |
| Configuration migration for users | Provide migration tool to convert old configs |
| Performance overhead from lazy loading | Cache config after first load |
| Complexity of new structure | Clear documentation and examples |

## Success Metrics

- [ ] No module-level side effects in config package
- [ ] 100% test coverage on configuration loading
- [ ] All 23 config-importing modules updated
- [ ] Configuration can be loaded non-interactively
- [ ] Config loading time < 10ms (cached)

## Related Refactorings

- **Configuration Coupling Reduction**: This enables dependency injection
- **Testing Infrastructure**: Makes entire codebase more testable
- **Type Safety**: Strong typing improves IDE support and catches errors
