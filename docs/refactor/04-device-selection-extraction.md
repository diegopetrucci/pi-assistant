# Refactoring: Device Selection Extraction

## Priority
**MEDIUM** - Estimated effort: 1-2 days

## Problem

Audio device selection logic is **duplicated** across capture and playback modules with similar patterns:

### Duplicate Code Locations

1. **Input Device Selection**: `src/pi_assistant/audio/capture.py:119-193` (75 lines)
2. **Output Device Selection**: `src/pi_assistant/audio/playback.py:79-135` (57 lines)

Both implement nearly identical logic:
- Query available devices
- Filter by device type (input/output)
- Handle device index from configuration
- Interactive device selection
- Device validation
- Default device fallback

### Current Code Smell

**capture.py:119-193**
```python
def _select_input_device() -> int:
    devices = sd.query_devices()

    # Filter input devices
    input_devices = [
        (idx, device)
        for idx, device in enumerate(devices)
        if device["max_input_channels"] > 0
    ]

    # Interactive selection
    if not input_devices:
        raise RuntimeError("No input devices found")

    print("Available input devices:")
    for idx, (device_idx, device) in enumerate(input_devices):
        print(f"  {idx}: {device['name']}")

    # ... 50+ more lines of selection logic
```

**playback.py:79-135**
```python
def _select_output_device() -> int:
    devices = sd.query_devices()

    # Filter output devices
    output_devices = [
        (idx, device)
        for idx, device in enumerate(devices)
        if device["max_output_channels"] > 0
    ]

    # Interactive selection
    if not output_devices:
        raise RuntimeError("No output devices found")

    print("Available output devices:")
    for idx, (device_idx, device) in enumerate(output_devices):
        print(f"  {idx}: {device['name']}")

    # ... 40+ more lines of selection logic
```

### Issues

1. **Code duplication**: ~80% of the logic is identical
2. **No caching**: Devices queried every time, even when unchanged
3. **No validation**: Device existence not validated until use
4. **Inconsistent errors**: Different error messages for same issues
5. **Hard to test**: Device selection mixed with I/O logic

## Proposed Solution

Create a **unified `AudioDeviceManager`** class that:
- Centralizes device enumeration and selection
- Caches device information
- Provides consistent validation
- Separates selection logic from I/O
- Enables easy testing with dependency injection

### Implementation

#### AudioDeviceManager (`src/pi_assistant/audio/device_manager.py`)

```python
"""Unified audio device management."""

from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum
import sounddevice as sd

DeviceType = Literal["input", "output"]


@dataclass(frozen=True)
class AudioDevice:
    """Represents an audio device."""
    index: int
    name: str
    channels: int
    sample_rate: float
    device_type: DeviceType

    @property
    def display_name(self) -> str:
        """Get display name for the device."""
        return f"{self.name} ({self.channels} ch, {self.sample_rate:.0f} Hz)"


class AudioDeviceManager:
    """Manages audio device enumeration and selection."""

    def __init__(self):
        self._device_cache: Optional[list[dict]] = None

    def get_devices(self, refresh: bool = False) -> list[dict]:
        """Get all audio devices (cached)."""
        if self._device_cache is None or refresh:
            self._device_cache = sd.query_devices()
        return self._device_cache

    def list_devices(
        self,
        device_type: DeviceType,
        refresh: bool = False
    ) -> list[AudioDevice]:
        """List all devices of a specific type."""
        devices = self.get_devices(refresh=refresh)

        filtered = []
        for idx, device in enumerate(devices):
            if self._is_device_type(device, device_type):
                filtered.append(
                    AudioDevice(
                        index=idx,
                        name=device["name"],
                        channels=self._get_channel_count(device, device_type),
                        sample_rate=device["default_samplerate"],
                        device_type=device_type
                    )
                )

        return filtered

    def get_device(
        self,
        device_index: int,
        device_type: DeviceType
    ) -> AudioDevice:
        """Get a specific device by index."""
        devices = self.get_devices()

        if device_index < 0 or device_index >= len(devices):
            raise ValueError(
                f"Invalid device index: {device_index}. "
                f"Available range: 0-{len(devices) - 1}"
            )

        device = devices[device_index]

        if not self._is_device_type(device, device_type):
            raise ValueError(
                f"Device {device_index} ({device['name']}) is not an "
                f"{device_type} device"
            )

        return AudioDevice(
            index=device_index,
            name=device["name"],
            channels=self._get_channel_count(device, device_type),
            sample_rate=device["default_samplerate"],
            device_type=device_type
        )

    def get_default_device(self, device_type: DeviceType) -> AudioDevice:
        """Get the system default device for a type."""
        if device_type == "input":
            default_idx = sd.default.device[0]
        else:
            default_idx = sd.default.device[1]

        return self.get_device(default_idx, device_type)

    def select_device_interactive(
        self,
        device_type: DeviceType,
        prompt: Optional[str] = None
    ) -> AudioDevice:
        """Interactively select a device."""
        devices = self.list_devices(device_type)

        if not devices:
            raise RuntimeError(f"No {device_type} devices found")

        if len(devices) == 1:
            print(f"Using only available {device_type} device: {devices[0].display_name}")
            return devices[0]

        # Display available devices
        print(f"\nAvailable {device_type} devices:")
        for idx, device in enumerate(devices):
            default_marker = " (default)" if self._is_default(device, device_type) else ""
            print(f"  {idx}: {device.display_name}{default_marker}")

        # Get user selection
        prompt_text = prompt or f"Select {device_type} device [0-{len(devices) - 1}]: "
        while True:
            try:
                selection = input(prompt_text).strip()
                if not selection:
                    # Use default
                    return self.get_default_device(device_type)

                idx = int(selection)
                if 0 <= idx < len(devices):
                    return devices[idx]
                else:
                    print(f"Please enter a number between 0 and {len(devices) - 1}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                raise RuntimeError("Device selection cancelled by user")

    def resolve_device(
        self,
        device_index: Optional[int],
        device_type: DeviceType,
        interactive: bool = True
    ) -> AudioDevice:
        """Resolve device from index, with fallback to interactive selection."""
        # If index provided, validate and use it
        if device_index is not None:
            try:
                return self.get_device(device_index, device_type)
            except ValueError as exc:
                if interactive:
                    print(f"Warning: {exc}")
                    print("Please select a valid device:")
                    return self.select_device_interactive(device_type)
                else:
                    raise

        # No index provided - interactive or default
        if interactive:
            return self.select_device_interactive(device_type)
        else:
            return self.get_default_device(device_type)

    @staticmethod
    def _is_device_type(device: dict, device_type: DeviceType) -> bool:
        """Check if device matches the specified type."""
        if device_type == "input":
            return device["max_input_channels"] > 0
        else:
            return device["max_output_channels"] > 0

    @staticmethod
    def _get_channel_count(device: dict, device_type: DeviceType) -> int:
        """Get channel count for device type."""
        if device_type == "input":
            return device["max_input_channels"]
        else:
            return device["max_output_channels"]

    def _is_default(self, device: AudioDevice, device_type: DeviceType) -> bool:
        """Check if device is the system default."""
        try:
            default_device = self.get_default_device(device_type)
            return device.index == default_device.index
        except Exception:
            return False
```

## Usage

### Before (Current)

**capture.py**
```python
# 75 lines of device selection logic
def _select_input_device() -> int:
    devices = sd.query_devices()
    # ... lots of code ...
    return device_index

class AudioCapture:
    def __init__(self):
        device_index = INPUT_DEVICE_INDEX or _select_input_device()
        self.device_index = device_index
```

**playback.py**
```python
# 57 lines of device selection logic
def _select_output_device() -> int:
    devices = sd.query_devices()
    # ... lots of code ...
    return device_index

class SpeechPlayer:
    def __init__(self):
        device_index = OUTPUT_DEVICE_INDEX or _select_output_device()
        self.device_index = device_index
```

### After (Proposed)

**capture.py**
```python
from .device_manager import AudioDeviceManager

class AudioCapture:
    def __init__(
        self,
        device_manager: Optional[AudioDeviceManager] = None,
        device_index: Optional[int] = None
    ):
        self.device_manager = device_manager or AudioDeviceManager()

        # Simple, consistent device resolution
        device = self.device_manager.resolve_device(
            device_index=device_index,
            device_type="input",
            interactive=True
        )
        self.device = device
```

**playback.py**
```python
from .device_manager import AudioDeviceManager

class SpeechPlayer:
    def __init__(
        self,
        device_manager: Optional[AudioDeviceManager] = None,
        device_index: Optional[int] = None
    ):
        self.device_manager = device_manager or AudioDeviceManager()

        device = self.device_manager.resolve_device(
            device_index=device_index,
            device_type="output",
            interactive=True
        )
        self.device = device
```

**app.py** (shared instance)
```python
from pi_assistant.audio import AudioDeviceManager

# Create shared manager for caching
device_manager = AudioDeviceManager()

# Pass to components
audio_capture = AudioCapture(
    device_manager=device_manager,
    device_index=config.audio.input_device_index
)

speech_player = SpeechPlayer(
    device_manager=device_manager,
    device_index=config.audio.output_device_index
)
```

## Benefits

1. **DRY principle**: Device selection logic in one place
2. **Caching**: Devices queried once and cached
3. **Testability**: Easy to mock device manager
4. **Consistency**: Same error messages and behavior
5. **Reusability**: Can be used by any audio component
6. **Type safety**: `AudioDevice` dataclass provides structure

## Migration Plan

### Phase 1: Create Device Manager (Day 1, Morning)
1. Create `src/pi_assistant/audio/device_manager.py`
2. Implement `AudioDeviceManager` class
3. Implement `AudioDevice` dataclass
4. Add unit tests (mock `sounddevice`)

### Phase 2: Update AudioCapture (Day 1, Afternoon)
1. Refactor `capture.py` to use `AudioDeviceManager`
2. Remove `_select_input_device()` function
3. Update constructor to accept device manager
4. Update tests

### Phase 3: Update SpeechPlayer (Day 2, Morning)
1. Refactor `playback.py` to use `AudioDeviceManager`
2. Remove `_select_output_device()` function
3. Update constructor to accept device manager
4. Update tests

### Phase 4: Update App (Day 2, Afternoon)
1. Create shared `AudioDeviceManager` instance in `app.py`
2. Pass to both capture and playback components
3. Integration testing
4. Verify caching works correctly

## Testing Strategy

```python
import pytest
from unittest.mock import Mock, patch
from pi_assistant.audio.device_manager import AudioDeviceManager, AudioDevice

@pytest.fixture
def mock_devices():
    """Mock device list."""
    return [
        {
            "name": "Built-in Mic",
            "max_input_channels": 2,
            "max_output_channels": 0,
            "default_samplerate": 48000.0
        },
        {
            "name": "Built-in Speaker",
            "max_input_channels": 0,
            "max_output_channels": 2,
            "default_samplerate": 48000.0
        }
    ]

@patch('sounddevice.query_devices')
def test_list_input_devices(mock_query, mock_devices):
    """Test listing input devices."""
    mock_query.return_value = mock_devices
    manager = AudioDeviceManager()

    input_devices = manager.list_devices("input")

    assert len(input_devices) == 1
    assert input_devices[0].name == "Built-in Mic"
    assert input_devices[0].channels == 2

@patch('sounddevice.query_devices')
def test_device_caching(mock_query, mock_devices):
    """Test that devices are cached."""
    mock_query.return_value = mock_devices
    manager = AudioDeviceManager()

    # First call
    devices1 = manager.get_devices()
    # Second call - should use cache
    devices2 = manager.get_devices()

    assert devices1 == devices2
    assert mock_query.call_count == 1  # Only called once!

@patch('sounddevice.query_devices')
def test_invalid_device_index(mock_query, mock_devices):
    """Test error handling for invalid device index."""
    mock_query.return_value = mock_devices
    manager = AudioDeviceManager()

    with pytest.raises(ValueError, match="Invalid device index"):
        manager.get_device(99, "input")

@patch('sounddevice.query_devices')
def test_wrong_device_type(mock_query, mock_devices):
    """Test error when device doesn't support requested type."""
    mock_query.return_value = mock_devices
    manager = AudioDeviceManager()

    # Device 1 is output only
    with pytest.raises(ValueError, match="not an input device"):
        manager.get_device(1, "input")
```

## Code Reduction

| File | Before | After | Savings |
|------|--------|-------|---------|
| `capture.py` | 193 lines | ~120 lines | 73 lines (38%) |
| `playback.py` | 135 lines | ~80 lines | 55 lines (41%) |
| **Total** | **328 lines** | **~200 lines** | **~128 lines (39%)** |

Plus adds ~200 lines in `device_manager.py`, but that's shared and well-tested.

## Success Metrics

- [ ] Device selection code removed from `capture.py`
- [ ] Device selection code removed from `playback.py`
- [ ] Shared `AudioDeviceManager` used in both modules
- [ ] Device caching working (verify single query)
- [ ] 90%+ test coverage on device manager
- [ ] No regression in device selection behavior

## Related Refactorings

- **Error Handling**: Use `AudioDeviceError` for device issues
- **Configuration**: Device indices come from config
- **Testing**: Mockable device manager improves testability
