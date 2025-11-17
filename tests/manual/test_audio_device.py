"""
Test script to verify audio setup and list available devices
"""

import sounddevice as sd  # pyright: ignore[reportMissingTypeStubs]

from pi_assistant.config import BUFFER_SIZE, CHANNELS, DTYPE, SAMPLE_RATE

print("=== Audio Device Test ===\n")

# List available audio devices
print("Available audio devices:")
print(sd.query_devices())
print()

# Get default input device
try:
    default_input = sd.query_devices(kind="input")
    print(f"Default input device: {default_input['name']}")  # pyright: ignore[reportArgumentType,reportCallIssue]
    print(f"  Max input channels: {default_input['max_input_channels']}")  # pyright: ignore[reportArgumentType,reportCallIssue]
    print(f"  Default sample rate: {default_input['default_samplerate']}")  # pyright: ignore[reportArgumentType,reportCallIssue]
    print()
except Exception as e:
    print(f"Error getting default input device: {e}")
    print()

# Verify configuration
print("=== Configuration Verification ===")
print(f"Sample rate: {SAMPLE_RATE} Hz")
print(f"Buffer size: {BUFFER_SIZE} frames")
print(f"Channels: {CHANNELS}")
print(f"Data type: {DTYPE}")
print(f"Expected bytes per chunk: {BUFFER_SIZE * CHANNELS * 2} bytes")
print()

print("Audio configuration loaded successfully!")
