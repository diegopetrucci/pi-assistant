"""
Test script to verify audio setup and list available devices
"""
import sounddevice as sd
import numpy as np
from config import SAMPLE_RATE, BUFFER_SIZE, CHANNELS, DTYPE

print('=== Audio Device Test ===\n')

# List available audio devices
print('Available audio devices:')
print(sd.query_devices())
print()

# Get default input device
try:
    default_input = sd.query_devices(kind='input')
    print(f'Default input device: {default_input["name"]}')
    print(f'  Max input channels: {default_input["max_input_channels"]}')
    print(f'  Default sample rate: {default_input["default_samplerate"]}')
    print()
except Exception as e:
    print(f'Error getting default input device: {e}')
    print()

# Verify configuration
print('=== Configuration Verification ===')
print(f'Sample rate: {SAMPLE_RATE} Hz')
print(f'Buffer size: {BUFFER_SIZE} frames')
print(f'Channels: {CHANNELS}')
print(f'Data type: {DTYPE}')
print(f'Expected bytes per chunk: {BUFFER_SIZE * CHANNELS * 2} bytes')
print()

print('Audio configuration loaded successfully!')
