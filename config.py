"""
Configuration settings for Raspberry Pi Audio Transcription System
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Audio Configuration
SAMPLE_RATE = 24000  # 24 kHz (OpenAI requirement)
BUFFER_SIZE = 1024   # frames (balanced for Pi 5 performance)
CHANNELS = 1         # Mono
DTYPE = 'int16'      # 16-bit PCM

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_REALTIME_ENDPOINT = 'wss://api.openai.com/v1/realtime?intent=transcription'
OPENAI_MODEL = 'gpt-4o-transcribe'

# WebSocket Headers
WEBSOCKET_HEADERS = {
    'Authorization': f'Bearer {OPENAI_API_KEY}',
    'OpenAI-Beta': 'realtime=v1'
}

# Session Configuration for OpenAI Realtime API (Transcription mode)
SESSION_CONFIG = {
    "type": "transcription_session.update",
    "input_audio_format": "pcm16",
    "input_audio_transcription": {
        "model": "gpt-4o-transcribe",
        "prompt": "",
        "language": ""
    },
    "turn_detection": {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500
    },
    "input_audio_noise_reduction": {
        "type": "near_field"
    },
    "include": [
        "item.input_audio_transcription.logprobs"
    ]
}

# Queue Configuration
AUDIO_QUEUE_MAX_SIZE = 100  # Limit queue size to prevent memory buildup

# Validation
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. "
        "Please create a .env file with your API key."
    )
