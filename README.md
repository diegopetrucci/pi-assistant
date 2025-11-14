# Raspberry Pi Audio Transcription System

Real-time speech-to-text transcription system for Raspberry Pi 5 that streams audio from a USB microphone to OpenAI's Realtime API and displays transcribed text in the terminal.

## Features

- Real-time audio capture from USB microphone
- Streams to OpenAI Realtime API for transcription
- Server-side Voice Activity Detection (VAD)
- Optional auto-stop logging after sustained silence
- Optimized for Raspberry Pi 5
- 24kHz, mono, 16-bit PCM audio

## Requirements

### Hardware
- Raspberry Pi 5
- USB microphone (plug-and-play, ALSA compatible)

### Software
- Python 3.8+
- OpenAI API key

## Installation

### 1. Clone or Navigate to Project

```bash
cd /path/to/pi-transcription
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### (Optional) Development Tools

```bash
pip install -r requirements-dev.txt
pre-commit install
```

This installs the Ruff formatter/linter, pre-commit framework, and registers the Git hook locally.

### 4. Configure API Key

Create a `.env` file with your OpenAI API key:

```bash
cp .env.example .env
```

Then edit `.env` and add your actual API key:

```
OPENAI_API_KEY=your_actual_api_key_here
```

Get your API key from: https://platform.openai.com/api-keys

## Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run full transcription pipeline (default mode)
python3 transcribe.py

# Test WebSocket connection to OpenAI (requires API key)
python3 transcribe.py test-websocket

# Test audio capture from microphone (no API key needed)
python3 transcribe.py test-audio
```

## Code Quality

Ruff is configured via `ruff.toml` to handle both formatting and linting.

```bash
# Format the codebase
ruff format .

# Run lint checks (apply autofixes when possible)
ruff check --fix .

# Run the Git hook suite manually
pre-commit run --all-files
```

These commands automatically exclude generated artifacts such as `venv/` and `test_recording.wav`.

## Raspberry Pi Setup

### System Dependencies

On Raspberry Pi, install the required system libraries:

```bash
# Update system
sudo apt-get update

# Install audio libraries
sudo apt-get install -y libportaudio2 portaudio19-dev python3-dev

# Install Python pip
sudo apt-get install -y python3-pip

# Optional: Set CPU governor to performance
sudo cpufreq-set -g performance
```

### Test Microphone with ALSA

```bash
# List available microphones
arecord -l

# Test microphone recording (replace hw:X,Y with your device)
arecord --device=hw:1,0 --format S16_LE --rate 24000 -c 1 test.wav
```

## Configuration

Audio and API settings are in `config.py`:

- **Sample Rate**: 24 kHz (OpenAI requirement)
- **Buffer Size**: 1024 frames
- **Channels**: Mono (1 channel)
- **Format**: 16-bit PCM
- **Model**: gpt-4o-transcribe
- **Endpoint**: wss://api.openai.com/v1/realtime
- **Authentication**: Direct API key (Authorization header)
- **VAD**: Server-side Voice Activity Detection
- **Noise Reduction**: Near-field (optimized for close-talking microphones)
- **Auto-stop**: Enable/disable, adjust silence threshold, and tweak timeout via
  `AUTO_STOP_*` settings (emits `[TURN]` logs when a pause ends)

## Project Structure

```
pi-transcription/
├── .env                  # API key (create from .env.example)
├── .env.example          # Template for .env
├── .gitignore           # Git ignore rules
├── .pre-commit-config.yaml # Pre-commit hooks (Ruff)
├── README.md            # This file
├── requirements.txt     # Python dependencies
├── requirements-dev.txt # Development dependencies (Ruff)
├── ruff.toml            # Ruff formatter/linter configuration
├── config.py            # Configuration settings
├── transcribe.py        # Main application with audio capture
├── test_audio.py        # Audio device listing utility
├── test_save_audio.py   # Audio capture verification (saves WAV file)
├── plan.md              # Implementation plan
└── venv/                # Virtual environment (created during setup)
```

## Implementation Status

- ✅ **Phase 1**: Project Setup - Virtual environment, dependencies, .env configuration
- ✅ **Phase 2**: Configuration - Audio settings, API configuration, session config
- ✅ **Phase 3**: Audio Capture - USB microphone input, sounddevice integration, async queue
- ✅ **Phase 4**: WebSocket Client - API key auth, connection management, event handling
- ✅ **Phase 5**: Integration - Audio capture bridged with WebSocket streaming
- ⏳ **Phase 6**: Error Handling - Graceful shutdown and resiliency improvements

## Troubleshooting

**Virtual environment not activated:**
```bash
source venv/bin/activate
```

**WebSocket connection errors:**
- Check API key in `.env` file
- Verify internet connection
- Ensure Realtime API access

**No audio devices found:**
```bash
# List microphones (Raspberry Pi)
arecord -l
```

**Microphone permission (macOS):**
System Settings → Privacy & Security → Microphone

## License

This project is for educational and development purposes.

## Support

For issues and questions, refer to:
- OpenAI Realtime API docs: https://platform.openai.com/docs/guides/realtime
- Plan document: `plan.md`
