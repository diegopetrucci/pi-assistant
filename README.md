# Raspberry Pi Audio Transcription System

Real-time speech-to-text transcription system for Raspberry Pi 5 that streams audio from a USB microphone to OpenAI's Realtime API and displays transcribed text in the terminal.

## Features

- Real-time audio capture from USB microphone
- Streams to OpenAI Realtime API for transcription
- Server-side Voice Activity Detection (VAD)
- Optional auto-stop logging after sustained silence
- Wake-word gated streaming with openWakeWord’s “hey jarvis” model (pre-roll included)
- Optimized for Raspberry Pi 5
- 24kHz, mono, 16-bit PCM audio

## Requirements

### Hardware
- Raspberry Pi 5
- USB microphone (plug-and-play, ALSA compatible)

### Software
- Python 3.9+
- uv (https://docs.astral.sh/uv/)
- OpenAI API key

## Installation

### 1. Clone or Navigate to Project

```bash
cd /path/to/pi-transcription
```

### 2. Install uv (one time per machine)

```bash
# macOS (Homebrew)
brew install uv

# Or use the official installer on Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Sync Dependencies (creates .venv automatically)

```bash
# Runtime dependencies only
uv sync

# Include Ruff + pre-commit (optional dev tools)
uv sync --group dev
uv run pre-commit install
```

`uv sync` creates a managed `.venv/` if one does not already exist and keeps it up to date as dependencies change.

Need to hand a `requirements.txt` to another system? Export one on demand:

```bash
uv export --format requirements-txt > requirements.txt
```

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

You can run commands directly through uv (no manual activation needed) or activate `.venv/` yourself.

```bash
# Full transcription pipeline (default mode)
uv run python transcribe.py

# Force streaming without the wake word (debug mode)
uv run python transcribe.py --force-always-on

# Explicitly require the wake word when FORCE_ALWAYS_ON=1 is set in the env
uv run python transcribe.py --no-force-always-on

# Test WebSocket connection to OpenAI (requires API key)
uv run python transcribe.py test-websocket

# Test audio capture from microphone (no API key needed)
uv run python transcribe.py test-audio
```

## Code Quality

Ruff is configured via `ruff.toml` to handle both formatting and linting.

```bash
# Format the codebase
uv run ruff format .

# Run lint checks (apply autofixes when possible)
uv run ruff check --fix .

# Run the Git hook suite manually
uv run pre-commit run --all-files
```

These commands automatically exclude generated artifacts such as `.venv/` and `test_recording.wav`.

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

### Wake Word Settings

Wake-word gating is enabled by default and uses the bundled openWakeWord “hey jarvis” model:

- `WAKE_WORD_MODEL_PATH` / `WAKE_WORD_MODEL_FALLBACK_PATH`: Paths to the `.tflite` and `.onnx` models stored in `models/`.
- `WAKE_WORD_MELSPEC_MODEL_PATH` / `WAKE_WORD_EMBEDDING_MODEL_PATH`: Feature-extractor assets (`melspectrogram.onnx` and `embedding_model.onnx`) bundled under `models/` so we can run openWakeWord without its download helper.
- `WAKE_WORD_SCORE_THRESHOLD` / `WAKE_WORD_CONSECUTIVE_FRAMES`: Confidence guard (default: score ≥ 0.5 for two consecutive frames).
- `PREROLL_DURATION_SECONDS`: Length of buffered audio (default: 1 second) that is prepended to the first streamed chunk after activation.
- `FORCE_ALWAYS_ON`: Set to `1` (or use `--force-always-on`) to bypass the wake word during troubleshooting. Use `--no-force-always-on` to override the env var.
- `WAKE_WORD_TARGET_SAMPLE_RATE`: Leave at 16 kHz unless you re-train a model that expects a different input rate.

Every detection above the configured threshold is logged with the `[WAKE]` label, and state transitions are reported with `[STATE]`. When the wake word fires, the controller flushes the pre-roll buffer plus live audio so the transcript includes the first spoken words after “hey jarvis”.

## Project Structure

```
pi-transcription/
├── .env                  # API key (create from .env.example)
├── .env.example          # Template for .env
├── .gitignore           # Git ignore rules
├── .pre-commit-config.yaml # Pre-commit hooks (Ruff)
├── README.md            # This file
├── pyproject.toml       # Project metadata and dependencies
├── uv.lock              # Resolved dependency lockfile
├── ruff.toml            # Ruff formatter/linter configuration
├── models/              # Bundled openWakeWord models (Apache 2.0 license)
├── config.py            # Configuration settings
├── transcribe.py        # Main application with audio capture
├── test_audio.py        # Audio device listing utility
├── test_save_audio.py   # Audio capture verification (saves WAV file)
├── plan.md              # Implementation plan
└── .venv/               # Managed virtual environment (generated by `uv sync`)
```

## Automated Tests

The `tests/test_wake_word.py` regression uses the generated `tests/hey_jarvis.wav` fixture to ensure the detector fires exactly once for the wake phrase. Run it with:

```bash
uv run python -m unittest tests/test_wake_word.py
```

> **Note:** The wake-word test is skipped automatically when `openwakeword` (and its runtimes) are unavailable.

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
source .venv/bin/activate
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

**Licensing for bundled wake-word models:**
- `models/hey_jarvis_v0.1.(onnx|tflite)`, `models/melspectrogram.onnx`, and `models/embedding_model.onnx` are distributed under the Apache 2.0 license from the [openWakeWord](https://github.com/dscripka/openWakeWord) project.

**Microphone permission (macOS):**
System Settings → Privacy & Security → Microphone

## License

This project is for educational and development purposes.

## Support

For issues and questions, refer to:
- OpenAI Realtime API docs: https://platform.openai.com/docs/guides/realtime
- Plan document: `plan.md`
