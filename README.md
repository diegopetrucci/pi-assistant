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

### Optional: Pin repository-local Python 3.11

If your Raspberry Pi workflow needs Python 3.11 without touching the system installation, let `uv` download a portable interpreter that only applies inside this repo:

```bash
uv python pin 3.11
```

This stores the interpreter under `.uv/python/...` and records the version in `.python-version`. Commit `.python-version`, keep `.uv/` ignored, and every `uv` command you run here (including `uv sync`) will automatically use that local 3.11 build so packages like `tflite-runtime` install cleanly. To switch later, run `uv python pin <other-version>` or remove `.python-version`.

### 4. Install PortAudio (Raspberry Pi)

`sounddevice` depends on the PortAudio shared libraries, which Raspberry Pi OS does not install by default. Run the helper script once per Pi to pull in the required packages:

```bash
./scripts/install-portaudio-deps.sh
```

The script performs an `apt-get update` and installs `libportaudio2`, `libportaudiocpp0`, and `portaudio19-dev` using `sudo`. If you're on a distro without `apt-get`, install the equivalent PortAudio development packages via your package manager before running `uv run pi-transcription`.

### 5. Configure API Key

Create a `.env` file with your OpenAI API key:

```bash
cp .env.example .env
```

Then edit `.env` and add your actual API key:

```
OPENAI_API_KEY=your_actual_api_key_here
```

Get your API key from: https://platform.openai.com/api-keys

On first launch, if `.env` is missing or `OPENAI_API_KEY` is empty, the CLI prompts you to paste the key so the setup can continue without manual file edits.

## Usage

You can run commands directly through uv (no manual activation needed) or activate `.venv/` yourself.

```bash
# Full transcription pipeline (default mode)
uv run pi-transcription

# Force streaming without the wake word (debug mode)
uv run pi-transcription --force-always-on

# Explicitly require the wake word when FORCE_ALWAYS_ON=1 is set in the env
uv run pi-transcription --no-force-always-on

# Test WebSocket connection to OpenAI (requires API key)
uv run pi-transcription test-websocket

# Test audio capture from microphone (no API key needed)
uv run pi-transcription test-audio

# Legacy shim (still available if you prefer python scripts)
uv run python start.py --help
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

Defaults live in `config/defaults.toml` and are loaded by `pi_transcription.config`.
The TOML file documents the baseline runtime (24 kHz mono PCM, VAD thresholds,
noise reduction, etc.), while the module exposes strongly typed constants and
environment-variable overrides:

- `OPENAI_API_KEY` (required), `OPENAI_MODEL`, `OPENAI_REALTIME_ENDPOINT`
- Audio pipeline knobs: `SAMPLE_RATE`, `BUFFER_SIZE`, `CHANNELS`, `DTYPE`,
  `PREROLL_DURATION_SECONDS`, `AUDIO_QUEUE_MAX_SIZE`
- Auto-stop tuning: `AUTO_STOP_ENABLED`, `AUTO_STOP_SILENCE_THRESHOLD`,
  `AUTO_STOP_MAX_SILENCE_SECONDS`
- Wake-word overrides (see below): `WAKE_WORD_*`, `FORCE_ALWAYS_ON`

Update the TOML file for new defaults (which can be committed) and use env vars
for per-device overrides.

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
├── config/
│   └── defaults.toml        # Baseline runtime configuration
├── models/                  # Bundled openWakeWord assets
├── scripts/                 # Repo automation helpers
├── src/
│   └── pi_transcription/
│       ├── __init__.py
│       ├── audio/capture.py
│       ├── cli/app.py       # CLI + orchestration
│       ├── config/__init__.py
│       ├── diagnostics.py
│       ├── network/websocket_client.py
│       └── wake_word.py
├── tests/
│   ├── hey_jarvis.wav
│   ├── manual/
│   │   ├── test_audio_device.py
│   │   └── test_save_audio.py
│   └── test_wake_word.py
├── start.py                 # Compatibility shim for legacy entry point
├── README.md
├── pyproject.toml
├── uv.lock / ruff.toml / docs/wake-word.md / todos.md
└── .venv/ (managed by `uv sync`)
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
- Plan document: `docs/wake-word.md`
