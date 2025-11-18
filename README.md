# Raspberry Pi Audio Transcription System

[![CI](https://github.com/diegopetrucci/pi-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/diegopetrucci/pi-assistant/actions/workflows/tests.yml)
[![Codecov](https://codecov.io/gh/diegopetrucci/pi-assistant/branch/main/graph/badge.svg)](https://codecov.io/gh/diegopetrucci/pi-assistant)

Real-time speech-to-text transcription system for Raspberry Pi 5 that streams audio from a USB microphone to OpenAI's Realtime API and displays transcribed text in the terminal.

## Features

- Real-time audio capture from USB microphone
- Streams to OpenAI Realtime API for transcription
- Server-side Voice Activity Detection (VAD)
- Optional auto-stop logging after sustained silence
- Wake-word gated streaming with bundled openWakeWord models (Alexa, Hey Jarvis, Hey Rhasspy) plus pre-roll
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
cd /path/to/pi-assistant
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

The script performs an `apt-get update` and installs `libportaudio2`, `libportaudiocpp0`, and `portaudio19-dev` using `sudo`. If you're on a distro without `apt-get`, install the equivalent PortAudio development packages via your package manager before running `uv run pi-assistant`.

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
uv run pi-assistant

# Force responses audio streaming (default)
uv run pi-assistant --assistant-audio-mode responses

# Fetch text first, then synthesize locally via the Audio API
uv run pi-assistant --assistant-audio-mode local-tts

# Inject a default simulated question once at startup (for silent testing)
uv run pi-assistant --simulate-query

# Inject a custom simulated question
uv run pi-assistant --simulate-query "Hey Rhasspy, what's the weather?"

# Test WebSocket connection to OpenAI (requires API key)
uv run pi-assistant test-websocket

# Test audio capture from microphone (no API key needed)
uv run pi-assistant test-audio

# Legacy shim (still available if you prefer python scripts)
uv run python start.py --help
```

### Capturing Verbose Logs Locally

Verbose logs are captured by default. Each `uv run pi-assistant` session:

- Writes a log file named with an ISO-8601 timestamp (e.g., `logs/2024-11-30T14-03-12.123.log`) under the `logs/` directory.
- Mirrors console output timestamps in the log.
- Strips ANSI colors for readability.

You can override the log folder with `VERBOSE_LOG_DIRECTORY=/path/to/dir`, or disable capture entirely via `VERBOSE_LOG_CAPTURE_ENABLED=0` to conserve space on constrained devices.
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

These commands automatically exclude generated artifacts such as `.venv/` and `tests/manual/test_recording.wav`.

Pyright enforces static typing using `pyrightconfig.json`, mirroring the same exclusions.

```bash
# Run a full type-checking pass
uv run pyright

# Keep Pyright running in watch mode while editing
uv run pyright -- --watch
```

## Editor Integration

If you use VS Code, install the recommended Pylance extension (added via `.vscode/extensions.json`) so you get fast type checking, inline docstrings, and completion hints that mirror the repo's configuration.

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

Defaults live in `config/defaults.toml` and are loaded by `pi_assistant.config`.
The TOML file documents the baseline runtime (24 kHz mono PCM, VAD thresholds,
noise reduction, etc.), while the module exposes strongly typed constants and
environment-variable overrides:

- `OPENAI_API_KEY` (required), `OPENAI_MODEL`, `OPENAI_REALTIME_ENDPOINT`
- Audio pipeline knobs: `SAMPLE_RATE`, `STREAM_SAMPLE_RATE`, `BUFFER_SIZE`,
  `CHANNELS`, `DTYPE`, `PREROLL_DURATION_SECONDS`, `AUDIO_QUEUE_MAX_SIZE`,
  `AUDIO_INPUT_DEVICE`
- Auto-stop tuning: `AUTO_STOP_ENABLED`, `AUTO_STOP_SILENCE_THRESHOLD`,
  `AUTO_STOP_MAX_SILENCE_SECONDS`
- Assistant replies: `ASSISTANT_MODEL`, `ASSISTANT_SYSTEM_PROMPT`,
  `ASSISTANT_WEB_SEARCH_ENABLED`, `ASSISTANT_TTS_*`
- Language locks: `TRANSCRIPTION_LANGUAGE` (forced speech-recognition hint) and
  `ASSISTANT_LANGUAGE` (LLM output language), both defaulting to `"en"`
- Wake-word overrides (see below): `WAKE_WORD_*`
- Verbose logging capture (enabled by default): `VERBOSE_LOG_CAPTURE_ENABLED`, `VERBOSE_LOG_DIRECTORY`

If you want the assistant to know its real-world context, set
`LOCATION_NAME="London, UK"` (or edit `device.location_name` in `config/defaults.toml`).
The value is sent as a system message (`Device location: London, UK`) alongside any
custom assistant prompt, and the CLI will interactively ask for it the first time it
runs if neither the env var nor the defaults provide one.

Update the TOML file for new defaults (which can be committed) and use env vars
for per-device overrides.

Set `ASSISTANT_TTS_RESPONSES_ENABLED=0` (or launch with `--assistant-audio-mode local-tts`)
to prefer the text+local-TTS round-trip when you want to compare perceived latency against
the streaming Responses audio path. Re-enable streaming with
`ASSISTANT_TTS_RESPONSES_ENABLED=1` or `--assistant-audio-mode responses`.

If you need to test without speaking, set `SIMULATED_QUERY_TEXT="Hey Rhasspy, is it going to rain today?"`
in `.env` (or pass `--simulate-query` / `--simulate-query "custom prompt"`). The CLI injects the text once
per `uv run pi-assistant` invocation and plays the resulting reply through the normal TTS path.

To keep every interaction in English (or another fixed language), set both
`TRANSCRIPTION_LANGUAGE` and `ASSISTANT_LANGUAGE` inside `.env`. The first value
is sent with the OpenAI Realtime session update so the recognizer never switches
languages mid-stream, while the second is injected into a system instruction so
assistant replies remain in that language even when the transcript briefly drifts.

When a USB microphone only exposes 48 kHz (or any rate different from the
OpenAI session’s 24 kHz expectation), set `SAMPLE_RATE` to the hardware-supported
value and leave `STREAM_SAMPLE_RATE` at the default 24 kHz. The controller will
resample buffered and live audio before sending it to OpenAI, so wake-word
detection and transcription stay in sync.

To persist the override in this repo, append it to `.env` once:

```bash
cd pi-assistant
echo "SAMPLE_RATE=48000" >> .env
```

### Wake Word Settings

Wake-word gating is enabled by default. On the first run the CLI prompts you to pick a wake
phrase (Alexa, Hey Jarvis, or Hey Rhasspy) and saves the choice to `.env` as `WAKE_WORD_NAME`
(Hey Rhasspy is the default if you just press Enter).
You can change it later by editing `.env` or exporting a new value before launching the CLI.

- `WAKE_WORD_NAME`: Canonical key for the selected phrase (`alexa`, `hey_jarvis`, or `hey_rhasspy`).
- `WAKE_WORD_MODEL_PATH` / `WAKE_WORD_MODEL_FALLBACK_PATH`: Default to the `.onnx` / `.tflite`
  assets for the chosen phrase under `models/`, but you can override them to point at a custom model.
- `WAKE_WORD_MELSPEC_MODEL_PATH` / `WAKE_WORD_EMBEDDING_MODEL_PATH`: Feature-extractor assets
  (`melspectrogram.onnx` and `embedding_model.onnx`) bundled under `models/` so we can run
  openWakeWord without its download helper.
- `WAKE_WORD_SCORE_THRESHOLD` / `WAKE_WORD_CONSECUTIVE_FRAMES`: Confidence guard (default: score ≥ 0.5 for two consecutive frames).
- `PREROLL_DURATION_SECONDS`: Length of buffered audio (default: 1 second) that is prepended to the first streamed chunk after activation.
- `WAKE_WORD_TARGET_SAMPLE_RATE`: Leave at 16 kHz unless you re-train a model that expects a different input rate.

Every detection above the configured threshold is logged with the `[WAKE]` label, and state transitions are reported with `[STATE]`. When the wake word fires, the controller flushes the pre-roll buffer plus live audio so the transcript includes the first spoken words after the selected phrase.

## Project Structure

```
pi-assistant/
├── config/
│   └── defaults.toml        # Baseline runtime configuration
├── models/                  # Bundled openWakeWord assets
├── scripts/                 # Repo automation helpers
├── src/
│   └── pi_assistant/
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

The `tests/test_wake_word.py` regression uses the generated `tests/hey_jarvis.wav` fixture (and explicitly loads the Jarvis model) to ensure the detector fires exactly once regardless of which wake word the CLI currently selects. Run it with:

```bash
uv run python -m unittest tests/test_wake_word.py
```

> **Note:** The wake-word test is skipped automatically when `openwakeword` (and its runtimes) are unavailable.

### Running Tests

You can run the full test suite through uv (no manual activation required):

```bash
uv run pytest
```

Add `-v` for verbose output:

```bash
uv run pytest -v
```

To generate coverage reports (powered by `pytest-cov`):

```bash
uv run pytest --cov
```

Async tests rely on `pytest-asyncio`; no extra setup is needed when using `uv sync --group dev`.

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

**`Error querying device -1`:**
- Run `uv run pi-assistant test-audio` to confirm sounddevice can capture samples.
- Use `arecord -l` (or `sd.query_devices()` in Python) to note the correct ALSA card/index.
- Export `AUDIO_INPUT_DEVICE=<index-or-name>` so the client selects the right microphone.

**Licensing for bundled wake-word models:**
- `models/alexa_v0.1.(onnx|tflite)`, `models/hey_jarvis_v0.1.(onnx|tflite)`, `models/hey_rhasspy_v0.1.(onnx|tflite)`, `models/melspectrogram.onnx`, and `models/embedding_model.onnx` are distributed under the Apache 2.0 license from the [openWakeWord](https://github.com/dscripka/openWakeWord) project.

**Microphone permission (macOS):**
System Settings → Privacy & Security → Microphone

## License

This project is for educational and development purposes.

## Support

For issues and questions, refer to:
- OpenAI Realtime API docs: https://platform.openai.com/docs/guides/realtime
- Plan document: `docs/wake-word.md`
