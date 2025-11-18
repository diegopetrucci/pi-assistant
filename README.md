# Raspberry Pi Audio Transcription System

[![CI](https://github.com/diegopetrucci/pi-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/diegopetrucci/pi-assistant/actions/workflows/tests.yml)
[![Codecov](https://codecov.io/gh/diegopetrucci/pi-assistant/branch/main/graph/badge.svg)](https://codecov.io/gh/diegopetrucci/pi-assistant)

Real-time speech-to-text transcription system for Raspberry Pi 5 that streams audio from a USB microphone to OpenAI's Realtime API, forwards finalized turns to a GPT-5-based assistant, and can speak or print the reply in the terminal.

## Features

- Real-time audio capture from USB microphone with wake-word gating, pre-roll, and silence auto-stop.
- Streams to OpenAI Realtime API for transcription, then routes each completed turn to GPT-5 Mini or GPT-5.1 with adjustable reasoning effort.
- Assistant replies can stream directly from the Responses API or fall back to local Audio API TTS with automatic sample-rate selection.
- Optional web-search tool calls, fixed system prompt, and location/language hints so the assistant stays on-topic.
- Voice stop commands (e.g., "Hey Jarvis stop") interrupt playback and clear the pending turn.
- Verbose logging captures wake-word scores, state transitions, and all console output in timestamped files under `logs/`.
- Optimized for Raspberry Pi 5, 24 kHz mono PCM audio throughout the capture pipeline.

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

See `docs/cli.md` for the complete CLI command and configuration guide, including execution modes, diagnostics, flags, environment variables, and wake-word tuning. The quick start remains:

```bash
uv run pi-assistant
```

Say "Hey Jarvis stop" (or "Jarvis stop") while the assistant is talking to immediately halt playback, clear the pending turn, and return to listening mode. Follow the CLI guide for verbose logging, simulated queries, assistant model overrides, and diagnostics such as `test-audio` and `test-websocket`.

### Assistant Models & Reasoning

`pi_assistant.assistant.LLMResponder` wraps the OpenAI Responses API and supports multiple presets. The CLI prompts for a default model/reasoning pair on first launch and stores the choice in `.env`, but you can override it per run via the flags above or by setting `ASSISTANT_MODEL` / `ASSISTANT_REASONING_EFFORT`.

| Preset | Model ID | Recommended use |
| --- | --- | --- |
| `mini` (default) | `gpt-5-mini-2025-08-07` | Low-latency replies with optional `minimal`, `low`, `medium`, or `high` reasoning. |
| `5.1` | `gpt-5.1-2025-11-13` | Higher accuracy with `none`, `low`, `medium`, or `high` reasoning. |

- `ASSISTANT_REASONING_EFFORT` falls back to `low` when unset; `minimal` cannot be used while `ASSISTANT_WEB_SEARCH_ENABLED=1`.
- `ASSISTANT_SYSTEM_PROMPT`, `LOCATION_NAME`, `ASSISTANT_LANGUAGE`, and `TRANSCRIPTION_LANGUAGE` are sent as system messages so you can keep the assistant short, localized, and aware of the device's location.
- Set `ASSISTANT_WEB_SEARCH_ENABLED=0` to disable tool calls entirely or leave it enabled to let GPT-5 issue web search requests when the Responses API determines they are useful.

### Assistant Audio Modes & TTS

Two delivery paths are supported:

- `responses` (default): stream assistant audio directly from the Responses API when `ASSISTANT_TTS_RESPONSES_ENABLED=1`. The CLI automatically verifies whether the selected model supports the feature and falls back if the server rejects it.
- `local-tts`: request text first, then synthesize locally via the Audio API using `ASSISTANT_TTS_MODEL`, `ASSISTANT_TTS_VOICE`, `ASSISTANT_TTS_FORMAT`, and `ASSISTANT_TTS_SAMPLE_RATE`.

`SpeechPlayer` handles sample-rate mismatches and exposes a stop hook so voice commands can interrupt playback. The optional confirmation cue ("Got it.") is controlled via `CONFIRMATION_CUE_ENABLED` and `CONFIRMATION_CUE_TEXT`; the phrase is pre-rendered and cached so the tone plays instantly after each wake-word trigger.

### Simulated Queries & Voice Controls

- Set `SIMULATED_QUERY_TEXT="What's the forecast today?"` in `.env` to auto-inject a prompt every time the pipeline starts, or use `--simulate-query` to test a single turn without speaking.
- Voice stop commands ("Hey Jarvis stop" / "Jarvis stop") signal the `SpeechPlayer` to halt audio and set `AUDIO_INPUT_DEVICE`-free capture back to listening mode with the current turn discarded.

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

Before sending a pull request, run `uv run pyright && uv run pytest` so CI sees the same status you validated locally.

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

Defaults live in `config/defaults.toml` and are surfaced via `pi_assistant.config`. Detailed environment-variable descriptions (assistant tuning, wake-word overrides, simulated queries, logging knobs, etc.) are documented in `docs/cli.md`.

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
│       ├── assistant/
│       │   ├── llm.py             # Responses API + TTS orchestration
│       │   └── transcript.py      # Turn-level transcript aggregation
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

**`Microphone <name> does not support SAMPLE_RATE=24000 Hz`:**
- Your USB mic only exposes 48 kHz (or similar). Set `SAMPLE_RATE` to the hinted value from the error and keep `STREAM_SAMPLE_RATE=24000` so the client resamples before streaming.
- Example for `.env`:
  ```bash
  echo "SAMPLE_RATE=48000" >> .env
  ```
- Re-run `uv run pi-assistant -v` to confirm audio capture succeeds.

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
