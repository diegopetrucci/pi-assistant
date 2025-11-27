# CLI & Configuration Guide

This guide consolidates every supported command, flag, and environment variable for the `pi-assistant` CLI. Use it as the single source of truth for running the transcription pipeline, exercising diagnostics, and tuning configuration.

## Running The CLI

All commands assume you run through `uv` so the managed virtual environment is activated automatically.

```bash
# Wake-word gated transcription pipeline (defaults from config/defaults.toml + .env)
uv run pi-assistant

# Print wake-word scores, state transitions, and other diagnostics to the console
uv run pi-assistant -v

# Force assistant replies to stream directly from the Responses API
uv run pi-assistant --audio-mode responses

# Fetch text first, then synthesize locally over the Audio API
uv run pi-assistant --audio-mode local-tts

# Pin the model and reasoning effort for this run only
uv run pi-assistant --model 5.1 --reasoning-effort high

# Inject a default simulated query (from SIMULATED_QUERY_TEXT or fallback prompt)
uv run pi-assistant --simulate-query

# Inject a custom simulated question once at startup
uv run pi-assistant --simulate-query "Hey Rhasspy, what's the weather?"
```

### Execution Modes

`pi-assistant` exposes a positional `mode` argument (defaults to `run`):

- `run`: Launches the full capture → streaming → assistant loop.
- `test-audio`: Runs `src/pi_assistant/diagnostics.py::test_audio_capture` to validate microphone capture without touching the network.
- `test-websocket`: Connects to the OpenAI Realtime API, streams fake events through `handle_transcription_event`, and exits—useful for API/latency checks.

Invoke them with:

```bash
uv run pi-assistant test-audio
uv run pi-assistant test-websocket
```

The legacy script remains available for direct Python use: `uv run python start.py --help`.

## CLI Flags

| Flag | Description |
| --- | --- |
| `-v`, `--verbose` | Enables verbose logging (wake-word scores, state transitions, controller state). |
| `--model <preset-or-id>` | Overrides the assistant model for one run. Accepts presets defined in `ASSISTANT_MODEL_REGISTRY` (`nano`, `mini`, `4.1`, `5.1`, etc.) or full model IDs. |
| `--audio-mode {responses,local-tts}` | Chooses how assistant audio is delivered. Defaults to `responses` if `ASSISTANT_TTS_RESPONSES_ENABLED=1`, otherwise `local-tts`. |
| `--simulate-query [text]` | Injects a one-off transcript instead of waiting for speech. Pass custom text or omit to use the fallback `"Hey Rhasspy, is it going to rain tomorrow?"`. If unset, the CLI checks `SIMULATED_QUERY_TEXT`. |
| `--reasoning-effort {none,minimal,low,medium,high}` | Overrides the GPT-5 reasoning effort. Only values supported by the selected model are allowed; models without reasoning (e.g., GPT-4.1) reject this flag. `minimal` is rejected when `ASSISTANT_WEB_SEARCH_ENABLED=1`. |
| `--reset` | Clears saved onboarding selections (assistant model, reasoning effort, location) from `.env` and exits so the prompts reappear next time. |

## Assistant Models & Reasoning

- `pi_assistant.assistant.LLMResponder` reads the selected preset from `.env` (`ASSISTANT_MODEL`) or the CLI flag.
- Presets include:
- `nano` (default) → `gpt-5-nano-2025-08-07`
- `mini` → `gpt-5-mini-2025-08-07`
- `5.1` → `gpt-5.1-2025-11-13`
- `4.1` → `gpt-4.1-2025-04-14` (reasoning disabled; ignores `ASSISTANT_REASONING_EFFORT`)
- `nano` runs with `low`/`medium`/`high` reasoning only. `minimal` is disabled so web search (which this tier needs) keeps working without the CLI raising errors.
- Reasoning effort falls back to `ASSISTANT_REASONING_EFFORT` or auto when unset (and is ignored for presets without reasoning).
- Set `ASSISTANT_WEB_SEARCH_ENABLED=0` if you need `minimal` reasoning with streaming tool calls disabled.

## Assistant Audio Modes & TTS

Two delivery paths exist:

1. `responses` (default): Streams OpenAI Responses audio directly when `ASSISTANT_TTS_RESPONSES_ENABLED=1`. The CLI verifies support at startup and falls back automatically.
2. `local-tts`: Requests text first, then synthesizes locally using Audio API parameters (`ASSISTANT_TTS_MODEL`, `ASSISTANT_TTS_VOICE`, `ASSISTANT_TTS_FORMAT`, `ASSISTANT_TTS_SAMPLE_RATE`).

`SpeechPlayer` coalesces playback, handles stop commands, and resamples audio to `ASSISTANT_TTS_SAMPLE_RATE` when necessary. Use the confirmation cue knobs to preload ("Got it.") responses:

- `CONFIRMATION_CUE_ENABLED=1`
- `CONFIRMATION_CUE_TEXT="Got it."`

## Simulated Queries & Voice Controls

- `SIMULATED_QUERY_TEXT="What's the forecast today?"` injects the text every time the pipeline starts (versus the CLI flag which triggers once).
- Saying "Hey Jarvis stop" or "Jarvis stop" interrupts `SpeechPlayer`, clears the active turn, and returns to listening mode.
- `ASSISTANT_LANGUAGE` and `TRANSCRIPTION_LANGUAGE` force the assistant/transcriber to stay in a specific language even if the transcript drifts.

## Logging

Verbose logs write to `logs/<iso-timestamp>.log` for every invocation. Configure with:

- `VERBOSE_LOG_CAPTURE_ENABLED=0` to disable file logging.
- `VERBOSE_LOG_DIRECTORY=/path/to/logs` to redirect the folder.

Each entry mirrors console output without ANSI colors.

## Environment Configuration Reference

Default values live in `config/defaults.toml` and can be overridden with environment variables (usually via `.env`). Key groups:

### Core Requirements

- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL`, `OPENAI_REALTIME_ENDPOINT`
- `ASSISTANT_MODEL`, `ASSISTANT_REASONING_EFFORT`, `ASSISTANT_SYSTEM_PROMPT`
- `ASSISTANT_WEB_SEARCH_ENABLED` (1/0)

### Audio Pipeline

- `SAMPLE_RATE`, `STREAM_SAMPLE_RATE`, `BUFFER_SIZE`, `CHANNELS`, `DTYPE`
- `PREROLL_DURATION_SECONDS` (wake-word buffer length)
- `AUDIO_QUEUE_MAX_SIZE`, `AUDIO_INPUT_DEVICE`
- Auto-stop: `AUTO_STOP_ENABLED`, `AUTO_STOP_SILENCE_THRESHOLD`, `AUTO_STOP_MAX_SILENCE_SECONDS`,
  `SERVER_STOP_MIN_SILENCE_SECONDS` (local silence required before honoring server VAD stop events),
  `SERVER_STOP_TIMEOUT_SECONDS` (max time to wait for the server's `speech_stopped` ack before finalizing locally)

When hardware only exposes 44.1/48 kHz, the client now probes PortAudio, saves the detected sample rate to `.env`, and asks you to restart (look for `[INFO] … Saved SAMPLE_RATE to .env`). Leave `STREAM_SAMPLE_RATE=24000` so capture audio continues to be resampled for OpenAI. Manual overrides are still supported if you want to pin a specific sample rate.
### Assistant Delivery & Language

- `ASSISTANT_TTS_RESPONSES_ENABLED` (toggles streaming audio)
- `ASSISTANT_TTS_MODEL`, `ASSISTANT_TTS_VOICE`, `ASSISTANT_TTS_FORMAT`, `ASSISTANT_TTS_SAMPLE_RATE`
- `ASSISTANT_LANGUAGE`, `TRANSCRIPTION_LANGUAGE`
- `ASSISTANT_TTS_RESPONSES_ENABLED=0` pairs well with `--audio-mode local-tts`.

### Location & Context

- `LOCATION_NAME="London, UK"` sets the device context shared with the assistant.
- Run `uv run pi-assistant --reset` any time you want to clear the saved onboarding answers and pick the model/reasoning/location again.
- Update `config/defaults.toml` for repo-wide defaults; use `.env` for per-device overrides.

### Wake Word Settings

- `WAKE_WORD_NAME` (`alexa`, `hey_jarvis`, `hey_rhasspy`)
- `WAKE_WORD_MODEL_PATH`, `WAKE_WORD_MODEL_FALLBACK_PATH`
- `WAKE_WORD_MELSPEC_MODEL_PATH`, `WAKE_WORD_EMBEDDING_MODEL_PATH`
- `WAKE_WORD_SCORE_THRESHOLD`, `WAKE_WORD_CONSECUTIVE_FRAMES`
- `WAKE_WORD_TARGET_SAMPLE_RATE`

Bundled models live in `models/`. Detection events above the configured threshold log with `[WAKE]` tags, and state changes emit `[STATE]`.

### UX Helpers

- `SIMULATED_QUERY_TEXT`
- `CONFIRMATION_CUE_ENABLED`, `CONFIRMATION_CUE_TEXT`

## Diagnostics Reminders

- `uv run pi-assistant test-audio` confirms the microphone works (helpful when ALSA reports `Error querying device -1`).
- `uv run pi-assistant test-websocket` validates API access (`OPENAI_API_KEY` must be set).
- `arecord -l` lists ALSA devices on Raspberry Pi; use the reported card/index with `AUDIO_INPUT_DEVICE`.

For additional Raspberry Pi provisioning (PortAudio dependencies, cpu governor suggestions, etc.), see the main README.
