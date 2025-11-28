# Starting Over Notes

Quick capture of key architecture pivots we would pursue if rebuilding the project from a clean slate.

## 1. Centralized Settings Object
- Current modules freeze `.env` defaults at import time (see `src/pi_assistant/assistant/transcription_session.py` and `src/pi_assistant/audio/capture.py`), which blocks dynamic overrides and complicates testing.
- Rebuild around a typed `Settings` container that loads `config/defaults.toml` + environment overrides once, then inject it into components so unit tests and future hot-reload UX can swap configurations without monkey-patching globals.

## 2. Event-Driven Controller ✅
- `_AudioControllerLoop` still mixes wake-word inference, silence tracking, WebSocket flow control, and assistant orchestration in a single class.
- Introduce an explicit event bus with small async actors (wake engine, speech gate, stream uploader, assistant responder). This keeps state transitions testable, enables replay tooling, and makes it easier to split functionality across devices later.
- Status: implemented on the `event-driven-controller` branch with the new controller event bus, actor set, and unit-test coverage (`tests/test_cli_controller_event_bus.py`, `tests/test_cli_controller_actors.py`).

## 3. Service-Oriented Session Lifecycle ✅
- `TranscriptionSession` owns cue warming, Responses audio probing, WebSocket start, and teardown logic, so any new transport re-opens that class.
- Move to a supervisor that treats capture, websocket, assistant, and diagnostics as services with `start/stop` hooks and readiness events. That yields granular restarts and simpler plug-in support for future transports (MQTT, on-device ASR).

## 4. Pluggable Audio Drivers
- The audio layer assumes `sounddevice`/PortAudio everywhere and tests rely on heavy patching to fake hardware.
- Define a `MicrophoneDriver` interface (USB mic, ALSA, CoreAudio, prerecorded WAV) plus a fixture-backed implementation so wake-word scoring, resampling, and buffering can be fuzzed offline and swapped for different boards without touching controller code.

## 5. Structured Observability
- Logging is ANSI-focused and event handling inspects raw dicts coming off the Realtime stream.
- Start with structured logging (JSON or dict-to-stdout) and typed event models so we can emit metrics, compute latency histograms, and replay timelines for regression analysis or UI mirroring.
