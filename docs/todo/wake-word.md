# Wake Word Integration Plan

## Objectives
- Gate streaming to OpenAI until the selected wake phrase fires.
- Preserve the first words after the trigger via a short pre-roll buffer.
- Keep the existing transcription UX (server VAD, auto-stop) once streaming begins.

## Requirements (locked)
1. **Wake phrase**: Support “Alexa”, “Hey Jarvis”, or “Hey Rhasspy” and allow the user to pick the active phrase.
2. **Pre-roll**: prepend ~1s of buffered audio captured before the trigger to the first transmitted chunk.
3. **Stream lifetime**: continue streaming until the current silence detector resets the turn.
4. **Confidence guard**: use openWakeWord’s pre-trained models (defaulting to “Hey Rhasspy”) and treat a detection as valid when score ≥0.5 for two consecutive frames; expose both numbers in `config.py` for tuning.
5. **Resource usage**: keep the capture pipeline at 24 kHz for OpenAI, branch a copy that is downsampled to 16 kHz for the wake-word detector so transcription quality stays untouched.
6. **Fallback**: add a debug-only override (CLI flag or env var) that bypasses the wake word when troubleshooting.
7. **Observability/tests**: log state transitions (listening → triggered → streaming → idle) and add an automated test that replays a “hey jarvis” WAV through the detector to prevent regressions.

## Architecture

### Audio capture & branching
- `AudioCapture` continues producing 24 kHz mono PCM16 chunks.
- A new `WakeWordEngine` consumes the same chunks through an asyncio fan-out:
  - One branch feeds the existing websocket pipeline (buffered) but stays paused until the engine unlocks it.
  - The other branch down-samples to 16 kHz and feeds openWakeWord.

### Detection pipeline
- Add `openwakeword` to dependencies and ship the bundled “hey_jarvis_v1.0.tflite` model (or similar) inside `models/`.
- Load the model once at startup; keep inference in a background task that receives audio slices (~0.5s overlap) to minimize latency.
- Apply the 0.5 score threshold plus the “two consecutive hits” rule before signaling a trigger.

### Pre-roll buffer
- Maintain a ring buffer (deque) that stores the last N milliseconds of 24 kHz PCM16 audio.
- On trigger, flush the buffer plus the latest chunks into the websocket queue so the transcript includes speech immediately after the wake phrase.

### State machine
- States: `LISTENING` → `TRIGGERED` (fire when wake word confirmed) → `STREAMING`.
- While streaming, forward all new audio chunks to OpenAI until both:
  - Server-side VAD (`AUTO_STOP_*` logic) reports silence, and
  - No wake word retrigger occurs.
- Transition back to `LISTENING` and resume wake-word-only mode.

### Debug override
- `--force-always-on` CLI flag or `FORCE_ALWAYS_ON=1` env var.
- When enabled, bypass the wake-word gate but keep logging that the override is active for clarity.

### Observability & testing
- Log every detection score ≥ threshold, accepted triggers, state transitions, and manual overrides.
- Add `tests/test_wake_word.py` that streams a known WAV containing “hey jarvis” to the detector and asserts the trigger fires once.

## Implementation Steps
1. **Dependencies & assets**
   - Add `openwakeword` to `pyproject.toml`, download/commit the Alexa, “hey jarvis”, and “hey rhasspy” models (if license permits) under `models/`.
2. **WakeWordEngine module**
   - Handles downsampling, buffering, scoring, and threshold logic.
   - Expose async methods: `submit_chunk(bytes)` and `async for detection`.
3. **Pre-roll buffer & gating**
   - Introduce an `AudioRouter` or extend `AudioCapture` consumer logic to keep a rolling buffer, flush on trigger, and queue audio for OpenAI once streaming.
4. **State management**
   - Embed a small controller in `start.py` that coordinates the wake-word engine, buffering, and websocket streaming tasks.
5. **Debug override & config**
   - Add CLI/env plumbing plus new constants in `config.py` for thresholds and buffer durations.
6. **Tests & scripts**
   - Create a fixture WAV (or reuse `tests/manual/test_recording.wav` if it contains the phrase), build `tests/test_wake_word.py`, and document a manual verification script in `README.md`.
7. **Docs & logging**
   - Update README usage section (wake-word mode, override flag) and ensure logs clearly indicate detection activity.

## Risks / Mitigations
- **False positives**: expose thresholds in config and log raw scores to tune; add optional “N detections within M ms” rule if needed.
- **Latency from downsampling**: keep chunks short (~100 ms) and reuse numpy/SciPy for efficient resampling.
- **Model asset size**: confirm openWakeWord license permits bundling; otherwise download on first run and cache locally.
