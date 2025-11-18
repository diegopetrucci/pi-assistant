# Potential Bugs

This document catalogs potential bugs and issues discovered through code analysis, organized by severity.

## CRITICAL SEVERITY

### 1. Assistant replies never fire when `--force-always-on` is used
- **Location:** `src/pi_assistant/cli/controller.py:209-299`
- **Impact:** The override skips wake-word gating but also disables every path that calls `_transition_stream_to_listening()`, so `schedule_turn_response()` is never invoked and no assistant response is produced.
- **Root Cause:** Auto-stop (line 380) and server `speech_stopped` events (line 288) are both guarded by `if not force_always_on`, leaving the controller stuck in `STREAMING` forever with no mechanism to finalize turns.
- **Fix Required:** Add alternative finalization mechanism for force-always-on mode, or remove the guard and rely on other state checks.

### 2. Race condition in concurrent access to `response_tasks` set
- **Location:** `src/pi_assistant/cli/controller.py:222, 229, 231-234`
- **Impact:** The `response_tasks` set is modified from both the main controller loop and task completion callbacks without lock protection. This can cause runtime errors or lost task references.
- **Root Cause:**
  - Line 229: `response_tasks.add(task)` in main loop
  - Line 232: `s.discard(fut)` in callback closure (runs in different context)
  - No synchronization primitive protects these concurrent modifications
- **Fix Required:** Add an `asyncio.Lock` to protect all modifications to `response_tasks`, or use a thread-safe collection.

### 3. WebSocket operations have no timeouts
- **Location:** `src/pi_assistant/network/websocket_client.py:135, 148`
- **Impact:** `await self.websocket.send()` and the async iteration in `receive_events()` can hang indefinitely if the connection stalls, causing the entire application to freeze.
- **Root Cause:** No timeout wrapper around WebSocket I/O operations.
- **Fix Required:** Wrap all WebSocket operations with `asyncio.wait_for()` with appropriate timeouts (e.g., 30 seconds).

### 4. PreRollBuffer size calculation missing channels multiplier
- **Location:** `src/pi_assistant/wake_word.py:48`
- **Impact:** Buffer size calculation is incorrect for multi-channel audio. Formula should be `max_seconds * sample_rate * sample_width * channels`, but currently omits channels.
- **Root Cause:** Line 48: `self.max_bytes = int(max_seconds * sample_rate * sample_width)` - missing `* channels` factor.
- **Current State:** Works correctly only because the system uses mono audio (CHANNELS=1), but would fail with stereo.
- **Fix Required:** Update formula to `int(max_seconds * sample_rate * sample_width * channels)` and add `channels` parameter to `__init__`.

## HIGH SEVERITY

### 5. Duplicate `item_id` handling removes legitimate transcript updates
- **Location:** `src/pi_assistant/assistant/transcript.py:53-60`
- **Impact:** OpenAI's Realtime API reuses an `item_id` when it sends updated completions. The current deduplication logic (line 53-57) drops every subsequent fragment with the same ID, so finalized turns frequently miss the final corrections.
- **Root Cause:** The `_seen_items` set only tracks item IDs, not revisions or sequence numbers.
- **Fix Required:** Either track revision numbers, or replace existing items instead of dropping updates, or use a timestamp-based approach to keep the most recent update.

### 6. Audio queue overflow silently drops frames
- **Location:** `src/pi_assistant/audio/capture.py:59-62`
- **Impact:** When the audio queue is full (max 100 items), incoming frames are silently dropped with only a stderr warning. Under heavy load or slow processing, this causes significant audio loss and poor transcription quality.
- **Root Cause:** Line 60-62: `put_nowait` in try/except that catches `QueueFull` and only prints a warning.
- **Potential Issues:**
  - No tracking of drop rate or consecutive drops
  - No backpressure mechanism to slow down the stream
  - No alert when drop rate is critically high
- **Fix Required:** Add drop tracking, consider dynamic buffer sizing, or implement flow control to pause/resume stream capture.

### 7. Wake word retriggering can bypass silence detection indefinitely
- **Location:** `src/pi_assistant/cli/controller.py:402-408`
- **Impact:** If `retrigger_budget > 0`, the silence detection completely resets (line 407-408), but this could keep the stream open indefinitely if the wake word keeps triggering due to background noise or repetitive sounds.
- **Root Cause:** No maximum retrigger limit or time-based cutoff.
- **Fix Required:** Add a maximum retrigger count or time window, after which the stream closes regardless.

### 8. Stop command substring matching triggers on unrelated phrases
- **Location:** `src/pi_assistant/cli/events.py:69`
- **Impact:** The code looks for `"jarvis stop"` as a substring (using `in` operator) inside the normalized transcript. Phrases like "Jarvis stoplight", "Jarvis stop sending alerts", or "don't stop" incorrectly clear the active turn and halt playback.
- **Root Cause:** Line 69: `if any(cmd in normalized for cmd in STOP_COMMANDS)` - substring matching instead of word boundary matching.
- **Fix Required:** Use word boundary regex matching (e.g., `\bhey jarvis stop\b`) or exact command matching.

### 9. No validation of WebSocket connection state before sending
- **Location:** `src/pi_assistant/cli/controller.py:360, 378`
- **Impact:** `ws_client.send_audio_chunk()` is called without checking if the WebSocket is still connected. If connection drops mid-stream, this causes exceptions that aren't gracefully handled.
- **Root Cause:**
  - Controller assumes WebSocket stays connected throughout operation
  - WebSocketClient.send_audio_chunk() (line 125-135) only checks `self.connected` flag which may be stale
- **Fix Required:** Add connection health checks before sending, and implement reconnection logic.

### 10. Missing cleanup of simulated query task on shutdown
- **Location:** `src/pi_assistant/cli/app.py:175-185`
- **Impact:** The `simulated_query_task` is added to the `pending` list but isn't explicitly cancelled during shutdown. If the task is still running when KeyboardInterrupt occurs, it may not terminate cleanly.
- **Root Cause:** Lines 181-185 add the task to pending for `gather()`, but there's no explicit cancellation in the `finally` block like there should be.
- **Fix Required:** Add explicit task cancellation in the `finally` block (lines 194-204) for all created tasks.

## MEDIUM SEVERITY

### 11. Unbounded phrase audio cache (memory leak)
- **Location:** `src/pi_assistant/assistant/llm.py:72`
- **Impact:** The `_phrase_audio_cache` dictionary grows unbounded as new phrases are cached. Over long-running sessions, this can consume significant memory.
- **Root Cause:** No eviction policy or maximum size limit on the cache.
- **Fix Required:** Implement LRU cache with maximum size limit, or use `functools.lru_cache` wrapper.

### 12. Linear resampler state not thread-safe
- **Location:** `src/pi_assistant/audio/resampler.py:24, 36-43`
- **Impact:** The `self._state` variable is mutated during `process()` (line 36). If the resampler is called from multiple threads concurrently, this causes race conditions and corrupted audio.
- **Current State:** Appears safe in current usage (separate resampler instances per use case), but dangerous if code is refactored.
- **Fix Required:** Document thread-safety requirements clearly, or add locking if multi-threaded use is intended.

### 13. Speech player missing validation for None sample rate
- **Location:** `src/pi_assistant/audio/playback.py:26-27, 36, 52-53`
- **Impact:** Multiple fallbacks exist (`sample_rate or self._default_sample_rate`), but if both are None/0, the code continues with invalid values, leading to errors in sounddevice or resampler.
- **Root Cause:** No assertion or explicit validation that final sample rate is valid before use.
- **Fix Required:** Add validation in `_play_blocking()` to raise a clear error if sample rate is invalid.

### 14. Confirmation cue errors not fully logged
- **Location:** `src/pi_assistant/cli/controller.py:85-91`
- **Impact:** Exception in confirmation cue playback is caught but only logged via `verbose_print`, which may be disabled. Full exception details and stack traces are lost.
- **Root Cause:** Line 88: `except Exception as exc` catches broadly but only logs `{exc}` without traceback.
- **Fix Required:** Use proper exception logging with `logging.exception()` or at minimum log the full traceback.

### 15. No validation of audio chunk size before calculations
- **Location:** `src/pi_assistant/cli/controller.py:381-382`
- **Impact:** The code calculates `frames = len(audio_bytes) / (2 * CHANNELS)` assuming the chunk size is always correct. Malformed chunks or partial data could cause incorrect duration calculations.
- **Root Cause:** No validation that `len(audio_bytes)` is a multiple of `2 * CHANNELS`.
- **Fix Required:** Add validation: `if len(audio_bytes) % (2 * CHANNELS) != 0: handle_error()`.

### 16. Race condition in speech_stopped_signal handling
- **Location:** `src/pi_assistant/cli/controller.py:288-298`
- **Impact:** Between checking `speech_stopped_signal.is_set()` (line 288) and clearing it (line 289), another signal could theoretically arrive, though unlikely with current event loop structure.
- **Root Cause:** Check-and-clear is not atomic.
- **Fix Required:** Use a lock or implement atomic check-and-clear pattern.

### 17. No maximum transcript finalize wait guarantee
- **Location:** `src/pi_assistant/assistant/transcript.py:113-124`
- **Impact:** The `while True` loop theoretically could run forever if `finalized` is never True, though the timeout logic makes this extremely unlikely.
- **Root Cause:** No hard upper bound on iterations.
- **Fix Required:** Add maximum iteration count as a safety measure.

### 18. Response task cancellation without awaiting completion
- **Location:** `src/pi_assistant/cli/controller.py:237-245`
- **Impact:** When cancelling response tasks (line 245), the code doesn't await their cancellation, so they may still be running when the next operation starts.
- **Root Cause:** `pending.cancel()` is called but not awaited.
- **Fix Required:** Add `await asyncio.gather(*response_tasks, return_exceptions=True)` after cancellation.

## LOW SEVERITY

### 19. WebSocket session timeout handling suboptimal
- **Location:** `src/pi_assistant/network/websocket_client.py:44-52`
- **Impact:** If `wait_for_session_created()` times out after 10 seconds, the code continues with a warning (line 49-50) but the session may not be properly initialized.
- **Root Cause:** Optimistic continuation after timeout rather than failing fast.
- **Fix Required:** Either retry the connection or raise an error instead of continuing.

### 20. No bounds checking on audio format assumptions
- **Location:** `src/pi_assistant/cli/controller.py:53-64`
- **Impact:** `calculate_rms()` assumes PCM16 format but doesn't validate. If audio format changes, this silently produces wrong results.
- **Root Cause:** Implicit format assumption in `np.frombuffer(audio_bytes, dtype=np.int16)`.
- **Fix Required:** Add format validation or make dtype configurable.

### 21. Potential division by zero in chunk duration calculation
- **Location:** `src/pi_assistant/cli/controller.py:382`
- **Impact:** If `SAMPLE_RATE` is 0 (due to misconfiguration), the division `frames / SAMPLE_RATE` raises `ZeroDivisionError`.
- **Root Cause:** No validation of configuration constants at startup.
- **Fix Required:** Add configuration validation during initialization.

### 22. Type coercion issues in audio device selection
- **Location:** `src/pi_assistant/audio/capture.py:153-160`
- **Impact:** `_coerce_input_index()` returns None for non-int devices, but the subsequent logic may not handle None correctly in all code paths.
- **Root Cause:** Implicit assumption that None means "use default", but not consistently enforced.
- **Fix Required:** Make None handling explicit with clear documentation.

### 23. Missing error context in event receiver exception handling
- **Location:** `src/pi_assistant/cli/events.py:111-113`
- **Impact:** Generic exception catch (line 111) logs the error but loses context about which event caused the failure.
- **Root Cause:** No event context in error message.
- **Fix Required:** Include event type and partial event data in error log.

### 24. Awaiting server stop without retry mechanism
- **Location:** `src/pi_assistant/cli/controller.py:263-267`
- **Impact:** If the server never sends the speech_stopped event (network issue, server bug), the system waits indefinitely in `awaiting_server_stop` state.
- **Root Cause:** No timeout on deferred finalization.
- **Fix Required:** Add a timeout (e.g., 5 seconds) after which finalization proceeds anyway.

## RECOMMENDATIONS

### Code Quality Improvements
1. Add comprehensive input validation for all configuration values at startup
2. Implement proper logging with levels (ERROR, WARNING, INFO, DEBUG) instead of mixing print/verbose_print
3. Add type hints to all function parameters and return values
4. Consider adding telemetry/metrics for dropped frames, queue sizes, and error rates
5. Implement proper resource cleanup patterns (context managers) for audio streams and WebSocket connections

### Architecture Improvements
1. Add health check mechanism for WebSocket connection with automatic reconnection
2. Implement backpressure mechanism for audio queue to prevent frame drops
3. Consider adding a state machine validator to catch invalid state transitions
4. Add timeout configurations for all async operations
5. Implement circuit breaker pattern for OpenAI API calls

### Testing Improvements
1. Add stress tests for audio queue overflow scenarios
2. Add tests for WebSocket disconnection/reconnection scenarios
3. Add tests for all edge cases in wake word detection (retriggers, rapid triggers)
4. Add tests for concurrent task cancellation scenarios
5. Add integration tests for force-always-on mode

### Documentation Improvements
1. Document thread-safety requirements for all shared state
2. Document expected behavior for all error conditions
3. Add sequence diagrams for main workflows
4. Document performance characteristics and resource limits
5. Add troubleshooting guide for common failure modes
