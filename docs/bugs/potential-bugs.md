# Potential Bugs

## 1. Assistant replies never fire when `--force-always-on` is used
- **Location:** `src/pi_transcription/cli/controller.py:169-299`
- **Impact:** The override skips wake-word gating but also disables every path that calls `_transition_stream_to_listening()`, so `schedule_turn_response()` is never invoked and no assistant response is produced.
- **Notes:** Auto-stop and server `speech_stopped` events are both guarded by `if not force_always_on`, leaving the controller stuck in `STREAMING` forever.

## 2. Duplicate `item_id` handling removes legitimate transcript updates
- **Location:** `src/pi_transcription/assistant.py:65-77`
- **Impact:** OpenAI’s realtime API reuses an `item_id` when it re-sends an updated completion. The current dedupe logic drops every subsequent fragment with the same ID, so finalized turns frequently miss the final corrections.
- **Notes:** Need to differentiate by revision (e.g., `item_revision` field) or simply keep the most recent payload.

## 3. Stop-command detection triggers on unrelated phrases
- **Location:** `src/pi_transcription/cli/events.py:65-73`
- **Impact:** The code looks for `"jarvis stop"` as a substring inside the normalized transcript. Phrases like “Jarvis stoplight” or “Jarvis stop sending alerts” incorrectly clear the active turn and halt playback.
- **Notes:** Should match whole words or require an exact command instead of a substring.
