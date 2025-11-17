# No-Turn Debug Notes

Investigating cases where wake and transcript logs appear but no turn is finalized.

- **Turn drains before transcript arrives**  
  `finalize_turn()` currently waits only 350 ms before clearing buffered segments (`src/pi_transcription/assistant.py`). When `input_audio_buffer.speech_stopped` fires and `_transition_stream_to_listening()` runs, the turn ends immediately. If the corresponding `conversation.item.input_audio_transcription.completed` event arrives after the drain window, the aggregator is already idle and drops the transcript.

- **Stream never returns to LISTENING**  
  The assistant response is scheduled only when `_transition_stream_to_listening()` executes (`src/pi_transcription/cli/controller.py`). If the server never sends `input_audio_buffer.speech_stopped`, or wake retriggers keep `retrigger_budget > 0`, auto-stop logic refuses to close the stream. Transcripts appear in the log, but no turn finalizes because the controller stays in STREAMING.

- **Stop-command heuristic flushes the buffer**  
  `maybe_stop_playback()` normalizes each completed transcript and matches `("hey jarvis stop", "jarvis stop")` (`src/pi_transcription/cli/events.py`). Any utterance that contains those words clears the current turn buffer and signals the controller to halt playback before `finalize_turn()` runs.

- **Duplicate item IDs suppress segments**  
  `TurnTranscriptAggregator.append_transcript()` ignores transcripts whose `item_id` was already processed. If the realtime service reuses the same `item_id` across multiple completions (seen during wake retriggers), only the first fragment is kept, leaving the finalized turn empty.

**Instrumentation:** timestamped debug prints now wrap `append_transcript()` and `finalize_turn()` so we can correlate item IDs, buffered segment counts, and wait times per device run.
