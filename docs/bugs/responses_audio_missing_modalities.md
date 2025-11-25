# Responses audio requests never succeed

**Location:** `src/pi_assistant/assistant/llm.py:139-154`

**Problem:** When TTS is enabled, `LLMResponder.generate_reply` adds an `audio` block to the Responses API payload but never sets the `modalities` field to include `"audio"`. The OpenAI Responses API rejects any request that asks for audio without declaring it in `modalities`, so the code always hits the `BadRequestError` path and immediately falls back to text-only processing.

**Impact:** Even if the org has Responses audio enabled, the Pi assistant can never stream audio directly from the Responses API because every request is malformed. Users are forced onto the slower two-call fallback (Responses for text + Audio API for TTS) and the startup probe will always log that Responses audio is unavailable.

**How to reproduce:**
1. Launch `pi-assistant` with `ASSISTANT_TTS_ENABLED=1` and `ASSISTANT_TTS_RESPONSES_ENABLED=1`.
2. Observe the console warning: `Responses API does not accept audio parameters yet; falling back to text-only responses`.
3. Inspect network traffic (or instrument `_send_response_request`) to see the 400 response complaining about `audio`/`modalities`.

**Suggested fix:** Whenever `_build_audio_extra_body()` returns data, add `request_kwargs["modalities"] = ["text", "audio"]` so the Responses API knows to return audio chunks. Keep the fallback logic for orgs that still lack the feature flag.
