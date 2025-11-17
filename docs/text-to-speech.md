# Text-to-Speech Options

The microphone → Realtime transcription flow already works; the missing piece is turning the LLM
reply back into speech. Below are the main ways we can add TTS, ordered roughly by how much code
change is required.

## Option 1 – Second OpenAI Request (Responses → Audio API)

- Keep `LLMResponder` as-is for text. After `finalize_turn_and_respond` prints the reply, call
  `AsyncOpenAI().audio.speech.create(model="gpt-4o-mini-tts", voice=..., input=reply)` and play the
  returned PCM via a tiny helper that feeds `sounddevice.OutputStream`.
- Pros: minimal code churn, no new dependencies (we already ship `openai`), easy to prototype.
- Cons: adds a second network round-trip per turn, no streaming; latency is text-response time plus
  TTS-generation time.

## Option 2 – Single Responses Call With Audio Modality

- Teach `LLMResponder.generate_reply` to request both text and audio in one `responses.create`
  payload (set the `audio` block with `voice`, `format`, and `sample_rate`) and return a structured
  object containing the reply text plus decoded audio bytes.
- Update `finalize_turn_and_respond` so it logs the text and forwards the audio to the playback
  helper. Everything stays in one API request and returns only when the audio is ready.
- Pros: one request handles both modalities, billing stays consolidated, easier rate limiting.
- Cons: still not streaming; requires modest refactor so reply handling deals with audio blobs.
- **Current status:** the Responses API is still rolling out audio support, so the implementation
  detects the `Unknown parameter: 'audio'` error and automatically falls back to the Audio API until
  the feature flag is enabled for our org. On launch the CLI now probes for Responses audio support
  and reports which path (Responses vs. Audio API) will be used for the current session.

## Option 3 – Full Realtime API Assistant Audio

- Use the existing WebSocket session (`WebSocketClient`) to request assistant output directly from
  the Realtime API (`response.create` event with `"modalities":["text","audio"]`). Subscribe to
  `conversation.item.output_audio.delta` events and feed the chunks into the playback coroutine as
  they arrive for near-real-time speech.
- Pros: lowest latency (audio begins while generation is in progress); no additional HTTP calls; all
  state stays inside the Realtime session.
- Cons: largest engineering lift. Needs outbound assistant events, a playback task, and coordination
  so speaker output does not re-trigger the microphone/wake pipeline.

## Option 4 – Local / Offline TTS Engine

- Integrate an on-device engine (Piper, Coqui-TTS, Mimic3, etc.). After the reply text is ready,
  stream it to the local engine and route PCM to the same playback helper.
- Pros: works without network, avoids additional OpenAI usage, customizable voices, predictable
  latency once models are downloaded.
- Cons: extra install footprint on the Pi, higher CPU usage, and we must ship/manage voice assets
  plus any license implications. Quality will vary with the chosen model.

## Shared Considerations

- **Playback plumbing:** Whichever option wins, we need a small module (e.g.
  `pi_assistant/audio/playback.py`) with a queue-driven coroutine so TTS never blocks the wake
  loop.
- **Echo suppression:** Consider ducking or muting the microphone input while the speaker plays back
  to avoid self-triggering the wake word.
- **Configuration:** Expose model/voice selection via `config/defaults.toml` to keep device-specific
  tweaks out of the codebase.
