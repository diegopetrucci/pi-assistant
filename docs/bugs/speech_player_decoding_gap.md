# Audio playback assumes PCM16 but Requests default to MP3

**Location:** `src/pi_assistant/assistant/llm.py:327-336` and `src/pi_assistant/audio/playback.py:27-79`

**Problem:** `LLMResponder` allows (and even defaults to) compressed formats like `mp3`, `opus`, `aac`, etc., but `SpeechPlayer` always treats `audio_bytes` as raw PCM16 and feeds them directly into `numpy.frombuffer(..., dtype=np.int16)`. Any non-PCM payload is interpreted as sample data, producing loud static or silence.

**Impact:** Whenever the Responses API (or the fallback Audio API) returns MP3/Opus/etc., playback is corrupted and may even blast random noise through the speakers. Users only get intelligible audio if they manually configure `ASSISTANT_TTS_FORMAT="pcm"` and ensure the API honors it, which is currently undocumented.

**How to reproduce:**
1. Leave `config/defaults.toml` as-is (default `tts_format = "pcm"`).
2. Change the server-side TTS format to `mp3` via env/CLI.
3. Trigger a response; playback outputs garbage because decoding never happens.

**Suggested fix:** Either restrict the allowed formats to PCM/wav (and validate user config) or introduce decoding before passing bytes to `SpeechPlayer` (e.g., use `pydub`, `soundfile`, or `ffmpeg` bindings to convert to PCM16). Until decoding exists, the assistant should refuse to request formats other than PCM to prevent data corruption.
