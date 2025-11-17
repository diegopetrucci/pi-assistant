"""LLM response helpers and cached TTS phrase playback."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import Optional, Tuple

from openai import AsyncOpenAI, BadRequestError

from pi_assistant.cli.logging_utils import verbose_print
from pi_assistant.config import (
    ASSISTANT_LANGUAGE,
    ASSISTANT_MODEL,
    ASSISTANT_SYSTEM_PROMPT,
    ASSISTANT_TTS_ENABLED,
    ASSISTANT_TTS_FORMAT,
    ASSISTANT_TTS_MODEL,
    ASSISTANT_TTS_RESPONSES_ENABLED,
    ASSISTANT_TTS_SAMPLE_RATE,
    ASSISTANT_TTS_VOICE,
    ASSISTANT_WEB_SEARCH_ENABLED,
    LOCATION_NAME,
    OPENAI_API_KEY,
)


@dataclass
class LLMReply:
    """Structured assistant response with optional audio payload."""

    text: Optional[str]
    audio_bytes: Optional[bytes] = None
    audio_format: Optional[str] = None
    audio_sample_rate: Optional[int] = None


class LLMResponder:
    """Thin wrapper around the OpenAI Responses API."""

    def __init__(
        self,
        *,
        model: str = ASSISTANT_MODEL,
        system_prompt: str = ASSISTANT_SYSTEM_PROMPT,
        location_name: str = LOCATION_NAME,
        enable_web_search: bool = ASSISTANT_WEB_SEARCH_ENABLED,
        enable_tts: bool = ASSISTANT_TTS_ENABLED,
        use_responses_audio: bool = ASSISTANT_TTS_RESPONSES_ENABLED,
        tts_model: str = ASSISTANT_TTS_MODEL,
        tts_voice: str = ASSISTANT_TTS_VOICE,
        tts_format: str = ASSISTANT_TTS_FORMAT,
        tts_sample_rate: int = ASSISTANT_TTS_SAMPLE_RATE,
        language: str = ASSISTANT_LANGUAGE,
        client: Optional[AsyncOpenAI] = None,
    ):
        self._client = client or AsyncOpenAI(api_key=OPENAI_API_KEY)
        self._model = model
        self._system_prompt = system_prompt.strip()
        self._location_name = location_name.strip()
        self._enable_web_search = enable_web_search
        self._enable_tts = enable_tts
        self._tts_model = tts_model
        self._tts_voice = tts_voice
        self._tts_format = tts_format
        self._tts_sample_rate = tts_sample_rate
        self._language = language.strip() if language else ""
        self._responses_audio_requested = enable_tts and use_responses_audio
        self._responses_audio_supported = self._responses_audio_requested
        self._audio_fallback_logged = False
        self._phrase_audio_cache: dict[str, tuple[bytes, int]] = {}
        self._phrase_locks: dict[str, asyncio.Lock] = {}

    async def generate_reply(self, transcript: str) -> Optional[LLMReply]:
        """Send the transcript to the LLM and return the response text."""

        prompt = transcript.strip()
        if not prompt:
            return None

        messages: list[dict] = []
        if self._system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": self._system_prompt}],
                }
            )
        if self._location_name:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Device location: {self._location_name}",
                        }
                    ],
                }
            )
        if self._language:
            language_instruction = (
                f"Always interpret the conversation in {self._language}. "
                f"Respond strictly in {self._language}, even if the user speaks another language "
                "or the transcript seems to switch languages."
            )
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": language_instruction}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        )

        request_kwargs = {
            "model": self._model,
            "input": messages,
        }
        if self._enable_web_search:
            request_kwargs["tools"] = [{"type": "web_search"}]

        extra_body = self._build_audio_extra_body()
        print("[ASSISTANT] Awaiting OpenAI response...")
        response = await self._send_response_request(request_kwargs, extra_body)

        text, audio_bytes, audio_format, sample_rate, chunk_count = self._extract_modalities(
            response, default_sample_rate=self._tts_sample_rate
        )
        print(
            "[ASSISTANT] Response received "
            f"(text={'yes' if text else 'no'}, audio_chunks={chunk_count})"
        )
        if self._enable_tts:
            if audio_bytes:
                verbose_print(
                    f"[ASSISTANT] Received {chunk_count} audio chunk(s) "
                    f"({len(audio_bytes)} bytes @ {sample_rate or 'unknown'} Hz, "
                    f"format={audio_format or 'unknown'})"
                )
            else:
                print("[ASSISTANT] No audio chunks returned; using text-only reply.")
        if self._enable_tts and audio_bytes is None and text:
            fallback_audio, fallback_rate = await self._synthesize_audio(text)
            if fallback_audio:
                audio_bytes = fallback_audio
                audio_format = self._tts_format
                sample_rate = fallback_rate or sample_rate

        if text is None and audio_bytes is None:
            return None
        return LLMReply(
            text=text,
            audio_bytes=audio_bytes,
            audio_format=audio_format,
            audio_sample_rate=sample_rate,
        )

    def peek_phrase_audio(self, text: str) -> Optional[Tuple[bytes, int]]:
        """Return cached audio for a fixed phrase if it exists."""

        if not text:
            return None
        key = text.strip()
        if not key:
            return None
        return self._phrase_audio_cache.get(key)

    async def warm_phrase_audio(self, text: str) -> Optional[Tuple[bytes, int]]:
        """Ensure TTS audio for a phrase is synthesized and cached."""

        if not (self._enable_tts and text):
            return None
        key = text.strip()
        if not key:
            return None
        cached = self._phrase_audio_cache.get(key)
        if cached:
            return cached
        lock = self._phrase_locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self._phrase_audio_cache.get(key)
            if cached:
                return cached
            audio_bytes, sample_rate = await self._synthesize_audio(key)
            if audio_bytes:
                playback_rate = sample_rate or self._tts_sample_rate or 24000
                payload = (audio_bytes, int(playback_rate))
                self._phrase_audio_cache[key] = payload
                return payload
        return None

    def _build_audio_extra_body(self) -> Optional[dict]:
        if not (self._enable_tts and self._responses_audio_supported):
            return None

        audio_config: dict[str, object] = {
            "voice": self._tts_voice,
            "format": self._tts_format,
        }
        if self._tts_sample_rate:
            audio_config["sample_rate"] = self._tts_sample_rate
        return {"audio": audio_config}

    async def _send_response_request(self, kwargs: dict, extra_body: Optional[dict]):
        if extra_body:
            try:
                return await self._client.responses.create(extra_body=extra_body, **kwargs)
            except BadRequestError as exc:
                if self._should_retry_without_audio(exc):
                    self.set_responses_audio_supported(False)
                    self._log_audio_fallback_warning(exc)
                    return await self._client.responses.create(**kwargs)
                raise
        return await self._client.responses.create(**kwargs)

    def set_responses_audio_supported(self, supported: bool) -> None:
        self._responses_audio_supported = bool(supported) and self._responses_audio_requested

    @property
    def responses_audio_supported(self) -> bool:
        return self._responses_audio_supported

    @property
    def tts_enabled(self) -> bool:
        return self._enable_tts

    async def verify_responses_audio_support(self) -> bool:
        """Probe the Responses API once at startup to see if audio output is enabled."""

        if not (self._enable_tts and self._responses_audio_requested):
            self.set_responses_audio_supported(False)
            return False

        extra_body = self._build_audio_extra_body()
        if not extra_body:
            self.set_responses_audio_supported(False)
            return False

        probe_prompt = "Respond with a single word so we can verify audio support."
        probe_kwargs = {
            "model": self._model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": probe_prompt}]}],
            "max_output_tokens": 5,
        }

        try:
            await self._client.responses.create(extra_body=extra_body, **probe_kwargs)
        except BadRequestError as exc:
            if self._should_retry_without_audio(exc):
                self.set_responses_audio_supported(False)
                return False
            raise

        self.set_responses_audio_supported(True)
        return True

    @staticmethod
    def _should_retry_without_audio(exc: BadRequestError) -> bool:
        body = exc.body if isinstance(exc.body, dict) else None
        error_info = body.get("error") if isinstance(body, dict) else None
        param = (error_info or {}).get("param") or exc.param
        if param in {"audio", "modalities"}:
            return True
        message = (error_info or {}).get("message") or exc.message
        if not message:
            return False
        lowered = message.lower()
        return "unknown parameter" in lowered and ("audio" in lowered or "modalities" in lowered)

    def _log_audio_fallback_warning(self, exc: BadRequestError) -> None:
        if self._audio_fallback_logged:
            return
        details = getattr(exc, "message", "") or "audio parameters not accepted"
        print(
            "[ASSISTANT] Responses API does not accept audio parameters yet; "
            f"falling back to text-only responses ({details})."
        )
        self._audio_fallback_logged = True

    async def _synthesize_audio(self, text: str) -> tuple[Optional[bytes], Optional[int]]:
        if not text or not self._enable_tts:
            return None, None
        try:
            response = await self._client.audio.speech.create(
                model=self._tts_model,
                voice=self._tts_voice,
                input=text,
                response_format=self._tts_format,  # pyright: ignore[reportArgumentType]
            )
            audio_bytes = await response.aread()
        except Exception:
            return None, None
        return audio_bytes, self._tts_sample_rate or None

    @staticmethod
    def _extract_modalities(
        response, default_sample_rate: Optional[int] = None
    ) -> tuple[Optional[str], Optional[bytes], Optional[str], Optional[int], int]:
        """Pull text and audio payloads out of a Responses API payload."""

        if hasattr(response, "model_dump"):
            data = response.model_dump()
        else:  # pragma: no cover - fallback for unexpected response shape
            data = response

        output: list = []
        if isinstance(data, dict):
            raw_output = data.get("output")
            if isinstance(raw_output, list):
                output = raw_output
        fragments: list[str] = []
        audio_chunks: list[bytes] = []
        audio_format: Optional[str] = None
        audio_sample_rate: Optional[int] = default_sample_rate
        audio_chunk_count = 0

        for block in output:
            if not isinstance(block, dict):
                continue
            contents = block.get("content")
            if not isinstance(contents, list):
                continue
            for content in contents:
                if not isinstance(content, dict):
                    continue
                content_type = content.get("type")
                if content_type == "output_text":
                    text = content.get("text", "").strip()
                    if text:
                        fragments.append(text)
                elif content_type == "output_audio":
                    audio_blob = content.get("audio") or {}
                    if not isinstance(audio_blob, dict):
                        continue
                    b64_data = audio_blob.get("data")
                    if not b64_data:
                        continue
                    try:
                        decoded = base64.b64decode(b64_data)
                    except (ValueError, TypeError):  # pragma: no cover - invalid payload
                        continue
                    if decoded:
                        audio_chunks.append(decoded)
                        audio_format = audio_blob.get("format", audio_format)
                        audio_sample_rate = audio_blob.get("sample_rate") or audio_sample_rate
                        audio_chunk_count += 1

        combined = "\n".join(fragments).strip()
        text_output = combined or None
        audio_output = b"".join(audio_chunks) if audio_chunks else None
        return text_output, audio_output, audio_format, audio_sample_rate, audio_chunk_count


__all__ = ["LLMReply", "LLMResponder"]
