"""LLM response helpers and cached TTS phrase playback."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, replace
from typing import Iterator, Literal, Mapping, Optional, Tuple, TypedDict, cast

from openai import AsyncOpenAI, BadRequestError
from typing_extensions import Unpack

from pi_assistant.cli.logging_utils import console_print, verbose_print
from pi_assistant.config import (
    ASSISTANT_LANGUAGE,
    ASSISTANT_MODEL,
    ASSISTANT_REASONING_EFFORT,
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

AudioResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
VALID_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high"}
MINIMAL_INCOMPATIBLE_TOOLS = {"web_search"}


@dataclass(slots=True)
class LLMResponderConfig:
    model: str = ASSISTANT_MODEL
    system_prompt: str = ASSISTANT_SYSTEM_PROMPT
    location_name: str = LOCATION_NAME
    enable_web_search: bool = ASSISTANT_WEB_SEARCH_ENABLED
    enable_tts: bool = ASSISTANT_TTS_ENABLED
    use_responses_audio: bool = ASSISTANT_TTS_RESPONSES_ENABLED
    tts_model: str = ASSISTANT_TTS_MODEL
    tts_voice: str = ASSISTANT_TTS_VOICE
    tts_format: str = ASSISTANT_TTS_FORMAT
    tts_sample_rate: int = ASSISTANT_TTS_SAMPLE_RATE
    language: str = ASSISTANT_LANGUAGE
    reasoning_effort: Optional[str] = ASSISTANT_REASONING_EFFORT


LLM_RESPONDER_CONFIG_FIELDS = frozenset(LLMResponderConfig.__dataclass_fields__.keys())


class LLMResponderOverrides(TypedDict, total=False):
    model: str
    system_prompt: str
    location_name: str
    enable_web_search: bool
    enable_tts: bool
    use_responses_audio: bool
    tts_model: str
    tts_voice: str
    tts_format: str
    tts_sample_rate: int
    language: str
    reasoning_effort: Optional[str]


@dataclass
class LLMReply:
    """Structured assistant response with optional audio payload."""

    text: Optional[str]
    audio_bytes: Optional[bytes] = None
    audio_format: Optional[str] = None
    audio_sample_rate: Optional[int] = None


def _validate_llm_overrides(overrides: Mapping[str, object]) -> None:
    if not overrides:
        return
    invalid = set(overrides) - set(LLM_RESPONDER_CONFIG_FIELDS)
    if invalid:
        joined = ", ".join(sorted(invalid))
        raise TypeError(f"Invalid responder config override(s): {joined}")


class LLMResponder:
    """Thin wrapper around the OpenAI Responses API."""

    def __init__(
        self,
        *,
        config: Optional[LLMResponderConfig] = None,
        client: Optional[AsyncOpenAI] = None,
        **overrides: Unpack[LLMResponderOverrides],
    ):
        config_obj = config or LLMResponderConfig()
        if overrides:
            _validate_llm_overrides(overrides)
            config_obj = replace(config_obj, **overrides)
        self._config = config_obj
        self._client = client or AsyncOpenAI(api_key=OPENAI_API_KEY)
        self._model = self._config.model
        self._system_prompt = self._config.system_prompt.strip()
        self._location_name = self._config.location_name.strip()
        self._enable_web_search = self._config.enable_web_search
        self._enable_tts = self._config.enable_tts
        self._tts_model = self._config.tts_model
        self._tts_voice = self._config.tts_voice
        self._tts_format: AudioResponseFormat = self._normalize_response_format(
            self._config.tts_format
        )
        self._tts_sample_rate = self._config.tts_sample_rate
        self._language = self._config.language.strip() if self._config.language else ""
        self._responses_audio_requested = (
            self._config.enable_tts and self._config.use_responses_audio
        )
        self._responses_audio_supported = self._responses_audio_requested
        self._audio_fallback_logged = False
        self._phrase_audio_cache: dict[str, tuple[bytes, int]] = {}
        self._phrase_locks: dict[str, asyncio.Lock] = {}
        normalized_reasoning = (self._config.reasoning_effort or "").strip().lower()
        self._reasoning_effort = (
            normalized_reasoning if normalized_reasoning in VALID_REASONING_EFFORTS else None
        )
        if self._reasoning_effort == "minimal" and self._enable_web_search:
            console_print(
                (
                    "[ASSISTANT] Reasoning effort 'minimal' is not supported when web search is "
                    "enabled; raising to 'low'."
                )
            )
            self._reasoning_effort = "low"

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
        if self._reasoning_effort:
            request_kwargs["reasoning"] = {"effort": self._reasoning_effort}

        extra_body = self._build_audio_extra_body()
        console_print("[ASSISTANT] Awaiting OpenAI response...")
        response = await self._send_response_request(request_kwargs, extra_body)

        text, audio_bytes, audio_format, sample_rate, chunk_count = self._extract_modalities(
            response, default_sample_rate=self._tts_sample_rate
        )
        console_print(
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
                console_print("[ASSISTANT] No audio chunks returned; using text-only reply.")
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

    @property
    def location_name(self) -> str:
        """Return the configured location context passed to the LLM."""

        return self._location_name

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
    def model_name(self) -> str:
        """Return the configured assistant LLM model identifier."""

        return self._model

    @property
    def enabled_tools(self) -> tuple[str, ...]:
        """Return the tuple of assistant tools currently enabled."""

        tools: list[str] = []
        if self._enable_web_search:
            tools.append("web_search")
        return tuple(tools)

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
        console_print(
            "[ASSISTANT] Responses API does not accept audio parameters yet; "
            f"falling back to text-only responses ({details})."
        )
        self._audio_fallback_logged = True

    def _normalize_response_format(self, requested: str) -> AudioResponseFormat:
        allowed: tuple[AudioResponseFormat, ...] = ("mp3", "opus", "aac", "flac", "wav", "pcm")
        normalized = (requested or "").strip().lower()
        if normalized in allowed:
            return cast(AudioResponseFormat, normalized)
        console_print(
            "[ASSISTANT] Unknown TTS format "
            f"'{requested or 'N/A'}'; defaulting to 'mp3' for compatibility."
        )
        return "mp3"

    async def _synthesize_audio(self, text: str) -> tuple[Optional[bytes], Optional[int]]:
        if not text or not self._enable_tts:
            return None, None
        try:
            response = await self._client.audio.speech.create(
                model=self._tts_model,
                voice=self._tts_voice,
                input=text,
                response_format=self._tts_format,
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

        payload = LLMResponder._normalize_response_payload(response)
        blocks = LLMResponder._coerce_output_blocks(payload)
        fragments, audio_chunks, audio_fmt, audio_rate = LLMResponder._collect_modalities(
            blocks,
            default_sample_rate=default_sample_rate,
        )
        combined_text = "\n".join(fragments).strip()
        text_output = combined_text or None
        audio_output = b"".join(audio_chunks) if audio_chunks else None
        return text_output, audio_output, audio_fmt, audio_rate, len(audio_chunks)

    @staticmethod
    def _normalize_response_payload(response) -> object:
        if hasattr(response, "model_dump"):
            return response.model_dump()
        return response  # pragma: no cover - fallback for unexpected response shape

    @staticmethod
    def _coerce_output_blocks(payload: object) -> list:
        if isinstance(payload, dict):
            raw_output = payload.get("output")
            if isinstance(raw_output, list):
                return raw_output
        return []

    @staticmethod
    def _collect_modalities(
        blocks: list,
        *,
        default_sample_rate: Optional[int],
    ) -> tuple[list[str], list[bytes], Optional[str], Optional[int]]:
        fragments: list[str] = []
        audio_chunks: list[bytes] = []
        audio_format: Optional[str] = None
        audio_sample_rate: Optional[int] = default_sample_rate
        for content in LLMResponder._iter_output_contents(blocks):
            content_type = content.get("type")
            if content_type == "output_text":
                text = content.get("text", "").strip()
                if text:
                    fragments.append(text)
                continue
            if content_type == "output_audio":
                (
                    chunk,
                    audio_format,
                    audio_sample_rate,
                ) = LLMResponder._decode_audio_chunk(content, audio_format, audio_sample_rate)
                if chunk:
                    audio_chunks.append(chunk)
        return fragments, audio_chunks, audio_format, audio_sample_rate

    @staticmethod
    def _iter_output_contents(blocks: list) -> Iterator[dict]:
        for block in blocks:
            if not isinstance(block, dict):
                continue
            contents = block.get("content")
            if not isinstance(contents, list):
                continue
            for content in contents:
                if isinstance(content, dict):
                    yield content

    @staticmethod
    def _decode_audio_chunk(
        content: dict,
        existing_format: Optional[str],
        existing_sample_rate: Optional[int],
    ) -> tuple[Optional[bytes], Optional[str], Optional[int]]:
        audio_blob = content.get("audio") or {}
        if not isinstance(audio_blob, dict):
            return None, existing_format, existing_sample_rate
        b64_data = audio_blob.get("data")
        if not b64_data:
            return None, existing_format, existing_sample_rate
        try:
            decoded = base64.b64decode(b64_data)
        except (ValueError, TypeError):  # pragma: no cover - invalid payload
            return None, existing_format, existing_sample_rate
        if not decoded:
            return None, existing_format, existing_sample_rate
        audio_format = audio_blob.get("format", existing_format)
        sample_rate = audio_blob.get("sample_rate") or existing_sample_rate
        return decoded, audio_format, sample_rate


__all__ = ["LLMReply", "LLMResponder", "LLMResponderConfig"]
