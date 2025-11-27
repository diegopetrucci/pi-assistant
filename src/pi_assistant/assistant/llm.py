"""LLM response helpers and cached TTS phrase playback."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Literal, Mapping, Optional, Tuple, TypedDict, cast

from openai import AsyncOpenAI, BadRequestError
from typing_extensions import Unpack

from pi_assistant.assistant.llm_payloads import extract_modalities
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

        messages = self._build_messages(prompt)

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

        return await self._reply_from_response(response)

    def _build_messages(self, prompt: str) -> list[dict]:
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
        language_instruction = self._language_instruction()
        if language_instruction:
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
        return messages

    def _language_instruction(self) -> Optional[str]:
        if not self._language:
            return None
        return (
            f"Always interpret the conversation in {self._language}. "
            f"Respond strictly in {self._language}, even if the user speaks another language "
            "or the transcript seems to switch languages."
        )

    async def _reply_from_response(self, response) -> Optional[LLMReply]:
        (
            text,
            audio_bytes,
            audio_format,
            sample_rate,
            chunk_count,
        ) = extract_modalities(response, default_sample_rate=self._tts_sample_rate)
        self._log_modalities_summary(text, audio_bytes, audio_format, sample_rate, chunk_count)
        audio_bytes, audio_format, sample_rate = await self._ensure_audio_payload(
            text,
            audio_bytes,
            audio_format,
            sample_rate,
        )
        if text is None and audio_bytes is None:
            return None
        return LLMReply(
            text=text,
            audio_bytes=audio_bytes,
            audio_format=audio_format,
            audio_sample_rate=sample_rate,
        )

    def _log_modalities_summary(
        self,
        text: Optional[str],
        audio_bytes: Optional[bytes],
        audio_format: Optional[str],
        sample_rate: Optional[int],
        chunk_count: int,
    ) -> None:
        console_print(
            "[ASSISTANT] Response received "
            f"(text={'yes' if text else 'no'}, audio_chunks={chunk_count})"
        )
        if not self._enable_tts:
            return
        if audio_bytes:
            verbose_print(
                f"[ASSISTANT] Received {chunk_count} audio chunk(s) "
                f"({len(audio_bytes)} bytes @ {sample_rate or 'unknown'} Hz, "
                f"format={audio_format or 'unknown'})"
            )
            return
        console_print(
            "[ASSISTANT] Assistant audio unavailable "
            f"({self._describe_audio_unavailability(chunk_count)}); "
            "falling back to synthesized speech."
        )

    async def _ensure_audio_payload(
        self,
        text: Optional[str],
        audio_bytes: Optional[bytes],
        audio_format: Optional[str],
        sample_rate: Optional[int],
    ) -> tuple[Optional[bytes], Optional[str], Optional[int]]:
        if not self._enable_tts or audio_bytes is not None or text is None:
            return audio_bytes, audio_format, sample_rate
        console_print("[ASSISTANT] Synthesizing fallback audio via the Audio API...")
        fallback_audio, fallback_rate = await self._synthesize_audio(text)
        if fallback_audio:
            new_sample_rate = fallback_rate or sample_rate or self._tts_sample_rate
            console_print(
                "[ASSISTANT] Fallback audio ready "
                f"({len(fallback_audio)} bytes @ {new_sample_rate or 'unknown'} Hz)."
            )
            return fallback_audio, self._tts_format, new_sample_rate
        console_print("[ASSISTANT] Fallback synthesis failed; assistant reply will be text-only.")
        return None, audio_format, sample_rate

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

    def _describe_audio_unavailability(self, chunk_count: int) -> str:
        if not self._responses_audio_requested:
            return "Responses audio disabled"
        if not self._responses_audio_supported:
            return "Responses audio probe failed"
        if chunk_count == 0:
            return "response returned zero audio chunks"
        return "response audio payload was empty"

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
        except Exception as exc:
            verbose_print(f"[ASSISTANT] Audio API synthesis failed: {exc}")
            return None, None
        verbose_print(
            "[ASSISTANT] Audio API returned fallback audio "
            f"({len(audio_bytes)} bytes @ {self._tts_sample_rate or 'unknown'} Hz)."
        )
        return audio_bytes, self._tts_sample_rate or None

    @staticmethod
    def _extract_modalities(
        response,
        default_sample_rate: Optional[int] = None,
    ) -> tuple[Optional[str], Optional[bytes], Optional[str], Optional[int], int]:
        return extract_modalities(response, default_sample_rate=default_sample_rate)


__all__ = ["LLMReply", "LLMResponder", "LLMResponderConfig"]
