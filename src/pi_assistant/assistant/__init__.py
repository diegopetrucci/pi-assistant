"""Assistant helper package exposing transcript + LLM utilities."""

from openai import BadRequestError

from .llm import LLMReply, LLMResponder, LLMResponderConfig
from .transcript import TurnTranscriptAggregator
from .transcription_session import (
    ASSISTANT_AUDIO_MODE_CHOICES,
    DEFAULT_ASSISTANT_AUDIO_MODE,
    TranscriptionComponentBuilder,
    TranscriptionComponents,
    TranscriptionConfigValidator,
    TranscriptionRunConfig,
    TranscriptionSession,
    TranscriptionTaskCoordinator,
    run_simulated_query_once,
)

__all__ = [
    "ASSISTANT_AUDIO_MODE_CHOICES",
    "BadRequestError",
    "DEFAULT_ASSISTANT_AUDIO_MODE",
    "LLMReply",
    "LLMResponder",
    "LLMResponderConfig",
    "TranscriptionComponentBuilder",
    "TranscriptionComponents",
    "TranscriptionConfigValidator",
    "TranscriptionRunConfig",
    "TranscriptionSession",
    "TranscriptionTaskCoordinator",
    "TurnTranscriptAggregator",
    "run_simulated_query_once",
]
