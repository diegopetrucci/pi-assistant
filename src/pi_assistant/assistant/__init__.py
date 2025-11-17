"""Assistant helper package exposing transcript + LLM utilities."""

from openai import BadRequestError

from .llm import LLMReply, LLMResponder
from .transcript import TurnTranscriptAggregator

__all__ = ["BadRequestError", "LLMReply", "LLMResponder", "TurnTranscriptAggregator"]
