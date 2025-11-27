"""Helpers for building assistant payloads and parsing Responses output."""

from __future__ import annotations

import base64
from typing import Iterator, Optional


def extract_modalities(
    response,
    default_sample_rate: Optional[int] = None,
) -> tuple[Optional[str], Optional[bytes], Optional[str], Optional[int], int]:
    """Pull text and audio payloads out of a Responses API payload."""

    payload = _normalize_response_payload(response)
    blocks = _coerce_output_blocks(payload)
    fragments, audio_chunks, audio_fmt, audio_rate = _collect_modalities(
        blocks,
        default_sample_rate=default_sample_rate,
    )
    combined_text = "\n".join(fragments).strip()
    text_output = combined_text or None
    audio_output = b"".join(audio_chunks) if audio_chunks else None
    return text_output, audio_output, audio_fmt, audio_rate, len(audio_chunks)


def _normalize_response_payload(response) -> object:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return response  # pragma: no cover - fallback for unexpected response shape


def _coerce_output_blocks(payload: object) -> list:
    if isinstance(payload, dict):
        raw_output = payload.get("output")
        if isinstance(raw_output, list):
            return raw_output
    return []


def _collect_modalities(
    blocks: list,
    *,
    default_sample_rate: Optional[int],
) -> tuple[list[str], list[bytes], Optional[str], Optional[int]]:
    fragments: list[str] = []
    audio_chunks: list[bytes] = []
    audio_format: Optional[str] = None
    audio_sample_rate: Optional[int] = default_sample_rate
    for content in _iter_output_contents(blocks):
        content_type = content.get("type")
        if content_type == "output_text":
            text = content.get("text", "").strip()
            if text:
                fragments.append(text)
            continue
        if content_type == "output_audio":
            chunk, audio_format, audio_sample_rate = _decode_audio_chunk(
                content,
                audio_format,
                audio_sample_rate,
            )
            if chunk:
                audio_chunks.append(chunk)
    return fragments, audio_chunks, audio_format, audio_sample_rate


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


__all__ = ["extract_modalities"]
