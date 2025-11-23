import asyncio
import base64
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
from openai import AsyncOpenAI

from pi_assistant.assistant import (
    BadRequestError as OpenAIBadRequestError,
)
from pi_assistant.assistant import (
    LLMResponder,
    TurnTranscriptAggregator,
)


class TurnTranscriptAggregatorTest(unittest.IsolatedAsyncioTestCase):
    async def test_combines_segments_in_order(self):
        aggregator = TurnTranscriptAggregator(drain_timeout_seconds=0)
        await aggregator.start_turn()
        await aggregator.append_transcript("alpha", "  Hello")
        await aggregator.append_transcript(None, "world  ")

        result = await aggregator.finalize_turn()

        self.assertEqual(result, "Hello world")

    async def test_deduplicates_by_item_id(self):
        aggregator = TurnTranscriptAggregator(drain_timeout_seconds=0)
        await aggregator.start_turn()
        await aggregator.append_transcript("same", "first")
        await aggregator.append_transcript("same", "duplicate should be ignored")
        await aggregator.append_transcript("other", "second")

        result = await aggregator.finalize_turn()

        self.assertEqual(result, "first second")

    async def test_ignores_appends_when_idle(self):
        aggregator = TurnTranscriptAggregator(drain_timeout_seconds=0)
        await aggregator.append_transcript("unused", "text")
        self.assertIsNone(await aggregator.finalize_turn())

        await aggregator.start_turn()
        await aggregator.append_transcript(None, "active")
        self.assertEqual(await aggregator.finalize_turn(), "active")

    async def test_finalize_waits_for_late_segments(self):
        aggregator = TurnTranscriptAggregator(
            drain_timeout_seconds=0.01, max_finalize_wait_seconds=0.05
        )
        await aggregator.start_turn()

        finalize_task = asyncio.create_task(aggregator.finalize_turn())
        await asyncio.sleep(0.005)
        await aggregator.append_transcript("first", "Hello")
        await aggregator.append_transcript("second", "world")

        result = await finalize_task

        self.assertEqual(result, "Hello world")

    async def test_finalize_returns_none_after_timeout(self):
        aggregator = TurnTranscriptAggregator(
            drain_timeout_seconds=0.0, max_finalize_wait_seconds=0.0
        )
        await aggregator.start_turn()

        result = await aggregator.finalize_turn()

        self.assertIsNone(result)

    async def test_clear_current_turn_resets_segments_and_seen_items(self):
        aggregator = TurnTranscriptAggregator(
            drain_timeout_seconds=0.0, max_finalize_wait_seconds=0.0
        )
        await aggregator.start_turn()
        await aggregator.append_transcript("shared", "first")

        await aggregator.clear_current_turn("test")
        await aggregator.append_transcript("shared", "second")

        result = await aggregator.finalize_turn()

        self.assertEqual(result, "second")

    async def test_append_transcript_ignores_whitespace_only_segments(self):
        aggregator = TurnTranscriptAggregator(drain_timeout_seconds=0.0)
        await aggregator.start_turn()
        await aggregator.append_transcript("first", "   ")
        await aggregator.append_transcript("second", "\t")
        await aggregator.append_transcript(None, "Valid")

        result = await aggregator.finalize_turn()

        self.assertEqual(result, "Valid")

    async def test_finalize_turn_handles_very_large_transcript(self):
        aggregator = TurnTranscriptAggregator(
            drain_timeout_seconds=0.0,
            max_finalize_wait_seconds=0.0,
        )
        await aggregator.start_turn()
        large_segment = "chunk" * 10000
        await aggregator.append_transcript(None, large_segment)
        await aggregator.append_transcript(None, "tail")

        result = await aggregator.finalize_turn()

        self.assertTrue(result.endswith("tail"))  # pyright: ignore[reportOptionalMemberAccess]
        self.assertEqual(len(result), len(("chunk" * 10000) + " tail"))  # pyright: ignore[reportArgumentType]

    async def test_finalize_turn_drops_segments_arriving_after_completion(self):
        aggregator = TurnTranscriptAggregator(
            drain_timeout_seconds=0.01,
            max_finalize_wait_seconds=0.02,
        )
        await aggregator.start_turn()
        seen_append_count = 0

        async def delayed_append(text, delay):
            nonlocal seen_append_count
            await asyncio.sleep(delay)
            await aggregator.append_transcript(None, text)
            seen_append_count += 1

        appenders = [
            asyncio.create_task(delayed_append("Hello", 0.0)),
            asyncio.create_task(delayed_append("world", 0.005)),
            asyncio.create_task(delayed_append("late chunk", 0.05)),
        ]

        finalize_task = asyncio.create_task(aggregator.finalize_turn())
        await asyncio.gather(finalize_task, *appenders)

        self.assertEqual(finalize_task.result(), "Hello world")
        self.assertEqual(seen_append_count, 3)
        self.assertEqual(aggregator._segments, [])


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


class FakeResponsesAPI:
    def __init__(self, outer_client):
        self._outer = outer_client

    async def create(self, **payload):
        self._outer.calls.append(payload)
        if self._outer.error_to_raise is not None:
            exc = self._outer.error_to_raise
            self._outer.error_to_raise = None
            raise exc
        return FakeResponse(self._outer.response_payload)


class FakeAudioResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    async def aread(self):
        return self._payload


class FakeAudioSpeechAPI:
    def __init__(self, outer_client):
        self._outer = outer_client

    async def create(self, **payload):
        self._outer.audio_calls.append(payload)
        if self._outer.audio_error_to_raise is not None:
            exc = self._outer.audio_error_to_raise
            self._outer.audio_error_to_raise = None
            raise exc
        return FakeAudioResponse(self._outer.audio_payload)


class FakeOpenAIClient:
    def __init__(self, payload):
        self.calls = []
        self.response_payload = payload
        self.responses = FakeResponsesAPI(self)
        self.audio_calls = []
        self.audio_payload = b""
        self.audio = SimpleNamespace(speech=FakeAudioSpeechAPI(self))
        self.error_to_raise = None
        self.audio_error_to_raise = None


class DummyBadRequest(OpenAIBadRequestError):
    def __init__(self, message: str = "", body=None, param=None):
        Exception.__init__(self, message)
        self.message = message
        self.body = body
        self.param = param


class LLMResponderTest(unittest.IsolatedAsyncioTestCase):
    async def test_generates_reply_and_respects_tools_flag(self):
        payload = {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "It is sunny."},
                    ]
                }
            ]
        }
        client = FakeOpenAIClient(payload)
        responder = LLMResponder(
            client=cast(AsyncOpenAI, client),
            enable_web_search=True,
            reasoning_effort="low",
        )

        reply = await responder.generate_reply("Weather update")

        self.assertIsNotNone(reply)
        self.assertEqual(reply.text, "It is sunny.")  # pyright: ignore[reportOptionalMemberAccess]
        self.assertEqual(len(client.calls), 1)
        sent = client.calls[0]
        self.assertEqual(sent["input"][0]["content"][0]["type"], "input_text")
        self.assertIn("tools", sent)
        self.assertEqual(sent.get("reasoning"), {"effort": "low"})

    async def test_includes_reasoning_effort_when_configured(self):
        payload = {"output": []}
        client = FakeOpenAIClient(payload)
        responder = LLMResponder(
            client=cast(AsyncOpenAI, client),
            enable_web_search=False,
            reasoning_effort="medium",
        )  # pyright: ignore[reportArgumentType]

        await responder.generate_reply("Hi")

        self.assertEqual(len(client.calls), 1)
        sent = client.calls[0]
        self.assertEqual(sent.get("reasoning"), {"effort": "medium"})

    async def test_ignores_invalid_reasoning_effort_values(self):
        payload = {"output": []}
        client = FakeOpenAIClient(payload)
        responder = LLMResponder(
            client=cast(AsyncOpenAI, client),
            enable_web_search=False,
            reasoning_effort="totally-invalid",
        )  # pyright: ignore[reportArgumentType]

        await responder.generate_reply("Hi")

        self.assertEqual(len(client.calls), 1)
        sent = client.calls[0]
        self.assertNotIn("reasoning", sent)

    async def test_minimal_reasoning_with_web_search_upgrades_to_low(self):
        payload = {"output": []}
        client = FakeOpenAIClient(payload)
        responder = LLMResponder(
            client=cast(AsyncOpenAI, client),
            enable_web_search=True,
            reasoning_effort="minimal",
        )  # pyright: ignore[reportArgumentType]

        await responder.generate_reply("Hi")

        self.assertEqual(len(client.calls), 1)
        sent = client.calls[0]
        self.assertEqual(sent.get("reasoning"), {"effort": "low"})

    async def test_returns_none_for_blank_transcript(self):
        client = FakeOpenAIClient({"output": []})
        responder = LLMResponder(client=client, enable_web_search=False)  # pyright: ignore[reportArgumentType]

        reply = await responder.generate_reply("   ")

        self.assertIsNone(reply)
        self.assertEqual(client.calls, [])

    async def test_handles_missing_output_text(self):
        client = FakeOpenAIClient({"output": [{"content": []}]})
        responder = LLMResponder(client=client, enable_web_search=False)  # pyright: ignore[reportArgumentType]

        reply = await responder.generate_reply("Hello")

        self.assertIsNone(reply)
        self.assertEqual(len(client.calls), 1)

    async def test_returns_audio_payload_from_responses(self):
        audio_chunk = base64.b64encode(b"\x01\x02\x03").decode()
        payload = {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_audio",
                            "audio": {
                                "data": audio_chunk,
                                "format": "pcm16",
                                "sample_rate": 22050,
                            },
                        }
                    ]
                }
            ]
        }
        client = FakeOpenAIClient(payload)
        responder = LLMResponder(client=client, enable_web_search=False, enable_tts=True)  # pyright: ignore[reportArgumentType]

        reply = await responder.generate_reply("Play audio")

        self.assertIsNotNone(reply)
        self.assertIsNone(reply.text)  # pyright: ignore[reportOptionalMemberAccess]
        self.assertEqual(reply.audio_bytes, b"\x01\x02\x03")  # pyright: ignore[reportOptionalMemberAccess]
        self.assertEqual(reply.audio_format, "pcm16")  # pyright: ignore[reportOptionalMemberAccess]
        self.assertEqual(reply.audio_sample_rate, 22050)  # pyright: ignore[reportOptionalMemberAccess]

    async def test_synthesizes_audio_when_responses_returns_text_only(self):
        payload = {
            "output": [
                {"content": [{"type": "output_text", "text": "Sure thing."}]},
            ]
        }
        client = FakeOpenAIClient(payload)
        responder = LLMResponder(client=client, enable_web_search=False, enable_tts=True)  # pyright: ignore[reportArgumentType]

        async def fake_synthesize(self, text):
            fake_synthesize.called_with = text  # pyright: ignore[reportFunctionMemberAccess]
            return b"pcm-bytes", 12345

        fake_synthesize.called_with = None  # pyright: ignore[reportFunctionMemberAccess]

        with patch.object(LLMResponder, "_synthesize_audio", new=fake_synthesize):
            reply = await responder.generate_reply("Need TTS please")

        self.assertEqual(fake_synthesize.called_with, "Sure thing.")  # pyright: ignore[reportFunctionMemberAccess]
        self.assertIsNotNone(reply)
        self.assertEqual(reply.audio_bytes, b"pcm-bytes")  # pyright: ignore[reportOptionalMemberAccess]
        self.assertEqual(reply.audio_sample_rate, 12345)  # pyright: ignore[reportOptionalMemberAccess]
        self.assertEqual(reply.audio_format, responder._tts_format)  # pyright: ignore[reportOptionalMemberAccess]

    def test_build_audio_extra_body_respects_tts_flags(self):
        client = FakeOpenAIClient({"output": []})
        responder = LLMResponder(client=cast(AsyncOpenAI, client), enable_tts=False)
        self.assertIsNone(responder._build_audio_extra_body())

        responder_local_mode = LLMResponder(
            client=cast(AsyncOpenAI, client),
            enable_tts=True,
            use_responses_audio=False,
        )
        self.assertIsNone(responder_local_mode._build_audio_extra_body())

        responder_tts = LLMResponder(
            client=cast(AsyncOpenAI, client),
            enable_tts=True,
            use_responses_audio=True,
            tts_voice="robot",
            tts_format="pcm",
            tts_sample_rate=1234,
        )

        self.assertEqual(
            responder_tts._build_audio_extra_body(),
            {"audio": {"voice": "robot", "format": "pcm", "sample_rate": 1234}},
        )

    async def test_send_response_request_retries_without_audio_on_known_error(self):
        client = FakeOpenAIClient({"output": []})
        responder = LLMResponder(
            client=cast(AsyncOpenAI, client),
            enable_tts=True,
            use_responses_audio=True,
        )
        responder.set_responses_audio_supported(True)
        client.error_to_raise = DummyBadRequest(  # pyright: ignore[reportAttributeAccessIssue]
            message="Unknown parameter audio",
            body={"error": {"param": "audio", "message": "Unknown parameter audio"}},
        )

        result = await responder._send_response_request({"input": []}, extra_body={"audio": {}})

        self.assertEqual(result.model_dump(), client.response_payload)
        self.assertEqual(len(client.calls), 2)
        self.assertIn("extra_body", client.calls[0])
        self.assertNotIn("extra_body", client.calls[1])
        self.assertFalse(responder.responses_audio_supported)

    async def test_send_response_request_propagates_unrelated_errors(self):
        client = FakeOpenAIClient({"output": []})
        responder = LLMResponder(client=client, enable_tts=True, use_responses_audio=True)  # pyright: ignore[reportArgumentType]
        responder.set_responses_audio_supported(True)
        client.error_to_raise = DummyBadRequest(message="Other failure")  # pyright: ignore[reportAttributeAccessIssue]

        with pytest.raises(DummyBadRequest):
            await responder._send_response_request({"input": []}, extra_body={"audio": {}})
        self.assertEqual(len(client.calls), 1)

    def test_should_retry_without_audio_checks_param_and_message(self):
        err_with_param = DummyBadRequest(body={"error": {"param": "audio"}})
        err_with_message = DummyBadRequest(message="Unknown parameter modalities")
        err_other = DummyBadRequest(message="Other failure")

        self.assertTrue(LLMResponder._should_retry_without_audio(err_with_param))
        self.assertTrue(LLMResponder._should_retry_without_audio(err_with_message))
        self.assertFalse(LLMResponder._should_retry_without_audio(err_other))

    def test_log_audio_fallback_warning_only_once(self):
        client = FakeOpenAIClient({"output": []})
        responder = LLMResponder(client=client)  # pyright: ignore[reportArgumentType]
        err = DummyBadRequest(message="Unknown parameter audio")

        with patch("builtins.print") as mock_print:
            responder._log_audio_fallback_warning(err)
            responder._log_audio_fallback_warning(err)

        mock_print.assert_called_once()

    async def test_verify_responses_audio_support_success(self):
        client = FakeOpenAIClient({"output": []})
        responder = LLMResponder(
            client=cast(AsyncOpenAI, client),
            enable_tts=True,
            use_responses_audio=True,
        )

        supported = await responder.verify_responses_audio_support()

        self.assertTrue(supported)
        self.assertTrue(responder.responses_audio_supported)
        self.assertEqual(len(client.calls), 1)
        self.assertIn("extra_body", client.calls[0])

    async def test_verify_responses_audio_support_handles_audio_errors(self):
        client = FakeOpenAIClient({"output": []})
        responder = LLMResponder(
            client=client,  # pyright: ignore[reportArgumentType]
            enable_tts=True,
            use_responses_audio=True,
        )
        client.error_to_raise = DummyBadRequest(  # pyright: ignore[reportAttributeAccessIssue]
            message="Unknown parameter audio",
            body={"error": {"param": "audio"}},
        )

        supported = await responder.verify_responses_audio_support()

        self.assertFalse(supported)
        self.assertFalse(responder.responses_audio_supported)
        self.assertEqual(len(client.calls), 1)

    async def test_verify_responses_audio_support_propagates_other_errors(self):
        client = FakeOpenAIClient({"output": []})
        responder = LLMResponder(
            client=client,  # pyright: ignore[reportArgumentType]
            enable_tts=True,
            use_responses_audio=True,
        )
        client.error_to_raise = DummyBadRequest(message="rate limited")  # pyright: ignore[reportAttributeAccessIssue]

        with pytest.raises(DummyBadRequest):
            await responder.verify_responses_audio_support()

        self.assertEqual(len(client.calls), 1)

    async def test_verify_responses_audio_support_skips_when_mode_disabled(self):
        client = FakeOpenAIClient({"output": []})
        responder = LLMResponder(
            client=cast(AsyncOpenAI, client),
            enable_tts=True,
            use_responses_audio=False,
        )

        supported = await responder.verify_responses_audio_support()

        self.assertFalse(supported)
        self.assertFalse(responder.responses_audio_supported)
        self.assertEqual(len(client.calls), 0)

    async def test_synthesize_audio_handles_exceptions(self):
        client = FakeOpenAIClient({"output": []})
        responder = LLMResponder(client=cast(AsyncOpenAI, client), enable_tts=True)
        client.audio_error_to_raise = RuntimeError("audio synth failure")  # pyright: ignore[reportAttributeAccessIssue]

        audio_bytes, sample_rate = await responder._synthesize_audio("hello")

        self.assertIsNone(audio_bytes)
        self.assertIsNone(sample_rate)
        self.assertEqual(len(client.audio_calls), 1)

    async def test_generate_reply_can_run_concurrently(self):
        payload = {
            "output": [
                {"content": [{"type": "output_text", "text": "Hello"}]},
            ]
        }
        client = FakeOpenAIClient(payload)
        responder = LLMResponder(client=client, enable_web_search=False)  # pyright: ignore[reportArgumentType]

        async def run_after_delay(text, delay):
            await asyncio.sleep(delay)
            return await responder.generate_reply(text)

        task1 = asyncio.create_task(run_after_delay("Query one", 0.01))
        task2 = asyncio.create_task(run_after_delay("Query two", 0.0))

        replies = await asyncio.gather(task1, task2)

        self.assertTrue(all(reply and reply.text == "Hello" for reply in replies))
        self.assertEqual(len(client.calls), 2)

    async def test_generate_reply_includes_system_and_location_prompts(self):
        payload = {
            "output": [
                {"content": [{"type": "output_text", "text": "Ack"}]},
            ]
        }
        client = FakeOpenAIClient(payload)
        responder = LLMResponder(
            client=client,  # pyright: ignore[reportArgumentType]
            enable_web_search=False,
            system_prompt="Use markdown & emojis 😊",
            location_name="São Paulo, BR",
        )

        await responder.generate_reply("Status?")

        self.assertEqual(len(client.calls), 1)
        sent = client.calls[0]
        system_entries = [msg for msg in sent["input"] if msg["role"] == "system"]
        self.assertEqual(len(system_entries), 3)
        self.assertIn("markdown & emojis 😊", system_entries[0]["content"][0]["text"])
        self.assertIn("São Paulo, BR", system_entries[1]["content"][0]["text"])
        self.assertIn("Respond strictly in en", system_entries[2]["content"][0]["text"])

    async def test_generate_reply_handles_location_with_special_characters(self):
        payload = {
            "output": [
                {"content": [{"type": "output_text", "text": "OK"}]},
            ]
        }
        client = FakeOpenAIClient(payload)
        responder = LLMResponder(
            client=client,  # pyright: ignore[reportArgumentType]
            enable_web_search=False,
            system_prompt="",
            location_name="München 🏰 / 東京",
        )

        await responder.generate_reply("Where am I?")

        self.assertEqual(len(client.calls), 1)
        sent = client.calls[0]
        system_entries = [msg for msg in sent["input"] if msg["role"] == "system"]
        self.assertEqual(len(system_entries), 2)
        self.assertIn("München 🏰 / 東京", system_entries[0]["content"][0]["text"])
        self.assertIn("Respond strictly in en", system_entries[1]["content"][0]["text"])

    async def test_generate_reply_truncates_very_long_system_prompt(self):
        payload = {
            "output": [
                {"content": [{"type": "output_text", "text": "OK"}]},
            ]
        }
        client = FakeOpenAIClient(payload)
        responder = LLMResponder(
            client=client,  # pyright: ignore[reportArgumentType]
            enable_web_search=False,
            system_prompt="X" * 5000,
            location_name="",
        )

        await responder.generate_reply("Check prompt length")

        self.assertEqual(len(client.calls), 1)
        sent = client.calls[0]
        system_entries = [msg for msg in sent["input"] if msg["role"] == "system"]
        self.assertEqual(len(system_entries), 2)
        self.assertEqual(len(system_entries[0]["content"][0]["text"]), 5000)
        self.assertIn("Respond strictly in en", system_entries[1]["content"][0]["text"])

    def test_extract_modalities_combines_text_and_audio_chunks(self):
        payload = {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "Line one"},
                        {
                            "type": "output_audio",
                            "audio": {
                                "data": base64.b64encode(b"\x00\x01").decode(),
                                "format": "pcm16",
                                "sample_rate": 123,
                            },
                        },
                        {"type": "output_text", "text": "Line two"},
                    ]
                },
                {
                    "content": [
                        {"type": "output_text", "text": "Line three"},
                        {
                            "type": "output_audio",
                            "audio": {
                                "data": base64.b64encode(b"\x02\x03").decode(),
                            },
                        },
                    ]
                },
            ]
        }

        text, audio_bytes, audio_fmt, sample_rate, chunk_count = LLMResponder._extract_modalities(
            payload, default_sample_rate=999
        )

        self.assertEqual(text, "Line one\nLine two\nLine three")
        self.assertEqual(audio_bytes, b"\x00\x01\x02\x03")
        self.assertEqual(audio_fmt, "pcm16")
        self.assertEqual(sample_rate, 123)
        self.assertEqual(chunk_count, 2)

    def test_extract_modalities_ignores_invalid_audio_payloads(self):
        payload = {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_audio",
                            "audio": {
                                "data": "!!not-base64!!",
                                "format": "pcm16",
                            },
                        }
                    ]
                }
            ]
        }

        text, audio_bytes, audio_fmt, sample_rate, chunk_count = LLMResponder._extract_modalities(
            payload, default_sample_rate=16000
        )

        self.assertIsNone(text)
        self.assertIsNone(audio_bytes)
        self.assertIsNone(audio_fmt)
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(chunk_count, 0)

    def test_extract_modalities_uses_default_sample_rate_when_missing(self):
        payload = {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_audio",
                            "audio": {
                                "data": base64.b64encode(b"\x10\x20").decode(),
                                "format": "wav",
                            },
                        }
                    ]
                }
            ]
        }

        text, audio_bytes, audio_fmt, sample_rate, chunk_count = LLMResponder._extract_modalities(
            payload, default_sample_rate=8000
        )

        self.assertIsNone(text)
        self.assertEqual(audio_bytes, b"\x10\x20")
        self.assertEqual(audio_fmt, "wav")
        self.assertEqual(sample_rate, 8000)
        self.assertEqual(chunk_count, 1)

    def test_extract_modalities_handles_missing_output_list(self):
        payload = {"output": None}

        text, audio_bytes, audio_fmt, sample_rate, chunk_count = LLMResponder._extract_modalities(
            payload, default_sample_rate=44100
        )

        self.assertIsNone(text)
        self.assertIsNone(audio_bytes)
        self.assertIsNone(audio_fmt)
        self.assertEqual(sample_rate, 44100)
        self.assertEqual(chunk_count, 0)

    def test_extract_modalities_skips_non_dict_blocks_and_contents(self):
        payload = {
            "output": [
                "text block",
                {"content": None},
                {"content": ["inner string"]},
                {
                    "content": [
                        {"type": "output_text", "text": "First chunk"},
                        {
                            "type": "output_audio",
                            "audio": {"data": base64.b64encode(b"ab").decode()},
                        },
                    ]
                },
            ]
        }

        text, audio_bytes, audio_fmt, sample_rate, chunk_count = LLMResponder._extract_modalities(
            payload, default_sample_rate=16000
        )

        self.assertEqual(text, "First chunk")
        self.assertEqual(audio_bytes, b"ab")
        self.assertIsNone(audio_fmt)
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(chunk_count, 1)
