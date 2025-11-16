import asyncio
import base64
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pi_transcription.assistant import (
    BadRequestError as OpenAIBadRequestError,
)
from pi_transcription.assistant import (
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
        responder = LLMResponder(client=client, enable_web_search=True)

        reply = await responder.generate_reply("Weather update")

        self.assertIsNotNone(reply)
        self.assertEqual(reply.text, "It is sunny.")
        self.assertEqual(len(client.calls), 1)
        sent = client.calls[0]
        self.assertEqual(sent["input"][0]["content"][0]["type"], "input_text")
        self.assertIn("tools", sent)

    async def test_returns_none_for_blank_transcript(self):
        client = FakeOpenAIClient({"output": []})
        responder = LLMResponder(client=client, enable_web_search=False)

        reply = await responder.generate_reply("   ")

        self.assertIsNone(reply)
        self.assertEqual(client.calls, [])

    async def test_handles_missing_output_text(self):
        client = FakeOpenAIClient({"output": [{"content": []}]})
        responder = LLMResponder(client=client, enable_web_search=False)

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
        responder = LLMResponder(client=client, enable_web_search=False, enable_tts=True)

        reply = await responder.generate_reply("Play audio")

        self.assertIsNotNone(reply)
        self.assertIsNone(reply.text)
        self.assertEqual(reply.audio_bytes, b"\x01\x02\x03")
        self.assertEqual(reply.audio_format, "pcm16")
        self.assertEqual(reply.audio_sample_rate, 22050)

    async def test_synthesizes_audio_when_responses_returns_text_only(self):
        payload = {
            "output": [
                {"content": [{"type": "output_text", "text": "Sure thing."}]},
            ]
        }
        client = FakeOpenAIClient(payload)
        responder = LLMResponder(client=client, enable_web_search=False, enable_tts=True)

        async def fake_synthesize(self, text):
            fake_synthesize.called_with = text
            return b"pcm-bytes", 12345

        fake_synthesize.called_with = None

        with patch.object(LLMResponder, "_synthesize_audio", new=fake_synthesize):
            reply = await responder.generate_reply("Need TTS please")

        self.assertEqual(fake_synthesize.called_with, "Sure thing.")
        self.assertIsNotNone(reply)
        self.assertEqual(reply.audio_bytes, b"pcm-bytes")
        self.assertEqual(reply.audio_sample_rate, 12345)
        self.assertEqual(reply.audio_format, responder._tts_format)

    def test_build_audio_extra_body_respects_tts_flags(self):
        client = FakeOpenAIClient({"output": []})
        responder = LLMResponder(client=client, enable_tts=False)
        self.assertIsNone(responder._build_audio_extra_body())

        responder_tts = LLMResponder(
            client=client,
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
        responder = LLMResponder(client=client, enable_tts=True, use_responses_audio=True)
        responder.set_responses_audio_supported(True)
        client.error_to_raise = DummyBadRequest(
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
        responder = LLMResponder(client=client, enable_tts=True, use_responses_audio=True)
        responder.set_responses_audio_supported(True)
        client.error_to_raise = DummyBadRequest(message="Other failure")

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
        responder = LLMResponder(client=client)
        err = DummyBadRequest(message="Unknown parameter audio")

        with patch("builtins.print") as mock_print:
            responder._log_audio_fallback_warning(err)
            responder._log_audio_fallback_warning(err)

        mock_print.assert_called_once()

    async def test_verify_responses_audio_support_success(self):
        client = FakeOpenAIClient({"output": []})
        responder = LLMResponder(
            client=client,
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
            client=client,
            enable_tts=True,
            use_responses_audio=True,
        )
        client.error_to_raise = DummyBadRequest(
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
            client=client,
            enable_tts=True,
            use_responses_audio=True,
        )
        client.error_to_raise = DummyBadRequest(message="rate limited")

        with pytest.raises(DummyBadRequest):
            await responder.verify_responses_audio_support()

        self.assertEqual(len(client.calls), 1)

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
