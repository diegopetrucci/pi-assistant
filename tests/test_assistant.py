import unittest

from pi_transcription.assistant import LLMResponder, TurnTranscriptAggregator


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
        return FakeResponse(self._outer.response_payload)


class FakeOpenAIClient:
    def __init__(self, payload):
        self.calls = []
        self.response_payload = payload
        self.responses = FakeResponsesAPI(self)


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

        self.assertEqual(reply, "It is sunny.")
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
