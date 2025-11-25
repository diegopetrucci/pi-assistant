# Realtime WebSocket handshake omits required subprotocol

**Location:** `src/pi_assistant/network/websocket_client.py:48-55`

**Problem:** The Realtime client calls `websockets.connect(OPENAI_REALTIME_ENDPOINT, additional_headers=WEBSOCKET_HEADERS)` but never declares the `Sec-WebSocket-Protocol: realtime` subprotocol that the OpenAI Realtime service mandates. Production servers reject such connections with 400/upgrade errors, so the CLI cannot reach the transcription API in real deployments.

**Impact:** Outside of mocked tests, every connection attempt fails before the session is created. Users see `Error connecting to OpenAI: ... invalid or missing subprotocol` and the assistant exits immediately.

**How to reproduce:**
1. Run `pi-assistant run` against the public Realtime endpoint.
2. Observe the connection failure (400 response mentioning `Sec-WebSocket-Protocol`).
3. Capture traffic via `tcpdump`/`mitmproxy` to confirm the header is missing.

**Suggested fix:** Pass `subprotocols=["realtime"]` (or whichever value OpenAI specifies) to `websockets.connect`. This ensures the client advertises the required protocol during the handshake.
