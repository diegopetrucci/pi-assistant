# LLM Optimisations

Context: Recent runs (see `~/.cache/pi-assistant/logs/2025-11-17T22-41-04.473.log`) show ~14 s between `[ASSISTANT] Awaiting OpenAI response...` and `[ASSISTANT] Response received`. The pause happens entirely inside `LLMResponder.generate_reply` (`src/pi_assistant/assistant/llm.py`), so we need to make the LLM stage faster or feel faster.

- **Adopt streaming responses** – Instead of waiting for `responses.create` to finish, switch to a streaming Responses invocation or reuse the existing Realtime socket for assistant replies. Show partial deltas in the CLI so users see progress immediately even if the model takes 10+ seconds.
- **Gate web search** – `ASSISTANT_WEB_SEARCH_ENABLED = true` means every request goes through retrieval before the model can answer, which is overkill for simple weather or chit-chat. Consider disabling it by default or triggering web search only when intents need it.
- **Use a faster model** – Even with the default `gpt-5-nano-2025-08-07`, pairing TTS + search routinely crosses 10 seconds. Evaluate `gpt-4o-mini` or a Realtime-optimized sibling to trade a bit of quality for much lower response times.
- **Optional UX tweak** – If we keep slower models, surface an "assistant is thinking…" animation or synthesized filler audio so the user gets immediate feedback while the model finishes.
