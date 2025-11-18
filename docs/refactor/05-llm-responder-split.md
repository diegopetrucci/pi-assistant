# Refactoring: LLM Responder Split

## Priority
**MEDIUM** - Estimated effort: 2 days

## Problem

The `LLMResponder.generate_reply` method in `src/pi_assistant/assistant/llm.py:75-162` is an 88-line function with **multiple responsibilities**:

1. **Message building**: Constructing request messages from transcript
2. **API communication**: Calling Anthropic API
3. **Response parsing**: Extracting content from response
4. **Modality handling**: Processing text vs audio responses
5. **Fallback logic**: Applying audio fallback when needed
6. **Error handling**: Managing API errors

### Current Code Structure

```python
# llm.py:75-162
async def generate_reply(
    self,
    transcript: str,
    conversation: list[dict],
    audio_fallback: bool = False
) -> LLMReply | None:
    """Generate a reply from the LLM."""

    # Lines 82-95: Message building
    messages = [...]
    if conversation:
        messages.extend(conversation)
    messages.append({"role": "user", "content": transcript})

    # Lines 97-118: API request
    response = await self.client.messages.create(
        model=self.model,
        max_tokens=self.max_tokens,
        messages=messages,
        # ... lots of parameters
    )

    # Lines 120-142: Response parsing
    text_content = None
    audio_content = None
    for block in response.content:
        if block.type == "text":
            text_content = block.text
        # ... more parsing

    # Lines 144-158: Audio fallback logic
    if audio_fallback and not audio_content:
        # Re-generate with different prompt
        # ... 14 more lines

    # Lines 160-162: Return
    return LLMReply(text=text_content, audio=audio_content)
```

### Issues

1. **High complexity**: Too many things happening in one method
2. **Hard to test**: Can't test message building without API call
3. **Tight coupling**: API details mixed with business logic
4. **Difficult to extend**: Adding new modalities or fallback strategies requires modifying this function
5. **Poor reusability**: Can't reuse message building or parsing logic independently

## Proposed Solution

Split into **focused, single-responsibility classes and methods**:

1. `MessageBuilder`: Construct API request messages
2. `LLMClient`: Handle API communication (thin wrapper)
3. `ResponseParser`: Extract content from API responses
4. `ModalityHandler`: Manage text/audio fallback logic
5. `LLMResponder`: Orchestrate the above components

### Implementation

#### 1. Message Builder (`message_builder.py`)

```python
"""Build messages for LLM requests."""

from typing import List, Dict, Any

class MessageBuilder:
    """Builds messages for LLM API requests."""

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    def build_messages(
        self,
        user_input: str,
        conversation_history: List[Dict[str, Any]] | None = None
    ) -> List[Dict[str, Any]]:
        """Build messages list for API request."""
        messages = []

        # Add conversation history if available
        if conversation_history:
            messages.extend(conversation_history)

        # Add current user input
        messages.append({
            "role": "user",
            "content": user_input
        })

        return messages

    def build_audio_fallback_messages(
        self,
        original_transcript: str,
        conversation_history: List[Dict[str, Any]] | None = None
    ) -> List[Dict[str, Any]]:
        """Build messages for audio fallback request."""
        # Add explicit audio request
        audio_prompt = (
            f"{original_transcript}\n\n"
            "Please provide your response in audio format."
        )

        return self.build_messages(audio_prompt, conversation_history)

    def get_system_prompt(self) -> str:
        """Get the system prompt."""
        return self.system_prompt
```

#### 2. Response Parser (`response_parser.py`)

```python
"""Parse LLM API responses."""

from typing import Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class ParsedResponse:
    """Parsed content from LLM response."""
    text: Optional[str] = None
    audio: Optional[bytes] = None
    has_text: bool = False
    has_audio: bool = False

    @property
    def is_empty(self) -> bool:
        """Check if response is empty."""
        return not (self.has_text or self.has_audio)


class ResponseParser:
    """Parse content from LLM API responses."""

    @staticmethod
    def parse(response: Any) -> ParsedResponse:
        """Extract text and audio content from API response."""
        text_content = None
        audio_content = None

        for block in response.content:
            if block.type == "text":
                text_content = block.text
            elif block.type == "audio":
                audio_content = ResponseParser._extract_audio_data(block)

        return ParsedResponse(
            text=text_content,
            audio=audio_content,
            has_text=text_content is not None,
            has_audio=audio_content is not None
        )

    @staticmethod
    def _extract_audio_data(audio_block: Any) -> Optional[bytes]:
        """Extract raw audio data from audio block."""
        try:
            if hasattr(audio_block, 'data'):
                import base64
                return base64.b64decode(audio_block.data)
            return None
        except Exception:
            return None
```

#### 3. Modality Handler (`modality_handler.py`)

```python
"""Handle different response modalities and fallback logic."""

from typing import Optional
from .response_parser import ParsedResponse
from .message_builder import MessageBuilder

class ModalityHandler:
    """Manages response modality preferences and fallback."""

    def __init__(self, prefer_audio: bool = False):
        self.prefer_audio = prefer_audio

    def should_apply_audio_fallback(
        self,
        response: ParsedResponse,
        audio_requested: bool
    ) -> bool:
        """Determine if audio fallback should be applied."""
        # Apply fallback if:
        # 1. Audio was requested AND
        # 2. Response has text but no audio AND
        # 3. Audio preference is enabled
        return (
            audio_requested and
            response.has_text and
            not response.has_audio and
            self.prefer_audio
        )

    async def apply_audio_fallback(
        self,
        original_transcript: str,
        conversation_history: list,
        llm_client: 'LLMClient',
        message_builder: MessageBuilder
    ) -> ParsedResponse:
        """Apply audio fallback by re-requesting with audio prompt."""
        # Build audio-specific messages
        messages = message_builder.build_audio_fallback_messages(
            original_transcript,
            conversation_history
        )

        # Make new request
        response = await llm_client.create_message(
            messages=messages,
            audio_output=True  # Force audio output
        )

        # Parse and return
        from .response_parser import ResponseParser
        return ResponseParser.parse(response)
```

#### 4. LLM Client Wrapper (`llm_client.py`)

```python
"""Thin wrapper around Anthropic API client."""

from typing import List, Dict, Any
from anthropic import AsyncAnthropic

class LLMClient:
    """Wrapper for Anthropic LLM API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 4096
    ):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    async def create_message(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        audio_output: bool = False
    ) -> Any:
        """Create a message using the Anthropic API."""
        # Build request parameters
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages
        }

        if system:
            params["system"] = system

        if audio_output:
            params["audio"] = {
                "voice": "alloy",
                "format": "pcm16"
            }

        # Make API request
        return await self.client.messages.create(**params)
```

#### 5. Refactored LLM Responder (`llm.py`)

```python
"""LLM response generation orchestration."""

from dataclasses import dataclass
from typing import Optional
from .llm_client import LLMClient
from .message_builder import MessageBuilder
from .response_parser import ResponseParser, ParsedResponse
from .modality_handler import ModalityHandler

@dataclass
class LLMReply:
    """Response from LLM."""
    text: Optional[str] = None
    audio: Optional[bytes] = None

class LLMResponder:
    """Orchestrates LLM response generation."""

    def __init__(
        self,
        api_key: str,
        model: str,
        system_prompt: str,
        prefer_audio: bool = False,
        max_tokens: int = 4096
    ):
        self.llm_client = LLMClient(api_key, model, max_tokens)
        self.message_builder = MessageBuilder(system_prompt)
        self.response_parser = ResponseParser()
        self.modality_handler = ModalityHandler(prefer_audio)

    async def generate_reply(
        self,
        transcript: str,
        conversation: list[dict] | None = None,
        audio_fallback: bool = False
    ) -> LLMReply | None:
        """Generate a reply from the LLM."""
        # 1. Build messages
        messages = self.message_builder.build_messages(
            user_input=transcript,
            conversation_history=conversation
        )

        # 2. Call LLM API
        response = await self.llm_client.create_message(
            messages=messages,
            system=self.message_builder.get_system_prompt()
        )

        # 3. Parse response
        parsed = self.response_parser.parse(response)

        # 4. Apply audio fallback if needed
        if self.modality_handler.should_apply_audio_fallback(parsed, audio_fallback):
            parsed = await self.modality_handler.apply_audio_fallback(
                original_transcript=transcript,
                conversation_history=conversation or [],
                llm_client=self.llm_client,
                message_builder=self.message_builder
            )

        # 5. Return formatted reply
        if parsed.is_empty:
            return None

        return LLMReply(text=parsed.text, audio=parsed.audio)
```

## Benefits

### Before: Single 88-line method
- Hard to test individual pieces
- High coupling between API and logic
- Difficult to extend with new modalities
- Can't reuse message building or parsing

### After: 5 focused classes
- **Testability**: Each component tested independently
- **Reusability**: Message building, parsing can be reused
- **Extensibility**: Easy to add new modalities or fallback strategies
- **Clarity**: Each class has single, clear responsibility
- **Maintainability**: Changes isolated to relevant component

## Testing Strategy

```python
import pytest
from pi_assistant.assistant.message_builder import MessageBuilder
from pi_assistant.assistant.response_parser import ResponseParser, ParsedResponse

def test_message_builder_simple():
    """Test building simple message."""
    builder = MessageBuilder(system_prompt="You are helpful")
    messages = builder.build_messages("Hello")

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"

def test_message_builder_with_history():
    """Test building message with conversation history."""
    builder = MessageBuilder(system_prompt="You are helpful")
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"}
    ]
    messages = builder.build_messages("How are you?", history)

    assert len(messages) == 3
    assert messages[0]["content"] == "Hi"
    assert messages[2]["content"] == "How are you?"

def test_response_parser_text_only():
    """Test parsing text-only response."""
    mock_response = MockResponse([
        MockBlock(type="text", text="Hello!")
    ])

    parsed = ResponseParser.parse(mock_response)

    assert parsed.has_text
    assert not parsed.has_audio
    assert parsed.text == "Hello!"

def test_response_parser_audio_only():
    """Test parsing audio-only response."""
    mock_response = MockResponse([
        MockBlock(type="audio", data="base64encodedaudio")
    ])

    parsed = ResponseParser.parse(mock_response)

    assert not parsed.has_text
    assert parsed.has_audio
    assert parsed.audio is not None

@pytest.mark.asyncio
async def test_llm_responder_integration():
    """Test full LLM responder flow."""
    responder = LLMResponder(
        api_key="test-key",
        model="claude-3-5-sonnet-20241022",
        system_prompt="Be helpful"
    )

    # Mock the LLM client
    responder.llm_client.create_message = AsyncMock(
        return_value=MockResponse([
            MockBlock(type="text", text="Test response")
        ])
    )

    reply = await responder.generate_reply("Hello")

    assert reply is not None
    assert reply.text == "Test response"
```

## Migration Plan

### Phase 1: Create New Components (Day 1, Morning)
1. Create `message_builder.py` with `MessageBuilder` class
2. Create `response_parser.py` with `ResponseParser` class
3. Add unit tests for both (no API calls needed!)

### Phase 2: Extract Client Wrapper (Day 1, Afternoon)
1. Create `llm_client.py` with thin API wrapper
2. Create `modality_handler.py` for fallback logic
3. Add tests with mocked API

### Phase 3: Refactor LLMResponder (Day 2, Morning)
1. Update `llm.py` to use new components
2. Replace `generate_reply` implementation
3. Keep same public interface (backward compatible)

### Phase 4: Testing and Cleanup (Day 2, Afternoon)
1. Integration tests for refactored `LLMResponder`
2. Remove old code
3. Update documentation

## Code Organization

```
src/pi_assistant/assistant/
├── __init__.py
├── llm.py                 # Main LLMResponder (orchestration only)
├── llm_client.py          # API wrapper
├── message_builder.py     # Message construction
├── response_parser.py     # Response parsing
├── modality_handler.py    # Modality/fallback logic
└── transcript.py          # (existing)
```

## Complexity Reduction

| Component | Before | After |
|-----------|--------|-------|
| `generate_reply` | 88 lines | ~30 lines (orchestration) |
| Message building | Embedded | ~40 lines (separate) |
| Response parsing | Embedded | ~50 lines (separate) |
| Fallback logic | Embedded | ~60 lines (separate) |
| **Total** | **88 lines** | **~180 lines** (better organized) |

Note: More lines total, but each piece is:
- Independently testable
- Reusable
- Easier to understand
- Simpler to modify

## Success Metrics

- [ ] `generate_reply` reduced to < 40 lines
- [ ] 90%+ test coverage on each component
- [ ] No regression in LLM functionality
- [ ] Message building testable without API calls
- [ ] Response parsing testable without API calls
- [ ] Can swap out API client implementation

## Related Refactorings

- **Error Handling**: Each component can use specific exceptions
- **Type Safety**: Strong typing for message/response structures
- **Testing**: Much easier to test with smaller components
