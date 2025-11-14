# Raspberry Pi Audio Transcription System - Implementation Plan

## Project Overview
Real-time speech-to-text transcription system for Raspberry Pi 5 that streams audio from a USB microphone to OpenAI's Realtime API and displays transcribed text in the terminal.

## Implementation Status
- ✅ **Phase 1**: Project Setup (repo structure, dependencies, .env scaffolding)
- ✅ **Phase 2**: Configuration (audio constants, API endpoint/model, env loading)
- ✅ **Phase 3**: Audio Capture (sounddevice integration, async queue, validation)
- ✅ **Phase 4**: WebSocket Client (API key auth, session config, event handling)
- ✅ **Phase 5**: Integration (audio streaming + websocket bridge, CLI modes)
- 🟡 **Phase 6**: Error Handling (basic logging complete; resiliency/retries pending)
- 🟡 **Phase 7**: Code Quality Tools (Ruff + pre-commit installed; future tuning TBD)

## Technical Specifications

### Hardware
- **Device**: Raspberry Pi 5
- **Input**: USB microphone (plug-and-play, ALSA compatible)

### Audio Configuration
- **Sample Rate**: 24 kHz (OpenAI requirement)
- **Bit Depth**: 16-bit PCM
- **Channels**: Mono (1 channel)
- **Buffer Size**: 1024 frames (balanced for Pi 5 performance)
- **Format**: Little-endian PCM16

### API Integration
- **Service**: OpenAI Realtime API
- **Endpoint**: `wss://api.openai.com/v1/realtime?intent=transcription`
- **Model**: gpt-4o-transcribe
- **Protocol**: WebSocket with base64-encoded audio chunks
- **Headers**:
  - Authorization: Bearer [API_KEY]
  - OpenAI-Beta: realtime=v1

### WebSocket Protocol - Message Examples

#### 1. Session Configuration (Client → Server)
Sent immediately after WebSocket connection is established:
```json
{
  "type": "transcription_session.update",
  "input_audio_format": "pcm16",
  "input_audio_transcription": {
    "model": "gpt-4o-transcribe",
    "prompt": "",
    "language": "en"
  },
  "turn_detection": {
    "type": "server_vad",
    "threshold": 0.5,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 500
  },
  "input_audio_noise_reduction": {
    "type": "near_field"
  },
  "include": [
    "item.input_audio_transcription.logprobs"
  ]
}
```

#### 2. Audio Input (Client → Server)
Sent continuously while streaming audio:
```json
{
  "type": "input_audio_buffer.append",
  "audio": "base64_encoded_audio_data_here..."
}
```

#### 3. Transcription Delta (Server → Client)
Received as transcription is being processed:
```json
{
  "type": "conversation.item.input_audio_transcription.delta",
  "item_id": "item_123",
  "content_index": 0,
  "delta": "Hello, this is a partial"
}
```

#### 4. Transcription Complete (Server → Client)
Received when transcription segment is finalized:
```json
{
  "type": "conversation.item.input_audio_transcription.completed",
  "item_id": "item_123",
  "content_index": 0,
  "transcript": "Hello, this is a complete sentence."
}
```

#### 5. Error Response (Server → Client)
Received if something goes wrong:
```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "code": "invalid_audio_format",
    "message": "Audio format must be pcm16 at 24kHz",
    "param": null
  }
}
```

## Project Structure

```
pi-transcription/
├── .env                      # Environment variables (API key)
├── .gitignore               # Git ignore (already exists)
├── requirements.txt         # Python dependencies
├── config.py                # Configuration settings
├── transcribe.py            # Main application
├── .pre-commit-config.yaml  # Pre-commit hooks configuration
├── ruff.toml                # Ruff formatter/linter configuration
└── plan.md                  # This file
```

## Dependencies

### Runtime Dependencies
```
sounddevice>=0.4.6          # Audio capture (simpler API, good for Pi 5)
websockets>=12.0            # WebSocket client for direct API communication
python-dotenv>=1.0.0       # Environment variable management
numpy>=1.24.0              # Audio data processing
```

### Development Dependencies
```
ruff>=0.1.0                # Fast Python linter and formatter
pre-commit>=3.0.0          # Git pre-commit hook framework
```

## Architecture

### Data Flow
```
[USB Microphone]
    ↓ (ALSA)
[sounddevice capture]
    ↓ (callback)
[asyncio.Queue]
    ↓ (base64 encode)
[WebSocket → OpenAI Realtime API]
    ↓ (WebSocket events)
[Transcription Handler]
    ↓
[Terminal Output]
```

### Threading Model
- **Main Thread**: asyncio event loop
- **Audio Capture**: sounddevice callback (runs in separate thread)
- **Communication**: asyncio.Queue for thread-safe data transfer
- **WebSocket**: async/await based communication

## Implementation Components

### 1. Configuration Module (`config.py`)
- Audio settings (sample rate: 24kHz, buffer size: 1024, channels: 1)
- OpenAI API settings (model, endpoint)
- Environment variable loading (API key)

### 2. Main Application (`transcribe.py`)

#### Core Functions:
1. **Audio Capture**
   - Initialize sounddevice with 24kHz, mono, 16-bit
   - Capture audio in callback function
   - Push audio chunks to asyncio.Queue

2. **WebSocket Client**
   - Connect to OpenAI Realtime API (websockets library handles keep-alive)
   - Send transcription session configuration with server-side VAD enabled
   - Stream base64-encoded audio chunks
   - Receive and parse transcription events

3. **Event Handlers**
   - Handle `conversation.item.input_audio_transcription.delta` events (partial transcripts)
   - Handle `conversation.item.input_audio_transcription.completed` events (final transcripts)
   - Handle `input_audio_buffer.committed` events (VAD detected speech)
   - Print transcriptions to terminal in real-time

4. **Error Handling**
   - Print audio stream errors to console
   - Print WebSocket connection errors to console
   - Print API errors to console
   - Fail fast - let user manually restart on errors

5. **Graceful Shutdown**
   - Handle SIGINT (Ctrl+C)
   - Close audio stream properly
   - Close WebSocket connection
   - Clean up resources

## Implementation Steps

### Phase 1: Project Setup
1. Create `requirements.txt` with dependencies
2. Create `.env` file template (user adds API key)
3. Update `.gitignore` to exclude `.env`

### Phase 2: Configuration
1. Build `config.py` with:
   - Audio configuration constants (24kHz, 1024 buffer, mono)
   - OpenAI API settings (endpoint, model)
   - Environment variable loading (API key)

### Phase 3: Audio Capture
1. Implement sounddevice initialization
2. Create audio callback function
3. Set up asyncio.Queue for audio data
4. Test audio capture and verify format

### Phase 4: WebSocket Client
1. Implement WebSocket connection to OpenAI
2. Send transcription session configuration with server-side VAD
3. Create audio streaming function
4. Handle WebSocket events

### Phase 5: Integration
1. Bridge audio capture with WebSocket
2. Implement base64 encoding for audio chunks
3. Connect event handlers for transcription
4. Add terminal output formatting

### Phase 6: Error Handling
1. Add basic error printing to console
2. Handle audio buffer errors
3. Add graceful shutdown handler (Ctrl+C)

### Phase 7: Code Quality Tools
1. Set up ruff configuration (`ruff.toml`):
   - Configure line length (88 or 100)
   - Set Python version target
   - Enable desired linting rules
   - Configure formatting options
2. Create `.pre-commit-config.yaml`:
   - Add ruff formatter hook
   - Add ruff linter hook
   - Configure to run on commit
3. Install and initialize pre-commit:
   - `pip install pre-commit`
   - `pre-commit install`
4. Run initial formatting:
   - `ruff format .`
   - `ruff check --fix .`

## Key Technical Considerations

### Latency Budget
- Audio capture (sounddevice): ~50-100ms
- Network to OpenAI: ~50-200ms
- OpenAI processing: ~200-500ms
- Network return: ~50-200ms
- **Total expected latency**: ~350-1000ms

### Performance Optimizations
1. Use appropriate buffer sizes (1024 frames)
2. Minimize processing in audio callback
3. Efficient base64 encoding
4. Async I/O for network operations

### Error Scenarios to Handle
- USB microphone disconnection
- Network interruptions
- API errors
- Audio buffer overflow/underflow
- Invalid API key

### Resource Management
- Limit queue size to prevent memory buildup
- Proper cleanup on exit

## Raspberry Pi Setup Requirements

### System Dependencies
```bash
# Update system
sudo apt-get update

# Install audio libraries
sudo apt-get install -y libportaudio2 portaudio19-dev python3-dev

# Install Python pip
sudo apt-get install -y python3-pip

# Optional: Set CPU governor to performance
sudo cpufreq-set -g performance
```

### ALSA Configuration
```bash
# List available microphones
arecord -l

# Test microphone (replace hw:X,Y with your device)
arecord --device=hw:1,0 --format S16_LE --rate 24000 -c 1 test.wav
```

## Demo Usage

### Setup
```bash
# Clone repository
cd /path/to/pi-transcription

# Install dependencies
pip3 install -r requirements.txt

# Configure API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Run
```bash
python3 transcribe.py
```

### Expected Output
```
Starting audio transcription...
Connected to OpenAI Realtime API
Listening... (Press Ctrl+C to stop)

[TRANSCRIPT] Hello, this is a test of the transcription system.
[TRANSCRIPT] The weather is nice today.
[TRANSCRIPT] Testing one, two, three.

^C
Shutting down...
Audio stream closed
WebSocket connection closed
```

## Future Enhancements
- Save transcriptions to file with timestamps
- Support for multiple languages
- Client-side VAD to reduce bandwidth (stop streaming during silence)
- Local audio preprocessing (noise reduction)
- Web interface for remote monitoring
- Integration with larger software system (as mentioned in requirements)

### Project Modernization (End Goal)
- **Migrate to uv**: Replace pip with uv for faster, more reliable dependency management
- **Switch to pyproject.toml**: Migrate from requirements.txt to modern pyproject.toml format
  - Move all dependencies to `[project.dependencies]`
  - Move dev dependencies to `[project.optional-dependencies]`
  - Add project metadata (name, version, description, authors)
  - Configure ruff in `[tool.ruff]` section
  - Benefits: Single source of truth, better tooling support, modern Python packaging standard

## Success Criteria
- [ ] Successfully captures audio from USB microphone
- [ ] Establishes WebSocket connection to OpenAI
- [ ] Streams audio in real-time at 24kHz
- [ ] Receives and displays transcriptions in terminal
- [ ] Prints errors to console
- [ ] Proper cleanup on exit (Ctrl+C)
