# Test Coverage Analysis and TODO

## 1. Critical Files with NO Tests

### Untested Files (551 total lines)

#### 1. `src/pi_transcription/config/__init__.py` (246 lines)
**Priority**: ⚠️ **HIGHEST - CRITICAL**

**Functions lacking tests**:
- [ ] `_env_bool()` - Boolean environment variable parsing
- [ ] `_env_int()` - Integer environment variable parsing
- [ ] `_env_float()` - Float environment variable parsing
- [ ] `_env_path()` - Path environment variable parsing
- [ ] `_prompt_for_api_key()` - Interactive API key prompting
- [ ] `_persist_env_value()` - .env file persistence
- [ ] `_prompt_for_location_name()` - Interactive location prompting
- [ ] `get_config()` - Main configuration loader

**Why critical**: Foundation for entire app configuration - errors here break everything

**Risk level**: Very High

#### 2. `src/pi_transcription/cli/app.py` (165 lines)
**Priority**: ⚠️ **VERY HIGH - CRITICAL**

**Functions lacking tests**:
- [ ] `parse_args()` - CLI argument parsing
- [ ] `run_transcription()` - Main application orchestration
- [ ] `main()` - Entry point

**Why critical**: Entry point for entire application, orchestrates all components

**Risk level**: Very High

#### 3. `src/pi_transcription/diagnostics.py` (102 lines)
**Priority**: 🟡 **MEDIUM**

**Functions lacking tests**:
- [ ] `test_audio_capture()` - Hardware audio validation
- [ ] `test_websocket_client()` - Network connection validation

**Why critical**: Diagnostic utilities for hardware validation

**Risk level**: High - used for troubleshooting

#### 4. `src/pi_transcription/cli/logging_utils.py` (38 lines)
**Priority**: 🟢 **LOW**

**Functions lacking tests**:
- [ ] `log_state_transition()` - State logging formatter

**Why critical**: Used throughout for debugging

**Risk level**: Medium - mostly formatting

---

## 3. Edge Cases and Gaps in Existing Tests

### Configuration Module Edge Cases
**File**: `src/pi_transcription/config/__init__.py`

Missing tests for:
- _none_ ✅

### Assistant Module Edge Cases
**File**: `src/pi_transcription/assistant.py`

Gaps identified:
- _none so far_ (new gaps welcome)

### Audio Capture Edge Cases
**File**: `src/pi_transcription/audio/capture.py`

Missing coverage:
- _none so far_

### Wake Word Engine Edge Cases
**File**: `src/pi_transcription/wake_word.py`

Gaps:
- ❌ `process_chunk()` with very large audio chunks
- ❌ Model loading with corrupt model files (exception path covered but not tested)

### CLI Controller Edge Cases
**File**: `src/pi_transcription/cli/controller.py`

Critical gaps:
- ❌ `run_audio_controller()` exception handling paths (marked `pragma: no cover`)
- ❌ Concurrent wake word triggers during streaming
- ❌ Stop signal during pre-roll flush
- ❌ Speech stopped signal timing edge cases
- ❌ Auto-stop with retrigger budget at boundary conditions
- ❌ RMS calculation with corrupted audio data
- ❌ `finalize_turn_and_respond()` network failures (covered by pragma but not tested)
- ❌ Task cleanup on cancellation

### WebSocket Client Edge Cases
**File**: `src/pi_transcription/network/websocket_client.py`

Missing:
- ❌ `connect()` timeout behavior
- ❌ `wait_for_session_created()` receiving error events
- ❌ `send_audio_chunk()` when not connected
- ❌ `receive_events()` handling malformed JSON
- ❌ Connection drops during `send_session_config()`
- ❌ Multiple connection attempts

### CLI Events Edge Cases
**File**: `src/pi_transcription/cli/events.py`

Gaps:
- ❌ `handle_transcription_event()` with unknown event types
- ❌ Event with missing required fields
- ❌ `receive_transcription_events()` exception handling (marked `pragma: no cover`)
- ❌ Malformed transcript text in stop command detection

### Speech Player Edge Cases
**File**: `src/pi_transcription/audio/playback.py`

Missing:
- ❌ `_play_blocking()` exception path (marked `pragma: no cover`)
- ❌ Invalid sample rate handling (≤ 0)
- ❌ Empty audio bytes playback
- ❌ Device detection failures
- ❌ `_is_rate_supported()` with various edge cases

---

## 4. High-Priority Areas Needing Tests

### Priority 1: 🔴 CRITICAL (Start Here)

#### 1. Configuration Module (`config/__init__.py`)
**Estimated effort**: 2-3 days

**Tests needed**:
- [ ] Test all environment variable parsing functions
-  - [x] `_env_bool()` with valid/invalid values
-  - [x] `_env_int()` with valid/invalid/overflow values
-  - [x] `_env_float()` with valid/invalid/special values
-  - [x] `_env_path()` with absolute/relative/`~` paths
- [x] Test path resolution with various inputs
- [x] Test .env file persistence
  - [x] Create new .env file
  - [x] Update existing .env file
  - [x] Handle permission errors
- [x] Test API key validation
- [x] Mock interactive prompts
  - [x] API key prompting
  - [x] Location name prompting

#### 2. Main Application Entry Point (`cli/app.py`)
**Estimated effort**: 1-2 days

**Tests needed**:
- [ ] Test `parse_args()` with various CLI flag combinations
  - [x] All valid flag combinations
  - [x] Invalid flag combinations
  - [x] Missing required arguments (N/A — parser has no required positional args)
- [ ] Test `run_transcription()` initialization flow
  - [x] Successful startup
  - [x] Component initialization failures
- [x] Test graceful shutdown on KeyboardInterrupt
- [x] Test error propagation from components
- [x] Mock all external dependencies

#### 3. Audio Capture Error Paths (`audio/capture.py`)
**Estimated effort**: 1 day

**Tests needed**:
- [ ] Test device initialization failures
- [x] Test device initialization failures
- [ ] Test all device selection edge cases
  - [x] When override device doesn't exist
  - [x] When no devices available
  - [x] When query_devices() raises exception
- [x] Test `stop_stream()` when stream is None
- [x] Test callback status warnings

### Priority 2: 🟡 HIGH (Important for Robustness)

#### 4. WebSocket Connection Edge Cases (`network/websocket_client.py`)
**Estimated effort**: 1-2 days

**Tests needed**:
- [ ] Test connection timeouts
- [x] Test connection timeouts
- [x] Test reconnection scenarios
- [x] Test malformed server responses
- [x] Test error event handling
- [x] Test `send_audio_chunk()` when not connected
- [x] Test multiple connection attempts

#### 5. CLI Controller Exception Handling (`cli/controller.py`)
**Estimated effort**: 2-3 days

**Tests needed**:
- [x] Test concurrent state transitions
- [x] Test all cancellation paths
- [x] Test resource cleanup
- [x] Test edge cases in auto-stop logic
- [x] Test retrigger scenarios
- [x] Test concurrent wake word triggers
- [x] Test stop signal during pre-roll flush

#### 6. Assistant API Error Handling (`assistant.py`)
**Estimated effort**: 1-2 days

**Tests needed**:
- [ ] Test concurrent reply generation
- [ ] Test API rate limiting scenarios
- [ ] Test very large transcripts (>10k chars)
- [ ] Test network failures during synthesis
- [ ] Test race conditions during finalization
- [ ] Test empty transcript handling

### Priority 3: 🟢 MEDIUM (Nice to Have)

#### 7. Diagnostics Module (`diagnostics.py`)
**Estimated effort**: 0.5-1 day

**Tests needed**:
- [ ] Test `test_audio_capture()` function
- [ ] Test `test_websocket_client()` function
- [ ] Mock hardware dependencies
- [ ] Test timeout behaviors

#### 8. Wake Word Edge Cases (`wake_word.py`)
**Estimated effort**: 1 day

**Tests needed**:
- [ ] Test model loading failures
- [ ] Test buffer overflow scenarios
- [ ] Test zero-length audio handling
- [ ] Test `PreRollBuffer` with invalid max_seconds
- [ ] Test very large audio chunks

#### 9. Integration Tests
**Estimated effort**: 2-3 days

**Tests needed**:
- [ ] Full pipeline from audio capture to transcription
- [ ] State machine transitions
- [ ] Error recovery flows
- [ ] End-to-end wake word to response flow

---

## 5. Specific Edge Cases to Test

### Null/Empty/Boundary Values
- [ ] Empty strings in all text processing functions
- [ ] None values in optional parameters
- [ ] Zero-length audio buffers
- [ ] Maximum queue sizes
- [ ] Very long transcripts (>10k characters)
- [ ] Unicode and special characters in transcripts
- [ ] Whitespace-only inputs

### Error Conditions
- [ ] API key missing or invalid
- [ ] Network timeouts and failures
- [ ] WebSocket disconnections mid-stream
- [ ] Audio device not found
- [ ] Corrupt model files
- [ ] File permission errors during .env persistence
- [ ] Out-of-memory scenarios with large audio buffers

### Race Conditions
- [ ] Concurrent transcription finalization
- [ ] Stop signal during state transition
- [ ] Multiple wake word triggers
- [ ] Audio queue overflow during high load
- [ ] Concurrent assistant reply generation

### Async/Promise Handling
- [ ] Task cancellation during long operations
- [ ] Exception propagation in fire-and-forget tasks
- [ ] Cleanup in finally blocks
- [ ] Event loop shutdown with pending tasks

### Validation Logic
- [ ] Config value range checks (sample rates, thresholds)
- [ ] Path validation (relative, absolute, ~ expansion)
- [ ] Device index validation
- [ ] Audio format validation

---

## 6. Recommended Testing Strategy

### Phase 1: Critical Gaps (Weeks 1-2)
**Goal**: Cover the most critical untested code

1. **Week 1**: Configuration module
   - [ ] Create `tests/test_config.py`
   - [ ] Test all environment variable parsing functions
   - [ ] Test .env file persistence
   - [ ] Test interactive prompts (mocked)
   - [ ] **Target**: 90%+ coverage of `config/__init__.py`

2. **Week 2**: Entry points and error paths
   - [ ] Create `tests/test_app.py`
   - [ ] Test CLI argument parsing
   - [ ] Test main application flow
   - [ ] Add audio capture error path tests
   - [ ] **Target**: 80%+ coverage of `cli/app.py`

### Phase 2: Robustness (Weeks 3-4)
**Goal**: Improve error handling and edge case coverage

3. **Week 3**: Network and async edge cases
   - [ ] Expand WebSocket client tests
   - [ ] Add CLI controller concurrent tests
   - [ ] Test async error handling
   - [ ] **Target**: Remove most `pragma: no cover` comments

4. **Week 4**: Assistant and integration
   - [ ] Add assistant edge case tests
   - [ ] Create integration test suite
   - [ ] Test state machine transitions
   - [ ] **Target**: End-to-end scenarios covered

### Phase 3: Polish (Week 5)
**Goal**: Achieve comprehensive coverage

5. **Week 5**: Final coverage push
   - [ ] Add diagnostics tests
   - [ ] Add performance/stress tests
   - [ ] Fill remaining coverage gaps
   - [ ] **Target**: 85%+ overall code coverage

---

## 7. Testing Best Practices

### Current Strengths ✓
- [ ] Good use of pytest fixtures
- [ ] Mocking external dependencies (OpenAI, sounddevice)
- [ ] Async test support with pytest-asyncio
- [ ] Separation of manual/hardware tests
- [ ] Clear test naming conventions

### Areas for Improvement
- [ ] Add property-based testing for config parsing (hypothesis library)
- [ ] Add integration tests using real audio fixtures
- [ ] Increase error path coverage (many `pragma: no cover` comments)
- [ ] Add stress tests for concurrent operations
- [ ] Consider adding contract tests for OpenAI API interactions
- [ ] Add mutation testing to verify test quality
- [ ] Set up coverage reporting in CI/CD
- [ ] Document testing patterns and conventions

---

## 8. Coverage Metrics Goal

### Current Estimate
- [ ] **Lines covered**: ~60-70% (estimated, needs measurement)
- [ ] **Critical paths covered**: ~50%
- [ ] **Error paths covered**: ~30%

### Target Metrics
- [ ] **Overall coverage**: 85%+
- [ ] **Critical modules**: 90%+ (`config`, `cli/app`, `assistant`, `controller`)
- [ ] **Error paths**: 70%+
- [ ] **Integration coverage**: At least 5 end-to-end scenarios

---

## 9. Tools and Infrastructure

### Required Tools
- [x] pytest
- [x] pytest-asyncio
- [x] pytest-cov
- [ ] pytest-timeout (for timeout testing)
- [ ] pytest-mock (enhanced mocking)
- [ ] hypothesis (property-based testing)
- [ ] pytest-xdist (parallel test execution)

### CI/CD Integration
- [ ] Add coverage reporting to CI pipeline
- [ ] Set minimum coverage threshold (start at 70%, increase to 85%)
- [ ] Generate HTML coverage reports
- [ ] Fail builds on coverage decrease
- [ ] Add mutation testing (optional)

---

## 10. Quick Start Guide

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/pi_transcription --cov-report=html

# Run specific test file
pytest tests/test_assistant.py

# Run tests matching pattern
pytest -k "test_config"

# Run with verbose output
pytest -v
```

### Writing New Tests
- [ ] 1. Create test file in `tests/` matching source file name
- [ ] 2. Use appropriate fixtures from `conftest.py`
- [ ] 3. Mock external dependencies (OpenAI API, sounddevice, etc.)
- [ ] 4. Test both success and error paths
- [ ] 5. Include edge cases (null, empty, boundary values)
- [ ] 6. Use descriptive test names: `test_function_name_with_condition_expects_result`

### Example Test Template
```python
import pytest
from unittest.mock import Mock, patch
from pi_transcription.module import function_to_test

def test_function_success_case():
    """Test normal operation of function."""
    result = function_to_test(valid_input)
    assert result == expected_output

def test_function_with_invalid_input():
    """Test function handles invalid input gracefully."""
    with pytest.raises(ValueError):
        function_to_test(invalid_input)

@pytest.mark.asyncio
async def test_async_function():
    """Test async function behavior."""
    result = await async_function()
    assert result is not None
```

---

## 11. Progress Tracking

### Completed ✓
- [ ] Initial test infrastructure setup
- [ ] Core functionality tests for assistant, wake word, audio

### In Progress 🔄
- [ ] (None currently)

### TODO 📋
- [ ] Create `tests/test_config.py` (Priority 1)
- [ ] Create `tests/test_app.py` (Priority 1)
- [ ] Expand `tests/test_audio_capture.py` with error paths (Priority 1)
- [ ] Expand `tests/test_network_websocket_client.py` (Priority 2)
- [ ] Expand `tests/test_cli_controller_run.py` (Priority 2)
- [ ] Expand `tests/test_assistant.py` with edge cases (Priority 2)
- [ ] Create `tests/test_diagnostics.py` (Priority 3)
- [ ] Create integration test suite (Priority 3)
- [ ] Set up coverage reporting in CI (Infrastructure)
- [ ] Add property-based tests (Enhancement)

---

## 12. Notes and Observations

### Code Quality Observations
- [ ] Many exception handlers use generic `except Exception` which makes testing harder
- [ ] Some areas marked with `pragma: no cover` could be tested with better mocking
- [ ] Good separation of concerns makes unit testing easier
- [ ] Async code is well-structured for testing

### Testing Challenges
- [ ] Hardware dependencies (audio devices) require careful mocking
- [ ] Interactive prompts need special handling in tests
- [ ] WebSocket connection testing requires async mocking
- [ ] State machine testing needs careful fixture setup

### Recommendations for New Code
- [ ] Write tests alongside new features (TDD approach)
- [ ] Keep functions small and testable
- [ ] Avoid global state
- [ ] Use dependency injection for external dependencies
- [ ] Document expected behavior in docstrings

---

## Appendix: File Reference

### Critical Untested Files
- [ ] 1. `src/pi_transcription/config/__init__.py` (246 lines)
- [ ] 2. `src/pi_transcription/cli/app.py` (165 lines)
- [ ] 3. `src/pi_transcription/diagnostics.py` (102 lines)
- [ ] 4. `src/pi_transcription/cli/logging_utils.py` (38 lines)

### Partially Tested Files Needing Expansion
- [ ] 1. `src/pi_transcription/assistant.py`
- [ ] 2. `src/pi_transcription/audio/capture.py`
- [ ] 3. `src/pi_transcription/cli/controller.py`
- [ ] 4. `src/pi_transcription/network/websocket_client.py`
- [ ] 5. `src/pi_transcription/cli/events.py`
- [ ] 6. `src/pi_transcription/audio/playback.py`
- [ ] 7. `src/pi_transcription/wake_word.py`

### Well-Tested Files (Maintain Coverage)
- [ ] 1. `src/pi_transcription/audio/resampler.py`
- [ ] 2. Test utilities and fixtures

---

**Last Updated**: 2025-11-16
**Next Review**: After Phase 1 completion
