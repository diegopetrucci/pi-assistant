# Test Coverage Analysis and TODO

## 1. Critical Files with NO Tests

### Untested Files (551 total lines)

#### 1. `src/pi_assistant/cli/logging.py` (38 lines)
**Priority**: 🟢 **LOW**

**Functions lacking tests**:
- [ ] `log_state_transition()` - State logging formatter

**Why critical**: Used throughout for debugging

**Risk level**: Medium - mostly formatting

---

## 3. Edge Cases and Gaps in Existing Tests

### Configuration Module Edge Cases
**File**: `src/pi_assistant/config/__init__.py`

Missing tests for:
- _none_ ✅

### Assistant Module Edge Cases
**File**: `src/pi_assistant/assistant.py`

Gaps identified:
- _none so far_ (new gaps welcome)

### Audio Capture Edge Cases
**File**: `src/pi_assistant/audio/capture.py`

Missing coverage:
- _none so far_

### Wake Word Engine Edge Cases
**File**: `src/pi_assistant/audio/wake_word/engine.py`

Gaps:
- ❌ `process_chunk()` with very large audio chunks
- ❌ Model loading with corrupt model files (exception path covered but not tested)

### CLI Controller Edge Cases
**File**: `src/pi_assistant/cli/controller.py`

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
**File**: `src/pi_assistant/network/websocket_client.py`

Missing:
- ❌ `connect()` timeout behavior
- ❌ `wait_for_session_created()` receiving error events
- ❌ `send_audio_chunk()` when not connected
- ❌ `receive_events()` handling malformed JSON
- ❌ Connection drops during `send_session_config()`
- ❌ Multiple connection attempts

### CLI Events Edge Cases
**File**: `src/pi_assistant/cli/events.py`

Gaps:
- ❌ `handle_transcription_event()` with unknown event types
- ❌ Event with missing required fields
- ❌ `receive_transcription_events()` exception handling (marked `pragma: no cover`)
- ❌ Malformed transcript text in stop command detection

### Speech Player Edge Cases
**File**: `src/pi_assistant/audio/playback.py`

Missing:
- ❌ `_play_blocking()` exception path (marked `pragma: no cover`)
- ❌ Invalid sample rate handling (≤ 0)
- ❌ Empty audio bytes playback
- ❌ Device detection failures
- ❌ `_is_rate_supported()` with various edge cases

---

## 4. High-Priority Areas Needing Tests

### Priority 1: 🔴 CRITICAL (Start Here)

#### 1. Audio Capture Error Paths (`audio/capture.py`)
**Estimated effort**: 1 day

**Tests needed**:
- [ ] Test device initialization failures
- [ ] Test all device selection edge cases

### Priority 2: 🟡 HIGH (Important for Robustness)

#### 1. WebSocket Connection Edge Cases (`network/websocket_client.py`)
**Estimated effort**: 1-2 days

**Tests needed**:
- [ ] Test connection timeouts

#### 2. Assistant API Error Handling (`assistant.py`)
**Estimated effort**: 1-2 days

**Tests needed**:
- [ ] Test concurrent reply generation
- [ ] Test API rate limiting scenarios
- [ ] Test very large transcripts (>10k chars)
- [ ] Test network failures during synthesis
- [ ] Test race conditions during finalization
- [ ] Test empty transcript handling

### Priority 3: 🟢 MEDIUM (Nice to Have)

#### 1. Wake Word Edge Cases (`audio/wake_word/engine.py`)
**Estimated effort**: 1 day

**Tests needed**:
- [ ] Test model loading failures
- [ ] Test buffer overflow scenarios
- [ ] Test zero-length audio handling
- [ ] Test `PreRollBuffer` with invalid max_seconds
- [ ] Test very large audio chunks

#### 2. Integration Tests
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
**Goal**: Finish the outstanding blockers that still lack basic regression coverage.

1. **Week 1**: Audio capture + logging
   - [ ] Cover device initialization/selection failures in `audio/capture.py`.
   - [ ] Exercise `cli/logging.log_state_transition()` with representative inputs.
   - [ ] **Target**: Remove remaining “no tests” entries for audio + logging helpers.

2. **Week 2**: Realtime connectivity
   - [ ] Simulate WebSocket connection timeouts/drops in `network/websocket_client.py`.
   - [ ] Add negative-path tests for assistant synthesis/network failures.
   - [ ] **Target**: Reduce crash-only code paths in networking + assistant layers.

### Phase 2: Robustness (Weeks 3-4)
**Goal**: Broaden error-path coverage for language + wake-word subsystems.

3. **Week 3**: Wake word + speech playback
   - [ ] Stress-test `audio/wake_word/engine.py` with oversize, zero-length, and corrupt buffers.
   - [ ] Cover playback failure scenarios (invalid sample rates, device issues).
   - [ ] **Target**: Confidence in on-device gating + playback fallbacks.

4. **Week 4**: Integration + controller flows
   - [ ] Build an integration harness from audio ingest through transcription events.
   - [ ] Exercise stop/retrigger boundaries and multi-turn cleanup in the controller.
   - [ ] **Target**: Stable end-to-end regression that survives retries/cancellations.

### Phase 3: Polish (Week 5)
**Goal**: Lock in coverage goals and regression tooling.

5. **Week 5**: Coverage consolidation
   - [ ] Backfill assistant edge cases (rate limits, large transcripts, empty turns).
   - [ ] Add state-machine + queue stress tests for async/race conditions.
   - [ ] **Target**: Sustain 85%+ overall coverage with CI gating.

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
- pytest
- pytest-asyncio
- pytest-cov
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
pytest --cov=src/pi_assistant --cov-report=html

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
from pi_assistant.module import function_to_test

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
- [ ] 1. `src/pi_assistant/config/__init__.py` (246 lines)
- [ ] 2. `src/pi_assistant/cli/app.py` (165 lines)
- [ ] 3. `src/pi_assistant/diagnostics.py` (102 lines)
- [ ] 4. `src/pi_assistant/cli/logging.py` (38 lines)

### Partially Tested Files Needing Expansion
- [ ] 1. `src/pi_assistant/assistant.py`
- [ ] 2. `src/pi_assistant/audio/capture.py`
- [ ] 3. `src/pi_assistant/cli/controller.py`
- [ ] 4. `src/pi_assistant/network/websocket_client.py`
- [ ] 5. `src/pi_assistant/cli/events.py`
- [ ] 6. `src/pi_assistant/audio/playback.py`
- [ ] 7. `src/pi_assistant/audio/wake_word/engine.py`

### Well-Tested Files (Maintain Coverage)
- [ ] 1. `src/pi_assistant/audio/resampler.py`
- [ ] 2. Test utilities and fixtures

---

**Last Updated**: 2025-11-16
**Next Review**: After Phase 1 completion
