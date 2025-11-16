import numpy as np
import pytest

from pi_transcription import wake_word


class DummyModel:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.models = {"dummy": None}
        self.predictions = []
        DummyModel.instances.append(self)

    def predict(self, samples):
        score = self.predictions.pop(0) if self.predictions else 0.0
        return {"dummy": score}


class DummyResampler:
    last_instance = None

    def __init__(self, *args, **kwargs):
        self.calls = []
        DummyResampler.last_instance = self

    def process(self, audio_bytes):
        self.calls.append(audio_bytes)
        if not audio_bytes:
            return np.array([], dtype=np.int16)
        return np.array([1, -1], dtype=np.int16)

    def reset(self):
        self.calls.clear()


@pytest.fixture(autouse=True)
def stub_wake_word_dependencies(monkeypatch):
    monkeypatch.setattr(wake_word, "Model", DummyModel)
    monkeypatch.setattr(wake_word, "LinearResampler", DummyResampler)
    DummyModel.instances.clear()
    DummyResampler.last_instance = None


def test_preroll_buffer_trim_and_flush():
    buffer = wake_word.PreRollBuffer(max_seconds=0.001, sample_rate=1000, sample_width=1)
    buffer.add(b"a")
    buffer.add(b"b" * 5)  # exceeds capacity, only most recent byte retained

    payload = buffer.flush()

    assert payload == b""
    assert buffer.flush() == b""


def test_preroll_buffer_clear_resets_state():
    buffer = wake_word.PreRollBuffer(max_seconds=0.001, sample_rate=1000, sample_width=1)
    buffer.add(b"abc")
    buffer.clear()
    assert buffer.flush() == b""


def test_wake_word_engine_triggers_after_consecutive_hits(tmp_path):
    model_path = tmp_path / "model.tflite"
    model_path.write_text("dummy")
    engine = wake_word.WakeWordEngine(
        model_path,
        threshold=0.5,
        consecutive_required=2,
        source_sample_rate=24000,
        target_sample_rate=16000,
    )

    dummy_model = DummyModel.instances[-1]
    dummy_model.predictions = [0.4, 0.6, 0.65, 0.3]

    chunk = np.zeros(400, dtype=np.int16).tobytes()
    first = engine.process_chunk(chunk)
    second = engine.process_chunk(chunk)
    third = engine.process_chunk(chunk)
    fourth = engine.process_chunk(chunk)

    assert first.triggered is False
    assert second.triggered is False
    assert third.triggered is True  # second consecutive hit
    assert fourth.triggered is False  # counter reset after trigger


def test_wake_word_engine_reset_detection(tmp_path):
    model_path = tmp_path / "model.tflite"
    model_path.write_text("dummy")
    engine = wake_word.WakeWordEngine(model_path, threshold=0.5, consecutive_required=2)
    dummy_model = DummyModel.instances[-1]
    dummy_model.predictions = [0.6, 0.6]

    engine.process_chunk(b"\x01\x02" * 10)
    engine.reset_detection()
    result = engine.process_chunk(b"\x01\x02" * 10)

    assert result.triggered is False


def test_load_model_uses_fallback_when_primary_missing(tmp_path):
    primary = tmp_path / "missing.tflite"
    fallback = tmp_path / "fallback.onnx"
    fallback.write_text("dummy")
    melspec = tmp_path / "melspec.onnx"
    embedding = tmp_path / "embedding.onnx"
    melspec.write_text("m")
    embedding.write_text("e")

    engine = wake_word.WakeWordEngine(
        primary,
        fallback_model_path=fallback,
        melspec_model_path=melspec,
        embedding_model_path=embedding,
    )

    dummy_model = DummyModel.instances[-1]
    assert str(fallback) in dummy_model.kwargs["wakeword_models"][0]
    assert dummy_model.kwargs["inference_framework"] == "onnx"
    assert dummy_model.kwargs["melspec_model_path"] == str(melspec)
    assert dummy_model.kwargs["embedding_model_path"] == str(embedding)
    assert isinstance(engine, wake_word.WakeWordEngine)
