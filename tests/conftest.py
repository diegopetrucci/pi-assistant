import importlib
import sys
import types

try:
    importlib.import_module("audioop")
except ModuleNotFoundError:
    stub = types.SimpleNamespace(
        ratecv=lambda audio_bytes, width, channels, src_rate, dst_rate, state: (audio_bytes, state),
    )
    sys.modules["audioop"] = stub  # pyright: ignore[reportArgumentType]
