"""
Real-time speech-to-text transcription system for Raspberry Pi 5
Streams audio from USB microphone to OpenAI's Realtime API
"""

import argparse
import asyncio
import sys
from functools import partial

from pi_transcription.assistant import LLMResponder, TurnTranscriptAggregator
from pi_transcription.audio import AudioCapture, SpeechPlayer
from pi_transcription.cli.controller import run_audio_controller
from pi_transcription.cli.events import handle_transcription_event, receive_transcription_events
from pi_transcription.cli.logging_utils import ASSISTANT_LOG_LABEL, ERROR_LOG_LABEL
from pi_transcription.config import ASSISTANT_TTS_SAMPLE_RATE, FORCE_ALWAYS_ON
from pi_transcription.diagnostics import test_audio_capture, test_websocket_client
from pi_transcription.network import WebSocketClient


async def run_transcription(force_always_on: bool = False) -> None:
    """
    Main integration function - runs real-time transcription
    Combines audio capture and WebSocket streaming
    """
    print("\n=== Starting Real-Time Transcription ===\n")

    # Create instances
    audio_capture = AudioCapture()
    ws_client = WebSocketClient()
    transcript_buffer = TurnTranscriptAggregator()
    assistant = LLMResponder()
    speech_player = SpeechPlayer(default_sample_rate=ASSISTANT_TTS_SAMPLE_RATE)

    responses_audio_enabled = False
    if assistant.tts_enabled:
        try:
            responses_audio_enabled = await assistant.verify_responses_audio_support()
        except Exception as exc:
            assistant.set_responses_audio_supported(False)
            print(
                f"{ASSISTANT_LOG_LABEL} Unable to verify Responses audio support: {exc}",
                file=sys.stderr,
            )

    if responses_audio_enabled:
        print(f"{ASSISTANT_LOG_LABEL} Responses audio enabled; streaming assistant replies.")
    elif assistant.tts_enabled:
        print(f"{ASSISTANT_LOG_LABEL} Responses audio not available; using Audio API for TTS.")

    # Get current event loop
    loop = asyncio.get_running_loop()

    try:
        # Connect to OpenAI via WebSocket
        await ws_client.connect()

        # Start audio stream
        audio_capture.start_stream(loop)

        print("\n✓ System ready")
        print("Listening... (Press Ctrl+C to stop)\n")

        # Create concurrent tasks for audio streaming and event receiving
        stop_signal = asyncio.Event()
        speech_stopped_signal = asyncio.Event()

        audio_task = asyncio.create_task(
            run_audio_controller(
                audio_capture,
                ws_client,
                force_always_on=force_always_on,
                transcript_buffer=transcript_buffer,
                assistant=assistant,
                speech_player=speech_player,
                stop_signal=stop_signal,
                speech_stopped_signal=speech_stopped_signal,
            )
        )
        event_task = asyncio.create_task(
            receive_transcription_events(
                ws_client,
                transcript_buffer,
                speech_player,
                stop_signal=stop_signal,
                speech_stopped_signal=speech_stopped_signal,
            )
        )

        # Wait for both tasks (they run until cancelled)
        await asyncio.gather(audio_task, event_task)

    except KeyboardInterrupt:
        print("\n\nShutdown requested...")

    except Exception as e:
        print(f"\n{ERROR_LOG_LABEL} Transcription failed: {e}", file=sys.stderr)
        raise

    finally:
        # Graceful shutdown
        print("Cleaning up...")

        # Stop audio stream
        audio_capture.stop_stream()

        # Close WebSocket connection
        await ws_client.close()

        print("✓ Shutdown complete\n")


def parse_args():
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="OpenAI transcription client with wake-word gating."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["run", "test-audio", "test-websocket"],
        default="run",
        help="Select an execution mode (default: run)",
    )
    parser.set_defaults(force_always_on=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--force-always-on",
        action="store_true",
        dest="force_always_on",
        help="Bypass wake-word gating and stream audio continuously.",
    )
    group.add_argument(
        "--no-force-always-on",
        action="store_false",
        dest="force_always_on",
        help="Explicitly disable the wake-word override even if the env var is set.",
    )
    return parser.parse_args()


def main():
    """Main entry point"""

    args = parse_args()
    if args.mode == "test-audio":
        run_func = test_audio_capture
    elif args.mode == "test-websocket":
        run_func = partial(test_websocket_client, handle_transcription_event)
    else:
        force_flag = FORCE_ALWAYS_ON if args.force_always_on is None else args.force_always_on
        run_func = partial(run_transcription, force_always_on=force_flag)

    try:
        asyncio.run(run_func())
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
