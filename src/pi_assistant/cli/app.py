"""
Real-time speech-to-text transcription system for Raspberry Pi 5
Streams audio from USB microphone to OpenAI's Realtime API
"""

import argparse
import asyncio
import sys
from functools import partial
from typing import Optional

from pi_assistant.assistant import LLMResponder, TurnTranscriptAggregator
from pi_assistant.audio import AudioCapture, SpeechPlayer
from pi_assistant.cli.controller import run_audio_controller
from pi_assistant.cli.events import handle_transcription_event, receive_transcription_events
from pi_assistant.cli.logging_utils import (
    ASSISTANT_LOG_LABEL,
    ERROR_LOG_LABEL,
    TURN_LOG_LABEL,
    console_print,
    set_verbose_logging,
)
from pi_assistant.config import (
    ASSISTANT_TTS_RESPONSES_ENABLED,
    ASSISTANT_TTS_SAMPLE_RATE,
    CONFIRMATION_CUE_ENABLED,
    CONFIRMATION_CUE_TEXT,
    FORCE_ALWAYS_ON,
    SIMULATED_QUERY_TEXT,
)
from pi_assistant.diagnostics import test_audio_capture, test_websocket_client
from pi_assistant.network import WebSocketClient

ASSISTANT_AUDIO_MODE_CHOICES = ("responses", "local-tts")
DEFAULT_ASSISTANT_AUDIO_MODE = "responses" if ASSISTANT_TTS_RESPONSES_ENABLED else "local-tts"
SIMULATED_QUERY_FALLBACK = "Hey Rhasspy, is it going to rain tomorrow?"


async def _run_simulated_query_once(
    query_text: str,
    assistant: LLMResponder,
    speech_player: SpeechPlayer,
) -> None:
    """Inject a one-off synthetic transcript to drive the assistant."""

    cleaned = query_text.strip()
    if not cleaned:
        return

    console_print(f"{TURN_LOG_LABEL} Injecting simulated query: {cleaned}")
    try:
        reply = await assistant.generate_reply(cleaned)
    except Exception as exc:  # pragma: no cover - network failure
        console_print(f"{ASSISTANT_LOG_LABEL} Simulated query failed: {exc}", file=sys.stderr)
        return

    if not reply:
        console_print(f"{ASSISTANT_LOG_LABEL} Simulated query returned no response.")
        return

    if reply.text:
        console_print(f"{ASSISTANT_LOG_LABEL} {reply.text}")
    else:
        console_print(f"{ASSISTANT_LOG_LABEL} (no text content)")

    if reply.audio_bytes:
        try:
            await speech_player.play(reply.audio_bytes, sample_rate=reply.audio_sample_rate)
        except Exception as exc:  # pragma: no cover - host audio failure
            console_print(
                f"{ASSISTANT_LOG_LABEL} Error playing simulated reply audio: {exc}",
                file=sys.stderr,
            )


async def run_transcription(
    *,
    force_always_on: bool = False,
    assistant_audio_mode: Optional[str] = None,
    simulate_query: Optional[str] = None,
) -> None:
    """
    Main integration function - runs real-time transcription
    Combines audio capture and WebSocket streaming
    """
    print("\n=== Starting Real-Time Transcription ===\n")

    # Create instances
    audio_capture = AudioCapture()
    ws_client = WebSocketClient()
    transcript_buffer = TurnTranscriptAggregator()
    audio_mode = assistant_audio_mode or DEFAULT_ASSISTANT_AUDIO_MODE
    use_responses_audio = audio_mode == "responses"
    configured_query = (
        simulate_query.strip() if simulate_query is not None else SIMULATED_QUERY_TEXT.strip()
    )

    assistant = LLMResponder(use_responses_audio=use_responses_audio)
    console_print(f"{ASSISTANT_LOG_LABEL} Using assistant model: {assistant.model_name}")
    enabled_tools = assistant.enabled_tools
    tools_summary = ", ".join(enabled_tools) if enabled_tools else "none"
    console_print(f"{ASSISTANT_LOG_LABEL} Tools enabled: {tools_summary}")
    speech_player = SpeechPlayer(default_sample_rate=ASSISTANT_TTS_SAMPLE_RATE)
    if assistant.tts_enabled and CONFIRMATION_CUE_ENABLED and CONFIRMATION_CUE_TEXT:
        cue_task = asyncio.create_task(assistant.warm_phrase_audio(CONFIRMATION_CUE_TEXT))

        def _log_cue_error(fut: asyncio.Task):
            try:
                fut.result()
            except Exception:
                pass

        cue_task.add_done_callback(_log_cue_error)

    responses_audio_enabled = False
    if assistant.tts_enabled and use_responses_audio:
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
        if use_responses_audio:
            print(f"{ASSISTANT_LOG_LABEL} Responses audio not available; using Audio API for TTS.")
        else:
            print(
                f"{ASSISTANT_LOG_LABEL} Local TTS mode active; "
                "synthesizing replies after receiving text."
            )

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
        simulated_query_task = None
        if configured_query:
            simulated_query_task = asyncio.create_task(
                _run_simulated_query_once(configured_query, assistant, speech_player)
            )

        # Wait for both tasks (they run until cancelled)
        pending = [audio_task, event_task]
        if simulated_query_task:
            pending.append(simulated_query_task)
        await asyncio.gather(*pending)

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
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed diagnostic logs (wake word, state changes, etc.).",
    )
    parser.add_argument(
        "--assistant-audio-mode",
        choices=ASSISTANT_AUDIO_MODE_CHOICES,
        help=(
            "How to deliver assistant replies: 'responses' streams audio directly from "
            "OpenAI, 'local-tts' fetches text then runs client-side TTS. "
            f"Defaults to {DEFAULT_ASSISTANT_AUDIO_MODE} "
            "(override via ASSISTANT_TTS_RESPONSES_ENABLED)."
        ),
    )
    parser.add_argument(
        "--simulate-query",
        nargs="?",
        const=SIMULATED_QUERY_FALLBACK,
        help=(
            "Inject a single synthetic transcript at startup. "
            "Pass custom text or omit it to send "
            f"{SIMULATED_QUERY_FALLBACK!r}. Defaults to the SIMULATED_QUERY_TEXT env var when set."
        ),
    )
    return parser.parse_args()


def main():
    """Main entry point"""

    args = parse_args()
    set_verbose_logging(args.verbose)
    if args.mode == "test-audio":
        run_func = test_audio_capture
    elif args.mode == "test-websocket":
        run_func = partial(test_websocket_client, handle_transcription_event)
    else:
        force_flag = FORCE_ALWAYS_ON if args.force_always_on is None else args.force_always_on
        run_func = partial(
            run_transcription,
            force_always_on=force_flag,
            assistant_audio_mode=args.assistant_audio_mode,
            simulate_query=args.simulate_query,
        )

    try:
        asyncio.run(run_func())
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
