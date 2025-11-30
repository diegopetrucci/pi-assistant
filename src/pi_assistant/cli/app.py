"""
Real-time speech-to-text transcription system for Raspberry Pi 5
Streams audio from USB microphone to OpenAI's Realtime API
"""

import argparse
import asyncio
import sys
from functools import partial
from typing import Optional

from pi_assistant.assistant import (
    ASSISTANT_AUDIO_MODE_CHOICES,
    DEFAULT_ASSISTANT_AUDIO_MODE,
    TranscriptionComponentBuilder,
    TranscriptionConfigValidator,
    TranscriptionSession,
)
from pi_assistant.cli.events import handle_transcription_event
from pi_assistant.cli.logging import (
    ASSISTANT_LOG_LABEL,
    ERROR_LOG_LABEL,
    LOGGER,
    set_chunk_progress_logging,
    set_verbose_logging,
)
from pi_assistant.config import (
    ASSISTANT_MODEL,
    ASSISTANT_MODEL_REGISTRY,
    ASSISTANT_REASONING_CHOICES,
    normalize_assistant_model_choice,
    reasoning_effort_choices_for_model,
    reset_first_launch_choices,
)
from pi_assistant.core.exceptions import AssistantRestartRequired
from pi_assistant.diagnostics import test_audio_capture, test_websocket_client

REASONING_EFFORT_CHOICES = ("none", "minimal", "low", "medium", "high")
SIMULATED_QUERY_FALLBACK = "Hey Rhasspy, is it going to rain tomorrow?"


def _assistant_model_help() -> str:
    lines: list[str] = []
    for key, data in ASSISTANT_MODEL_REGISTRY.items():
        model_id = data["id"]
        description = data["description"]
        lines.append(f"  {key}: {description} [{model_id}]")
    return "\n".join(lines)


def _assistant_model_label(model_id: str) -> str:
    for key, data in ASSISTANT_MODEL_REGISTRY.items():
        if data["id"] == model_id:
            return f"{key} ({model_id})"
    return model_id


def _parse_assistant_model_arg(value: str) -> str:
    normalized = normalize_assistant_model_choice(value)
    if normalized:
        return normalized
    valid_tokens = sorted(
        list(ASSISTANT_MODEL_REGISTRY.keys())
        + [entry["id"] for entry in ASSISTANT_MODEL_REGISTRY.values()]
    )
    raise argparse.ArgumentTypeError(
        f"Unknown assistant model '{value}'. Choose from: {', '.join(valid_tokens)}."
    )


async def run_transcription(
    *,
    assistant_audio_mode: Optional[str] = None,
    simulate_query: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    assistant_model: Optional[str] = None,
) -> None:
    """
    Main integration function - runs real-time transcription
    Combines audio capture and WebSocket streaming
    """
    LOGGER.log("SYSTEM", "Starting real-time transcription")

    validator = TranscriptionConfigValidator()
    config = validator.resolve(
        assistant_audio_mode=assistant_audio_mode,
        simulate_query=simulate_query,
        reasoning_effort=reasoning_effort,
        assistant_model=assistant_model,
    )
    builder = TranscriptionComponentBuilder(config)
    components = builder.build()
    session = TranscriptionSession(config, components)

    try:
        async with session:
            await session.run()
    except KeyboardInterrupt:
        LOGGER.log("SYSTEM", "Shutdown requested")
    except AssistantRestartRequired as e:
        LOGGER.log(ASSISTANT_LOG_LABEL, str(e), error=True)
        raise SystemExit(1) from None
    except Exception as e:
        LOGGER.log(ERROR_LOG_LABEL, f"Transcription failed: {e}", error=True)
        raise


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
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed diagnostic logs (wake word, state changes, etc.).",
    )
    parser.add_argument(
        "--log-chunks",
        action="store_true",
        help="Emit per-100 chunk counters inside verbose logs (useful for throughput debugging).",
    )
    parser.add_argument(
        "--model",
        dest="assistant_model",
        type=_parse_assistant_model_arg,
        help=(
            "Override which assistant LLM to use for this run. "
            "Accepts the preset key (nano, mini, 4.1, 5.1) or the full model id.\n"
            f"{_assistant_model_help()}\n"
            f"Default: {ASSISTANT_MODEL}"
        ),
    )
    parser.add_argument(
        "--audio-mode",
        dest="assistant_audio_mode",
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
    supported_reasoning_label = (
        ", ".join(ASSISTANT_REASONING_CHOICES) if ASSISTANT_REASONING_CHOICES else "none (disabled)"
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        help=(
            "Override the GPT-5 reasoning level for this run. "
            "Defaults to your saved selection (low is recommended). "
            f"Supported for the current model: {supported_reasoning_label}. "
            "Models without reasoning (e.g., GPT-4.1) do not accept this flag. "
            "Note: 'minimal' requires web search to be disabled."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Remove saved first-launch answers (assistant model, reasoning effort, location) "
            "from .env and exit."
        ),
    )
    args = parser.parse_args()
    if args.reasoning_effort:
        selected_model = args.assistant_model or ASSISTANT_MODEL
        allowed = reasoning_effort_choices_for_model(selected_model)
        if args.reasoning_effort not in allowed:
            label = _assistant_model_label(selected_model)
            allowed_label = ", ".join(allowed) if allowed else "none (reasoning disabled)"
            parser.error(
                f"Reasoning effort '{args.reasoning_effort}' is not supported by {label}. "
                f"Allowed values: {allowed_label}."
            )
    return args


def main():
    """Main entry point"""

    args = parse_args()
    if getattr(args, "reset", False):
        cleared = sorted(reset_first_launch_choices())
        if cleared:
            LOGGER.log(
                ASSISTANT_LOG_LABEL,
                (
                    f"Cleared saved selections: {', '.join(cleared)}. "
                    "They will be requested again on next run."
                ),
            )
        else:
            LOGGER.log(
                ASSISTANT_LOG_LABEL,
                "No saved first-launch selections were present. Defaults will be used on next run.",
            )
        return

    set_verbose_logging(args.verbose)
    set_chunk_progress_logging(getattr(args, "log_chunks", False))
    if args.mode == "test-audio":
        run_func = test_audio_capture
    elif args.mode == "test-websocket":
        run_func = partial(test_websocket_client, handle_transcription_event)
    else:
        run_func = partial(
            run_transcription,
            assistant_audio_mode=args.assistant_audio_mode,
            simulate_query=args.simulate_query,
            reasoning_effort=args.reasoning_effort,
            assistant_model=args.assistant_model,
        )

    try:
        asyncio.run(run_func())
    except KeyboardInterrupt:
        LOGGER.log("SYSTEM", "Shutdown requested")
    except Exception as e:
        LOGGER.log(ERROR_LOG_LABEL, f"CLI error: {e}", error=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
