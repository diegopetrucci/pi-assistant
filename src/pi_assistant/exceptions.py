"""Custom exception types shared across the Pi Assistant package."""


class AssistantRestartRequired(RuntimeError):
    """Raised when the assistant must be relaunched to apply config changes."""
