"""Compatibility shim for the legacy start.py entry point."""

from pi_assistant.cli.app import main

if __name__ == "__main__":
    main()
