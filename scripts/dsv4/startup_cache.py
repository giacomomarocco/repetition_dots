"""Launcher entry point; cache publication belongs to the validation workflow."""
import sys
from filler.dsv4.startup_cache import launch

if __name__ == "__main__":
    command = sys.argv[1:]
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("usage: python -m scripts.dsv4.startup_cache -- COMMAND [ARG ...]")
    launch(command)
