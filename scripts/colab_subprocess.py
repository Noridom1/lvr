"""Small Colab helper for streaming subprocess output with useful failures."""

from collections import deque
from pathlib import Path
import shlex
import subprocess
from typing import Mapping, Optional, Sequence, Union


CommandPart = Union[str, Path]


def run_streaming(
    command: Sequence[CommandPart],
    *,
    cwd: Optional[CommandPart] = None,
    env: Optional[Mapping[str, str]] = None,
    tail_lines: int = 100,
) -> None:
    """Run a command, echo merged output live, and include its tail on failure."""
    normalized_command = [str(part) for part in command]
    display_command = shlex.join(normalized_command)
    print(f"Running: {display_command}", flush=True)

    recent_output = deque(maxlen=tail_lines)
    process = subprocess.Popen(
        normalized_command,
        cwd=None if cwd is None else str(cwd),
        env=None if env is None else dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        recent_output.append(line)

    return_code = process.wait()
    if return_code:
        tail = "".join(recent_output)
        raise RuntimeError(
            f"Command failed with exit code {return_code}: {display_command}\n\n"
            f"Last {len(recent_output)} output lines:\n{tail}"
        )
    print(f"Completed successfully: {display_command}", flush=True)
