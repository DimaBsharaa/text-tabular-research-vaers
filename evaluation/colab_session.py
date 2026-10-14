"""Stream subprocess diagnostics into the notebook and optional persistent log."""
from contextlib import nullcontext
from pathlib import Path
import subprocess


def run_logged_command(args, log_path=None, **kwargs):
    args = list(map(str, args))
    print(" ".join(args), flush=True)
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    with (log_path.open("a") if log_path is not None else nullcontext()) as log:
        with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace", bufsize=1, **kwargs) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                lines.append(line)
                if log is not None:
                    log.write(line)
                    log.flush()
            returncode = process.wait()
    if returncode:
        if log_path is not None:
            print(f"Full command output saved to {log_path}", flush=True)
        raise subprocess.CalledProcessError(returncode, args, output="".join(lines))
    return subprocess.CompletedProcess(args, returncode, stdout="".join(lines))
