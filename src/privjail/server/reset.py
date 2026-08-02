"""reset.py -- forcibly reset gateway state: unmount the encrypted directory
and kill any data server holding the configured port.

Mount state and data-server lifetime are currently managed loosely (gocryptfs
daemonizes itself, and the data server keeps running independent of any
particular ssh session), so things can end up stuck or inconsistent -- e.g.
a data server surviving with a stale view of an unmounted directory. This
gives a single command to tear all of that down so the next
`privjail-gateway` invocation starts clean.
"""
import argparse
import os
import subprocess
import sys


def unmount(data: str) -> bool:
    """Unmounts `data` if mounted. Returns True if it was mounted."""
    if not os.path.ismount(data):
        return False
    subprocess.run(["fusermount", "-u", data])
    if os.path.ismount(data):
        print(f"[reset] normal unmount of {data} failed, forcing lazy unmount", file=sys.stderr)
        subprocess.run(["fusermount", "-uz", data])
    return True


def kill_port(port: int) -> bool:
    """Kills whatever is listening on `port`. Returns True if anything was killed."""
    result = subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="mount point to unmount")
    parser.add_argument("--port", type=int, default=12345, help="privjail server port to clear")
    args = parser.parse_args()

    if unmount(args.data):
        print(f"[reset] unmounted {args.data}")
    else:
        print(f"[reset] {args.data} was not mounted")

    if kill_port(args.port):
        print(f"[reset] killed process(es) listening on port {args.port}")
    else:
        print(f"[reset] no process was listening on port {args.port}")


if __name__ == "__main__":
    main()
