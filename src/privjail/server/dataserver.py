"""dataserver.py -- brings up the privjail server-side process, listening
for privjail clients on the given port.

  python -m privjail.server.dataserver [PORT] [--root MOUNT PATH ...]

With --root (as launched by spawn_session / gateway.py), PATH appears to the
client as its absolute local directory MOUNT, and only files under the
mounts can be opened (see privjail.datapath); without it, paths are used as
given.
"""
import argparse
import io
import sys

import privjail as pj
from privjail.datapath import set_roots


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("port", type=int, nargs="?", default=0, help="port to listen on (0 = any)")
    parser.add_argument("--root", action="append", nargs=2, default=None, metavar=("MOUNT", "PATH"),
                        help="expose directory PATH to clients as their absolute path MOUNT (repeatable)")
    args = parser.parse_args()
    if args.root is not None:
        for mount, _ in args.root:
            if not mount.startswith("/"):
                parser.error(f"--root MOUNT must be absolute, got {mount!r}")
        set_roots(dict(args.root))
    pj.serve(args.port)


if __name__ == "__main__":
    main()
