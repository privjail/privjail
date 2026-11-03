"""dataserver.py -- brings up the privjail server-side process, listening
for privjail clients on the given port.

  python -m privjail.server.dataserver [PORT] [--root NAME=PATH ...]

With --root (as launched by gateway.py), clients can open files only as
"NAME/..." under one of the given directories (see privjail.datapath);
without it, paths are used as given.
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
    parser.add_argument("--root", action="append", default=None, metavar="NAME=PATH",
                        help="expose PATH to clients as NAME (repeatable)")
    args = parser.parse_args()
    if args.root is not None:
        roots = {}
        for spec in args.root:
            name, sep, path = spec.partition("=")
            if not sep or not name or "/" in name:
                parser.error(f"--root must be NAME=PATH, got {spec!r}")
            roots[name] = path
        set_roots(roots)
    pj.serve(args.port)


if __name__ == "__main__":
    main()
