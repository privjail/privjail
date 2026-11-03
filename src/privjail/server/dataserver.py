"""dataserver.py -- brings up the privjail server-side process, listening
for privjail clients on the given port.

  python -m privjail.server.dataserver [PORT] [--root DIR REAL CANONICAL ...]

With --root (as launched by spawn_session / gateway.py), clients may open
files only under the given DIRs (the directories they mounted, as they name
them); DIR is actually found at REAL, and named CANONICAL for budget
accounting (see privjail.datapath). Without it, paths are used as given.
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
    parser.add_argument("--root", action="append", nargs=3, default=None, metavar=("DIR", "REAL", "CANONICAL"),
                        help="let clients open files under DIR, found at REAL, named CANONICAL (repeatable)")
    args = parser.parse_args()
    if args.root is not None:
        set_roots({d: (real, canon) for d, real, canon in args.root})
    pj.serve(args.port)


if __name__ == "__main__":
    main()
