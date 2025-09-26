# Copyright 2025 TOYOTA MOTOR CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from typing import Sequence
import argparse

from . import serve as start_server

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m privjail",
        description="Utilities for interacting with privjail from the command line.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start a privjail gRPC server using privjail.serve().",
    )
    serve_parser.add_argument(
        "host",
        nargs="?",
        default=None,
        help="Host/IP address to bind. Defaults to all interfaces.",
    )
    serve_parser.add_argument(
        "port",
        type=int,
        help="Port number to listen on.",
    )
    serve_parser.set_defaults(func=_handle_serve)

    return parser

def _handle_serve(args: argparse.Namespace) -> None:
    try:
        start_server(args.port, host=args.host)
    except KeyboardInterrupt:
        pass

def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "func")
    handler(args)

if __name__ == "__main__":
    main()
