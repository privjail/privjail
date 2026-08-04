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

"""session.py -- unified session/lifecycle interface for privjail's
deployment modes:

  local_session()              -- everything runs in-process; no server.
  connect_session(host, port)  -- connect to a server someone else launched
                                   and manages; never touches its lifecycle.
  spawn_session(port=0)        -- launch a dataserver as a local subprocess,
                                   own its whole lifecycle.
  gateway_session(host, ...)   -- launch the gateway over ssh (attestation,
                                   mount, dataserver), own its whole
                                   lifecycle including both ssh processes.

All four return a Session, usable as a context manager (or via explicit
.close()) with the same shape regardless of mode -- what close() actually
does depends on how the session was created. The rule that makes this
coherent: whichever constructor launched a server is the one whose close()
retires it; connect_session never touches a server it didn't create.
"""
from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

from egrpc import connect as _egrpc_connect, disconnect as _egrpc_disconnect

from .helper import shutdown_remote_server

_SERVER_STARTED_RE = re.compile(r"Server started on port (\d+) \(pid = (\d+)\)")


class LaunchError(Exception):
    pass


class Session:
    """Returned by local_session()/connect_session()/spawn_session()/
    gateway_session(). Use as a context manager, or call .close()
    explicitly; safe to call close() more than once."""

    def __init__(self, port: Optional[int], close_fn: Callable[[], None]):
        self.port = port
        self._close_fn = close_fn
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_fn()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def local_session() -> Session:
    """Mode 1: no server; everything runs in-process. close() is a no-op."""
    return Session(port=None, close_fn=lambda: None)


def connect_session(host: str, port: int) -> Session:
    """Mode 2: connect to a server someone else launched and manages.
    close() only disconnects -- it never shuts the server down, since this
    session doesn't own it."""
    _egrpc_connect(host, port)
    return Session(port=port, close_fn=_egrpc_disconnect)


def _wait_for_banner(proc: subprocess.Popen[str], label: str, timeout: float) -> int:
    """Waits for `proc`'s stdout to print "Server started on port <port>
    (pid = <pid>)." and returns the port. Keeps draining stdout for the
    life of the process (even after the port is found) so the child never
    blocks on a full pipe. Raises LaunchError on early exit or timeout."""
    port_found = threading.Event()
    result: dict[str, int] = {}
    output_lines: list[str] = []

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            output_lines.append(line)
            if "port" not in result:
                m = _SERVER_STARTED_RE.search(line)
                if m:
                    result["port"] = int(m.group(1))
                    port_found.set()

    threading.Thread(target=reader, daemon=True).start()

    deadline = time.monotonic() + timeout
    while not port_found.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            proc.terminate()
            raise LaunchError(
                f"{label} did not report a server port within {timeout}s. "
                f"Output so far:\n{''.join(output_lines)}"
            )
        if proc.poll() is not None:
            raise LaunchError(
                f"{label} exited (code {proc.returncode}) before starting "
                f"the server. Output:\n{''.join(output_lines)}"
            )
        port_found.wait(timeout=min(0.2, remaining))

    return result["port"]


def _shutdown_and_disconnect() -> None:
    try:
        shutdown_remote_server()
    finally:
        _egrpc_disconnect()


def spawn_session(port: int = 0, timeout: float = 10.0) -> Session:
    """Mode 3a: launch a dataserver as a local subprocess and connect to
    it. close() asks it to shut down over RPC and disconnects, falling
    back to killing the subprocess if it doesn't exit promptly."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "privjail.server.dataserver", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assigned_port = _wait_for_banner(proc, "local dataserver", timeout)
    _egrpc_connect("localhost", assigned_port)

    def _close() -> None:
        try:
            _shutdown_and_disconnect()
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    return Session(port=assigned_port, close_fn=_close)


def gateway_session(host: str, local_port: int, timeout: float = 30.0) -> Session:
    """Mode 3b: launch the gateway over ssh (attestation, mount,
    dataserver), tunnel a local port to it, and connect. close() asks the
    remote server to shut down, disconnects, and tears down both ssh
    processes."""
    gateway_proc = subprocess.Popen(
        ["ssh", host],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    remote_port = _wait_for_banner(gateway_proc, f"`ssh {host}`", timeout)

    tunnel_proc = subprocess.Popen(["ssh", "-L", f"{local_port}:localhost:{remote_port}", host, "-N"])

    _egrpc_connect("localhost", local_port)

    def _close() -> None:
        try:
            _shutdown_and_disconnect()
        finally:
            tunnel_proc.terminate()
            if gateway_proc.poll() is None:
                gateway_proc.terminate()

    return Session(port=local_port, close_fn=_close)
