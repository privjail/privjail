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

"""launch.py -- client-side helper to bring up a privjail gateway over ssh
and tunnel to it.

`ssh host` runs the gateway's forced command (see privjail.server.gateway),
which mounts the encrypted directory, attests, and finally execs into
privjail.server.dataserver, which binds a (by default OS-assigned) port and
prints "Server started on port <port> (pid = <pid>)." launch_server()
parses that port out of the ssh session's output and opens a second ssh
connection to forward a local port to it.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time

_SERVER_STARTED_RE = re.compile(r"Server started on port (\d+) \(pid = (\d+)\)")

# Keep references to launched ssh processes alive for the life of the
# interpreter. The first ssh connection IS the remote gateway/dataserver
# process (via the forced command); closing it would end that process, so
# it -- and the tunnel -- must keep running in the background rather than
# be dropped once launch_server() returns.
_background_processes: list[subprocess.Popen] = []


class LaunchError(Exception):
    pass


def launch_server(host: str, local_port: int, timeout: float = 30.0) -> int:
    """Launches the privjail gateway on `host` over ssh and forwards
    `local_port` to whatever port it ends up listening on.

    Runs `ssh host` (triggering the gateway's forced command), waits up to
    `timeout` seconds for it to print "Server started on port <port> (pid =
    <pid>).", then runs `ssh -L <local_port>:localhost:<port> host -N` to
    establish the tunnel.

    Both ssh processes are left running in the background (see module-level
    note) so the gateway and tunnel stay up. Raises LaunchError if the
    gateway's ssh session exits before printing the expected line, or
    doesn't print it within `timeout` seconds.

    Returns `local_port`.
    """
    gateway_proc = subprocess.Popen(
        ["ssh", host],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    remote_port: int | None = None
    output_lines: list[str] = []
    port_found = threading.Event()

    def reader() -> None:
        nonlocal remote_port
        assert gateway_proc.stdout is not None
        for line in gateway_proc.stdout:
            output_lines.append(line)
            if remote_port is None:
                m = _SERVER_STARTED_RE.search(line)
                if m:
                    remote_port = int(m.group(1))
                    port_found.set()
        # Keep draining stdout even after the port is found, for the life of
        # the process, so the remote side never blocks on a full pipe.

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    deadline = time.monotonic() + timeout
    while not port_found.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            gateway_proc.terminate()
            raise LaunchError(
                f"`ssh {host}` did not report a server port within {timeout}s. "
                f"Output so far:\n{''.join(output_lines)}"
            )
        if gateway_proc.poll() is not None:
            raise LaunchError(
                f"`ssh {host}` exited (code {gateway_proc.returncode}) before "
                f"starting the server. Output:\n{''.join(output_lines)}"
            )
        port_found.wait(timeout=min(0.2, remaining))

    tunnel_proc = subprocess.Popen(["ssh", "-L", f"{local_port}:localhost:{remote_port}", host, "-N"])

    _background_processes.append(gateway_proc)
    _background_processes.append(tunnel_proc)

    return local_port
