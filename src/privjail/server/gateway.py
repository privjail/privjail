"""gateway.py -- runs on the gateway host as the forced command behind
`ssh -L 12345:localhost:12345 you@gateway` (see README). It:

  1. retrieves the key: acts as the attester, proving to the key server (via
     TDX remote attestation) that it is running inside the intended TDX VM,
     and receives the gocryptfs mount password in return.
  2. mounts the encrypted directory with that password.
  3. brings up the privjail server-side process (dataserver.py) listening
     on the tunneled port, or -- if one from a previous session is already
     listening -- just keeps this ssh session alive so its -L tunnel stays
     valid.

Fetching a TDX quote needs root (creating the configfs-tsm entry); this
process itself doesn't, so it elevates only for that one step by shelling
out to `sudo ... -m privjail.server.attestation.quote`.
"""
import argparse
import os
import socket
import subprocess
import sys
import time

from privjail.server.attestation.net import read_full, recv_frame, send_frame
from privjail.server.attestation.quote import REPORTDATA_LEN

MAX_RESPONSE_LEN = 1 << 20

def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        ## ???? doesn't this confuse the server?
        return s.connect_ex(("localhost", port)) == 0

def get_quote_privileged(nonce: bytes) -> bytes:
    proc = subprocess.run(
        ["sudo", sys.executable, "-m", "privjail.server.attestation.quote"],
        input=nonce,
        capture_output=True,
        check=True,
    )
    return proc.stdout


def request_key(key_server_host: str, key_server_port: int) -> bytes:
    with socket.create_connection((key_server_host, key_server_port)) as sock:
        nonce = read_full(sock, REPORTDATA_LEN)
        quote = get_quote_privileged(nonce)
        send_frame(sock, quote)
        status = read_full(sock, 1)[0]
        response = recv_frame(sock, MAX_RESPONSE_LEN)
    if status != 1:
        raise RuntimeError(f"key server refused to release key: {response.decode(errors='replace')}")
    return response


def parse_key_server(spec: str) -> tuple[str, int]:
    host, sep, port_str = spec.rpartition(":")
    if not sep:
        raise ValueError(f"--key-server must be HOST:PORT, got {spec!r}")
    return host, int(port_str)


def open_key_server_tunnel(host: str, port: int, timeout: float = 10.0) -> subprocess.Popen | None:
    """Ensures the key server is reachable at localhost:port, by opening
    `ssh -L port:localhost:port host -N` if nothing is listening on that
    port locally yet.

    Returns the tunnel's Popen if a new one was started (the caller is then
    responsible for terminating it once done with the key server), or None
    if localhost:port was already reachable (assumed to be a tunnel from a
    previous session; left alone since we didn't start it).
    """
    if port_in_use(port):
        print(f"[gateway] localhost:{port} is already reachable; assuming a "
              f"tunnel to {host} is already up", flush=True)
        return None

    print(f"[gateway] opening ssh tunnel to {host}:{port}", flush=True)
    proc = subprocess.Popen(["ssh", "-L", f"{port}:localhost:{port}", host, "-N"])

    deadline = time.monotonic() + timeout
    while not port_in_use(port):
        if proc.poll() is not None:
            raise RuntimeError(f"ssh tunnel to {host} exited early (code {proc.returncode})")
        if time.monotonic() > deadline:
            proc.terminate()
            raise TimeoutError(f"timed out waiting for ssh tunnel to {host}:{port}")
        time.sleep(0.2)
    return proc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-enc", default="data_enc", help="gocryptfs-encrypted source directory")
    parser.add_argument("--data", default="data", help="mount point for the decrypted view")
    parser.add_argument("--key-server", default="ks:9443", help="key server as host:port; reached via an ssh tunnel")
    # TODO: get rid of --port option and always use dynamic port instead (port=0)
    parser.add_argument("--port", type=int, default=0, help="privjail server port")
    args = parser.parse_args()

    if os.path.ismount(args.data):
        print(f"[gateway] {args.data} is already mounted, skipping mount", flush=True)
    else:
        key_server_host, key_server_port = parse_key_server(args.key_server)
        print(f"[gateway] {args.data} is not mounted; requesting key from "
              f"{key_server_host}:{key_server_port}", flush=True)
        tunnel_proc = open_key_server_tunnel(key_server_host, key_server_port)
        try:
            password = request_key("localhost", key_server_port)
            subprocess.run(
                ["gocryptfs", args.data_enc, args.data],
                input=password,
                check=True,
            )
        finally:
            if tunnel_proc is not None:
                tunnel_proc.terminate()
        print(f"[gateway] mounted {args.data_enc} at {args.data}", flush=True)

    if args.port and port_in_use(args.port):
        # A server from a previous session is already listening; just keep
        # this session alive so its -L tunnel stays valid.
        # TODO: get rid of this. this is not going to work anyway
        print(f"[gateway] port {args.port} is already in use; assuming a data "
              f"server from a previous session is serving it, keeping this "
              f"session alive for the tunnel", flush=True)
        while True:
            time.sleep(3600)
    else:
        # if port == 0, we do not know what the port actually is
        # no point in showing "... on port 0"
        # we should print it in the dataserver
        print(f"[gateway] launching a fresh data server on port {args.port}", flush=True)
        os.execv(sys.executable, [sys.executable, "-m", "privjail.server.dataserver", str(args.port)])


if __name__ == "__main__":
    main()
