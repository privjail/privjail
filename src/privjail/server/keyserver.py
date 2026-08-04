"""keyserver.py -- holds the gocryptfs mount password and releases it only
to callers that prove, via TDX remote attestation, that they are running
inside a genuine TDX VM.

Reads the password from a file once at startup (a plaintext file is a
placeholder -- see the README for the plan to replace it with something
better). Runs indefinitely, serving this protocol on a TCP socket:
  server -> client : 64 raw bytes            (nonce challenge)
  client -> server : length-prefixed frame   (TDX quote bytes)
  server -> client : 1 status byte (1=ok, 0=fail)
                      + length-prefixed frame (secret, or an error message)

This currently runs on the same host as the gateway (see gateway.py); it is
meant to move to a separate host, hence talking over a TCP socket rather
than a pipe or shared file.
"""
import argparse
import importlib.resources
import os
import socket
import sys

from privjail.server.attestation.net import recv_frame, send_frame
from privjail.server.attestation.quote import REPORTDATA_LEN
from privjail.server.attestation.verify import TdxVerifyError, verify_quote

DEFAULT_PORT = 9443
MAX_QUOTE_LEN = 64 * 1024


def _default_root_pem() -> str:
    return str(importlib.resources.files("privjail.server.attestation").joinpath("trusted_root.pem"))


def handle_client(conn: socket.socket, addr, root_pem_path: str, secret: bytes) -> None:
    nonce = os.urandom(REPORTDATA_LEN)
    print(f"[keyserver] connection from {addr[0]}:{addr[1]}, issuing nonce challenge")
    conn.sendall(nonce)  # nonce is sent raw, not framed

    try:
        quote = recv_frame(conn, MAX_QUOTE_LEN)
    except (ConnectionError, ValueError) as e:
        print(f"[keyserver] failed to receive quote: {e}", file=sys.stderr)
        return
    print(f"[keyserver] received quote ({len(quote)} bytes), verifying...")

    try:
        verify_quote(quote, root_pem_path, nonce)
        ok = True
    except TdxVerifyError as e:
        print(f"[keyserver] attestation failed: {e}", file=sys.stderr)
        ok = False

    conn.sendall(bytes([1 if ok else 0]))
    if ok:
        print("[keyserver] attestation OK -- releasing key")
        send_frame(conn, secret)
    else:
        print("[keyserver] attestation FAILED -- refusing to release key")
        send_frame(conn, b"attestation failed")


def run(port: int, host: str, root_pem_path: str, secret: bytes) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as lsock:
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind((host, port))
        lsock.listen(8)
        print(f"[keyserver] listening on {host}:{port} (root={root_pem_path})")
        while True:
            conn, addr = lsock.accept()
            with conn:
                handle_client(conn, addr, root_pem_path, secret)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1",
                        help="IP address to listen to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="port number to listen to")
    parser.add_argument("--root-pem", default=None,
                        help="trusted root CA cert (default: bundled Intel SGX root)")
    parser.add_argument("--pw-file", default="pw",
                        help="file holding the secret to serve (default: 'pw' in the current directory)")
    args = parser.parse_args()

    root_pem_path = args.root_pem or _default_root_pem()
    with open(args.pw_file, "rb") as f:
        secret = f.read()

    run(args.port, args.host, root_pem_path, secret)


if __name__ == "__main__":
    main()
