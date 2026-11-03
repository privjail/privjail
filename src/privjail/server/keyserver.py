"""keyserver.py -- holds one gocryptfs password per encrypted directory and
releases it only to a granted user whose request comes from a genuine TDX VM.

State (in --state-dir, default ~/privjail-ks; owner-only):
  keys/<dir>   the gocryptfs password of encrypted directory <dir> (one line)
  grants       which user may mount which directory, one "<user> <dir>" per
               line ('#' starts a comment); re-read on every request, so
               edits (e.g. by privjail-admin grant/revoke) apply immediately

Who is asking: clients reach this server only through an ssh tunnel into
this host (`ssh -L ...:localhost:<port> <key server>`, restricted by
authorized_keys to exactly this port). The forwarded TCP connection is
opened by that user's own sshd process, so the peer socket's owner uid --
looked up in /proc/net/tcp -- identifies the ssh account, i.e. the user.

Protocol (frames are 4-byte little-endian length + payload):
  client -> server : frame(directory name)
  server -> client : 1 byte; 0 = refused, followed by frame(error message)
                             1 = go on,   followed by 64 raw bytes (nonce)
  client -> server : frame(TDX quote over the nonce)
  server -> client : 1 status byte (1=ok, 0=fail)
                      + frame(password, or an error message)
"""
import argparse
import importlib.resources
import io
import os
import pwd
import re
import socket
import sys

from privjail.server.attestation.net import recv_frame, send_frame
from privjail.server.attestation.quote import REPORTDATA_LEN
from privjail.server.attestation.verify import TdxVerifyError, verify_quote

DEFAULT_PORT = 9443
DEFAULT_STATE_DIR = "~/privjail-ks"
MAX_QUOTE_LEN = 64 * 1024
MAX_NAME_LEN = 255
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _default_root_pem() -> str:
    return str(importlib.resources.files("privjail.server.attestation").joinpath("trusted_root.pem"))


def _hex_endpoint(addr: str, port: int) -> str:
    """127.0.0.1:9443 -> "0100007F:24E3", as written in /proc/net/tcp."""
    packed = socket.inet_aton(addr)
    return f"{int.from_bytes(packed, 'little'):08X}:{port:04X}"


def peer_uid(conn: socket.socket) -> int | None:
    """uid owning the other end of a loopback TCP connection."""
    local = _hex_endpoint(*conn.getsockname())
    remote = _hex_endpoint(*conn.getpeername())
    with open("/proc/net/tcp") as f:
        next(f)
        for line in f:
            fields = line.split()
            # the peer's socket: its local end is our remote end and vice versa
            if fields[1] == remote and fields[2] == local:
                return int(fields[7])
    return None


def read_grants(path: str) -> set[tuple[str, str]]:
    grants: set[tuple[str, str]] = set()
    try:
        with open(path) as f:
            for line in f:
                fields = line.split("#", 1)[0].split()
                if len(fields) == 2:
                    grants.add((fields[0], fields[1]))
    except FileNotFoundError:
        pass
    return grants


def refuse(conn: socket.socket, who: str, message: str) -> None:
    print(f"[keyserver] {who}: refused: {message}", file=sys.stderr)
    conn.sendall(b"\x00")
    send_frame(conn, message.encode())


def handle_client(conn: socket.socket, root_pem_path: str, state_dir: str) -> None:
    uid = peer_uid(conn)
    try:
        user = pwd.getpwuid(uid).pw_name if uid is not None else None
    except KeyError:
        user = None
    who = user or f"uid {uid}"

    try:
        name = recv_frame(conn, MAX_NAME_LEN).decode()
    except (ConnectionError, ValueError, UnicodeDecodeError) as e:
        print(f"[keyserver] {who}: bad request: {e}", file=sys.stderr)
        return
    who = f"{who} -> {name}"

    if user is None:
        return refuse(conn, who, "cannot identify the requesting user")
    if not NAME_RE.match(name):
        return refuse(conn, who, f"invalid directory name {name!r}")
    if (user, name) not in read_grants(os.path.join(state_dir, "grants")):
        return refuse(conn, who, f"{user} is not granted access to {name}")
    key_path = os.path.join(state_dir, "keys", name)
    if not os.path.isfile(key_path):
        return refuse(conn, who, f"no key for {name}")

    nonce = os.urandom(REPORTDATA_LEN)
    print(f"[keyserver] {who}: granted, issuing nonce challenge")
    conn.sendall(b"\x01" + nonce)

    try:
        quote = recv_frame(conn, MAX_QUOTE_LEN)
    except (ConnectionError, ValueError) as e:
        print(f"[keyserver] {who}: failed to receive quote: {e}", file=sys.stderr)
        return

    try:
        verify_quote(quote, root_pem_path, nonce)
    except TdxVerifyError as e:
        print(f"[keyserver] {who}: attestation FAILED: {e}", file=sys.stderr)
        conn.sendall(b"\x00")
        send_frame(conn, b"attestation failed")
        return

    with open(key_path, "rb") as f:
        secret = f.read()
    print(f"[keyserver] {who}: attestation OK -- releasing key")
    conn.sendall(b"\x01")
    send_frame(conn, secret)


def run(port: int, host: str, root_pem_path: str, state_dir: str) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as lsock:
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind((host, port))
        lsock.listen(8)
        print(f"[keyserver] listening on {host}:{port} (state={state_dir}, root={root_pem_path})")
        while True:
            conn, _ = lsock.accept()
            with conn:
                try:
                    handle_client(conn, root_pem_path, state_dir)
                except OSError as e:
                    print(f"[keyserver] connection error: {e}", file=sys.stderr)


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1",
                        help="IPv4 loopback address to listen on (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to listen on (default: {DEFAULT_PORT})")
    parser.add_argument("--root-pem", default=None, help="trusted root CA cert (default: bundled Intel SGX root)")
    parser.add_argument("--state-dir", default=DEFAULT_STATE_DIR,
                        help=f"directory holding keys/ and grants (default: {DEFAULT_STATE_DIR})")
    args = parser.parse_args()

    run(args.port, args.host, args.root_pem or _default_root_pem(), os.path.expanduser(args.state_dir))


if __name__ == "__main__":
    main()
