"""keyserver.py -- holds one gocryptfs password per encrypted directory and
releases it only to a granted user whose request comes from a genuine TDX VM.

Directories are identified by their absolute path on the gateway.

State (in --state-dir, default ~/privjail-ks; owner-only), both re-read on
every request, so edits apply immediately; '#' starts a comment:
  passwords    "<path> <password>" per line: the gocryptfs password of the
               encrypted directory at <path>
  grants       "<user> <path>" per line: <user> may mount <path>
               (maintained by privjail-admin grant/revoke)

Who is asking: clients reach this server only through an ssh tunnel into
this host (`ssh -L ...:localhost:<port> <key server>`, restricted by
authorized_keys to exactly this port). The forwarded TCP connection is
opened by that user's own sshd process, so the peer socket's owner uid --
looked up in /proc/net/tcp -- identifies the ssh account, i.e. the user.

Protocol (frames are 4-byte little-endian length + payload):
  client -> server : frame(absolute path of the encrypted directory)
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
import socket
import sys

from privjail.dirspec import path_error
from privjail.server.attestation.net import recv_frame, send_frame
from privjail.server.attestation.quote import REPORTDATA_LEN
from privjail.server.attestation.verify import TdxVerifyError, verify_quote

DEFAULT_PORT = 9443
DEFAULT_STATE_DIR = "~/privjail-ks"
MAX_QUOTE_LEN = 64 * 1024
MAX_PATH_LEN = 4096


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
            # the peer's socket: its local end is our remote end and vice versa.
            # Only a live (ESTABLISHED, still owned by a process) socket counts:
            # once its owner closes it, the kernel reports it as uid 0.
            if fields[1] == remote and fields[2] == local:
                return int(fields[7]) if fields[3] == "01" and fields[9] != "0" else None
    return None


def read_pairs(path: str) -> list[tuple[str, str]]:
    """Lines of two whitespace-separated fields; '#' comments, others ignored."""
    pairs = []
    try:
        with open(path) as f:
            for line in f:
                fields = line.split("#", 1)[0].split()
                if len(fields) == 2:
                    pairs.append((fields[0], fields[1]))
    except FileNotFoundError:
        pass
    return pairs


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
        dirpath = recv_frame(conn, MAX_PATH_LEN).decode()
    except (ConnectionError, ValueError, UnicodeDecodeError) as e:
        print(f"[keyserver] {who}: bad request: {e}", file=sys.stderr)
        return
    who = f"{who} -> {dirpath}"

    if user is None:
        return refuse(conn, who, "cannot identify the requesting user")
    if not dirpath.startswith("/") or path_error(dirpath):
        return refuse(conn, who, f"invalid directory path {dirpath!r}")
    if (user, dirpath) not in read_pairs(os.path.join(state_dir, "grants")):
        return refuse(conn, who, f"{user} is not granted access to {dirpath}")
    passwords = dict(read_pairs(os.path.join(state_dir, "passwords")))
    if dirpath not in passwords:
        return refuse(conn, who, f"no password registered for {dirpath}")

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

    secret = (passwords[dirpath] + "\n").encode()
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
                        help=f"directory holding passwords and grants (default: {DEFAULT_STATE_DIR})")
    args = parser.parse_args()

    run(args.port, args.host, args.root_pem or _default_root_pem(), os.path.expanduser(args.state_dir))


if __name__ == "__main__":
    main()
