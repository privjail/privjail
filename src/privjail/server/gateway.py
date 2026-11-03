"""gateway.py -- runs on the gateway host as the forced command of a
privjail user's ssh key (see privjail-admin add-user). The client
(privjail.gateway_session) runs `ssh <gateway> NAME=PATH ...`; each word asks
for the encrypted directory at PATH (absolute, or ~/... in this user's home)
to be visible as NAME (a bare PATH means NAME is its last component). For
this session it:

  1. opens an ssh tunnel to the key server (authenticated with the user's
     forwarded agent), whose local end is a Unix socket only this user can
     use;
  2. for each requested PATH, proves via TDX remote attestation that it runs
     inside a genuine TDX VM and receives PATH's gocryptfs password -- the
     key server releases it only if this user is granted PATH;
  3. mounts PATH read-only at a private, per-session mount point
     (--mnt-root/<pid>/<n>, never shown to the client);
  4. runs the privjail dataserver with the NAME -> mount point table, so the
     client opens "NAME/file" and nothing outside the table;
  5. when the dataserver exits, or the ssh session ends (EOF on stdin), stops
     the dataserver and unmounts everything it mounted.

Fetching a TDX quote needs root (creating the configfs-tsm entry); this
process elevates only for that one step via `sudo <bin>/privjail-get-quote`,
where <bin> is the directory of this interpreter (the venv's bin/), which
privjail-admin add-user allows in sudoers.
"""
import argparse
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time

from privjail.dirspec import NAME_RE, default_name, expand, path_error
from privjail.server.attestation.net import read_full, recv_frame, send_frame
from privjail.server.attestation.quote import REPORTDATA_LEN

MAX_RESPONSE_LEN = 1 << 20


def log(msg: str) -> None:
    print(f"[gateway] {msg}", flush=True)


def parse_requests(words: list[str], home: str) -> dict[str, str]:
    """["mydata=~/enc", "/srv/x"] -> {"mydata": "<home>/enc", "x": "/srv/x"}"""
    requests: dict[str, str] = {}
    for word in words:
        name, sep, path = word.partition("=")
        if not sep:
            name, path = default_name(word), word
        err = path_error(path)
        if err:
            raise SystemExit(f"[gateway] invalid directory {path!r}: {err}")
        if not NAME_RE.match(name):
            raise SystemExit(f"[gateway] invalid name {name!r} in {word!r}")
        if name in requests:
            raise SystemExit(f"[gateway] name {name!r} used twice")
        requests[name] = expand(path, home)
    return requests


def get_quote_privileged(nonce: bytes) -> bytes:
    proc = subprocess.run(
        ["sudo", "-n", os.path.join(os.path.dirname(sys.executable), "privjail-get-quote")],
        input=nonce, capture_output=True, check=True,
    )
    return proc.stdout


class KeyServerTunnel:
    """`ssh -N -L <private unix socket>:localhost:<port> <host>`."""

    def __init__(self, host: str, port: int, sock_path: str, timeout: float = 15.0):
        self.sock_path = sock_path
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        log(f"opening ssh tunnel to {host}:{port}")
        self.proc = subprocess.Popen(
            ["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "BatchMode=yes",
             "-L", f"{sock_path}:localhost:{port}", host],
            stdin=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + timeout
        while not os.path.exists(sock_path):
            if self.proc.poll() is not None:
                raise RuntimeError(f"ssh tunnel to {host} exited early (code {self.proc.returncode})")
            if time.monotonic() > deadline:
                self.close()
                raise TimeoutError(f"timed out waiting for ssh tunnel to {host}:{port}")
            time.sleep(0.1)

    def request_key(self, dirpath: str) -> bytes:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(self.sock_path)
            send_frame(sock, dirpath.encode())
            if read_full(sock, 1)[0] != 1:
                raise PermissionError(recv_frame(sock, MAX_RESPONSE_LEN).decode(errors="replace"))
            nonce = read_full(sock, REPORTDATA_LEN)
            send_frame(sock, get_quote_privileged(nonce))
            status = read_full(sock, 1)[0]
            response = recv_frame(sock, MAX_RESPONSE_LEN)
        if status != 1:
            raise PermissionError(f"key server refused: {response.decode(errors='replace')}")
        return response

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait()
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)


def unmount(path: str) -> None:
    if os.path.ismount(path):
        subprocess.run(["fusermount", "-u", path], check=False)
        if os.path.ismount(path):
            subprocess.run(["fusermount", "-uz", path], check=False)
    try:
        os.rmdir(path)
    except OSError:
        pass


def cleanup_session_dir(session_dir: str) -> None:
    if not os.path.isdir(session_dir):
        return
    for entry in os.listdir(session_dir):
        unmount(os.path.join(session_dir, entry))
    try:
        os.rmdir(session_dir)
    except OSError:
        pass


def cleanup_stale_sessions(mnt_root: str) -> None:
    """Unmounts what earlier sessions left behind (e.g. after a crash)."""
    for entry in os.listdir(mnt_root):
        if not entry.isdigit():
            continue
        try:
            os.kill(int(entry), 0)
            continue  # still running
        except ProcessLookupError:
            pass
        except PermissionError:
            continue
        cleanup_session_dir(os.path.join(mnt_root, entry))


def parse_key_server(spec: str) -> tuple[str, int]:
    host, sep, port_str = spec.rpartition(":")
    if not sep:
        return spec, 9443
    return host, int(port_str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="*", metavar="NAME=DIR",
                        help="for local testing; over ssh, the words come from the client's command")
    parser.add_argument("--key-server", default="privjail-ks:9443",
                        help="key server as host[:port]; reached via an ssh tunnel (default: privjail-ks:9443)")
    parser.add_argument("--mnt-root", default="~/.privjail/mnt",
                        help="where per-session mount points are made (default: ~/.privjail/mnt)")
    parser.add_argument("--port", type=int, default=0, help="dataserver port (default: any free port)")
    args = parser.parse_args()

    words = shlex.split(os.environ.get("SSH_ORIGINAL_COMMAND", "")) or args.dirs
    requests = parse_requests(words, os.path.expanduser("~"))
    if not requests:
        raise SystemExit("[gateway] no directory requested (e.g. `ssh <gateway> mydata=~/data_enc`)")

    mnt_root = os.path.expanduser(args.mnt_root)
    os.makedirs(mnt_root, mode=0o700, exist_ok=True)
    os.chmod(mnt_root, 0o700)
    cleanup_stale_sessions(mnt_root)
    session_dir = os.path.join(mnt_root, str(os.getpid()))
    os.makedirs(session_dir, mode=0o700)

    roots: dict[str, str] = {}
    try:
        tunnel = KeyServerTunnel(*parse_key_server(args.key_server),
                                 sock_path=os.path.join(mnt_root, f".ks-{os.getpid()}.sock"))
        try:
            mounts: dict[str, str] = {}
            for i, dirpath in enumerate(sorted(set(requests.values()))):
                mount_point = os.path.join(session_dir, str(i))
                password = tunnel.request_key(dirpath)
                os.makedirs(mount_point, mode=0o700)
                subprocess.run(["gocryptfs", "-ro", "-q", dirpath, mount_point], input=password, check=True)
                mounts[dirpath] = mount_point
                log(f"mounted {dirpath}")
        finally:
            tunnel.close()
        for name, dirpath in requests.items():
            roots[name] = mounts[dirpath]

        log("launching the data server for " + ", ".join(f"{name}={p}" for name, p in requests.items()))
        server = subprocess.Popen(
            [sys.executable, "-m", "privjail.server.dataserver", str(args.port),
             *[f"--root={name}={path}" for name, path in roots.items()]],
            stdin=subprocess.DEVNULL,
        )

        def stop(*_: object) -> None:
            if server.poll() is None:
                server.terminate()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGHUP, stop)

        def watch_stdin() -> None:
            # the ssh session going away closes our stdin
            try:
                while sys.stdin.buffer.read(4096):
                    pass
            except OSError:
                pass
            stop()

        threading.Thread(target=watch_stdin, daemon=True).start()
        server.wait()
    except (PermissionError, RuntimeError, TimeoutError, subprocess.CalledProcessError) as e:
        log(f"error: {e}")
        sys.exit(1)
    finally:
        cleanup_session_dir(session_dir)
        log("unmounted and exiting")


if __name__ == "__main__":
    main()
