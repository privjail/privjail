"""privjail-admin -- manage who may use a PrivJail gateway, and on which data.

Run by the data owner, from a machine that can `ssh` into both the gateway
and the key server as an account with sudo rights there.

Two kinds of permission
-----------------------
add-user / remove-user   whether an OS account (existing on both hosts) can
                         use the gateway at all: its ssh key, and what the
                         gateway needs to run for it.
grant / revoke           which encrypted directories that user may mount.
                         Recorded on the key server, which holds the
                         directories' passwords and hands one out only for
                         a granted (user, directory) pair.

An encrypted directory is a gocryptfs directory on the gateway, identified
by its path there: absolute, or ~/... for the granted user's home (expanded
when granting; the key server only records absolute paths). Its password is
a "<path> <password>" line in <ks-dir>/passwords on the key server (default
~/privjail-ks), written by add-dir (or by hand); this tool never copies that
file's contents off the key server. Filling a directory with data is the
owner's business, outside this tool. The user must be able to read its
gocryptfs.conf.

Subcommands
-----------
status       users, their keys, the gateway setup, and their grants, with
             what is missing. Without arguments, the users are those this
             tool knows about: a privjail-admin sudoers entry on the gateway
             (made by add-user) or a line in the key server's grants.
add-user     registers the user's key (--user-pubkey) on both hosts --
             gateway: forced command privjail-gateway, agent forwarding
             allowed; key server: nothing but forwarding to the key server
             port -- plus, on the gateway, a sudoers entry for
             privjail-get-quote and a ~/.ssh/config block for the key server.
remove-user  removes those (all, or the key matching --match), and the
             user's grants once no key is left.
grant        USER PATH: lets USER mount the encrypted directory at PATH.
revoke       USER PATH: takes it back (applies from the next session).
add-dir      DIR: creates DIR on the gateway, initializes it with gocryptfs
             under a new random password, and registers it on the key
             server. With --existing, DIR must already be a gocryptfs
             directory: its password (--dir-passfile, or asked) is checked
             by a read-only test mount and then registered. Never "use it if
             it exists, else create it" -- a mistyped path should fail.
remove-dir   DIR: unregisters DIR on the key server (refused while granted);
             the directory and its data are left alone.

Every subcommand inspects each host with one ssh session (both hosts in
parallel) and changes it with at most one more; --dry-run shows the changes
without making them.
"""
from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import os
import re
import secrets
import shlex
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, NoReturn

from privjail.dirspec import path_error

DEFAULT_GATEWAY = "privjail-gw"
DEFAULT_KEY_SERVER_HOST = "privjail-ks"
DEFAULT_KEY_SERVER_PORT = 9443
DEFAULT_KS_DIR = "~/privjail-ks"
DEFAULT_PRIVJAIL_GATEWAY_CMD = "/opt/privjail/.venv/bin/privjail-gateway"

SUDOERS_DIR = "/etc/sudoers.d"
SUDOERS_MARK = "# managed by privjail-admin"
QUOTE_COMMAND = "privjail-get-quote"
SSH_CONFIG_BEGIN = "# BEGIN privjail-admin (managed block; edits here are overwritten)"
SSH_CONFIG_END = "# END privjail-admin"


# --------------------------------------------------------------------------
# ssh plumbing

def run_remote_script(host: str, script: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    """Runs `script` with bash on `host` (fed on stdin, so no quoting issues)."""
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "LogLevel=ERROR",
         host, "bash", "-s"],
        input=script, capture_output=True, text=True, timeout=timeout, check=False,
    )


# Advisory banners that are never the actual reason a remote command failed
# (OpenSSH's post-quantum notice; ProxyJump hops' known_hosts warnings).
_NOISE_PREFIXES = ("**", "Warning: Permanently added")


def last_error_line(text: str) -> str:
    lines = [l for l in text.strip().splitlines()
             if l.strip() and not l.strip().startswith(_NOISE_PREFIXES)]
    return lines[-1] if lines else ""


def resolve_ssh_target(alias: str) -> tuple[str, int]:
    """Resolves `alias` through the *operator's* ssh config (`ssh -G`)."""
    result = subprocess.run(["ssh", "-G", alias], capture_output=True, text=True, timeout=10, check=False)
    hostname, port = alias, 22
    for line in result.stdout.splitlines():
        key, _, value = line.partition(" ")
        if key == "hostname":
            hostname = value
        elif key == "port":
            port = int(value)
    return hostname, port


# --------------------------------------------------------------------------
# probing: one ssh per host, emitting "key<TAB>value" lines

PROBE_SCRIPT = r"""
set -u
e() { printf '%s\t%s\n' "$1" "$2"; }
yn() { "$@" >/dev/null 2>&1 && echo 1 || echo 0; }
b64() { base64 -w0; }

e login 1
e root "$(yn sudo -n true)"
if [ "$ROLE" = ks ]; then
  KS_DIR=${KS_DIR/#\~/$HOME}
  e listening "$(yn timeout 3 bash -c "exec 3<>/dev/tcp/127.0.0.1/$KS_PORT")"
  # only the registered paths, never the passwords
  e dirs "$(sed 's/#.*//' "$KS_DIR/passwords" 2>/dev/null | awk 'NF == 2 {print $1}' | tr '\n' ' ')"
  e grants "$(cat "$KS_DIR/grants" 2>/dev/null | b64)"
else
  # users that add-user has set up: the sudoers entries it wrote (marked)
  e sudoers_users "$(sudo -n find "$SUDOERS_DIR" -maxdepth 1 -type f -name 'privjail-*' 2>/dev/null |
    while IFS= read -r f; do
      [ "$(sudo -n head -n1 "$f")" = "$SUDOERS_MARK" ] && sudo -n awk '/^[^#]/ {print $1; exit}' "$f"
    done | tr '\n' ' ')"
  e gocryptfs "$(command -v gocryptfs || true)"
  [ -n "$REACH_HOST" ] && e reach "$(yn timeout 5 bash -c "exec 3<>/dev/tcp/$REACH_HOST/$REACH_PORT")"
  for d in ${DIRS[@]+"${DIRS[@]}"}; do
    case "$d" in /*) e "dir:$d" "$(yn sudo -n test -d "$d")$(yn sudo -n test -e "$d/gocryptfs.conf")" ;; esac
  done
fi


for u in ${USERS[@]+"${USERS[@]}"}; do
  p="u:$u"
  if ! id -u "$u" >/dev/null 2>&1; then e "$p:exists" 0; continue; fi
  e "$p:exists" 1
  h=$(getent passwd "$u" | cut -d: -f6); e "$p:home" "$h"
  if ! sudo -n -u "$u" true 2>/dev/null; then e "$p:sudo" 0; continue; fi
  e "$p:sudo" 1
  ak=$(sudo -n -u "$u" cat "$h/.ssh/authorized_keys" 2>/dev/null || true)
  e "$p:ak" "$(printf '%s' "$ak" | b64)"
  [ "$ROLE" = gw ] || continue

  e "$p:config" "$(sudo -n -u "$u" cat "$h/.ssh/config" 2>/dev/null | b64)"
  e "$p:sudoers" "$(sudo -n cat "$SUDOERS_DIR/privjail-${u//./_}" 2>/dev/null | b64)"
  for d in ${DIRS[@]+"${DIRS[@]}"}; do
    x=$d; case "$d" in "~/"*) x="$h/${d#\~/}" ;; esac
    e "$p:dir:$d" "$x $(yn sudo -n -u "$u" test -d "$x")$(yn sudo -n -u "$u" test -e "$x/gocryptfs.conf")$(yn sudo -n -u "$u" test -r "$x/gocryptfs.conf")"
  done
  cmds=()
  [ -n "$CMD" ] && cmds+=("$CMD")
  while IFS= read -r l; do
    case "$l" in *privjail-gateway*) ;; *) continue ;; esac
    c=$(printf '%s' "$l" | sed -n 's/^command="\([^ "]*\).*/\1/p'); [ -n "$c" ] && cmds+=("$c")
  done <<< "$ak"
  for c in ${cmds[@]+"${cmds[@]}"}; do
    sb=$(sudo -n -u "$u" head -n1 "$c" 2>/dev/null || true)
    py=${sb#\#!}; py=${py%% *}; q=""
    case "$sb" in '#!/'*) case "$py" in */env) ;; *) q="$(dirname "$py")/$QUOTE_COMMAND" ;; esac ;; esac
    qx=0; [ -n "$q" ] && qx=$(yn sudo -n -u "$u" test -x "$q")
    e "$p:cmd:$c" "$(yn sudo -n -u "$u" test -x "$c") $qx $q"
  done
done
"""


@dataclass
class CmdState:
    executable: bool
    quote_executable: bool
    quote_path: str


@dataclass
class UserFacts:
    exists: bool = False
    sudo: bool = False
    home: str = ""
    authorized_keys: list[str] = field(default_factory=list)
    ssh_config: list[str] = field(default_factory=list)
    sudoers: str = ""
    cmds: dict[str, CmdState] = field(default_factory=dict)
    # dir spec as asked -> (expanded path, exists, initialized, gocryptfs.conf readable), as this user
    dirs: dict[str, tuple[str, bool, bool, bool]] = field(default_factory=dict)


@dataclass
class HostFacts:
    name: str
    login: bool = False
    error: str = ""
    root: bool = False
    # key server
    listening: bool = False
    registered: list[str] = field(default_factory=list)  # paths in <ks-dir>/passwords
    grants_lines: list[str] = field(default_factory=list)
    # gateway
    sudoers_users: list[str] = field(default_factory=list)
    gocryptfs: str = ""
    reach: bool | None = None
    dirs: dict[str, tuple[bool, bool]] = field(default_factory=dict)  # abs path -> (exists, initialized)
    users: dict[str, UserFacts] = field(default_factory=dict)

    @property
    def grants(self) -> set[tuple[str, str]]:
        return parse_grants(self.grants_lines)


def _b64text(value: str) -> str:
    return base64.b64decode(value).decode(errors="replace") if value else ""


def probe(host: str, role: str, users: Sequence[str], args: argparse.Namespace,
          cmd: str = "", reach: tuple[str, int] | None = None, dirs: Sequence[str] = ()) -> HostFacts:
    params = {
        "ROLE": role, "SUDOERS_MARK": SUDOERS_MARK,
        "KS_PORT": str(args.key_server_port), "KS_DIR": args.ks_dir,
        "CMD": cmd, "REACH_HOST": reach[0] if reach else "", "REACH_PORT": str(reach[1]) if reach else "",
        "SUDOERS_DIR": SUDOERS_DIR, "QUOTE_COMMAND": QUOTE_COMMAND,
    }
    header = "".join(f"{k}={shlex.quote(v)}\n" for k, v in params.items())
    header += "USERS=(" + " ".join(shlex.quote(u) for u in users) + ")\n"
    header += "DIRS=(" + " ".join(shlex.quote(d) for d in dirs) + ")\n"
    facts = HostFacts(host)
    try:
        result = run_remote_script(host, header + PROBE_SCRIPT)
    except subprocess.TimeoutExpired:
        facts.error = "timed out"
        return facts
    if "login\t1" not in result.stdout:
        facts.error = last_error_line(result.stderr) or "ssh failed"
        return facts

    for line in result.stdout.splitlines():
        key, _, value = line.partition("\t")
        if key == "login":
            facts.login = True
        elif key == "root":
            facts.root = value == "1"
        elif key == "listening":
            facts.listening = value == "1"
        elif key == "dirs":
            facts.registered = value.split()
        elif key == "grants":
            facts.grants_lines = _b64text(value).splitlines()
        elif key == "sudoers_users":
            facts.sudoers_users = value.split()
        elif key == "gocryptfs":
            facts.gocryptfs = value
        elif key == "reach":
            facts.reach = value == "1"
        elif key.startswith("dir:"):
            facts.dirs[key[4:]] = (value[0] == "1", value[1] == "1")
        elif key.startswith("u:"):
            _, user, attr = key.split(":", 2)
            uf = facts.users.setdefault(user, UserFacts())
            if attr == "exists":
                uf.exists = value == "1"
            elif attr == "sudo":
                uf.sudo = value == "1"
            elif attr == "home":
                uf.home = value
            elif attr == "ak":
                uf.authorized_keys = [l for l in _b64text(value).splitlines() if l.strip()]
            elif attr == "config":
                uf.ssh_config = _b64text(value).splitlines()
            elif attr == "sudoers":
                uf.sudoers = _b64text(value)
            elif attr.startswith("dir:"):
                expanded, flags = value.rsplit(" ", 1)
                uf.dirs[attr[4:]] = (expanded, flags[0] == "1", flags[1] == "1", flags[2] == "1")
            elif attr.startswith("cmd:"):
                x, qx, *q = value.split(" ", 2)
                uf.cmds[attr[4:]] = CmdState(x == "1", qx == "1", q[0] if q else "")
    return facts


def probe_both(args: argparse.Namespace, users: Sequence[str], cmd: str = "",
               reach: tuple[str, int] | None = None, dirs: Sequence[str] = ()) -> tuple[HostFacts, HostFacts]:
    with ThreadPoolExecutor(2) as pool:
        ks = pool.submit(probe, args.key_server_ssh, "ks", users, args)
        gw = pool.submit(probe, args.gateway_ssh, "gw", users, args, cmd, reach, dirs)
        return ks.result(), gw.result()


# --------------------------------------------------------------------------
# authorized_keys / grants / sudoers / ssh config helpers

_KEY_RE = re.compile(r"(?:^|\s)((?:ssh-|ecdsa-|sk-)\S+)\s+([A-Za-z0-9+/]+={0,3})(?:\s+(.*))?$")


@dataclass
class KeyLine:
    line: str
    options: str
    blob: str
    comment: str

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256(base64.b64decode(self.blob)).digest()
        return "SHA256:" + base64.b64encode(digest).decode().rstrip("=")

    @property
    def command(self) -> str:
        m = re.search(r'command="((?:[^"\\]|\\.)*)"', self.options)
        return m.group(1) if m else ""

    @property
    def is_gateway_entry(self) -> bool:
        return "privjail-gateway" in self.command

    @property
    def is_keyserver_entry(self) -> bool:
        return self.command == "/bin/true" and 'permitopen="localhost:' in self.options

    @property
    def label(self) -> str:
        return self.comment or self.fingerprint


def parse_key_line(line: str) -> KeyLine | None:
    m = _KEY_RE.search(line)
    if not m:
        return None
    return KeyLine(line, line[:m.start(1)].strip(), m.group(2), (m.group(3) or "").strip())


def key_lines(uf: UserFacts | None) -> list[KeyLine]:
    return [k for k in (parse_key_line(l) for l in (uf.authorized_keys if uf else [])) if k]


def parse_grants(lines: list[str]) -> set[tuple[str, str]]:
    """Same format the key server reads: "<user> <dir>" per line, '#' comments."""
    grants: set[tuple[str, str]] = set()
    for line in lines:
        fields = line.split("#", 1)[0].split()
        if len(fields) == 2:
            grants.add((fields[0], fields[1]))
    return grants


def without_grants(lines: list[str], drop: set[tuple[str, str]]) -> list[str]:
    """The grants file minus the lines granting any (user, dir) in `drop`."""
    return [l for l in lines if not parse_grants([l]) & drop]


def sudoers_path(user: str) -> str:
    # sudo ignores files in sudoers.d whose name contains "." or ends in "~"
    return f"{SUDOERS_DIR}/privjail-{user.replace('.', '_')}"


def sudoers_content(user: str, quote_path: str) -> str:
    # the first line marks it as ours (status lists users by it);
    # the trailing "" forbids any arguments
    return f'{SUDOERS_MARK}\n{user} ALL=(root) NOPASSWD: {quote_path} ""\n'


def ssh_config_block(alias: str, hostname: str, port: int) -> list[str]:
    block = [SSH_CONFIG_BEGIN, f"Host {alias}", f"    HostName {hostname}"]
    if port != 22:
        block.append(f"    Port {port}")
    block += ["    StrictHostKeyChecking accept-new", SSH_CONFIG_END]
    return block


def managed_block(lines: list[str]) -> list[str]:
    if SSH_CONFIG_BEGIN not in lines:
        return []
    start = lines.index(SSH_CONFIG_BEGIN)
    end = lines.index(SSH_CONFIG_END, start) if SSH_CONFIG_END in lines[start:] else len(lines) - 1
    return lines[start:end + 1]


def strip_managed_block(lines: list[str]) -> list[str]:
    out: list[str] = []
    inside = False
    for line in lines:
        if line == SSH_CONFIG_BEGIN:
            inside = True
        elif inside:
            inside = line != SSH_CONFIG_END
        else:
            out.append(line)
    while out and not out[0].strip():
        out.pop(0)
    return out


def with_managed_block(lines: list[str], block: list[str]) -> list[str]:
    """Puts the managed block at the very top: ssh takes the first value it
    sees for each option, so a later `Host *` cannot override it."""
    rest = strip_managed_block(lines)
    return block + ([""] + rest if rest else [])


def read_pubkey_line(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        line = f.read().strip()
    if parse_key_line(line) is None:
        raise SystemExit(f"{path} does not look like an SSH public key")
    return line


# --------------------------------------------------------------------------
# writing: at most one ssh per host

WRITE_PRELUDE = r"""
set -eu
put_user_file() {  # user name b64: atomically replace ~user/.ssh/<name>
  printf '%s' "$3" | base64 -d | sudo -n -u "$1" -H bash -c \
    'umask 077; mkdir -p ~/.ssh && cat > ~/.ssh/'"$2"'.tmp && mv ~/.ssh/'"$2"'.tmp ~/.ssh/'"$2"
}
put_own_file() {  # path b64: atomically replace a file of the operator ("~" = operator's home)
  f=${1/#\~/$HOME}
  (umask 077; mkdir -p "$(dirname "$f")" && printf '%s' "$2" | base64 -d > "$f.tmp" && mv "$f.tmp" "$f")
}
append_own_line() {  # path b64line: append to a file of the operator, kept mode 600
  f=${1/#\~/$HOME}
  (umask 077; mkdir -p "$(dirname "$f")" && printf '%s' "$2" | base64 -d >> "$f")
}
drop_own_entry() {  # path key: atomically drop the lines whose first field is exactly key
  f=${1/#\~/$HOME}
  (umask 077; awk -v k="$2" '$1 != k' "$f" > "$f.tmp" && mv "$f.tmp" "$f")
}
create_encrypted_dir() {  # dir owner b64password: mkdir + gocryptfs -init as owner
  sudo -n mkdir -- "$1"
  sudo -n chown -- "$2:$(id -gn "$2")" "$1"
  sudo -n chmod 755 "$1"
  printf '%s' "$3" | base64 -d | sudo -n -u "$2" gocryptfs -q -init "$1" > /dev/null
  sudo -n chmod 444 "$1/gocryptfs.conf"
}
put_sudoers() {  # path b64: validated with visudo before going live
  tmp="$(dirname "$1")/.$(basename "$1").tmp"   # leading "." -> ignored by sudo
  printf '%s' "$2" | base64 -d | sudo -n tee "$tmp" >/dev/null
  sudo -n chmod 0440 "$tmp"
  if ! sudo -n visudo -cqf "$tmp"; then sudo -n rm -f "$tmp"; echo "visudo rejected $1" >&2; exit 1; fi
  sudo -n mv "$tmp" "$1"
}
"""


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def _content(lines: list[str]) -> str:
    return "\n".join(lines) + ("\n" if lines else "")


class Writer:
    def __init__(self) -> None:
        self.steps: list[str] = []

    def put_user_file(self, user: str, name: str, lines: list[str]) -> None:
        self.steps.append(f"put_user_file {shlex.quote(user)} {shlex.quote(name)} {_b64(_content(lines))}")

    def put_own_file(self, path: str, lines: list[str]) -> None:
        self.steps.append(f"put_own_file {shlex.quote(path)} {_b64(_content(lines))}")

    def put_sudoers(self, path: str, content: str) -> None:
        self.steps.append(f"put_sudoers {shlex.quote(path)} {_b64(content)}")

    def remove_sudoers(self, path: str) -> None:
        self.steps.append(f"sudo -n rm -f {shlex.quote(path)}")

    def append_own_line(self, path: str, line: str) -> None:
        self.steps.append(f"append_own_line {shlex.quote(path)} {_b64(line + chr(10))}")

    def drop_own_entry(self, path: str, key: str) -> None:
        self.steps.append(f"drop_own_entry {shlex.quote(path)} {shlex.quote(key)}")

    def create_encrypted_dir(self, path: str, owner: str, password: str) -> None:
        self.steps.append(f"create_encrypted_dir {shlex.quote(path)} {shlex.quote(owner)} {_b64(password + chr(10))}")

    def run(self, host: str) -> None:
        if not self.steps:
            return
        result = run_remote_script(host, WRITE_PRELUDE + "\n".join(self.steps) + "\n")
        if result.returncode != 0:
            raise SystemExit(f"failed to apply changes on {host}: {last_error_line(result.stderr) or 'error'}")


# --------------------------------------------------------------------------
# reporting

class Report:
    """What a command found and did. The CLI echoes it as text as it goes;
    the web UI (privjail-admin web) returns `entries` as JSON."""

    def __init__(self, echo: bool = True) -> None:
        self.errors = 0
        self.echo = echo
        self.entries: list[dict[str, str]] = []
        self.outcome = ""   # problems / nothing / dry-run / done (changing commands)
        self.data: dict[str, Any] = {}   # structured results (status), for the web UI

    def line(self, text: str = "") -> None:
        self.entries.append({"kind": "line", "text": text})
        if self.echo:
            print(text)

    def item(self, label: str, value: str, level: str = "ok") -> None:
        mark = {"ok": " ", "warn": "!", "ng": "x"}[level]
        if level == "ng":
            self.errors += 1
        self.entries.append({"kind": "item", "label": label, "value": value, "level": level})
        if self.echo:
            print(f"  {mark} {label:<22} {value}")


def host_header(rep: Report, title: str, host: str, facts: HostFacts) -> bool:
    rep.line(f"{title} {host}")
    if not facts.login:
        rep.item("ssh", facts.error, "ng")
    return facts.login


def user_ready(rep: Report, user: str, uf: UserFacts | None, label: str = "account") -> bool:
    if uf is None or not uf.exists:
        rep.item(label, f"{user} does not exist", "ng")
        return False
    if not uf.sudo:
        rep.item(label, f"cannot `sudo -u {user}`", "ng")
        return False
    return True


def dir_problems(spec: str, user: str, ks: HostFacts, gw: HostFacts) -> tuple[str, list[str]]:
    """(absolute path, problems) for directory `spec` as seen by `user`;
    needs gw probed with `spec` in dirs for `user`."""
    uf = gw.users.get(user)
    if not (gw.login and uf and spec in uf.dirs):
        return spec, ([] if not gw.login else ["cannot check on gateway"])
    path, exists, initialized, readable = uf.dirs[spec]
    problems = []
    if ks.login and path not in ks.registered:
        problems.append("no password registered on key server")
    if not exists:
        problems.append("does not exist on gateway (or not accessible to the user)")
    elif not initialized:
        problems.append("not a gocryptfs directory")
    elif not readable:
        problems.append("gocryptfs.conf not readable by the user")
    return path, problems


def finish(rep: Report, args: argparse.Namespace, writers: list[tuple[str, Writer]], done: str) -> None:
    rep.line()
    if rep.errors:
        rep.outcome = "problems"
        rep.line(f"{rep.errors} problem(s) (marked x); nothing changed.")
        return
    if not any(w.steps for _, w in writers):
        rep.outcome = "nothing"
        rep.line("Nothing to change.")
        return
    if args.dry_run:
        rep.outcome = "dry-run"
        rep.line("--dry-run: nothing changed.")
        return
    for host, w in writers:
        w.run(host)
    rep.outcome = "done"
    rep.line(done)


# --------------------------------------------------------------------------
# status

def cmd_status(args: argparse.Namespace, rep: Report) -> None:
    # round 1: who (privjail-admin sudoers entries, grants) and which directories;
    # round 2: the per-user details
    ks0, gw0 = probe_both(args, [])
    names = args.users or sorted(set(gw0.sudoers_users) | {u for u, _ in ks0.grants})
    dirs = sorted(set(ks0.registered) | {d for _, d in ks0.grants})
    ks, gw = probe_both(args, names, dirs=dirs)
    data: dict[str, Any] = {
        "key_server": {"host": args.key_server_ssh, "login": ks.login, "error": ks.error,
                       "listening": ks.listening, "port": args.key_server_port},
        "gateway": {"host": args.gateway_ssh, "login": gw.login, "error": gw.error},
        "dirs": [], "users": [],
    }
    rep.data = data

    if ks.login:
        rep.line(f"key server {args.key_server_ssh}: "
              + (f"listening on {args.key_server_port}" if ks.listening else f"NOT listening on {args.key_server_port}"))
        if not ks.listening:
            rep.errors += 1
    else:
        rep.line(f"key server {args.key_server_ssh}: ssh failed ({ks.error})")
    rep.line(f"gateway    {args.gateway_ssh}: " + ("reachable" if gw.login else f"ssh failed ({gw.error})"))

    if ks.login:
        rep.line()
        rep.line("encrypted dirs (passwords registered on the key server)")
        if not ks.registered:
            rep.line("    (none)")
        for d in ks.registered:
            exists, initialized = gw.dirs.get(d, (False, False))
            if not gw.login:
                note, level = "", "ok"
            elif not exists:
                note, level = "does not exist on gateway", "warn"
            else:
                note, level = ("", "ok") if initialized else ("not a gocryptfs directory", "warn")
            rep.item(d, note, level)
            data["dirs"].append({"path": d, "registered": True, "note": note, "level": level})
    for d in sorted({d for _, d in ks.grants} - set(ks.registered)):
        data["dirs"].append({"path": d, "registered": False, "note": "no password registered", "level": "ng"})

    if not names:
        rep.line()
        rep.line("no users (no privjail-admin sudoers entry on the gateway, no grants on the key server)")
        return

    for user in names:
        rep.line()
        rep.line(f"{user}")
        first, errors0 = len(rep.entries), rep.errors
        kf, gf = ks.users.get(user), gw.users.get(user)
        if ks.login and not (kf and kf.exists and kf.sudo):
            user_ready(rep, user, kf, "account (key server)")
        if gw.login and not (gf and gf.exists and gf.sudo):
            user_ready(rep, user, gf, "account (gateway)")
        ks_entries = [k for k in key_lines(kf) if k.is_keyserver_entry]
        gw_entries = [k for k in key_lines(gf) if k.is_gateway_entry]
        ks_blobs, gw_blobs = {k.blob for k in ks_entries}, {k.blob for k in gw_entries}

        if not gw_entries and not ks_entries:
            rep.item("user", "not added")
        for g in gw_entries:
            on_ks = g.blob in ks_blobs
            rep.item(f"key {g.label}", g.fingerprint + ("" if on_ks else "  (missing on key server)"),
                     "ok" if on_ks else "ng")
        for k in ks_entries:
            if k.blob not in gw_blobs:
                rep.item(f"key {k.label}", f"{k.fingerprint}  (only on key server)", "warn")

        if gf and gw_entries:
            for cmd in sorted({g.command.split()[0] for g in gw_entries}):
                cs = gf.cmds.get(cmd)
                ok = bool(cs and cs.executable)
                rep.item("forced command", cmd + ("" if ok else "  (not executable)"), "ok" if ok else "ng")
                if cs:
                    want = sudoers_content(user, cs.quote_path) if cs.quote_executable else None
                    good = want is not None and gf.sudoers == want
                    rep.item("sudoers", sudoers_path(user) if good else
                             ("missing or different" if want else f"{QUOTE_COMMAND} not found"),
                             "ok" if good else "ng")
            block = managed_block(gf.ssh_config)
            host_line = next((l.split()[1] for l in block if l.strip().startswith("HostName")), "")
            alias = next((l.split()[1] for l in block if l.startswith("Host ")), "")
            rep.item("ssh config", f"Host {alias} -> {host_line}" if block else "no managed Host block",
                     "ok" if block else "warn")
        elif gf:
            if gf.sudoers:
                rep.item("leftover", sudoers_path(user), "warn")
            if managed_block(gf.ssh_config):
                rep.item("leftover", "managed ~/.ssh/config block", "warn")

        details = rep.entries[first:]   # the account-level checks above
        granted = sorted(d for u, d in ks.grants if u == user)
        if not granted:
            rep.item("grants", "none")
        grants: dict[str, list[str]] = {}
        for d in granted:
            _, problems = dir_problems(d, user, ks, gw)
            grants[d] = problems
            rep.item("grant", d + ("  (" + "; ".join(problems) + ")" if problems else ""),
                     "ng" if problems else "ok")
        keys = {k.blob: k for k in gw_entries + ks_entries}
        data["users"].append({
            "name": user,
            "added": bool(gw_entries),
            "problems": rep.errors - errors0,
            "details": details,
            "keys": [{"label": k.label, "fingerprint": k.fingerprint,
                      "match": k.comment or k.blob[-40:],   # a substring that identifies the line
                      "gateway": b in gw_blobs, "key_server": b in ks_blobs} for b, k in keys.items()],
            "grants": grants,
        })



# --------------------------------------------------------------------------
# add-user / remove-user

def cmd_add_user(args: argparse.Namespace, rep: Report) -> None:
    user = args.user
    pubkey = read_pubkey_line(args.user_pubkey)
    key = parse_key_line(pubkey)
    assert key is not None
    cmd = args.privjail_gateway_cmd.format(user=user)
    ks_addr = (args.key_server_hostname, 22) if args.key_server_hostname \
        else resolve_ssh_target(args.key_server_host)

    ks, gw = probe_both(args, [user], cmd=cmd, reach=ks_addr)
    ks_writer, gw_writer = Writer(), Writer()

    if host_header(rep, "key server", args.key_server_ssh, ks):
        kf = ks.users.get(user)
        rep.item("key server", f"listening on {args.key_server_port}" if ks.listening
                 else f"NOT listening on {args.key_server_port}", "ok" if ks.listening else "warn")
        if user_ready(rep, user, kf):
            assert kf is not None
            line = (f'command="/bin/true",no-pty,no-agent-forwarding,no-X11-forwarding,'
                    f'permitopen="localhost:{args.key_server_port}" {pubkey}')
            if line in kf.authorized_keys:
                rep.item("authorized_keys", f"present ({key.label})")
            else:
                rep.item("authorized_keys", f"add ({key.label})")
                ks_writer.put_user_file(user, "authorized_keys", kf.authorized_keys + [line])

    rep.line()
    if host_header(rep, "gateway", args.gateway_ssh, gw):
        gf = gw.users.get(user)
        if not gw.root:
            rep.item("root", "operator cannot `sudo` (needed for sudoers)", "ng")
        rep.item("gocryptfs", gw.gocryptfs or "not found", "ok" if gw.gocryptfs else "ng")
        rep.item("key server reachable", f"{ks_addr[0]}:{ks_addr[1]}" + ("" if gw.reach else "  (cannot connect)"),
                 "ok" if gw.reach else "warn")
        if user_ready(rep, user, gf):
            assert gf is not None
            cs = gf.cmds.get(cmd)
            rep.item("privjail-gateway", cmd if cs and cs.executable else f"{cmd}  (not executable)",
                     "ok" if cs and cs.executable else "ng")
            quote_ok = bool(cs and cs.quote_executable)
            rep.item(QUOTE_COMMAND, cs.quote_path if cs and quote_ok else "not found next to its python",
                     "ok" if quote_ok else "ng")

            forced = f"{cmd} --key-server {args.key_server_host}:{args.key_server_port}"
            gw_line = f'command="{forced}",no-pty,no-X11-forwarding,permitopen="localhost:*" {pubkey}'
            if gw_line in gf.authorized_keys:
                rep.item("authorized_keys", f"present ({key.label})")
            else:
                rep.item("authorized_keys", f"add ({key.label})")
                gw_writer.put_user_file(user, "authorized_keys", gf.authorized_keys + [gw_line])

            if cs and quote_ok:
                want, path = sudoers_content(user, cs.quote_path), sudoers_path(user)
                if gf.sudoers == want:
                    rep.item("sudoers", f"present ({path})")
                else:
                    rep.item("sudoers", f"{'replace' if gf.sudoers else 'install'} {path}")
                    gw_writer.put_sudoers(path, want)

            config = with_managed_block(gf.ssh_config, ssh_config_block(args.key_server_host, *ks_addr))
            if gf.ssh_config in (config, config + [""]):
                rep.item("ssh config", f"present (Host {args.key_server_host} -> {ks_addr[0]})")
            else:
                rep.item("ssh config", f"write Host {args.key_server_host} -> {ks_addr[0]}")
                gw_writer.put_user_file(user, "config", config)

    finish(rep, args, [(args.key_server_ssh, ks_writer), (args.gateway_ssh, gw_writer)],
           f"Added '{user}'. Next: privjail-admin grant {user} <dir>; the user connects with "
           f"this key and `ForwardAgent yes`.")


def cmd_remove_user(args: argparse.Namespace, rep: Report) -> None:
    user = args.user
    ks, gw = probe_both(args, [user])
    removals: dict[str, list[str]] = {}

    for title, host, facts, gateway_side in (("key server", args.key_server_ssh, ks, False),
                                             ("gateway", args.gateway_ssh, gw, True)):
        if host_header(rep, title, host, facts) and user_ready(rep, user, facts.users.get(user)):
            entries = [k for k in key_lines(facts.users[user])
                       if (k.is_gateway_entry if gateway_side else k.is_keyserver_entry)]
            if args.match:
                entries = [k for k in entries if args.match in k.line]
                if len(entries) > 1:
                    rep.item("--match", f"matches {len(entries)} keys; refine it", "ng")
            for k in entries:
                rep.item("authorized_keys", f"remove ({k.label})")
            if not entries:
                rep.item("authorized_keys", "no matching entry")
            removals[title] = [k.line for k in entries]
        rep.line()

    ks_writer, gw_writer = Writer(), Writer()
    if removals.get("key server"):
        ks_writer.put_user_file(user, "authorized_keys",
                                [l for l in ks.users[user].authorized_keys if l not in removals["key server"]])
    if "gateway" in removals:
        gf = gw.users[user]
        remaining = [l for l in gf.authorized_keys if l not in removals["gateway"]]
        if removals["gateway"]:
            gw_writer.put_user_file(user, "authorized_keys", remaining)
        if not any(k.is_gateway_entry for k in (parse_key_line(l) for l in remaining) if k):
            if gf.sudoers:
                rep.line(f"no key left on the gateway: removing {sudoers_path(user)}")
                gw_writer.remove_sudoers(sudoers_path(user))
            if managed_block(gf.ssh_config):
                rep.line("no key left on the gateway: removing the managed ~/.ssh/config block")
                gw_writer.put_user_file(user, "config", strip_managed_block(gf.ssh_config))
            granted = {(u, d) for u, d in ks.grants if u == user}
            if granted:
                rep.line(f"no key left on the gateway: revoking {', '.join(sorted(d for _, d in granted))}")
                ks_writer.put_own_file(f"{args.ks_dir}/grants", without_grants(ks.grants_lines, granted))

    finish(rep, args, [(args.key_server_ssh, ks_writer), (args.gateway_ssh, gw_writer)], f"Removed '{user}'.")


# --------------------------------------------------------------------------
# grant / revoke

def check_path_arg(path: str) -> None:
    err = path_error(path)
    if err:
        raise SystemExit(f"invalid directory {path!r}: {err}")


def cmd_grant(args: argparse.Namespace, rep: Report) -> None:
    user, spec = args.user, args.path
    check_path_arg(spec)
    ks, gw = probe_both(args, [user], dirs=[spec])
    writer = Writer()

    path, problems = dir_problems(spec, user, ks, gw)
    rep.line(f"{user} -> {path}")
    if not ks.login:
        rep.item("key server", ks.error, "ng")
    if not gw.login:
        rep.item("gateway", gw.error, "ng")
    elif not user_ready(rep, user, gw.users.get(user)):
        pass
    elif not any(k.is_gateway_entry for k in key_lines(gw.users.get(user))):
        rep.item("user", f"not added (run privjail-admin add-user {user} first)", "ng")
    for problem in problems:
        rep.item("directory", problem, "ng")
    if ks.login and path.startswith("/"):
        if (user, path) in ks.grants:
            rep.item("grant", "already present")
        else:
            rep.item("grant", f"add to {args.ks_dir}/grants")
            writer.put_own_file(f"{args.ks_dir}/grants", ks.grants_lines + [f"{user} {path}"])

    finish(rep, args, [(args.key_server_ssh, writer)], f"Granted {path} to '{user}'.")


def cmd_revoke(args: argparse.Namespace, rep: Report) -> None:
    user, spec = args.user, args.path
    check_path_arg(spec)
    if spec.startswith("~/"):
        ks, gw = probe_both(args, [user], dirs=[spec])   # to expand ~ as on the gateway
        uf = gw.users.get(user)
        if not (uf and spec in uf.dirs):
            raise SystemExit(f"cannot expand {spec!r} for '{user}' on the gateway; give the absolute path")
        path = uf.dirs[spec][0]
    else:
        ks, path = probe(args.key_server_ssh, "ks", [], args), spec
    writer = Writer()

    rep.line(f"{user} -> {path}")
    if not ks.login:
        rep.item("key server", ks.error, "ng")
    elif (user, path) not in ks.grants:
        rep.item("grant", "not present")
    else:
        rep.item("grant", f"remove from {args.ks_dir}/grants")
        writer.put_own_file(f"{args.ks_dir}/grants", without_grants(ks.grants_lines, {(user, path)}))

    finish(rep, args, [(args.key_server_ssh, writer)],
           f"Revoked {path} from '{user}' (a session already running keeps its mount until it ends).")


# --------------------------------------------------------------------------
# add-dir / remove-dir

ADD_DIR_CHECK = r"""
set -u
e() { printf '%s\t%s\n' "$1" "$2"; }
yn() { "$@" >/dev/null 2>&1 && echo 1 || echo 0; }
e login 1
e root "$(yn sudo -n true)"
e gocryptfs "$(command -v gocryptfs || true)"
e exists "$(yn sudo -n test -e "$DIR")"
e isdir "$(yn sudo -n test -d "$DIR")"
e parent "$(yn sudo -n test -d "$(dirname "$DIR")")"
e conf "$(yn sudo -n test -f "$DIR/gocryptfs.conf")"
e conf_mode "$(sudo -n stat -c %A "$DIR/gocryptfs.conf" 2>/dev/null)"
e owner "$(yn id -u "$OWNER")"
if [ -n "$PWB64" ] && sudo -n test -f "$DIR/gocryptfs.conf"; then
  # check the password the way the gateway will use it: a read-only mount
  m=$(sudo -n mktemp -d)
  if err=$(printf '%s' "$PWB64" | base64 -d | sudo -n gocryptfs -ro -q "$DIR" "$m" 2>&1 >/dev/null); then
    e pw_ok 1
    sudo -n fusermount -u "$m" 2>/dev/null || sudo -n umount "$m"
  else
    e pw_ok 0
    e pw_err "$(printf '%s' "$err" | grep -v '^$' | tail -1)"
  fi
  sudo -n rmdir "$m"
fi
"""


def read_dir_password(args: argparse.Namespace) -> str:
    """From --dir-passfile (a file, or - for stdin), else asked on the terminal."""
    if args.dir_passfile == "-":
        text = sys.stdin.readline()
    elif args.dir_passfile:
        with open(os.path.expanduser(args.dir_passfile), encoding="utf-8") as f:
            text = f.readline()
    elif sys.stdin.isatty():
        text = getpass.getpass(f"gocryptfs password of {args.path}: ")
    else:
        raise SystemExit("no terminal to ask the password on: give --dir-passfile FILE (or - for stdin)")
    password = text.rstrip("\r\n")
    if not password or any(c.isspace() for c in password):
        raise SystemExit("the password must be one non-empty line without spaces")
    return password


def check_dir_arg(path: str) -> None:
    err = path_error(path)
    if err or not path.startswith("/"):
        raise SystemExit(f"invalid directory {path!r}: {err or 'must be an absolute path'}")


def cmd_add_dir(args: argparse.Namespace, rep: Report) -> None:
    path, existing = args.path, args.existing
    check_dir_arg(path)
    password = read_dir_password(args) if existing or args.dir_passfile else secrets.token_urlsafe(32)
    owner = args.owner or args.gateway_user
    params = {"DIR": path, "OWNER": owner,
              "PWB64": base64.b64encode((password + "\n").encode()).decode() if existing else ""}
    script = "".join(f"{k}={shlex.quote(v)}\n" for k, v in params.items()) + ADD_DIR_CHECK

    with ThreadPoolExecutor(2) as pool:
        ks_f = pool.submit(probe, args.key_server_ssh, "ks", [], args)
        gw_f = pool.submit(run_remote_script, args.gateway_ssh, script)
        ks, gw_result = ks_f.result(), gw_f.result()
    gw = dict(line.split("\t", 1) for line in gw_result.stdout.splitlines() if "\t" in line)
    ks_writer, gw_writer = Writer(), Writer()

    rep.line(f"{'register existing' if existing else 'create'} {path}")
    if not gw.get("login"):
        rep.item("gateway", last_error_line(gw_result.stderr) or "ssh failed", "ng")
    else:
        if gw.get("root") != "1":
            rep.item("root", "operator cannot `sudo` on the gateway", "ng")
        rep.item("gocryptfs", gw.get("gocryptfs") or "not found", "ok" if gw.get("gocryptfs") else "ng")
        if existing:
            if gw.get("isdir") != "1":
                rep.item("directory", "does not exist on the gateway (to create it, omit --existing)", "ng")
            elif gw.get("conf") != "1":
                rep.item("directory", "exists, but is not a gocryptfs directory", "ng")
            else:
                rep.item("directory", "exists, gocryptfs")
                if gw.get("pw_ok") == "1":
                    rep.item("password", "verified (read-only test mount)")
                else:
                    rep.item("password", f"does not unlock it: {gw.get('pw_err') or 'mount failed'}", "ng")
                if gw.get("conf_mode", "")[7:8] != "r":
                    rep.item("gocryptfs.conf", f"{gw.get('conf_mode')}: only its owner can read it; other "
                             f"users granted it need chmod 444", "warn")
        else:
            if gw.get("exists") == "1":
                rep.item("directory", "already exists on the gateway (to register it, use --existing)", "ng")
            elif gw.get("parent") != "1":
                rep.item("directory", f"parent {os.path.dirname(path)} does not exist", "ng")
            elif gw.get("owner") != "1":
                rep.item("owner", f"{owner} does not exist on the gateway", "ng")
            else:
                rep.item("directory", f"create, owner {owner}, gocryptfs -init with a new random password"
                         if not args.dir_passfile else
                         f"create, owner {owner}, gocryptfs -init with the given password")
                gw_writer.create_encrypted_dir(path, owner, password)

    if not ks.login:
        rep.item("key server", ks.error, "ng")
    elif path in ks.registered:
        rep.item("key server", f"{path} is already registered", "ng")
    else:
        rep.item("key server", f"register in {args.ks_dir}/passwords")
        ks_writer.append_own_line(f"{args.ks_dir}/passwords", f"{path} {password}")

    # the gateway first: nothing gets registered for a directory that failed to be created
    finish(rep, args, [(args.gateway_ssh, gw_writer), (args.key_server_ssh, ks_writer)],
           f"{'Registered' if existing else 'Created and registered'} {path}. Next: put data in it, "
           f"then privjail-admin grant <user> {path}.")


def cmd_remove_dir(args: argparse.Namespace, rep: Report) -> None:
    path = args.path
    check_dir_arg(path)
    ks = probe(args.key_server_ssh, "ks", [], args)
    writer = Writer()

    rep.line(f"unregister {path}")
    if not ks.login:
        rep.item("key server", ks.error, "ng")
    elif path not in ks.registered:
        rep.item("key server", "not registered")
    else:
        holders = sorted(u for u, d in ks.grants if d == path)
        if holders:
            rep.item("grants", f"still granted to {', '.join(holders)}; revoke first", "ng")
        else:
            rep.item("key server", f"remove from {args.ks_dir}/passwords (the directory itself is kept)")
            writer.drop_own_entry(f"{args.ks_dir}/passwords", path)

    finish(rep, args, [(args.key_server_ssh, writer)], f"Unregistered {path}; its data is untouched.")


# --------------------------------------------------------------------------
# CLI

class DefaultsFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Skips "(default: None)" for options that explain their own fallback."""
    def _get_help_string(self, action: argparse.Action) -> str:
        help_str = action.help or ""
        if not help_str or "%(default)" in help_str or action.default in (None, "", argparse.SUPPRESS):
            return help_str
        return help_str + " (default: %(default)s)"


def parse_key_server(spec: str) -> tuple[str, int]:
    """Parses "[host][:port]", defaulting either side that's omitted."""
    if not spec:
        return DEFAULT_KEY_SERVER_HOST, DEFAULT_KEY_SERVER_PORT
    host, sep, port_str = spec.rpartition(":")
    if not sep:
        return spec, DEFAULT_KEY_SERVER_PORT
    return host or DEFAULT_KEY_SERVER_HOST, int(port_str) if port_str else DEFAULT_KEY_SERVER_PORT


def add_common_args(parser: argparse.ArgumentParser, dry_run: bool = True) -> None:
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY, help="gateway host (as in your ssh config)")
    parser.add_argument("--key-server", default="", metavar="[HOST][:PORT]",
                        help=f"key server as [host][:port]; defaults to "
                             f"{DEFAULT_KEY_SERVER_HOST}:{DEFAULT_KEY_SERVER_PORT}")
    parser.add_argument("--ks-dir", default=DEFAULT_KS_DIR,
                        help="key server state directory (passwords, grants), as given to privjail-keyserver "
                             "--state-dir; '~' is the operator's home there")
    parser.add_argument("--adm-user", default=None,
                        help="SSH username used to administer both hosts; defaults to the current user")
    parser.add_argument("--gateway-adm-user", default=None, help="SSH username for the gateway; defaults to --adm-user")
    parser.add_argument("--key-server-adm-user", default=None,
                        help="SSH username for the key server; defaults to --adm-user")
    if dry_run:
        parser.add_argument("--dry-run", action="store_true", help="check and show what would change; change nothing")


class UsageError(Exception):
    pass


class _RaisingParser(argparse.ArgumentParser):
    """For the web UI: report bad arguments as an exception, not exit(2)."""
    def error(self, message: str) -> NoReturn:
        raise UsageError(message)


def build_parser(raising: bool = False) -> argparse.ArgumentParser:
    cls = _RaisingParser if raising else argparse.ArgumentParser
    parser = cls(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def subparser(name: str, help: str) -> argparse.ArgumentParser:
        p: argparse.ArgumentParser = sub.add_parser(name, help=help, formatter_class=DefaultsFormatter)
        return p

    p = subparser("status", "show users, their keys and grants, and what is missing")
    p.add_argument("users", nargs="*", metavar="user",
                   help="accounts to show; default: those with a privjail-admin sudoers entry or a grant")
    add_common_args(p, dry_run=False)
    p.set_defaults(func=cmd_status)

    p = subparser("add-user", "let an existing account use the gateway with its key")
    p.add_argument("user", help="OS account name, must already exist on both gateway and key server")
    p.add_argument("--user-pubkey", required=True, help="the user's SSH public key file (planted on both hosts)")
    p.add_argument("--privjail-gateway-cmd", default=DEFAULT_PRIVJAIL_GATEWAY_CMD,
                   help="privjail-gateway path on the gateway; '{user}' is substituted")
    p.add_argument("--key-server-hostname", default=None,
                   help="address the gateway should use for the key server (e.g. its internal IP); "
                        "defaults to what the key server host resolves to in your ssh config")
    add_common_args(p)
    p.set_defaults(func=cmd_add_user)

    p = subparser("remove-user", "remove a user's keys (and, with the last one, its setup and grants)")
    p.add_argument("user", help="OS account name")
    p.add_argument("--match", default=None,
                   help="only remove the key whose line contains this (e.g. its comment); default: all")
    add_common_args(p)
    p.set_defaults(func=cmd_remove_user)

    p = subparser("grant", "let a user mount an encrypted directory")
    p.add_argument("user", help="OS account name")
    p.add_argument("path", help="encrypted directory on the gateway: absolute, or ~/... in the user's home")
    add_common_args(p)
    p.set_defaults(func=cmd_grant)

    p = subparser("revoke", "take back a user's access to an encrypted directory")
    p.add_argument("user", help="OS account name")
    p.add_argument("path", help="encrypted directory on the gateway: absolute, or ~/... in the user's home")
    add_common_args(p)
    p.set_defaults(func=cmd_revoke)

    p = subparser("add-dir", "create and register an encrypted directory (or, with --existing, register one)")
    p.add_argument("path", help="absolute path of the directory on the gateway")
    p.add_argument("--existing", action="store_true",
                   help="the directory already exists as a gocryptfs directory: check its password and "
                        "register it (without this, it must not exist yet)")
    p.add_argument("--owner", default=None,
                   help="account to own a newly created directory; defaults to the admin account on the gateway")
    p.add_argument("--dir-passfile", default=None, metavar="FILE",
                   help="read the directory's gocryptfs password from FILE ('-' = stdin); with --existing "
                        "and no FILE it is asked on the terminal; when creating, defaults to a random one")
    add_common_args(p)
    p.set_defaults(func=cmd_add_dir)

    p = subparser("remove-dir", "unregister an encrypted directory (its data is kept)")
    p.add_argument("path", help="absolute path of the directory on the gateway")
    add_common_args(p)
    p.set_defaults(func=cmd_remove_dir)

    p = subparser("web", "serve a local web page for status and these commands (no authentication)")
    p.add_argument("--host", default="127.0.0.1", help="address to listen on; keep it local")
    p.add_argument("--port", type=int, default=8881, help="port to listen on")
    add_common_args(p, dry_run=False)
    p.set_defaults(func=cmd_web)
    return parser


def finalize_args(args: argparse.Namespace) -> None:
    args.key_server_host, args.key_server_port = parse_key_server(args.key_server)
    adm_user = args.adm_user or getpass.getuser()
    args.gateway_user = args.gateway_adm_user or adm_user
    args.gateway_ssh = f"{args.gateway_user}@{args.gateway}"
    args.key_server_ssh = f"{args.key_server_adm_user or adm_user}@{args.key_server_host}"


def run_command(argv: list[str], echo: bool) -> Report:
    """Runs one subcommand (not web) and returns what it reported.
    Raises UsageError for bad arguments when echo is False (web UI)."""
    args = build_parser(raising=not echo).parse_args(argv)
    if args.command == "web":
        raise UsageError("web cannot be run from the web UI")
    finalize_args(args)
    rep = Report(echo)
    args.func(args, rep)
    return rep


def cmd_web(args: argparse.Namespace, rep: Report) -> None:
    from .web import serve
    serve(args)


def main() -> None:
    args = build_parser().parse_args()
    finalize_args(args)
    rep = Report(echo=True)
    args.func(args, rep)
    sys.exit(1 if rep.errors else 0)


if __name__ == "__main__":
    main()
