"""privjail-admin -- grant/revoke/inspect users' access to a PrivJail gateway.

Run by the data owner, from a machine that can `ssh` into both the gateway
and the key server as an account with sudo rights there (`sudo -u <user>`,
and root on the gateway for the sudoers entry).

Subcommands
-----------
status  Shows, for each user (default: every account that has a privjail
        entry on either host), what is granted on the key server and the
        gateway and whether the pieces fit together: same key on both
        hosts, forced command / privjail-get-quote present, encrypted dir
        initialized, sudoers entry and ~/.ssh/config block in place, key
        server listening.

grant   Idempotently registers the user's "mount key" (--user-pubkey) on
        both hosts:
          - gateway authorized_keys: forced command privjail-gateway with
            --data-enc/--data/--key-server; agent forwarding allowed.
          - key server authorized_keys: nothing but local port forwarding
            to the key server port; agent forwarding disabled.
        and sets up on the gateway what the forced command needs at run time:
          - /etc/sudoers.d/privjail-<user>: run exactly
            `<bin>/privjail-get-quote` (no arguments) as root, where <bin> is
            the directory of privjail-gateway's interpreter. Checked with
            `visudo -c` before being put in place.
          - a managed block at the top of <user>'s ~/.ssh/config defining
            `Host <key server>` (address resolved through the operator's own
            ssh config, or --key-server-hostname) with
            `StrictHostKeyChecking accept-new`.
        No key material is generated. The gateway reaches the key server via
        the user's forwarded agent (`ForwardAgent yes`), i.e. the same key,
        which is why it is registered on both hosts.
        Nothing is written unless every check passes; --dry-run shows what
        would change.

revoke  Removes the authorized_keys line matching --match (a substring,
        typically the pubkey's comment) on each host. Aborts without changes
        if --match is ambiguous on either host. When the user's last
        privjail-gateway entry goes, the sudoers entry and the managed
        ~/.ssh/config block go too.

Each host is inspected with a single ssh session (both hosts in parallel) and
changed with at most one more.
"""
from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

DEFAULT_GATEWAY = "privjail-gw"
DEFAULT_KEY_SERVER_HOST = "privjail-ks"
DEFAULT_KEY_SERVER_PORT = 9443
DEFAULT_PRIVJAIL_GATEWAY_CMD = "/opt/privjail/.venv/bin/privjail-gateway"
DEFAULT_ENC_DIR = "~/data_enc"
DEFAULT_MOUNT_POINT = "~/data"

SUDOERS_DIR = "/etc/sudoers.d"
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
# (OpenSSH's post-quantum notice; ProxyJump hops' known_hosts warnings, which
# don't see our -o LogLevel=ERROR).
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
  e listening "$(yn timeout 3 bash -c "exec 3<>/dev/tcp/127.0.0.1/$KS_PORT")"
else
  e gocryptfs "$(command -v gocryptfs || true)"
  [ -n "$REACH_HOST" ] && e reach "$(yn timeout 5 bash -c "exec 3<>/dev/tcp/$REACH_HOST/$REACH_PORT")"
fi

if [ "$ALL" = 1 ]; then
  USERS=()
  while IFS=: read -r name _ uid _ _ home _; do
    [ "$uid" -ge 1000 ] 2>/dev/null && [ "$uid" -lt 65534 ] && USERS+=("$name")
  done < <(getent passwd)
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
  cmds=(); dirs=()
  [ -n "$CMD" ] && cmds+=("$CMD")
  [ -n "$ENC" ] && dirs+=("$ENC")
  [ -n "$MNT" ] && dirs+=("$MNT")
  while IFS= read -r l; do
    case "$l" in *privjail-gateway*) ;; *) continue ;; esac
    c=$(printf '%s' "$l" | sed -n 's/^command="\([^ "]*\).*/\1/p'); [ -n "$c" ] && cmds+=("$c")
    for o in --data-enc --data; do
      v=$(printf '%s' "$l" | sed -n "s/.*$o \([^ \"]*\).*/\1/p"); [ -n "$v" ] && dirs+=("$v")
    done
  done <<< "$ak"
  for c in ${cmds[@]+"${cmds[@]}"}; do
    c=${c/#\~/$h}; c=${c//\$HOME/$h}
    sb=$(sudo -n -u "$u" head -n1 "$c" 2>/dev/null || true)
    py=${sb#\#!}; py=${py%% *}; q=""
    case "$sb" in '#!/'*) case "$py" in */env) ;; *) q="$(dirname "$py")/$QUOTE_COMMAND" ;; esac ;; esac
    qx=0; [ -n "$q" ] && qx=$(yn sudo -n -u "$u" test -x "$q")
    e "$p:cmd:$c" "$(yn sudo -n -u "$u" test -x "$c") $qx $q"
  done
  for d in ${dirs[@]+"${dirs[@]}"}; do
    d=${d/#\~/$h}; d=${d//\$HOME/$h}
    e "$p:dir:$d" "$(yn sudo -n -u "$u" test -d "$d")$(yn sudo -n -u "$u" test -f "$d/gocryptfs.conf")$(yn sudo -n -u "$u" mountpoint -q "$d")"
  done
done
"""


@dataclass
class DirState:
    exists: bool
    gocryptfs: bool
    mounted: bool


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
    dirs: dict[str, DirState] = field(default_factory=dict)


@dataclass
class HostFacts:
    name: str
    login: bool = False
    error: str = ""
    root: bool = False
    listening: bool = False
    gocryptfs: str = ""
    reach: bool | None = None
    users: dict[str, UserFacts] = field(default_factory=dict)


def _b64lines(value: str) -> list[str]:
    return base64.b64decode(value).decode(errors="replace").splitlines() if value else []


def probe(host: str, role: str, users: list[str] | None, *, ks_port: int = DEFAULT_KEY_SERVER_PORT,
          cmd: str = "", enc: str = "", mnt: str = "",
          reach: tuple[str, int] | None = None) -> HostFacts:
    """users=None means: every account with a privjail entry on this host."""
    params = {
        "ROLE": role, "ALL": "1" if users is None else "0", "KS_PORT": str(ks_port),
        "CMD": cmd, "ENC": enc, "MNT": mnt,
        "REACH_HOST": reach[0] if reach else "", "REACH_PORT": str(reach[1]) if reach else "",
        "SUDOERS_DIR": SUDOERS_DIR, "QUOTE_COMMAND": QUOTE_COMMAND,
    }
    header = "".join(f"{k}={shlex.quote(v)}\n" for k, v in params.items())
    header += "USERS=(" + " ".join(shlex.quote(u) for u in users or []) + ")\n"
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
        elif key == "gocryptfs":
            facts.gocryptfs = value
        elif key == "reach":
            facts.reach = value == "1"
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
                uf.authorized_keys = [l for l in _b64lines(value) if l.strip()]
            elif attr == "config":
                uf.ssh_config = _b64lines(value)
            elif attr == "sudoers":
                uf.sudoers = base64.b64decode(value).decode(errors="replace") if value else ""
            elif attr.startswith("cmd:"):
                x, qx, *q = value.split(" ", 2)
                uf.cmds[attr[4:]] = CmdState(x == "1", qx == "1", q[0] if q else "")
            elif attr.startswith("dir:"):
                uf.dirs[attr[4:]] = DirState(value[0] == "1", value[1] == "1", value[2] == "1")
    return facts


def probe_both(args: argparse.Namespace, users: list[str] | None, **gw_kwargs: object) -> tuple[HostFacts, HostFacts]:
    with ThreadPoolExecutor(2) as pool:
        ks = pool.submit(probe, args.key_server_ssh, "ks", users, ks_port=args.key_server_port)
        gw = pool.submit(probe, args.gateway_ssh, "gw", users, **gw_kwargs)  # type: ignore[arg-type]
        return ks.result(), gw.result()


# --------------------------------------------------------------------------
# authorized_keys / sudoers / ssh config helpers

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

    def option_value(self, name: str) -> str:
        """Value of `--name X` inside the forced command."""
        m = re.search(rf"(?:^|\s){re.escape(name)}\s+(\S+)", self.command)
        return m.group(1) if m else ""

    @property
    def is_gateway_grant(self) -> bool:
        return "privjail-gateway" in self.command

    @property
    def is_keyserver_grant(self) -> bool:
        return self.command == "/bin/true" and 'permitopen="localhost:' in self.options

    @property
    def label(self) -> str:
        return self.comment or self.fingerprint


def parse_key_line(line: str) -> KeyLine | None:
    m = _KEY_RE.search(line)
    if not m:
        return None
    return KeyLine(line, line[:m.start(1)].strip(), m.group(2), (m.group(3) or "").strip())


def sudoers_path(user: str) -> str:
    # sudo ignores files in sudoers.d whose name contains "." or ends in "~"
    return f"{SUDOERS_DIR}/privjail-{user.replace('.', '_')}"


def sudoers_content(user: str, quote_path: str) -> str:
    # the trailing "" forbids any arguments
    return f'# managed by privjail-admin\n{user} ALL=(root) NOPASSWD: {quote_path} ""\n'


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


def config_mentions_host(lines: list[str], alias: str) -> bool:
    return any(l.split()[:1] == ["Host"] and alias in l.split()[1:] for l in lines if l.strip())


def expand_user_path(path: str, home: str) -> str:
    if path == "~":
        path = home
    elif path.startswith("~/"):
        path = home + path[1:]
    return path.replace("$HOME", home)


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
put_sudoers() {  # path b64: validated with visudo before going live
  tmp="$(dirname "$1")/.$(basename "$1").tmp"   # leading "." -> ignored by sudo
  printf '%s' "$2" | base64 -d | sudo -n tee "$tmp" >/dev/null
  sudo -n chmod 0440 "$tmp"
  if ! sudo -n visudo -cqf "$tmp"; then sudo -n rm -f "$tmp"; echo "visudo rejected $1" >&2; exit 1; fi
  sudo -n mv "$tmp" "$1"
}
"""


class Writer:
    def __init__(self) -> None:
        self.steps: list[str] = []

    @staticmethod
    def _b64(text: str) -> str:
        return base64.b64encode(text.encode()).decode()

    def put_user_file(self, user: str, name: str, lines: list[str]) -> None:
        content = "\n".join(lines) + ("\n" if lines else "")
        self.steps.append(f"put_user_file {shlex.quote(user)} {shlex.quote(name)} {self._b64(content)}")

    def put_sudoers(self, path: str, content: str) -> None:
        self.steps.append(f"put_sudoers {shlex.quote(path)} {self._b64(content)}")

    def remove_sudoers(self, path: str) -> None:
        self.steps.append(f"sudo -n rm -f {shlex.quote(path)}")

    def run(self, host: str) -> None:
        if not self.steps:
            return
        result = run_remote_script(host, WRITE_PRELUDE + "\n".join(self.steps) + "\n")
        if result.returncode != 0:
            raise SystemExit(f"failed to apply changes on {host}: {last_error_line(result.stderr) or 'error'}")


# --------------------------------------------------------------------------
# reporting

class Report:
    def __init__(self) -> None:
        self.errors = 0

    def head(self, text: str) -> None:
        print(text)

    def item(self, label: str, value: str, level: str = "ok") -> None:
        mark = {"ok": " ", "warn": "!", "ng": "x"}[level]
        if level == "ng":
            self.errors += 1
        print(f"  {mark} {label:<22} {value}")


def host_header(rep: Report, title: str, host: str, facts: HostFacts) -> bool:
    if not facts.login:
        rep.head(f"{title} {host}")
        rep.item("ssh", facts.error, "ng")
        return False
    rep.head(f"{title} {host}")
    return True


def user_ready(rep: Report, user: str, uf: UserFacts | None, label: str = "account") -> bool:
    if uf is None or not uf.exists:
        rep.item(label, f"{user} does not exist", "ng")
        return False
    if not uf.sudo:
        rep.item(label, f"cannot `sudo -u {user}`", "ng")
        return False
    return True


def tilde(path: str, home: str) -> str:
    return "~" + path[len(home):] if home and (path == home or path.startswith(home + "/")) else path


# --------------------------------------------------------------------------
# status

def has_privjail_state(uf: UserFacts | None) -> bool:
    """Any privjail grant line, sudoers entry or managed ssh config block."""
    if uf is None:
        return False
    keys = (parse_key_line(l) for l in uf.authorized_keys)
    return (any(k and (k.is_gateway_grant or k.is_keyserver_grant) for k in keys)
            or bool(uf.sudoers) or bool(managed_block(uf.ssh_config)))


def cmd_status(args: argparse.Namespace) -> None:
    users = args.users or None
    ks, gw = probe_both(args, users)
    names = users or sorted(u for u in set(ks.users) | set(gw.users)
                            if has_privjail_state(ks.users.get(u)) or has_privjail_state(gw.users.get(u)))
    rep = Report()

    rep.head(f"key server {args.key_server_ssh}: " + (
        ("listening on " if ks.listening else "NOT listening on ") + str(args.key_server_port)
        if ks.login else f"ssh failed ({ks.error})"))
    rep.head(f"gateway    {args.gateway_ssh}: " + ("reachable" if gw.login else f"ssh failed ({gw.error})"))
    if not ks.listening and ks.login:
        rep.errors += 1
    if not names:
        print("\nno user has a privjail entry on either host")
        return

    for user in names:
        print(f"\n{user}")
        kf, gf = ks.users.get(user), gw.users.get(user)
        ks_keys = [k for k in (parse_key_line(l) for l in (kf.authorized_keys if kf else [])) if k]
        gw_keys = [k for k in (parse_key_line(l) for l in (gf.authorized_keys if gf else [])) if k]
        ks_grants = [k for k in ks_keys if k.is_keyserver_grant]
        gw_grants = [k for k in gw_keys if k.is_gateway_grant]
        ks_blobs = {k.blob for k in ks_keys}
        gw_blobs = {k.blob for k in gw_grants}

        if ks.login and not (kf and kf.exists and kf.sudo):
            user_ready(rep, user, kf, "account (key server)")
        if gw.login and not (gf and gf.exists and gf.sudo):
            user_ready(rep, user, gf, "account (gateway)")
        if not gw_grants and not ks_grants:
            rep.item("grants", "none")
            continue

        home = gf.home if gf else ""
        for g in gw_grants:
            ks_spec = g.option_value("--key-server") or "(default)"
            enc, data = g.option_value("--data-enc"), g.option_value("--data")
            rep.item(f"key {g.label}", g.fingerprint)
            rep.item("  key server entry", "present" if g.blob in ks_blobs else "MISSING",
                     "ok" if g.blob in ks_blobs else "ng")
            cmd = g.command.split()[0]
            cs = gf.cmds.get(cmd) if gf else None
            rep.item("  forced command", cmd + ("" if cs and cs.executable else "  (not executable)"),
                     "ok" if cs and cs.executable else "ng")
            if enc and data and gf:
                es, ds = gf.dirs.get(enc), gf.dirs.get(data)
                enc_ok = bool(es and es.exists and es.gocryptfs)
                rep.item("  data", f"{tilde(enc, home)} -> {tilde(data, home)}"
                         + ("" if enc_ok else "  (encrypted dir missing/uninitialized)")
                         + ("" if ds and ds.exists else "  (mount point missing)")
                         + ("  [mounted]" if ds and ds.mounted else ""),
                         "ok" if enc_ok and ds and ds.exists else "ng")
            if gf and cs:
                want = sudoers_content(user, cs.quote_path) if cs.quote_executable else None
                ok = want is not None and gf.sudoers == want
                rep.item("  sudoers", sudoers_path(user) if ok else
                         ("missing or different" if want else f"{QUOTE_COMMAND} not found"),
                         "ok" if ok else "ng")
            if gf:
                alias = ks_spec.rsplit(":", 1)[0] if ":" in ks_spec else DEFAULT_KEY_SERVER_HOST
                block = managed_block(gf.ssh_config)
                if config_mentions_host(block, alias):
                    host_line = next((l.split()[1] for l in block if l.strip().startswith("HostName")), "?")
                    rep.item("  ssh config", f"Host {alias} -> {host_line}")
                elif config_mentions_host(gf.ssh_config, alias):
                    rep.item("  ssh config", f"Host {alias} (not managed by privjail-admin)", "warn")
                else:
                    rep.item("  ssh config", f"no Host {alias} (relies on DNS/hosts)", "warn")
        for k in ks_grants:
            if k.blob not in gw_blobs:
                rep.item(f"key {k.label}", f"{k.fingerprint}  only on key server", "warn")
        if gf and gf.exists and not gw_grants:
            if gf.sudoers:
                rep.item("leftover", sudoers_path(user), "warn")
            if managed_block(gf.ssh_config):
                rep.item("leftover", "managed ~/.ssh/config block", "warn")

    if rep.errors:
        sys.exit(1)


# --------------------------------------------------------------------------
# grant

def cmd_grant(args: argparse.Namespace) -> None:
    user = args.user
    pubkey = read_pubkey_line(args.user_pubkey)
    key = parse_key_line(pubkey)
    assert key is not None
    cmd = args.privjail_gateway_cmd.format(user=user)
    ks_addr = (args.key_server_hostname, 22) if args.key_server_hostname \
        else resolve_ssh_target(args.key_server_host)

    ks, gw = probe_both(args, [user], cmd=cmd, enc=args.enc_dir, mnt=args.mount_point, reach=ks_addr)
    rep = Report()
    ks_writer, gw_writer = Writer(), Writer()

    # key server
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

    # gateway
    print()
    if host_header(rep, "gateway", args.gateway_ssh, gw):
        gf = gw.users.get(user)
        if not gw.root:
            rep.item("root", "operator cannot `sudo` (needed for sudoers)", "ng")
        rep.item("gocryptfs", gw.gocryptfs or "not found", "ok" if gw.gocryptfs else "ng")
        rep.item("key server reachable", f"{ks_addr[0]}:{ks_addr[1]}" + ("" if gw.reach else "  (cannot connect)"),
                 "ok" if gw.reach else "warn")
        if user_ready(rep, user, gf):
            assert gf is not None
            home = gf.home
            cmd_abs, enc, mnt = (expand_user_path(p, home) for p in (cmd, args.enc_dir, args.mount_point))
            cs = gf.cmds.get(cmd_abs)
            rep.item("privjail-gateway", cmd_abs if cs and cs.executable else f"{cmd_abs}  (not executable)",
                     "ok" if cs and cs.executable else "ng")
            quote_ok = bool(cs and cs.quote_executable)
            rep.item(QUOTE_COMMAND, cs.quote_path if cs and quote_ok else "not found next to its python",
                     "ok" if quote_ok else "ng")
            es, ms = gf.dirs.get(enc), gf.dirs.get(mnt)
            rep.item("encrypted dir", tilde(enc, home) + (
                "" if es and es.gocryptfs else "  (missing)" if not (es and es.exists)
                else "  (not gocryptfs-initialized)"), "ok" if es and es.gocryptfs else "ng")
            rep.item("mount point", tilde(mnt, home) + ("" if ms and ms.exists else "  (missing)"),
                     "ok" if ms and ms.exists else "ng")

            gw_line = (f'command="{cmd_abs} --data-enc {enc} --data {mnt} '
                       f'--key-server {args.key_server_host}:{args.key_server_port}",'
                       f'no-pty,no-X11-forwarding,permitopen="localhost:*" {pubkey}')
            if gw_line in gf.authorized_keys:
                rep.item("authorized_keys", f"present ({key.label})")
            else:
                rep.item("authorized_keys", f"add ({key.label})")
                gw_writer.put_user_file(user, "authorized_keys", gf.authorized_keys + [gw_line])

            if cs and quote_ok:
                want = sudoers_content(user, cs.quote_path)
                path = sudoers_path(user)
                if gf.sudoers == want:
                    rep.item("sudoers", f"present ({path})")
                else:
                    rep.item("sudoers", f"{'replace' if gf.sudoers else 'install'} {path}")
                    gw_writer.put_sudoers(path, want)

            config = with_managed_block(gf.ssh_config,
                                        ssh_config_block(args.key_server_host, *ks_addr))
            if gf.ssh_config == config or gf.ssh_config == config + [""]:
                rep.item("ssh config", f"present (Host {args.key_server_host} -> {ks_addr[0]})")
            else:
                rep.item("ssh config", f"write Host {args.key_server_host} -> {ks_addr[0]}")
                gw_writer.put_user_file(user, "config", config)

    print()
    if rep.errors:
        print(f"{rep.errors} problem(s) (marked x); nothing changed.")
        sys.exit(1)
    if not ks_writer.steps and not gw_writer.steps:
        print("Already granted; nothing to do.")
        return
    if args.dry_run:
        print("Ready to grant. (--dry-run: nothing changed)")
        return
    ks_writer.run(args.key_server_ssh)
    gw_writer.run(args.gateway_ssh)
    print(f"Granted. '{user}' connects to the gateway with this key and `ForwardAgent yes`.")


# --------------------------------------------------------------------------
# revoke

def cmd_revoke(args: argparse.Namespace) -> None:
    user = args.user
    ks, gw = probe_both(args, [user])
    rep = Report()
    plans: list[tuple[str, HostFacts, list[str]]] = []

    for title, host, facts in (("key server", args.key_server_ssh, ks), ("gateway", args.gateway_ssh, gw)):
        if host_header(rep, title, host, facts):
            uf = facts.users.get(user)
            if user_ready(rep, user, uf):
                assert uf is not None
                matches = [l for l in uf.authorized_keys if args.match in l]
                for m in matches:
                    k = parse_key_line(m)
                    rep.item("match", k.label if k else m[:60], "ng" if len(matches) > 1 else "ok")
                if not matches:
                    rep.item("match", "none (left unchanged)")
                plans.append((title, facts, matches))
        print()

    if rep.errors:
        print("nothing changed" + (" (--match must hit at most one line per host)"
                                   if any(len(m) > 1 for _, _, m in plans) else ""))
        sys.exit(1)
    if not any(m for _, _, m in plans):
        print("No matching entry on either host; nothing to revoke.")
        return

    ks_writer, gw_writer = Writer(), Writer()
    for title, facts, matches in plans:
        if not matches:
            continue
        uf = facts.users[user]
        remaining = [l for l in uf.authorized_keys if l != matches[0]]
        (ks_writer if title == "key server" else gw_writer).put_user_file(user, "authorized_keys", remaining)
        if title == "gateway" and not any("privjail-gateway" in l for l in remaining):
            print(f"last privjail-gateway entry for '{user}': also removing "
                  f"{sudoers_path(user)} and the managed ~/.ssh/config block")
            gw_writer.remove_sudoers(sudoers_path(user))
            stripped = strip_managed_block(uf.ssh_config)
            if stripped != uf.ssh_config:
                gw_writer.put_user_file(user, "config", stripped)

    if args.dry_run:
        print("--dry-run: nothing changed")
        return
    ks_writer.run(args.key_server_ssh)
    gw_writer.run(args.gateway_ssh)
    print("Revoked.")


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


def add_host_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY, help="gateway host (as in your ssh config)")
    parser.add_argument("--key-server", default="", metavar="[HOST][:PORT]",
                        help=f"key server as [host][:port]; defaults to "
                             f"{DEFAULT_KEY_SERVER_HOST}:{DEFAULT_KEY_SERVER_PORT}")
    parser.add_argument("--adm-user", default=None,
                        help="SSH username used to administer both hosts; defaults to the current user")
    parser.add_argument("--gateway-adm-user", default=None, help="SSH username for the gateway; defaults to --adm-user")
    parser.add_argument("--key-server-adm-user", default=None,
                        help="SSH username for the key server; defaults to --adm-user")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="show users' grants and whether they are complete",
                            formatter_class=DefaultsFormatter)
    status.add_argument("users", nargs="*", metavar="user",
                        help="accounts to show; default: every account with a privjail entry")
    add_host_args(status)
    status.set_defaults(func=cmd_status)

    grant = sub.add_parser("grant", help="grant a user access to the gateway", formatter_class=DefaultsFormatter)
    grant.add_argument("user", help="OS account name, must already exist on both gateway and key server")
    add_host_args(grant)
    grant.add_argument("--user-pubkey", required=True,
                       help="the user's SSH public key file (planted on both hosts)")
    grant.add_argument("--privjail-gateway-cmd", default=DEFAULT_PRIVJAIL_GATEWAY_CMD,
                       help="privjail-gateway path on the gateway; '~' is the user's home, '{user}' is substituted")
    grant.add_argument("--enc-dir", default=DEFAULT_ENC_DIR,
                       help="gocryptfs-encrypted data directory on the gateway ('~' = the user's home)")
    grant.add_argument("--mount-point", default=DEFAULT_MOUNT_POINT,
                       help="plaintext mount point on the gateway ('~' = the user's home)")
    grant.add_argument("--key-server-hostname", default=None,
                       help="address the gateway should use for the key server (e.g. its internal IP); "
                            "defaults to what the key server host resolves to in your ssh config")
    grant.add_argument("--dry-run", action="store_true", help="check and show what would change; change nothing")
    grant.set_defaults(func=cmd_grant)

    revoke = sub.add_parser("revoke", help="revoke a user's access to the gateway", formatter_class=DefaultsFormatter)
    revoke.add_argument("user", help="OS account name")
    add_host_args(revoke)
    revoke.add_argument("--match", required=True,
                        help="substring identifying the authorized_keys line to remove (e.g. the key's comment)")
    revoke.add_argument("--dry-run", action="store_true", help="show what would be removed; change nothing")
    revoke.set_defaults(func=cmd_revoke)

    args = parser.parse_args()
    args.key_server_host, args.key_server_port = parse_key_server(args.key_server)
    adm_user = args.adm_user or getpass.getuser()
    args.gateway_ssh = f"{args.gateway_adm_user or adm_user}@{args.gateway}"
    args.key_server_ssh = f"{args.key_server_adm_user or adm_user}@{args.key_server_host}"
    args.func(args)


if __name__ == "__main__":
    main()
