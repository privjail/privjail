"""privjail-admin -- grant/revoke a user's access to a PrivJail gateway.

Run by the data owner, from a machine that can `ssh` into both the gateway
and the key server as an account with `sudo -u <user>` rights there.

Subcommands
-----------
grant   Idempotently appends the SAME user-supplied "mount key" as a
        forced-command entry in <user>'s authorized_keys on BOTH hosts:
          - on the gateway: restricted to running privjail-gateway with the
            given --data-enc/--data/--key-server, with agent forwarding
            *allowed* (see below).
          - on the key server: restricted to nothing but local port
            forwarding to the key server's own port, with agent forwarding
            disabled (this leaf doesn't need to forward any further).
        No new key material is ever generated or stored by this script.

        Reaching the key server from the gateway (step 2a) relies on SSH
        agent forwarding: the user connects to the gateway with agent
        forwarding enabled (`ssh -A gateway`, or `ForwardAgent yes`), so
        privjail-gateway -- running as the forced command in that same
        session -- inherits SSH_AUTH_SOCK. When it shells out
        `ssh -L 9443:localhost:9443 key-server`, that child process
        authenticates via the forwarded agent, i.e. the user's own laptop
        signs the challenge; the private key itself never leaves the
        laptop or touches the gateway's disk. Since it's the same keypair,
        the same public key needs to be (and is) authorized on the key
        server too.

        This does widen what the live session can do (a compromise of the
        gateway, as that account, during the session, could ride the agent
        to authenticate elsewhere the same pubkey is trusted) but the
        exposure is bounded and time-limited: it requires code execution as
        that specific account while the session is open, it can't exfiltrate
        the private key itself (only use it live), and what it can
        authenticate *as* is exactly these two already-tightly-scoped
        forced commands -- as long as this mount key is never reused
        anywhere with looser restrictions.

revoke  Removes the line matching --match (a substring, e.g. the pubkey's
        own comment like "tau@namo") from each host's authorized_keys.
        Substring match rather than line number: it's content-addressed, so
        it stays correct even if something else concurrently changed the
        file, whereas a line number could silently shift out from under you.
        If --match hits more than one line on either host, nothing is
        touched anywhere and both hosts' matches are printed so you can
        refine it; if it hits zero lines on a host, that host is just left
        alone (not an error -- you may only want to revoke one side).

Idempotency here is purely by exact line content: a grant re-run with the
same pubkey and the same options (--enc-dir/--mount-point/--privjail-gateway-
cmd/--key-server) is a no-op; if any of those differ, it appends a second,
distinct entry rather than silently rewriting the first one -- no separate
state file is kept, and no synthetic marker is added to the line; the
authorized_keys files themselves are the only state, and the pubkey's own
comment (e.g. "tau@namo") is left untouched for the admin to read.

`grant` writes nothing until every hard check passes (see --dry-run).
"""
from __future__ import annotations

import argparse
import getpass
import shlex
import subprocess
import sys
from dataclasses import dataclass

DEFAULT_KEY_SERVER_HOST = "privjail-ks"
DEFAULT_KEY_SERVER_PORT = 9443


class DefaultsFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Like ArgumentDefaultsHelpFormatter, but skips the auto-appended
    "(default: ...)" for options whose default is None or "" -- those already
    explain their effective fallback/default in their own help text, and
    "(default: None)" would just be noise."""
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


def ssh(host: str, remote_cmd: str, user: str | None = None,
        input_data: str | None = None, timeout: float = 30.0) -> subprocess.CompletedProcess:
    # ssh flattens every argument after `host` into one space-joined string
    # for the remote shell to re-parse, so remote_cmd must be pre-quoted to
    # survive as a single word -- otherwise its own spaces/quotes/redirects
    # get split apart before reaching the inner `bash -c` (see: `ssh host
    # bash -c 'cat file'` hangs on bare `cat` for exactly this reason).
    if user is not None:
        inner = ["sudo", "-u", user, "-H", "bash", "-c", shlex.quote(remote_cmd)]
    else:
        inner = ["bash", "-c", shlex.quote(remote_cmd)]
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "LogLevel=ERROR", host, *inner]
    return subprocess.run(
        command, input=input_data, capture_output=True, text=True, timeout=timeout,
    )


@dataclass
class Check:
    label: str       # fixed description, same wording regardless of outcome
    ok: bool
    hard: bool = True
    bracket: str = ""   # the key piece of info this check is about (host, path, port, ...)
    suffix: str = ""    # short trailing explanation, mainly used on failure


# Advisory banners ssh clients/servers print that aren't diagnostic of the
# remote command's own failure, so they should never be mistaken for "the
# reason it failed" -- e.g. OpenSSH's post-quantum-KEX upgrade notice
# ("** The server may need to be upgraded. See https://openssh.com/pq.html"),
# printed at a severity -o LogLevel=ERROR doesn't suppress; and
# "Warning: Permanently added '<host>' ..." from a ProxyJump hop, which spawns
# its own nested ssh process using ~/.ssh/config and so never sees our
# -o LogLevel=ERROR (that only applies to the outer connection).
_NOISE_PREFIXES = ("**", "Warning: Permanently added")


def _last_line(text: str) -> str:
    lines = [
        l for l in text.strip().splitlines()
        if l.strip() and not l.strip().startswith(_NOISE_PREFIXES)
    ]
    return lines[-1] if lines else ""


def print_check(c: Check) -> None:
    status = "OK" if c.ok else ("WARN" if not c.hard else "NG")
    line = f"  {c.label} ... {status} [{c.bracket}]"
    if c.suffix:
        line += f" {c.suffix}"
    print(line)


def check(section: list[Check], label: str, result: subprocess.CompletedProcess,
          bracket: str, hard: bool = True, suffix_if_failed: str | None = None,
          prefer_suffix: bool = False) -> bool:
    """prefer_suffix=True skips the real stderr/stdout and always uses
    suffix_if_failed -- for checks (like a raw /dev/tcp probe) whose real
    output is boilerplate, never a useful diagnostic."""
    ok = result.returncode == 0
    c = Check(
        label, ok, hard, bracket,
        suffix="" if ok else (
            suffix_if_failed if prefer_suffix else (
                _last_line(result.stderr)
                or _last_line(result.stdout)
                or suffix_if_failed
                or "failed"
            )
        ),
    )
    section.append(c)
    print_check(c)
    return ok




def expand_user_path(path: str, home: str) -> str:
    """Expands a leading "~" and any "$HOME" occurrences against `home` -- the
    granted *user's* home directory on the remote host, not the local operator's."""
    if path == "~":
        path = home
    elif path.startswith("~/"):
        path = home + path[1:]
    return path.replace("$HOME", home)


def account_checks(section: list[Check], host: str, user: str, title: str) -> tuple[bool, str]:
    """Appends SSH login / user exists / sudo -u / authorized_keys path / existing
    keys to `section`, printing each as it's performed. Returns (ok, home) -- ok
    iff all of these succeeded, i.e. it's safe to read or write that user's
    authorized_keys on this host; home is that user's $HOME on `host` (used to
    expand "~"/"$HOME" in other options). Assumes the operator running this
    script can `sudo -u <user>` on `host` -- that's what the "sudo" check below
    verifies."""
    print(f"{title}:")
    if not check(section, "SSH login", ssh(host, "true"), f"ssh {host}"):
        return False, ""
    if not check(section, "user exists", ssh(host, f"id -u {shlex.quote(user)}"), user,
                 suffix_if_failed="does not exist"):
        return False, ""
    if not check(section, "sudo", ssh(host, "id", user=user), f"sudo -u {user} id",
                 suffix_if_failed=f"cannot `sudo -u {user}`"):
        return False, ""
    home_result = ssh(host, 'echo "$HOME"', user=user)
    home = home_result.stdout.strip()
    ak_path = f"{home}/.ssh/authorized_keys" if home else ""
    if not check(section, "authorized_keys", home_result, ak_path,
                 suffix_if_failed="could not resolve path"):
        return False, ""
    line_count = len(read_authorized_keys(host, user))
    lines_check = Check("existing keys", True, True, f"{line_count} line(s)")
    section.append(lines_check)
    print_check(lines_check)
    return True, home


def all_hard_ok(checks: list[Check]) -> bool:
    return all(c.ok for c in checks if c.hard)


def report_already_granted(section: list[Check], host: str, user: str, line: str) -> None:
    already_present = line in read_authorized_keys(host, user)
    c = Check("this exact entry", True, True, "already present" if already_present else "not present yet")
    section.append(c)
    print_check(c)


def run_checks(args: argparse.Namespace, pubkey: str) -> tuple[list[Check], list[Check]]:
    gateway: list[Check] = []
    key_server: list[Check] = []

    ks_ok, _ = account_checks(key_server, args.key_server_ssh, args.user,
                               f"Key server [{args.key_server_host}]")
    if ks_ok:
        listening = ssh(args.key_server_ssh,
                         f"bash -c 'exec 3<>/dev/tcp/127.0.0.1/{args.key_server_port}'")
        check(key_server, "key server listening", listening, str(args.key_server_port),
              hard=False, suffix_if_failed="is not listened", prefer_suffix=True)
        args.keyserver_line = (
            f'command="/bin/true",no-pty,no-agent-forwarding,no-X11-forwarding,'
            f'permitopen="localhost:{args.key_server_port}" {pubkey}'
        )
        report_already_granted(key_server, args.key_server_ssh, args.user, args.keyserver_line)
    print()

    gw_ok, gw_home = account_checks(gateway, args.gateway_ssh, args.user,
                                     f"Gateway [{args.gateway}]")
    if gw_ok:
        args.privjail_gateway_cmd = expand_user_path(args.privjail_gateway_cmd, gw_home)
        args.enc_dir = expand_user_path(args.enc_dir, gw_home)
        args.mount_point = expand_user_path(args.mount_point, gw_home)

        privjail_gateway = ssh(args.gateway_ssh, f"command -v {shlex.quote(args.privjail_gateway_cmd)}",
                                user=args.user)
        resolved_cmd = privjail_gateway.stdout.strip()
        if check(gateway, "privjail-gateway command", privjail_gateway, resolved_cmd,
                 suffix_if_failed="not found on PATH"):
            # store the resolved absolute path so the forced command written to
            # authorized_keys is explicit and auditable, not a bare PATH-dependent name
            args.privjail_gateway_cmd = resolved_cmd
        check(gateway, "--enc-dir exists",
              ssh(args.gateway_ssh, f"test -d {shlex.quote(args.enc_dir)}", user=args.user), args.enc_dir,
              suffix_if_failed="does not exist")
        check(gateway, "--mount-point exists",
              ssh(args.gateway_ssh, f"test -d {shlex.quote(args.mount_point)}", user=args.user), args.mount_point,
              suffix_if_failed="does not exist")
        gocryptfs = ssh(args.gateway_ssh, "command -v gocryptfs", user=args.user)
        check(gateway, "gocryptfs is on PATH", gocryptfs, gocryptfs.stdout.strip(),
              suffix_if_failed="not found on PATH")

        gateway_command = (
            f"{args.privjail_gateway_cmd} --data-enc {args.enc_dir} --data {args.mount_point} "
            f"--key-server {args.key_server_host}:{args.key_server_port}"
        )
        args.gateway_line = (
            f'command="{gateway_command}",no-pty,no-X11-forwarding,'
            f'permitopen="localhost:*" {pubkey}'
        )
        report_already_granted(gateway, args.gateway_ssh, args.user, args.gateway_line)

    return gateway, key_server


def read_pubkey_line(path: str) -> str:
    """Returns the pubkey file's line verbatim (including its own comment,
    e.g. "tau@namo") -- nothing here rewrites or drops it."""
    with open(path, "r", encoding="utf-8") as f:
        line = f.read().strip()
    fields = line.split()
    if len(fields) < 2:
        raise ValueError(f"{path} does not look like an SSH public key")
    return line


def read_authorized_keys(host: str, user: str) -> list[str]:
    result = ssh(host, 'cat "$HOME/.ssh/authorized_keys" 2>/dev/null', user=user)
    return [l for l in result.stdout.splitlines() if l.strip()]


def write_authorized_keys(host: str, user: str, lines: list[str]) -> None:
    content = "\n".join(lines) + ("\n" if lines else "")
    setup = (
        'mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh" && '
        'cat > "$HOME/.ssh/authorized_keys.tmp" && '
        'chmod 600 "$HOME/.ssh/authorized_keys.tmp" && '
        'mv "$HOME/.ssh/authorized_keys.tmp" "$HOME/.ssh/authorized_keys"'
    )
    result = ssh(host, setup, user=user, input_data=content)
    if result.returncode != 0:
        raise RuntimeError(f"failed to update authorized_keys on {host} for {user}: {result.stderr}")


def append_if_missing(host: str, user: str, line: str) -> bool:
    """Appends `line` to authorized_keys unless that exact line is already
    present. Returns True iff it was newly added."""
    lines = read_authorized_keys(host, user)
    if line in lines:
        return False
    lines.append(line)
    write_authorized_keys(host, user, lines)
    return True


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("user", help="OS account name, must already exist on both gateway and key server")
    parser.add_argument("--gateway", default="privjail-gw", help="gateway hostname to administer")
    parser.add_argument("--key-server", default="", metavar="[HOST][:PORT]",
                         help=f"key server as [host][:port]; defaults to "
                              f"{DEFAULT_KEY_SERVER_HOST}:{DEFAULT_KEY_SERVER_PORT}")
    parser.add_argument("--adm-user", default=None,
                         help="SSH username this script uses to administer both hosts; "
                              "defaults to the current user")
    parser.add_argument("--gateway-adm-user", default=None,
                         help="SSH username for the gateway; defaults to --adm-user")
    parser.add_argument("--key-server-adm-user", default=None,
                         help="SSH username for the key server; defaults to --adm-user")


def cmd_grant(args: argparse.Namespace) -> None:
    args.privjail_gateway_cmd = args.privjail_gateway_cmd.format(user=args.user)
    pubkey = read_pubkey_line(args.user_pubkey)

    print(f"Checking environment (assumes this script's operator can `sudo -u {args.user}` "
          f"on both hosts):")
    gateway_checks, key_server_checks = run_checks(args, pubkey)
    all_ok = all_hard_ok(gateway_checks) and all_hard_ok(key_server_checks)

    if all_ok:
        print("\nAll requirements satisfied; ready to grant access.")
    else:
        print("\nOne or more requirements are not satisfied; cannot grant access.")

    if args.dry_run:
        print("no changes made (--dry-run)")
        sys.exit(0 if all_ok else 1)

    if not all_ok:
        sys.exit(1)

    added_gw = append_if_missing(args.gateway_ssh, args.user, args.gateway_line)
    print(f"\nGateway authorized_keys for '{args.user}': "
          f"{'added new entry' if added_gw else 'entry already present, unchanged'}.")

    added_ks = append_if_missing(args.key_server_ssh, args.user, args.keyserver_line)
    print(f"Key-server authorized_keys for '{args.user}': "
          f"{'added new entry' if added_ks else 'entry already present, unchanged'}.")

    print(
        f"\nGrant complete for '{args.user}'. The same mount key now authenticates on both "
        f"hosts; the user must connect to the gateway with agent forwarding enabled "
        f"(ssh -A / ForwardAgent yes) so privjail-gateway can reach the key server on port "
        f"{args.key_server_port} (see this script's module docstring for details)."
    )


def cmd_revoke(args: argparse.Namespace) -> None:
    print(f"Checking environment (assumes this script's operator can `sudo -u {args.user}` "
          f"on both hosts):")
    key_server_checks: list[Check] = []
    gateway_checks: list[Check] = []
    account_checks(key_server_checks, args.key_server_ssh, args.user,
                    f"Key server [{args.key_server_host}]")
    print()
    account_checks(gateway_checks, args.gateway_ssh, args.user, f"Gateway [{args.gateway}]")
    all_ok = all_hard_ok(gateway_checks) and all_hard_ok(key_server_checks)

    if all_ok:
        print("\nAll requirements satisfied.")
    else:
        print("\nOne or more requirements are not satisfied; cannot proceed.")
        sys.exit(1)

    ks_lines = read_authorized_keys(args.key_server_ssh, args.user)
    gw_lines = read_authorized_keys(args.gateway_ssh, args.user)
    ks_matches = [l for l in ks_lines if args.match in l]
    gw_matches = [l for l in gw_lines if args.match in l]

    def report(label: str, matches: list[str]) -> None:
        print(f"\n{label}: {len(matches)} matching line(s)")
        for line in matches:
            print(f"  {line}")

    report(f"Key server [{args.key_server_host}]", ks_matches)
    report(f"Gateway [{args.gateway}]", gw_matches)

    if len(ks_matches) > 1 or len(gw_matches) > 1:
        print("\n--match matched more than one line on at least one host; refine it. "
              "No changes made.", file=sys.stderr)
        sys.exit(1)

    if not ks_matches and not gw_matches:
        print("\nNo matching entry on either host; nothing to revoke.")
        return

    if args.dry_run:
        print("\nno changes made (--dry-run)")
        return

    if ks_matches:
        write_authorized_keys(args.key_server_ssh, args.user,
                               [l for l in ks_lines if l != ks_matches[0]])
    if gw_matches:
        write_authorized_keys(args.gateway_ssh, args.user,
                               [l for l in gw_lines if l != gw_matches[0]])

    print(f"\nKey server: {'removed matching entry' if ks_matches else 'no match, left unchanged'}.")
    print(f"Gateway: {'removed matching entry' if gw_matches else 'no match, left unchanged'}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    grant = subparsers.add_parser("grant", help="grant a user access to the gateway",
                               formatter_class=DefaultsFormatter)
    add_common_args(grant)
    grant.add_argument("--user-pubkey", required=True,
                        help="path to the user's SSH public key file (the 'mount key', "
                             "planted on both the gateway and the key server)")
    grant.add_argument("--privjail-gateway-cmd", default="privjail-gateway",
                        help="privjail-gateway command name or path on the gateway; a bare name "
                             "is looked up via the user's PATH (like gocryptfs), a path containing "
                             "'/' is used as-is; '{user}' is substituted if present")
    grant.add_argument("--enc-dir", default="/data/enc",
                        help="gocryptfs-encrypted data directory on the gateway")
    grant.add_argument("--mount-point", default="/data/plain",
                        help="plaintext mount point on the gateway")
    grant.add_argument("--dry-run", action="store_true",
                        help="check the environment and report what would change; make no changes")
    grant.set_defaults(func=cmd_grant)

    revoke = subparsers.add_parser("revoke", help="revoke a user's access to the gateway",
                                formatter_class=DefaultsFormatter)
    add_common_args(revoke)
    revoke.add_argument("--match", required=True,
                         help="substring identifying which authorized_keys line to remove "
                              "(e.g. the pubkey's own comment, like 'tau@namo'); must match "
                              "exactly one line per host, or that host is left unchanged")
    revoke.add_argument("--dry-run", action="store_true",
                         help="show what would be removed; make no changes")
    revoke.set_defaults(func=cmd_revoke)

    args = parser.parse_args()
    args.key_server_host, args.key_server_port = parse_key_server(args.key_server)

    adm_user = args.adm_user or getpass.getuser()
    args.gateway_adm_user = args.gateway_adm_user or adm_user
    args.key_server_adm_user = args.key_server_adm_user or adm_user
    # All admin-side SSH connections (login checks, sudo -u, authorized_keys
    # read/write) go to these; args.gateway/args.key_server_host stay as bare
    # hostnames for display and for the runtime --key-server value baked into
    # the gateway's forced command.
    args.gateway_ssh = f"{args.gateway_adm_user}@{args.gateway}"
    args.key_server_ssh = f"{args.key_server_adm_user}@{args.key_server_host}"

    args.func(args)


if __name__ == "__main__":
    main()
