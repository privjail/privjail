"""web.py -- `privjail-admin web`: a local page for status and the admin commands.

The page runs exactly the CLI subcommands (same argument parsing, defaults,
checks and changes) and shows what they report, so the CLI and the page can
never disagree. Each request becomes an argv for privjail-admin; the
equivalent command line is shown on the page.

Local use only, no authentication: it listens on 127.0.0.1 by default, only
answers requests whose Host is localhost/127.0.0.1 (against DNS rebinding),
and requires a custom header on POSTs (so other web sites cannot submit
forms to it). One command runs at a time.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any, ClassVar

from .cli import UsageError, run_command

COMMON_OPTIONS = {   # field -> CLI option, shared by all commands
    "gateway": "--gateway",
    "key_server": "--key-server",
    "ks_dir": "--ks-dir",
    "adm_user": "--adm-user",
    "gateway_adm_user": "--gateway-adm-user",
    "key_server_adm_user": "--key-server-adm-user",
}
COMMANDS: dict[str, dict[str, Any]] = {   # positional fields, option fields
    "status": {"positional": [], "options": {}},
    "add-user": {"positional": ["user"],
                 "options": {"privjail_gateway_cmd": "--privjail-gateway-cmd",
                             "key_server_hostname": "--key-server-hostname"}},
    "remove-user": {"positional": ["user"], "options": {"match": "--match"}},
    "grant": {"positional": ["user", "path"], "options": {}},
    "revoke": {"positional": ["user", "path"], "options": {}},
}
HEADER = "X-Privjail-Admin"

_lock = threading.Lock()


def _text(fields: dict[str, Any], name: str) -> str:
    value = fields.get(name, "")
    if not isinstance(value, str):
        raise UsageError(f"{name} must be a string")
    return value.strip()


def run_request(req: dict[str, Any]) -> dict[str, Any]:
    command = req.get("command")
    if command not in COMMANDS:
        raise UsageError(f"unknown command {command!r}")
    spec = COMMANDS[command]
    fields = req.get("fields") or {}
    common = req.get("common") or {}
    argv: list[str] = [command]
    shown: list[str] = ["privjail-admin", command]

    if command == "status":
        users = _text(fields, "users").split()
        argv += users
        shown += users
    for name in spec["positional"]:
        value = _text(fields, name)
        if not value:
            raise UsageError(f"{name} is required")
        argv.append(value)
        shown.append(value)
    for name, opt in [*spec["options"].items(), *COMMON_OPTIONS.items()]:
        value = _text(common if name in COMMON_OPTIONS else fields, name)
        if value:
            argv += [opt, value]
            shown += [opt, value]

    with tempfile.TemporaryDirectory(prefix="privjail-admin-web-") as tmp:
        if command == "add-user":
            key = _text(fields, "pubkey")
            if not key:
                raise UsageError("the user's public key is required")
            key_file = os.path.join(tmp, "user.pub")
            with open(key_file, "w") as f:
                f.write(key + "\n")
            argv += ["--user-pubkey", key_file]
            shown += ["--user-pubkey", "USER_KEY.pub"]
        if command != "status" and req.get("dry_run", True):
            argv.append("--dry-run")
            shown.append("--dry-run")
        rep = run_command(argv, echo=False)

    return {"command": shlex.join(shown), "entries": rep.entries,
            "errors": rep.errors, "outcome": rep.outcome, "data": rep.data}


class Handler(BaseHTTPRequestHandler):
    server_version = "privjail-admin-web"
    defaults: ClassVar[dict[str, str]] = {}

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} {format % args}", flush=True)

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        return host in ("localhost", "127.0.0.1", "::1")

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj: Any) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:
        if not self._host_ok():
            return self._json(HTTPStatus.FORBIDDEN, {"error": "bad Host"})
        if self.path == "/":
            page = resources.files("privjail.admin").joinpath("web.html").read_bytes()
            return self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
        if self.path == "/api/defaults":
            return self._json(HTTPStatus.OK, self.defaults)
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._host_ok() or self.headers.get(HEADER) != "1":
            return self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
        if self.path != "/api/run":
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(min(length, 1 << 20)) or b"{}")
        except ValueError:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "bad JSON"})
        if not _lock.acquire(blocking=False):
            return self._json(HTTPStatus.CONFLICT, {"error": "another command is running"})
        try:
            self._json(HTTPStatus.OK, run_request(req))
        except UsageError as e:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
        except SystemExit as e:   # the CLI's own fatal errors (bad key file, failed write, ...)
            self._json(HTTPStatus.OK, {"error": str(e.code)})
        except Exception as e:  # noqa: BLE001 -- report anything to the page
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(e).__name__}: {e}"})
        finally:
            _lock.release()


def serve(args: argparse.Namespace) -> None:
    Handler.defaults = {   # what the page shows as defaults in the common fields
        "gateway": args.gateway, "key_server": f"{args.key_server_host}:{args.key_server_port}",
        "ks_dir": args.ks_dir, "adm_user": args.adm_user or "",
        "gateway_adm_user": args.gateway_adm_user or "", "key_server_adm_user": args.key_server_adm_user or "",
    }
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"[web] WARNING: listening on {args.host}; this page has no authentication", flush=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[web] privjail-admin web on http://{args.host}:{args.port}/",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
