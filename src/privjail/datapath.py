"""datapath.py -- "mounting" data directories into the client's file namespace.

A session opened with mounts={R: L} (local_session, spawn_session,
gateway_session) makes directory R -- wherever the data server runs --
appear to privjail's file-reading functions (read_csv, load, and their
schema files) as the client's local directory L, like `mount R L`:

  - L must be an existing, empty local directory; mount points may not be
    nested or repeated. Only privjail's functions see R there -- `ls L`
    stays empty.
  - client side (`to_request_path`): a path given to those functions is made
    absolute against the current directory at call time, and must lie under
    one of the mount points; it is then sent to the server as that absolute
    path.
  - server side (`resolve`): the server holds the table {absolute L: R}
    (keyed by mount point, so one R could be mounted at several places),
    replaces the L prefix with R, and opens nothing outside the table.

Without mounts, paths are passed and opened as given (unchanged behavior;
also for connect_session, which does not take mounts).
"""
from __future__ import annotations

import os
import posixpath
from collections.abc import Mapping

_client_mounts: list[str] | None = None   # absolute local mount points, client side
_server_roots: dict[str, str] | None = None  # absolute local mount point -> real dir, server side


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def check_mount_points(points: list[str]) -> list[str]:
    """Absolute forms of the client-side mount points; each must be an
    existing empty directory, and none may contain another."""
    absolute = [os.path.normpath(os.path.abspath(os.path.expanduser(p))) for p in points]
    for given, a in zip(points, absolute):
        if not os.path.isdir(a):
            raise FileNotFoundError(f"mount point {given!r} ({a}) is not an existing directory")
        if os.listdir(a):
            raise OSError(f"mount point {given!r} ({a}) is not empty")
    for i, a in enumerate(absolute):
        for b in absolute[i + 1:]:
            if _under(a, b) or _under(b, a):
                raise ValueError(f"mount points may not be nested or repeated: {a}, {b}")
    return absolute


# -------------------------------------------------------------- client side

def set_client_mounts(mounts: list[str] | None) -> None:
    global _client_mounts
    _client_mounts = list(mounts) if mounts is not None else None


def to_request_path(path: str) -> str:
    if _client_mounts is None:
        return path
    p = os.path.normpath(os.path.join(os.getcwd(), os.path.expanduser(path)))
    if not any(_under(p, m) for m in _client_mounts):
        raise PermissionError(f"{path!r} ({p}) is not under a mounted directory; "
                              f"mounted: {', '.join(_client_mounts)}")
    return p


# -------------------------------------------------------------- server side

def set_roots(roots: Mapping[str, str] | None) -> None:
    global _server_roots
    _server_roots = dict(roots) if roots is not None else None


def resolve(path: str) -> str:
    if _server_roots is None:
        return path
    if posixpath.isabs(path) and ".." not in path.split("/"):
        for mount, real in _server_roots.items():
            if _under(path, mount):
                return real + path[len(mount.rstrip("/")):]
    raise PermissionError(f"{path!r} is not under a mounted directory; "
                          f"mounted: {', '.join(_server_roots) or '(none)'}")
