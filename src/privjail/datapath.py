"""datapath.py -- "mounting" data directories into the client's file namespace.

A session opened with mounts={R: L} (local_session, spawn_session,
gateway_session) makes directory R -- wherever the data server runs --
appear to privjail's file-reading functions (read_csv, load, and their
schema files) as the client's local directory L, like `mount R L`:

  - L must be an existing, empty local directory; mount points may not be
    nested or repeated. Only privjail's functions see R there -- `ls L`
    stays empty.
  - client side (`to_request_path`): holds {L: R}. A path given to those
    functions is made absolute against the current directory at call time,
    must lie under some L, and is sent as the corresponding path under R.
    The server never learns L.
  - server side (`resolve`, `canonical`): holds {R: (real directory,
    canonical name)} and opens nothing outside it. The real directory is
    where R is actually available (on the gateway, a hidden per-session
    mount point); the canonical name is R's absolute path (on the gateway,
    with ~ expanded), used to name budgets, so they follow the data, not
    where or how the client mounted it.

Without mounts, paths are passed and opened as given (unchanged behavior;
also for connect_session, which does not take mounts).
"""
from __future__ import annotations

import os
from collections.abc import Mapping

_client_mounts: list[tuple[str, str]] | None = None      # (absolute local L, R), client side
_server_roots: dict[str, tuple[str, str]] | None = None  # R -> (real dir, canonical name), server side


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _replace_prefix(path: str, old: str, new: str) -> str:
    return new.rstrip("/") + path[len(old.rstrip("/")):] if path != old else new


def check_mount_points(points: list[str]) -> list[str]:
    """Absolute forms of the client-side mount points; each must be an
    existing empty directory, and none may contain or repeat another."""
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

def set_client_mounts(mounts: list[tuple[str, str]] | None) -> None:
    """mounts: [(absolute local mount point L, R as the server knows it)]"""
    global _client_mounts
    _client_mounts = list(mounts) if mounts is not None else None


def to_request_path(path: str) -> str:
    if _client_mounts is None:
        return path
    p = os.path.normpath(os.path.join(os.getcwd(), os.path.expanduser(path)))
    for local, remote in _client_mounts:
        if _under(p, local):
            return _replace_prefix(p, local, remote)
    raise PermissionError(f"{path!r} ({p}) is not under a mounted directory; "
                          f"mounted: {', '.join(local for local, _ in _client_mounts)}")


# -------------------------------------------------------------- server side

def set_roots(roots: Mapping[str, tuple[str, str]] | None) -> None:
    """roots: {R: (real directory, canonical name)}"""
    global _server_roots
    _server_roots = dict(roots) if roots is not None else None


def _match(path: str) -> tuple[str, str, str]:
    assert _server_roots is not None
    if ".." not in path.split("/"):
        for remote, (real, canon) in _server_roots.items():
            if _under(path, remote):
                return remote, real, canon
    raise PermissionError(f"{path!r} is not under a mounted directory; "
                          f"mounted: {', '.join(_server_roots) or '(none)'}")


def resolve(path: str) -> str:
    """Where to actually open `path`."""
    if _server_roots is None:
        return path
    remote, real, _ = _match(path)
    return _replace_prefix(path, remote, real)


def canonical(path: str) -> str:
    """The name of `path` for budget accounting."""
    if _server_roots is None:
        return path
    remote, _, canon = _match(path)
    return _replace_prefix(path, remote, canon)
