"""dirspec.py -- how encrypted directories and their client-side names are written.

An encrypted directory is identified by its path on the gateway: an
absolute path, or "~/..." meaning the home directory of the user it is
granted to (expanded on the gateway, so the key server only ever sees
absolute paths). Paths are kept to a plain, normalized form -- no spaces,
no "." / ".." / empty components, no trailing "/" -- so that each directory
has exactly one spelling.

A name is what the client calls a directory in its own file paths
("NAME/file.csv").
"""
from __future__ import annotations

import posixpath
import re

_PATH_CHARS_RE = re.compile(r"^[A-Za-z0-9._/~-]+$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def path_error(path: str) -> str | None:
    """None if `path` is a valid directory spec, else why not."""
    if not path.startswith(("/", "~/")):
        return "must be an absolute path or start with ~/"
    if not _PATH_CHARS_RE.match(path) or "~" in path[1:]:
        return "may contain only letters, digits, '.', '_', '-' and '/'"
    rest = path[1:] if path.startswith("/") else path[2:]
    if not rest or any(part in ("", ".", "..") for part in rest.split("/")):
        return "must be normalized (no empty, '.' or '..' components, no trailing '/')"
    return None


def expand(path: str, home: str) -> str:
    """'~/x' -> '<home>/x'; absolute paths unchanged."""
    return posixpath.join(home, path[2:]) if path.startswith("~/") else path


def default_name(path: str) -> str:
    return path.rsplit("/", 1)[-1]
