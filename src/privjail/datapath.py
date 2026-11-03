"""datapath.py -- maps the paths a client asks for to files on the server.

By default paths are used as given (local mode, or a dataserver someone
started by hand). A dataserver launched by the gateway instead gets a table
of names, e.g. {"mydata": "/home/u/.privjail/mnt/1234/0"}: the client
then says "mydata/train.csv", and only paths below one of those names
can be opened -- absolute paths, "..", and unknown first components are
refused. The real mount points are never shown to the client.
"""
from __future__ import annotations

import os
from pathlib import PurePosixPath

_roots: dict[str, str] | None = None


def set_roots(roots: dict[str, str]) -> None:
    global _roots
    _roots = dict(roots)


def resolve(path: str) -> str:
    if _roots is None:
        return path
    p = PurePosixPath(path)
    names = ", ".join(sorted(_roots)) or "(none)"
    if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] not in _roots:
        raise PermissionError(f"{path!r} is not under an available directory; available: {names}")
    return os.path.join(_roots[p.parts[0]], *p.parts[1:])
