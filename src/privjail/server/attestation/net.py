"""net.py -- exact-length socket I/O and length-prefixed framing.

A 4-byte little-endian length prefix followed by the payload; shared wire
format between the gateway (attester) and the key server (verifier).
"""
import socket
import struct


def read_full(sock: socket.socket, length: int) -> bytes:
    chunks = []
    got = 0
    while got < length:
        chunk = sock.recv(length - got)
        if not chunk:
            raise ConnectionError("connection closed early")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def write_full(sock: socket.socket, data: bytes) -> None:
    sock.sendall(data)


def send_frame(sock: socket.socket, payload: bytes) -> None:
    write_full(sock, struct.pack("<I", len(payload)))
    if payload:
        write_full(sock, payload)


def recv_frame(sock: socket.socket, max_len: int) -> bytes:
    (length,) = struct.unpack("<I", read_full(sock, 4))
    if length > max_len:
        raise ValueError(f"frame too large ({length} bytes)")
    if length == 0:
        return b""
    return read_full(sock, length)
