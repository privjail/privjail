"""privjail_fs.attestation -- TDX remote attestation building blocks.

- net: exact-length socket I/O and a length-prefixed framing shared by the
  gateway and the key server.
- quote: attester-side -- fetch a TDX ECDSA quote via configfs-tsm.
- verify: verifier-side -- verify a TDX ECDSA quote (v4).
"""
