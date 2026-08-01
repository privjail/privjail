"""quote.py -- attester-side: fetch a TDX ECDSA quote via configfs-tsm.

Requires root (creating the configfs entry needs CAP_SYS_ADMIN-ish privilege
on most kernels) and must run on the TDX guest (needs
/sys/kernel/config/tsm/report).

Also runnable as a CLI (`python -m privjail_fs.attestation.quote`, or the
installed `privjail-fs-get-quote` script): reads a 64-byte reportdata from
stdin, writes the quote to stdout. This lets an unprivileged caller obtain a
quote via `sudo privjail-fs-get-quote < nonce > quote` without needing root
for anything but this one step.
"""
import os
import sys

TSM_REPORT_DIR = "/sys/kernel/config/tsm/report"
REPORTDATA_LEN = 64
MAX_QUOTE_LEN = 64 * 1024


def get_quote(reportdata: bytes) -> bytes:
    """Fetches a TDX ECDSA quote over the given 64-byte report data
    (typically a verifier-supplied nonce).

    Returns the raw quote bytes. Raises OSError/ValueError on failure.
    """
    if len(reportdata) != REPORTDATA_LEN:
        raise ValueError(f"reportdata must be {REPORTDATA_LEN} bytes, got {len(reportdata)}")

    entry_path = os.path.join(TSM_REPORT_DIR, f"entry.{os.getpid()}")
    os.mkdir(entry_path, 0o755)
    try:
        inblob_path = os.path.join(entry_path, "inblob")
        outblob_path = os.path.join(entry_path, "outblob")
        provider_path = os.path.join(entry_path, "provider")

        with open(inblob_path, "wb") as f:
            f.write(reportdata)

        with open(provider_path, "r") as f:
            provider = f.read()
        if not provider.startswith("tdx_guest"):
            raise ValueError(f"unexpected provider (expected tdx_guest, got {provider!r})")

        with open(outblob_path, "rb") as f:
            quote = f.read(MAX_QUOTE_LEN)
        if not quote:
            raise ValueError("outblob was empty")
        return quote
    finally:
        os.rmdir(entry_path)


def main() -> None:
    reportdata = sys.stdin.buffer.read(REPORTDATA_LEN)
    sys.stdout.buffer.write(get_quote(reportdata))


if __name__ == "__main__":
    main()
