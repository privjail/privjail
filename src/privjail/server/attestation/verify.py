"""verify.py -- verifier-side: verify a TDX ECDSA quote (v4).

Same steps as go-tdx-guest's verify.go verifyQuote() minus the parts that
need live Intel PCS collateral (no TCB-status / revocation checks):
  1. sha256(Header||TdQuoteBody) matches the quote signature under the
     embedded ECDSA attestation public key.
  2. The QE (enclave) report signature verifies under the PCK leaf
     certificate's public key.
  3. sha256(attestation_key||qe_auth_data) matches the QE report's
     REPORTDATA field (binds the ephemeral attestation key to the QE).
  4. The PCK certificate chain (leaf -> intermediate -> root) verifies up to
     the trusted root CA's key (root_pem_path as the trust anchor).
  5. If expected_reportdata is given, the TdQuoteBody's REPORTDATA field
     (the original 64-byte input, e.g. a verifier nonce) equals it. This is
     what turns "a genuine quote" into "a genuine, FRESH quote responding to
     my specific challenge".

verify_quote() raises TdxVerifyError on failure.
"""
import hashlib
import re
import struct

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils
from cryptography import x509

TDX_REPORTDATA_LEN = 64

# ---- Quote v4 ABI offsets (from go-tdx-guest/abi/abi.go) ------------------

HEADER_SIZE = 0x30
TD_QUOTE_BODY_SIZE = 0x248
QUOTE_BODY_START = HEADER_SIZE
QUOTE_BODY_END = QUOTE_BODY_START + TD_QUOTE_BODY_SIZE  # 0x278
SIGNED_DATA_SIZE_START = QUOTE_BODY_END  # 0x278
SIGNED_DATA_SIZE_LEN = 4
SIGNED_DATA_START = SIGNED_DATA_SIZE_START + SIGNED_DATA_SIZE_LEN  # 0x27C

TD_REPORT_DATA_START = 0x208
TD_REPORT_DATA_LEN = 0x40

SD_SIGNATURE_START = 0x00
SD_SIGNATURE_LEN = 0x40
SD_ATTESTATION_KEY_START = 0x40
SD_ATTESTATION_KEY_LEN = 0x40
SD_CERTIFICATION_DATA_START = 0x80

CD_TYPE_LEN = 2
CD_SIZE_LEN = 4
CD_HEADER_LEN = CD_TYPE_LEN + CD_SIZE_LEN
QE_REPORT_CERTIFICATION_DATA_TYPE = 6
PCK_CERT_CHAIN_DATA_TYPE = 5

QE_REPORT_LEN = 0x180
QE_REPORT_SIGNATURE_LEN = 0x40

ER_REPORT_DATA_START = 0x140
ER_REPORT_DATA_LEN = 0x40

_PEM_CERT_RE = re.compile(
    rb"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", re.DOTALL
)


class TdxVerifyError(Exception):
    pass


def _le16(buf: bytes, off: int) -> int:
    return struct.unpack_from("<H", buf, off)[0]


def _le32(buf: bytes, off: int) -> int:
    return struct.unpack_from("<I", buf, off)[0]


def _raw_pubkey_to_eckey(raw64: bytes) -> ec.EllipticCurvePublicKey:
    x = int.from_bytes(raw64[:32], "big")
    y = int.from_bytes(raw64[32:64], "big")
    return ec.EllipticCurvePublicNumbers(x, y, ec.SECP256R1()).public_key()


def _verify_raw_ecdsa_sha256(pubkey: ec.EllipticCurvePublicKey, msg: bytes, raw_sig64: bytes) -> bool:
    r = int.from_bytes(raw_sig64[:32], "big")
    s = int.from_bytes(raw_sig64[32:64], "big")
    der_sig = utils.encode_dss_signature(r, s)
    try:
        pubkey.verify(der_sig, msg, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


def _parse_pem_certs(pem_blob: bytes) -> list:
    return [x509.load_pem_x509_certificate(m.group(0)) for m in _PEM_CERT_RE.finditer(pem_blob)]


def _verify_cert_signature(cert: x509.Certificate, issuer_pubkey: ec.EllipticCurvePublicKey) -> bool:
    try:
        issuer_pubkey.verify(
            cert.signature,
            cert.tbs_certificate_bytes,
            ec.ECDSA(cert.signature_hash_algorithm),
        )
        return True
    except InvalidSignature:
        return False


def verify_quote(quote: bytes, root_pem_path: str, expected_reportdata: bytes = None) -> None:
    """Verifies a TDX ECDSA quote (v4). Raises TdxVerifyError on any failure."""
    n = len(quote)
    if n < SIGNED_DATA_START:
        raise TdxVerifyError(f"quote too small ({n} bytes)")

    header = quote[0:HEADER_SIZE]
    body = quote[QUOTE_BODY_START:QUOTE_BODY_END]
    signed_data_len = _le32(quote, SIGNED_DATA_SIZE_START)
    if SIGNED_DATA_START + signed_data_len > n:
        raise TdxVerifyError("signed data size exceeds quote length")
    signed_data = quote[SIGNED_DATA_START:SIGNED_DATA_START + signed_data_len]

    version = _le16(header, 0)
    if version != 4:
        raise TdxVerifyError(f"only QuoteV4 is supported (got version {version})")

    quote_signature = signed_data[SD_SIGNATURE_START:SD_SIGNATURE_START + SD_SIGNATURE_LEN]
    attestation_key = signed_data[SD_ATTESTATION_KEY_START:SD_ATTESTATION_KEY_START + SD_ATTESTATION_KEY_LEN]

    cert_data = signed_data[SD_CERTIFICATION_DATA_START:]
    cert_data_type = _le16(cert_data, 0)
    if cert_data_type != QE_REPORT_CERTIFICATION_DATA_TYPE:
        raise TdxVerifyError(f"unexpected certification data type {cert_data_type}")
    qe_cert_data = cert_data[CD_HEADER_LEN:]

    qe_report = qe_cert_data[0:QE_REPORT_LEN]
    qe_report_signature = qe_cert_data[QE_REPORT_LEN:QE_REPORT_LEN + QE_REPORT_SIGNATURE_LEN]
    qe_auth_data_hdr_off = QE_REPORT_LEN + QE_REPORT_SIGNATURE_LEN
    qe_auth_data_size = _le16(qe_cert_data, qe_auth_data_hdr_off)
    qe_auth_data_off = qe_auth_data_hdr_off + 2
    qe_auth_data = qe_cert_data[qe_auth_data_off:qe_auth_data_off + qe_auth_data_size]

    pck_chain_cd = qe_cert_data[qe_auth_data_off + qe_auth_data_size:]
    pck_cd_type = _le16(pck_chain_cd, 0)
    pck_cd_size = _le32(pck_chain_cd, 2)
    if pck_cd_type != PCK_CERT_CHAIN_DATA_TYPE:
        raise TdxVerifyError(f"unexpected certification data type {pck_cd_type} (expected PCK cert chain)")
    pck_chain_pem = pck_chain_cd[CD_HEADER_LEN:CD_HEADER_LEN + pck_cd_size]

    # Step: nonce binding (if the caller supplied an expected value).
    if expected_reportdata is not None:
        actual = body[TD_REPORT_DATA_START:TD_REPORT_DATA_START + TD_REPORT_DATA_LEN]
        if actual != expected_reportdata:
            raise TdxVerifyError("quote REPORTDATA does not match expected nonce (stale or mismatched quote)")

    # Step 1: header||body signature under attestation key.
    attest_key = _raw_pubkey_to_eckey(attestation_key)
    message = header + body
    if not _verify_raw_ecdsa_sha256(attest_key, message, quote_signature):
        raise TdxVerifyError("quote signature verification failed")

    # Parse PCK certificate chain (leaf || intermediate || root).
    certs = _parse_pem_certs(pck_chain_pem)
    if len(certs) < 3:
        raise TdxVerifyError("failed to parse PCK certificate chain")
    pck_leaf, pck_intermediate, pck_root = certs[0], certs[1], certs[2]

    # Step 2: QE report signature under PCK leaf public key.
    pck_leaf_pubkey = pck_leaf.public_key()
    if not isinstance(pck_leaf_pubkey, ec.EllipticCurvePublicKey):
        raise TdxVerifyError("PCK leaf public key is not an EC key")
    if not _verify_raw_ecdsa_sha256(pck_leaf_pubkey, qe_report, qe_report_signature):
        raise TdxVerifyError("QE report signature verification failed against PCK leaf certificate")

    # Step 3: sha256(attestation_key||qe_auth_data) == QE report REPORTDATA.
    digest = hashlib.sha256(attestation_key + qe_auth_data).digest()
    expected = digest.ljust(ER_REPORT_DATA_LEN, b"\x00")
    actual = qe_report[ER_REPORT_DATA_START:ER_REPORT_DATA_START + ER_REPORT_DATA_LEN]
    if expected != actual:
        raise TdxVerifyError("QE report REPORTDATA does not match sha256(attestation_key||qe_auth_data)")

    # Step 4: PCK cert chain up to the trusted root.
    with open(root_pem_path, "rb") as f:
        trusted_root = x509.load_pem_x509_certificate(f.read())

    if not _verify_cert_signature(pck_leaf, pck_intermediate.public_key()):
        raise TdxVerifyError("PCK certificate chain verification failed: leaf not signed by intermediate")
    if not _verify_cert_signature(pck_intermediate, pck_root.public_key()):
        raise TdxVerifyError("PCK certificate chain verification failed: intermediate not signed by root")
    if pck_root.public_key().public_numbers() != trusted_root.public_key().public_numbers():
        raise TdxVerifyError("PCK certificate chain verification failed: root does not match trusted root")
