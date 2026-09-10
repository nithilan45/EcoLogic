"""Secret scanner. Reports file, line, and type — never the secret value."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from woais_experiments.paths import ROOT

# Patterns must not be logged with their matches. Group 0 is the secret.
RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_access_key_id", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("aws_secret_access_key", re.compile(r"(?i)aws_secret_access_key\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{30,}")),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("openai_api_key", re.compile(r"sk-[A-Za-z0-9_\-]{20,}")),
    ("anthropic_api_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("github_pat", re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("github_fine_grained_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("slack_token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----")),
    ("authorization_header", re.compile(r"(?i)authorization\s*[:=]\s*(bearer|basic)\s+\S{8,}")),
    ("cookie_header", re.compile(r"(?i)(?:set-)?cookie\s*[:=]\s*\S{16,}")),
    ("password_assignment", re.compile(r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"]?[^\s'\"]{8,}")),
    ("dotenv_secret", re.compile(
        r"(?i)^(export\s+)?([A-Z][A-Z0-9_]*_)?(API_KEY|SECRET_ACCESS_KEY|SECRET_KEY|ACCESS_KEY_ID|PASSWORD|TOKEN|SECRET)\s*="
    )),
    ("url_with_embedded_credentials", re.compile(r"(?i)https?://[^/\s:]+:[^/\s]+@")),
    ("google_client_secret", re.compile(r'(?i)"client_secret"\s*:\s*"[^"]{12,}"')),
)

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    "hf_cache",
}
SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico",
    ".pkl", ".npz", ".npy", ".so", ".dylib", ".woff", ".woff2",
    ".pyc", ".pyo", ".zip", ".gz",
}
MAX_BYTES = 20_000_000


def _is_binary(sample: bytes) -> bool:
    return b"\x00" in sample[:8192]


def iter_text_files(root: Path, *, extra_skip: Iterable[str] = ()) -> Iterable[Path]:
    skip = set(extra_skip) | {"woais_experiments/reproducibility/secret_scan.py"}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        rel = str(path)
        if any(tok in rel.replace("\\", "/") for tok in skip):
            continue
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
        except OSError:
            continue
        yield path


PLACEHOLDER_VALUE = re.compile(
    r"(?i)^(your[-_]?|changeme|xxx+|todo|insert|<|\$\{|none|null|placeholder|example|dummy|fake)"
)
CODE_VALUE = re.compile(
    r"(?i)(os\.environ|os\.getenv|getenv\(|environ\[|[(){}\[\]])"
)
LIVE_ENV_NAMES = {".env", ".env.local", ".env.rc", ".env.production", ".env.development"}
SSH_KEY_NAMES = {"id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"}


def _assigned_value(line: str) -> str | None:
    if "=" not in line:
        return None
    left, right = line.split("=", 1)
    if not left.strip():
        return None
    return right.split("#", 1)[0].strip().strip("'\"")


def _looks_live(value: str | None, *, min_len: int = 16) -> bool:
    if not value:
        return False
    if PLACEHOLDER_VALUE.search(value):
        return False
    if CODE_VALUE.search(value):
        return False
    if any(ch.isspace() for ch in value):
        return False
    return len(value) >= min_len


def _decode_for_scan(raw: bytes) -> str | None:
    if raw[:5] == b"%PDF-":
        chunks = re.findall(rb"[\x20-\x7e]{8,}", raw[:MAX_BYTES])
        return "\n".join(c.decode("ascii", errors="ignore") for c in chunks[:4000])
    if _is_binary(raw):
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def scan_file(path: Path) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    name = path.name
    if name in SSH_KEY_NAMES:
        hits.append({"file": _rel(path), "line": 1, "secret_type": "ssh_or_pem_key_file"})
        return hits
    if name in {"credentials.json", "service-account.json"} or name.endswith("-credentials.json"):
        hits.append({"file": _rel(path), "line": 1, "secret_type": "google_credentials_file"})
        return hits
    try:
        raw = path.read_bytes()
    except OSError:
        return hits
    text = _decode_for_scan(raw)
    if text is None:
        return hits
    if name in LIVE_ENV_NAMES:
        for i, line in enumerate(text.splitlines(), 1):
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                val = _assigned_value(line)
                if _looks_live(val):
                    hits.append({"file": _rel(path), "line": i, "secret_type": "dotenv_file"})
        return hits
    min_len = {"password_assignment": 8, "authorization_header": 8}
    for i, line in enumerate(text.splitlines(), 1):
        for stype, pat in RULES:
            if not pat.search(line):
                continue
            if stype in {"dotenv_secret", "password_assignment", "authorization_header", "cookie_header"}:
                val = _assigned_value(line)
                if val is None:
                    parts = line.split(None, 2)
                    val = parts[-1] if parts else ""
                if not _looks_live(val, min_len=min_len.get(stype, 16)):
                    continue
            hits.append({"file": _rel(path), "line": i, "secret_type": stype})
            break
    return hits


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def scan_tree(root: Path | None = None) -> dict[str, Any]:
    base = Path(root) if root is not None else ROOT
    findings: list[dict[str, Any]] = []
    n_files = 0
    for path in iter_text_files(base):
        n_files += 1
        findings.extend(scan_file(path))
    return {
        "ok": len(findings) == 0,
        "n_files_scanned": n_files,
        "n_hits": len(findings),
        "findings": findings,
        "note": "Secret values are omitted. Review the listed file:line locations.",
    }
