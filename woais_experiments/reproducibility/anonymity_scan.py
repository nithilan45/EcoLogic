"""Double-blind anonymity scanner. Never deletes files."""

from __future__ import annotations

import fnmatch
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from woais_experiments.paths import PACKAGE, ROOT
from woais_experiments.reproducibility.secret_scan import SKIP_DIR_NAMES

DEFAULT_CONFIG = PACKAGE / "reproducibility" / "anonymity_config.yaml"
MAX_BYTES = 8_000_000


def load_anonymity_config(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_CONFIG
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{p} must be a mapping")
    return raw


def _compile_rules(cfg: dict[str, Any]) -> list[tuple[str, re.Pattern[str]]]:
    rules: list[tuple[str, re.Pattern[str]]] = []
    ident = cfg.get("identities") or {}
    for kind, values in ident.items():
        for raw in values or []:
            text = str(raw).strip()
            if not text:
                continue
            rules.append((f"identity:{kind}", re.compile(re.escape(text), re.IGNORECASE)))
    for spec in cfg.get("path_patterns") or []:
        name = str(spec.get("name") or "pattern")
        flags = re.IGNORECASE if spec.get("ignore_case", True) else 0
        rules.append((name, re.compile(str(spec["regex"]), flags)))
    return rules


def _excluded(rel: str, cfg: dict[str, Any]) -> bool:
    rel_n = rel.replace("\\", "/")
    for glob in cfg.get("exclude_globs") or []:
        if fnmatch.fnmatch(rel_n, str(glob)):
            return True
    suffixes = {str(s).lower() for s in (cfg.get("exclude_suffixes") or [])}
    path = Path(rel_n)
    if path.suffix.lower() in suffixes:
        return True
    return False


def _decode(raw: bytes) -> str | None:
    if b"\x00" in raw[:8192]:
        # PDFs are binary; keep extractable ASCII runs.
        if raw[:5] == b"%PDF-":
            chunks = re.findall(rb"[\x20-\x7e]{8,}", raw[:MAX_BYTES])
            return "\n".join(c.decode("ascii", errors="ignore") for c in chunks[:4000])
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def scan_text(text: str, rules: list[tuple[str, re.Pattern[str]]], *, source: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for i, line in enumerate(text.splitlines(), 1):
        for name, pat in rules:
            if pat.search(line):
                hits.append({
                    "file": source,
                    "line": i,
                    "pattern": name,
                    "excerpt": "[redacted — open the file at this line]",
                })
                break
    return hits


def scan_git_metadata(root: Path, rules: list[tuple[str, re.Pattern[str]]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    commands = (
        ("git:log", ["git", "-C", str(root), "log", "-n", "50", "--format=%an <%ae> %s"]),
        ("git:config", ["git", "-C", str(root), "config", "--local", "--list"]),
    )
    for label, cmd in commands:
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        except OSError:
            continue
        if proc.returncode != 0:
            continue
        hits.extend(scan_text(proc.stdout or "", rules, source=label))
    return hits


def scan_tree(root: Path | None = None, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    base = Path(root) if root is not None else ROOT
    cfg = config if config is not None else load_anonymity_config()
    rules = _compile_rules(cfg)
    findings: list[dict[str, Any]] = []
    n_files = 0
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES or part == ".git" for part in path.parts):
            continue
        rel = _rel(path, base)
        if _excluded(rel, cfg):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > MAX_BYTES and path.suffix.lower() != ".pdf":
            continue
        n_files += 1
        path_hits = scan_text(rel, rules, source=f"path:{rel}")
        findings.extend(path_hits)
        try:
            raw = path.read_bytes()[:MAX_BYTES]
        except OSError:
            continue
        text = _decode(raw)
        if text is None:
            continue
        findings.extend(scan_text(text, rules, source=rel))
    if cfg.get("scan_git_metadata", True):
        findings.extend(scan_git_metadata(base, rules))
    files = sorted({h["file"] for h in findings})
    return {
        "ok": len(findings) == 0,
        "severity": str(cfg.get("severity") or "critical"),
        "n_files_scanned": n_files,
        "n_hits": len(findings),
        "n_files_requiring_review": len(files),
        "files_requiring_review": files,
        "findings": findings,
        "note": "No files were modified. Review each listed path before an anonymous release.",
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Anonymity report",
        "",
        "Double-blind scan. Nothing was deleted. Secret values and identity strings are not copied here.",
        "",
        f"- hits: {report['n_hits']}",
        f"- files requiring manual review: {report['n_files_requiring_review']}",
        f"- severity: {report.get('severity')}",
        "",
        "## Files requiring manual review",
        "",
    ]
    files = report.get("files_requiring_review") or []
    if not files:
        lines.append("None.")
    else:
        for f in files:
            lines.append(f"- `{f}`")
    lines.extend(["", "## Locations (file, line, pattern)", ""])
    if not report.get("findings"):
        lines.append("None.")
    else:
        lines.append("| file | line | pattern |")
        lines.append("|---|---:|---|")
        for hit in report["findings"]:
            lines.append(
                f"| `{hit['file']}` | {hit['line']} | `{hit['pattern']}` |"
            )
    lines.append("")
    return "\n".join(lines)
