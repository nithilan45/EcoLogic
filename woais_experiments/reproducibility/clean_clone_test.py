"""Copy the repo into a fresh tree, install locked deps, and run reproduction checks.

Never copies virtualenvs, caches, or local secret files. Does not call paid APIs.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from woais_experiments.paths import ROOT
from woais_experiments.reproducibility.environment_capture import capture_environment

IGNORE_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "node_modules",
    ".cursor",
    "hf_cache",
    "_stress_unittest",
}
IGNORE_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.rc",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    ".DS_Store",
}
IGNORE_SUFFIXES = {".pyc", ".pyo", ".pyd", ".pem", ".p12"}
IGNORE_REL_PREFIXES = (
    "woais_experiments/results/runs/",
    "woais_experiments/data/xroutebench/hf_cache/",
)

LOCKFILE = "woais_experiments/requirements-lock.txt"


def _ignore(src: str, names: list[str]) -> set[str]:
    skipped: set[str] = set()
    src_path = Path(src)
    try:
        rel_dir = src_path.resolve().relative_to(ROOT.resolve()).as_posix() + "/"
    except ValueError:
        rel_dir = ""
    for name in names:
        path = src_path / name
        rel = (rel_dir + name).lstrip("/")
        if name in IGNORE_DIR_NAMES or name in IGNORE_FILE_NAMES:
            skipped.add(name)
            continue
        if path.suffix.lower() in IGNORE_SUFFIXES:
            skipped.add(name)
            continue
        if any(rel == p.rstrip("/") or rel.startswith(p) for p in IGNORE_REL_PREFIXES):
            skipped.add(name)
            continue
        if name.startswith(".env"):
            skipped.add(name)
    return skipped


def copy_clean_tree(dest: Path, *, src: Path | None = None) -> Path:
    source = Path(src) if src is not None else ROOT
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    shutil.copytree(source, dest, ignore=_ignore, symlinks=False)
    return dest


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def create_venv(root: Path, *, python: str | None = None) -> Path:
    venv = root / ".venv"
    exe = python or sys.executable
    subprocess.run([exe, "-m", "venv", str(venv)], check=True, cwd=str(root))
    return venv


def install_locked(venv: Path, root: Path) -> dict[str, Any]:
    py = _venv_python(venv)
    lock = root / LOCKFILE
    if not lock.exists():
        raise FileNotFoundError(f"missing lockfile {lock}")
    proc = subprocess.run(
        [str(py), "-m", "pip", "install", "--upgrade", "pip"],
        check=False,
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {"ok": False, "step": "pip_upgrade", "returncode": proc.returncode}
    proc = subprocess.run(
        [str(py), "-m", "pip", "install", "-r", str(lock)],
        check=False,
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    return {
        "ok": proc.returncode == 0,
        "step": "pip_install_lock",
        "returncode": proc.returncode,
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


SCRUB_ENV = (
    "OPENAI_API_KEY",
    "TOGETHER_API_KEY",
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
)


def _clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in SCRUB_ENV}
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run(py: Path, args: list[str], *, cwd: Path, timeout: float) -> dict[str, Any]:
    proc = subprocess.run(
        [str(py), *args],
        check=False,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_clean_env(),
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "cmd": [str(py), *args],
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
    }


def run_reproduction_checks(root: Path, *, python: Path, timeout_s: float) -> dict[str, Any]:
    """Unit tests, audit smoke, accounting/static reconstruction, public external smoke."""
    steps: dict[str, Any] = {}
    steps["unit_tests"] = _run(
        python,
        ["-m", "unittest", "discover", "-s", "woais_experiments/tests", "-t", ".", "-q"],
        cwd=root,
        timeout=min(timeout_s, 600.0),
    )
    steps["frozen_hashes"] = _run(
        python,
        [
            "-c",
            (
                "from woais_experiments.frozen import verify_frozen_hashes; "
                "import sys; r=verify_frozen_hashes(); "
                "sys.exit(0 if r.get('ok') else 1)"
            ),
        ],
        cwd=root,
        timeout=min(timeout_s, 120.0),
    )
    steps["smoke_audit"] = _run(
        python,
        ["run_woais.py", "audit", "--run-id", "artifact_smoke", "--no-symlink"],
        cwd=root,
        timeout=min(timeout_s, 120.0),
    )
    recon = _run(
        python,
        [
            "-c",
            (
                "from woais_experiments.reproducibility.reproduce_tables import reconstruct, smoke_external_public; "
                "import json,sys; "
                "r=reconstruct(); e=smoke_external_public(); "
                "print(json.dumps({'reconstruct': r['ok'], 'external': e, 'n_failed': r['n_failed']})); "
                "sys.exit(0 if r['ok'] and e.get('ok') else 1)"
            ),
        ],
        cwd=root,
        timeout=min(timeout_s, 180.0),
    )
    steps["accounting_static_external"] = recon
    ok = all(v.get("ok") for v in steps.values())
    return {"ok": ok, "steps": steps}


def run_clean_clone(
    *,
    src: Path | None = None,
    timeout_s: float = 1200.0,
    keep: bool = False,
) -> dict[str, Any]:
    tmp = Path(tempfile.mkdtemp(prefix="woais_clean_clone_"))
    dest = tmp / "repo"
    payload: dict[str, Any] = {
        "tmp": "<redacted-absolute>",
        "dest": "<redacted-absolute>",
        "ok": False,
    }
    try:
        copy_clean_tree(dest, src=src)
        payload["copied"] = True
        venv = create_venv(dest)
        inst = install_locked(venv, dest)
        payload["install"] = {"ok": inst["ok"], "returncode": inst.get("returncode")}
        if not inst["ok"]:
            payload["install_error"] = inst.get("stderr_tail")
            return payload
        py = _venv_python(venv)
        checks = run_reproduction_checks(dest, python=py, timeout_s=timeout_s)
        payload["checks"] = checks
        env_proc = subprocess.run(
            [
                str(py),
                "-c",
                "from woais_experiments.reproducibility.environment_capture import capture_environment; "
                "import json; print(json.dumps(capture_environment()))",
            ],
            check=False,
            cwd=str(dest),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if env_proc.returncode == 0 and env_proc.stdout:
            try:
                import json
                payload["clone_environment"] = json.loads(env_proc.stdout.splitlines()[-1])
            except json.JSONDecodeError:
                payload["clone_environment"] = None
        else:
            payload["clone_environment"] = capture_environment()
        payload["ok"] = bool(checks.get("ok"))
        return payload
    except subprocess.TimeoutExpired as exc:
        payload["ok"] = False
        payload["error"] = f"timeout: {exc}"
        return payload
    except Exception as exc:  # noqa: BLE001
        payload["ok"] = False
        payload["error"] = f"{type(exc).__name__}: {exc}"
        return payload
    finally:
        if not keep:
            def _chmod(func: Callable, p: str, _exc: BaseException) -> None:
                os.chmod(p, stat.S_IWRITE)
                func(p)
            shutil.rmtree(tmp, onerror=_chmod)
            payload["cleaned"] = True
        else:
            payload["cleaned"] = False
