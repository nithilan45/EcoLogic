"""RouteLLM GSM8K adapter: inventory, optional public download, local scoring.

What this repository already contains
-------------------------------------
- ``stage7_10/s9_static_baselines.json``: aggregate threshold sweep, n=1307.
  No query ids, no scores, no tokens, no assignments.
- ``stage7_10/external_generalization.json``: same reconstruction, aggregates.
- ``woais_experiments/results/external/routellm_tables.json``: re-tabulation of
  those aggregates.

What the public RouteLLM release contains (not in this repo)
------------------------------------------------------------
Documented at github.com/lm-sys/RouteLLM @ ``0b64fda`` (Ong et al., ICLR 2025):

- ``evals/gsm8k/gsm8k_responses.csv``: prompt, both models' correctness, both
  models' response text. **No token counts, no cost, no router scores.**
- ``evals/gsm8k/contaminated_prompts.jsonl``: decontamination list.
- HuggingFace ``routellm/bert_gpt4_augmented``: BERT router checkpoint.
  Scores are produced locally; this is not a paid inference API.

MMLU/MT-Bench in that release have no response text, so per-query cost cannot
be reconstructed. This adapter refuses those files rather than inventing tokens.

Token counts are derived from released response text with the models' tokenizers
when those libraries are installed. They are never guessed from character length.
"""

from __future__ import annotations

import csv
import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from woais_experiments.accounting.per_query_cost import load_price_table
from woais_experiments.external.adapter import ROUTELLM_CSV_CANDIDATES, find_routellm_csv
from woais_experiments.external_routing.panel import ExternalRouterPanel, from_arrays
from woais_experiments.frozen import load_json
from woais_experiments.paths import PACKAGE, ROOT, public_relpath, repo_rel

ROUTELLM_COMMIT = "0b64fda"
ROUTELLM_GITHUB = f"https://raw.githubusercontent.com/lm-sys/RouteLLM/{ROUTELLM_COMMIT}"
GSM8K_CSV_URL = f"{ROUTELLM_GITHUB}/routellm/evals/gsm8k/gsm8k_responses.csv"
CONTAM_URL = f"{ROUTELLM_GITHUB}/routellm/evals/gsm8k/contaminated_prompts.jsonl"
BERT_REPO = "routellm/bert_gpt4_augmented"

WEAK = "mistralai/Mixtral-8x7B-Instruct-v0.1"
STRONG = "gpt-4-1106-preview"

CACHE = PACKAGE / "data" / "routellm"
GSM8K_CSV = CACHE / "gsm8k_responses.csv"
CONTAM_JSONL = CACHE / "contaminated_prompts.jsonl"
SCORES_NPY = CACHE / "bert_win_rates_gsm8k.npy"
TOKENS_NPZ = CACHE / "tokens_gsm8k.npz"

# BERTRouter.calculate_strong_win_rate: softmax, then 1 - P(tie)-P(weak).
# Copied from stage7_10/external_check.py / RouteLLM routers.py. Not fitted here.
BERT_BATCH = 32


def _env_file(name: str) -> Path | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None


def scoring_dependencies() -> dict[str, Any]:
    out = {"torch": False, "transformers": False, "tiktoken": False, "sentencepiece": False}
    try:
        import torch  # noqa: F401

        out["torch"] = True
    except ImportError:
        pass
    try:
        import transformers  # noqa: F401

        out["transformers"] = True
    except ImportError:
        pass
    try:
        import tiktoken  # noqa: F401

        out["tiktoken"] = True
    except ImportError:
        pass
    try:
        import sentencepiece  # noqa: F401

        out["sentencepiece"] = True
    except ImportError:
        pass
    return out


def historical_stage9_pointer() -> dict[str, Any]:
    """Committed Stage 9 aggregates. Different grid; not this protocol's answer."""
    return {
        "not_this_protocol": True,
        "query_level": False,
        "threshold_grid": "pd.qcut on BERT scores (chosen after scoring; not the frozen linspace)",
        "paths": {
            "s9_static_baselines": "stage7_10/s9_static_baselines.json",
            "external_generalization": "stage7_10/external_generalization.json",
        },
        "note": (
            "Do not treat Stage 9 qcut aggregates as the query-level linspace sweep "
            "in woais_experiments/results/external_router/."
        ),
    }


def inventory() -> dict[str, Any]:
    """Describe committed vs downloadable RouteLLM assets. Never invents files."""
    s9 = ROOT / "stage7_10" / "s9_static_baselines.json"
    gen = ROOT / "stage7_10" / "external_generalization.json"
    tables = PACKAGE / "results" / "external" / "routellm_tables.json"
    csv_path = find_routellm_csv() or (GSM8K_CSV if GSM8K_CSV.is_file() else None)
    scores = _env_file("ROUTELLM_WIN_RATES") or (SCORES_NPY if SCORES_NPY.is_file() else None)
    tokens = _env_file("ROUTELLM_TOKENS") or (TOKENS_NPZ if TOKENS_NPZ.is_file() else None)
    s9_blob = load_json(s9) if s9.is_file() else {}
    return {
        "router": "RouteLLM BERT (routellm/bert_gpt4_augmented)",
        "router_kind": "EXTERNAL_LEARNED_ROUTER",
        "commit": ROUTELLM_COMMIT,
        "committed_aggregates": {
            "s9_static_baselines": {
                "path": repo_rel(s9) if s9.is_file() else None,
                "available": s9.is_file(),
                "n_items": s9_blob.get("n_items"),
                "query_level": False,
                "has_router_scores": False,
                "has_assignments": False,
                "has_tokens": False,
                "has_costs": "mean_cost_per_item_usd" in s9_blob,
            },
            "external_generalization": {
                "path": repo_rel(gen) if gen.is_file() else None,
                "available": gen.is_file(),
                "query_level": False,
            },
            "routellm_tables": {
                "path": repo_rel(tables) if tables.is_file() else None,
                "available": tables.is_file(),
                "query_level": False,
            },
        },
        "query_level_on_disk": {
            "gsm8k_responses_csv": public_relpath(csv_path) if csv_path else None,
            "bert_win_rates_npy": public_relpath(scores) if scores else None,
            "tokens_npz": public_relpath(tokens) if tokens else None,
        },
        "missing_unless_downloaded": [
            name
            for name, present in (
                ("query-level RouteLLM GSM8K responses", csv_path is not None),
                (
                    "BERT router scores (computed locally from the public checkpoint)",
                    scores is not None,
                ),
                (
                    "per-query token counts (reconstructed from released response text)",
                    tokens is not None,
                ),
            )
            if not present
        ],
        "not_available_anywhere": [
            "published numeric token counts",
            "published per-query USD",
            "MMLU/MT-Bench response text (cannot reconstruct cost)",
        ],
        "scoring_dependencies": scoring_dependencies(),
        "env_overrides": {
            "ROUTELLM_WIN_RATES": (
                public_relpath(_env_file("ROUTELLM_WIN_RATES"))
                if _env_file("ROUTELLM_WIN_RATES")
                else None
            ),
            "ROUTELLM_TOKENS": (
                public_relpath(_env_file("ROUTELLM_TOKENS"))
                if _env_file("ROUTELLM_TOKENS")
                else None
            ),
        },
        "csv_search_paths": [
            public_relpath(p) for p in ROUTELLM_CSV_CANDIDATES if p is not None
        ],
        "download_urls": {"gsm8k_responses.csv": GSM8K_CSV_URL, "contaminated_prompts.jsonl": CONTAM_URL},
        "paid_inference": False,
        "historical_stage9_aggregates": historical_stage9_pointer(),
    }


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "woais-external-routing/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            tmp.write_bytes(resp.read())
    except Exception:
        import subprocess

        subprocess.check_call(["curl", "-fsSL", "-o", str(tmp), url], timeout=120)
    tmp.replace(dest)
    return dest


def download_gsm8k(*, force: bool = False) -> dict[str, str]:
    """Fetch RouteLLM's documented GSM8K eval files. No undocumented endpoints."""
    written = {}
    if force or not GSM8K_CSV.is_file():
        _download(GSM8K_CSV_URL, GSM8K_CSV)
        written["gsm8k_responses.csv"] = repo_rel(GSM8K_CSV)
    else:
        written["gsm8k_responses.csv"] = repo_rel(GSM8K_CSV)
    if force or not CONTAM_JSONL.is_file():
        _download(CONTAM_URL, CONTAM_JSONL)
        written["contaminated_prompts.jsonl"] = repo_rel(CONTAM_JSONL)
    else:
        written["contaminated_prompts.jsonl"] = repo_rel(CONTAM_JSONL)
    return written


def _contam_prompts(path: Path) -> set[str]:
    prompts: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        p = row.get("eval_prompt") or row.get("prompt")
        if p:
            prompts.add(str(p))
    return prompts


def load_gsm8k_rows(csv_path: Path | None = None, contam_path: Path | None = None) -> list[dict[str, Any]]:
    csv_path = csv_path or find_routellm_csv() or (GSM8K_CSV if GSM8K_CSV.is_file() else None)
    if csv_path is None:
        raise FileNotFoundError(
            "RouteLLM GSM8K CSV is not on disk. Pass --download to fetch the "
            f"public file at {GSM8K_CSV_URL}"
        )
    csv_path = Path(csv_path)
    contam_path = contam_path or (CONTAM_JSONL if CONTAM_JSONL.is_file() else None)
    blocked = _contam_prompts(Path(contam_path)) if contam_path else set()
    rows: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for i, raw in enumerate(csv.DictReader(fh)):
            prompt = str(raw.get("prompt") or "")
            if prompt in blocked:
                continue
            if WEAK not in raw or STRONG not in raw:
                raise ValueError(f"{csv_path} is missing correctness columns {WEAK!r} / {STRONG!r}")
            rows.append({
                "query_id": f"gsm8k-{i:04d}",
                "prompt": prompt,
                "cheap_correct": _as_bool(raw[WEAK]),
                "strong_correct": _as_bool(raw[STRONG]),
                "cheap_response": str(raw.get(f"{WEAK}_response") or ""),
                "strong_response": str(raw.get(f"{STRONG}_response") or ""),
            })
    if not rows:
        raise ValueError("no GSM8K rows after decontamination")
    return rows


def _as_bool(v: Any) -> float:
    if isinstance(v, bool):
        return float(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes"}:
        return 1.0
    if s in {"0", "false", "no", ""}:
        return 0.0
    return float(bool(int(float(s))))


def bert_win_rates(prompts: Sequence[str], *, cache: Path | None = None, force: bool = False) -> np.ndarray:
    """Public RouteLLM BERT checkpoint, local forward pass. Not a paid API."""
    cache = cache or _env_file("ROUTELLM_WIN_RATES") or SCORES_NPY
    if cache.is_file() and not force and cache.stat().st_size > 0:
        arr = np.load(cache)
        if arr.shape[0] == len(prompts):
            return np.asarray(arr, dtype=float)
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for a local RouteLLM BERT forward pass (not a paid API)."
        ) from exc
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "RouteLLM scores are not cached and the `transformers` package is not installed. "
            "Install transformers+tiktoken for local BERT scoring, or place "
            f"{repo_rel(cache)} with one float per decontaminated prompt. "
            "This is not a paid inference call."
        ) from exc
    import torch

    model = AutoModelForSequenceClassification.from_pretrained(BERT_REPO, num_labels=3)
    tok = AutoTokenizer.from_pretrained(BERT_REPO)
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    out: list[float] = []
    with torch.no_grad():
        for i in range(0, len(prompts), BERT_BATCH):
            batch = list(prompts[i:i + BERT_BATCH])
            inp = tok(batch, return_tensors="pt", padding=True, truncation=True)
            inp = {k: v.to(device) for k, v in inp.items()}
            logits = model(**inp).logits.detach().cpu().numpy()
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            sm = e / e.sum(axis=1, keepdims=True)
            out.extend(1.0 - sm[:, -2:].sum(axis=1))
    arr = np.asarray(out, dtype=float)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


def reconstruct_tokens(rows: Sequence[Mapping[str, Any]], *, cache: Path | None = None, force: bool = False) -> dict[str, np.ndarray]:
    """Tokenize released prompts/responses. Refuses to guess if tokenizers are missing."""
    n = len(rows)
    cache = cache or _env_file("ROUTELLM_TOKENS") or TOKENS_NPZ
    if cache.is_file() and not force:
        blob = np.load(cache)
        if int(blob["n"][0]) == n:
            return {k: blob[k] for k in ("cheap_in", "cheap_out", "strong_in", "strong_out")}
    prompts = [str(r["prompt"]) for r in rows]
    cheap_resp = [str(r["cheap_response"]) for r in rows]
    strong_resp = [str(r["strong_response"]) for r in rows]
    if not any(cheap_resp) or not any(strong_resp):
        raise RuntimeError(
            "GSM8K CSV has no response text; token counts cannot be reconstructed "
            "and will not be invented."
        )
    try:
        import tiktoken
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "tiktoken/transformers are required to reconstruct RouteLLM token counts "
            "from released response text. Counts are not published as numeric fields."
        ) from exc
    enc = tiktoken.encoding_for_model("gpt-4-1106-preview")
    mx = AutoTokenizer.from_pretrained(WEAK)
    strong_in = np.array([len(enc.encode(p)) for p in prompts], dtype=float)
    strong_out = np.array([len(enc.encode(t)) for t in strong_resp], dtype=float)
    cheap_in = np.array([len(mx.encode(p)) for p in prompts], dtype=float)
    cheap_out = np.array([len(mx.encode(t)) for t in cheap_resp], dtype=float)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        n=np.array([n]),
        cheap_in=cheap_in,
        cheap_out=cheap_out,
        strong_in=strong_in,
        strong_out=strong_out,
    )
    return {"cheap_in": cheap_in, "cheap_out": cheap_out, "strong_in": strong_in, "strong_out": strong_out}


def load_routellm_panel(*, download: bool = False, force: bool = False) -> ExternalRouterPanel:
    if download:
        download_gsm8k(force=force)
    rows = load_gsm8k_rows()
    prompts = [r["prompt"] for r in rows]
    scores = bert_win_rates(prompts, force=force)
    tok = reconstruct_tokens(rows, force=force)
    notes = [
        f"RouteLLM GSM8K after their decontamination list, n={len(rows)}",
        "Scores: local forward pass of public BERT checkpoint routellm/bert_gpt4_augmented",
        "Tokens: reconstructed from released response text; not published numeric fields",
        "Assignments are score>=threshold; quality labels are not inputs to routing",
        "Oracle is computed separately and labeled ORACLE_ASSIGNMENT",
    ]
    return from_arrays(
        router_name="routellm_bert_gpt4_augmented",
        dataset="routellm_gsm8k",
        query_ids=[r["query_id"] for r in rows],
        scores=scores,
        cheap_model=WEAK,
        strong_model=STRONG,
        cheap_quality=[r["cheap_correct"] for r in rows],
        strong_quality=[r["strong_correct"] for r in rows],
        cheap_input_tokens=tok["cheap_in"],
        cheap_output_tokens=tok["cheap_out"],
        strong_input_tokens=tok["strong_in"],
        strong_output_tokens=tok["strong_out"],
        notes=notes,
        source=repo_rel(find_routellm_csv() or GSM8K_CSV),
        score_provenance=f"{BERT_REPO} local logits; 1 - P(tie) - P(weak)",
        prices=load_price_table(),
    )
