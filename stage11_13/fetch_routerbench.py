#!/usr/bin/env python3
"""Fetch the RouterBench releases used by Stage 11 and verify their contents.

The two pickles are ~270 MB together, so they are gitignored rather than
committed. That makes the download a reproducibility step, and this script is
it: it fetches both releases into `external_data/`, hashes them, and compares
the hashes against the ones the analysis in the repository was actually run on.

A hash mismatch is not a warning to be clicked through. The Hub revision is not
pinnable from the dataset id alone, so a file that hashes differently is a
different dataset, and none of the numbers in `s11_routerbench_*.json`,
`SUMMARY.md` or the paper describe it.

    python3 stage11_13/fetch_routerbench.py            # download if absent, then verify
    python3 stage11_13/fetch_routerbench.py --verify    # hash what is on disk, download nothing
"""
import hashlib
import json
import os
import shutil
import sys

REPO = "withmartian/routerbench"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "external_data")
OUT = os.path.join(HERE, "s11_routerbench_sha256.json")

# The files as analysed. Recorded independently in the `sha256` field of each
# s11_routerbench_{shot}.json, which is what the paper's provenance table cites.
EXPECTED = {
    "routerbench_0shot.pkl":
        "ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d",
    "routerbench_5shot.pkl":
        "fbd7d3d16fba2759a18fa0ad44409d3e0e92ba80d03d90827eed1a6d084c9ffe",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    verify_only = "--verify" in sys.argv
    os.makedirs(DATA, exist_ok=True)
    got, ok = {}, True

    for name, expect in EXPECTED.items():
        dest = os.path.join(DATA, name)
        if not os.path.exists(dest):
            if verify_only:
                print(f"{name}: ABSENT (run without --verify to download)")
                ok = False
                continue
            from huggingface_hub import hf_hub_download
            print(f"downloading {name} ...", flush=True)
            cached = hf_hub_download(REPO, name, repo_type="dataset")
            # Copy out of the Hub cache so the analysis reads a stable path that
            # a cache eviction cannot pull out from under it.
            shutil.copyfile(cached, dest)

        digest = sha256(dest)
        got[name] = digest
        match = digest == expect
        ok &= match
        print(f"{name}: {digest} {'OK' if match else 'MISMATCH, expected ' + expect}")

    json.dump({"repo": REPO, "expected": EXPECTED, "observed": got,
               "all_match": ok}, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    if not ok:
        raise SystemExit("hashes do not match the analysed files; see the docstring")
    print("both releases match the files the Stage 11 results were computed on")


if __name__ == "__main__":
    main()
