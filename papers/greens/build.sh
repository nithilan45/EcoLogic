#!/usr/bin/env bash
# Build the GREENS 2027 IEEE workshop paper.
# Requires: pdflatex + bibtex, or tectonic.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f artifacts/fig_energy_accuracy.pdf ]]; then
  echo "Figures missing; running analysis.py first."
  python3 analysis.py
fi

run_pdflatex() {
  local engine="$1"
  "$engine" -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  "$engine" -interaction=nonstopmode -halt-on-error main.tex
  "$engine" -interaction=nonstopmode -halt-on-error main.tex
}

if command -v tectonic >/dev/null 2>&1; then
  tectonic --keep-logs --keep-intermediates main.tex
elif command -v pdflatex >/dev/null 2>&1; then
  run_pdflatex pdflatex
elif [[ -x /Library/TeX/texbin/pdflatex ]]; then
  export PATH="/Library/TeX/texbin:$PATH"
  run_pdflatex pdflatex
else
  echo "ERROR: no tectonic or pdflatex in PATH." >&2
  echo "Install MacTeX/BasicTeX or: brew install tectonic" >&2
  exit 1
fi

# Page count from PDF trailer if pdfinfo/python available
if command -v pdfinfo >/dev/null 2>&1; then
  pdfinfo main.pdf | grep Pages
elif python3 -c "import pathlib; print(pathlib.Path('main.pdf').stat().st_size)" >/dev/null 2>&1; then
  python3 - <<'PY'
from pathlib import Path
p = Path("main.pdf").read_bytes()
# Count /Type /Page but not /Pages
n = p.count(b"/Type /Page") - p.count(b"/Type /Pages")
print(f"Pages (heuristic): {n}")
PY
fi

echo "Built $(pwd)/main.pdf"
