#!/usr/bin/env bash
# Build AIPerf 2027 Paper C. Run from papers/aiperf/ or via this script's directory.
set -euo pipefail
cd "$(dirname "$0")"
python3 analysis.py
if command -v tectonic >/dev/null 2>&1; then
  tectonic -o . main.tex
elif command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode main.tex
  bibtex main
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
else
  echo "No LaTeX engine (tectonic or pdflatex) on PATH" >&2
  exit 1
fi
echo "Built $(pwd)/main.pdf"
