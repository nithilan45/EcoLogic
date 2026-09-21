#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
echo "Reconstructing tables..."
python3 analysis.py

count_chars() {
python3 - <<'PY'
from pathlib import Path
import re
tex = Path("main.tex").read_text()
lines=[]
for line in tex.splitlines():
    if line.lstrip().startswith("%"):
        continue
    lines.append(re.sub(r"(^|[^\\])%.*", r"\1", line))
body="\n".join(lines)
print("main.tex chars excluding whitespace:", len(re.sub(r"\s+","", body)))
print("main.tex chars including whitespace:", len(body))
PY
}

if command -v tectonic >/dev/null 2>&1; then
  echo "Building PDF with tectonic..."
  tectonic -o . main.tex
elif command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode main.tex
  bibtex main
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
else
  echo "WARNING: no LaTeX engine found. Tables reconstructed; PDF not built." >&2
  count_chars
  exit 0
fi
count_chars
echo "pdf bytes: $(wc -c < main.pdf)"
