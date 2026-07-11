#!/usr/bin/env bash
# Fast LaTeX build: compile in local /tmp (SSD) to avoid slow ceph I/O,
# then copy the resulting PDF back into the repo's revise/ folder.
set -euo pipefail
SRC=/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD/latex
BUILD=/tmp/latexbuild
mkdir -p "$BUILD"
# sync sources (only the files LaTeX needs)
cp -r "$SRC/main.tex" "$SRC/main.bib" "$SRC/acl.sty" "$SRC/acl_natbib.bst" \
      "$SRC/chapters" "$SRC/tables" "$SRC/figures" "$BUILD/" 2>/dev/null
cd "$BUILD"
pdflatex -interaction=nonstopmode -draftmode main.tex >/tmp/build_p1.log 2>&1 || true
bibtex main >/tmp/build_bib.log 2>&1 || true
pdflatex -interaction=nonstopmode -draftmode main.tex >/tmp/build_p2.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/tmp/build_p3.log 2>&1
echo "pages: $(grep -oE 'Output written on main.pdf \([0-9]+ pages' /tmp/build_p3.log)"
echo "errors: $(grep -cE '^!' /tmp/build_p3.log)"
echo "undefined refs/cites: $(grep -c 'undefined' /tmp/build_p3.log)"
cp "$BUILD/main.pdf" "$SRC/revise/EasyOPD-revised-0710.pdf"
echo "PDF -> $SRC/revise/EasyOPD-revised-0710.pdf"
