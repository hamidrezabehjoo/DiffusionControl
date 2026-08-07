#!/bin/bash
# Rebuild pnas.pdf and si.pdf (tectonic, cross-reference cycle).
set -e
B=/tmp/pnas_build
rm -rf $B && mkdir -p $B
cp /mnt/agents/output/pnas.tex /mnt/agents/output/si.tex $B/
cp -r /mnt/agents/output/figs $B/
cd $B
for i in 1 2; do
  tectonic --keep-intermediates si.tex >/dev/null
  tectonic --keep-intermediates pnas.tex >/dev/null
done
tectonic --keep-intermediates si.tex >/dev/null
cp pnas.pdf si.pdf /mnt/agents/output/
echo "pnas: $(pdfinfo pnas.pdf | grep Pages)"
echo "si:   $(pdfinfo si.pdf | grep Pages)"
echo "?? in pnas: $(pdftotext pnas.pdf - | grep -c '??' || true)"
echo "?? in si:   $(pdftotext si.pdf - | grep -c '??' || true)"
