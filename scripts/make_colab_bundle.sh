#!/usr/bin/env bash
# Bundle the source needed to train the patch on Colab.
#
# Code, protocol, and the frozen manifests only -- no dataset, no weights, no
# results. Colab re-fetches the public inputs itself, so what crosses the
# boundary is exactly the code under review plus the pool definitions it must
# honour.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${REPO_ROOT}/adahuman_colab_src.tar.gz"

cd "${REPO_ROOT}"
tar --exclude='__pycache__' -czf "${OUT}" \
  adahuman configs scripts requirements.txt PROTOCOL.md

echo "bundle: ${OUT}"
echo "size:   $(du -h "${OUT}" | cut -f1)"
echo "sha256: $(shasum -a 256 "${OUT}" | cut -d' ' -f1)"
echo
echo "Copy this file to Google Drive (top level of My Drive, or an AdAHuman/"
echo "folder there), or to /content in the Colab session. The notebook locates"
echo "it on the filesystem -- there is no upload widget to click."
