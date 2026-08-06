#!/usr/bin/env bash
# Fetch COCO val2017 images and annotations.
#
# The dataset is public and is downloaded rather than redistributed in this
# repository. Pool membership is pinned by image id in configs/manifests/, so
# the exact archive contents are what the manifests are checked against.
#
# Usage: scripts/00_fetch_coco.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/coco"

IMAGES_URL="http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

fetch() {
  local url="$1" archive="$2" marker="$3"
  if [[ -e "${marker}" ]]; then
    echo "present, skipping: ${marker}"
    return
  fi
  if [[ ! -f "${archive}" ]]; then
    echo "downloading ${url}"
    curl -fL --retry 3 --retry-delay 5 -o "${archive}.partial" "${url}"
    mv "${archive}.partial" "${archive}"
  fi
  echo "extracting ${archive}"
  unzip -q -o "${archive}"
}

fetch "${IMAGES_URL}"      "val2017.zip"                  "val2017"
fetch "${ANNOTATIONS_URL}" "annotations_trainval2017.zip" "annotations/instances_val2017.json"

# The train2017 annotations ship in the same archive and are not used here.
rm -f annotations/instances_train2017.json \
      annotations/person_keypoints_train2017.json \
      annotations/captions_train2017.json

echo
echo "images:      $(find "${DATA_DIR}/val2017" -name '*.jpg' | wc -l | tr -d ' ')"
echo "annotations: ${DATA_DIR}/annotations/instances_val2017.json"
echo "sha256(annotations): $(shasum -a 256 "${DATA_DIR}/annotations/instances_val2017.json" | cut -d' ' -f1)"
