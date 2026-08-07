# AdAHuman

A deployment-aware evaluation of adversarial patch attacks against an
edge-oriented person detector.

The question is narrow and empirical: when a person-detection model is attacked
with an optimized patch, and when that same model is converted for a CPU
deployment path, what actually happens to detection performance, to runtime
cost, and to a cheap feature-space diagnostic that might flag the attack?

Nothing here claims to solve adversarial robustness. It measures one attack
against one model under stated conditions and reports the result, including
when the result is negative.

## Status

| Stage | State |
|-------|-------|
| Protocol frozen | done — `configs/protocol_v1.yaml`, pools frozen 2026-08-05 |
| Pools frozen | done — 4 disjoint pools, manifests digested |
| RQ1a clean baseline | done — `results/rq1_clean_baseline.json` |
| RQ1b patch attack | implemented and dry-run; GPU training pending |
| RQ3 monitor fit | done — `results/rq3_monitor_fit.json` |
| RQ3 monitor evaluation | implemented and dry-run; needs the trained patch |
| RQ4 ONNX export and CPU benchmark | done — `results/rq4_export.json`, `results/rq4_benchmark.json` |
| RQ5 reproducibility package | in progress |

Stages 05 and 07 have been exercised end-to-end with an untrained patch on a
24-image subset, so the post-training path is known to run. Those smoke outputs
were deleted; their run logs remain in `logs/` and are marked as smoke tests.

No result is claimed until it appears in `results/` with a dated run log in
`logs/`. See [`LIMITATIONS.md`](LIMITATIONS.md) for what these measurements do
not establish.

## Clean baseline

`ssdlite320_mobilenet_v3_large`, COCO_V1 weights, person class, score threshold
0.5, IoU 0.5:

| Pool | Images | People | Recall | Precision | AP@[.50:.95] | AP@.50 |
|------|--------|--------|--------|-----------|--------------|--------|
| reference | 500 | 1876 | 0.455 | 0.929 | 0.370 | 0.630 |
| attack_dev | 300 | 1203 | 0.449 | 0.933 | 0.367 | 0.638 |
| eval_untouched | 500 | 1722 | 0.472 | 0.932 | 0.386 | 0.648 |
| negative | 250 | 0 | — | — | — | — |

The negative pool produced 1 false person detection across 250 person-free
images (0.004 per image). Recall near 0.46 reflects the deliberately high 0.5
score threshold on a small 320x320 detector, not a broken baseline; the AP
figures are the threshold-independent view.

The three person-bearing pools agree to within their confidence intervals,
which is the intended check that the random split produced exchangeable pools.

## Deployment comparison (RQ4)

The full detector, including non-maximum suppression, exports to ONNX opset 17
at 320x320, batch size 1. Conversion fidelity is effectively exact: across 50
reference images, **zero** disagreements in person-detection count at the 0.5
operating threshold, maximum box deviation 7.6e-05 px, maximum score deviation
1.9e-06.

Latency on an Intel i5-8257U at 4 threads, PyTorch and onnxruntime interleaved
over 200 iterations each:

| | PyTorch | onnxruntime | ratio |
|---|---|---|---|
| median | 67.7 ms | 86.3 ms | 0.78x |
| p95 | 74.7 ms | 107.7 ms | 0.69x |
| p99 | 77.9 ms | 110.3 ms | 0.71x |
| min | 56.1 ms | 37.5 ms | 1.50x |
| throughput | 14.8 fps | 11.6 fps | 0.78x |
| load memory | 19.3 MiB | 29.2 MiB | |
| model size | 13.25 MB | 13.46 MB | |

**ONNX Runtime is slower here, not faster** — about 22% at the median, with a
noticeably wider spread. That runs against the common assumption that exporting
to ONNX buys throughput. It is reported as measured.

The two paths were timed *interleaved* rather than one after the other. A first
sequential run gave the same 0.78x median ratio but much dirtier tails
(PyTorch p99 109 ms versus 78 ms interleaved), because running one path to
completion before the other confounds the execution path with CPU thermal state
and background load.

This is a single machine at a single thread count. See
[`LIMITATIONS.md`](LIMITATIONS.md) — the relative ordering can invert elsewhere.

## Design

Three properties are enforced by code rather than by care:

**Pool entitlement.** Each pipeline stage declares which pools it may read.
`PoolDataset` refuses to construct against any other, so patch optimization
cannot see the held-out set even by mistake.

**Protocol freezing.** Environment-dependent protocol fields start as `PENDING`
and are filled by probe scripts. A stage that reads the held-out pool cannot
load the protocol while any field in the sections governing its result is still
`PENDING`.

**Provenance.** Every stage writes a JSON run record: timestamp, git commit and
dirty flag, library and hardware versions, pools read, and the sha256 of every
input and output. Results are traceable to the code that produced them.

These are not decoration. The held-out-pool check caught a real error during
construction: the RQ3 monitor's threshold was being calibrated on the same
images it was fit from, reporting a 5% false-positive rate while the true
held-out rate was 60%. See [`LIMITATIONS.md`](LIMITATIONS.md).

## Reproducing

```bash
brew install python@3.11
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash scripts/00_fetch_coco.sh                 # ~1.8 GB

.venv/bin/python scripts/01_probe_model.py --write    # resolve env-dependent fields
.venv/bin/python scripts/02_freeze_pools.py --write   # fix pool membership
.venv/bin/python scripts/03_clean_baseline.py         # RQ1a
```

Patch optimization needs a GPU and runs on Colab:

```bash
bash scripts/make_colab_bundle.sh    # then open notebooks/colab_train_patch.ipynb
```

Bring `artifacts/patch_v1.pt` back before running the evaluation stages. The
patch crosses that boundary as a plain tensor with a recorded digest, so the
training device does not enter any locally measured number.

## Layout

```
adahuman/
  config.py          protocol loading, stage entitlements, freeze enforcement
  data/pools.py      deterministic pool selection and digested manifests
  data/dataset.py    manifest-backed COCO loading
  models/detector.py the frozen detector and its feature tap
  attack/patch.py    EOT patch, differentiable placement, suppression loss
  eval/detection.py  COCO mAP and operating-point metrics
  utils/             hashing, seeding, run records
configs/             protocol and frozen pool manifests
scripts/             numbered pipeline stages
results/  logs/      measurements and their provenance
```

## Provenance and licence

All code is independently written for this project from public literature and
public documentation. No employer or client source code, data, thresholds,
learned parameters, or internal implementation details are used. Model weights
and images are public and downloaded at run time rather than redistributed.

Original code is released under the Apache License 2.0 — see
[`LICENSE`](LICENSE). That covers `adahuman/`, `scripts/`, `configs/`,
`notebooks/`, and the documentation, and nothing else.

Third-party libraries, the pretrained detector weights, and the COCO dataset
remain under their own terms and are not redistributed here; each is fetched at
run time from its own distributor. [`NOTICE`](NOTICE) lists them with their
licenses, and records why the attack code is published in full rather than
limited or withheld.
