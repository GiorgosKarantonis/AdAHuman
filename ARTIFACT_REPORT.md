# Artifact Report

**AdAHuman: Deployment-Aware Adversarial Resilience Evaluation for
Edge-Oriented Computer Vision Systems**

| | |
|---|---|
| Version | 1.0 |
| Date | 2026-08-07 |
| Repository | https://github.com/GiorgosKarantonis/AdAHuman |
| Tag | `v1.0` |
| Archive | `adahuman-1.0.tar.gz`, produced by `git archive` from the tag |
| Licence | Apache 2.0 for original code; see `NOTICE` for scope |

Cite this artifact by its **tag**. A commit hash cannot appear in a document the
commit contains, and the tag resolves to exactly one commit, which every run
record in `logs/` names independently. The archive digest is published with the
release; `git archive` is deterministic, so anyone can regenerate a
byte-identical archive from the tag and check it.

This report describes what was built, what was measured, what the measurements
support, and what they do not. Every figure quoted here is reproduced from a
named file in `results/`, and every result file has a dated run record in
`logs/` naming the commit that produced it.

---

## 1. Summary

| | Question | Finding |
|---|---|---|
| **RQ1** | Does an optimized patch degrade detection beyond simple occlusion? | **Yes.** 41.6% of held-out targets suppressed, against 9.9% for a random patch of identical size and placement. |
| RQ2 | Does an established resilience control retain robustness after deployment optimization? | **Deferred.** Not implemented, not run, not claimed. |
| **RQ3** | Can a feature-distance score flag adversarial inputs at runtime without excessive false alarms? | **No.** AUROC 0.586 against ordinarily-corrupted inputs, 0.498 against motion blur. Not usable as a defence, alone or in combination. |
| **RQ4** | What do accuracy, latency and memory cost on a deployment path? | **Conversion is exact, execution is not free.** Zero detection-count disagreements; 18–22% slower at the median under onnxruntime. |
| **RQ5** | Can an independent party reproduce this? | **Yes.** Pool manifests, clean baseline and fitted monitor reproduce byte- or bit-identically from the seed and protocol. |

An adversarial patch trained to suppress person detections removes **41.6%**
[38.1–45.2] of the targets an edge-oriented detector found on held-out data. A
random patch of identical size and placement removes 9.9% [8.0–12.3] by
occlusion alone, so most of the effect is adversarial rather than physical
obstruction.

A runtime feature-distance monitor intended to flag such inputs **does not
work**. It separates adversarial from ordinarily-corrupted inputs at AUROC
0.586, and against motion blur specifically at 0.498 — chance. This is reported
as the result.

Exporting the same detector to ONNX preserves its outputs almost exactly — zero
disagreements in detection count, maximum score deviation 1.9e-06 — while
running 18–22% slower at the median on the measured CPU.

The three findings are informative together. Each would have been reported more
favourably by an evaluation that stopped at the model boundary: the attack
without an occlusion control, the monitor without a benign-shift comparison, the
conversion without a timing measurement. The gap between what is measured during
development and what holds in deployment is the object of study.

## 2. Scope

**What this is.** A bounded, reproducible study of one attack against one model,
carried through runtime monitoring and deployment-format conversion, with the
evaluation protocol frozen before the results were produced.

**What this is not.** Not a framework, not a general robustness benchmark, and
not a defence. One model family, one attack objective, digital rather than
physical patches, a non-adaptive adversary, one dataset, one CPU. Section 7 and
`LIMITATIONS.md` state the boundaries in full.

## 3. Method

### 3.1 Target system

`ssdlite320_mobilenet_v3_large` from TorchVision 0.17.2 with `COCO_V1`
pretrained weights, checkpoint sha256 `a79551df90c79834…`, evaluated on person
detection at 320×320. Operating point: score threshold 0.5, IoU 0.5. Mean
average precision is reported separately over the standard COCO IoU sweep.

Weights and images are public and fetched at run time. Nothing proprietary is
used; all code is written for this project from public literature.

### 3.2 Pool discipline

Four disjoint pools are fixed from COCO val2017 by seed `20260805`, before any
measurement, and recorded as manifests with content digests.

| Pool | n | Purpose | Manifest digest |
|---|---|---|---|
| `reference` | 500 | Monitor statistics; conversion-fidelity probe | `fec2c62d1daee321…` |
| `attack_dev` | 300 | Patch optimization; all tuning | `34922d023f02dcb6…` |
| `eval_untouched` | 500 | Reported results, measured once | `e99c5f68a1da8bf0…` |
| `negative` | 250 | False positives on person-free images | `52f6d9362ba5b11b…` |

Each pipeline stage declares which pools it may read, and `PoolDataset` refuses
to construct against any other. Patch optimization cannot reach the held-out
pool, and the development-evaluation stage added later is entitled to
`attack_dev` alone, so it cannot become a second held-out look.

Environment-dependent protocol fields begin as `PENDING`. A stage that reads the
held-out pool cannot load the protocol while any field governing its result is
unfrozen. The freeze is a property of the code path, not of the author's memory.

### 3.3 Attack

A person-suppression patch after Thys, Van Ranst and Goedemé (CVPRW 2019),
independently reimplemented against SSDLite person-class scores. Patch area is
25% of the target bounding box, applied under expectation-over-transformation
across scale, rotation, brightness, contrast, perspective and sensor noise, with
a total-variation penalty so the result is not pure per-pixel noise.

Optimization ran on a Colab T4 for 109 epochs of a 300-epoch cap and stopped on
a plateau: no improvement above 0.003 in the 5-epoch trailing mean for 30
epochs. The patch saved is the best checkpoint (epoch 103, 3848 steps), not the
final one — the final epoch scored worse than the checkpoint.

### 3.4 Runtime monitor

A Mahalanobis-style distance from an input's pooled backbone features to clean
in-distribution statistics, after Lee et al. (NeurIPS 2018), independently
implemented. 480-dimensional pooled features are projected to 64 PCA components
retaining ~97% of variance, then a Ledoit-Wolf shrunk covariance is estimated.

The reference pool is split: covariance is estimated on 350 images and the
operating threshold calibrated on a held-out 150 at a 5% false-positive target.
Neither split is ever the evaluation pool.

### 3.5 Deployment path

The full detector including non-maximum suppression is exported to ONNX opset
17 at 320×320, batch size 1, and executed under onnxruntime 1.23.2 on an Intel
i5-8257U at 4 intra-op threads. Both paths receive identical pre-resized input.
Latency is measured interleaved rather than in sequential blocks, so CPU thermal
state and background load fall on both paths equally.

## 4. Results

### 4.1 Clean baseline — `results/rq1_clean_baseline.json`

| Pool | n | People | Recall | Precision | AP@[.50:.95] | AP@.50 |
|---|---|---|---|---|---|---|
| `reference` | 500 | 1876 | 0.455 | 0.929 | 0.370 | 0.630 |
| `attack_dev` | 300 | 1203 | 0.449 | 0.933 | 0.367 | 0.638 |
| `eval_untouched` | 500 | 1722 | 0.472 | 0.932 | 0.386 | 0.648 |
| `negative` | 250 | 0 | — | — | — | — |

One false person detection across 250 person-free images (0.004 per image).
Recall near 0.46 reflects the deliberately high 0.5 score threshold on a small
320×320 detector; the AP figures give the threshold-independent view. The three
person-bearing pools agree within their confidence intervals, which is the
intended check that the split produced exchangeable pools.

### 4.2 Attack — `results/rq1_attack.json`, `results/rq1_attack_control.json`

Held-out pool, 500 images, 917 patched targets. Each frame is scored twice,
clean and patched, so the effect is a paired difference on identical images.

| | recall, patched targets | recall, unpatched targets | suppression rate |
|---|---|---|---|
| clean | 0.8015 | 0.0969 | — |
| **trained patch** | 0.4766 | 0.0919 | **0.4163** [0.3812–0.4523] |
| random patch (control) | 0.7546 | 0.1006 | 0.0993 [0.0797–0.1231] |

AP@.50 falls 0.648 → 0.432; AP@[.50:.95] falls 0.386 → 0.222.

*Suppression rate* is the paired quantity: of targets detected when clean, the
fraction lost under attack (306 of 735). It cannot be inflated by targets the
detector was already missing.

Three controls bound the interpretation:

1. **Occlusion floor.** An unoptimized patch of identical size and placement
   suppresses 9.9%. Quoting 41.6% without this would overstate the adversarial
   contribution by roughly a quarter.
2. **Unpatched targets.** Recall on targets in the same frames that received no
   patch moved −0.005. The patch suppresses what it covers rather than degrading
   the detector globally.
3. **False positives.** On the person-free pool, 0.004 per image under both
   conditions. The attack does not induce spurious detections.

Development-pool suppression was 0.5054 [0.4600–0.5507] against 0.4163 held out.
The patch was optimized on the development pool, so that ~9-point gap is
ordinary overfitting to the images it trained against. The held-out figure is
the one that should be quoted.

### 4.3 Runtime monitor — `results/rq3_monitor_eval.json` — negative

| Comparison | AUROC | Avg. precision | TPR at threshold | FPR at threshold |
|---|---|---|---|---|
| clean vs adversarial | 0.6388 | 0.6120 | 0.108 | 0.052 |
| **ordinary shift vs adversarial** | **0.5863** | 0.5692 | 0.108 | 0.074 |
| clean vs ordinary shift | 0.5622 | 0.5432 | 0.074 | 0.052 |

Per corruption, against adversarial inputs:

| Corruption | AUROC |
|---|---|
| gaussian noise | 0.6287 |
| JPEG compression | 0.6035 |
| brightness | 0.6013 |
| fog | 0.5998 |
| **motion blur** | **0.4980** |

At the frozen threshold the monitor flags 10.8% of attacked inputs while raising
false alarms on 5.2% of clean ones. That is not a usable trade at any deployment
scale.

The second row is the one that matters. A monitor separating adversarial from
clean inputs but not from fog, blur and codec artefacts is a novelty detector,
not an attack detector; deployed, it would alarm on weather and be switched off.
Against motion blur it does not even do that.

The hypothesis — that adversarially patched inputs sit far enough from the clean
feature distribution to be flagged at runtime — is **not supported** for this
model, this feature representation and this attack. The verdict is produced by a
function written and committed before any of these numbers existed, so the
conclusion follows thresholds fixed in advance.

Two constraints bound how far this generalizes. The monitor is image-level: a
patch covering a quarter of one bounding box is a small perturbation of a
globally pooled feature vector, and per-proposal monitoring is untested. And the
attacker is non-adaptive, having never optimized against the monitor — an
adaptive attacker would only make this worse.

### 4.4 Deployment path — `results/rq4_export.json`, `results/rq4_benchmark.json`

**Conversion fidelity**, 50 reference images at the 0.5 operating threshold:

| | |
|---|---|
| Detection-count disagreements | **0** |
| Max absolute box deviation | 9.2e-05 px |
| Max absolute score deviation | 1.9e-06 |

**Latency**, interleaved, 200 iterations each:

| | PyTorch | onnxruntime | ratio |
|---|---|---|---|
| median | 72.0 ms | 87.3 ms | 0.82× |
| p95 | 78.3 ms | 108.6 ms | 0.72× |
| p99 | 87.1 ms | 111.0 ms | 0.78× |
| throughput | 13.9 fps | 11.5 fps | 0.82× |
| load memory | 19.4 MiB | 30.2 MiB | |
| model size | 13.25 MB | 13.46 MB | |

ONNX Runtime is **slower**, not faster, which runs against the common assumption
that export buys throughput. Three independent runs gave median ratios of 0.78×,
0.80× and 0.82× — consistently the same direction, but the point estimate is not
stable to better than a few percent, and p99 ratios ranged 0.66× to 0.78×. The
honest claim is the ordering plus a range of 18–22%, not any single figure. The
machine is a passively cooled laptop shared with other work.

Numerically the converted model is equivalent; operationally it is not. Only one
of those questions is answered by comparing outputs.

## 5. Corrections made during construction

Recorded because a reviewer should be able to see that the controls caught real
errors rather than merely being described.

**Monitor threshold calibrated in-sample** (found and fixed 2026-08-05, before
any held-out measurement). The first monitor estimated a 480-dimensional
covariance from 500 reference images and selected its threshold on those same
images. It reported the target 5% false-positive rate — which an in-sample
quantile does by construction — while the true rate on held-out clean images was
**60%**. Calibration now uses a reference split held out from the covariance
fit. Diagnostically the out-of-sample calibration was the fix; the PCA
projection mattered less than expected. No held-out measurement was ever
produced under the original version. The superseded fit and its run log remain
in `logs/`.

**`.gitignore` silently excluded a source package.** An unanchored `data/`
pattern, intended for the downloaded dataset, matched `adahuman/data/` at any
depth and kept four source files out of eleven commits. The repository ran
locally, where the files sit on disk, and failed on the first fresh clone. Both
dataset patterns are now root-anchored.

**Attack step count recorded against a superseded patch.** `attack.steps` in the
protocol described a 30-epoch run after a 109-epoch run had replaced it, because
the writer refused to overwrite a non-`PENDING` value. It now overwrites and
reports the change.

## 6. Provenance and reproducibility

Every stage writes a JSON run record naming the commit, whether the source tree
differed from it, library and hardware versions, pools read, and the sha256 of
every input and output.

Nine of ten stages record a commit with `dirty: false`. The exception is patch
training, which shows `dirty: true` at commit `df5e5e3`: that run executed on
Colab from a fresh clone, so the tree was not modified, but the dirty check in
force at that commit was unscoped and counted untracked run logs written earlier
in the same session. Correcting it would require retraining, which would produce
a different patch and invalidate every held-out result measured against this
one. The explanation is recorded rather than the log regenerated.

Reproducibility was checked by re-running every locally reproducible stage on a
clean tree. Pool manifests, the clean baseline and the fitted monitor came back
byte- or bit-identical, and the development attack evaluation reproduced digit
for digit. The ONNX export is not bit-reproducible — constant folding and node
ordering vary — and timings move by a few percent, both noted above.

**Artifact digests at this commit:**

| File | sha256 |
|---|---|
| `artifacts/patch_v1.pt` | `6fad168ad8ef191e5d5c09b7e96407bb229591e07727ffc43656511c2f673040` |
| `artifacts/monitor_v1.npz` | `e177fbc4a84d3d2c4e2a5692d52820854871d2fd94abc2788cce4c6145ad569b` |
| `artifacts/detector_v1.onnx` | `0cf0ae4524ab5575214e0ce0c281708de2e98b8e2c0ef8496dd2275c23d679cb` |

## 7. Limitations

`LIMITATIONS.md` is part of this artifact and states the boundaries in full. The
ones that most constrain the conclusions:

- **One model family, one attack objective.** No claim of transfer to other
  detectors or other attacks.
- **Digital, not physical.** No patch was printed, photographed or tested in a
  physical environment.
- **Non-adaptive adversary.** The attacker never optimized against the monitor.
  The RQ3 negative result is therefore obtained under the *easier* condition.
- **Deployment-format comparison, not edge hardware.** A laptop CPU, no
  accelerator, no quantization or pruning. The exported graph is
  trace-specialized to 320×320 batch 1.
- **General-purpose dataset.** COCO val2017 is not a critical-infrastructure or
  public-safety camera corpus.
- **Training and evaluation environments differ.** The patch was optimized under
  torch 2.11 / torchvision 0.26 on Colab and evaluated under 2.2.2 / 0.17.2
  locally. Weights are identical, verified by digest, but the implementations
  differ. The reported attack strength is therefore a lower bound including an
  unquantified amount of implementation transfer.

## 8. Reproduction

```bash
git clone https://github.com/GiorgosKarantonis/AdAHuman.git
cd AdAHuman && git checkout v1.0

python3.11 -m venv .venv && .venv/bin/pip install -r requirements.lock
bash scripts/00_fetch_coco.sh                      # ~1.8 GB, public

.venv/bin/python scripts/01_probe_model.py
.venv/bin/python scripts/02_freeze_pools.py --write
.venv/bin/python scripts/03_clean_baseline.py
.venv/bin/python scripts/06_fit_monitor.py
.venv/bin/python scripts/05_eval_attack.py         # uses artifacts/patch_v1.pt
.venv/bin/python scripts/05_eval_attack.py --untrained-patch
.venv/bin/python scripts/07_eval_monitor.py
.venv/bin/python scripts/08_export_onnx.py
.venv/bin/python scripts/09_benchmark_cpu.py
```

Patch optimization requires a GPU and runs from
`notebooks/colab_train_patch.ipynb`, which clones this repository rather than
unpacking an archive so the training run records the commit it used.

Reusing the shipped `artifacts/patch_v1.pt` reproduces the reported attack
numbers exactly. Retraining will not: GPU non-determinism and a different
library version produce a different patch.

## 9. Inventory

| Path | Contents |
|---|---|
| `configs/protocol_v1.yaml` | The frozen protocol; every parameter governing a result |
| `configs/manifests/` | Pool membership with content digests |
| `adahuman/` | Library: pools, attack, monitor, evaluation, deployment, provenance |
| `scripts/` | Ten pipeline stages, run in the order numbered |
| `results/` | Nine result files, one per measurement |
| `logs/` | Dated run records, one per stage execution |
| `artifacts/` | Patch, fitted monitor, exported model, feature archives |
| `LIMITATIONS.md` | Boundaries, corrections, and result-dependent caveats |
| `PROTOCOL.md` | The protocol explained |
| `NOTICE` | Licence scope, dependency terms, dissemination reasoning |

---

*Every figure in this report is drawn from a file in `results/` at tag `v1.0`.
`scripts/verify_report.py` re-derives each one from its source file
and fails if the document and the data disagree — it caught a p95 latency
written here as 78.4 ms when the recorded value rounds to 78.3.*
