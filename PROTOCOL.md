# AdAHuman Evaluation Protocol

A deployment-aware evaluation of adversarial patch attacks against an
edge-oriented person detector, with a runtime feature-distance diagnostic and a
CPU deployment-format comparison.

The machine-readable protocol is [`configs/protocol_v1.yaml`](configs/protocol_v1.yaml).
This document explains what it means and why the experiment is structured this
way. Where the two disagree, the YAML governs — it is what the code reads.

## Research questions

| ID | Question | Reported in |
|----|----------|-------------|
| RQ1 | How does an optimized person-suppression patch affect detection performance, before and after deployment-format conversion? | `results/rq1_*` |
| RQ3 | Can an independently implemented feature-distance score separate adversarial inputs from clean and ordinarily-shifted inputs at an acceptable false-positive rate? | `results/rq3_*` |

| RQ4 | What conversion-fidelity, latency, memory, and model-size tradeoffs arise on a CPU deployment path? | `results/rq4_*` |
| RQ5 | Can an independent party reproduce the above from the frozen protocol? | this repository |

RQ2 (evaluating a separate resilience control such as adversarial training) is
**deferred**. It is not implemented and no RQ2 result is claimed.

## Why the pools are split four ways

The credibility of every number here rests on a single discipline: **the
untouched evaluation pool is measured once, after all decisions are frozen.**

| Pool | Size | May be used for | Never used for |
|------|------|-----------------|----------------|
| `reference` | 500 | Fitting monitor statistics (mu, Sigma); conversion-fidelity probe | Attack optimization; final metrics |
| `attack_dev` | 300 | Optimizing the patch; tuning attack and threshold hyperparameters | Any reported final metric |
| `eval_untouched` | 500 | Final attack, robustness, and monitoring results | Any tuning or selection decision |
| `negative` | 250 | False-positive rates on images with no annotated person | Attack optimization |

Pools are disjoint, fixed by sorted COCO image ID under seed `20260805`, and
recorded as explicit manifests with sha256 digests in `configs/manifests/`. Pool
membership is reproducible from the seed alone; the manifests exist so that a
reviewer can verify membership without rerunning the selection.

If a hyperparameter is chosen by looking at `eval_untouched`, the resulting
number is no longer an out-of-sample measurement. The code is structured to make
that mistake hard: pool loading is centralized, and every script declares which
pools it is entitled to read.

## Freeze rule

`configs/protocol_v1.yaml` contains fields marked `PENDING`. These are values
that can only be determined by probing the environment (exact torchvision module
paths, feature tensor dimensions, library versions, hardware identifiers).

`adahuman.config.load_protocol(stage=...)` raises if a script that touches
`eval_untouched` is invoked while any field it depends on is still `PENDING`.
The freeze is therefore enforced by the code, not by discipline alone.

Once frozen, a field is never edited in place. A change requires publishing
`protocol_v2.yaml` and rerunning, with both versions kept in history.

## Conditions compared

The attack is not measured against nothing. Every held-out image is evaluated
under matched conditions, so each reported difference is paired on identical
images:

| Condition | What it isolates |
|-----------|------------------|
| clean | the baseline |
| **untrained random patch** | occlusion alone — a patch of the same size in the same place, carrying no optimization |
| trained patch | the attack |
| ordinary shift | benign corruption, the monitor's control condition |

The occlusion control matters more than it looks. A random patch already
suppresses some detections purely by covering the target; a pilot run at n=24
lost 13% of patched targets that way. Reporting the trained patch's suppression
rate against zero would credit the optimizer with work the occlusion did. The
honest comparison is trained-versus-random, and both are reported.

Similarly, recall is split between *patched* and *unpatched* targets in the
same frames. The patch is local, so unpatched recall should barely move; if it
collapses too, the finding is global degradation and must be described that
way.

## Execution split

Patch optimization needs a GPU and runs on Google Colab
(`notebooks/colab_train_patch.ipynb`). Everything else — baselines, evaluation,
the monitor, ONNX export, and all timing measurements — runs locally on CPU.

This split is methodologically correct rather than a compromise: RQ4 is
explicitly a *CPU deployment-path* measurement, so the laptop is the intended
measurement device. The patch produced on Colab is a frozen input to local
evaluation, transferred as a `.pt` tensor plus its sha256, and its provenance is
recorded in the run log.

## Reporting rules

These are commitments made before results exist.

- Negative and mixed findings are reported. The RQ3 monitor is a hypothesis; if
  it fails to separate adversarial from benign shift, that is the result.
- No result is described as evidence of an effective defense. A distribution-
  shift diagnostic is not an adversarial defense.
- Every reported number carries its exact conditions: model, weights, pool,
  attack parameters, thresholds, software versions, hardware.
- Failed runs, non-convergence, and numerical problems are logged and disclosed.
- Nothing in this repository is described as having existed before the date of
  the commit that introduced it. The git history is the record.

## Status

Under construction. No result is claimed until it appears in `results/` with a
dated run log. See [`LIMITATIONS.md`](LIMITATIONS.md).
