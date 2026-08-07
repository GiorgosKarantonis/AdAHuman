# Known Limitations

This memorandum is part of the artifact. It is written before results exist and
updated as results are produced, so that limitations are stated as commitments
rather than added retroactively.

## Scope limitations, fixed by the protocol

1. **One model family.** All results use `ssdlite320_mobilenet_v3_large` at
   320x320. Nothing here establishes that findings transfer to other detector
   architectures. A second architecture is deliberately out of scope for the
   initial artifact.

2. **One attack objective.** Only person-suppression via an optimized patch is
   evaluated. Secondary digital evasion is deferred and not reported.

3. **Simulated, not physical.** Patches are applied and transformed digitally
   under expectation-over-transformation. No patch was printed, photographed, or
   tested in a physical environment. Digital EOT does not reproduce field
   conditions such as sensor response, real optics, motion, or weather.

4. **Non-adaptive adversary.** The attacker does not know about, and does not
   optimize against, the runtime monitor. Published work shows detectors of
   adversarial inputs frequently fail against adaptive attackers who target the
   detector directly. No claim of robustness to an adaptive adversary is made.

5. **Deployment-format comparison, not edge hardware.** RQ4 compares a PyTorch
   baseline against an ONNX Runtime CPU execution path on a general-purpose
   laptop CPU. It is not a measurement on a production edge accelerator, NPU, or
   mobile device. No quantization or pruning is performed.

   Specific limits on the RQ4 numbers:

   - **One machine, one thread count.** All timings come from a single Intel
     i5-8257U at 4 intra-op threads. Relative performance between the two
     runtimes can invert on different hardware, thread counts, or builds.
   - **The exported graph is trace-specialized.** Export raised TracerWarnings
     indicating that Python-level control flow and some tensor values were
     baked into the graph as constants. The artifact is therefore valid for the
     traced configuration -- 320x320, batch size 1 -- and should not be assumed
     correct at other shapes.
   - **Memory figures are coarse.** Resident-set growth is shared with the rest
     of the process and the allocator does not promptly return freed pages.
     These are indicative comparisons, not exact footprints.
   - **Timing uses eight cycled frames.** Enough to average over per-frame NMS
     cost, not enough to characterize the full distribution of scene
     complexity.

6. **Single dataset, single domain.** COCO val2017 is a general-purpose object
   detection dataset. It is not a critical-infrastructure or public-safety
   camera dataset, and results should not be read as characterizing performance
   in those environments.

7. **Bounded sample sizes.** Pool sizes were chosen to fit the available
   compute. Metrics carry sampling error that the reported confidence intervals
   quantify but do not eliminate.

## Corrections made during construction

Recorded here rather than quietly fixed, because the failure mode is common and
a reviewer should be able to see it was caught.

**Monitor threshold was calibrated in-sample (found and fixed 2026-08-05,
before any held-out measurement).** The first version of the RQ3 monitor
estimated a 480-dimensional covariance from 500 reference images and selected
its operating threshold on those same images. It reported the target 5%
false-positive rate — which an in-sample quantile does by construction — while
the true rate on held-out clean images from the same pool was **60%**. The
threshold is now calibrated on a reference split held out from the covariance
fit, and features are projected to 64 PCA components before estimation. Held-out
false-positive rate is now within sampling error of the 5% target.

Diagnostically, the out-of-sample calibration was the fix; the PCA projection
was a secondary stabilization that mattered less than expected. No held-out
measurement was ever produced under the original version. The superseded fit
and its run log remain in `logs/`.

## The patch was optimized in a different environment than it is evaluated in

The adversarial patch is optimized on a Google Colab GPU and evaluated locally
on CPU. Those two environments do not run the same library versions:

| | Patch optimization | Everything else |
|---|---|---|
| torch | 2.11.0+cu128 | 2.2.2 |
| torchvision | 0.26.0+cu128 | 0.17.2 |
| device | Tesla T4 | Intel i5-8257U |

The local versions are the ones pinned in `configs/protocol_v1.yaml` and
`requirements.lock`; the Colab versions are recorded in the training run log
under `logs/`. The local pin is not arbitrary — torch 2.2.2 is the last release
with macOS x86_64 wheels, and the RQ4 timings must be measured on the CPU they
describe.

The detector weights are identical in both, verified by the sha256 pinned in the
protocol. But nine minor versions separate the two torchvision releases, and the
SSDLite implementation, its anchor generation, and its postprocessing may differ
in ways that change the gradients the patch was optimized against.

Two consequences:

1. **The reported attack strength is a lower bound**, not a white-box result. A
   patch optimized against the exact implementation that scores it would be at
   least as strong. What is measured here includes an unquantified amount of
   implementation transfer.

2. **A weak attack result is ambiguous.** Under-training and cross-version
   transfer loss are not separable from the held-out numbers alone. Attributing
   a weak result to one rather than the other would require an experiment that
   has not been run — optimizing under matched versions, or evaluating the same
   patch under both.

This is disclosed rather than corrected because correcting it means either
pinning Colab to versions it does not ship (uncertain, given its Python
version), or moving the local pin forward and discarding the RQ4 timings and
clean baseline already measured. Neither is obviously right, and the honest
description costs less than a misleading fix.

## Result-dependent limitations

To be completed when results exist. Any negative or mixed finding for the RQ3
monitor is reported here and in the artifact report, not omitted.

- RQ1 (attack effect): _pending_
- RQ3 (monitor separation): _pending_
- RQ4 (conversion fidelity, latency, memory): _pending_

## Provenance

All code in this repository is independently written for this project from
public literature and public documentation. No employer or client source code,
data, thresholds, learned parameters, feature representations, or internal
implementation details are used or reproduced. All model weights and images are
public and are downloaded at run time rather than redistributed here.
