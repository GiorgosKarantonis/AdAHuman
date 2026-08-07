#!/usr/bin/env python
"""Check that every figure quoted in the documentation matches results/.

A report is written by hand, so its numbers can drift from the data they
describe -- through a transcription slip, or because a stage was re-run and the
prose was not updated. This check caught one of each during construction: a p95
latency written as 78.4 ms when the recorded value rounds to 78.3, and an RQ4
table left behind by a re-run.

Every figure asserted here is formatted from the result file and required to
appear verbatim in the document. That makes the claim "these numbers come from
results/" verifiable rather than asserted, which is the same standard the
artifact applies to everything else.

Usage:
    scripts/verify_report.py
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

DOCUMENTS = ("ARTIFACT_REPORT.md", "README.md")


def load(name: str) -> dict:
    with (RESULTS / f"{name}.json").open() as handle:
        return json.load(handle)


def figures() -> list[tuple[str, float | int, str]]:
    """Every (label, value, format) the documentation is expected to quote."""
    out: list[tuple[str, float | int, str]] = []

    attack = load("rq1_attack")["eval_untouched"]
    control = load("rq1_attack_control")["eval_untouched"]
    dev = load("rq1_attack_dev")["attack_dev"]

    for label, source in (("held-out", attack), ("control", control), ("dev", dev)):
        s = source["suppression"]
        out += [
            (f"{label} suppression", s["rate"], "{:.4f}"),
            (f"{label} ci low", s["ci95"][0], "{:.4f}"),
            (f"{label} ci high", s["ci95"][1], "{:.4f}"),
        ]
    for label, source in (("held-out", attack), ("control", control)):
        for cond in ("clean", "attacked"):
            for group in ("patched", "unpatched"):
                out.append((
                    f"{label} {cond} {group} recall",
                    source[cond]["grouped_recall"][group]["recall"],
                    "{:.4f}",
                ))

    monitor = load("rq3_monitor_eval")
    for key in (
        "clean_vs_adversarial",
        "ordinary_shift_vs_adversarial",
        "clean_vs_ordinary_shift",
    ):
        out.append((f"{key} auroc", monitor[key]["auroc"], "{:.4f}"))
    for name, row in monitor["per_corruption_vs_adversarial"].items():
        out.append((f"corruption {name}", row["auroc"], "{:.4f}"))

    benchmark = load("rq4_benchmark")
    for path in ("pytorch", "onnxruntime"):
        for metric in ("median_ms", "p95_ms", "p99_ms"):
            out.append((
                f"{path} {metric}", benchmark["latency"][path][metric], "{:.1f}"
            ))

    export = load("rq4_export")
    out.append((
        "conversion count mismatches",
        export["fidelity"]["n_count_mismatches"],
        "{:d}",
    ))

    baseline = load("rq1_clean_baseline")
    for pool in ("reference", "attack_dev", "eval_untouched"):
        out.append((
            f"{pool} recall", baseline[pool]["operating_point"]["recall"], "{:.3f}"
        ))
        out.append((f"{pool} AP50", baseline[pool]["coco_map"]["AP@.50"], "{:.3f}"))

    return out


def main() -> int:
    texts = {}
    for name in DOCUMENTS:
        path = ROOT / name
        if not path.is_file():
            print(f"missing document: {name}")
            return 2
        texts[name] = path.read_text()

    combined = "\n".join(texts.values())
    expected = figures()
    missing = [
        (label, fmt.format(value))
        for label, value, fmt in expected
        if fmt.format(value) not in combined
    ]

    print(f"checked {len(expected)} figures against {RESULTS.name}/")
    if missing:
        print(f"\n{len(missing)} figure(s) in results/ do not appear in the docs:")
        for label, rendered in missing:
            print(f"  {label:36s} expected {rendered}")
        print(
            "\nEither the documentation is stale, or a value was transcribed "
            "incorrectly. Update the prose to match results/, never the reverse."
        )
        return 1

    print("all figures match their source files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
