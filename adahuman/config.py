"""Protocol loading with enforced freeze semantics.

The evaluation protocol lives in ``configs/protocol_v1.yaml``. Some of its
fields cannot be written by hand because they describe the environment (exact
module paths, tensor shapes, library and hardware identifiers). Those start as
the sentinel ``PENDING`` and are filled in by probe scripts.

The rule this module enforces is the one that makes the reported numbers
meaningful: **no script may read the untouched evaluation pool while a protocol
field that determines its result is still PENDING.** Freezing is therefore a
property of the code path, not a habit the author has to remember.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Iterable

import yaml

PENDING = "PENDING"

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL = REPO_ROOT / "configs" / "protocol_v1.yaml"

#: The evaluation pool that may be measured exactly once, after freeze.
HELD_OUT_POOL = "eval_untouched"


@dataclasses.dataclass(frozen=True)
class Stage:
    """One step of the pipeline, and what it is entitled to do.

    Attributes:
        name: Stage identifier, used on the command line and in run logs.
        pools: Pools this stage may read. Reading anything else is a bug.
        requires_frozen: Dotted protocol paths that must not be ``PENDING``
            before this stage runs.
        sections: Top-level protocol sections whose values determine this
            stage's output. When the stage touches the held-out pool, *every*
            field in these sections must be frozen -- not merely the ones the
            stage names explicitly, since an unfrozen neighbour in the same
            section is a value someone could still be tuning.

            Deliberately scoped rather than global: requiring the whole file to
            be frozen would force placeholder values into fields that legitimately
            cannot be determined yet, which defeats the purpose of the sentinel.
    """

    name: str
    pools: frozenset[str]
    requires_frozen: tuple[str, ...]
    sections: tuple[str, ...] = ()

    @property
    def touches_held_out(self) -> bool:
        return HELD_OUT_POOL in self.pools


_MODEL_FIELDS = (
    "model.weights_sha256",
    "model.torch_version",
    "model.torchvision_version",
)
_ATTACK_FIELDS = ("attack.steps",)
_MONITOR_FIT_FIELDS = (
    "monitor.feature_hook",
    "monitor.feature_unpooled_shape",
    "monitor.feature_dim",
)
_MONITOR_EVAL_FIELDS = _MONITOR_FIT_FIELDS + ("monitor.threshold_value",)
_DEPLOY_FIELDS = ("deploy.runtime_version", "deploy.hardware")

STAGES: dict[str, Stage] = {
    # Selects pool membership and writes the manifests. Reads no pixels.
    "freeze_pools": Stage("freeze_pools", frozenset(), ()),
    # Loads the detector and reports its module tree and feature shapes so the
    # PENDING monitor fields can be filled in. Reference pool only.
    "probe_model": Stage("probe_model", frozenset({"reference"}), ()),
    # Clean detection performance. Touches the held-out pool, so the model,
    # task definition, and pool membership must all already be pinned. The
    # attack and monitor sections are irrelevant here and stay unfrozen.
    "clean_baseline": Stage(
        "clean_baseline",
        frozenset({"reference", "attack_dev", "eval_untouched", "negative"}),
        _MODEL_FIELDS,
        sections=("model", "task", "data"),
    ),
    # Patch optimization. Development pools only -- never the held-out pool.
    "train_patch": Stage(
        "train_patch", frozenset({"attack_dev", "reference"}), _MODEL_FIELDS
    ),
    # Measures the patch on the development pool. Exists so the question "is
    # this attack worth spending the held-out pool on?" can be answered without
    # spending it -- the held-out pool is measured once, and a disappointing
    # result there cannot be acted on afterwards without invalidating it.
    # Entitled to attack_dev only, so it cannot become a second held-out look.
    "eval_attack_dev": Stage(
        "eval_attack_dev", frozenset({"attack_dev"}), _MODEL_FIELDS
    ),
    # Single scored run of the frozen patch.
    "eval_attack": Stage(
        "eval_attack",
        frozenset({"eval_untouched", "negative"}),
        _MODEL_FIELDS + _ATTACK_FIELDS,
        sections=("model", "task", "data", "attack"),
    ),
    # Fit mu/Sigma and select the threshold, on development pools only.
    "fit_monitor": Stage(
        "fit_monitor",
        frozenset({"reference", "attack_dev"}),
        _MODEL_FIELDS + _MONITOR_FIT_FIELDS,
    ),
    # Score the frozen monitor against clean, shifted, and attacked inputs.
    "eval_monitor": Stage(
        "eval_monitor",
        frozenset({"eval_untouched", "negative"}),
        _MODEL_FIELDS + _ATTACK_FIELDS + _MONITOR_EVAL_FIELDS,
        sections=("model", "task", "data", "attack", "monitor", "ordinary_shift"),
    ),
    "export_onnx": Stage("export_onnx", frozenset({"reference"}), _MODEL_FIELDS),
    "benchmark_cpu": Stage(
        "benchmark_cpu", frozenset({"reference"}), _MODEL_FIELDS + _DEPLOY_FIELDS
    ),
}


class ProtocolNotFrozen(RuntimeError):
    """Raised when a stage would produce a result it is not entitled to produce."""


class PoolAccessError(RuntimeError):
    """Raised when a stage reads a pool outside its declared entitlement."""


class Protocol:
    """A loaded protocol, scoped to the stage that loaded it."""

    def __init__(self, raw: dict[str, Any], stage: Stage, path: pathlib.Path):
        self._raw = raw
        self.stage = stage
        self.path = path

    def get(self, dotted: str, default: Any = dataclasses.MISSING) -> Any:
        """Fetch a value by dotted path, e.g. ``"attack.eot.scale_range"``."""
        node: Any = self._raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is not dataclasses.MISSING:
                    return default
                raise KeyError(f"{dotted!r} not present in {self.path.name}")
            node = node[part]
        return node

    def pool(self, name: str) -> dict[str, Any]:
        """Fetch a pool spec, refusing pools this stage may not read."""
        if name not in self.stage.pools:
            raise PoolAccessError(
                f"stage {self.stage.name!r} may read {sorted(self.stage.pools)}, "
                f"not {name!r}. If this access is legitimate, widen the stage "
                f"definition deliberately -- do not work around this check."
            )
        return self.get(f"data.pools.{name}")

    @property
    def raw(self) -> dict[str, Any]:
        return self._raw

    def pending_fields(self) -> list[str]:
        """Every dotted path in the protocol still set to ``PENDING``."""
        found: list[str] = []

        def walk(node: Any, prefix: str) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, f"{prefix}.{key}" if prefix else key)
            elif node == PENDING:
                found.append(prefix)

        walk(self._raw, "")
        return sorted(found)


def load_protocol(
    stage: str,
    path: pathlib.Path | str = DEFAULT_PROTOCOL,
    allow_unfrozen: bool = False,
) -> Protocol:
    """Load the protocol for ``stage``, enforcing the freeze rule.

    Args:
        stage: Key into :data:`STAGES`.
        path: Protocol YAML to load.
        allow_unfrozen: Skip the freeze check. Reserved for smoke tests that
            exercise a code path without producing a reported measurement.
            Callers that pass this must name their outputs distinctly and
            record the fact in the run log -- the escape hatch is visible by
            design, because a silent one would make the whole check decorative.

    Returns:
        A :class:`Protocol` scoped to the stage.

    Raises:
        ProtocolNotFrozen: If a field this stage depends on is still PENDING,
            or if the stage touches the held-out pool while a field in its
            governing sections is PENDING.
    """
    if stage not in STAGES:
        raise KeyError(f"unknown stage {stage!r}; known: {sorted(STAGES)}")
    spec = STAGES[stage]

    path = pathlib.Path(path)
    with path.open() as handle:
        raw = yaml.safe_load(handle)

    protocol = Protocol(raw, spec, path)
    if not allow_unfrozen:
        _assert_frozen(protocol, spec)
    return protocol


def _assert_frozen(protocol: Protocol, spec: Stage) -> None:
    missing = [
        field
        for field in spec.requires_frozen
        if protocol.get(field, default=PENDING) == PENDING
    ]
    if missing:
        raise ProtocolNotFrozen(
            f"stage {spec.name!r} requires these protocol fields to be frozen "
            f"first: {missing}. Run the probe scripts that fill them."
        )

    # Touching the held-out pool is the irreversible step: it converts an
    # out-of-sample measurement into an in-sample one if anything that shaped
    # the result was still adjustable. So it demands that every field in the
    # sections governing this stage be frozen, not merely those it names.
    if spec.touches_held_out:
        pending = [
            field
            for field in protocol.pending_fields()
            if field.split(".")[0] in spec.sections
        ]
        if pending:
            raise ProtocolNotFrozen(
                f"stage {spec.name!r} reads the held-out pool "
                f"{HELD_OUT_POOL!r}, so every field in sections "
                f"{list(spec.sections)} must be frozen first. "
                f"Still PENDING: {pending}"
            )


def describe_stages() -> Iterable[str]:
    """Human-readable summary of the pipeline, for `--help` output and docs."""
    for name, spec in STAGES.items():
        pools = ", ".join(sorted(spec.pools)) or "none"
        yield f"{name:16s} pools=[{pools}]"
