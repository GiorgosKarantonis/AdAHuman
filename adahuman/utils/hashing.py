"""Content hashing for artifact provenance.

Every input and output that a reviewer might want to verify -- pool manifests,
the optimized patch, the exported model, result files -- is identified by a
sha256 digest recorded in the run log. This is what lets an expert letter cite
the artifact by hash rather than by description.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

_CHUNK = 1 << 20


def sha256_file(path: pathlib.Path | str) -> str:
    """Digest a file's bytes."""
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    """Digest a JSON-serializable object with stable key ordering.

    Used for pool manifests, so that the digest depends on pool membership and
    not on dictionary iteration order or whitespace.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def short(digest: str, length: int = 12) -> str:
    """Abbreviate a digest for log lines. Full digests go in the run record."""
    return digest[:length]
