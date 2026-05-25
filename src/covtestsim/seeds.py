"""Stable seed helpers.

Python's built-in hash is intentionally randomized between processes, so all
study seeds are derived from BLAKE2 instead.
"""

from __future__ import annotations

import hashlib

BASE_SEED = 20260521


def stable_seed(*parts: object, base_seed: int = BASE_SEED) -> int:
    """Return a reproducible unsigned 32-bit seed from arbitrary labels."""

    text = "|".join([str(base_seed), *map(str, parts)])
    digest = hashlib.blake2s(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)

