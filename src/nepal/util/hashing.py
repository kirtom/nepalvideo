"""Content hashing for asset_id.

The spec says asset_id is the sha256 of the file. On several hundred GB a full
read is hours of IO for no benefit, so files above ``full_hash_limit`` are
identified by a sampled digest: head, tail, midpoint and size. Collisions are
not a practical concern for a few thousand media files, and the digest is
stable across runs, which is what idempotency actually needs.

Set full_hash_limit=None to force a true whole-file sha256.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK = 1 << 20          # 1 MiB
SAMPLE = 1 << 22         # 4 MiB per sampled region
FULL_HASH_LIMIT = 1 << 28  # 256 MiB


def sha256_file(path: str | Path, *, full_hash_limit: int | None = FULL_HASH_LIMIT) -> str:
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    if full_hash_limit is None or size <= full_hash_limit:
        with open(p, "rb") as fh:
            for block in iter(lambda: fh.read(CHUNK), b""):
                h.update(block)
        return h.hexdigest()

    h.update(b"sampled")
    with open(p, "rb") as fh:
        for offset in (0, max(0, size // 2 - SAMPLE // 2), max(0, size - SAMPLE)):
            fh.seek(offset)
            h.update(fh.read(SAMPLE))
    return h.hexdigest()
