#!/usr/bin/env python3
"""Build a training corpus: generated samples, some joined into longer ones.

`corpus.py` yields one field-sized sample at a time, and most are much
shorter than the 512-byte window that the model sees. At inference a window
is often full, for example inside a long body. This script joins several
generated samples into one for a share of the output, so that the model also
trains on dense windows and on payloads surrounded by unrelated content. The
label of a joined sample is the union of its parts.
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import Iterator

import corpus

SEPARATORS = (b" ", b" ", b"\n", b"&", b", ", b"; ", b"\t", b"")


def compose(seed: int, count: int, join_rate: float) -> Iterator[tuple[bytes, int]]:
    """Yield `count` samples, deterministic for `seed`."""
    rng = random.Random(seed ^ 0x5EED)
    # Joining consumes several generated samples per output sample.
    source = corpus.generate(seed, count * 8)
    for _ in range(count):
        payload, mask = next(source)
        if rng.random() < join_rate:
            separator = rng.choice(SEPARATORS)
            for _ in range(rng.randint(1, 9)):
                extra, extra_mask = next(source)
                if len(payload) + len(separator) + len(extra) > corpus.MAX_LEN:
                    # Whole parts only, so that no label loses its payload.
                    break
                if rng.random() < 0.5:
                    payload = payload + separator + extra
                else:
                    payload = extra + separator + payload
                mask |= extra_mask
        yield payload, mask


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--join-rate", type=float, default=0.35)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)
    written = corpus.write_corpus(
        arguments.out, compose(arguments.seed, arguments.count, arguments.join_rate)
    )
    print(f"wrote {written} records to {arguments.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
