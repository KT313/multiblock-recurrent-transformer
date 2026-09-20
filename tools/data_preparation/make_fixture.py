# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Generate deterministic synthetic original text for reproducing the local measurements."""

import argparse
import json
import random
from pathlib import Path

WORDS = [
    "the", "model", "learns", "from", "original", "documents", "code", "mathematics", "reasoning", "token", "sequence",
    "memory", "language", "distributed", "systems", "records", "gradients", "multilingual", "café", "日本語", "😀",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--case", choices=("mixed", "long"), required=True)
    args = parser.parse_args()
    long = args.case == "long"
    rng = random.Random(219 if long else 19)
    with args.output.open("x", encoding="utf-8") as handle:
        for index in range(192 if long else 23_001):
            words = 24_000 if long else (400 if index % 29 == 0 else 45)
            text = " ".join(rng.choices(WORDS, k=words)) + f" document {index}"
            row: dict[str, str | bool]
            if long:
                row = {"source": f"source_{index % 4}", "text": text}
            else:
                passive = index >= 20_001
                row = {"source": "passive" if passive else "main", "group": "mixed", "passive": passive, "text": text}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
