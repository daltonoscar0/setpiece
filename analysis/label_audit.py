"""Describe what a corners dataset actually is, without training anything.

Two facts bound every number in `docs/metrics.md` and neither is visible from the
model tables: what fraction of open-data corners could have been in the dataset
at all, and what the plausibility guards on the label windows threw away.
`setpiece train` prints both into the metrics file, but it costs five fits per
head per repeat to get there. This script prints the same two sections from the
dataset and its fetch ledger alone, in seconds:

    python analysis/label_audit.py
    python analysis/label_audit.py --before data/interim/corners_preguard.jsonl

``--before`` is a build of the same corners from before the guards. It changes
nothing; it only lets the audit say which class each guarded row used to carry,
which the guarded dataset cannot recover on its own — the guard records that a
touch was implausible, not whose it would have been.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:  # running the file directly, not as a module
    sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="data/interim/corners.jsonl")
    parser.add_argument(
        "--before",
        default=None,
        help="a build of the same corners from before the label windows grew guards",
    )
    parser.add_argument("--json", default=None, help="also write the raw summaries here")
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(1)  # featurising only; keep it off the other agents' cores

    import numpy as np

    from setpiece import train as trainmod

    rows = trainmod.load_rows(args.dataset)
    labels = trainmod.label_window_summary(rows, baseline_path=args.before)
    coverage = trainmod.coverage_summary(args.dataset)
    data = {
        "path": args.dataset,
        "n_corners": len(rows),
        "n_receiver_rows": sum(1 for r in rows if r.receiver_valid),
        "shot_base_rate": float(np.mean([r.y_shot for r in rows])),
        "labels": labels,
    }
    if coverage is not None:
        data["coverage"] = coverage

    print(f"# {args.dataset} — {len(rows)} corners, {len({r.group for r in rows})} matches\n")
    if coverage is None:
        print("No fetch ledger beside the dataset, so 360 coverage is not reconstructable.\n")
    print(trainmod.format_dataset_markdown(data))
    if args.before is None:
        print(
            "Pass --before to compare against a pre-guard build; without one the classes "
            "the guarded rows used to carry are not recoverable."
        )
    if args.json:
        Path(args.json).write_text(json.dumps(data, indent=2, default=str))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
