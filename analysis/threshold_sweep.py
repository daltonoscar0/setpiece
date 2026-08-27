"""Re-run the receiver-label match-threshold audit from cached predictions.

`RECEIVER_MATCH_MAX_M = 8.0` in the frozen `setpiece/_io.py` decides which
attacker-first corners get a receiver label: the first touch is matched to the
nearest candidate attacker in the freeze frame and the row survives only if that
candidate is within the threshold. `setpiece.train.label_threshold_sweep`
re-derives the labels at other thresholds from the `receiver_match_dist` the
dataset stores for exactly that purpose and re-scores stored out-of-fold
predictions against them, so nothing is retrained.

`setpiece train` already runs this and writes the result into `docs/metrics.md`.
This script is the same sweep without the training run attached, for when the
question is about the threshold rather than about the models:

    python analysis/threshold_sweep.py --cache-dir runs/oof
    python analysis/threshold_sweep.py --oof runs/latest/oof.npz --thresholds 5,8,15

It needs out-of-fold predictions that already exist — from `--cache-dir`, which
`analysis/precompute_oof.py` and `setpiece train --cache-dir` both fill, or from
a single `--oof` file. It will not train anything to produce them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:  # running the file directly, not as a module
    sys.path.insert(0, str(REPO))


def parse_thresholds(text: str) -> list[float | None]:
    """``"4,6,8,none"`` -> ``[4.0, 6.0, 8.0, None]``."""
    out: list[float | None] = []
    for piece in text.split(","):
        token = piece.strip().lower()
        if not token:
            continue
        out.append(None if token in {"none", "inf", "unthresholded"} else float(token))
    if not out:
        raise ValueError(f"no thresholds parsed from {text!r}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="data/interim/corners.jsonl")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="directory of oof-seed*.npz files; every one found is used as a repeat",
    )
    parser.add_argument("--oof", default=None, help="a single .npz of out-of-fold predictions")
    parser.add_argument(
        "--thresholds",
        default="4,6,8,10,12,none",
        help="comma separated metres, or 'none' for unthresholded (default: %(default)s)",
    )
    parser.add_argument("--json", default=None, help="also write the raw sweep here")
    args = parser.parse_args(argv)

    from setpiece import train as trainmod

    sources: list[Path] = []
    if args.oof:
        sources.append(Path(args.oof))
    if args.cache_dir:
        sources.extend(sorted(Path(args.cache_dir).glob("oof-seed*.npz")))
    if not sources:
        parser.error(
            "no out-of-fold predictions given; pass --oof or --cache-dir. This script "
            "re-scores existing predictions and never trains."
        )

    rows = trainmod.load_rows(args.dataset)
    # The signature is deliberately not checked here: the sweep only reads the
    # per-row candidate distributions and the labels, both of which are valid
    # whatever hyperparameters produced them. It *is* checked when a cache is
    # reused to build the metrics table, where the configuration matters.
    cvs = [trainmod.load_oof(p) for p in sources]
    for cv in cvs:
        cv.check(len(rows))
    print(f"{len(rows)} corners, {len(cvs)} repeat(s) from {[p.name for p in sources]}\n")

    sweep = trainmod.label_threshold_sweep(
        rows, cvs, thresholds=parse_thresholds(args.thresholds)
    )
    print(sweep["summary"], end="\n\n")
    print(sweep["table"], end="\n\n")
    for note in sweep["notes"]:
        print(f"- {note}")
    if args.json:
        Path(args.json).write_text(json.dumps(sweep, indent=2, default=str))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
