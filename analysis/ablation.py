"""Attribute a change in the metrics table to the thing that caused it.

Two things moved under `docs/metrics.md` at once — the featuriser started
reflecting every frame about ``y = GOAL_Y``, and the label windows grew
plausibility guards — and a single before-and-after cannot say which of them did
what. This runs the comparison as a chain of cells, each a directory of cached
out-of-fold passes plus the dataset whose labels those passes should be scored
against, and reports the step between consecutive cells.

    python analysis/ablation.py \\
        --cell "before:data/interim/corners_v2.jsonl:runs/oof" \\
        --cell "mirrored:data/interim/corners_v2.jsonl:runs/oof-preguard" \\
        --cell "mirrored + guarded labels:data/interim/corners.jsonl:runs/oof2"

Nothing is trained here; every cell must already have been computed by
`analysis/precompute_oof.py`. Cells are matched **by seed**, so a step differences
the two numbers a single fold draw produced and the shared split noise cancels —
the same within-repeat differencing the metrics table uses to compare two models.
Read a step against the spread printed beside it: the seeds are common to both
cells, so a step smaller than its own spread is not a result.

One honest limit. Two cells that differ in their *labels* are not scored on the
same rows, so the step between them is the change in the reported number, not a
model improvement. Only a step between two cells sharing a dataset isolates the
featuriser.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:  # running the file directly, not as a module
    sys.path.insert(0, str(REPO))

KEYS: tuple[tuple[str, str, str], ...] = (
    ("receiver", "top1_attacker", "receiver top-1"),
    ("receiver", "top3_attacker", "receiver top-3"),
    ("receiver", "overall_accuracy", "receiver overall acc."),
    ("receiver", "defence_auc", "defence-wins AUC"),
    ("shot", "auc", "shot AUC"),
)


def read_metric(metrics: dict, family: str, key: str, model: str) -> float:
    """One number from a seed's metrics, for a model or for a difference of two.

    ``"gnn-logreg"`` reads the *gap*, differenced inside the seed before anything
    else happens. That is the quantity the metrics table's reading section is
    about — "the GNN beats the baseline by X" — and stepping it across
    configurations answers a question stepping the two columns separately cannot:
    whether a change that lifted the GNN lifted the baseline by more.
    """
    if "-" in model:
        left, right = model.split("-", 1)
        return float(metrics[family][left][key]) - float(metrics[family][right][key])
    return float(metrics[family][model][key])


def parse_cell(text: str) -> tuple[str, str, str]:
    """``"label:dataset:cache_dir"`` -> its three parts."""
    parts = text.split(":")
    if len(parts) != 3 or not all(p.strip() for p in parts):
        raise argparse.ArgumentTypeError(
            f"expected 'label:dataset:cache_dir', got {text!r}"
        )
    return tuple(p.strip() for p in parts)  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cell",
        action="append",
        required=True,
        type=parse_cell,
        metavar="LABEL:DATASET:CACHE_DIR",
        help="one column of the comparison; repeat, in the order to step through",
    )
    parser.add_argument(
        "--model",
        default="gnn",
        choices=["gnn", "logreg", "heuristic", "gnn-logreg", "gnn-heuristic"],
        help="which column to step, or a difference of two columns (default: %(default)s)",
    )
    parser.add_argument("--json", default=None, help="also write the raw numbers here")
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(1)  # scoring cached predictions; keep off the other cores

    import numpy as np

    from setpiece import train as trainmod

    cells: list[dict] = []
    for label, dataset, cache_dir in args.cell:
        paths = sorted(Path(cache_dir).glob("oof-seed*.npz"))
        if not paths:
            parser.error(f"no cached passes in {cache_dir}; run analysis/precompute_oof.py first")
        rows = trainmod.load_rows(dataset)
        per_seed: dict[int, dict] = {}
        for path in paths:
            cv = trainmod.load_oof(path)
            cv.check(len(rows))
            per_seed[cv.seed] = trainmod.metrics_from_oof(rows, cv)
        cells.append(
            {
                "label": label,
                "dataset": dataset,
                "cache_dir": cache_dir,
                "seeds": sorted(per_seed),
                "n_receiver_rows": int(
                    next(iter(per_seed.values()))["receiver"]["gnn"]["n_rows"]
                ),
                "per_seed": per_seed,
            }
        )

    print(f"model: {args.model}\n")
    for cell in cells:
        print(
            f"  {cell['label']}: {len(cell['seeds'])} repeats from {cell['cache_dir']}, "
            f"labels from {cell['dataset']}, {cell['n_receiver_rows']} receiver rows"
        )
    print()

    header = "| metric | " + " | ".join(c["label"] for c in cells) + " |"
    print(header)
    print("|" + "---|" * (len(cells) + 1))
    table: dict[str, dict[str, str]] = {}
    for family, key, pretty in KEYS:
        cellsv = []
        for cell in cells:
            vals = [
                read_metric(cell["per_seed"][s], family, key, args.model) for s in cell["seeds"]
            ]
            cellsv.append(f"{np.mean(vals):.4f} ±{np.std(vals, ddof=1):.4f}")
        table[pretty] = dict(zip([c["label"] for c in cells], cellsv, strict=True))
        print(f"| {pretty} | " + " | ".join(cellsv) + " |")
    print()

    steps: list[dict] = []
    for before, after in zip(cells, cells[1:], strict=False):
        shared = sorted(set(before["seeds"]) & set(after["seeds"]))
        same_rows = before["dataset"] == after["dataset"]
        print(f"### {before['label']} -> {after['label']} ({len(shared)} shared seeds)")
        if not same_rows:
            print(
                "The labels differ, so the two cells are not scored on the same rows: this "
                "step is the change in the reported number, not a model improvement."
            )
        print()
        print("| metric | step | its own spread | ratio | reading |")
        print("|---|---|---|---|---|")
        for family, key, pretty in KEYS:
            deltas = np.asarray(
                [
                    read_metric(after["per_seed"][s], family, key, args.model)
                    - read_metric(before["per_seed"][s], family, key, args.model)
                    for s in shared
                ],
                dtype=np.float64,
            )
            mean, sd = float(deltas.mean()), float(deltas.std(ddof=1))
            ratio = abs(mean) / sd if sd > 0 else float("inf")
            verdict = "outside the noise floor" if ratio >= 2.0 else "**inside the noise floor**"
            steps.append(
                {"from": before["label"], "to": after["label"], "metric": pretty,
                 "mean": mean, "sd": sd, "ratio": ratio, "same_rows": same_rows}
            )
            print(f"| {pretty} | {mean:+.4f} | ±{sd:.4f} | {ratio:.1f}x | {verdict} |")
        print()

    if args.json:
        Path(args.json).write_text(
            json.dumps({"model": args.model, "table": table, "steps": steps}, indent=2)
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
