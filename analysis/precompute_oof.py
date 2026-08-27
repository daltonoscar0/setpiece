"""Compute one seed's purged out-of-fold predictions and cache them.

`docs/metrics.md` reports every figure as a mean over repeats of the whole
cross-validation protocol, and each repeat costs five fits per head. Repeats are
independent, so this script does exactly one of them and writes the result to the
cache `setpiece.train.cached_cross_validate` reads. Several can therefore run at
once, one process per seed, and `setpiece train --seeds N --cache-dir ...` then
collects them without retraining anything.

    for s in 0 1000 2000 3000 4000; do
        python analysis/precompute_oof.py --seed $s --cache-dir runs/oof &
    done; wait

Pin the thread count. GNN training is only bit-reproducible at a fixed intra-op
thread count, so a cache written at four threads and one written at one thread
are not interchangeable — and the count is not part of the cache signature
because it is a property of the machine, not of the configuration. The default
here is one thread, which is also what the shipped metrics were measured at.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:  # running the file directly, not as a module
    sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", default="data/interim/corners.jsonl")
    parser.add_argument("--cache-dir", default="runs/oof")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    import torch

    torch.set_num_threads(args.threads)

    from setpiece import train as trainmod

    started = time.perf_counter()
    rows = trainmod.load_rows(args.dataset)
    hp = dict(trainmod.DEFAULTS)
    hp["folds"] = args.folds
    cv = trainmod.cached_cross_validate(
        rows,
        dataset_path=args.dataset,
        folds=args.folds,
        seed=args.seed,
        hp=hp,
        cache_dir=args.cache_dir,
        log=lambda m: print(m, flush=True),
    )
    metrics = trainmod.metrics_from_oof(rows, cv)
    print(
        f"seed {args.seed}: shot AUC gnn {metrics['shot']['gnn']['auc']:.4f} "
        f"logreg {metrics['shot']['logreg']['auc']:.4f}; "
        f"receiver top-1 gnn {metrics['receiver']['gnn']['top1_attacker']:.4f} "
        f"logreg {metrics['receiver']['logreg']['top1_attacker']:.4f}; "
        f"{time.perf_counter() - started:.0f}s at {args.threads} thread(s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
