"""Training, cross-validation and honest evaluation for the two corner heads.

Everything in here is organised around one rule: **corners from the same match
never straddle a split**. Corners inside a match share a coaching plan, a set of
personnel and a scouting report, so a random row-level split leaks the answer
across the fold boundary and inflates every number. Folds are therefore built by
grouping on ``match_id`` and assigning whole matches.

Evaluation is out-of-fold. Every corner is scored exactly once, by a model that
never saw its match, and the pooled out-of-fold predictions are what the metrics
table is computed from. The two baselines are fitted on the same training folds
and scored on the same held-out rows, so the three columns of the table are
directly comparable.

Receiver rows with ``y_receiver_valid == False`` are dropped from the loss *and*
from every receiver metric: those are corners where the first touch was an
attacker who was not visible in the freeze frame, so there is no node to point
at. Corners where the defence won first contact are **kept** — "the defence wins"
is an explicit class of the receiver softmax, not a missing label.

The whole protocol is run **once per seed and reported as a mean with a spread**.
GNN training is only bit-reproducible at a fixed intra-op thread count (see
:func:`set_seed`), so a single run quoted to three decimals invites the reader to
believe a third-decimal difference. Each seed re-draws both the fold assignment
and the network initialisation, so the spread covers the two things that would
actually change on a re-run, and the deterministic baselines get a spread too —
theirs comes purely from the split.

Both binary probabilities — the shot sigmoid and the receiver head's
"defence wins first contact" class — are checked against observed frequencies by
:mod:`setpiece.calibration`, and the calibrator that ships in a checkpoint's card
is fitted on out-of-fold predictions only.

Freeze frames carry no velocities and nothing here invents any; every model card
written from this module records ``assume_static: True``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.loader import DataLoader

from setpiece import features as featmod
from setpiece._io import RECEIVER_MATCH_MAX_M, read_corners
from setpiece.calibration import (
    Calibrator,
    calibration_report,
    expected_calibration_error,
    fit_calibrator,
    format_reliability_markdown,
)
from setpiece.models import (
    ReceiverModel,
    ShotModel,
    count_parameters,
    receiver_log_softmax,
    save_model,
)
from setpiece.types import GOAL_X, GOAL_Y, CornerFrame

__all__ = [
    "DEFAULTS",
    "LABEL_THRESHOLDS",
    "CVResult",
    "Row",
    "add_train_arguments",
    "build_parser",
    "cached_cross_validate",
    "calibration_from_oof",
    "cross_validate",
    "label_threshold_sweep",
    "load_oof",
    "load_rows",
    "main",
    "metrics_from_oof",
    "nested_calibration",
    "oof_signature",
    "purged_fold_assignment",
    "receiver_metrics",
    "roc_auc",
    "run_from_args",
    "save_oof",
    "seed_list",
    "set_seed",
    "shot_metrics",
    "summarise_seeds",
    "train_all",
]

#: Default hyperparameters. Small on purpose: a smoke run has to be quick.
DEFAULTS: dict[str, Any] = {
    "folds": 5,
    "epochs": 20,
    "hidden": 64,
    "layers": 3,
    "dropout": 0.1,
    "lr": 3e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "seed": 0,
    "baseline_iters": 400,
    "baseline_lr": 0.05,
    "baseline_weight_decay": 1e-3,
    "seeds": 5,
}

#: Seeds are spaced this far apart so that the per-fold seed of one repeat
#: (``seed + fold``) can never collide with the per-fold seed of another. Two
#: repeats sharing an initialisation would understate the spread they exist to
#: measure.
SEED_STRIDE: int = 1000

#: Bin count for every reliability curve and calibration error quoted here.
CALIBRATION_BINS: int = 10

#: Calibrators tried on each binary head, in the order the table lists them.
CALIBRATORS: tuple[str, ...] = ("platt", "isotonic")

#: Softness of the nearest-to-goal heuristic, in metres. It only has to produce a
#: monotone ranking; the temperature turns that ranking into a distribution so the
#: heuristic can be compared against the models on overall accuracy too.
HEURISTIC_TEMPERATURE_M: float = 3.0

#: Column keys in the emitted receiver table, in order.
RECEIVER_METRIC_KEYS: tuple[str, ...] = (
    "top1_attacker",
    "top3_attacker",
    "overall_accuracy",
    "defence_auc",
)


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed python, numpy and torch.

    CPU training is bit-reproducible *at a fixed intra-op thread count*, and only
    then. Torch's parallel reductions sum in thread-partition order, so the same
    seed on the same data yields different weights under a different
    ``torch.get_num_threads()`` — measured here as a change in the fitted
    receiver weights between 4 threads and 1, and as GNN metrics that move in the
    third decimal between runs on differently loaded machines. The thread count
    is recorded in every model card so a number can be traced back to it. The
    linear baselines are dense single-kernel ops and do reproduce exactly.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# metrics, implemented here so there is no scikit-learn dependency
# --------------------------------------------------------------------------- #
def _average_ranks(scores: np.ndarray) -> np.ndarray:
    """Competition-free ranks 1..n, ties sharing their mean rank."""
    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    i = 0
    while i < ordered.size:
        j = i
        while j + 1 < ordered.size and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def roc_auc(y_true: Sequence[float] | np.ndarray, y_score: Sequence[float] | np.ndarray) -> float:
    """Area under the ROC curve, via the rank (Mann-Whitney U) identity.

    ``y_true`` is thresholded at 0.5, so booleans and 0/1 floats both work. Ties
    in ``y_score`` get their average rank, which is the standard convention and
    makes a constant score come out at exactly 0.5. Returns ``nan`` when one of
    the two classes is absent, because the quantity is then undefined.

    A non-finite score is rejected rather than ranked. ``np.argsort`` sorts
    ``nan`` to the end, which silently hands every missing prediction the *best*
    rank and inflates the AUC — a hole in an out-of-fold prediction array would
    then read as skill instead of failing loudly.
    """
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1) > 0.5
    scores = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if truth.size != scores.size:
        raise ValueError(f"length mismatch: {truth.size} labels, {scores.size} scores")
    if scores.size and not bool(np.isfinite(scores).all()):
        raise ValueError(
            f"{int((~np.isfinite(scores)).sum())} of {scores.size} scores are not finite; "
            "an unscored row must not be ranked"
        )
    n_pos = int(truth.sum())
    n_neg = int(truth.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    return float((ranks[truth].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _rank_of_target(probs: np.ndarray, target: int) -> int:
    """0-based rank of ``target`` when ``probs`` is sorted descending.

    Ties are broken by index, which is deterministic and never flatters the model:
    a tied target sits behind every lower-indexed tie.
    """
    order = np.argsort(-probs, kind="stable")
    return int(np.nonzero(order == target)[0][0])


def receiver_metrics(
    cand_probs: Sequence[np.ndarray],
    p_defence: Sequence[float] | np.ndarray,
    target_pos: Sequence[int] | np.ndarray,
) -> dict[str, float]:
    """Receiver metrics over a set of held-out rows.

    Parameters
    ----------
    cand_probs:
        Per row, the probability assigned to each candidate attacker, in
        candidate order. Need not sum to one; ``p_defence`` carries the rest.
    p_defence:
        Per row, the probability that the defence wins first contact.
    target_pos:
        Per row, the index into that row's candidates of the true receiver, or
        ``-1`` when the defence won first contact.

    Top-1 and top-3 are reported on the attacker-first subset only, ranking the
    candidates against each other. That is the question "which attacker gets on
    the end of it", and it is the only question the nearest-to-goal heuristic can
    answer, so it is where the three models are compared head to head. Overall
    accuracy is the argmax over candidates *and* the defence class across every
    row, and the AUC is for predicting that the defence wins.
    """
    targets = np.asarray(target_pos, dtype=np.int64).reshape(-1)
    defence = np.asarray(p_defence, dtype=np.float64).reshape(-1)
    if len(cand_probs) != targets.size or defence.size != targets.size:
        raise ValueError("cand_probs, p_defence and target_pos must be the same length")

    top1 = top3 = n_attacker = 0
    correct = 0
    for probs, target, p_def in zip(cand_probs, targets, defence, strict=True):
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        best = int(np.argmax(probs)) if probs.size else -1
        best_p = float(probs[best]) if probs.size else -1.0
        predicted_defence = probs.size == 0 or p_def > best_p
        if target < 0:
            correct += int(predicted_defence)
            continue
        n_attacker += 1
        rank = _rank_of_target(probs, int(target))
        top1 += int(rank == 0)
        top3 += int(rank < 3)
        correct += int(not predicted_defence and best == int(target))

    n_rows = int(targets.size)
    return {
        "top1_attacker": top1 / n_attacker if n_attacker else float("nan"),
        "top3_attacker": top3 / n_attacker if n_attacker else float("nan"),
        "overall_accuracy": correct / n_rows if n_rows else float("nan"),
        "defence_auc": roc_auc(targets < 0, defence),
        "n_rows": float(n_rows),
        "n_attacker_first": float(n_attacker),
    }


def shot_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """AUC plus the base rate, for the binary shot head."""
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    return {
        "auc": roc_auc(truth, y_score),
        "base_rate": float(truth.mean()) if truth.size else float("nan"),
        "n_rows": float(truth.size),
    }


# --------------------------------------------------------------------------- #
# rows
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    """One corner, featurised once and reused by the GNN and both baselines."""

    corner_id: str
    group: str
    match_id: int | None
    competition: str | None
    season: str | None
    graph: Any
    flat: np.ndarray
    pooled: np.ndarray
    cand_nodes: np.ndarray
    cand_goal_dist: np.ndarray
    target_pos: int
    receiver_valid: bool
    y_shot: float
    # Label provenance, carried for the threshold audit in ``analysis/``. These
    # are label-adjacent diagnostics, never features: nothing downstream of
    # ``build_graph`` sees them.
    first_touch: str = "unknown"
    receiver_match_dist: float | None = None
    first_touch_dt: float | None = None


def _pool(flat: np.ndarray, is_attacker: np.ndarray) -> np.ndarray:
    """Graph-level summary of the flat node features, ``[4 * N_NODE_FEATURES]``.

    Mean over everyone, mean over the attackers, mean over the defenders, and an
    elementwise max. This is the only representation the linear baselines get of
    the whole box, and it mirrors the GNN's readout so the comparison is fair.
    """
    width = flat.shape[1] if flat.ndim == 2 else featmod.N_NODE_FEATURES
    if flat.size == 0:
        return np.zeros(4 * width, dtype=np.float64)
    flat = flat.astype(np.float64)
    attack = flat[is_attacker] if is_attacker.any() else np.zeros((1, width))
    defend = flat[~is_attacker] if (~is_attacker).any() else np.zeros((1, width))
    return np.concatenate(
        [flat.mean(axis=0), attack.mean(axis=0), defend.mean(axis=0), flat.max(axis=0)]
    )


def _row_from_frame(frame: CornerFrame) -> Row:
    graph = featmod.build_graph(frame)
    flat = featmod.build_flat(frame)
    candidate = graph.candidate_mask.numpy().astype(bool)
    cand_nodes = np.nonzero(candidate)[0].astype(np.int64)

    positions = np.asarray([[p.x, p.y] for p in frame.players], dtype=np.float64).reshape(-1, 2)
    if cand_nodes.size:
        cand_pos = positions[cand_nodes]
        goal_dist = np.hypot(GOAL_X - cand_pos[:, 0], GOAL_Y - cand_pos[:, 1])
    else:
        goal_dist = np.zeros(0, dtype=np.float64)

    y_receiver = int(graph.y_receiver.reshape(-1)[0].item())
    valid = bool(graph.y_receiver_valid.reshape(-1)[0].item())
    if y_receiver < 0:
        target_pos = -1
    else:
        hits = np.nonzero(cand_nodes == y_receiver)[0]
        if hits.size == 0:  # pragma: no cover - features guarantees this cannot happen
            target_pos, valid = -1, False
        else:
            target_pos = int(hits[0])

    is_attacker = np.asarray([p.is_attacker for p in frame.players], dtype=bool)
    # Grouping key for the purged split. A corner with no ``match_id`` — user CSV
    # and ad-hoc JSON inputs carry none — becomes its own singleton group rather
    # than joining one shared bucket, because collapsing them would put every
    # unlabelled corner in a single fold. The consequence is that such corners
    # are *not* purged against each other: if two of them come from the same
    # match, that match can straddle a fold boundary and nothing here can tell.
    # ``_split_summary`` counts them so the metrics table can say so out loud.
    group = str(frame.match_id) if frame.match_id is not None else f"corner:{frame.corner_id}"
    return Row(
        corner_id=frame.corner_id,
        group=group,
        match_id=frame.match_id,
        competition=frame.competition,
        season=frame.season,
        graph=graph,
        flat=flat,
        pooled=_pool(flat, is_attacker),
        cand_nodes=cand_nodes,
        cand_goal_dist=goal_dist,
        target_pos=target_pos,
        receiver_valid=valid,
        y_shot=float(graph.y_shot.reshape(-1)[0].item()),
        first_touch=str(frame.first_touch),
        receiver_match_dist=frame.receiver_match_dist,
        first_touch_dt=(frame.meta or {}).get("first_touch_dt"),
    )


def load_rows(dataset_path: str | Path, *, limit: int | None = None) -> list[Row]:
    """Read a corners JSONL file and featurise every corner once."""
    frames = read_corners(Path(dataset_path))
    if limit is not None:
        frames = frames[:limit]
    if not frames:
        raise ValueError(f"no corners in {dataset_path}")
    return [_row_from_frame(f) for f in frames]


# --------------------------------------------------------------------------- #
# purged splits
# --------------------------------------------------------------------------- #
def purged_fold_assignment(groups: Sequence[str], n_folds: int, *, seed: int = 0) -> np.ndarray:
    """Assign every row to a fold so that a group is never split.

    ``groups`` is one key per row — the ``match_id``. Whole groups are shuffled
    and then handed to whichever fold currently holds the fewest rows, which
    balances fold sizes without ever splitting a match. Returns an ``int64``
    array of fold indices, one per row.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    keys = [str(g) for g in groups]
    members: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        members.setdefault(key, []).append(i)

    n_folds = min(n_folds, len(members))
    if n_folds < 2:
        raise ValueError(
            f"cannot build {len(members)}-way purged folds: the dataset has only "
            f"{len(members)} distinct match groups"
        )

    order = sorted(members)
    rng = random.Random(seed)
    rng.shuffle(order)
    # Largest first, so the greedy balance cannot be wrecked by a late big match.
    order.sort(key=lambda k: -len(members[k]))

    assignment = np.full(len(keys), -1, dtype=np.int64)
    load = [0] * n_folds
    for key in order:
        fold = min(range(n_folds), key=lambda f: (load[f], f))
        for i in members[key]:
            assignment[i] = fold
        load[fold] += len(members[key])
    return assignment


def _split_summary(rows: Sequence[Row], assignment: np.ndarray) -> dict[str, Any]:
    n_folds = int(assignment.max()) + 1
    fold_rows = [int((assignment == f).sum()) for f in range(n_folds)]
    fold_groups = [len({rows[i].group for i in np.nonzero(assignment == f)[0]}) for f in
                   range(n_folds)]
    ungrouped = sum(1 for r in rows if r.match_id is None)
    return {
        "scheme": "purged group k-fold on match_id; every corner of a match in one fold",
        "n_folds": n_folds,
        "rows_per_fold": fold_rows,
        "matches_per_fold": fold_groups,
        "n_matches": len({r.group for r in rows}),
        # Corners with no match_id are singleton groups, so they are not purged
        # against one another. Nonzero here means the purging claim is weaker
        # than the scheme string suggests.
        "n_ungrouped_corners": ungrouped,
        "evaluation": "pooled out-of-fold predictions; each corner scored once, unseen match",
    }


# --------------------------------------------------------------------------- #
# GNN training
# --------------------------------------------------------------------------- #
def _receiver_loss(model: ReceiverModel, batch: Any) -> Tensor | None:
    node_logits, defence_logit = model(batch)
    n_graphs = int(batch.num_graphs)
    node_lp, defence_lp = receiver_log_softmax(
        node_logits, defence_logit, batch.batch, n_graphs
    )
    valid = batch.y_receiver_valid.reshape(-1)
    target = featmod.global_receiver_index(batch)
    to_node = valid & (target >= 0)
    to_defence = valid & (target < 0)
    n = int(valid.sum())
    if n == 0:
        return None
    total = torch.zeros((), dtype=node_lp.dtype, device=node_lp.device)
    if bool(to_node.any()):
        total = total - node_lp[target[to_node]].sum()
    if bool(to_defence.any()):
        total = total - defence_lp[to_defence].sum()
    return total / n


def _train_receiver(
    rows: Sequence[Row],
    *,
    epochs: int,
    hidden: int,
    layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
) -> ReceiverModel:
    set_seed(seed)
    model = ReceiverModel(
        featmod.N_NODE_FEATURES,
        featmod.N_EDGE_FEATURES,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
    )
    usable = [r for r in rows if r.receiver_valid]
    if not usable:
        return model
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        [r.graph for r in usable],
        batch_size=min(batch_size, len(usable)),
        shuffle=True,
        generator=generator,
    )
    model.train()
    for _ in range(epochs):
        for batch in loader:
            loss = _receiver_loss(model, batch)
            if loss is None:
                continue
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
    model.eval()
    return model


def _train_shot(
    rows: Sequence[Row],
    *,
    epochs: int,
    hidden: int,
    layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
) -> ShotModel:
    set_seed(seed + 1)
    model = ShotModel(
        featmod.N_NODE_FEATURES,
        featmod.N_EDGE_FEATURES,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
    )
    if not rows:
        return model
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed + 1)
    loader = DataLoader(
        [r.graph for r in rows],
        batch_size=min(batch_size, len(rows)),
        shuffle=True,
        generator=generator,
    )
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for batch in loader:
            logit = model(batch).reshape(-1)
            loss = criterion(logit, batch.y_shot.reshape(-1))
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
    model.eval()
    return model


@torch.no_grad()
def _predict_receiver_gnn(
    model: ReceiverModel, rows: Sequence[Row], batch_size: int
) -> tuple[list[np.ndarray], np.ndarray]:
    model.eval()
    cand_probs: list[np.ndarray] = []
    defence: list[float] = []
    if not rows:
        return cand_probs, np.zeros(0, dtype=np.float64)
    loader = DataLoader(
        [r.graph for r in rows], batch_size=min(batch_size, len(rows)), shuffle=False
    )
    for batch in loader:
        node_logits, defence_logit = model(batch)
        n_graphs = int(batch.num_graphs)
        node_lp, defence_lp = receiver_log_softmax(
            node_logits, defence_logit, batch.batch, n_graphs
        )
        probs = node_lp.exp().to(torch.float64)
        candidate = batch.candidate_mask.to(torch.bool)
        for g in range(n_graphs):
            here = (batch.batch == g) & candidate
            cand_probs.append(probs[here].cpu().numpy())
        defence.extend(defence_lp.exp().to(torch.float64).cpu().numpy().tolist())
    return cand_probs, np.asarray(defence, dtype=np.float64)


@torch.no_grad()
def _predict_shot_gnn(model: ShotModel, rows: Sequence[Row], batch_size: int) -> np.ndarray:
    model.eval()
    if not rows:
        return np.zeros(0, dtype=np.float64)
    loader = DataLoader(
        [r.graph for r in rows], batch_size=min(batch_size, len(rows)), shuffle=False
    )
    out: list[float] = []
    for batch in loader:
        logit = model(batch).reshape(-1)
        out.extend(torch.sigmoid(logit).to(torch.float64).cpu().numpy().tolist())
    return np.asarray(out, dtype=np.float64)


# --------------------------------------------------------------------------- #
# baseline 1: heuristics
# --------------------------------------------------------------------------- #
def _heuristic_receiver(
    rows: Sequence[Row], p_defence_prior: float
) -> tuple[list[np.ndarray], np.ndarray]:
    """Nearest-to-goal candidate wins; the defence class gets the training prior."""
    cand_probs: list[np.ndarray] = []
    for row in rows:
        d = row.cand_goal_dist
        if d.size == 0:
            cand_probs.append(np.zeros(0, dtype=np.float64))
            continue
        logits = -d / HEURISTIC_TEMPERATURE_M
        logits = logits - logits.max()
        weights = np.exp(logits)
        weights = weights / weights.sum()
        cand_probs.append(weights * (1.0 - p_defence_prior))
    return cand_probs, np.full(len(rows), p_defence_prior, dtype=np.float64)


# --------------------------------------------------------------------------- #
# baseline 2: logistic regression on the flat features
# --------------------------------------------------------------------------- #
@dataclass
class _Standardiser:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> _Standardiser:
        if x.size == 0:
            width = x.shape[1] if x.ndim == 2 else 1
            return cls(np.zeros(width), np.ones(width))
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-8] = 1.0
        return cls(mean, std)

    def apply(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x.astype(np.float64)
        return (x.astype(np.float64) - self.mean) / self.std


@dataclass
class _ShotLogReg:
    scaler: _Standardiser
    weight: np.ndarray
    bias: float

    def predict(self, rows: Sequence[Row]) -> np.ndarray:
        if not rows:
            return np.zeros(0, dtype=np.float64)
        x = self.scaler.apply(np.stack([r.pooled for r in rows]))
        z = x @ self.weight + self.bias
        return 1.0 / (1.0 + np.exp(-z))


def _fit_shot_logreg(
    rows: Sequence[Row], *, iters: int, lr: float, weight_decay: float, seed: int
) -> _ShotLogReg:
    x_raw = np.stack([r.pooled for r in rows]) if rows else np.zeros((0, 1))
    scaler = _Standardiser.fit(x_raw)
    width = x_raw.shape[1]
    weight = torch.zeros(width, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    if not rows:
        return _ShotLogReg(scaler, weight.detach().numpy(), 0.0)

    torch.manual_seed(seed)
    x = torch.from_numpy(scaler.apply(x_raw))
    y = torch.tensor([r.y_shot for r in rows], dtype=torch.float64)
    optimiser = torch.optim.Adam([weight, bias], lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(iters):
        optimiser.zero_grad()
        loss = criterion(x @ weight + bias, y)
        loss.backward()
        optimiser.step()
    return _ShotLogReg(scaler, weight.detach().numpy(), float(bias.detach().item()))


@dataclass
class _ReceiverLogReg:
    node_scaler: _Standardiser
    pooled_scaler: _Standardiser
    node_weight: np.ndarray
    node_bias: float
    pooled_weight: np.ndarray
    pooled_bias: float

    def predict(self, rows: Sequence[Row]) -> tuple[list[np.ndarray], np.ndarray]:
        cand_probs: list[np.ndarray] = []
        defence: list[float] = []
        for row in rows:
            if row.cand_nodes.size == 0:
                cand_probs.append(np.zeros(0, dtype=np.float64))
                defence.append(1.0)
                continue
            feats = self.node_scaler.apply(row.flat[row.cand_nodes])
            node_logits = feats @ self.node_weight + self.node_bias
            pooled = self.pooled_scaler.apply(row.pooled.reshape(1, -1)).reshape(-1)
            defence_logit = float(pooled @ self.pooled_weight + self.pooled_bias)
            logits = np.concatenate([node_logits, [defence_logit]])
            logits = logits - logits.max()
            weights = np.exp(logits)
            weights = weights / weights.sum()
            cand_probs.append(weights[:-1])
            defence.append(float(weights[-1]))
        return cand_probs, np.asarray(defence, dtype=np.float64)


def _fit_receiver_logreg(
    rows: Sequence[Row], *, iters: int, lr: float, weight_decay: float, seed: int
) -> _ReceiverLogReg:
    """Conditional logit over the candidates plus a pooled defence-wins class.

    The candidate scores are a linear function of exactly the flat node features
    the GNN sees per node; the defence score is a linear function of the pooled
    summary. Trained with the same per-graph softmax likelihood as the GNN head,
    so the only difference between this baseline and the model is the message
    passing.
    """
    usable = [r for r in rows if r.receiver_valid and r.cand_nodes.size > 0]
    node_width = featmod.N_NODE_FEATURES
    pooled_width = 4 * node_width
    if not usable:
        return _ReceiverLogReg(
            _Standardiser.fit(np.zeros((0, node_width))),
            _Standardiser.fit(np.zeros((0, pooled_width))),
            np.zeros(node_width),
            0.0,
            np.zeros(pooled_width),
            0.0,
        )

    node_raw = np.concatenate([r.flat[r.cand_nodes] for r in usable]).astype(np.float64)
    pooled_raw = np.stack([r.pooled for r in usable])
    node_scaler = _Standardiser.fit(node_raw)
    pooled_scaler = _Standardiser.fit(pooled_raw)

    segment: list[int] = []
    targets: list[int] = []
    offset = 0
    for i, row in enumerate(usable):
        n_cand = int(row.cand_nodes.size)
        segment.extend([i] * n_cand)
        targets.append(-1 if row.target_pos < 0 else offset + row.target_pos)
        offset += n_cand

    x_node = torch.from_numpy(node_scaler.apply(node_raw))
    x_pooled = torch.from_numpy(pooled_scaler.apply(pooled_raw))
    seg = torch.tensor(segment, dtype=torch.long)
    target = torch.tensor(targets, dtype=torch.long)
    to_node = target >= 0
    to_defence = ~to_node
    n_rows = len(usable)

    torch.manual_seed(seed)
    node_weight = torch.zeros(x_node.shape[1], dtype=torch.float64, requires_grad=True)
    node_bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    pooled_weight = torch.zeros(x_pooled.shape[1], dtype=torch.float64, requires_grad=True)
    pooled_bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.Adam(
        [node_weight, node_bias, pooled_weight, pooled_bias], lr=lr, weight_decay=weight_decay
    )
    for _ in range(iters):
        optimiser.zero_grad()
        node_logits = x_node @ node_weight + node_bias
        defence_logit = x_pooled @ pooled_weight + pooled_bias
        node_lp, defence_lp = receiver_log_softmax(node_logits, defence_logit, seg, n_rows)
        total = torch.zeros((), dtype=torch.float64)
        if bool(to_node.any()):
            total = total - node_lp[target[to_node]].sum()
        if bool(to_defence.any()):
            total = total - defence_lp[to_defence].sum()
        loss = total / n_rows
        loss.backward()
        optimiser.step()

    return _ReceiverLogReg(
        node_scaler,
        pooled_scaler,
        node_weight.detach().numpy(),
        float(node_bias.detach().item()),
        pooled_weight.detach().numpy(),
        float(pooled_bias.detach().item()),
    )


# --------------------------------------------------------------------------- #
# one cross-validation pass
# --------------------------------------------------------------------------- #
MODEL_KEYS: tuple[str, ...] = ("gnn", "logreg", "heuristic")


@dataclass
class CVResult:
    """Pooled out-of-fold predictions from one purged cross-validation pass.

    Every array is indexed by position in the ``rows`` list that produced it, and
    **covers every row**, including the ones excluded from the receiver metrics.
    Scoring a row is free once its fold model exists, and keeping the excluded
    predictions is what makes the label-threshold audit possible without
    retraining: a row dropped at 8 m is still a genuine out-of-fold prediction
    from a model that never saw its match.
    """

    seed: int
    assignment: np.ndarray
    splits: dict[str, Any]
    #: model -> per row, the probability of each candidate attacker.
    cand_probs: dict[str, list[np.ndarray]]
    #: model -> per row, P(defence wins first contact).
    p_defence: dict[str, np.ndarray]
    #: model -> per row, P(shot within five events).
    shot: dict[str, np.ndarray]

    def check(self, n_rows: int) -> None:
        """Fail loudly on a hole. An unscored row must never reach a metric."""
        for key in MODEL_KEYS:
            if len(self.cand_probs[key]) != n_rows:
                raise RuntimeError(
                    f"{key} produced {len(self.cand_probs[key])} candidate distributions "
                    f"for {n_rows} rows"
                )
            for name, arr in (("p_defence", self.p_defence[key]), ("shot", self.shot[key])):
                if arr.shape != (n_rows,) or not bool(np.isfinite(arr).all()):
                    raise RuntimeError(f"{key}/{name} is incomplete for {n_rows} rows")


def _fit_fold_models(
    train_rows: Sequence[Row], *, seed: int, hp: dict[str, Any]
) -> tuple[ReceiverModel, ShotModel]:
    """Both GNN heads on one training partition. Kept in one place so the
    cross-validation loop and the nested-calibration loop cannot drift apart."""
    kwargs = dict(
        epochs=hp["epochs"],
        hidden=hp["hidden"],
        layers=hp["layers"],
        dropout=hp["dropout"],
        lr=hp["lr"],
        weight_decay=hp["weight_decay"],
        batch_size=hp["batch_size"],
        seed=seed,
    )
    return _train_receiver(train_rows, **kwargs), _train_shot(train_rows, **kwargs)


def cross_validate(
    rows: Sequence[Row],
    *,
    folds: int,
    seed: int,
    hp: dict[str, Any],
    log: Any = None,
) -> CVResult:
    """One purged group k-fold pass: train per fold, score the held-out rows.

    ``seed`` drives the fold assignment *and* the initialisation, so repeating
    this with a different seed re-draws both. The two baselines are fitted on the
    identical training partition and scored on the identical held-out rows.
    """
    n = len(rows)
    assignment = purged_fold_assignment([r.group for r in rows], folds, seed=seed)
    n_folds = int(assignment.max()) + 1
    result = CVResult(
        seed=seed,
        assignment=assignment,
        splits=_split_summary(rows, assignment),
        cand_probs={k: [np.zeros(0, dtype=np.float64)] * n for k in MODEL_KEYS},
        p_defence={k: np.full(n, np.nan, dtype=np.float64) for k in MODEL_KEYS},
        shot={k: np.full(n, np.nan, dtype=np.float64) for k in MODEL_KEYS},
    )

    for fold in range(n_folds):
        test_idx = np.nonzero(assignment == fold)[0]
        train_idx = np.nonzero(assignment != fold)[0]
        train_rows = [rows[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]
        if log is not None:
            log(f"  seed {seed} fold {fold + 1}/{n_folds}: train {len(train_rows)}, "
                f"test {len(test_rows)}")

        receiver_model, shot_model = _fit_fold_models(train_rows, seed=seed + fold, hp=hp)

        train_valid = [r for r in train_rows if r.receiver_valid]
        prior = float(np.mean([r.target_pos < 0 for r in train_valid])) if train_valid else 0.5
        shot_rate = float(np.mean([r.y_shot for r in train_rows])) if train_rows else 0.5
        logreg_recv = _fit_receiver_logreg(
            train_rows,
            iters=hp["baseline_iters"],
            lr=DEFAULTS["baseline_lr"],
            weight_decay=DEFAULTS["baseline_weight_decay"],
            seed=seed + fold,
        )
        logreg_shot = _fit_shot_logreg(
            train_rows,
            iters=hp["baseline_iters"],
            lr=DEFAULTS["baseline_lr"],
            weight_decay=DEFAULTS["baseline_weight_decay"],
            seed=seed + fold,
        )

        preds: dict[str, tuple[list[np.ndarray], np.ndarray]] = {
            "gnn": _predict_receiver_gnn(receiver_model, test_rows, hp["batch_size"]),
            "heuristic": _heuristic_receiver(test_rows, prior),
            "logreg": logreg_recv.predict(test_rows),
        }
        for name, (cand_probs, p_def) in preds.items():
            # The table is only a comparison if all three models are scored on
            # exactly these rows in exactly this order. A predictor returning a
            # longer list would otherwise misalign silently and never raise.
            if len(cand_probs) != len(test_idx) or len(p_def) != len(test_idx):
                raise RuntimeError(
                    f"{name} returned {len(cand_probs)} candidate distributions and "
                    f"{len(p_def)} defence probabilities for {len(test_idx)} held-out "
                    "rows; every model must be scored on the identical rows"
                )
            for slot, row_id in enumerate(test_idx):
                result.cand_probs[name][int(row_id)] = np.asarray(
                    cand_probs[slot], dtype=np.float64
                )
                result.p_defence[name][int(row_id)] = float(p_def[slot])

        result.shot["gnn"][test_idx] = _predict_shot_gnn(shot_model, test_rows, hp["batch_size"])
        result.shot["heuristic"][test_idx] = shot_rate
        result.shot["logreg"][test_idx] = logreg_shot.predict(test_rows)

    result.check(n)
    return result


def metrics_from_oof(
    rows: Sequence[Row], cv: CVResult, *, keep: np.ndarray | None = None
) -> dict[str, Any]:
    """Receiver and shot tables from one pass of out-of-fold predictions.

    ``keep`` optionally restricts the receiver tables to a boolean subset of the
    rows — used by the label-threshold audit, which re-scores identical
    predictions against a stricter definition of "this row has a usable receiver
    label". The shot table always covers every row; the shot label has no such
    threshold.
    """
    valid = np.asarray([r.receiver_valid for r in rows], dtype=bool)
    if keep is not None:
        valid = valid & np.asarray(keep, dtype=bool).reshape(-1)
    ids = np.nonzero(valid)[0]
    target_pos = np.asarray([rows[i].target_pos for i in ids], dtype=np.int64)

    receiver: dict[str, Any] = {}
    for name in MODEL_KEYS:
        receiver[name] = receiver_metrics(
            [cv.cand_probs[name][i] for i in ids],
            cv.p_defence[name][ids],
            target_pos,
        )

    n_cands = np.asarray([rows[i].cand_nodes.size for i in ids], dtype=np.float64)
    attacker_first = target_pos >= 0
    receiver["chance"] = {
        "top1_attacker": float(np.mean(1.0 / n_cands[attacker_first]))
        if attacker_first.any()
        else float("nan"),
        "top3_attacker": float(
            np.mean(np.minimum(3.0, n_cands[attacker_first]) / n_cands[attacker_first])
        )
        if attacker_first.any()
        else float("nan"),
    }
    receiver["always_defence_accuracy"] = (
        float(np.mean(target_pos < 0)) if target_pos.size else float("nan")
    )

    y_shot = np.asarray([r.y_shot for r in rows], dtype=np.float64)
    shot = {name: shot_metrics(y_shot, cv.shot[name]) for name in MODEL_KEYS}
    return {"receiver": receiver, "shot": shot, "receiver_row_ids": ids}


# --------------------------------------------------------------------------- #
# repeated seeds
# --------------------------------------------------------------------------- #
def seed_list(seed: int, n_seeds: int) -> list[int]:
    """``n_seeds`` well-separated seeds starting at ``seed``. See :data:`SEED_STRIDE`."""
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be at least 1, got {n_seeds}")
    return [seed + SEED_STRIDE * k for k in range(n_seeds)]


def _aggregate(values: Sequence[float]) -> dict[str, Any]:
    """Mean, spread and the raw repeats behind one number.

    ``sd`` is the *sample* standard deviation (ddof=1) and is ``0.0`` for a
    single repeat rather than ``nan``, because "we ran this once" is better said
    by ``n`` than by a missing spread. Non-finite repeats are dropped, so a
    metric that is undefined on one split does not poison the mean.
    """
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return {"mean": float("nan"), "sd": float("nan"), "min": float("nan"),
                "max": float("nan"), "n": 0, "values": []}
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
        "values": [round(v, 6) for v in finite],
    }


def summarise_seeds(per_seed: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collapse one metrics dict per seed into ``(means, spread)``.

    ``means`` has the shape of a single-seed report — every leaf a float — so
    everything downstream that reads a metrics dict keeps working. ``spread``
    mirrors it with ``{mean, sd, min, max, n, values}`` at each leaf.
    """
    means: dict[str, Any] = {}
    spread: dict[str, Any] = {}
    for family in ("receiver", "shot"):
        means[family], spread[family] = {}, {}
        for model in per_seed[0][family]:
            entry = per_seed[0][family][model]
            if not isinstance(entry, dict):  # always_defence_accuracy and friends
                agg = _aggregate([s[family][model] for s in per_seed])
                means[family][model] = agg["mean"]
                spread[family][model] = agg
                continue
            means[family][model], spread[family][model] = {}, {}
            for key in entry:
                agg = _aggregate([s[family][model][key] for s in per_seed])
                means[family][model][key] = agg["mean"]
                spread[family][model][key] = agg
    return means, spread


def per_seed_summary(per_seed: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The headline figures of each repeat, small enough to sit in a model card.

    :func:`metrics_from_oof` also returns ``receiver_row_ids``, an array as long
    as the dataset; a card is a description, not a data dump, so only the numbers
    a reader would compare across repeats come through.
    """
    out: list[dict[str, Any]] = []
    for metrics in per_seed:
        out.append(
            {
                "receiver": {
                    model: {key: float(metrics["receiver"][model][key])
                            for key in RECEIVER_METRIC_KEYS}
                    for model in MODEL_KEYS
                },
                "shot": {
                    model: float(metrics["shot"][model]["auc"]) for model in MODEL_KEYS
                },
            }
        )
    return out


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def _binary_targets(rows: Sequence[Row]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The two genuinely binary labels, with the row mask each is defined on.

    ``shot`` is defined on every row. ``defence`` — "the defence wins first
    contact" — is only defined where the receiver label is usable, which is the
    same subset the receiver table is scored on.
    """
    valid = np.asarray([r.receiver_valid for r in rows], dtype=bool)
    return {
        "shot": (
            np.asarray([r.y_shot for r in rows], dtype=np.float64),
            np.ones(len(rows), dtype=bool),
        ),
        "defence": (
            np.asarray([r.target_pos < 0 for r in rows], dtype=np.float64),
            valid,
        ),
    }


def _head_scores(cv: CVResult, head: str, model: str) -> np.ndarray:
    return cv.shot[model] if head == "shot" else cv.p_defence[model]


def calibration_from_oof(
    rows: Sequence[Row], cv: CVResult, *, bins: int = CALIBRATION_BINS
) -> dict[str, Any]:
    """Measure calibration of both binary heads on one pass of out-of-fold scores.

    Reports the raw state of each head, then what each candidate calibrator would
    do **fitted and evaluated on the same out-of-fold rows**. That in-sample view
    is an optimistic bound, not an estimate of held-out calibration, and it is
    labelled as such everywhere it is used; :func:`nested_calibration` is the
    honest version. It is still worth having: if a calibrator cannot improve ECE
    even when it is allowed to see the answers, it will certainly not improve it
    out of sample.
    """
    targets = _binary_targets(rows)
    out: dict[str, Any] = {}
    for head, (y, mask) in targets.items():
        out[head] = {}
        for model in MODEL_KEYS:
            p = _head_scores(cv, head, model)[mask]
            entry: dict[str, Any] = {"raw": calibration_report(y[mask], p, bins=bins)}
            for kind in CALIBRATORS:
                cal = fit_calibrator(y[mask], p, kind=kind)
                entry[kind] = {
                    "in_sample": calibration_report(y[mask], cal.predict(p), bins=bins),
                    "params": cal.to_dict(),
                }
            out[head][model] = entry
    return out


def nested_calibration(
    rows: Sequence[Row],
    cv: CVResult,
    *,
    hp: dict[str, Any],
    kinds: Sequence[str] = CALIBRATORS,
    log: Any = None,
) -> dict[str, Any]:
    """Held-out calibration error, with no held-out row ever seen by its calibrator.

    Fitting a calibrator on the rows it is then scored on flatters it, and
    fitting one on a model's *training* rows flatters it the other way: an
    overfit model looks sharper in sample than it is, so the calibrator learns a
    correction the held-out rows do not need. The only honest arrangement is a
    second level of held-out data.

    For each outer fold ``f`` this trains an extra pair of heads on
    ``rows - (f + partner)``, scores its ``partner`` fold — rows that model never
    saw and that do not overlap ``f`` — and fits the calibrator on those scores.
    That calibrator is then applied to fold ``f``'s ordinary out-of-fold
    predictions from ``cv``. Folds are paired (0,1), (2,3), … so one extra model
    serves both members of a pair; with an odd fold count the last fold borrows
    fold 0 as its partner. Five folds therefore cost three extra fits per head,
    not five.

    The one approximation left, stated rather than hidden: the calibrator is
    fitted on scores from a model trained on three fifths of the data while it is
    applied to a model trained on four fifths. A model trained on less data is
    slightly less confident, so this is expected to *understate* how much
    shrinking the shipped model needs.
    """
    n_folds = int(cv.assignment.max()) + 1
    targets = _binary_targets(rows)
    calibrated = {
        head: {kind: np.full(len(rows), np.nan, dtype=np.float64) for kind in kinds}
        for head in targets
    }

    pair_of: dict[int, int] = {}
    for f in range(n_folds):
        partner = f + 1 if f % 2 == 0 else f - 1
        pair_of[f] = partner if partner < n_folds else 0
    cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    for fold in range(n_folds):
        partner = pair_of[fold]
        key = (min(fold, partner), max(fold, partner))
        if key not in cache:
            if log is not None:
                log(f"  nested calibration: extra fit excluding folds {key}")
            held = np.isin(cv.assignment, key)
            inner_train = [rows[i] for i in np.nonzero(~held)[0]]
            recv_model, shot_model = _fit_fold_models(
                inner_train, seed=cv.seed + 500 + key[0], hp=hp
            )
            p_shot = np.full(len(rows), np.nan, dtype=np.float64)
            p_def = np.full(len(rows), np.nan, dtype=np.float64)
            held_idx = np.nonzero(held)[0]
            held_rows = [rows[i] for i in held_idx]
            p_shot[held_idx] = _predict_shot_gnn(shot_model, held_rows, hp["batch_size"])
            _, defence = _predict_receiver_gnn(recv_model, held_rows, hp["batch_size"])
            p_def[held_idx] = defence
            cache[key] = (p_shot, p_def)
        p_shot, p_def = cache[key]

        fit_on = cv.assignment == partner
        apply_to = cv.assignment == fold
        for head, (y, mask) in targets.items():
            inner = np.asarray(p_shot if head == "shot" else p_def, dtype=np.float64)
            fit_mask = fit_on & mask & np.isfinite(inner)
            for kind in kinds:
                cal = fit_calibrator(y[fit_mask], inner[fit_mask], kind=kind)
                here = apply_to & mask
                calibrated[head][kind][here] = cal.predict(_head_scores(cv, head, "gnn")[here])

    out: dict[str, Any] = {}
    for head, (y, mask) in targets.items():
        raw = _head_scores(cv, head, "gnn")[mask]
        out[head] = {
            "raw_ece": expected_calibration_error(y[mask], raw, bins=CALIBRATION_BINS),
            "raw_ece_quantile": expected_calibration_error(
                y[mask], raw, bins=CALIBRATION_BINS, strategy="quantile"
            ),
        }
        for kind in kinds:
            p = calibrated[head][kind][mask]
            if not bool(np.isfinite(p).all()):  # pragma: no cover - every fold is covered
                raise RuntimeError(f"nested calibration left holes in {head}/{kind}")
            out[head][kind] = {
                "ece": expected_calibration_error(y[mask], p, bins=CALIBRATION_BINS),
                "ece_quantile": expected_calibration_error(
                    y[mask], p, bins=CALIBRATION_BINS, strategy="quantile"
                ),
                # Deliberately not called "auc". A single monotone calibrator
                # cannot change a ranking, but this pooled column is *five
                # different* per-fold maps stitched together, and two rows in
                # different folds can swap order across the seam. The shipped
                # calibrator is one global map and does preserve the AUC exactly;
                # this number is the price of measuring calibration honestly, not
                # a property of what ships.
                "auc_pooled_per_fold_maps": roc_auc(y[mask], p),
            }
        out[head]["raw_auc"] = roc_auc(y[mask], raw)
    return out


def _pick_calibrator(nested: dict[str, Any], head: str) -> tuple[str, float, float]:
    """The calibrator with the lowest honestly-measured ECE, and the two numbers.

    Ties and near-ties go to the simpler map: isotonic has to beat Platt by more
    than a tenth of the raw ECE to be worth its extra freedom on a few thousand
    rows. If nothing beats the raw head, ``"identity"`` is returned and the
    checkpoint ships uncalibrated.
    """
    raw = float(nested[head]["raw_ece"])
    scores = {kind: float(nested[head][kind]["ece"]) for kind in CALIBRATORS}
    kind = min(scores, key=lambda k: scores[k])
    if kind == "isotonic" and "platt" in scores and scores["platt"] <= scores[kind] + 0.1 * raw:
        kind = "platt"
    if scores[kind] >= raw:
        return "identity", raw, raw
    return kind, raw, scores[kind]


def fit_shipped_calibrators(
    rows: Sequence[Row], cv: CVResult, nested: dict[str, Any]
) -> dict[str, Any]:
    """Fit the calibrators that travel in the checkpoints' model cards.

    The *choice* of calibrator comes from :func:`nested_calibration`, which is the
    only honest measurement of it. The calibrator that actually ships is then
    refitted on the full pooled out-of-fold predictions, because the shipped
    model is itself refitted on every corner and the pooled out-of-fold scores
    are the largest set of honest predictions available for it. No held-out row
    informs the number quoted for it; every corner informs the map that ships.
    """
    targets = _binary_targets(rows)
    out: dict[str, Any] = {}
    for head, (y, mask) in targets.items():
        kind, raw_ece, nested_ece = _pick_calibrator(nested, head)
        p = _head_scores(cv, head, "gnn")[mask]
        cal: Calibrator = fit_calibrator(y[mask], p, kind=kind)
        out[head] = {
            "kind": kind,
            "calibrator": cal.to_dict(),
            "fitted_on": "pooled out-of-fold predictions, primary seed",
            "ece_raw": raw_ece,
            "ece_calibrated_nested": nested_ece,
            "n_rows": int(mask.sum()),
        }
    return out


def summarise_calibration(
    in_sample: dict[str, Any], nested: dict[str, Any], shipped: dict[str, Any]
) -> dict[str, Any]:
    """One flat entry per head, joining the raw description to the nested result.

    The descriptive numbers — how many rows, what the head actually predicted,
    the raw reliability curve — come from the pooled out-of-fold predictions. The
    *calibrated* ECE comes from :func:`nested_calibration`, because an ECE
    measured on the rows the calibrator was fitted on is a lower bound rather
    than a result.
    """
    out: dict[str, Any] = {}
    for head, entry in shipped.items():
        raw = in_sample[head]["gnn"]["raw"]
        kind = entry["kind"]
        n = nested[head]
        out[head] = {
            "n": raw["n"],
            "base_rate": raw["base_rate"],
            "mean_prediction": raw["mean_prediction"],
            "min_prediction": raw["min_prediction"],
            "max_prediction": raw["max_prediction"],
            "brier_raw": raw["brier"],
            "auc_raw": n["raw_auc"],
            "ece_raw": n["raw_ece"],
            "ece_raw_quantile": n["raw_ece_quantile"],
            "ece_calibrated": n["raw_ece"] if kind == "identity" else n[kind]["ece"],
            "ece_calibrated_quantile": (
                n["raw_ece_quantile"] if kind == "identity" else n[kind]["ece_quantile"]
            ),
            "chosen": kind,
            "curve": raw["curve"],
        }
    return out


# --------------------------------------------------------------------------- #
# receiver-label threshold audit
# --------------------------------------------------------------------------- #
#: Match thresholds the receiver-label audit sweeps, in metres. ``None`` means
#: "accept any match, however far". ``_io.RECEIVER_MATCH_MAX_M`` is 8.0 and is
#: frozen; the sweep re-derives labels from the stored ``receiver_match_dist``
#: rather than changing it.
LABEL_THRESHOLDS: tuple[float | None, ...] = (4.0, 6.0, 8.0, 10.0, 12.0, None)


def _threshold_masks(rows: Sequence[Row]) -> dict[str, np.ndarray]:
    """The row masks the label audit works from."""
    attacker = np.asarray([r.first_touch == "attacker" for r in rows], dtype=bool)
    valid = np.asarray([r.receiver_valid for r in rows], dtype=bool)
    dist = np.asarray(
        [np.nan if r.receiver_match_dist is None else float(r.receiver_match_dist)
         for r in rows],
        dtype=np.float64,
    )
    return {
        "attacker": attacker,
        "valid": valid,
        "dist": dist,
        # A match exists (some candidate attacker was nearest to the touch), at
        # whatever distance. Attacker-first rows without one have no visible
        # candidate at all and can never be labelled.
        "has_match": attacker & np.isfinite(dist),
        "defence": valid & ~attacker,
    }


def _bracket(correct: float, kept: int, added: int) -> tuple[float, float]:
    """The range a rate could take once ``added`` unscoreable rows are included.

    Low assumes every added row is wrong, high assumes every one is right. There
    is no way to narrow this from the shipped dataset — see
    :func:`label_threshold_sweep`.
    """
    total = kept + added
    if total == 0:
        return float("nan"), float("nan")
    return correct / total, (correct + added) / total


def label_threshold_sweep(
    rows: Sequence[Row],
    cvs: Sequence[CVResult],
    *,
    thresholds: Sequence[float | None] = LABEL_THRESHOLDS,
    shipped: float = RECEIVER_MATCH_MAX_M,
) -> dict[str, Any]:
    """How the receiver metrics move with the receiver-label match threshold.

    ``_io.RECEIVER_MATCH_MAX_M = 8.0`` decides which attacker-first corners get a
    receiver label: the first touch is matched to the nearest candidate attacker
    in the freeze frame, and the row is kept only if that nearest candidate is
    within the threshold. It is a judgement call, so this re-derives the labels at
    other thresholds from the ``receiver_match_dist`` the dataset stores for
    exactly this purpose, and re-scores the *same* out-of-fold predictions.

    Nothing is retrained. Only the evaluation subset moves, so this measures the
    sensitivity of the reported numbers to the threshold, not the sensitivity of
    the fitted models to it. Retraining at each threshold would change the
    training rows too and is a larger experiment than the question asks for.

    **Above 8 m the numbers can only be bracketed, and that is a property of the
    dataset rather than a shortcut.** ``_io`` drops ``receiver_idx`` when the
    match exceeds its own threshold and keeps only the distance, so for a row
    matched at 11 m the dataset records that a match existed and how far away it
    was, but not *which* attacker it was. The label can therefore be tightened
    from the shipped file and only loosened by re-fetching the events. Rows added
    above 8 m are reported as a range: every one wrong at the low end, every one
    right at the high end.

    Averaged over ``cvs`` — one entry per seed — so a threshold row carries the
    same kind of spread as the main table.
    """
    masks = _threshold_masks(rows)
    dist, attacker, defence = masks["dist"], masks["attacker"], masks["defence"]
    has_match = masks["has_match"]
    n_defence = int(defence.sum())
    n_matched_shipped = int((has_match & (dist <= shipped)).sum())

    entries: list[dict[str, Any]] = []
    base: dict[str, dict[str, Any]] | None = None
    for t in thresholds:
        kept_attacker = has_match if t is None else (has_match & (dist <= float(t)))
        n_attacker = int(kept_attacker.sum())
        entry: dict[str, Any] = {
            "threshold": t,
            "n_attacker_first": n_attacker,
            "n_defence": n_defence,
            "n_rows": n_attacker + n_defence,
            "attacker_share_kept": (
                n_attacker / int(has_match.sum()) if has_match.any() else float("nan")
            ),
            "scoreable": t is not None and float(t) <= shipped,
        }
        if entry["scoreable"]:
            keep = defence | kept_attacker
            per_seed = [metrics_from_oof(rows, cv, keep=keep)["receiver"] for cv in cvs]
            entry["metrics"] = {
                model: {
                    key: _aggregate([m[model][key] for m in per_seed])
                    for key in RECEIVER_METRIC_KEYS
                }
                for model in MODEL_KEYS
            }
            if t is not None and float(t) == shipped:
                base = entry["metrics"]
        entries.append(entry)

    # Bracket the thresholds whose labels the shipped dataset cannot reconstruct.
    for entry in entries:
        if entry["scoreable"] or base is None:
            continue
        added = entry["n_attacker_first"] - n_matched_shipped
        entry["added_rows"] = added
        entry["bounds"] = {
            model: {
                "top1_attacker": _bracket(
                    base[model]["top1_attacker"]["mean"] * n_matched_shipped,
                    n_matched_shipped,
                    added,
                ),
                "top3_attacker": _bracket(
                    base[model]["top3_attacker"]["mean"] * n_matched_shipped,
                    n_matched_shipped,
                    added,
                ),
                "overall_accuracy": _bracket(
                    base[model]["overall_accuracy"]["mean"]
                    * (n_matched_shipped + n_defence),
                    n_matched_shipped + n_defence,
                    added,
                ),
            }
            for model in MODEL_KEYS
        }

    finite = dist[np.isfinite(dist)]
    quantiles = (
        {
            str(q): float(np.quantile(finite, q))
            for q in (0.5, 0.75, 0.9, 0.95, 0.99)
        }
        if finite.size
        else {}
    )
    return {
        "shipped_threshold": shipped,
        "entries": entries,
        "n_attacker_first": int(attacker.sum()),
        "n_with_match": int(has_match.sum()),
        "match_distance_quantiles": quantiles,
        "n_seeds": len(cvs),
        "summary": _threshold_summary(entries, quantiles, shipped, n_matched_shipped),
        "table": _threshold_tables(entries, shipped),
        "notes": _threshold_notes(entries, shipped, len(cvs)),
    }


def _threshold_summary(
    entries: Sequence[dict[str, Any]],
    quantiles: dict[str, float],
    shipped: float,
    n_matched_shipped: int,
) -> str:
    at = {e["threshold"]: e for e in entries}
    unlimited = at.get(None)
    tight = next((e for e in entries if e["scoreable"] and e["threshold"] == 4.0), None)
    base = at.get(shipped)
    parts = [
        f"`RECEIVER_MATCH_MAX_M = {shipped}` in the frozen `setpiece/_io.py` decides which "
        "attacker-first corners get a receiver label at all: the first touch is matched to "
        "the nearest candidate attacker in the freeze frame and the row survives only if that "
        "candidate is within the threshold. It was a judgement call and had never been "
        "checked, so the labels here are re-derived at other thresholds from the "
        "`receiver_match_dist` the dataset stores for exactly this purpose, and the *same* "
        "out-of-fold predictions are re-scored against them. Nothing is retrained: this "
        "measures how sensitive the reported numbers are to the threshold, not how sensitive "
        "a refitted model would be."
    ]
    if quantiles:
        parts.append(
            "Match distances are short where they are usable at all — median "
            f"{quantiles['0.5']:.2f} m, 75th percentile {quantiles['0.75']:.2f} m, 90th "
            f"{quantiles['0.9']:.2f} m, 95th {quantiles['0.95']:.2f} m — so the threshold "
            "sits far out in the tail rather than through the middle of the distribution."
        )
    if unlimited is not None and base is not None:
        extra = unlimited["n_attacker_first"] - n_matched_shipped
        parts.append(
            f"Dropping the threshold entirely would add {_plural(extra, 'attacker-first row')} "
            f"to the {n_matched_shipped} that survive at {shipped:g} m "
            f"({extra / max(unlimited['n_attacker_first'], 1):.1%} of every attacker-first "
            "corner with a match), and each of those is a corner whose *nearest visible "
            f"attacker* was more than {shipped:g} m from where the ball was actually touched. "
            "That is not a receiver who happened to be mislocated; it is the closest thing on "
            "camera to an event whose real actor the frame does not show."
        )
    if tight is not None and base is not None:
        cost = base["n_attacker_first"] - tight["n_attacker_first"]
        moves = [
            f"{MODEL_LABELS[k]} {delta:+.3f}"
            for k in MODEL_ORDER
            for delta in [
                tight["metrics"][k]["top1_attacker"]["mean"]
                - base["metrics"][k]["top1_attacker"]["mean"]
            ]
        ]
        parts.append(
            f"Tightening to 4 m costs {_plural(cost, 'attacker-first row')}, leaving "
            f"{tight['n_attacker_first']}, and moves top-1 by "
            + ", ".join(moves[:-1])
            + f" and {moves[-1]}."
        )
    return " ".join(parts)


def _range_cell(low: float, high: float) -> str:
    if math.isnan(low) or math.isnan(high):
        return "n/a"
    # A threshold that adds no rows collapses the bracket; print it as the number
    # it is rather than as a range from a value to itself.
    if abs(high - low) < 5e-4:
        return f"{low:.3f}"
    return f"{low:.3f}-{high:.3f}"


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n} {singular}" if n == 1 else f"{n} {plural or singular + 's'}"


def _threshold_label(t: float | None) -> str:
    return "unthresholded" if t is None else f"{t:g} m"


def _threshold_tables(entries: Sequence[dict[str, Any]], shipped: float) -> str:
    lines = [
        "| threshold | corners with a usable label | attacker-first among them | "
        "share of matchable attacker-first rows kept |",
        "|---|---|---|---|",
    ]
    for e in entries:
        mark = " *(shipped)*" if e["threshold"] == shipped else ""
        lines.append(
            f"| {_threshold_label(e['threshold'])}{mark} | {e['n_rows']} | "
            f"{e['n_attacker_first']} | {e['attacker_share_kept']:.3f} |"
        )
    lines.append("")
    lines.append(
        "| threshold | GNN top-1 | logreg top-1 | heuristic top-1 | GNN top-3 | "
        "logreg top-3 | heuristic top-3 | GNN overall acc. |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for e in entries:
        mark = " *(shipped)*" if e["threshold"] == shipped else ""
        if e["scoreable"]:
            m = e["metrics"]
            cells = [
                f"{m[k]['top1_attacker']['mean']:.3f} ±{m[k]['top1_attacker']['sd']:.3f}"
                for k in MODEL_ORDER
            ] + [
                f"{m[k]['top3_attacker']['mean']:.3f} ±{m[k]['top3_attacker']['sd']:.3f}"
                for k in MODEL_ORDER
            ] + [
                f"{m['gnn']['overall_accuracy']['mean']:.3f} "
                f"±{m['gnn']['overall_accuracy']['sd']:.3f}"
            ]
        elif "bounds" in e:
            b = e["bounds"]
            cells = [_range_cell(*b[k]["top1_attacker"]) for k in MODEL_ORDER]
            cells += [_range_cell(*b[k]["top3_attacker"]) for k in MODEL_ORDER]
            cells += [_range_cell(*b["gnn"]["overall_accuracy"])]
        else:  # pragma: no cover - only when 8 m itself was not swept
            cells = ["n/a"] * 7
        lines.append(f"| {_threshold_label(e['threshold'])}{mark} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _threshold_notes(
    entries: Sequence[dict[str, Any]], shipped: float, n_seeds: int
) -> list[str]:
    notes = [
        f"Every cell is a mean ±sd over the same {n_seeds} repeats as the main table, "
        "re-scoring stored out-of-fold predictions. The models are the ones trained under "
        f"the shipped {shipped:g} m labels throughout; only the rows they are scored on "
        "change.",
        "Rows above the shipped threshold are given as a **range, not an estimate**. "
        "`setpiece/_io.py` keeps `receiver_match_dist` but drops `receiver_idx` once the "
        "match exceeds its own threshold, so for a corner matched at 11 m the dataset "
        "records that a match existed and how far it was, but not which attacker it was. "
        "The range is every added row wrong at the low end and every added row right at the "
        "high end. Loosening the threshold for real needs a re-fetch, not a re-read.",
        "The defence-wins AUC is not swept: loosening the threshold changes the negative "
        "class of that binary problem as well as its size, so a bracket on it would be "
        "vacuous rather than informative.",
    ]
    scoreable = [e for e in entries if e["scoreable"]]
    if len(scoreable) > 1:
        keys = ("top1_attacker", "top3_attacker")
        gaps = [
            max(e["metrics"][k][key]["mean"] for e in scoreable)
            - min(e["metrics"][k][key]["mean"] for e in scoreable)
            for k in MODEL_KEYS
            for key in keys
        ]
        # One distinct ordering per metric means the ranking never changed. A
        # metric that reorders the models contributes two or more.
        pretty = {"top1_attacker": "top-1", "top3_attacker": "top-3"}
        reorders = [
            pretty[key]
            for key in keys
            if len(
                {
                    tuple(sorted(MODEL_KEYS, key=lambda k: -e["metrics"][k][key]["mean"]))
                    for e in scoreable
                }
            )
            > 1
        ]
        labels = ", ".join(_threshold_label(e["threshold"]) for e in scoreable)
        note = (
            f"Across {labels} no model's top-1 or top-3 moves by more than {max(gaps):.3f} "
            "from end to end, so the headline receiver comparison does not rest on the "
            "threshold. "
        )
        if reorders:
            note += (
                "The model ranking is not quite fixed, though: "
                + " and ".join(reorders)
                + " reorders somewhere in that range, by a margin smaller than the spread "
                "beside it — which is a statement about how close the GNN and the logistic "
                "regression are, not about the threshold."
            )
        else:
            note += "The ranking of the three models is identical at every threshold."
        notes.append(note)
    return notes


# --------------------------------------------------------------------------- #
# out-of-fold predictions on disk
# --------------------------------------------------------------------------- #
def oof_signature(
    rows: Sequence[Row], dataset_path: str | Path, folds: int, hp: dict[str, Any]
) -> str:
    """A string identifying what a cached out-of-fold pass was computed from.

    Cached predictions are only reusable if the rows, the fold count, the feature
    layout and every hyperparameter that touches training are unchanged. Storing
    the signature next to the arrays turns "is this cache stale?" from a judgement
    call into an equality test, so a stale cache raises instead of quietly
    reporting numbers from a configuration nobody is running any more.
    """
    keys = ("epochs", "hidden", "layers", "dropout", "lr", "weight_decay",
            "batch_size", "baseline_iters", "baseline_lr", "baseline_weight_decay")
    return json.dumps(
        {
            "dataset": str(dataset_path),
            "n_rows": len(rows),
            "folds": int(folds),
            "n_node_features": int(featmod.N_NODE_FEATURES),
            "n_edge_features": int(featmod.N_EDGE_FEATURES),
            "shot_head": ShotModel(
                featmod.N_NODE_FEATURES, featmod.N_EDGE_FEATURES, hidden=1, layers=1
            ).head,
            "hp": {k: hp[k] for k in keys if k in hp},
        },
        sort_keys=True,
    )


def save_oof(
    rows: Sequence[Row], cv: CVResult, path: str | Path, *, signature: str = ""
) -> Path:
    """Write one pass of out-of-fold predictions to ``.npz``.

    Ragged per-row candidate distributions are stored flat with an offset array.
    The row-level label columns travel with them so an audit script — the
    label-threshold sweep in ``analysis/`` — can re-score these predictions
    against a different definition of a usable label without retraining anything.

    ``signature`` is stored verbatim and checked by :func:`load_oof`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = np.asarray([r.cand_nodes.size for r in rows], dtype=np.int64)
    payload: dict[str, Any] = {
        "seed": np.asarray(cv.seed),
        "signature": np.asarray(signature),
        "splits_json": np.asarray(json.dumps(cv.splits, default=str)),
        "assignment": cv.assignment,
        "cand_offsets": np.concatenate([[0], np.cumsum(lengths)]),
        "target_pos": np.asarray([r.target_pos for r in rows], dtype=np.int64),
        "receiver_valid": np.asarray([r.receiver_valid for r in rows], dtype=bool),
        "y_shot": np.asarray([r.y_shot for r in rows], dtype=np.float64),
        "n_candidates": lengths,
        "match_dist": np.asarray(
            [np.nan if r.receiver_match_dist is None else r.receiver_match_dist for r in rows],
            dtype=np.float64,
        ),
        "first_touch_dt": np.asarray(
            [np.nan if r.first_touch_dt is None else r.first_touch_dt for r in rows],
            dtype=np.float64,
        ),
        "attacker_first": np.asarray([r.first_touch == "attacker" for r in rows], dtype=bool),
    }
    for key in MODEL_KEYS:
        payload[f"cand_{key}"] = (
            np.concatenate([np.asarray(c, dtype=np.float64) for c in cv.cand_probs[key]])
            if rows
            else np.zeros(0, dtype=np.float64)
        )
        payload[f"defence_{key}"] = cv.p_defence[key]
        payload[f"shot_{key}"] = cv.shot[key]
    np.savez_compressed(path, **payload)
    return path


def load_oof(path: str | Path, *, signature: str | None = None) -> CVResult:
    """Rebuild a :class:`CVResult` written by :func:`save_oof`.

    With ``signature``, a cache written from a different dataset, fold count or
    hyperparameter set raises instead of being reused. A silent mismatch here
    would put numbers from an abandoned configuration into `docs/metrics.md`, so
    it is an error rather than a warning.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        stored = str(data["signature"]) if "signature" in data else ""
        if signature is not None and stored != signature:
            raise ValueError(
                f"cached out-of-fold predictions at {path} were computed from a "
                f"different configuration:\n  cached: {stored}\n  wanted: {signature}"
            )
        offsets = data["cand_offsets"]
        cand: dict[str, list[np.ndarray]] = {}
        p_def: dict[str, np.ndarray] = {}
        shot: dict[str, np.ndarray] = {}
        for key in MODEL_KEYS:
            flat = data[f"cand_{key}"]
            cand[key] = [
                np.asarray(flat[offsets[i]:offsets[i + 1]], dtype=np.float64)
                for i in range(len(offsets) - 1)
            ]
            p_def[key] = np.asarray(data[f"defence_{key}"], dtype=np.float64)
            shot[key] = np.asarray(data[f"shot_{key}"], dtype=np.float64)
        return CVResult(
            seed=int(data["seed"]),
            assignment=np.asarray(data["assignment"], dtype=np.int64),
            splits=json.loads(str(data["splits_json"])) if "splits_json" in data else {},
            cand_probs=cand,
            p_defence=p_def,
            shot=shot,
        )


def cached_cross_validate(
    rows: Sequence[Row],
    *,
    dataset_path: str | Path,
    folds: int,
    seed: int,
    hp: dict[str, Any],
    cache_dir: str | Path | None = None,
    log: Any = None,
) -> CVResult:
    """:func:`cross_validate`, reusing a cached pass when one matches.

    A full pass costs five fits per head, and the report is regenerated far more
    often than the predictions change. With ``cache_dir`` each seed's pass is
    written to ``oof-seed<seed>.npz`` under a signature, so re-running the report
    — or computing the seeds in parallel processes and then collecting them — is
    free. Without it, nothing is cached and nothing is written.
    """
    if cache_dir is None:
        return cross_validate(rows, folds=folds, seed=seed, hp=hp, log=log)
    signature = oof_signature(rows, dataset_path, folds, hp)
    path = Path(cache_dir) / f"oof-seed{seed}.npz"
    if path.exists():
        cv = load_oof(path, signature=signature)
        cv.check(len(rows))
        if log is not None:
            log(f"  seed {seed}: reused cached out-of-fold predictions from {path}")
        return cv
    cv = cross_validate(rows, folds=folds, seed=seed, hp=hp, log=log)
    save_oof(rows, cv, path, signature=signature)
    if log is not None:
        log(f"  seed {seed}: wrote {path}")
    return cv


# --------------------------------------------------------------------------- #
# markdown reporting
# --------------------------------------------------------------------------- #
def _fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.3f}"


def _better(models: dict[str, dict[str, float]], key: str) -> tuple[list[str], float]:
    """Every model tied for the best value of ``key``, and that value.

    Ties are returned as ties rather than resolved by table order: silently
    handing a three-way tie to the GNN because it happens to be listed first is
    exactly the kind of flattery this table exists to avoid.
    """
    best: list[str] = []
    best_value = float("-inf")
    for name, metrics in models.items():
        value = metrics.get(key, float("nan"))
        if value is None or math.isnan(value):
            continue
        if value > best_value + 1e-12:
            best, best_value = [name], value
        elif abs(value - best_value) <= 1e-12:
            best.append(name)
    return best, best_value


MODEL_LABELS: dict[str, str] = {
    "gnn": "GNN",
    "heuristic": "heuristic",
    "logreg": "logistic regression",
}
MODEL_ORDER: tuple[str, ...] = ("gnn", "logreg", "heuristic")

#: A gap has to be this many times the spread of its own paired difference before
#: the table calls it real. Two is deliberately modest — it is a readability rule
#: for a five-repeat sample, not a significance test.
DECISIVE_RATIO: float = 2.0


def _agg(report: dict[str, Any], family: str, model: str, key: str) -> dict[str, Any]:
    """The spread entry behind one cell, or a degenerate one if none was recorded."""
    try:
        return report["spread"][family][model][key]
    except (KeyError, TypeError):
        value = float(report[family][model][key])
        return {"mean": value, "sd": 0.0, "min": value, "max": value, "n": 1, "values": [value]}


def _cell(report: dict[str, Any], family: str, model: str, key: str) -> str:
    """``mean ±sd`` across repeats, or a bare mean when there was only one."""
    agg = _agg(report, family, model, key)
    if agg["n"] == 0 or math.isnan(agg["mean"]):
        return "n/a"
    if agg["n"] < 2:
        return f"{agg['mean']:.3f}"
    return f"{agg['mean']:.3f} ±{agg['sd']:.3f}"


def _paired_gap(
    report: dict[str, Any], family: str, key: str, challenger: str, incumbent: str
) -> dict[str, Any] | None:
    """``challenger - incumbent`` seed by seed: the mean gap and its own spread.

    Both models see the identical split within a seed, so differencing them
    inside a seed cancels the split noise that dominates the individual columns.
    A gap is only worth reporting as a result if it is large against the spread
    of *this* difference, which is usually much tighter than the spread of either
    column on its own.
    """
    a = _agg(report, family, key=key, model=challenger)
    b = _agg(report, family, key=key, model=incumbent)
    if a["n"] < 1 or a["n"] != b["n"]:
        return None
    diffs = np.asarray(a["values"], dtype=np.float64) - np.asarray(b["values"], dtype=np.float64)
    if diffs.size == 0:
        return None
    sd = float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0
    mean = float(diffs.mean())
    ratio = abs(mean) / sd if sd > 0 else float("inf")
    return {"mean": mean, "sd": sd, "n": int(diffs.size), "ratio": ratio}


def _describe_gap(gap: dict[str, Any] | None, winner: str, loser: str, pretty: str) -> str:
    if gap is None or gap["n"] < 2:
        return ""
    if math.isinf(gap["ratio"]):
        return " Every repeat agreed, so the sign of this gap is not in doubt."
    verdict = (
        "clear of the noise floor"
        if gap["ratio"] >= DECISIVE_RATIO
        else "**inside the noise floor** and should not be leaned on"
    )
    return (
        f" Seed by seed, {winner} minus {loser} on {pretty} is "
        f"{abs(gap['mean']):.3f} ±{gap['sd']:.3f} over {gap['n']} repeats — "
        f"{gap['ratio']:.1f}x its own spread, {verdict}."
    )


def format_metrics_markdown(report: dict[str, Any]) -> str:
    """Render the full metrics report as markdown."""
    data = report["data"]
    splits = report["splits"]
    receiver = report["receiver"]
    shot = report["shot"]
    params = report["params"]

    lines: list[str] = []
    lines.append("# metrics")
    lines.append("")
    lines.append(
        f"Dataset `{data['path']}` — {data['n_corners']} corners from "
        f"{data['n_matches']} matches"
        + (f", {', '.join(data['competitions'])}" if data["competitions"] else "")
        + f". Trained {report['date']}."
    )
    lines.append("")
    lines.append(
        f"Validation is **purged {splits['n_folds']}-fold cross-validation grouped by "
        "`match_id`**: every corner of a match sits in exactly one fold, so no model is "
        "ever scored on a match it trained on. The numbers below are pooled out-of-fold "
        "predictions — each corner scored exactly once — and both baselines are fitted on "
        "the same training folds and scored on the same held-out rows as the GNN."
    )
    lines.append("")
    lines.append(
        f"Rows per fold: {splits['rows_per_fold']}. Matches per fold: "
        f"{splits['matches_per_fold']}."
    )
    lines.append("")
    seeds = report.get("seeds") or [report.get("training", {}).get("seed", 0)]
    threads = report.get("training", {}).get("torch_threads")
    if len(seeds) > 1:
        lines.append(
            f"**Every figure is a mean over {len(seeds)} repeats of the whole protocol, "
            "written `mean ±sd`** with the sample standard deviation across those repeats. "
            f"Seeds {seeds} each re-draw the fold assignment *and* the network "
            "initialisation, so the spread covers both of the things that change on a "
            "re-run. The two baselines are deterministic given a split, so their spread is "
            "pure split noise; the GNN's carries initialisation noise on top. A difference "
            "smaller than the spread beside it is not a result. Where two models are "
            "compared below, the gap is differenced *within* each repeat first, which "
            "cancels the shared split noise and is a tighter test than eyeballing two "
            "columns."
        )
        lines.append("")
    if threads is not None:
        lines.append(
            f"Run at `torch.set_num_threads({threads})`. That is not decoration: torch's "
            "parallel reductions sum in thread-partition order, so the GNN column is only "
            "reproducible at a pinned thread count, and every checkpoint's card records the "
            "count it was trained at."
        )
        lines.append("")
    if splits.get("n_ungrouped_corners"):
        lines.append(
            f"**Caveat:** {splits['n_ungrouped_corners']} of these corners carry no "
            "`match_id`, so each was treated as a match of its own. Any two of them that "
            "actually came from the same match were *not* purged against each other, and "
            "the match counts above count those corners individually."
        )
        lines.append("")
    lines.append("Freeze frames carry no velocities, so every model here is `assume_static`.")
    lines.append("")

    lines.append("## receiver head")
    lines.append("")
    dropped = int(data["n_corners"]) - int(receiver["gnn"]["n_rows"])
    lines.append(
        f"Scored on {int(receiver['gnn']['n_rows'])} corners with a usable receiver label, of "
        f"which {int(receiver['gnn']['n_attacker_first'])} were won by an attacker. Top-1 and "
        "top-3 rank the candidate attackers against each other on the attacker-first subset, "
        "which is the only question the nearest-to-goal heuristic can answer. Overall accuracy "
        "is the argmax over the candidates *and* the \"defence wins first contact\" class, "
        "across every scored row."
    )
    lines.append("")
    if dropped > 0:
        lines.append(
            f"The other {dropped} corners carry no usable receiver label — almost all of them "
            "attacker-first corners whose receiver was not visible in the freeze frame, so "
            "there is no node to point at. That filter removes attacker-first rows far more "
            "often than defence-wins rows, which leaves the scored subset more defence-heavy "
            "than the dataset as a whole. All three models are scored on exactly these rows, "
            "so the columns compare cleanly against each other, but the overall-accuracy and "
            "defence-AUC figures are not population estimates."
        )
        lines.append("")
    lines.append(
        "| model | top-1 (attacker-first) | top-3 (attacker-first) | "
        "overall acc. incl. defence | defence-wins AUC |"
    )
    lines.append("|---|---|---|---|---|")
    for key in MODEL_ORDER:
        cells = " | ".join(_cell(report, "receiver", key, k) for k in RECEIVER_METRIC_KEYS)
        lines.append(f"| {MODEL_LABELS[key]} | {cells} |")
    lines.append("")
    lines.append(
        f"Chance level over the candidates is {_fmt(receiver['chance']['top1_attacker'])} "
        f"top-1 and {_fmt(receiver['chance']['top3_attacker'])} top-3. Top-3 has a high floor "
        "by construction: a corner with three or fewer candidate attackers is a free hit for "
        "every model, so read the top-3 column against that floor and not against zero. "
        "Always predicting \"defence wins\" would score "
        f"{_fmt(receiver['always_defence_accuracy'])} overall."
    )
    lines.append("")

    lines.append("## shot head")
    lines.append("")
    lines.append(
        f"Binary `shot_within_5`, scored on all {int(shot['gnn']['n_rows'])} held-out corners. "
        f"Base rate {_fmt(shot['gnn']['base_rate'])}. The heuristic is the training-fold shot "
        "rate — a constant within a fold, so it carries no information about an individual "
        "corner. It lands a little *below* 0.5 rather than on it, and that is mechanical "
        "rather than noise: the training-fold rate is a strictly decreasing function of the "
        "held-out fold's own rate, so the score is mildly anti-correlated with the label by "
        "construction. Read it as the floor; the gap between it and 0.5 is not a signal."
    )
    lines.append("")
    lines.append("| model | AUC |")
    lines.append("|---|---|")
    for key in MODEL_ORDER:
        lines.append(f"| {MODEL_LABELS[key]} | {_cell(report, 'shot', key, 'auc')} |")
    lines.append("")

    lines.append("## reading")
    lines.append("")
    for family, key, pretty in (
        ("receiver", "top1_attacker", "top-1 on the attacker-first subset"),
        ("receiver", "top3_attacker", "top-3 on the attacker-first subset"),
        ("receiver", "defence_auc", "AUC for the defence-wins class"),
        ("shot", "auc", "shot AUC"),
    ):
        table = report[family]
        subset = {k: table[k] for k in MODEL_ORDER if k in table}
        winners, value = _better(subset, key)
        if not winners:
            lines.append(f"- {pretty.capitalize()}: undefined on this data.")
            continue
        gnn_value = subset["gnn"].get(key, float("nan"))
        if len(winners) > 1:
            names = ", ".join(MODEL_LABELS[w] for w in winners)
            lines.append(f"- {pretty.capitalize()}: {names} tie at {_fmt(value)}.")
            continue
        runner_up = max(
            (k for k in MODEL_ORDER if k != winners[0] and k in subset),
            key=lambda k: subset[k].get(key, float("-inf")),
        )
        gap = _paired_gap(report, family, key, winners[0], runner_up)
        if winners[0] == "gnn":
            lines.append(
                f"- The GNN wins on {pretty} ({_fmt(value)}), ahead of the "
                f"{MODEL_LABELS[runner_up]} at {_fmt(subset[runner_up].get(key, float('nan')))}."
                + _describe_gap(gap, "the GNN", f"the {MODEL_LABELS[runner_up]}", pretty)
            )
        else:
            lines.append(
                f"- The {MODEL_LABELS[winners[0]]} baseline **beats** the GNN on {pretty}: "
                f"{_fmt(value)} against {_fmt(gnn_value)}."
                + _describe_gap(
                    _paired_gap(report, family, key, winners[0], "gnn"),
                    f"the {MODEL_LABELS[winners[0]]}",
                    "the GNN",
                    pretty,
                )
            )
    lines.append("")
    lines.append(
        f"Parameters: receiver {params['receiver']:,}, shot {params['shot']:,}."
    )
    lines.append("")

    lines.extend(_calibration_section(report))
    lines.extend(_threshold_section(report))
    return "\n".join(lines)


def _calibration_section(report: dict[str, Any]) -> list[str]:
    """The calibration part of the table, or a line saying it was not measured."""
    cal = report.get("calibration")
    if not cal:
        return []
    lines = ["## calibration", ""]
    lines.append(
        "Two of the numbers this project emits are genuine binary probabilities and can be "
        "checked against observed frequencies: the shot head's sigmoid, and the receiver "
        "head's `P(defence wins first contact)`. The per-attacker receiver probabilities are "
        "not in that family — they are a distribution over a variable-length candidate set — "
        "so they are ranked, not calibrated. Expected calibration error (ECE) below is the "
        "count-weighted mean gap between predicted and observed frequency over "
        f"{CALIBRATION_BINS} bins of the pooled out-of-fold predictions, reported under both "
        "binnings: equal-width bins are the textbook definition, and equal-count bins are the "
        "honest view for a head whose predictions all sit inside a narrow band, where "
        "equal-width bins drop every row into two buckets and report a flatteringly small "
        "number. Calibration is a monotone rescaling, so **every AUC above is identical "
        "before and after it**: this section changes what a number means, never how well it "
        "ranks."
    )
    lines.append("")

    measured = cal.get("measured", {})
    lines.append(
        "| head | rows | base rate | mean predicted | predicted range | ECE raw | "
        "ECE calibrated | ECE raw (equal-count) | ECE calibrated (equal-count) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for head, label in (("shot", "shot"), ("defence", "defence wins first contact")):
        entry = measured.get(head)
        if not entry:
            continue
        lines.append(
            f"| {label} | {entry['n']} | {_fmt(entry['base_rate'])} | "
            f"{_fmt(entry['mean_prediction'])} | "
            f"{_fmt(entry['min_prediction'])}-{_fmt(entry['max_prediction'])} | "
            f"{_fmt(entry['ece_raw'])} | {_fmt(entry['ece_calibrated'])} | "
            f"{_fmt(entry.get('ece_raw_quantile'))} | "
            f"{_fmt(entry.get('ece_calibrated_quantile'))} |"
        )
    lines.append("")
    for head, label in (("shot", "shot"), ("defence", "defence-wins")):
        entry = measured.get(head)
        if not entry:
            continue
        verdict = (
            "so no calibrator ships for it"
            if entry["chosen"] == "identity"
            else f"so a `{entry['chosen']}` map ships with it"
        )
        lines.append(
            f"- The raw {label} head spans "
            f"{_fmt(entry['min_prediction'])}-{_fmt(entry['max_prediction'])} with a mean of "
            f"{_fmt(entry['mean_prediction'])} against a base rate of "
            f"{_fmt(entry['base_rate'])}. Raw ECE {_fmt(entry['ece_raw'])}, Brier "
            f"{_fmt(entry.get('brier_raw'))}; the best nested calibrator reaches "
            f"{_fmt(entry['ece_calibrated'])}, {verdict}."
        )
    lines.append("")
    for head, label in (("shot", "shot"), ("defence", "defence wins first contact")):
        entry = measured.get(head)
        if not entry or not entry.get("curve"):
            continue
        lines.append(
            format_reliability_markdown(
                entry["curve"],
                title=f"reliability of the raw {label} head, "
                f"{CALIBRATION_BINS} equal-count bins",
            )
        )
        lines.append("")
    lines.append(
        "The calibrated column is measured **nested**: for each fold the calibrator was "
        "fitted on scores from an extra model that had seen neither that fold nor the fold "
        "the calibrator was fitted on, so no held-out row ever contributed to the map applied "
        "to it. The one approximation, stated rather than hidden: those extra models were "
        "trained on three fifths of the data instead of four fifths, and a model trained on "
        "less data is a little less confident, so this understates rather than overstates how "
        "much shrinking the shipped head needs."
    )
    lines.append("")
    lines.append(
        "One consequence of measuring it that way, since it looks like a contradiction "
        "otherwise: the *nested* column stitches together a different calibrator per fold, and "
        "two rows in different folds can swap order across that seam, so the nested column is "
        "not AUC-preserving. What ships is a single global map, and a single monotone map "
        "cannot reorder anything — which is why the AUC tables above stand unchanged."
    )
    lines.append("")
    shipped = cal.get("shipped", {})
    if shipped:
        files = {"shot": "`setpiece/assets/shot.pt`", "defence": "`setpiece/assets/receiver.pt`"}
        carried = [h for h, e in shipped.items() if e["kind"] != "identity"]
        bare = [h for h, e in shipped.items() if e["kind"] == "identity"]
        if not carried:
            lines.append(
                "**The shipped checkpoints are uncalibrated, and deliberately so:** no "
                "calibrator improved held-out ECE on either head, so neither checkpoint ships "
                "one. The probabilities they emit are ranking scores."
            )
        else:
            listed = " and ".join(
                f"{files[h]} stores a `{shipped[h]['kind']}` map for its "
                f"{'shot' if h == 'shot' else 'defence-wins'} probability"
                for h in carried
            )
            lines.append(
                f"**A calibrator ships inside the checkpoint it belongs to.** {listed}, under "
                "`card[\"calibration\"]`, refitted on the full pooled out-of-fold predictions "
                "once the nested measurement had chosen it. "
                "`setpiece.calibration.calibrator_from_card` rebuilds one."
            )
            if bare:
                lines.append("")
                lines.append(
                    "The remaining head — "
                    + ", ".join("shot" if h == "shot" else "defence-wins" for h in bare)
                    + " — ships **no** calibrator, because no candidate improved its held-out "
                    "ECE. That is a measurement, not an omission."
                )
        lines.append("")
        lines.append(
            "Calibrators are **not** applied automatically. `score`, `suggest`, `demo` and the "
            "app all print the raw sigmoid, so a probability from the CLI today is a ranking "
            "score with a documented correction available beside it, not a calibrated "
            "probability."
        )
        lines.append("")
    return lines


def _threshold_section(report: dict[str, Any]) -> list[str]:
    """The receiver-label threshold audit, when an analysis run has attached one."""
    sweep = report.get("label_threshold")
    if not sweep:
        return []
    lines = ["## receiver label threshold", "", sweep["summary"], ""]
    lines.append(sweep["table"])
    lines.append("")
    for note in sweep.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# the training entry point
# --------------------------------------------------------------------------- #
def train_all(
    dataset_path: str | Path,
    out_dir: str | Path,
    *,
    folds: int = DEFAULTS["folds"],
    epochs: int = DEFAULTS["epochs"],
    hidden: int = DEFAULTS["hidden"],
    layers: int = DEFAULTS["layers"],
    dropout: float = DEFAULTS["dropout"],
    lr: float = DEFAULTS["lr"],
    weight_decay: float = DEFAULTS["weight_decay"],
    batch_size: int = DEFAULTS["batch_size"],
    seed: int = DEFAULTS["seed"],
    seeds: int = DEFAULTS["seeds"],
    baseline_iters: int = DEFAULTS["baseline_iters"],
    limit: int | None = None,
    metrics_path: str | Path | None = None,
    quiet: bool = False,
    install: bool = False,
    calibrate: bool = True,
    sweep_threshold: bool = True,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Cross-validate both heads, evaluate the baselines, then fit and save.

    Runs purged group k-fold on ``dataset_path`` ``seeds`` times over — each
    repeat re-drawing both the fold assignment and the initialisation — pools the
    out-of-fold predictions of the GNN and of both baselines, and reports every
    figure as a mean with the spread across repeats. It writes a markdown metrics
    table to stdout and to ``metrics_path`` (``docs/metrics.md`` by default),
    then refits each head on the whole dataset and saves ``receiver.pt`` and
    ``shot.pt`` into ``out_dir`` with a fully populated model card.

    With ``calibrate``, both binary heads are additionally measured against
    observed frequencies, a calibrator is chosen on a nested protocol that costs
    three extra fits per head, and the chosen map travels in the model cards. It
    is on by default and can be turned off for a smoke run.

    With ``install``, the two refitted checkpoints are also copied over the
    bundled ones in ``setpiece/assets/``, which is what ``score``, ``suggest``,
    ``demo`` and the app load by default.

    Returns the report dict: ``data``, ``splits``, ``receiver``, ``shot``,
    ``spread``, ``seeds``, ``calibration``, ``label_threshold``, ``params``,
    ``paths``, ``date``, ``training``.
    """
    set_seed(seed)
    dataset_path = Path(dataset_path)
    out_dir = Path(out_dir)
    rows = load_rows(dataset_path, limit=limit)

    def log(message: str) -> None:
        if not quiet:
            print(message, flush=True)

    hp = {
        "epochs": epochs,
        "hidden": hidden,
        "layers": layers,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "baseline_iters": baseline_iters,
        "baseline_lr": DEFAULTS["baseline_lr"],
        "baseline_weight_decay": DEFAULTS["baseline_weight_decay"],
    }
    seed_values = seed_list(seed, seeds)
    log(
        f"{len(rows)} corners, {len({r.group for r in rows})} matches, {folds} purged folds, "
        f"{len(seed_values)} repeat(s) at seeds {seed_values}, "
        f"torch.get_num_threads()={torch.get_num_threads()}"
    )

    passes: list[CVResult] = []
    per_seed: list[dict[str, Any]] = []
    for s in seed_values:
        cv = cached_cross_validate(
            rows,
            dataset_path=dataset_path,
            folds=folds,
            seed=s,
            hp=hp,
            cache_dir=cache_dir,
            log=log,
        )
        passes.append(cv)
        per_seed.append(metrics_from_oof(rows, cv))

    means, spread = summarise_seeds(per_seed)
    receiver_report = means["receiver"]
    shot_report = means["shot"]
    primary = passes[0]
    splits = primary.splits
    n_folds = splits["n_folds"]

    calibration: dict[str, Any] | None = None
    if calibrate:
        log("measuring calibration of both binary heads")
        in_sample = calibration_from_oof(rows, primary)
        nested = nested_calibration(rows, primary, hp=hp, log=log)
        shipped_cal = fit_shipped_calibrators(rows, primary, nested)
        calibration = {
            "bins": CALIBRATION_BINS,
            "seed": primary.seed,
            "measured": summarise_calibration(in_sample, nested, shipped_cal),
            "nested": nested,
            "in_sample": in_sample,
            "shipped": shipped_cal,
        }

    label_threshold = (
        label_threshold_sweep(rows, passes) if sweep_threshold else None
    )

    y_shot = np.asarray([r.y_shot for r in rows], dtype=np.float64)
    receiver_ids = [i for i in range(len(rows)) if rows[i].receiver_valid]

    log("fitting final models on the full dataset")
    final_receiver = _train_receiver(
        rows,
        epochs=epochs,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
    )
    final_shot = _train_shot(
        rows,
        epochs=epochs,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
    )

    data_summary = {
        "path": str(dataset_path),
        "n_corners": len(rows),
        "n_matches": splits["n_matches"],
        "n_receiver_rows": len(receiver_ids),
        "competitions": sorted({r.competition for r in rows if r.competition}),
        "seasons": sorted({r.season for r in rows if r.season}),
        "shot_base_rate": float(y_shot.mean()),
    }
    training = {
        "folds": n_folds,
        "epochs": epochs,
        "hidden": hidden,
        "layers": layers,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "seed": seed,
        "seeds": seed_values,
        "baseline_iters": baseline_iters,
        "optimiser": "adam",
        "final_fit": "refit on every corner after cross-validation",
        # Reproducing these numbers exactly needs the thread count too; see set_seed.
        "torch_threads": int(torch.get_num_threads()),
    }
    report: dict[str, Any] = {
        "date": date.today().isoformat(),
        "data": data_summary,
        "splits": splits,
        "receiver": receiver_report,
        "shot": shot_report,
        "spread": spread,
        "seeds": seed_values,
        "per_seed": per_seed_summary(per_seed),
        "training": training,
        "params": {
            "receiver": count_parameters(final_receiver),
            "shot": count_parameters(final_shot),
        },
    }
    if calibration is not None:
        report["calibration"] = calibration
    if label_threshold is not None:
        report["label_threshold"] = label_threshold

    feature_card = {
        "node_features": list(featmod.FEATURE_NAMES),
        "edge_features": list(featmod.EDGE_FEATURE_NAMES),
        "n_node_features": featmod.N_NODE_FEATURES,
        "n_edge_features": featmod.N_EDGE_FEATURES,
        "excluded": ["meta.end_xy", "meta.pass_outcome"],
        "note": "positional only; freeze frames carry no velocities",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    # Each checkpoint carries the calibrator for its own binary head, and only
    # that one: the receiver head's calibratable probability is "the defence wins
    # first contact", the shot head's is the shot sigmoid.
    card_calibration = {"receiver": "defence", "shot": "shot"}
    for name, model, metrics in (
        ("receiver", final_receiver, receiver_report),
        ("shot", final_shot, shot_report),
    ):
        card = {
            "name": name,
            "data": data_summary,
            "splits": splits,
            "metrics": metrics,
            "metrics_spread": spread[name],
            "seeds": seed_values,
            "date": report["date"],
            "features": feature_card,
            "params": count_parameters(model),
            "training": training,
            "assume_static": True,
        }
        if calibration is not None:
            head = card_calibration[name]
            card["calibration"] = {head: calibration["shipped"][head]}
        if label_threshold is not None and name == "receiver":
            card["label_threshold"] = {
                "shipped_threshold_m": label_threshold["shipped_threshold"],
                "entries": [
                    {k: v for k, v in e.items() if k != "metrics"}
                    for e in label_threshold["entries"]
                ],
            }
        path = out_dir / f"{name}.pt"
        save_model(model, path, card)
        paths[name] = str(path)
    report["paths"] = paths

    markdown = format_metrics_markdown(report)
    print(markdown)
    if metrics_path is not None:
        target = Path(metrics_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown)
        report["paths"]["metrics"] = str(target)
        log(f"wrote {target}")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    report["paths"]["report"] = str(out_dir / "report.json")
    # The primary pass's raw predictions, so the label audit and any later
    # re-scoring can be redone without retraining a thing.
    report["paths"]["oof"] = str(
        save_oof(
            rows,
            primary,
            out_dir / "oof.npz",
            signature=oof_signature(rows, dataset_path, folds, hp),
        )
    )

    if install:
        from setpiece.models import ASSETS_DIR

        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        installed = {}
        for name in ("receiver", "shot"):
            target = ASSETS_DIR / f"{name}.pt"
            shutil.copyfile(out_dir / f"{name}.pt", target)
            installed[name] = str(target)
            log(f"installed {target}")
        report["paths"]["installed"] = installed

    return report


# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #
def add_train_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach every training flag to ``parser``. Used by :mod:`setpiece.cli`."""
    parser.add_argument(
        "--dataset",
        default="data/interim/corners.jsonl",
        help="corners JSONL to train on (default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/latest",
        help="where to write receiver.pt, shot.pt and report.json (default: %(default)s)",
    )
    parser.add_argument(
        "--metrics-path",
        default="docs/metrics.md",
        help="markdown metrics table destination (default: %(default)s)",
    )
    parser.add_argument("--no-metrics-file", action="store_true", help="print the table only")
    parser.add_argument(
        "--install",
        action="store_true",
        help="also copy the refitted checkpoints over the bundled ones in setpiece/assets/",
    )
    parser.add_argument("--folds", type=int, default=DEFAULTS["folds"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--hidden", type=int, default=DEFAULTS["hidden"])
    parser.add_argument("--layers", type=int, default=DEFAULTS["layers"])
    parser.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULTS["seeds"],
        help="repeat the whole cross-validation this many times and report mean ±sd "
        "(default: %(default)s). GNN training is only reproducible at a fixed thread "
        "count, so a single run quoted to three decimals overstates its own precision.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="reuse and write per-seed out-of-fold predictions here, so repeats can be "
        "precomputed in parallel (see analysis/precompute_oof.py) and the report "
        "regenerated without retraining",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="skip the calibration measurement; it costs three extra fits per head",
    )
    parser.add_argument(
        "--no-threshold-sweep",
        action="store_true",
        help="skip the receiver-label match-threshold audit",
    )
    parser.add_argument("--baseline-iters", type=int, default=DEFAULTS["baseline_iters"])
    parser.add_argument("--limit", type=int, default=None, help="use only the first N corners")
    parser.add_argument("--quiet", action="store_true", help="suppress progress lines")
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Standalone parser for ``python -m setpiece.train``."""
    parser = argparse.ArgumentParser(
        prog="setpiece train",
        description="Cross-validate and fit the corner receiver and shot heads.",
    )
    return add_train_arguments(parser)


def run_from_args(args: argparse.Namespace) -> int:
    """Run :func:`train_all` from a parsed namespace. Returns a process exit code."""
    metrics_path = None if getattr(args, "no_metrics_file", False) else args.metrics_path
    train_all(
        args.dataset,
        args.out_dir,
        folds=args.folds,
        epochs=args.epochs,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
        seeds=getattr(args, "seeds", DEFAULTS["seeds"]),
        baseline_iters=args.baseline_iters,
        limit=args.limit,
        metrics_path=metrics_path,
        quiet=getattr(args, "quiet", False),
        install=getattr(args, "install", False),
        calibrate=not getattr(args, "no_calibrate", False),
        sweep_threshold=not getattr(args, "no_threshold_sweep", False),
        cache_dir=getattr(args, "cache_dir", None),
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
