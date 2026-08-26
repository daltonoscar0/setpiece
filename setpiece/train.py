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
from setpiece._io import read_corners
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
    "Row",
    "add_train_arguments",
    "build_parser",
    "load_rows",
    "main",
    "purged_fold_assignment",
    "receiver_metrics",
    "roc_auc",
    "run_from_args",
    "set_seed",
    "shot_metrics",
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
}

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
        m = receiver[key]
        lines.append(
            f"| {MODEL_LABELS[key]} | {_fmt(m['top1_attacker'])} | {_fmt(m['top3_attacker'])} "
            f"| {_fmt(m['overall_accuracy'])} | {_fmt(m['defence_auc'])} |"
        )
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
        lines.append(f"| {MODEL_LABELS[key]} | {_fmt(shot[key]['auc'])} |")
    lines.append("")

    lines.append("## reading")
    lines.append("")
    for label, table, key, pretty in (
        ("receiver top-1", receiver, "top1_attacker", "top-1 on the attacker-first subset"),
        ("receiver defence AUC", receiver, "defence_auc", "AUC for the defence-wins class"),
        ("shot", shot, "auc", "shot AUC"),
    ):
        subset = {k: table[k] for k in MODEL_ORDER if k in table}
        winners, value = _better(subset, key)
        if not winners:
            lines.append(f"- {label}: undefined on this data.")
            continue
        gnn_value = subset["gnn"].get(key, float("nan"))
        if len(winners) > 1:
            names = ", ".join(MODEL_LABELS[w] for w in winners)
            lines.append(f"- {pretty.capitalize()}: {names} tie at {_fmt(value)}.")
        elif winners[0] == "gnn":
            lines.append(f"- The GNN wins on {pretty} ({_fmt(value)}).")
        else:
            lines.append(
                f"- The {MODEL_LABELS[winners[0]]} baseline **beats** the GNN on {pretty}: "
                f"{_fmt(value)} against {_fmt(gnn_value)}."
            )
    lines.append("")
    lines.append(
        f"Parameters: receiver {params['receiver']:,}, shot {params['shot']:,}."
    )
    lines.append("")
    return "\n".join(lines)


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
    baseline_iters: int = DEFAULTS["baseline_iters"],
    limit: int | None = None,
    metrics_path: str | Path | None = None,
    quiet: bool = False,
    install: bool = False,
) -> dict[str, Any]:
    """Cross-validate both heads, evaluate the baselines, then fit and save.

    Runs purged group k-fold on ``dataset_path``, pools the out-of-fold
    predictions of the GNN and of both baselines, writes a markdown metrics table
    to stdout and to ``metrics_path`` (``docs/metrics.md`` by default), then
    refits each head on the whole dataset and saves ``receiver.pt`` and
    ``shot.pt`` into ``out_dir`` with a fully populated model card.

    With ``install``, the two refitted checkpoints are also copied over the
    bundled ones in ``setpiece/assets/``, which is what ``score``, ``suggest``,
    ``demo`` and the app load by default.

    Returns the report dict: ``data``, ``splits``, ``receiver``, ``shot``,
    ``params``, ``paths``, ``date``, ``training``.
    """
    set_seed(seed)
    dataset_path = Path(dataset_path)
    out_dir = Path(out_dir)
    rows = load_rows(dataset_path, limit=limit)

    def log(message: str) -> None:
        if not quiet:
            print(message, flush=True)

    assignment = purged_fold_assignment([r.group for r in rows], folds, seed=seed)
    n_folds = int(assignment.max()) + 1
    splits = _split_summary(rows, assignment)
    log(
        f"{len(rows)} corners, {splits['n_matches']} matches, {n_folds} purged folds "
        f"(rows per fold {splits['rows_per_fold']})"
    )

    n = len(rows)
    receiver_ids = [i for i in range(n) if rows[i].receiver_valid]
    oof_receiver: dict[str, dict[int, tuple[np.ndarray, float]]] = {
        "gnn": {},
        "heuristic": {},
        "logreg": {},
    }
    oof_shot: dict[str, np.ndarray] = {
        key: np.full(n, np.nan, dtype=np.float64) for key in ("gnn", "heuristic", "logreg")
    }

    for fold in range(n_folds):
        test_idx = np.nonzero(assignment == fold)[0]
        train_idx = np.nonzero(assignment != fold)[0]
        train_rows = [rows[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]
        test_recv_idx = [i for i in test_idx if rows[i].receiver_valid]
        test_recv_rows = [rows[i] for i in test_recv_idx]
        log(f"fold {fold + 1}/{n_folds}: train {len(train_rows)}, test {len(test_rows)}")

        receiver_model = _train_receiver(
            train_rows,
            epochs=epochs,
            hidden=hidden,
            layers=layers,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            seed=seed + fold,
        )
        shot_model = _train_shot(
            train_rows,
            epochs=epochs,
            hidden=hidden,
            layers=layers,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            seed=seed + fold,
        )

        train_valid = [r for r in train_rows if r.receiver_valid]
        prior = (
            float(np.mean([r.target_pos < 0 for r in train_valid])) if train_valid else 0.5
        )
        shot_rate = float(np.mean([r.y_shot for r in train_rows])) if train_rows else 0.5
        logreg_recv = _fit_receiver_logreg(
            train_rows,
            iters=baseline_iters,
            lr=DEFAULTS["baseline_lr"],
            weight_decay=DEFAULTS["baseline_weight_decay"],
            seed=seed + fold,
        )
        logreg_shot = _fit_shot_logreg(
            train_rows,
            iters=baseline_iters,
            lr=DEFAULTS["baseline_lr"],
            weight_decay=DEFAULTS["baseline_weight_decay"],
            seed=seed + fold,
        )

        preds: dict[str, tuple[list[np.ndarray], np.ndarray]] = {
            "gnn": _predict_receiver_gnn(receiver_model, test_recv_rows, batch_size),
            "heuristic": _heuristic_receiver(test_recv_rows, prior),
            "logreg": logreg_recv.predict(test_recv_rows),
        }
        for name, (cand_probs, p_def) in preds.items():
            # The table is only a comparison if all three models are scored on
            # exactly these rows in exactly this order. A predictor returning a
            # longer list would otherwise misalign silently and never raise.
            if len(cand_probs) != len(test_recv_idx) or len(p_def) != len(test_recv_idx):
                raise RuntimeError(
                    f"{name} returned {len(cand_probs)} candidate distributions and "
                    f"{len(p_def)} defence probabilities for {len(test_recv_idx)} held-out "
                    "receiver rows; every model must be scored on the identical rows"
                )
            for slot, row_id in enumerate(test_recv_idx):
                oof_receiver[name][row_id] = (cand_probs[slot], float(p_def[slot]))

        oof_shot["gnn"][test_idx] = _predict_shot_gnn(shot_model, test_rows, batch_size)
        oof_shot["heuristic"][test_idx] = shot_rate
        oof_shot["logreg"][test_idx] = logreg_shot.predict(test_rows)

    target_pos = np.asarray([rows[i].target_pos for i in receiver_ids], dtype=np.int64)
    receiver_report: dict[str, Any] = {}
    for name in ("gnn", "heuristic", "logreg"):
        cand_probs = [oof_receiver[name][i][0] for i in receiver_ids]
        p_def = np.asarray([oof_receiver[name][i][1] for i in receiver_ids], dtype=np.float64)
        receiver_report[name] = receiver_metrics(cand_probs, p_def, target_pos)

    n_cands = np.asarray([rows[i].cand_nodes.size for i in receiver_ids], dtype=np.float64)
    attacker_first = target_pos >= 0
    receiver_report["chance"] = {
        "top1_attacker": float(np.mean(1.0 / n_cands[attacker_first]))
        if attacker_first.any()
        else float("nan"),
        "top3_attacker": float(np.mean(np.minimum(3.0, n_cands[attacker_first]) /
                                       n_cands[attacker_first]))
        if attacker_first.any()
        else float("nan"),
    }
    receiver_report["always_defence_accuracy"] = (
        float(np.mean(target_pos < 0)) if target_pos.size else float("nan")
    )

    y_shot = np.asarray([r.y_shot for r in rows], dtype=np.float64)
    shot_report = {name: shot_metrics(y_shot, oof_shot[name]) for name in oof_shot}

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
        "training": training,
        "params": {
            "receiver": count_parameters(final_receiver),
            "shot": count_parameters(final_shot),
        },
    }

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
    for name, model, metrics in (
        ("receiver", final_receiver, receiver_report),
        ("shot", final_shot, shot_report),
    ):
        card = {
            "name": name,
            "data": data_summary,
            "splits": splits,
            "metrics": metrics,
            "date": report["date"],
            "features": feature_card,
            "params": count_parameters(model),
            "training": training,
            "assume_static": True,
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
        baseline_iters=args.baseline_iters,
        limit=args.limit,
        metrics_path=metrics_path,
        quiet=getattr(args, "quiet", False),
        install=getattr(args, "install", False),
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
