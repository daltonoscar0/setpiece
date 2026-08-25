"""Graph featuriser for corner freeze frames.

Turns a :class:`~setpiece.types.CornerFrame` into either a
:class:`torch_geometric.data.Data` graph (for the GNN) or a flat ``[N, F]``
matrix of the same per-node features (for the logistic-regression baseline).

Freeze frames carry no velocities, so every feature here is positional. Nothing
in this module reads ``frame.meta["end_xy"]`` or ``frame.meta["pass_outcome"]``:
both are label-adjacent and are diagnostics only.

Receiver target and batching
----------------------------
``y_receiver`` is a **node index**, and it is stored **graph-local**: it indexes
into that corner's own nodes, and it is *not* shifted when graphs are collated.
``CornerData.__inc__`` pins the increment to ``0`` so PyG leaves it alone.

That is deliberate. PyG's collation applies to graph ``b`` the *cumulative sum
of the increments of graphs 0..b-1*, so the increment cannot depend on graph
``b``'s own value — a "shift real indices but leave ``-1`` alone" rule is
impossible to express through ``__inc__``, and half-implementing it silently
turns the ``-1`` defence-wins sentinel into a valid-looking node index.

So, for ``batch = Batch.from_data_list([g0, g1, ...])``:

* ``batch.y_receiver[b] >= 0``  -> a node index **local to graph b**; the
  batch-global index is ``batch.ptr[b] + batch.y_receiver[b]``;
* ``batch.y_receiver[b] == -1`` -> the defence-wins class, unshifted;
* ``batch.to_data_list()`` returns exactly the per-graph values.

Use :func:`global_receiver_index` to do the offsetting; it works on a single
graph and on a batch. Rows with ``y_receiver_valid == False`` must be dropped
from the receiver loss.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from setpiece.types import (
    GOAL_X,
    GOAL_Y,
    PITCH_LENGTH,
    PITCH_WIDTH,
    POST_LEFT_Y,
    POST_RIGHT_Y,
    CornerFrame,
    Player,
)

# --------------------------------------------------------------------------- #
# graph construction knobs
# --------------------------------------------------------------------------- #
#: Nearest neighbours per node before symmetrisation.
K_NEIGHBOURS: int = 8

#: Penalty area and six-yard box, in metres from the goal being attacked.
_PENALTY_DEPTH: float = 16.5
_PENALTY_HALF_WIDTH: float = 20.16
_SIX_YARD_DEPTH: float = 5.5
_SIX_YARD_HALF_WIDTH: float = 9.16

# Scales chosen so that typical values land in roughly [-1, 1].
_GOAL_DIST_SCALE: float = 30.0
_TAKER_DIST_SCALE: float = 50.0
_NEIGHBOUR_DIST_SCALE: float = 20.0
_EDGE_DIST_SCALE: float = 20.0
_DENSITY_3M_SCALE: float = 3.0
_DENSITY_5M_SCALE: float = 5.0
_RIGHT_ANGLE: float = math.pi / 2.0

_NEAR_RADIUS: float = 3.0
_MID_RADIUS: float = 5.0

FEATURE_NAMES: list[str] = [
    "x_norm",  # x / 105
    "y_norm",  # y / 68
    "dx_goal",  # (goal_x - x) / 30
    "dy_goal",  # (y - goal_y) / 30, signed: positive is the far side
    "dist_goal",  # |p - goal| / 30
    "goal_angle_cos",  # direction to the goal centre, cyclic-safe encoding
    "goal_angle_sin",
    "post_angle",  # angle subtended by the two posts, / (pi/2)
    "dist_taker",  # |p - taker| / 50
    "taker_angle_cos",  # direction to the taker
    "taker_angle_sin",
    "is_attacker",
    "is_taker",
    "is_keeper",
    "n_within_3m",  # other players within 3 m, / 3
    "n_within_5m",  # other players within 5 m, / 5
    "n_opponents_within_3m",  # opponents within 3 m, / 3
    "dist_nearest_opponent",  # min(d, 20) / 20
    "dist_nearest_teammate",  # min(d, 20) / 20
    "in_penalty_area",
    "in_six_yard_box",
]

EDGE_FEATURE_NAMES: list[str] = [
    "dx",  # (x_dst - x_src) / 20
    "dy",  # (y_dst - y_src) / 20
    "dist",  # |dst - src| / 20
    "inv_dist",  # 1 / (1 + |dst - src|)
    "same_team",
    "taker_edge",  # either endpoint is the corner taker
]

N_NODE_FEATURES: int = len(FEATURE_NAMES)
N_EDGE_FEATURES: int = len(EDGE_FEATURE_NAMES)


# --------------------------------------------------------------------------- #
# Data subclass: makes y_receiver survive batching
# --------------------------------------------------------------------------- #
class CornerData(Data):
    """``Data`` whose ``y_receiver`` is a graph-local node index.

    The increment is pinned to ``0`` and the concat dimension to ``0``, so
    batching concatenates one target per graph and changes no values. See the
    module docstring for why the target is not shifted into batch-global space.
    """

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "y_receiver":
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "y_receiver":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


def global_receiver_index(data: Data) -> torch.Tensor:
    """Batch-global receiver node index, ``int64 [B]``.

    Adds each graph's node offset to its local ``y_receiver``; the ``-1``
    defence-wins sentinel passes through untouched. On a single unbatched graph
    this is the identity, because the offset is zero.
    """
    y = data.y_receiver.reshape(-1)
    ptr = getattr(data, "ptr", None)
    offset = torch.zeros_like(y) if ptr is None else ptr[:-1].to(dtype=y.dtype, device=y.device)
    return torch.where(y >= 0, y + offset, y)


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _positions(players: list[Player]) -> np.ndarray:
    return np.asarray([[p.x, p.y] for p in players], dtype=np.float64).reshape(-1, 2)


def _pairwise_distances(pos: np.ndarray) -> np.ndarray:
    diff = pos[:, None, :] - pos[None, :, :]
    return np.hypot(diff[:, :, 0], diff[:, :, 1])


def _post_angles(pos: np.ndarray) -> np.ndarray:
    """Angle in radians subtended at each position by the two goalposts."""
    left = np.stack([GOAL_X - pos[:, 0], POST_LEFT_Y - pos[:, 1]], axis=1)
    right = np.stack([GOAL_X - pos[:, 0], POST_RIGHT_Y - pos[:, 1]], axis=1)
    cross = np.abs(left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0])
    dot = left[:, 0] * right[:, 0] + left[:, 1] * right[:, 1]
    return np.arctan2(cross, dot)


def taker_position(frame: CornerFrame) -> tuple[float, float]:
    """Where the corner is taken from.

    Normally the freeze frame flags the taker. User-supplied frames sometimes do
    not, in which case we fall back to the corner flag nearest to any player —
    a geometric inference, never an invented node.
    """
    idx = frame.taker_index
    if idx is not None:
        p = frame.players[idx]
        return (p.x, p.y)
    flags = [(PITCH_LENGTH, 0.0), (PITCH_LENGTH, PITCH_WIDTH)]
    if not frame.players:
        return flags[0]
    pos = _positions(frame.players)
    best = min(
        flags,
        key=lambda f: float(np.min(np.hypot(pos[:, 0] - f[0], pos[:, 1] - f[1]))),
    )
    return best


# --------------------------------------------------------------------------- #
# node features
# --------------------------------------------------------------------------- #
def _node_features(frame: CornerFrame) -> np.ndarray:
    players = frame.players
    n = len(players)
    feats = np.zeros((n, N_NODE_FEATURES), dtype=np.float64)
    if n == 0:
        return feats.astype(np.float32)

    pos = _positions(players)
    x = pos[:, 0]
    y = pos[:, 1]
    is_attacker = np.asarray([p.is_attacker for p in players], dtype=bool)
    is_taker = np.asarray([p.is_taker for p in players], dtype=bool)
    is_keeper = np.asarray([p.is_keeper for p in players], dtype=bool)

    gdx = GOAL_X - x
    gdy = GOAL_Y - y
    goal_dist = np.hypot(gdx, gdy)
    goal_angle = np.arctan2(gdy, gdx)

    tx, ty = taker_position(frame)
    tdx = tx - x
    tdy = ty - y
    taker_dist = np.hypot(tdx, tdy)
    taker_angle = np.arctan2(tdy, tdx)

    dist = _pairwise_distances(pos)
    off_diagonal = ~np.eye(n, dtype=bool)
    near = (dist <= _NEAR_RADIUS) & off_diagonal
    mid = (dist <= _MID_RADIUS) & off_diagonal
    opponent = is_attacker[:, None] != is_attacker[None, :]
    teammate = (is_attacker[:, None] == is_attacker[None, :]) & off_diagonal

    big = np.float64(1e9)
    opp_dist = np.where(opponent, dist, big)
    team_dist = np.where(teammate, dist, big)
    nearest_opp = np.minimum(opp_dist.min(axis=1), _NEIGHBOUR_DIST_SCALE)
    nearest_team = np.minimum(team_dist.min(axis=1), _NEIGHBOUR_DIST_SCALE)

    in_box = (x >= GOAL_X - _PENALTY_DEPTH) & (np.abs(y - GOAL_Y) <= _PENALTY_HALF_WIDTH)
    in_six = (x >= GOAL_X - _SIX_YARD_DEPTH) & (np.abs(y - GOAL_Y) <= _SIX_YARD_HALF_WIDTH)

    feats[:, 0] = x / PITCH_LENGTH
    feats[:, 1] = y / PITCH_WIDTH
    feats[:, 2] = gdx / _GOAL_DIST_SCALE
    feats[:, 3] = -gdy / _GOAL_DIST_SCALE
    feats[:, 4] = goal_dist / _GOAL_DIST_SCALE
    feats[:, 5] = np.cos(goal_angle)
    feats[:, 6] = np.sin(goal_angle)
    feats[:, 7] = _post_angles(pos) / _RIGHT_ANGLE
    feats[:, 8] = taker_dist / _TAKER_DIST_SCALE
    feats[:, 9] = np.cos(taker_angle)
    feats[:, 10] = np.sin(taker_angle)
    feats[:, 11] = is_attacker
    feats[:, 12] = is_taker
    feats[:, 13] = is_keeper
    feats[:, 14] = near.sum(axis=1) / _DENSITY_3M_SCALE
    feats[:, 15] = mid.sum(axis=1) / _DENSITY_5M_SCALE
    feats[:, 16] = (near & opponent).sum(axis=1) / _DENSITY_3M_SCALE
    feats[:, 17] = nearest_opp / _NEIGHBOUR_DIST_SCALE
    feats[:, 18] = nearest_team / _NEIGHBOUR_DIST_SCALE
    feats[:, 19] = in_box
    feats[:, 20] = in_six
    return feats.astype(np.float32)


# --------------------------------------------------------------------------- #
# edges
# --------------------------------------------------------------------------- #
def _adjacency(frame: CornerFrame, dist: np.ndarray) -> np.ndarray:
    """k-NN adjacency, symmetrised, plus every edge to and from the taker."""
    n = dist.shape[0]
    adj = np.zeros((n, n), dtype=bool)
    if n < 2:
        return adj

    k = min(K_NEIGHBOURS, n - 1)
    masked = dist.copy()
    np.fill_diagonal(masked, np.inf)
    # stable ordering: ties broken by node index, which argsort(kind="stable") gives
    order = np.argsort(masked, axis=1, kind="stable")[:, :k]
    rows = np.repeat(np.arange(n), k)
    adj[rows, order.reshape(-1)] = True

    taker = frame.taker_index
    if taker is not None:
        adj[taker, :] = True
        adj[:, taker] = True

    adj |= adj.T
    np.fill_diagonal(adj, False)
    return adj


def _edge_features(
    frame: CornerFrame, pos: np.ndarray, dist: np.ndarray, edge_index: np.ndarray
) -> np.ndarray:
    e = edge_index.shape[1]
    attr = np.zeros((e, N_EDGE_FEATURES), dtype=np.float64)
    if e == 0:
        return attr.astype(np.float32)

    src = edge_index[0]
    dst = edge_index[1]
    is_attacker = np.asarray([p.is_attacker for p in frame.players], dtype=bool)
    is_taker = np.asarray([p.is_taker for p in frame.players], dtype=bool)
    d = dist[src, dst]

    attr[:, 0] = (pos[dst, 0] - pos[src, 0]) / _EDGE_DIST_SCALE
    attr[:, 1] = (pos[dst, 1] - pos[src, 1]) / _EDGE_DIST_SCALE
    attr[:, 2] = d / _EDGE_DIST_SCALE
    attr[:, 3] = 1.0 / (1.0 + d)
    attr[:, 4] = is_attacker[src] == is_attacker[dst]
    attr[:, 5] = is_taker[src] | is_taker[dst]
    return attr.astype(np.float32)


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def _candidate_mask(frame: CornerFrame) -> np.ndarray:
    """Attackers, excluding the taker."""
    return np.asarray(
        [p.is_attacker and not p.is_taker for p in frame.players], dtype=bool
    ).reshape(-1)


def _receiver_target(frame: CornerFrame, candidates: np.ndarray) -> tuple[int, bool]:
    """``(y_receiver, y_receiver_valid)``.

    * ``first_touch == "defender"``  -> ``(-1, True)``: "the defence wins" is a
      real class, not a missing label.
    * ``first_touch == "attacker"`` with a matched receiver -> ``(idx, True)``.
    * ``first_touch == "attacker"`` with ``receiver_idx is None`` -> the
      receiver was off camera, so ``(-1, False)`` and the row is dropped.
    * unlabelled / ``"unknown"`` -> ``(-1, False)``.
    """
    touch = frame.first_touch
    if touch == "defender":
        return -1, bool(candidates.any())
    if touch == "attacker":
        idx = frame.receiver_idx
        if idx is None or not (0 <= idx < len(frame.players)) or not bool(candidates[idx]):
            return -1, False
        return int(idx), True
    return -1, False


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def build_flat(frame: CornerFrame) -> np.ndarray:
    """Per-node feature matrix, ``float32 [N, N_NODE_FEATURES]``.

    Identical to ``build_graph(frame).x`` as a numpy array; used by the flat
    baseline so the GNN and the baseline see the same information per node.
    """
    return _node_features(frame)


def build_graph(frame: CornerFrame) -> CornerData:
    """Featurise one corner into a graph.

    Works on an unlabelled frame: then ``y_receiver = -1``,
    ``y_receiver_valid = False`` and ``y_shot = 0.0``.
    """
    n = len(frame.players)
    pos = _positions(frame.players)
    dist = _pairwise_distances(pos) if n else np.zeros((0, 0))

    node_x = _node_features(frame)
    adj = _adjacency(frame, dist)
    src, dst = np.nonzero(adj)
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    edge_attr = _edge_features(frame, pos, dist, edge_index)

    candidates = _candidate_mask(frame)
    y_receiver, y_valid = _receiver_target(frame, candidates)

    data = CornerData(
        x=torch.from_numpy(node_x),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
    )
    data.is_attacker = torch.tensor([p.is_attacker for p in frame.players], dtype=torch.bool)
    data.is_taker = torch.tensor([p.is_taker for p in frame.players], dtype=torch.bool)
    data.is_keeper = torch.tensor([p.is_keeper for p in frame.players], dtype=torch.bool)
    data.candidate_mask = torch.from_numpy(candidates)
    data.y_receiver = torch.tensor([y_receiver], dtype=torch.int64)
    data.y_receiver_valid = torch.tensor([y_valid], dtype=torch.bool)
    data.y_shot = torch.tensor([1.0 if frame.shot_within_5 else 0.0], dtype=torch.float32)
    data.corner_id = frame.corner_id
    data.num_nodes = n
    return data
