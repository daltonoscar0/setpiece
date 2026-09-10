"""Graph neural networks for corner-kick receiver and shot prediction.

Both heads share the same backbone: :class:`CornerEncoder`, three rounds of
message passing over the freeze-frame graph produced by
:func:`setpiece.features.build_graph`. Each head owns its own encoder instance
so the two checkpoints are independent; each stays comfortably under 1M
parameters and scores a single corner on CPU in a few milliseconds.

Two things in here are easy to get wrong and are therefore spelled out.

**The receiver distribution is per graph.** The receiver head emits one logit
per node plus one "the defence wins first contact" logit per graph. The valid
distribution is a softmax over *that graph's* candidate nodes together with
*that graph's* defence logit. Under batching the node logits of every graph are
concatenated into a single vector, so a plain ``softmax`` over that vector mixes
graphs together and is wrong. :func:`receiver_log_softmax` does the segment-wise
normalisation and is what training and inference must both use.

**Freeze frames carry no velocities.** Nothing here invents any; every model
card written by :func:`save_model` records ``assume_static: True``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.functional import relu
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

from setpiece.types import CornerFrame

__all__ = [
    "ASSETS_DIR",
    "CornerEncoder",
    "ReceiverModel",
    "ShotModel",
    "load_bundled",
    "load_model",
    "receiver_log_softmax",
    "receiver_probabilities",
    "save_model",
    "shot_probability",
]

#: Where :func:`load_bundled` looks for trained checkpoints.
ASSETS_DIR: Path = Path(__file__).resolve().parent / "assets"

#: Checkpoint layout version, bumped if the on-disk structure ever changes.
CHECKPOINT_FORMAT: int = 1


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _feature_dims() -> tuple[int, int]:
    """Node and edge feature widths, read from :mod:`setpiece.features`."""
    from setpiece import features

    return int(features.N_NODE_FEATURES), int(features.N_EDGE_FEATURES)


def _batch_vector(data: Any) -> Tensor:
    """Graph assignment per node; a vector of zeros for a single unbatched graph."""
    batch = getattr(data, "batch", None)
    if batch is None:
        return torch.zeros(int(data.num_nodes), dtype=torch.long, device=data.x.device)
    return batch


def _graph_count(data: Any, batch: Tensor) -> int:
    n = getattr(data, "num_graphs", None)
    if n is not None:
        return int(n)
    return int(batch.max().item()) + 1 if batch.numel() else 1


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


def _segment_log_softmax(src: Tensor, index: Tensor, n_segments: int) -> Tensor:
    """Numerically stable log-softmax within each segment of ``index``.

    ``src`` may contain ``-inf`` entries (masked-out nodes); those come back as
    ``-inf`` with zero gradient. Every segment must contain at least one finite
    entry, which the per-graph defence logit guarantees.
    """
    peak = scatter(src, index, dim=0, dim_size=n_segments, reduce="max")
    centred = src - peak[index]
    total = scatter(centred.exp(), index, dim=0, dim_size=n_segments, reduce="sum")
    return centred - total[index].log()


def _masked_mean(h: Tensor, batch: Tensor, n_graphs: int, mask: Tensor) -> Tensor:
    """Mean of ``h`` over the masked nodes of each graph; zeros where empty."""
    weight = mask.to(h.dtype).unsqueeze(-1)
    total = scatter(h * weight, batch, dim=0, dim_size=n_graphs, reduce="sum")
    count = scatter(weight, batch, dim=0, dim_size=n_graphs, reduce="sum")
    return total / count.clamp(min=1.0)


# --------------------------------------------------------------------------- #
# message passing
# --------------------------------------------------------------------------- #
class CornerConv(MessagePassing):
    """One round of edge-conditioned message passing with a residual update.

    Messages are built from the source embedding, the target embedding and the
    edge features, mean-aggregated over the neighbourhood, then folded back into
    the target node through a gated residual and a layer norm.
    """

    def __init__(self, hidden: int, edge_dim: int, dropout: float = 0.0) -> None:
        super().__init__(aggr="mean", node_dim=0)
        self.message_mlp = _mlp(2 * hidden + edge_dim, hidden, hidden, dropout)
        self.update_mlp = _mlp(2 * hidden, hidden, hidden, dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        aggregated = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        delta = self.update_mlp(torch.cat([x, aggregated], dim=-1))
        return self.norm(x + delta)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return relu(self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1)))


class CornerEncoder(nn.Module):
    """Three rounds of message passing over a corner freeze-frame graph.

    Parameters
    ----------
    node_dim, edge_dim:
        Input feature widths. Default to ``setpiece.features.N_NODE_FEATURES``
        and ``N_EDGE_FEATURES``; pass them explicitly to build a model without
        importing the featuriser.
    hidden:
        Embedding width.
    layers:
        Message passing rounds. Three, per the architecture contract.
    """

    def __init__(
        self,
        node_dim: int | None = None,
        edge_dim: int | None = None,
        *,
        hidden: int = 64,
        layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if node_dim is None or edge_dim is None:
            default_node, default_edge = _feature_dims()
            node_dim = default_node if node_dim is None else node_dim
            edge_dim = default_edge if edge_dim is None else edge_dim

        self.node_dim_in = int(node_dim)
        self.edge_dim_in = int(edge_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)

        self.node_encoder = _mlp(self.node_dim_in, hidden, hidden, dropout)
        self.edge_encoder = _mlp(self.edge_dim_in, hidden, hidden, dropout)
        self.convs = nn.ModuleList(
            CornerConv(hidden, hidden, dropout) for _ in range(self.layers)
        )

    @property
    def init_kwargs(self) -> dict[str, Any]:
        return {
            "node_dim": self.node_dim_in,
            "edge_dim": self.edge_dim_in,
            "hidden": self.hidden,
            "layers": self.layers,
            "dropout": self.dropout,
        }

    def forward(self, data: Any) -> Tensor:
        """Node embeddings ``[N_total, hidden]``."""
        x = data.x
        if x.size(-1) != self.node_dim_in:
            raise ValueError(
                f"node feature width {x.size(-1)} does not match this encoder's "
                f"{self.node_dim_in}; the checkpoint and setpiece.features disagree"
            )
        edge_attr = data.edge_attr
        if edge_attr.size(-1) != self.edge_dim_in:
            raise ValueError(
                f"edge feature width {edge_attr.size(-1)} does not match this "
                f"encoder's {self.edge_dim_in}"
            )

        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        for conv in self.convs:
            h = conv(h, data.edge_index, e)
        return h


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #
class _GraphReadout(nn.Module):
    """Pool node embeddings into one vector per graph.

    Concatenates a mean and a max over all nodes with separate means over the
    attackers and the defenders, so the head sees both sides of the box rather
    than a single blended average.
    """

    out_scale: int = 4

    def forward(self, h: Tensor, data: Any, batch: Tensor, n_graphs: int) -> Tensor:
        mean_all = scatter(h, batch, dim=0, dim_size=n_graphs, reduce="mean")
        max_all = scatter(h, batch, dim=0, dim_size=n_graphs, reduce="max")
        is_attacker = data.is_attacker.to(torch.bool)
        mean_att = _masked_mean(h, batch, n_graphs, is_attacker)
        mean_def = _masked_mean(h, batch, n_graphs, ~is_attacker)
        return torch.cat([mean_all, max_all, mean_att, mean_def], dim=-1)


class ReceiverModel(nn.Module):
    """Who makes first contact with the delivery.

    The output is one logit per node plus one "defence wins first contact"
    logit per graph. Non-candidate nodes (defenders, and the taker) are masked
    to ``-inf`` so a softmax over a graph's candidates together with its own
    defence logit is a valid distribution. Use :func:`receiver_log_softmax` to
    perform that normalisation — it is segment-wise and batch safe.
    """

    def __init__(
        self,
        node_dim: int | None = None,
        edge_dim: int | None = None,
        *,
        hidden: int = 64,
        layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = CornerEncoder(
            node_dim, edge_dim, hidden=hidden, layers=layers, dropout=dropout
        )
        self.readout = _GraphReadout()
        self.node_head = _mlp(hidden, hidden, 1, dropout)
        self.defence_head = _mlp(self.readout.out_scale * hidden, hidden, 1, dropout)

    @property
    def init_kwargs(self) -> dict[str, Any]:
        return dict(self.encoder.init_kwargs)

    def forward(self, data: Any) -> tuple[Tensor, Tensor]:
        """Return ``(node_logits [N_total], defence_logit [B])``."""
        batch = _batch_vector(data)
        n_graphs = _graph_count(data, batch)

        h = self.encoder(data)
        raw = self.node_head(h).squeeze(-1)
        candidate = data.candidate_mask.to(torch.bool)
        neg_inf = torch.full_like(raw, float("-inf"))
        node_logits = torch.where(candidate, raw, neg_inf)

        pooled = self.readout(h, data, batch, n_graphs)
        defence_logit = self.defence_head(pooled).squeeze(-1)
        return node_logits, defence_logit


class ShotModel(nn.Module):
    """Binary head for ``shot_within_5``: one logit per graph.

    The readout feeds a **single linear layer**, not a two-layer MLP. That is a
    measured choice, not a stylistic one. ``shot_within_5`` has a 0.275 base rate
    over 2464 corners, so the head sees roughly 677 positives; a hidden layer on
    top of a ``4 * hidden`` pooled vector adds ~16k parameters that the label
    cannot pay for, and the extra capacity showed up as variance rather than
    skill.

    Selected on an inner protocol carved out of the training folds only — five
    inner validation splits grouped by ``match_id``, five seeds, 25 fits per
    configuration, never an outer test fold. Mean AUC over the inner validation
    splits, and the spread across seeds:

    ========================  =========  ===============
    shot head                 inner AUC  across 5 seeds
    ========================  =========  ===============
    two-layer MLP (previous)  0.5629     0.5459 - 0.5782
    single linear (this)      0.5741     0.5650 - 0.5831
    ========================  =========  ===============

    **The logistic-regression baseline still wins the shot head.** On the
    reporting protocol — purged 5-fold by ``match_id``, pooled out-of-fold, the
    identical rows — over four seeds at one thread:

    ========================  ===============================  ======
    shot head                 out-of-fold AUC per seed         mean
    ========================  ===============================  ======
    two-layer MLP (previous)  0.5569 0.5662 0.5688 0.5762      0.5670
    single linear (this)      0.5735 0.5771 0.5801 0.5886      0.5798
    logistic regression       0.5935 (deterministic)           0.5935
    ========================  ===============================  ======

    So the change buys the GNN +0.0128 AUC and closes about half of a 0.027
    deficit; it does not overturn it. Nothing here licenses a claim that the GNN
    predicts shots better than the pooled linear model.

    The other measured effect is variance: the seed-to-seed spread of the inner
    score drops from 0.032 to 0.018, which matters because these numbers are only
    reproducible at a fixed thread count.

    ``head`` selects between the two, and it is recorded in :attr:`init_kwargs`
    so every checkpoint states which one it was trained with. Checkpoints written
    before the head became configurable carry the MLP; see :func:`load_model`.
    """

    #: Readout-to-logit heads this class can build.
    HEADS: tuple[str, ...] = ("linear", "mlp")

    def __init__(
        self,
        node_dim: int | None = None,
        edge_dim: int | None = None,
        *,
        hidden: int = 64,
        layers: int = 3,
        dropout: float = 0.1,
        head: str = "linear",
    ) -> None:
        super().__init__()
        if head not in self.HEADS:
            raise ValueError(f"unknown shot head {head!r}; expected one of {self.HEADS}")
        self.encoder = CornerEncoder(
            node_dim, edge_dim, hidden=hidden, layers=layers, dropout=dropout
        )
        self.readout = _GraphReadout()
        self.head = head
        width = self.readout.out_scale * hidden
        self.shot_head: nn.Module = (
            nn.Linear(width, 1) if head == "linear" else _mlp(width, hidden, 1, dropout)
        )

    @property
    def init_kwargs(self) -> dict[str, Any]:
        return {**self.encoder.init_kwargs, "head": self.head}

    def forward(self, data: Any) -> Tensor:
        """Return the shot logit ``[B]``."""
        batch = _batch_vector(data)
        n_graphs = _graph_count(data, batch)
        h = self.encoder(data)
        pooled = self.readout(h, data, batch, n_graphs)
        return self.shot_head(pooled).squeeze(-1)


# --------------------------------------------------------------------------- #
# receiver normalisation
# --------------------------------------------------------------------------- #
def receiver_log_softmax(
    node_logits: Tensor,
    defence_logit: Tensor,
    batch: Tensor,
    n_graphs: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Normalise receiver logits per graph.

    Each graph's candidate node logits and its own defence logit form one
    distribution. Returns ``(node_log_probs [N_total], defence_log_prob [B])``;
    for every graph ``g`` the exponentials of ``node_log_probs[batch == g]``
    plus ``defence_log_prob[g]`` sum to one. Masked nodes come back as ``-inf``,
    i.e. probability exactly zero.

    A plain ``log_softmax`` over the concatenated node logits would normalise
    across graph boundaries and is not a valid substitute.
    """
    if n_graphs is None:
        n_graphs = int(defence_logit.numel())
    if defence_logit.numel() != n_graphs:
        raise ValueError(
            f"expected one defence logit per graph, got {defence_logit.numel()} for "
            f"{n_graphs} graphs"
        )
    combined = torch.cat([node_logits, defence_logit])
    index = torch.cat(
        [batch, torch.arange(n_graphs, device=node_logits.device, dtype=torch.long)]
    )
    log_probs = _segment_log_softmax(combined, index, n_graphs)
    split = int(node_logits.numel())
    return log_probs[:split], log_probs[split:]


# --------------------------------------------------------------------------- #
# single-corner inference
# --------------------------------------------------------------------------- #
def _build_graph(frame: CornerFrame) -> Any:
    from setpiece.features import build_graph

    data = build_graph(frame)
    if int(data.num_nodes) != len(frame.players):
        raise ValueError(
            f"build_graph produced {int(data.num_nodes)} nodes for a frame with "
            f"{len(frame.players)} players; node i must correspond to players[i]"
        )
    return data


class _EvalMode:
    """Context manager that runs a module in eval mode and restores its state."""

    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.was_training = module.training

    def __enter__(self) -> nn.Module:
        self.module.eval()
        return self.module

    def __exit__(self, *exc: object) -> None:
        self.module.train(self.was_training)


@torch.no_grad()
def receiver_probabilities(model: ReceiverModel, frame: CornerFrame) -> tuple[np.ndarray, float]:
    """Score one corner's receiver distribution.

    Returns ``(probs, p_defence)`` where ``probs`` has one entry per player in
    ``frame.players`` — exactly ``0.0`` at every non-candidate — and
    ``p_defence`` is the probability that the defence wins first contact.
    ``probs.sum() + p_defence == 1.0``.
    """
    data = _build_graph(frame)
    with _EvalMode(model):
        node_logits, defence_logit = model(data)
    batch = _batch_vector(data)
    node_log_probs, defence_log_prob = receiver_log_softmax(node_logits, defence_logit, batch, 1)

    probs = node_log_probs.exp().to(torch.float64).cpu().numpy()
    p_defence = float(defence_log_prob.exp().to(torch.float64).item())

    out = np.zeros(len(frame.players), dtype=np.float64)
    candidate = data.candidate_mask.to(torch.bool).cpu().numpy()
    out[candidate] = probs[candidate]

    # Renormalise away float32 accumulation so the contract sum(probs) + p_defence
    # == 1.0 holds exactly for callers that assert on it.
    total = out.sum() + p_defence
    if total > 0.0:
        out /= total
        p_defence /= total
    return out, p_defence


@torch.no_grad()
def shot_probability(model: ShotModel, frame: CornerFrame) -> float:
    """Probability that the attacking team shoots within five events."""
    data = _build_graph(frame)
    with _EvalMode(model):
        logit = model(data)
    return float(torch.sigmoid(logit.reshape(-1)[0]).item())


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, type[nn.Module]] = {
    "CornerEncoder": CornerEncoder,
    "ReceiverModel": ReceiverModel,
    "ShotModel": ShotModel,
}

#: Keys the architecture contract requires in a model card.
CARD_KEYS: tuple[str, ...] = (
    "data",
    "splits",
    "metrics",
    "date",
    "features",
    "params",
    "assume_static",
)


def count_parameters(model: nn.Module) -> int:
    """Total number of parameters, trainable or not."""
    return sum(p.numel() for p in model.parameters())


def save_model(model: nn.Module, path: str | Path, card: dict) -> None:
    """Write a checkpoint carrying the weights, the recipe, and the model card.

    The checkpoint records the class name and the constructor kwargs so
    :func:`load_model` can rebuild the module without the caller knowing any
    shapes. The card is stored as JSON, which keeps the payload loadable under
    ``weights_only=True``.
    """
    name = type(model).__name__
    if name not in _REGISTRY:
        raise ValueError(f"{name} is not a registered setpiece model: {sorted(_REGISTRY)}")
    kwargs = getattr(model, "init_kwargs", None)
    if kwargs is None:
        raise ValueError(f"{name} does not expose init_kwargs, so it cannot be reconstructed")

    card = dict(card)
    card.setdefault("params", count_parameters(model))
    card.setdefault("assume_static", True)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "class": name,
        "kwargs_json": json.dumps(dict(kwargs), sort_keys=True),
        "card_json": json.dumps(card, default=str, sort_keys=True),
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    }
    torch.save(payload, path)


def load_model(path: str | Path) -> tuple[nn.Module, dict]:
    """Rebuild a model and its card from a checkpoint. The model is in eval mode."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Older or hand-written checkpoints may hold non-tensor objects.
        payload = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict) or "class" not in payload:
        raise ValueError(f"{path} is not a setpiece checkpoint")
    name = payload["class"]
    if name not in _REGISTRY:
        raise ValueError(f"checkpoint {path} names unknown model class {name!r}")

    kwargs = json.loads(payload.get("kwargs_json", "{}"))
    card = json.loads(payload.get("card_json", "{}"))
    if name == "ShotModel" and "head" not in kwargs:
        # The shot head used to be a two-layer MLP and was not configurable, so
        # it was not recorded either. Every checkpoint missing the key therefore
        # holds the MLP head, the absence is what dates it, not a guess, and
        # rebuilding it with today's default would fail on the state dict.
        kwargs["head"] = "mlp"
    model = _REGISTRY[name](**kwargs)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, card


def load_bundled(name: str) -> tuple[nn.Module, dict]:
    """Load a checkpoint shipped in ``setpiece/assets/``. Names: ``receiver``, ``shot``."""
    known = ("receiver", "shot")
    if name not in known:
        raise ValueError(f"unknown bundled model {name!r}; expected one of {known}")
    path = ASSETS_DIR / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"no bundled '{name}' model at {path}. Train one first:\n"
            f"    uv run setpiece train\n"
            f"which writes {ASSETS_DIR / (name + '.pt')}, or pass an explicit "
            f"checkpoint with --model and load it via load_model()."
        )
    return load_model(path)
