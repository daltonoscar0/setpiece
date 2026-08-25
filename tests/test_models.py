"""Tests for the corner GNN heads.

Graphs are built by hand here rather than through :mod:`setpiece.features`, so
the model contract is tested independently of the featuriser. The handful of
tests that genuinely need real data (single-corner inference, latency) skip
themselves when the featuriser or the smoke dataset is absent.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from setpiece._io import read_corners
from setpiece.models import (
    CornerEncoder,
    ReceiverModel,
    ShotModel,
    count_parameters,
    load_bundled,
    load_model,
    receiver_log_softmax,
    receiver_probabilities,
    save_model,
    shot_probability,
)

try:  # written concurrently; these tests must pass with or without it
    from setpiece import features as sp_features
except Exception:  # pragma: no cover - depends on repo state
    sp_features = None

needs_features = pytest.mark.skipif(
    sp_features is None, reason="setpiece.features is not available yet"
)

REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / "data" / "interim" / "smoke.jsonl"

NODE_DIM = 13
EDGE_DIM = 4
PARAM_BUDGET = 1_000_000


# --------------------------------------------------------------------------- #
# hand-built graphs
# --------------------------------------------------------------------------- #
def make_data(n_nodes: int, n_candidates: int, *, seed: int = 0) -> Data:
    """A fully connected graph shaped like ``features.build_graph`` output.

    Nodes ``0 .. n_candidates-1`` are candidate attackers, node ``n_candidates``
    is the taker, the rest are defenders (the last one is the keeper).
    """
    assert n_nodes > n_candidates + 1
    gen = torch.Generator().manual_seed(seed)

    idx = torch.arange(n_nodes)
    src = idx.repeat_interleave(n_nodes)
    dst = idx.repeat(n_nodes)
    keep = src != dst
    edge_index = torch.stack([src[keep], dst[keep]])

    is_attacker = torch.zeros(n_nodes, dtype=torch.bool)
    is_attacker[: n_candidates + 1] = True
    is_taker = torch.zeros(n_nodes, dtype=torch.bool)
    is_taker[n_candidates] = True
    is_keeper = torch.zeros(n_nodes, dtype=torch.bool)
    is_keeper[-1] = True

    return Data(
        x=torch.randn(n_nodes, NODE_DIM, generator=gen),
        edge_index=edge_index,
        edge_attr=torch.randn(edge_index.size(1), EDGE_DIM, generator=gen),
        is_attacker=is_attacker,
        is_taker=is_taker,
        is_keeper=is_keeper,
        candidate_mask=is_attacker & ~is_taker,
        y_receiver=torch.tensor([0], dtype=torch.long),
        y_receiver_valid=torch.tensor([True]),
        y_shot=torch.tensor([0.0], dtype=torch.float32),
    )


@pytest.fixture
def graphs() -> list[Data]:
    """Three graphs of deliberately different sizes and candidate counts."""
    return [
        make_data(11, 4, seed=1),
        make_data(17, 7, seed=2),
        make_data(22, 9, seed=3),
    ]


def receiver_model(**kw: Any) -> ReceiverModel:
    torch.manual_seed(0)
    model = ReceiverModel(NODE_DIM, EDGE_DIM, **kw)
    model.eval()
    return model


def shot_model(**kw: Any) -> ShotModel:
    torch.manual_seed(0)
    model = ShotModel(NODE_DIM, EDGE_DIM, **kw)
    model.eval()
    return model


def real_corners() -> list:
    if not SMOKE.exists():
        pytest.skip(f"no smoke dataset at {SMOKE}")
    corners = read_corners(SMOKE)
    if not corners:
        pytest.skip("smoke dataset is empty")
    return corners


# --------------------------------------------------------------------------- #
# architecture
# --------------------------------------------------------------------------- #
def test_encoder_does_three_rounds_of_message_passing() -> None:
    enc = CornerEncoder(NODE_DIM, EDGE_DIM)
    assert enc.layers == 3
    assert len(enc.convs) == 3


def test_encoder_returns_node_embeddings() -> None:
    enc = CornerEncoder(NODE_DIM, EDGE_DIM, hidden=64)
    enc.eval()
    data = make_data(15, 6)
    with torch.no_grad():
        h = enc(data)
    assert h.shape == (15, 64)
    assert torch.isfinite(h).all()


def test_parameter_budget_per_head() -> None:
    for model in (receiver_model(), shot_model()):
        n = count_parameters(model)
        assert 0 < n < PARAM_BUDGET, f"{type(model).__name__} has {n} parameters"


def test_encoder_rejects_mismatched_feature_width() -> None:
    enc = CornerEncoder(NODE_DIM, EDGE_DIM)
    data = make_data(12, 5)
    data.x = torch.randn(12, NODE_DIM + 1)
    with pytest.raises(ValueError, match="node feature width"):
        enc(data)


def test_three_hops_of_information_actually_propagate() -> None:
    """A change three hops away must move a node's embedding."""
    enc = CornerEncoder(NODE_DIM, EDGE_DIM)
    enc.eval()
    # path graph 0 - 1 - 2 - 3, both directions
    pairs = [(0, 1), (1, 2), (2, 3)]
    src = [a for a, b in pairs] + [b for a, b in pairs]
    dst = [b for a, b in pairs] + [a for a, b in pairs]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    data = Data(
        x=torch.zeros(4, NODE_DIM),
        edge_index=edge_index,
        edge_attr=torch.ones(edge_index.size(1), EDGE_DIM),
        is_attacker=torch.tensor([True, True, False, False]),
        candidate_mask=torch.tensor([True, True, False, False]),
    )
    with torch.no_grad():
        before = enc(data)[0].clone()
        data.x = data.x.clone()
        data.x[3] = 5.0
        after = enc(data)[0]
    assert not torch.allclose(before, after, atol=1e-6)


# --------------------------------------------------------------------------- #
# receiver head: masking and the per-graph softmax
# --------------------------------------------------------------------------- #
def test_non_candidate_nodes_are_masked_to_negative_infinity() -> None:
    model = receiver_model()
    data = make_data(18, 7)
    with torch.no_grad():
        node_logits, defence_logit = model(data)

    cand = data.candidate_mask
    assert node_logits.shape == (18,)
    assert defence_logit.shape == (1,)
    assert torch.isneginf(node_logits[~cand]).all()
    assert torch.isfinite(node_logits[cand]).all()
    assert torch.isfinite(defence_logit).all()


def test_receiver_distribution_is_per_graph_not_global(graphs: list[Data]) -> None:
    """The batched softmax must normalise within each graph, not across the batch."""
    model = receiver_model()
    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        node_logits, defence_logit = model(batch)

    assert node_logits.shape == (batch.num_nodes,)
    assert defence_logit.shape == (3,)

    node_lp, def_lp = receiver_log_softmax(node_logits, defence_logit, batch.batch, 3)
    node_p = node_lp.exp()
    def_p = def_lp.exp()

    for g in range(3):
        sel = batch.batch == g
        total = float(node_p[sel].sum() + def_p[g])
        assert total == pytest.approx(1.0, abs=1e-5), f"graph {g} sums to {total}"
        assert float(node_p[sel & ~batch.candidate_mask].sum()) == 0.0

    # the whole batch therefore sums to B, which a naive global softmax cannot do
    assert float(node_p.sum() + def_p.sum()) == pytest.approx(3.0, abs=1e-5)

    naive = torch.softmax(torch.cat([node_logits, defence_logit]), dim=0)
    assert float(naive.sum()) == pytest.approx(1.0, abs=1e-5)
    assert not torch.allclose(naive[: node_logits.numel()], node_p, atol=1e-4)


def test_batched_receiver_matches_one_graph_at_a_time(graphs: list[Data]) -> None:
    model = receiver_model()
    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        node_logits, defence_logit = model(batch)
    node_lp, def_lp = receiver_log_softmax(node_logits, defence_logit, batch.batch, 3)

    offset = 0
    for g, data in enumerate(graphs):
        with torch.no_grad():
            single_nodes, single_def = model(data)
        single_lp, single_def_lp = receiver_log_softmax(
            single_nodes, single_def, torch.zeros(data.num_nodes, dtype=torch.long), 1
        )
        chunk = node_lp[offset : offset + data.num_nodes]
        cand = data.candidate_mask
        assert torch.allclose(chunk[cand], single_lp[cand], atol=1e-5)
        assert float(def_lp[g]) == pytest.approx(float(single_def_lp[0]), abs=1e-5)
        offset += data.num_nodes
    assert offset == batch.num_nodes


def test_receiver_log_softmax_matches_a_hand_rolled_reference(graphs: list[Data]) -> None:
    model = receiver_model()
    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        node_logits, defence_logit = model(batch)
    node_lp, def_lp = receiver_log_softmax(node_logits, defence_logit, batch.batch, 3)

    offset = 0
    for g, data in enumerate(graphs):
        cand_logits = node_logits[offset : offset + data.num_nodes][data.candidate_mask]
        reference = torch.log_softmax(torch.cat([cand_logits, defence_logit[g : g + 1]]), dim=0)
        got = torch.cat(
            [
                node_lp[offset : offset + data.num_nodes][data.candidate_mask],
                def_lp[g : g + 1],
            ]
        )
        assert torch.allclose(got, reference, atol=1e-6)
        offset += data.num_nodes


def test_graph_with_no_candidates_gives_the_defence_everything() -> None:
    model = receiver_model()
    data = make_data(12, 3)
    data.candidate_mask = torch.zeros(12, dtype=torch.bool)
    with torch.no_grad():
        node_logits, defence_logit = model(data)
    node_lp, def_lp = receiver_log_softmax(
        node_logits, defence_logit, torch.zeros(12, dtype=torch.long), 1
    )
    assert float(node_lp.exp().sum()) == 0.0
    assert float(def_lp.exp()) == pytest.approx(1.0, abs=1e-6)


def test_masked_softmax_gradients_are_finite(graphs: list[Data]) -> None:
    """-inf masking must not leak NaNs into the backward pass."""
    torch.manual_seed(0)
    model = ReceiverModel(NODE_DIM, EDGE_DIM)
    batch = Batch.from_data_list(graphs)
    node_logits, defence_logit = model(batch)
    node_lp, def_lp = receiver_log_softmax(node_logits, defence_logit, batch.batch, 3)

    targets = torch.tensor([0, 1, 2], dtype=torch.long)
    offsets = torch.tensor([0, graphs[0].num_nodes, graphs[0].num_nodes + graphs[1].num_nodes])
    loss = -node_lp[offsets + targets].mean() - def_lp.mean()
    loss.backward()

    assert torch.isfinite(loss)
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


# --------------------------------------------------------------------------- #
# shot head
# --------------------------------------------------------------------------- #
def test_shot_model_returns_one_logit_per_graph(graphs: list[Data]) -> None:
    model = shot_model()
    with torch.no_grad():
        single = model(graphs[0])
        batched = model(Batch.from_data_list(graphs))
    assert single.shape == (1,)
    assert batched.shape == (3,)
    assert torch.isfinite(batched).all()
    assert float(batched[0]) == pytest.approx(float(single[0]), abs=1e-5)


def test_shot_model_sees_the_defence() -> None:
    """Moving defenders must change the shot logit; the readout is not attacker-only."""
    model = shot_model()
    data = make_data(18, 7)
    with torch.no_grad():
        before = float(model(data)[0])
        data.x = data.x.clone()
        data.x[~data.is_attacker] += 1.5
        after = float(model(data)[0])
    assert abs(before - after) > 1e-6


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def _card() -> dict:
    return {
        "data": {"competitions": ["Euro 2024"], "seasons": ["2024"], "n_corners": 18},
        "splits": {"scheme": "grouped by match_id", "train": 12, "val": 3, "test": 3},
        "metrics": {"top1": 0.25, "top3": 0.5},
        "date": "2026-08-25",
        "features": ["x", "y"],
        # "params" is deliberately omitted: save_model fills it from the module.
        "assume_static": True,
    }


@pytest.mark.parametrize("factory", [receiver_model, shot_model])
def test_save_load_round_trip_is_identical(tmp_path: Path, factory: Any, graphs: list[Data]) -> None:
    model = factory()
    model.train()  # the loader must not inherit train mode
    batch = Batch.from_data_list(graphs)

    model.eval()
    with torch.no_grad():
        before = model(batch)

    path = tmp_path / "model.pt"
    save_model(model, path, _card())
    loaded, card = load_model(path)

    assert type(loaded) is type(model)
    assert not loaded.training, "load_model must set eval mode"
    with torch.no_grad():
        after = loaded(batch)

    before_t = before if isinstance(before, tuple) else (before,)
    after_t = after if isinstance(after, tuple) else (after,)
    for a, b in zip(before_t, after_t, strict=True):
        assert torch.equal(torch.nan_to_num(a, neginf=-1e30), torch.nan_to_num(b, neginf=-1e30))

    assert card["date"] == "2026-08-25"
    assert card["assume_static"] is True
    assert card["params"] == count_parameters(model)


def test_load_model_reconstructs_without_the_caller_knowing_shapes(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = ReceiverModel(NODE_DIM + 3, EDGE_DIM + 2, hidden=48, layers=3, dropout=0.05)
    path = tmp_path / "odd.pt"
    save_model(model, path, {})
    loaded, _ = load_model(path)
    assert loaded.encoder.node_dim_in == NODE_DIM + 3
    assert loaded.encoder.edge_dim_in == EDGE_DIM + 2
    assert loaded.encoder.hidden == 48
    assert len(loaded.encoder.convs) == 3


def test_save_model_fills_in_the_required_card_fields(tmp_path: Path) -> None:
    model = shot_model()
    path = tmp_path / "m.pt"
    save_model(model, path, {"date": "2026-08-25"})
    _, card = load_model(path)
    assert card["assume_static"] is True
    assert card["params"] == count_parameters(model)


def test_load_model_on_a_missing_path_is_clear(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        load_model(tmp_path / "nope.pt")


def test_load_bundled_missing_asset_is_actionable() -> None:
    try:
        load_bundled("receiver")
    except FileNotFoundError as exc:
        msg = str(exc)
        assert "assets" in msg
        assert "train" in msg
    else:
        pytest.skip("a bundled receiver model exists, so there is nothing to report")


def test_load_bundled_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown bundled model"):
        load_bundled("goalkeeper")


# --------------------------------------------------------------------------- #
# single-corner inference on real data
# --------------------------------------------------------------------------- #
@needs_features
def test_receiver_probabilities_form_a_distribution() -> None:
    corners = real_corners()
    torch.manual_seed(0)
    model = ReceiverModel()
    model.eval()

    for frame in corners[:5]:
        probs, p_defence = receiver_probabilities(model, frame)
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (len(frame.players),)
        assert float(probs.sum() + p_defence) == pytest.approx(1.0, abs=1e-9)
        assert 0.0 <= p_defence <= 1.0
        assert (probs >= 0.0).all()

        taker = frame.taker_index
        for i, player in enumerate(frame.players):
            is_candidate = player.is_attacker and not player.is_taker
            if not is_candidate:
                assert probs[i] == 0.0, f"player {i} is not a candidate but scored"
        if taker is not None:
            assert probs[taker] == 0.0


@needs_features
def test_shot_probability_is_a_probability() -> None:
    corners = real_corners()
    torch.manual_seed(0)
    model = ShotModel()
    model.eval()
    for frame in corners[:5]:
        p = shot_probability(model, frame)
        assert isinstance(p, float)
        assert 0.0 < p < 1.0
        assert math.isfinite(p)


@needs_features
def test_inference_is_under_100ms_per_corner() -> None:
    corners = real_corners()
    frame = corners[0]
    torch.manual_seed(0)
    receiver = ReceiverModel()
    shot = ShotModel()
    receiver.eval()
    shot.eval()

    for _ in range(3):  # warm up lazy kernels
        receiver_probabilities(receiver, frame)
        shot_probability(shot, frame)

    timings = []
    for corner in (corners * 3)[:20]:
        t0 = time.perf_counter()
        receiver_probabilities(receiver, corner)
        shot_probability(shot, corner)
        timings.append((time.perf_counter() - t0) * 1000.0)

    median = float(np.median(timings))
    assert median < 100.0, f"median {median:.1f} ms per corner exceeds the 100 ms budget"


@needs_features
def test_real_corners_batch_into_per_graph_distributions() -> None:
    """End-to-end: real frames of differing sizes, batched, still normalise per graph."""
    corners = real_corners()
    frames = corners[:6]
    sizes = {len(f.players) for f in frames}
    assert len(sizes) > 1, "expected variable-size freeze frames in the smoke set"

    batch = Batch.from_data_list([sp_features.build_graph(f) for f in frames])
    torch.manual_seed(0)
    model = ReceiverModel()
    model.eval()
    with torch.no_grad():
        node_logits, defence_logit = model(batch)
    node_lp, def_lp = receiver_log_softmax(
        node_logits, defence_logit, batch.batch, len(frames)
    )
    node_p = node_lp.exp()

    for g, frame in enumerate(frames):
        sel = batch.batch == g
        assert int(sel.sum()) == len(frame.players)
        total = float(node_p[sel].sum() + def_lp[g].exp())
        assert total == pytest.approx(1.0, abs=1e-5)

    # and the batched per-graph answer equals the single-corner helper
    probs, p_defence = receiver_probabilities(model, frames[0])
    sel = batch.batch == 0
    assert np.allclose(probs, node_p[sel].numpy(), atol=1e-5)
    assert p_defence == pytest.approx(float(def_lp[0].exp()), abs=1e-5)


@needs_features
def test_probabilities_respond_to_moving_a_player() -> None:
    """Sanity check that the featuriser and the model are actually coupled."""
    corners = real_corners()
    frame = corners[0]
    torch.manual_seed(0)
    model = ShotModel()
    model.eval()

    base = shot_probability(model, frame)
    movable = [i for i, p in enumerate(frame.players) if not p.is_attacker and not p.is_keeper]
    if not movable:
        pytest.skip("no movable defender in this corner")
    players = list(frame.players)
    players[movable[0]] = players[movable[0]].moved(-4.0, 3.0)
    moved = shot_probability(model, frame.with_players(players))
    assert abs(moved - base) > 1e-9
