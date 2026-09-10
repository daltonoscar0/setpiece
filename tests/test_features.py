"""Tests for the graph featuriser.

The interesting failure modes here are silent: a target that shifts by the wrong
offset under batching, or a feature that quietly leaks the delivery destination.
Both are tested explicitly.

A third silent failure mode is the mirroring. ``build_graph`` reflects every
frame about ``y = GOAL_Y`` so corners from the two sides land on top of each
other, and a fixture whose taker sits at ``y < GOAL_Y`` is reflected by the
identity — so every assertion written against such a fixture stays green whether
or not the reflection happens at all, and tests nothing. Every geometry test
below is therefore parametrised over :func:`near_side_frame` *and*
:func:`far_side_frame`, and states its expectations in mirrored coordinates.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from setpiece._io import read_corners
from setpiece.features import (
    EDGE_FEATURE_NAMES,
    FEATURE_NAMES,
    N_EDGE_FEATURES,
    N_NODE_FEATURES,
    build_flat,
    build_graph,
    global_receiver_index,
    mirror_side,
    taker_position,
)
from setpiece.types import GOAL_X, GOAL_Y, PITCH_WIDTH, CornerFrame, Player

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "data" / "interim" / "smoke.jsonl"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def corners() -> list[CornerFrame]:
    if not SMOKE.exists():
        pytest.skip(f"missing {SMOKE}")
    frames = read_corners(SMOKE)
    if not frames:
        pytest.skip("empty smoke dataset")
    return frames


@pytest.fixture(scope="module")
def graphs(corners: list[CornerFrame]) -> list[Data]:
    """One graph per smoke corner, built once for the whole module.

    Most tests below only read the graph they are handed, and ``build_graph`` is
    deterministic (``test_deterministic`` asserts exactly that on freshly built
    pairs), so rebuilding all eighteen graphs inside every test bought nothing
    but wall clock. Anything that needs a *fresh* graph — determinism, or the
    poisoned-input comparisons — still calls ``build_graph`` itself.
    """
    return [build_graph(f) for f in corners]


def reflect(frame: CornerFrame) -> CornerFrame:
    """The same corner taken from the other flag: every ``y`` reflected about ``GOAL_Y``.

    Player order, team flags and labels are untouched, so node ``i`` of the
    reflection is the mirror image of node ``i`` of the original. That is what
    lets the invariance tests compare the two graphs row by row.
    """
    return frame.with_players(
        [dataclasses.replace(p, y=GOAL_Y + (GOAL_Y - p.y)) for p in frame.players]
    )


def mirrored_y(frame: CornerFrame, y: float) -> float:
    """``y`` in the featuriser's mirrored coordinates, derived independently.

    Deliberately re-derives the rule from :func:`taker_position` rather than
    calling :func:`~setpiece.features._mirror_y`, so the geometry tests below
    check the featuriser against a second statement of the convention instead of
    against itself.
    """
    side = 1.0 if taker_position(frame)[1] <= GOAL_Y else -1.0
    return GOAL_Y + (y - GOAL_Y) * side


def synthetic_frame(**kwargs: object) -> CornerFrame:
    """A small hand-built corner: 3 attackers (one taker), 3 defenders (one keeper).

    Taken from the ``y < GOAL_Y`` flag, so the featuriser's reflection is the
    identity on it. Use :func:`far_side_frame` for the case that actually
    exercises the mirror.
    """
    players = [
        Player(x=96.0, y=30.0, is_attacker=True),
        Player(x=99.0, y=34.0, is_attacker=True),
        Player(x=97.5, y=38.0, is_attacker=False),
        Player(x=103.5, y=34.0, is_attacker=False, is_keeper=True),
        Player(x=90.0, y=44.0, is_attacker=False),
        Player(x=104.5, y=0.5, is_attacker=True, is_taker=True),
    ]
    return CornerFrame(corner_id="synthetic", players=players, **kwargs)  # type: ignore[arg-type]


def near_side_frame(**kwargs: object) -> CornerFrame:
    """:func:`synthetic_frame`, named for what makes it different from its twin."""
    return synthetic_frame(**kwargs)


def far_side_frame(**kwargs: object) -> CornerFrame:
    """:func:`synthetic_frame` reflected: the taker sits at ``y > GOAL_Y``.

    Every geometry assertion in this module runs against this fixture as well as
    the near-side one. Without it the mirror could be deleted outright and the
    suite would stay green.
    """
    frame = reflect(synthetic_frame(**kwargs))
    return dataclasses.replace(frame, corner_id="synthetic-far")


#: Both hand-built fixtures, for parametrising the geometry tests.
SIDED_FRAMES = [near_side_frame, far_side_frame]


# --------------------------------------------------------------------------- #
# contract: shapes and dtypes
# --------------------------------------------------------------------------- #
def test_feature_name_lengths() -> None:
    assert len(FEATURE_NAMES) == N_NODE_FEATURES
    assert len(set(FEATURE_NAMES)) == N_NODE_FEATURES
    assert len(EDGE_FEATURE_NAMES) == N_EDGE_FEATURES
    assert len(set(EDGE_FEATURE_NAMES)) == N_EDGE_FEATURES
    for required in ("is_attacker", "is_taker", "is_keeper", "dist_goal", "post_angle"):
        assert required in FEATURE_NAMES


def test_graph_attribute_contract(corners: list[CornerFrame], graphs: list[Data]) -> None:
    for frame, g in zip(corners, graphs, strict=True):
        n = len(frame.players)

        assert g.num_nodes == n
        assert isinstance(g.num_nodes, int)

        assert g.x.dtype == torch.float32
        assert tuple(g.x.shape) == (n, N_NODE_FEATURES)

        assert g.edge_index.dtype == torch.int64
        assert g.edge_index.dim() == 2
        assert g.edge_index.shape[0] == 2

        assert g.edge_attr.dtype == torch.float32
        assert tuple(g.edge_attr.shape) == (g.edge_index.shape[1], N_EDGE_FEATURES)

        for key in ("is_attacker", "is_taker", "is_keeper", "candidate_mask"):
            t = getattr(g, key)
            assert t.dtype == torch.bool, key
            assert tuple(t.shape) == (n,), key

        assert g.y_receiver.dtype == torch.int64
        assert tuple(g.y_receiver.shape) == (1,)
        assert g.y_receiver_valid.dtype == torch.bool
        assert tuple(g.y_receiver_valid.shape) == (1,)
        assert g.y_shot.dtype == torch.float32
        assert tuple(g.y_shot.shape) == (1,)
        assert g.y_shot.item() in (0.0, 1.0)


def test_flags_match_the_frame(corners: list[CornerFrame], graphs: list[Data]) -> None:
    for frame, g in zip(corners, graphs, strict=True):
        assert g.is_attacker.tolist() == [p.is_attacker for p in frame.players]
        assert g.is_taker.tolist() == [p.is_taker for p in frame.players]
        assert g.is_keeper.tolist() == [p.is_keeper for p in frame.players]


def test_no_nans_and_roughly_unit_range(graphs: list[Data]) -> None:
    for g in graphs:
        assert torch.isfinite(g.x).all()
        assert torch.isfinite(g.edge_attr).all()
        assert g.x.abs().max().item() <= 5.0
        assert g.edge_attr.abs().max().item() <= 5.0
        # the features are not all constant
        assert g.x.std(dim=0).max().item() > 0.0


def test_build_flat_matches_graph_x(corners: list[CornerFrame], graphs: list[Data]) -> None:
    for frame, g in zip(corners, graphs, strict=True):
        flat = build_flat(frame)
        assert isinstance(flat, np.ndarray)
        assert flat.dtype == np.float32
        assert flat.shape == (len(frame.players), N_NODE_FEATURES)
        np.testing.assert_allclose(flat, g.x.numpy())


def test_deterministic(corners: list[CornerFrame]) -> None:
    for frame in corners[:5]:
        a, b = build_graph(frame), build_graph(frame)
        assert torch.equal(a.x, b.x)
        assert torch.equal(a.edge_index, b.edge_index)
        assert torch.equal(a.edge_attr, b.edge_attr)


# --------------------------------------------------------------------------- #
# feature semantics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("make_frame", SIDED_FRAMES)
def test_geometry_features_are_what_they_claim(make_frame) -> None:
    """Every signed-``y`` feature is stated in mirrored coordinates.

    On the far-side fixture the raw and mirrored ``y`` differ for every player,
    so a featuriser that skipped the reflection would fail here rather than
    quietly agreeing.
    """
    frame = make_frame()
    x = build_flat(frame)
    col = {name: i for i, name in enumerate(FEATURE_NAMES)}

    for i, p in enumerate(frame.players):
        py = mirrored_y(frame, p.y)

        assert x[i, col["x_norm"]] == pytest.approx(p.x / 105.0, abs=1e-5)
        assert x[i, col["y_norm"]] == pytest.approx(py / PITCH_WIDTH, abs=1e-5)
        assert x[i, col["dy_goal"]] * 30.0 == pytest.approx(py - GOAL_Y, abs=1e-4)

        d = math.hypot(GOAL_X - p.x, GOAL_Y - py)
        assert x[i, col["dist_goal"]] * 30.0 == pytest.approx(d, abs=1e-4)

        ang = math.atan2(GOAL_Y - py, GOAL_X - p.x)
        assert x[i, col["goal_angle_cos"]] == pytest.approx(math.cos(ang), abs=1e-5)
        assert x[i, col["goal_angle_sin"]] == pytest.approx(math.sin(ang), abs=1e-5)

        taker = frame.players[frame.taker_index]  # type: ignore[index]
        ty = mirrored_y(frame, taker.y)
        dt = math.hypot(taker.x - p.x, ty - py)
        assert x[i, col["dist_taker"]] * 50.0 == pytest.approx(dt, abs=1e-4)

        tang = math.atan2(ty - py, taker.x - p.x)
        assert x[i, col["taker_angle_cos"]] == pytest.approx(math.cos(tang), abs=1e-5)
        assert x[i, col["taker_angle_sin"]] == pytest.approx(math.sin(tang), abs=1e-5)

        assert x[i, col["is_attacker"]] == float(p.is_attacker)
        assert x[i, col["is_taker"]] == float(p.is_taker)
        assert x[i, col["is_keeper"]] == float(p.is_keeper)

    # a player level with the goal line between the posts sees a wide angle;
    # the taker in the corner sees a very narrow one
    post_angle = x[:, col["post_angle"]] * (math.pi / 2.0)
    assert post_angle[1] > post_angle[5]
    assert post_angle[5] < math.radians(15.0)


@pytest.mark.parametrize("side", ["near", "far"])
def test_density_counts(side: str) -> None:
    players = [
        Player(x=100.0, y=34.0, is_attacker=True),
        Player(x=101.0, y=34.0, is_attacker=False),  # 1 m away
        Player(x=104.0, y=34.0, is_attacker=True),  # 4 m away
        Player(x=90.0, y=10.0, is_attacker=False),  # far
        Player(x=104.9, y=0.2, is_attacker=True, is_taker=True),
    ]
    frame = CornerFrame(corner_id="density", players=players)
    if side == "far":
        frame = reflect(frame)
    # densities are distances only, so the reflection must not move them at all
    x = build_flat(frame)
    col = {name: i for i, name in enumerate(FEATURE_NAMES)}

    assert x[0, col["n_within_3m"]] * 3.0 == pytest.approx(1.0)
    assert x[0, col["n_within_5m"]] * 5.0 == pytest.approx(2.0)
    assert x[0, col["n_opponents_within_3m"]] * 3.0 == pytest.approx(1.0)
    assert x[0, col["dist_nearest_opponent"]] * 20.0 == pytest.approx(1.0, abs=1e-4)
    assert x[0, col["dist_nearest_teammate"]] * 20.0 == pytest.approx(4.0, abs=1e-4)
    assert x[3, col["n_within_3m"]] == pytest.approx(0.0)


@pytest.mark.parametrize("make_frame", SIDED_FRAMES)
def test_box_flags(make_frame) -> None:
    """The box is symmetric about ``GOAL_Y``, so both sides give the same flags."""
    frame = make_frame()
    x = build_flat(frame)
    col = {name: i for i, name in enumerate(FEATURE_NAMES)}
    # index 1 (99.0, 34.0) is inside the penalty area but outside the six-yard box;
    # index 3 (103.5, 34.0) is inside both; the taker is outside both.
    assert x[1, col["in_penalty_area"]] == 1.0
    assert x[1, col["in_six_yard_box"]] == 0.0
    assert x[3, col["in_six_yard_box"]] == 1.0
    assert x[5, col["in_penalty_area"]] == 0.0


@pytest.mark.parametrize("make_frame", SIDED_FRAMES)
def test_features_respond_to_movement(make_frame) -> None:
    frame = make_frame()
    moved = frame.with_players(
        [p.moved(2.0, -1.0) if i == 0 else p for i, p in enumerate(frame.players)]
    )
    before, after = build_flat(frame), build_flat(moved)
    assert not np.allclose(before[0], after[0])
    # the taker did not move, and neither did its own geometry to the goal
    col = FEATURE_NAMES.index("dist_goal")
    assert before[5, col] == pytest.approx(after[5, col])


# --------------------------------------------------------------------------- #
# mirroring
#
# A corner from the left and its mirror image from the right are the same
# coaching situation. build_graph reflects every frame about y = GOAL_Y so the
# two land on top of each other; these tests pin that down, on real corners from
# both flags, and on the degenerate frames where the rule has nothing to go on.
# --------------------------------------------------------------------------- #
def test_smoke_corners_cover_both_flags(corners: list[CornerFrame]) -> None:
    """Guard for every test below: a one-sided fixture set would test nothing."""
    sides = [mirror_side(f) for f in corners]
    assert sides.count(1.0) > 0, "no corner taken from the y < GOAL_Y flag"
    assert sides.count(-1.0) > 0, "no corner taken from the y > GOAL_Y flag"


def test_mirror_side_follows_the_taker(corners: list[CornerFrame]) -> None:
    for frame in corners:
        ty = taker_position(frame)[1]
        assert ty != GOAL_Y, "no real corner is taken from the centre line"
        assert mirror_side(frame) == (1.0 if ty < GOAL_Y else -1.0)
        # the mirror image is taken from the other flag, so it picks the other side
        assert mirror_side(reflect(frame)) == -mirror_side(frame)


def test_build_graph_is_invariant_to_reflection(
    corners: list[CornerFrame], graphs: list[Data]
) -> None:
    """Real corners: reflecting a frame about ``y = GOAL_Y`` changes no feature.

    ``reflect`` preserves node order, so this is checked row by row — the
    strictest reading of "identical up to node ordering". Exact equality is the
    bar, not a tolerance: the reflection is applied once to the coordinates, so
    the two featurisations run the same arithmetic on the same numbers.
    """
    worst_x = 0.0
    worst_edge = 0.0
    for frame, g in zip(corners, graphs, strict=True):
        h = build_graph(reflect(frame))
        worst_x = max(worst_x, (g.x - h.x).abs().max().item())
        if g.edge_attr.numel():
            worst_edge = max(worst_edge, (g.edge_attr - h.edge_attr).abs().max().item())
        assert torch.equal(g.edge_index, h.edge_index), frame.corner_id
        assert torch.equal(g.x, h.x), frame.corner_id
        assert torch.equal(g.edge_attr, h.edge_attr), frame.corner_id
    assert worst_x == 0.0 and worst_edge == 0.0


def test_reflection_invariance_survives_batching(corners: list[CornerFrame]) -> None:
    """The invariance is a property of the features, so batches must agree too."""
    subset = corners[:8]
    a = next(iter(DataLoader([build_graph(f) for f in subset], batch_size=8, shuffle=False)))
    b = next(
        iter(
            DataLoader(
                [build_graph(reflect(f)) for f in subset], batch_size=8, shuffle=False
            )
        )
    )
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.edge_attr, b.edge_attr)
    assert torch.equal(a.edge_index, b.edge_index)


def test_the_two_sided_fixtures_featurise_identically() -> None:
    near = build_graph(near_side_frame())
    far = build_graph(far_side_frame())
    assert torch.equal(near.x, far.x)
    assert torch.equal(near.edge_attr, far.edge_attr)
    assert torch.equal(near.edge_index, far.edge_index)
    # ...while the frames themselves genuinely differ
    assert [p.y for p in near_side_frame().players] != [p.y for p in far_side_frame().players]


def test_far_side_features_are_not_the_raw_coordinates() -> None:
    """The mirror really ran: raw ``y`` would give different node and edge values.

    Without this the previous test would also pass on a featuriser that read raw
    coordinates and happened to be handed a symmetric fixture.
    """
    frame = far_side_frame()
    assert mirror_side(frame) == -1.0
    g = build_graph(frame)
    col = {name: i for i, name in enumerate(FEATURE_NAMES)}
    ecol = {name: i for i, name in enumerate(EDGE_FEATURE_NAMES)}

    off_centre = [i for i, p in enumerate(frame.players) if p.y != GOAL_Y]
    assert off_centre, "fixture must have players off the centre line"
    for i in off_centre:
        raw = frame.players[i].y / PITCH_WIDTH
        assert g.x[i, col["y_norm"]].item() != pytest.approx(raw, abs=1e-6)
        # the reflected value is the one that is there
        assert g.x[i, col["y_norm"]].item() == pytest.approx(
            mirrored_y(frame, frame.players[i].y) / PITCH_WIDTH, abs=1e-6
        )

    # and the edge features are reflected in step, not left in raw coordinates
    flipped = 0
    for e, (a, b) in enumerate(g.edge_index.t().tolist()):
        raw_dy = frame.players[b].y - frame.players[a].y
        if raw_dy == 0.0:
            continue
        flipped += 1
        assert g.edge_attr[e, ecol["dy"]].item() * 20.0 == pytest.approx(-raw_dy, abs=1e-4)
    assert flipped > 0


def test_reflection_invariance_without_a_taker_flag() -> None:
    """User CSV input: no taker flag, so the side comes from the inferred corner flag."""
    body = [
        Player(x=96.0, y=30.0, is_attacker=True),
        Player(x=99.0, y=34.0, is_attacker=True),
        Player(x=97.5, y=38.0, is_attacker=False),
        Player(x=103.5, y=34.0, is_attacker=False, is_keeper=True),
        Player(x=104.5, y=0.5, is_attacker=True),  # by the flag, but unflagged
    ]
    frame = CornerFrame(corner_id="no-taker", players=body)
    assert frame.taker_index is None
    assert taker_position(frame) == (105.0, 0.0)
    assert mirror_side(frame) == 1.0

    flipped = reflect(frame)
    assert taker_position(flipped) == (105.0, 68.0)
    assert mirror_side(flipped) == -1.0

    g, h = build_graph(frame), build_graph(flipped)
    assert torch.equal(g.x, h.x)
    assert torch.equal(g.edge_attr, h.edge_attr)
    assert torch.equal(g.edge_index, h.edge_index)


def test_taker_exactly_on_the_centre_line_is_left_unreflected() -> None:
    """Degenerate: a taker at ``y == GOAL_Y`` picks no side, so the frame is left alone.

    No corner is taken from the middle of the pitch, but the featuriser must
    still produce finite, well-formed features rather than dividing by a side it
    cannot determine. Reflection invariance provably cannot hold here for any
    taker-based rule: the frame and its mirror put the taker on the same line and
    therefore choose the same side, so this asserts the documented behaviour
    (identity) instead of an invariance that does not exist.
    """
    players = [
        Player(x=96.0, y=30.0, is_attacker=True),
        Player(x=99.0, y=40.0, is_attacker=True),
        Player(x=103.5, y=34.0, is_attacker=False, is_keeper=True),
        Player(x=104.5, y=GOAL_Y, is_attacker=True, is_taker=True),
    ]
    frame = CornerFrame(corner_id="centre-line", players=players)
    assert mirror_side(frame) == 1.0

    g = build_graph(frame)
    assert torch.isfinite(g.x).all()
    assert torch.isfinite(g.edge_attr).all()
    col = FEATURE_NAMES.index("y_norm")
    for i, p in enumerate(players):
        assert g.x[i, col].item() == pytest.approx(p.y / PITCH_WIDTH, abs=1e-6)

    # its mirror is also left alone, so the two differ, stated, not hidden
    h = build_graph(reflect(frame))
    assert mirror_side(reflect(frame)) == 1.0
    assert not torch.equal(g.x, h.x)


def test_taker_just_either_side_of_the_centre_line() -> None:
    """The rule is a clean threshold at ``GOAL_Y``, with no dead band around it."""
    body = [
        Player(x=96.0, y=30.0, is_attacker=True),
        Player(x=103.5, y=34.0, is_attacker=False, is_keeper=True),
    ]

    def with_taker(ty: float) -> CornerFrame:
        return CornerFrame(
            corner_id=f"t{ty}",
            players=[*body, Player(x=104.5, y=ty, is_attacker=True, is_taker=True)],
        )

    assert mirror_side(with_taker(GOAL_Y - 1e-3)) == 1.0
    assert mirror_side(with_taker(GOAL_Y + 1e-3)) == -1.0

    # and a taker only half a metre off the line, an offset that reflects
    # exactly in binary, so the two featurisations are bit-identical, is still
    # mirrored like any other corner
    near = with_taker(GOAL_Y - 0.5)
    assert mirror_side(near) == 1.0
    assert mirror_side(reflect(near)) == -1.0
    g, h = build_graph(near), build_graph(reflect(near))
    assert torch.equal(g.x, h.x)
    assert torch.equal(g.edge_attr, h.edge_attr)


# --------------------------------------------------------------------------- #
# edges
# --------------------------------------------------------------------------- #
def test_edges_are_symmetric_and_loop_free(graphs: list[Data]) -> None:
    for g in graphs:
        edges = {(int(a), int(b)) for a, b in g.edge_index.t().tolist()}
        assert edges, "expected a non-empty edge set"
        for a, b in edges:
            assert a != b
            assert (b, a) in edges
            assert 0 <= a < g.num_nodes and 0 <= b < g.num_nodes


def test_taker_is_connected_to_everyone(corners: list[CornerFrame], graphs: list[Data]) -> None:
    for frame, g in zip(corners, graphs, strict=True):
        taker = frame.taker_index
        if taker is None:
            continue
        src, dst = g.edge_index
        nbrs = set(dst[src == taker].tolist())
        assert nbrs == set(range(g.num_nodes)) - {taker}


@pytest.mark.parametrize("make_frame", SIDED_FRAMES)
def test_edge_attr_semantics(make_frame) -> None:
    """Edge ``dy`` is mirrored too — the reflection is not a node-only change."""
    frame = make_frame()
    g = build_graph(frame)
    ecol = {name: i for i, name in enumerate(EDGE_FEATURE_NAMES)}
    for e, (a, b) in enumerate(g.edge_index.t().tolist()):
        pa, pb = frame.players[a], frame.players[b]
        ya, yb = mirrored_y(frame, pa.y), mirrored_y(frame, pb.y)
        attr = g.edge_attr[e]
        assert attr[ecol["dx"]].item() * 20.0 == pytest.approx(pb.x - pa.x, abs=1e-4)
        assert attr[ecol["dy"]].item() * 20.0 == pytest.approx(yb - ya, abs=1e-4)
        d = math.hypot(pb.x - pa.x, yb - ya)
        assert attr[ecol["dist"]].item() * 20.0 == pytest.approx(d, abs=1e-4)
        assert attr[ecol["same_team"]].item() == float(pa.is_attacker == pb.is_attacker)
        assert attr[ecol["taker_edge"]].item() == float(pa.is_taker or pb.is_taker)


def test_single_player_frame_has_no_edges() -> None:
    frame = CornerFrame(corner_id="lonely", players=[Player(x=50.0, y=34.0, is_attacker=True)])
    g = build_graph(frame)
    assert g.num_nodes == 1
    assert tuple(g.edge_index.shape) == (2, 0)
    assert tuple(g.edge_attr.shape) == (0, N_EDGE_FEATURES)
    assert tuple(g.x.shape) == (1, N_NODE_FEATURES)


# --------------------------------------------------------------------------- #
# candidate mask
# --------------------------------------------------------------------------- #
def test_candidate_mask_is_attackers_minus_taker(
    corners: list[CornerFrame], graphs: list[Data]
) -> None:
    for frame, g in zip(corners, graphs, strict=True):
        expected = set(frame.attacker_indices) - {frame.taker_index}
        assert set(torch.nonzero(g.candidate_mask).flatten().tolist()) == expected
        assert not (g.candidate_mask & g.is_taker).any()
        assert (g.candidate_mask <= g.is_attacker).all()
        assert g.candidate_mask.sum().item() > 0


def test_candidate_mask_synthetic() -> None:
    g = build_graph(synthetic_frame())
    assert g.candidate_mask.tolist() == [True, True, False, False, False, False]


# --------------------------------------------------------------------------- #
# labels, including the unlabelled inference path
# --------------------------------------------------------------------------- #
def test_unlabelled_frame() -> None:
    frame = synthetic_frame()
    assert frame.first_touch is None and frame.shot_within_5 is None
    g = build_graph(frame)
    assert g.y_receiver.item() == -1
    assert g.y_receiver_valid.item() is False
    assert g.y_shot.item() == 0.0


def test_unknown_first_touch_is_invalid() -> None:
    g = build_graph(synthetic_frame(first_touch="unknown", shot_within_5=True))
    assert g.y_receiver.item() == -1
    assert g.y_receiver_valid.item() is False
    assert g.y_shot.item() == 1.0


def test_defender_first_touch_is_a_real_label() -> None:
    g = build_graph(synthetic_frame(first_touch="defender", shot_within_5=False))
    assert g.y_receiver.item() == -1
    assert g.y_receiver_valid.item() is True


def test_attacker_first_touch_with_receiver() -> None:
    g = build_graph(synthetic_frame(first_touch="attacker", receiver_idx=1))
    assert g.y_receiver.item() == 1
    assert g.y_receiver_valid.item() is True
    assert bool(g.candidate_mask[g.y_receiver.item()])


def test_attacker_first_touch_without_receiver_is_dropped() -> None:
    """Receiver off camera: the row must be excluded from the receiver loss."""
    g = build_graph(synthetic_frame(first_touch="attacker", receiver_idx=None))
    assert g.y_receiver.item() == -1
    assert g.y_receiver_valid.item() is False


def test_receiver_outside_the_candidate_set_is_dropped() -> None:
    for bad in (2, 5, 99):  # a defender, the taker, out of range
        g = build_graph(synthetic_frame(first_touch="attacker", receiver_idx=bad))
        assert g.y_receiver_valid.item() is False
        assert g.y_receiver.item() == -1


def test_real_label_semantics(corners: list[CornerFrame], graphs: list[Data]) -> None:
    seen_valid_receiver = False
    seen_defence = False
    seen_dropped = False
    for frame, g in zip(corners, graphs, strict=True):
        y = g.y_receiver.item()
        valid = g.y_receiver_valid.item()
        if frame.first_touch == "attacker" and frame.receiver_idx is not None:
            assert (y, valid) == (frame.receiver_idx, True)
            assert bool(g.candidate_mask[y])
            seen_valid_receiver = True
        elif frame.first_touch == "attacker":
            assert (y, valid) == (-1, False)
            seen_dropped = True
        elif frame.first_touch == "defender":
            assert (y, valid) == (-1, True)
            seen_defence = True
        assert g.y_shot.item() == float(bool(frame.shot_within_5))
    assert seen_valid_receiver and seen_defence and seen_dropped


# --------------------------------------------------------------------------- #
# batching
# --------------------------------------------------------------------------- #
def test_batching_lines_up_with_per_graph_values(graphs: list[Data]) -> None:
    graphs = graphs[:8]
    loader = DataLoader(graphs, batch_size=len(graphs), shuffle=False)
    batch = next(iter(loader))

    assert batch.num_graphs == len(graphs)
    assert batch.num_nodes == sum(g.num_nodes for g in graphs)
    assert batch.x.shape == (batch.num_nodes, N_NODE_FEATURES)
    assert batch.ptr.tolist() == [0, *np.cumsum([g.num_nodes for g in graphs]).tolist()]

    for i, g in enumerate(graphs):
        lo, hi = int(batch.ptr[i]), int(batch.ptr[i + 1])

        # per-node attributes land in the right slice
        assert torch.equal(batch.x[lo:hi], g.x)
        assert torch.equal(batch.candidate_mask[lo:hi], g.candidate_mask)
        assert torch.equal(batch.is_attacker[lo:hi], g.is_attacker)
        assert torch.equal(batch.is_taker[lo:hi], g.is_taker)
        assert torch.equal(batch.is_keeper[lo:hi], g.is_keeper)
        assert (batch.batch[lo:hi] == i).all()

        # per-graph targets stay one-per-graph and unmodified
        assert batch.y_receiver[i].item() == g.y_receiver.item()
        assert batch.y_receiver_valid[i].item() == g.y_receiver_valid.item()
        assert batch.y_shot[i].item() == g.y_shot.item()

    assert tuple(batch.y_receiver.shape) == (len(graphs),)
    assert tuple(batch.y_shot.shape) == (len(graphs),)


def test_batched_edge_index_is_offset_per_graph(graphs: list[Data]) -> None:
    graphs = graphs[:6]
    batch = next(iter(DataLoader(graphs, batch_size=len(graphs), shuffle=False)))
    assert batch.edge_index.shape[1] == sum(g.edge_index.shape[1] for g in graphs)
    off = 0
    e0 = 0
    for i, g in enumerate(graphs):
        e = g.edge_index.shape[1]
        assert torch.equal(batch.edge_index[:, e0 : e0 + e], g.edge_index + off)
        assert torch.equal(batch.edge_attr[e0 : e0 + e], g.edge_attr)
        # every edge stays inside its own graph
        assert (batch.batch[batch.edge_index[0, e0 : e0 + e]] == i).all()
        off += g.num_nodes
        e0 += e


def test_global_receiver_index_points_at_the_right_node(graphs: list[Data]) -> None:
    """The target must select the receiver's own feature row after batching."""
    batch = next(iter(DataLoader(graphs, batch_size=len(graphs), shuffle=False)))
    gidx = global_receiver_index(batch)

    assert tuple(gidx.shape) == (len(graphs),)
    labelled = 0
    for i, g in enumerate(graphs):
        local = g.y_receiver.item()
        if local < 0:
            assert gidx[i].item() == -1
            continue
        labelled += 1
        expected = int(batch.ptr[i]) + local
        assert gidx[i].item() == expected
        assert torch.equal(batch.x[expected], g.x[local])
        assert bool(batch.candidate_mask[expected])
    assert labelled > 0

    # identity on a single unbatched graph
    for g in graphs:
        assert global_receiver_index(g).tolist() == g.y_receiver.tolist()


def test_batch_round_trips(graphs: list[Data]) -> None:
    graphs = graphs[:7]
    batch = next(iter(DataLoader(graphs, batch_size=len(graphs), shuffle=False)))
    for original, restored in zip(graphs, batch.to_data_list(), strict=True):
        assert restored.num_nodes == original.num_nodes
        assert torch.equal(restored.x, original.x)
        assert torch.equal(restored.edge_index, original.edge_index)
        assert torch.equal(restored.edge_attr, original.edge_attr)
        assert torch.equal(restored.candidate_mask, original.candidate_mask)
        assert torch.equal(restored.y_receiver, original.y_receiver)
        assert torch.equal(restored.y_receiver_valid, original.y_receiver_valid)
        assert torch.equal(restored.y_shot, original.y_shot)


def test_variable_size_graphs_batch(graphs: list[Data]) -> None:
    """Freeze frames vary in size; several batch sizes must all work."""
    assert len({g.num_nodes for g in graphs}) > 1, "expected variable frame sizes"
    for bs in (1, 2, 3, 5):
        seen = 0
        for batch in DataLoader(graphs, batch_size=bs, shuffle=False):
            assert batch.y_receiver.shape[0] == batch.num_graphs
            assert batch.y_shot.shape[0] == batch.num_graphs
            assert batch.candidate_mask.shape[0] == batch.num_nodes
            seen += batch.num_graphs
        assert seen == len(graphs)


# --------------------------------------------------------------------------- #
# label leakage
# --------------------------------------------------------------------------- #
def test_features_ignore_label_adjacent_meta(
    corners: list[CornerFrame], graphs: list[Data]
) -> None:
    """end_xy and pass_outcome are label-adjacent: perturbing them changes nothing."""
    for frame, g in zip(corners[:6], graphs[:6], strict=True):
        flat = build_flat(frame)

        poisoned = frame.with_players(list(frame.players))
        poisoned.meta = dict(frame.meta)
        poisoned.meta["end_xy"] = [0.0, 0.0]
        poisoned.meta["pass_outcome"] = "Incomplete"
        poisoned.meta["pass_length"] = -999.0
        poisoned.meta["pass_height"] = "Ground Pass"
        h = build_graph(poisoned)

        assert torch.equal(g.x, h.x)
        assert torch.equal(g.edge_index, h.edge_index)
        assert torch.equal(g.edge_attr, h.edge_attr)
        np.testing.assert_array_equal(flat, build_flat(poisoned))


def test_features_ignore_the_supervision_labels(
    corners: list[CornerFrame], graphs: list[Data]
) -> None:
    """The outcome labels themselves must not reach ``x``, ``edge_attr`` or the mask.

    ``meta`` is the obvious trap and is covered above, but ``first_touch``,
    ``receiver_idx``, ``receiver_match_dist`` and ``shot_within_5`` are the labels
    the heads are trained against: if any of them changed a single feature the
    cross-validated numbers would be measuring themselves.
    """
    for frame, g in zip(corners, graphs, strict=True):
        flat = build_flat(frame)

        poisoned = frame.with_players(list(frame.players))
        poisoned.first_touch = "defender" if frame.first_touch != "defender" else "attacker"
        poisoned.receiver_idx = None
        poisoned.receiver_match_dist = 999.0
        poisoned.shot_within_5 = not bool(frame.shot_within_5)
        h = build_graph(poisoned)

        assert torch.equal(g.x, h.x)
        assert torch.equal(g.edge_index, h.edge_index)
        assert torch.equal(g.edge_attr, h.edge_attr)
        assert torch.equal(g.candidate_mask, h.candidate_mask)
        np.testing.assert_array_equal(flat, build_flat(poisoned))


def test_features_work_without_meta() -> None:
    frame = synthetic_frame()
    assert frame.meta == {}
    g = build_graph(frame)
    assert torch.isfinite(g.x).all()


def test_source_never_touches_meta() -> None:
    """Static guard: no attribute access to ``.meta`` anywhere in features.py."""
    src = (ROOT / "setpiece" / "features.py").read_text()
    tree = ast.parse(src)
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "meta"
    ]
    assert offenders == [], f"features.py reads .meta at lines {offenders}"
