"""Tests for the pure half of the pitch editor.

Nothing here starts a Streamlit server: every function under test is an
ordinary function that takes coordinates, frames or dicts and returns the same.
The coordinate round trip gets the most attention, because a drag is only
trustworthy if canvas pixels and pitch metres convert both ways exactly.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from setpiece import app
from setpiece._io import read_corners
from setpiece.suggest import Move
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

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE = REPO_ROOT / "data" / "interim" / "smoke.jsonl"

VIEWS = [
    app.DEFAULT_VIEW,
    app.PitchView(x_min=70.0, x_max=105.0, y_min=0.0, y_max=68.0, px_per_metre=8.0),
    app.PitchView(x_min=60.5, x_max=106.25, y_min=-3.5, y_max=71.5, px_per_metre=15.5),
]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def synthetic_frame() -> CornerFrame:
    """A small hand-built corner: taker, keeper, four attackers, five defenders."""
    players = [
        Player(x=105.0, y=0.2, is_attacker=True, is_taker=True),
        Player(x=104.0, y=34.0, is_attacker=False, is_keeper=True),
        Player(x=99.0, y=30.0, is_attacker=True),
        Player(x=98.0, y=36.0, is_attacker=True),
        Player(x=95.5, y=33.0, is_attacker=True),
        Player(x=92.0, y=40.0, is_attacker=True),
        Player(x=99.5, y=31.5, is_attacker=False),
        Player(x=98.5, y=37.5, is_attacker=False),
        Player(x=96.0, y=34.5, is_attacker=False),
        Player(x=93.0, y=41.0, is_attacker=False),
        Player(x=88.0, y=34.0, is_attacker=False),
    ]
    return CornerFrame(corner_id="synthetic", players=players)


@pytest.fixture
def frame() -> CornerFrame:
    return synthetic_frame()


@pytest.fixture
def real_frame() -> CornerFrame:
    if not SMOKE.exists():
        pytest.skip(f"no extracted corners at {SMOKE}")
    corners = read_corners(SMOKE)
    if not corners:
        pytest.skip("smoke dataset is empty")
    return corners[0]


# --------------------------------------------------------------------------- #
# coordinate conversion
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("view", VIEWS)
def test_canvas_pitch_canvas_round_trip_is_identity(view: app.PitchView) -> None:
    for cx in np.linspace(0.0, view.width_px, 17):
        for cy in np.linspace(0.0, view.height_px, 17):
            x, y = app.canvas_to_pitch(view, float(cx), float(cy))
            back_cx, back_cy = app.pitch_to_canvas(view, x, y)
            assert back_cx == pytest.approx(cx, abs=1e-9)
            assert back_cy == pytest.approx(cy, abs=1e-9)


@pytest.mark.parametrize("view", VIEWS)
def test_pitch_canvas_pitch_round_trip_is_identity(view: app.PitchView) -> None:
    for x in np.linspace(0.0, PITCH_LENGTH, 13):
        for y in np.linspace(0.0, PITCH_WIDTH, 13):
            cx, cy = app.pitch_to_canvas(view, float(x), float(y))
            back_x, back_y = app.canvas_to_pitch(view, cx, cy)
            assert back_x == pytest.approx(x, abs=1e-9)
            assert back_y == pytest.approx(y, abs=1e-9)


def test_view_orientation_puts_the_goal_at_the_top() -> None:
    view = app.DEFAULT_VIEW
    goal_cx, goal_cy = app.pitch_to_canvas(view, GOAL_X, GOAL_Y)
    deep_cx, deep_cy = app.pitch_to_canvas(view, GOAL_X - 20.0, GOAL_Y)
    assert goal_cy < deep_cy  # nearer the goal is higher up the canvas
    assert goal_cx == pytest.approx(deep_cx)
    assert goal_cx == pytest.approx(view.width_px / 2.0, abs=1.0)

    # pitch y increases left to right on screen
    left_cx, _ = app.pitch_to_canvas(view, GOAL_X, 5.0)
    right_cx, _ = app.pitch_to_canvas(view, GOAL_X, 60.0)
    assert left_cx < right_cx


def test_view_pixel_dimensions_follow_the_scale() -> None:
    view = app.PitchView(x_min=70.0, x_max=105.0, y_min=0.0, y_max=68.0, px_per_metre=10.0)
    assert view.width_px == 680
    assert view.height_px == 350


def test_clamp_to_pitch() -> None:
    assert app.clamp_to_pitch(-5.0, -3.0) == (0.0, 0.0)
    assert app.clamp_to_pitch(200.0, 900.0) == (PITCH_LENGTH, PITCH_WIDTH)
    assert app.clamp_to_pitch(50.0, 20.0) == (50.0, 20.0)


# --------------------------------------------------------------------------- #
# pitch markings
# --------------------------------------------------------------------------- #
def test_markings_have_the_real_dimensions() -> None:
    markings = {m.name: m for m in app.pitch_markings()}
    assert set(markings) >= {
        "goal_line",
        "goal",
        "penalty_area",
        "six_yard_box",
        "penalty_spot",
        "penalty_arc",
        "corner_arc_near",
        "corner_arc_far",
    }

    goal_line = markings["goal_line"].points
    assert all(px == pytest.approx(GOAL_X) for px, _ in goal_line)
    assert min(py for _, py in goal_line) == pytest.approx(0.0)
    assert max(py for _, py in goal_line) == pytest.approx(PITCH_WIDTH)

    box = markings["penalty_area"].points
    assert min(px for px, _ in box) == pytest.approx(GOAL_X - 16.5)
    assert max(py for _, py in box) - min(py for _, py in box) == pytest.approx(40.32)

    six = markings["six_yard_box"].points
    assert min(px for px, _ in six) == pytest.approx(GOAL_X - 5.5)
    assert max(py for _, py in six) - min(py for _, py in six) == pytest.approx(18.32)

    goal = markings["goal"].points
    assert {round(py, 2) for _, py in goal} == {round(POST_LEFT_Y, 2), round(POST_RIGHT_Y, 2)}
    assert max(px for px, _ in goal) > GOAL_X  # the net sits behind the line

    spot = markings["penalty_spot"].points
    spot_x = (min(px for px, _ in spot) + max(px for px, _ in spot)) / 2.0
    spot_y = (min(py for _, py in spot) + max(py for _, py in spot)) / 2.0
    assert spot_x == pytest.approx(GOAL_X - 11.0, abs=1e-6)
    assert spot_y == pytest.approx(GOAL_Y, abs=1e-6)

    arc = markings["penalty_arc"].points
    radii = [math.hypot(px - (GOAL_X - 11.0), py - GOAL_Y) for px, py in arc]
    assert max(radii) == pytest.approx(9.15, abs=1e-6)
    assert min(radii) == pytest.approx(9.15, abs=1e-6)
    assert max(px for px, _ in arc) <= GOAL_X - 16.5 + 1e-6  # the D stays outside the box


def test_markings_are_clipped_to_the_view() -> None:
    view = app.PitchView(x_min=80.0)
    for marking in app.pitch_markings(view):
        for px, _ in marking.points:
            assert px >= view.x_min - 1e-9


# --------------------------------------------------------------------------- #
# canvas objects and dragging
# --------------------------------------------------------------------------- #
def test_initial_drawing_is_one_object_per_player(frame: CornerFrame) -> None:
    scene = app.initial_drawing(frame)
    assert len(scene["objects"]) == len(frame.players)
    assert {o["type"] for o in scene["objects"]} == {"circle"}
    for obj in scene["objects"]:
        cx, cy = app.object_centre_px(obj)
        assert -50.0 <= cx <= app.DEFAULT_VIEW.width_px + 50.0
        assert -50.0 <= cy <= app.DEFAULT_VIEW.height_px + 50.0


def test_object_centre_handles_both_origins() -> None:
    left_origin = {"type": "circle", "left": 100.0, "top": 40.0, "radius": 10.0}
    centred = {
        "type": "circle",
        "left": 110.0,
        "top": 50.0,
        "radius": 10.0,
        "originX": "center",
        "originY": "center",
    }
    assert app.object_centre_px(left_origin) == (110.0, 50.0)
    assert app.object_centre_px(centred) == (110.0, 50.0)

    scaled = {"type": "circle", "left": 0.0, "top": 0.0, "radius": 10.0, "scaleX": 2.0, "scaleY": 3.0}
    assert app.object_centre_px(scaled) == (20.0, 30.0)


@pytest.mark.parametrize("view", VIEWS)
def test_untouched_canvas_reproduces_the_frame(view: app.PitchView, frame: CornerFrame) -> None:
    scene = app.initial_drawing(frame, view)
    players = app.players_from_canvas(frame, scene, view)
    assert len(players) == len(frame.players)
    for original, restored in zip(frame.players, players, strict=True):
        assert restored.x == pytest.approx(original.x, abs=1e-6)
        assert restored.y == pytest.approx(original.y, abs=1e-6)
        assert restored.is_attacker == original.is_attacker
        assert restored.is_taker == original.is_taker
        assert restored.is_keeper == original.is_keeper


def test_drag_moves_exactly_one_player(frame: CornerFrame) -> None:
    view = app.DEFAULT_VIEW
    scene = app.initial_drawing(frame, view)
    idx = 4
    shift_px = 3.0 * view.px_per_metre  # three metres to the right on screen
    scene["objects"][idx]["left"] += shift_px

    players = app.players_from_canvas(frame, scene, view)
    moved = players[idx]
    assert moved.y == pytest.approx(frame.players[idx].y + 3.0, abs=1e-6)
    assert moved.x == pytest.approx(frame.players[idx].x, abs=1e-6)
    for i, (before, after) in enumerate(zip(frame.players, players, strict=True)):
        if i == idx:
            continue
        assert (after.x, after.y) == pytest.approx((before.x, before.y), abs=1e-6)


def test_drag_off_the_canvas_is_clamped_onto_the_pitch(frame: CornerFrame) -> None:
    view = app.DEFAULT_VIEW
    scene = app.initial_drawing(frame, view)
    scene["objects"][2]["left"] = -10_000.0
    scene["objects"][3]["top"] = -10_000.0
    players = app.players_from_canvas(frame, scene, view)
    for player in players:
        assert 0.0 <= player.x <= PITCH_LENGTH
        assert 0.0 <= player.y <= PITCH_WIDTH


def test_players_from_canvas_rejects_a_desynchronised_scene(frame: CornerFrame) -> None:
    scene = app.initial_drawing(frame)
    scene["objects"] = scene["objects"][:-2]
    with pytest.raises(ValueError, match="resynchronise"):
        app.players_from_canvas(frame, scene)


def test_players_from_canvas_ignores_non_circle_objects(frame: CornerFrame) -> None:
    scene = app.initial_drawing(frame)
    scene["objects"].append({"type": "path", "left": 5.0, "top": 5.0})
    players = app.players_from_canvas(frame, scene)
    assert len(players) == len(frame.players)


def test_apply_drag(frame: CornerFrame) -> None:
    moved = app.apply_drag(frame, 2, 101.0, 25.0)
    assert (moved.players[2].x, moved.players[2].y) == (101.0, 25.0)
    assert moved.players[2].is_attacker is frame.players[2].is_attacker
    assert frame.players[2].x == 99.0  # the original frame is untouched

    clamped = app.apply_drag(frame, 2, 999.0, -4.0)
    assert (clamped.players[2].x, clamped.players[2].y) == (PITCH_LENGTH, 0.0)

    with pytest.raises(IndexError):
        app.apply_drag(frame, 99, 50.0, 30.0)


def test_canvas_key_changes_with_revision_corner_and_view() -> None:
    assert app.canvas_key("abc", 0) != app.canvas_key("abc", 1)
    assert app.canvas_key("abc", 1) != app.canvas_key("xyz", 1)
    assert app.canvas_key("abc", 1) == app.canvas_key("abc", 1)
    zoomed = app.PitchView(px_per_metre=app.DEFAULT_VIEW.px_per_metre + 1.0)
    assert app.canvas_key("abc", 1, zoomed) != app.canvas_key("abc", 1, app.DEFAULT_VIEW)


def test_canvas_background_support_is_available() -> None:
    import streamlit_drawable_canvas as canvas_module

    assert app.ensure_canvas_background_support() is True
    assert callable(canvas_module.st_image.image_to_url)


def test_full_width_kwargs_is_a_single_supported_flag() -> None:
    kwargs = app.full_width_kwargs()
    assert len(kwargs) == 1
    assert set(kwargs) <= {"width", "use_container_width"}


# --------------------------------------------------------------------------- #
# styling and labels
# --------------------------------------------------------------------------- #
def test_roles_are_visually_distinct(frame: CornerFrame) -> None:
    roles = {app.player_role(p) for p in frame.players}
    assert roles == {"taker", "keeper", "attacker", "defender"}
    colours = {app.player_style(p).colour for p in frame.players}
    assert len(colours) == 4  # one colour per role, none shared


def test_player_label(frame: CornerFrame) -> None:
    assert app.player_label(frame, 0) == "A0"
    assert app.player_label(frame, 1) == "D1"


def test_formatters() -> None:
    assert app.format_probability(0.1234) == "12.3%"
    assert app.format_delta(-0.042) == "-4.2 pp"
    assert app.format_delta(0.042) == "+4.2 pp"


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def test_build_figure_without_scores(real_frame: CornerFrame) -> None:
    fig = app.build_figure(real_frame)
    assert isinstance(fig, go.Figure)
    marker_traces = [t for t in fig.data if t.name != "pitch"]
    assert sum(len(t.x) for t in marker_traces) == len(real_frame.players)
    assert fig.layout.yaxis.range == (app.DEFAULT_VIEW.x_min, app.DEFAULT_VIEW.x_max)
    assert fig.layout.xaxis.range == (app.DEFAULT_VIEW.y_min, app.DEFAULT_VIEW.y_max)


def test_build_figure_with_scores_rings_the_top_receiver(frame: CornerFrame) -> None:
    probs = np.zeros(len(frame.players))
    probs[3] = 0.6
    probs[2] = 0.1
    scores = app.Scores(receiver_probs=probs, p_defence=0.3, shot_prob=0.25)
    fig = app.build_figure(frame, scores=scores, title="scored")
    names = [t.name for t in fig.data]
    assert "most likely receiver" in names
    ring = fig.data[names.index("most likely receiver")]
    assert (float(ring.x[0]), float(ring.y[0])) == (frame.players[3].y, frame.players[3].x)
    assert fig.layout.title.text == "scored"


def test_build_figure_draws_one_annotation_per_move(frame: CornerFrame) -> None:
    moves = [Move(player_idx=6, dx=1.0, dy=-2.0), Move(player_idx=8, dx=-1.5, dy=0.5)]
    arrows = app.move_arrows(frame, moves)
    fig = app.build_figure(frame, arrows=arrows)
    assert len(fig.layout.annotations) == 2
    first = fig.layout.annotations[0]
    assert (first.ax, first.ay) == (frame.players[6].y, frame.players[6].x)
    assert (first.x, first.y) == (frame.players[6].y - 2.0, frame.players[6].x + 1.0)


def test_positions_signature_tracks_moves(frame: CornerFrame) -> None:
    assert app.positions_signature(frame) == app.positions_signature(frame)
    moved = app.apply_drag(frame, 5, frame.players[5].x + 0.5, frame.players[5].y)
    assert app.positions_signature(moved) != app.positions_signature(frame)


def test_move_arrows_skip_out_of_range_indices(frame: CornerFrame) -> None:
    arrows = app.move_arrows(frame, [Move(player_idx=999, dx=1.0, dy=1.0)])
    assert arrows == []


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("view", VIEWS)
def test_pitch_image_matches_the_canvas_size(view: app.PitchView) -> None:
    img = app.render_pitch_image(view)
    assert img.size == (view.width_px, view.height_px)


def test_render_png_returns_a_png(real_frame: CornerFrame) -> None:
    data = app.render_png(real_frame)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) > 2000


def test_render_png_includes_scores_and_arrows(frame: CornerFrame) -> None:
    probs = np.zeros(len(frame.players))
    probs[2] = 0.5
    scores = app.Scores(receiver_probs=probs, p_defence=0.5, shot_prob=0.2)
    arrows = app.move_arrows(frame, [Move(player_idx=6, dx=2.0, dy=1.0)])
    plain = app.render_png(frame)
    decorated = app.render_png(frame, scores=scores, arrows=arrows)
    assert decorated[:8] == b"\x89PNG\r\n\x1a\n"
    assert decorated != plain


# --------------------------------------------------------------------------- #
# corner sources
# --------------------------------------------------------------------------- #
def test_load_uploaded_csv() -> None:
    csv = b"team,x,y,is_taker,is_keeper\nattack,105,0.3,1,0\ndefend,104,34,0,1\nattack,98,32,0,0\n"
    frame = app.load_uploaded_corner("corner.csv", csv)
    assert len(frame.players) == 3
    assert frame.players[0].is_taker
    assert frame.players[1].is_keeper
    assert not frame.players[1].is_attacker


def test_load_uploaded_statsbomb_json() -> None:
    payload = {
        "event_uuid": "abc-123",
        "freeze_frame": [
            {"location": [120.0, 0.5], "teammate": True, "actor": True, "keeper": False},
            {"location": [118.0, 40.0], "teammate": False, "actor": False, "keeper": True},
            {"location": [112.0, 36.0], "teammate": True, "actor": False, "keeper": False},
        ],
    }
    frame = app.load_uploaded_corner("three-sixty.json", json.dumps(payload).encode())
    assert frame.corner_id == "abc-123"
    assert len(frame.players) == 3
    assert frame.players[0].x == pytest.approx(105.0)
    assert frame.players[1].is_keeper


def test_load_uploaded_rejects_other_suffixes() -> None:
    with pytest.raises(ValueError, match="unsupported upload"):
        app.load_uploaded_corner("frame.txt", b"nonsense")


def test_load_uploaded_reports_bad_json() -> None:
    with pytest.raises(ValueError):
        app.load_uploaded_corner("bad.json", json.dumps({"nope": 1}).encode())


def test_demo_corners_load() -> None:
    corners = app.demo_corners(limit=3)
    if not corners:
        pytest.skip("no bundled demo corners and no extracted dataset")
    assert len(corners) <= 3
    for corner in corners:
        assert corner.players
        assert app.corner_caption(corner)


def test_jsonl_reader_skips_a_half_written_line(tmp_path: Path) -> None:
    good = json.dumps(
        {
            "corner_id": "x",
            "players": [{"x": 100.0, "y": 30.0, "is_attacker": True}],
        }
    )
    path = tmp_path / "partial.jsonl"
    path.write_text(good + "\n" + good[: len(good) // 2])
    corners = app._read_jsonl_corners(path, limit=10)
    assert len(corners) == 1


# --------------------------------------------------------------------------- #
# models and scoring
# --------------------------------------------------------------------------- #
def test_missing_model_message_is_actionable(tmp_path: Path) -> None:
    bundle = app.load_bundle(str(tmp_path / "nope.pt"), str(tmp_path / "nope.pt"))
    assert not bundle.ready
    message = app.missing_model_message(bundle)
    assert "setpiece train" in message
    assert "receiver" in message and "shot" in message
    assert bundle.errors


def test_resolve_model_paths(tmp_path: Path) -> None:
    assert app.resolve_model_paths(None) == (None, None)
    assert app.resolve_model_paths("") == (None, None)

    (tmp_path / "receiver.pt").write_bytes(b"")
    assert app.resolve_model_paths(str(tmp_path)) == (str(tmp_path / "receiver.pt"), None)

    single = tmp_path / "shot.pt"
    single.write_bytes(b"")
    assert app.resolve_model_paths(str(single)) == (str(single), str(single))


def test_model_arg_from_argv() -> None:
    assert app.model_arg_from_argv(["--model", "/tmp/x.pt"]) == "/tmp/x.pt"
    assert app.model_arg_from_argv(["--model=/tmp/y.pt"]) == "/tmp/y.pt"
    assert app.model_arg_from_argv([]) is None
    assert app.model_arg_from_argv(["--model"]) is None


def test_score_frame_returns_none_without_models(frame: CornerFrame) -> None:
    assert app.score_frame(app.ModelBundle(), frame) is None


def _untrained_bundle() -> app.ModelBundle:
    from setpiece.models import ReceiverModel, ShotModel

    return app.ModelBundle(receiver=ReceiverModel(), shot=ShotModel())


def test_score_frame_round_trip_is_fast_enough(real_frame: CornerFrame) -> None:
    bundle = _untrained_bundle()
    app.score_frame(bundle, real_frame)  # warm up lazy imports

    start = time.perf_counter()
    scores = app.score_frame(bundle, real_frame)
    elapsed = time.perf_counter() - start

    assert scores is not None
    assert len(scores.receiver_probs) == len(real_frame.players)
    assert scores.receiver_probs.sum() + scores.p_defence == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= scores.shot_prob <= 1.0
    assert elapsed < 1.0


def test_scoring_reacts_to_a_drag(real_frame: CornerFrame) -> None:
    bundle = _untrained_bundle()
    before = app.score_frame(bundle, real_frame)
    dragged = app.apply_drag(real_frame, real_frame.attacker_indices[0], 101.0, 33.0)
    after = app.score_frame(bundle, dragged)
    assert before is not None and after is not None
    assert not np.allclose(before.receiver_probs, after.receiver_probs)


def test_receiver_rows_are_ranked(frame: CornerFrame) -> None:
    probs = np.zeros(len(frame.players))
    probs[2] = 0.4
    probs[3] = 0.1
    probs[4] = 0.2
    scores = app.Scores(receiver_probs=probs, p_defence=0.3, shot_prob=0.2)
    rows = app.receiver_rows(frame, scores, top=4)

    assert [row["player"] for row in rows][:2] == ["A2", "defence wins first contact"]
    assert [float(row["p"]) for row in rows] == sorted(
        (float(row["p"]) for row in rows), reverse=True
    )
    assert all(row["role"] != "defender" for row in rows)


def test_top_receiver_index(frame: CornerFrame) -> None:
    probs = np.zeros(len(frame.players))
    scores = app.Scores(receiver_probs=probs, p_defence=1.0, shot_prob=0.1)
    assert app.top_receiver_index(frame, scores) is None
    probs[5] = 0.7
    assert app.top_receiver_index(frame, scores) == 5


# --------------------------------------------------------------------------- #
# torch thread pinning
# --------------------------------------------------------------------------- #
def _pin_in_a_fresh_interpreter(env_overrides: dict[str, str]) -> tuple[int, int]:
    """Import the app in a clean process, pin, and report the thread count.

    A subprocess is the only honest way to check the ``OMP_NUM_THREADS`` branch:
    torch reads that variable once, when it is imported, so it has to be set
    before the interpreter starts. Nothing here launches a Streamlit server; it
    imports the module and calls one function.
    """
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    env.update(env_overrides)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; from setpiece import app; "
            "print(app.pin_torch_threads(), torch.get_num_threads())",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    returned, effective = proc.stdout.strip().splitlines()[-1].split()
    return (int(returned), int(effective))


def test_pin_torch_threads_pins_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    before = torch.get_num_threads()
    try:
        assert app.pin_torch_threads() == 1
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(before)


def test_pin_torch_threads_respects_an_explicit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An OMP_NUM_THREADS the user set is left alone, whatever its value."""
    import torch

    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)

    def _refuse(_: int) -> None:
        raise AssertionError("pin_torch_threads overrode an explicit OMP_NUM_THREADS")

    monkeypatch.setattr(torch, "set_num_threads", _refuse)
    assert app.pin_torch_threads() == torch.get_num_threads()


@pytest.mark.slow  # about 4 s: a second interpreter has to import torch
def test_pin_torch_threads_survives_a_fresh_interpreter() -> None:
    """The pin has to work in the process ``streamlit run`` actually starts."""
    returned, effective = _pin_in_a_fresh_interpreter({})
    assert (returned, effective) == (1, 1)


@pytest.mark.slow
def test_an_explicit_thread_count_reaches_a_fresh_interpreter() -> None:
    returned, effective = _pin_in_a_fresh_interpreter({"OMP_NUM_THREADS": "2"})
    assert (returned, effective) == (2, 2)


def test_main_pins_before_it_touches_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` pins on every Streamlit rerun, before any scoring happens.

    Streamlit re-executes the script top to bottom on each interaction, so the
    pin cannot live at import time only. This drives ``main`` far enough to see
    the call and then stops it, without starting a server.
    """
    calls: list[str] = []

    def _record_pin() -> int:
        calls.append("pin")
        return 1

    class _StopEarlyError(Exception):
        pass

    def _stop(*_args: object, **_kwargs: object) -> None:
        calls.append("page_config")
        raise _StopEarlyError

    monkeypatch.setattr(app, "pin_torch_threads", _record_pin)
    monkeypatch.setattr(app.st, "set_page_config", _stop)
    with pytest.raises(_StopEarlyError):
        app.main()
    assert calls == ["pin", "page_config"]


def test_static_note_is_visible_text() -> None:
    assert "no velocities" in app.STATIC_NOTE
    assert "static" in app.STATIC_NOTE
