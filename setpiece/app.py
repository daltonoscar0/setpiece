"""Streamlit pitch editor for corner kicks.

Drag the players around an attacking-third pitch view and watch the receiver
distribution and the shot probability move with them, then ask for a defensive
adjustment and see the proposed moves as arrows.

Everything that is not a Streamlit call lives in a plain function in this
module: the view geometry, the canvas-pixel <-> pitch-metre conversions, the
drag application, the figure builder and the PNG renderer. ``main`` is a thin
shell over those, which is what makes the interesting half testable without
starting a server.

Two caveats the UI repeats, because they bound what the numbers mean:

* freeze frames carry **no velocities**, so scoring treats the frame as static.
  A dragged player is a different *shape*, not a run, and the model has no idea
  whether anyone could actually cover the ground;
* ``meta["end_xy"]`` and ``meta["pass_outcome"]`` are label-adjacent and are
  never shown as inputs; nothing here feeds them to a model.
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_drawable_canvas import st_canvas

from setpiece._io import corner_from_dict, load_corner
from setpiece.suggest import Move, Suggestion, apply_moves, suggest
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

__all__ = [
    "DEFAULT_VIEW",
    "Arrow",
    "Marking",
    "ModelBundle",
    "PitchView",
    "Scores",
    "apply_drag",
    "build_figure",
    "canvas_key",
    "canvas_to_pitch",
    "clamp_to_pitch",
    "demo_corners",
    "ensure_canvas_background_support",
    "format_delta",
    "format_probability",
    "initial_drawing",
    "load_bundle",
    "load_uploaded_corner",
    "main",
    "missing_model_message",
    "model_arg_from_argv",
    "move_arrows",
    "object_centre_px",
    "pin_torch_threads",
    "pitch_markings",
    "pitch_to_canvas",
    "player_label",
    "player_style",
    "players_from_canvas",
    "positions_signature",
    "receiver_rows",
    "render_pitch_image",
    "render_png",
    "resolve_model_paths",
    "score_frame",
]

# --------------------------------------------------------------------------- #
# pitch dimensions not already in types.py
# --------------------------------------------------------------------------- #
SIX_YARD_DEPTH: float = 5.5
SIX_YARD_HALF_WIDTH: float = 9.16
PENALTY_DEPTH: float = 16.5
PENALTY_HALF_WIDTH: float = 20.16
PENALTY_SPOT_DEPTH: float = 11.0
PENALTY_ARC_RADIUS: float = 9.15
CORNER_ARC_RADIUS: float = 1.0
GOAL_DEPTH: float = 1.8

# --------------------------------------------------------------------------- #
# palette
# --------------------------------------------------------------------------- #
GRASS_DARK: str = "#2e6b41"
GRASS_LIGHT: str = "#347a4a"
LINE_COLOUR: str = "#eaf3ec"
ATTACKER_COLOUR: str = "#e8533a"
DEFENDER_COLOUR: str = "#3f7fe0"
KEEPER_COLOUR: str = "#f4c542"
TAKER_COLOUR: str = "#b64ad0"
ARROW_COLOUR: str = "#ffd166"

#: The banner the app never stops showing.
STATIC_NOTE: str = (
    "Freeze frames carry no velocities. Scoring assumes a single static frame: "
    "dragging a player changes the shape, not anyone's momentum, and the model "
    "cannot tell whether a defender could actually cover the ground."
)


# --------------------------------------------------------------------------- #
# view geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PitchView:
    """The rectangle of pitch shown, and how many pixels a metre is worth.

    The canvas is drawn with the goal at the **top**: screen x runs along pitch
    ``y`` (left to right) and screen y runs along pitch ``x`` inverted, so the
    goal line at ``x = 105`` sits at the top edge and the attacking team plays
    up the screen. The mapping is a pure affine scale, so
    :func:`canvas_to_pitch` and :func:`pitch_to_canvas` invert each other
    exactly.
    """

    x_min: float = 66.0
    x_max: float = 107.0
    y_min: float = -2.0
    y_max: float = 70.0
    px_per_metre: float = 11.0

    @property
    def width_px(self) -> int:
        return int(round((self.y_max - self.y_min) * self.px_per_metre))

    @property
    def height_px(self) -> int:
        return int(round((self.x_max - self.x_min) * self.px_per_metre))

    @property
    def key(self) -> tuple[float, float, float, float, float]:
        return (self.x_min, self.x_max, self.y_min, self.y_max, self.px_per_metre)


DEFAULT_VIEW = PitchView()


def pitch_to_canvas(view: PitchView, x: float, y: float) -> tuple[float, float]:
    """Pitch metres ``(x, y)`` to canvas pixels ``(cx, cy)``."""
    cx = (y - view.y_min) * view.px_per_metre
    cy = (view.x_max - x) * view.px_per_metre
    return (cx, cy)


def canvas_to_pitch(view: PitchView, cx: float, cy: float) -> tuple[float, float]:
    """Canvas pixels ``(cx, cy)`` back to pitch metres ``(x, y)``."""
    y = view.y_min + cx / view.px_per_metre
    x = view.x_max - cy / view.px_per_metre
    return (x, y)


def clamp_to_pitch(x: float, y: float) -> tuple[float, float]:
    """Clamp a position onto the 105x68 playing area."""
    return (
        min(max(float(x), 0.0), PITCH_LENGTH),
        min(max(float(y), 0.0), PITCH_WIDTH),
    )


# --------------------------------------------------------------------------- #
# pitch markings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Marking:
    """One pitch marking as a polyline in pitch metres."""

    name: str
    points: tuple[tuple[float, float], ...]
    width: float = 2.0


def _arc(
    cx: float,
    cy: float,
    radius: float,
    start_deg: float,
    end_deg: float,
    steps: int = 32,
) -> tuple[tuple[float, float], ...]:
    """Sample a circular arc in pitch coordinates, centre ``(cx, cy)``."""
    return tuple(
        (
            cx + radius * math.cos(math.radians(start_deg + (end_deg - start_deg) * i / steps)),
            cy + radius * math.sin(math.radians(start_deg + (end_deg - start_deg) * i / steps)),
        )
        for i in range(steps + 1)
    )


def pitch_markings(view: PitchView = DEFAULT_VIEW) -> list[Marking]:
    """Every line drawn on the attacking third, clipped to ``view``.

    Coordinates are metres on the 105x68 pitch, so the goal line is at
    ``x = GOAL_X`` and the posts sit at ``POST_LEFT_Y`` / ``POST_RIGHT_Y``.
    """
    deep = max(view.x_min, 0.0)
    box_x = GOAL_X - PENALTY_DEPTH
    six_x = GOAL_X - SIX_YARD_DEPTH
    spot = (GOAL_X - PENALTY_SPOT_DEPTH, GOAL_Y)

    # the D: the sweep of the penalty arc that lies outside the penalty area,
    # measured either side of the 180 degree bearing (straight away from goal)
    cos_limit = (box_x - spot[0]) / PENALTY_ARC_RADIUS
    half = 180.0 - math.degrees(math.acos(max(-1.0, min(1.0, cos_limit))))

    markings: list[Marking] = [
        Marking("touchline_near", ((deep, 0.0), (GOAL_X, 0.0))),
        Marking("touchline_far", ((deep, PITCH_WIDTH), (GOAL_X, PITCH_WIDTH))),
        Marking("goal_line", ((GOAL_X, 0.0), (GOAL_X, PITCH_WIDTH))),
        Marking(
            "penalty_area",
            (
                (GOAL_X, GOAL_Y - PENALTY_HALF_WIDTH),
                (box_x, GOAL_Y - PENALTY_HALF_WIDTH),
                (box_x, GOAL_Y + PENALTY_HALF_WIDTH),
                (GOAL_X, GOAL_Y + PENALTY_HALF_WIDTH),
            ),
        ),
        Marking(
            "six_yard_box",
            (
                (GOAL_X, GOAL_Y - SIX_YARD_HALF_WIDTH),
                (six_x, GOAL_Y - SIX_YARD_HALF_WIDTH),
                (six_x, GOAL_Y + SIX_YARD_HALF_WIDTH),
                (GOAL_X, GOAL_Y + SIX_YARD_HALF_WIDTH),
            ),
        ),
        Marking("penalty_spot", _arc(spot[0], spot[1], 0.22, 0.0, 360.0, steps=12), width=2.5),
        Marking("penalty_arc", _arc(spot[0], spot[1], PENALTY_ARC_RADIUS, 180.0 - half, 180.0 + half)),
        Marking("corner_arc_near", _arc(GOAL_X, 0.0, CORNER_ARC_RADIUS, 90.0, 180.0, steps=10)),
        Marking(
            "corner_arc_far",
            _arc(GOAL_X, PITCH_WIDTH, CORNER_ARC_RADIUS, 180.0, 270.0, steps=10),
        ),
        Marking(
            "goal",
            (
                (GOAL_X, POST_LEFT_Y),
                (GOAL_X + GOAL_DEPTH, POST_LEFT_Y),
                (GOAL_X + GOAL_DEPTH, POST_RIGHT_Y),
                (GOAL_X, POST_RIGHT_Y),
            ),
            width=3.5,
        ),
    ]
    return markings


# --------------------------------------------------------------------------- #
# player styling
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlayerStyle:
    role: str
    colour: str
    outline: str
    radius_m: float
    symbol: str


_STYLES: dict[str, PlayerStyle] = {
    "taker": PlayerStyle("taker", TAKER_COLOUR, "#ffffff", 1.05, "diamond"),
    "keeper": PlayerStyle("keeper", KEEPER_COLOUR, "#1b1b1b", 1.1, "square"),
    "attacker": PlayerStyle("attacker", ATTACKER_COLOUR, "#1b1b1b", 1.0, "circle"),
    "defender": PlayerStyle("defender", DEFENDER_COLOUR, "#1b1b1b", 1.0, "circle"),
}


def player_role(player: Player) -> str:
    """``taker``, ``keeper``, ``attacker`` or ``defender``, in that priority."""
    if player.is_taker:
        return "taker"
    if player.is_keeper:
        return "keeper"
    return "attacker" if player.is_attacker else "defender"


def player_style(player: Player) -> PlayerStyle:
    return _STYLES[player_role(player)]


def player_label(frame: CornerFrame, idx: int) -> str:
    """Short cross-reference label, e.g. ``A4`` or ``D11``."""
    player = frame.players[idx]
    return f"{'A' if player.is_attacker else 'D'}{idx}"


# --------------------------------------------------------------------------- #
# canvas objects
# --------------------------------------------------------------------------- #
def _object_radius_px(view: PitchView, player: Player) -> float:
    return player_style(player).radius_m * view.px_per_metre


def initial_drawing(frame: CornerFrame, view: PitchView = DEFAULT_VIEW) -> dict[str, Any]:
    """A fabric.js scene with one draggable circle per player, in frame order.

    ``left``/``top`` are the bounding-box corner, fabric's default origin, so a
    scene we write and a scene fabric hands back are read the same way.
    """
    objects: list[dict[str, Any]] = []
    for idx, player in enumerate(frame.players):
        style = player_style(player)
        radius = _object_radius_px(view, player)
        cx, cy = pitch_to_canvas(view, player.x, player.y)
        objects.append(
            {
                "type": "circle",
                "originX": "left",
                "originY": "top",
                "left": cx - radius,
                "top": cy - radius,
                "radius": radius,
                "fill": style.colour,
                "stroke": style.outline,
                "strokeWidth": 2,
                "scaleX": 1,
                "scaleY": 1,
                "angle": 0,
                "opacity": 0.95,
                "selectable": True,
                "hasControls": False,
                "hasBorders": True,
                "lockRotation": True,
                "lockScalingX": True,
                "lockScalingY": True,
                "name": player_label(frame, idx),
            }
        )
    return {"version": "4.4.0", "objects": objects}


def object_centre_px(obj: dict[str, Any]) -> tuple[float, float]:
    """Centre of a fabric circle in canvas pixels, whatever origin it uses."""
    left = float(obj.get("left", 0.0))
    top = float(obj.get("top", 0.0))
    radius = float(obj.get("radius", 0.0))
    scale_x = float(obj.get("scaleX", 1.0) or 1.0)
    scale_y = float(obj.get("scaleY", 1.0) or 1.0)
    cx = left if obj.get("originX", "left") == "center" else left + radius * scale_x
    cy = top if obj.get("originY", "top") == "center" else top + radius * scale_y
    return (cx, cy)


def _circles(json_data: Any) -> list[dict[str, Any]]:
    objects = json_data.get("objects", []) if isinstance(json_data, dict) else (json_data or [])
    return [o for o in objects if str(o.get("type", "")).lower() == "circle"]


def players_from_canvas(
    frame: CornerFrame,
    json_data: Any,
    view: PitchView = DEFAULT_VIEW,
) -> list[Player]:
    """Read dragged positions back out of a canvas scene.

    Objects are matched to players by position in the scene, which is the order
    :func:`initial_drawing` wrote them in and the order fabric preserves. Team,
    taker and keeper flags come from ``frame``: dragging changes coordinates and
    nothing else. Positions are clamped onto the pitch.
    """
    circles = _circles(json_data)
    if len(circles) != len(frame.players):
        raise ValueError(
            f"canvas holds {len(circles)} player markers but the frame has "
            f"{len(frame.players)}; reload the corner to resynchronise"
        )
    out: list[Player] = []
    for player, obj in zip(frame.players, circles, strict=True):
        cx, cy = object_centre_px(obj)
        x, y = clamp_to_pitch(*canvas_to_pitch(view, cx, cy))
        out.append(replace(player, x=x, y=y))
    return out


def apply_drag(frame: CornerFrame, idx: int, x: float, y: float) -> CornerFrame:
    """Move one player to ``(x, y)``, clamped onto the pitch."""
    if not 0 <= idx < len(frame.players):
        raise IndexError(f"no player {idx} in a frame of {len(frame.players)}")
    cx, cy = clamp_to_pitch(x, y)
    players = list(frame.players)
    players[idx] = replace(players[idx], x=cx, y=cy)
    return frame.with_players(players)


def canvas_key(corner_id: str, revision: int, view: PitchView = DEFAULT_VIEW) -> str:
    """Widget key for the canvas; changing it forces fabric to reload the scene.

    The view is part of the key because the scene is written in pixels: rescale
    the view and the old objects would sit at the wrong metres.
    """
    shape = "x".join(f"{v:g}" for v in view.key)
    return f"pitch-{corner_id}-{revision}-{shape}"


# --------------------------------------------------------------------------- #
# suggestion overlay
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Arrow:
    """A proposed displacement, from ``(x0, y0)`` to ``(x1, y1)`` in metres."""

    x0: float
    y0: float
    x1: float
    y1: float
    label: str


def move_arrows(frame: CornerFrame, moves: list[Move]) -> list[Arrow]:
    """Turn suggested moves into arrows anchored on the current frame."""
    arrows: list[Arrow] = []
    for move in moves:
        if not 0 <= move.player_idx < len(frame.players):
            continue
        player = frame.players[move.player_idx]
        arrows.append(
            Arrow(
                x0=player.x,
                y0=player.y,
                x1=player.x + move.dx,
                y1=player.y + move.dy,
                label=player_label(frame, move.player_idx),
            )
        )
    return arrows


def positions_signature(frame: CornerFrame) -> tuple[tuple[float, float], ...]:
    """Millimetre-rounded positions, used to notice that a shape has changed."""
    return tuple((round(p.x, 3), round(p.y, 3)) for p in frame.players)


def format_probability(p: float) -> str:
    return f"{100.0 * float(p):.1f}%"


def format_delta(delta: float) -> str:
    """A probability change in percentage points, signed."""
    return f"{100.0 * float(delta):+.1f} pp"


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
@dataclass
class Scores:
    """Model output for one frame."""

    receiver_probs: np.ndarray
    p_defence: float
    shot_prob: float


@dataclass
class ModelBundle:
    """The two heads, plus whatever went wrong loading them."""

    receiver: Any = None
    shot: Any = None
    receiver_card: dict = None  # type: ignore[assignment]
    shot_card: dict = None  # type: ignore[assignment]
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.receiver_card is None:
            self.receiver_card = {}
        if self.shot_card is None:
            self.shot_card = {}

    @property
    def ready(self) -> bool:
        return self.receiver is not None and self.shot is not None


def pin_torch_threads() -> int:
    """Score single threaded unless the user asked for something else.

    ``setpiece app`` launches this module as a fresh ``streamlit run``
    subprocess, so ``_tune_torch_threads`` in :mod:`setpiece.cli` never reaches
    it — the child inherits the parent's environment and none of its in-process
    torch settings. A corner graph has twenty-odd nodes; spreading that over the
    machine's cores costs more in synchronisation than it saves, and the
    suggestion search pays that cost hundreds of times over.

    An ``OMP_NUM_THREADS`` or ``MKL_NUM_THREADS`` the user set explicitly wins:
    torch has already read it at import time and somebody who set it meant it.
    Returns the number of threads torch will actually use, so a caller can see
    which of the two happened.
    """
    import os

    import torch

    if not any(os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")):
        torch.set_num_threads(1)
    return torch.get_num_threads()


def model_arg_from_argv(argv: list[str] | None = None) -> str | None:
    """Read ``--model PATH`` out of the script arguments.

    ``setpiece app`` forwards its ``--model`` flag to ``streamlit run`` after a
    ``--`` separator, so the app picks the same checkpoint the CLI would.
    """
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    for i, token in enumerate(args):
        if token == "--model" and i + 1 < len(args):
            return args[i + 1]
        if token.startswith("--model="):
            return token.split("=", 1)[1]
    return None


def resolve_model_paths(model_arg: str | None) -> tuple[str | None, str | None]:
    """Turn a ``--model`` value into ``(receiver_path, shot_path)``.

    A directory is expected to hold ``receiver.pt`` and ``shot.pt``; a single
    file is offered to both heads and :func:`load_bundle` keeps whichever one it
    actually is. ``None`` means "use the bundled checkpoints".
    """
    if not model_arg:
        return (None, None)
    path = Path(model_arg).expanduser()
    if path.is_dir():
        receiver = path / "receiver.pt"
        shot = path / "shot.pt"
        return (str(receiver) if receiver.exists() else None, str(shot) if shot.exists() else None)
    return (str(path), str(path))


def _load_head(path: str | None, name: str, expected: str) -> tuple[Any, dict, str | None]:
    """Load one head; return ``(model, card, error)`` with the model ``None`` on failure."""
    from setpiece import models as _models

    try:
        model, card = _models.load_model(path) if path else _models.load_bundled(name)
    except Exception as exc:  # missing, corrupt, or shape-mismatched checkpoint
        return (None, {}, f"{name} model: {exc}")
    if type(model).__name__ != expected:
        return (
            None,
            {},
            f"{name} model: {path} holds a {type(model).__name__}, not a {expected}",
        )
    return (model, card, None)


def load_bundle(receiver_path: str | None = None, shot_path: str | None = None) -> ModelBundle:
    """Load the receiver and shot checkpoints, collecting failures instead of raising.

    With no paths the bundled ``setpiece/assets/*.pt`` checkpoints are used. A
    head that cannot be loaded comes back as ``None`` with its error recorded,
    so the app can show one actionable message rather than a traceback.
    """
    receiver, receiver_card, receiver_error = _load_head(receiver_path, "receiver", "ReceiverModel")
    shot, shot_card, shot_error = _load_head(shot_path, "shot", "ShotModel")
    return ModelBundle(
        receiver=receiver,
        shot=shot,
        receiver_card=receiver_card,
        shot_card=shot_card,
        errors=tuple(e for e in (receiver_error, shot_error) if e),
    )


def missing_model_message(bundle: ModelBundle) -> str:
    """What to tell the user when a head is missing. Empty when both loaded."""
    if bundle.ready:
        return ""
    missing = []
    if bundle.receiver is None:
        missing.append("receiver")
    if bundle.shot is None:
        missing.append("shot")
    lines = [
        f"No trained {' and '.join(missing)} model available, so the pitch is "
        "editable but nothing can be scored.",
        "",
        "Train one, which writes the checkpoints into setpiece/assets/:",
        "",
        "    uv run setpiece train",
        "",
        "or point the sidebar at an existing checkpoint with --model / the "
        "checkpoint path box.",
    ]
    if bundle.errors:
        lines += ["", "Loader said:"] + [f"- {e}" for e in bundle.errors]
    return "\n".join(lines)


def score_frame(bundle: ModelBundle, frame: CornerFrame) -> Scores | None:
    """Score a frame, or ``None`` if either head is missing."""
    if not bundle.ready:
        return None
    from setpiece import models as _models

    probs, p_defence = _models.receiver_probabilities(bundle.receiver, frame)
    shot = _models.shot_probability(bundle.shot, frame)
    return Scores(receiver_probs=np.asarray(probs, dtype=float), p_defence=p_defence, shot_prob=shot)


def receiver_rows(
    frame: CornerFrame,
    scores: Scores,
    top: int = 6,
) -> list[dict[str, Any]]:
    """Ranked receiver table: candidate attackers plus the defence-wins class."""
    rows: list[dict[str, Any]] = []
    for idx, player in enumerate(frame.players):
        p = float(scores.receiver_probs[idx])
        if p <= 0.0:
            continue
        rows.append(
            {
                "player": player_label(frame, idx),
                "role": player_role(player),
                "x": round(player.x, 1),
                "y": round(player.y, 1),
                "p": p,
            }
        )
    rows.append(
        {
            "player": "defence wins first contact",
            "role": "defence",
            "x": float("nan"),
            "y": float("nan"),
            "p": float(scores.p_defence),
        }
    )
    rows.sort(key=lambda r: (-float(r["p"]), str(r["player"])))
    return rows[:top]


def top_receiver_index(frame: CornerFrame, scores: Scores) -> int | None:
    """Index of the most likely receiver, or ``None`` when nobody is a candidate."""
    if not len(scores.receiver_probs):
        return None
    idx = int(np.argmax(scores.receiver_probs))
    return idx if scores.receiver_probs[idx] > 0.0 else None


# --------------------------------------------------------------------------- #
# plotly figure
# --------------------------------------------------------------------------- #
def _screen(x: float, y: float) -> tuple[float, float]:
    """Pitch metres to figure axes: horizontal is pitch y, vertical is pitch x."""
    return (y, x)


def build_figure(
    frame: CornerFrame,
    *,
    view: PitchView = DEFAULT_VIEW,
    scores: Scores | None = None,
    arrows: list[Arrow] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Build the read-only pitch figure: markings, players, labels, arrows.

    ``scores`` adds each candidate's receive probability to the hover text and
    rings the most likely receiver. ``arrows`` overlays proposed moves.
    """
    fig = go.Figure()

    line_x: list[float | None] = []
    line_y: list[float | None] = []
    for marking in pitch_markings(view):
        for px, py in marking.points:
            sx, sy = _screen(px, py)
            line_x.append(sx)
            line_y.append(sy)
        line_x.append(None)
        line_y.append(None)
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line={"color": LINE_COLOUR, "width": 2},
            hoverinfo="skip",
            showlegend=False,
            name="pitch",
        )
    )

    by_role: dict[str, list[int]] = {}
    for idx, player in enumerate(frame.players):
        by_role.setdefault(player_role(player), []).append(idx)

    for role in ("defender", "keeper", "attacker", "taker"):
        idxs = by_role.get(role, [])
        if not idxs:
            continue
        style = _STYLES[role]
        xs, ys, text, hover = [], [], [], []
        for idx in idxs:
            player = frame.players[idx]
            sx, sy = _screen(player.x, player.y)
            xs.append(sx)
            ys.append(sy)
            text.append(player_label(frame, idx))
            p = None if scores is None else float(scores.receiver_probs[idx])
            hover.append(
                f"{player_label(frame, idx)} ({role})<br>"
                f"x={player.x:.1f} m, y={player.y:.1f} m"
                + ("" if p is None else f"<br>P(first contact) = {format_probability(p)}")
            )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker={
                    "size": 22,
                    "color": style.colour,
                    "symbol": style.symbol,
                    "line": {"color": style.outline, "width": 2},
                },
                text=text,
                textposition="middle center",
                textfont={"color": "#10130f", "size": 9},
                hovertext=hover,
                hoverinfo="text",
                name=role,
            )
        )

    if scores is not None:
        best = top_receiver_index(frame, scores)
        if best is not None:
            sx, sy = _screen(frame.players[best].x, frame.players[best].y)
            fig.add_trace(
                go.Scatter(
                    x=[sx],
                    y=[sy],
                    mode="markers",
                    marker={
                        "size": 40,
                        "color": "rgba(0,0,0,0)",
                        "symbol": "circle-open",
                        "line": {"color": "#ffffff", "width": 3},
                    },
                    hoverinfo="skip",
                    showlegend=False,
                    name="most likely receiver",
                )
            )

    for arrow in arrows or []:
        x0, y0 = _screen(arrow.x0, arrow.y0)
        x1, y1 = _screen(arrow.x1, arrow.y1)
        fig.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=3,
            arrowcolor=ARROW_COLOUR,
            text="",
        )

    fig.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        plot_bgcolor=GRASS_DARK,
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 10, "r": 10, "t": 36 if title else 10, "b": 10},
        height=int(view.height_px),
        # the legend sits inside the plot, on the empty grass at the back of the
        # view, so it never fights the title for the top strip
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 0.01,
            "xanchor": "left",
            "x": 0.02,
            "bgcolor": "rgba(20,24,15,0.55)",
            "font": {"size": 11},
        },
        dragmode="pan",
        font={"color": "#e9efe9"},
    )
    fig.update_xaxes(
        range=[view.y_min, view.y_max],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        constrain="domain",
    )
    fig.update_yaxes(
        range=[view.x_min, view.x_max],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
    )
    return fig


# --------------------------------------------------------------------------- #
# PIL rendering: canvas background and PNG export
# --------------------------------------------------------------------------- #
def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Helvetica.ttc", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _draw_markings(draw: ImageDraw.ImageDraw, view: PitchView, dy: float = 0.0) -> None:
    for marking in pitch_markings(view):
        pts = [pitch_to_canvas(view, px, py) for px, py in marking.points]
        draw.line(
            [(cx, cy + dy) for cx, cy in pts],
            fill=LINE_COLOUR,
            width=max(1, int(round(marking.width))),
            joint="curve",
        )


def render_pitch_image(view: PitchView = DEFAULT_VIEW) -> Image.Image:
    """The empty pitch, exactly ``view.width_px`` x ``view.height_px``.

    Used as the canvas background, so its size must match the canvas exactly.
    """
    img = Image.new("RGB", (view.width_px, view.height_px), GRASS_DARK)
    draw = ImageDraw.Draw(img)

    stripe = 6.0 * view.px_per_metre
    n = int(math.ceil(view.height_px / stripe))
    for i in range(n):
        if i % 2:
            top = i * stripe
            draw.rectangle([0, top, view.width_px, top + stripe], fill=GRASS_LIGHT)

    _draw_markings(draw, view)
    return img


def _draw_players(
    draw: ImageDraw.ImageDraw,
    frame: CornerFrame,
    view: PitchView,
    scores: Scores | None,
    dy: float,
) -> None:
    label_font = _font(11)
    for idx, player in enumerate(frame.players):
        style = player_style(player)
        radius = _object_radius_px(view, player)
        cx, cy = pitch_to_canvas(view, player.x, player.y)
        cy += dy
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=style.colour,
            outline=style.outline,
            width=2,
        )
        draw.text(
            (cx, cy),
            player_label(frame, idx),
            fill="#10130f",
            font=label_font,
            anchor="mm",
        )
        if scores is not None:
            p = float(scores.receiver_probs[idx])
            if p > 0.0:
                draw.text(
                    (cx, cy - radius - 8),
                    format_probability(p),
                    fill="#f4f8f4",
                    font=label_font,
                    anchor="mm",
                )


def _draw_arrows(
    draw: ImageDraw.ImageDraw,
    view: PitchView,
    arrows: list[Arrow],
    dy: float,
) -> None:
    for arrow in arrows:
        x0, y0 = pitch_to_canvas(view, arrow.x0, arrow.y0)
        x1, y1 = pitch_to_canvas(view, arrow.x1, arrow.y1)
        y0 += dy
        y1 += dy
        draw.line([(x0, y0), (x1, y1)], fill=ARROW_COLOUR, width=3)
        angle = math.atan2(y1 - y0, x1 - x0)
        head = 9.0
        for sign in (1, -1):
            theta = angle + sign * math.radians(150.0)
            draw.line(
                [(x1, y1), (x1 + head * math.cos(theta), y1 + head * math.sin(theta))],
                fill=ARROW_COLOUR,
                width=3,
            )


def render_png(
    frame: CornerFrame,
    *,
    view: PitchView = DEFAULT_VIEW,
    scores: Scores | None = None,
    arrows: list[Arrow] | None = None,
    title: str = "setpiece - corner",
) -> bytes:
    """Render the current pitch state to PNG bytes.

    A header band carries the title, the shot probability and the static-frame
    caveat; the pitch below is the same drawing the canvas uses.
    """
    header = 58
    pitch = render_pitch_image(view)
    img = Image.new("RGB", (view.width_px, view.height_px + header), "#14180f")
    img.paste(pitch, (0, header))
    draw = ImageDraw.Draw(img)

    draw.text((12, 8), title, fill="#f4f8f4", font=_font(15), anchor="la")
    line = "no trained model loaded" if scores is None else (
        f"P(shot within 5 events) = {format_probability(scores.shot_prob)}   |   "
        f"P(defence wins first contact) = {format_probability(scores.p_defence)}"
    )
    draw.text((12, 28), line, fill="#cfe0cf", font=_font(12), anchor="la")
    draw.text(
        (12, 44),
        "static frame, no velocities",
        fill="#93a893",
        font=_font(11),
        anchor="la",
    )

    _draw_players(draw, frame, view, scores, dy=header)
    if arrows:
        _draw_arrows(draw, view, arrows, dy=header)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# corner sources
# --------------------------------------------------------------------------- #
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEMO_DIR: Path = Path(__file__).resolve().parent / "assets" / "demo"
INTERIM_FILES: tuple[Path, ...] = (
    REPO_ROOT / "data" / "interim" / "smoke.jsonl",
    REPO_ROOT / "data" / "interim" / "corners.jsonl",
)


def _read_jsonl_corners(path: Path, limit: int) -> list[CornerFrame]:
    """Read up to ``limit`` corners, tolerating a half-written final line."""
    out: list[CornerFrame] = []
    try:
        with path.open() as fh:
            for line in fh:
                if len(out) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(corner_from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue
    except OSError:
        return []
    return out


def demo_corners(limit: int = 12) -> list[CornerFrame]:
    """Bundled corners for the demo picker.

    Prefers ``setpiece/assets/demo/*.json``, then whatever extracted dataset is
    sitting in ``data/interim``. Returns an empty list if nothing is available,
    which the app reports rather than crashing on.
    """
    out: list[CornerFrame] = []
    if DEMO_DIR.is_dir():
        for path in sorted(DEMO_DIR.glob("*.json")):
            if len(out) >= limit:
                break
            try:
                out.append(load_corner(path))
            except (ValueError, OSError, json.JSONDecodeError):
                continue
    for path in INTERIM_FILES:
        if len(out) >= limit:
            break
        if path.exists():
            out.extend(_read_jsonl_corners(path, limit - len(out)))
    return out[:limit]


def corner_caption(frame: CornerFrame) -> str:
    """One line describing where a corner came from."""
    bits = [b for b in (frame.competition, frame.season) if b]
    if frame.minute is not None:
        bits.append(f"{frame.minute}'")
    bits.append(f"{frame.n_attackers()} att / {frame.n_defenders()} def")
    return " - ".join(bits)


def load_uploaded_corner(name: str, data: bytes) -> CornerFrame:
    """Parse an uploaded file through :func:`setpiece._io.load_corner`.

    The loader is path-based and handles StatsBomb 360 JSON, our own JSON and
    the simple CSV, so the upload is spooled to a temp file with its original
    suffix and handed over unchanged.
    """
    suffix = Path(name).suffix.lower() or ".json"
    if suffix not in {".json", ".csv"}:
        raise ValueError(f"unsupported upload {name!r}: expected a .json or .csv corner")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        frame = load_corner(path)
    if not frame.players:
        raise ValueError(f"no players parsed from {name!r}")
    return replace(frame, corner_id=frame.corner_id or Path(name).stem)


# --------------------------------------------------------------------------- #
# streamlit shell
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _cached_bundle(receiver_path: str | None, shot_path: str | None) -> ModelBundle:
    # ``main`` pins first on every rerun, but this is the one place that touches
    # torch weights, and a cached resource outlives whichever run created it.
    pin_torch_threads()
    return load_bundle(receiver_path, shot_path)


@st.cache_data(show_spinner=False)
def _cached_pitch_png(key: tuple[float, float, float, float, float]) -> bytes:
    view = PitchView(*key)
    buf = BytesIO()
    render_pitch_image(view).save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def _cached_demo(limit: int) -> list[CornerFrame]:
    return demo_corners(limit)


def _background_image(view: PitchView) -> Image.Image:
    return Image.open(BytesIO(_cached_pitch_png(view.key)))


class _ImageUrlShim:
    """Adapter for the helper ``streamlit-drawable-canvas`` expects to import.

    The canvas (0.9.3) calls ``streamlit.elements.image.image_to_url(image,
    width, ...)``. Streamlit later moved that function to
    ``streamlit.elements.lib.image_utils`` and replaced the integer width with a
    ``LayoutConfig``. This forwards the old call to the new function so the
    pitch background still renders; without it the canvas raises and the editor
    would have nothing under the players.
    """

    @staticmethod
    def image_to_url(
        image: Any,
        width: Any,
        clamp: bool,
        channels: str,
        output_format: str,
        image_id: str,
    ) -> str:
        from streamlit.elements.lib import image_utils
        from streamlit.elements.lib.layout_utils import LayoutConfig

        layout = LayoutConfig(width=width) if isinstance(width, int) else width
        return image_utils.image_to_url(image, layout, clamp, channels, output_format, image_id)


def full_width_kwargs() -> dict[str, Any]:
    """How to ask for a full-width element on the installed Streamlit.

    ``use_container_width`` was deprecated in favour of ``width="stretch"`` in
    Streamlit 1.49; the project supports back to 1.33, so pick per version
    rather than emit a deprecation warning on every rerun.
    """
    try:
        major, minor = (int(part) for part in st.__version__.split(".")[:2])
    except (ValueError, AttributeError):
        return {"use_container_width": True}
    return {"width": "stretch"} if (major, minor) >= (1, 49) else {"use_container_width": True}


_canvas_background_ok: bool | None = None


def ensure_canvas_background_support() -> bool:
    """Make the canvas able to show a background image; report whether it can.

    Returns ``False`` if this Streamlit is too different to bridge, in which
    case the editor falls back to a flat green canvas and still works.
    """
    global _canvas_background_ok
    if _canvas_background_ok is not None:
        return _canvas_background_ok

    import streamlit_drawable_canvas as canvas_module

    try:
        if not hasattr(canvas_module.st_image, "image_to_url"):
            from streamlit.elements.lib import image_utils  # noqa: F401

            canvas_module.st_image = _ImageUrlShim
        _canvas_background_ok = True
    except Exception:
        _canvas_background_ok = False
    return _canvas_background_ok


def _sidebar_frame() -> CornerFrame | None:
    """Corner picker: bundled demo, uploaded StatsBomb 360 JSON, or CSV."""
    st.sidebar.header("corner")
    source = st.sidebar.radio(
        "source",
        ("bundled demo", "upload JSON / CSV"),
        key="source",
        label_visibility="collapsed",
    )

    if source == "bundled demo":
        corners = _cached_demo(12)
        if not corners:
            st.sidebar.error(
                "No bundled corners found. Build a dataset first:\n\n"
                "    uv run setpiece fetch --max-matches 5"
            )
            return None
        labels = [f"{i + 1}. {corner_caption(c)}" for i, c in enumerate(corners)]
        chosen = st.sidebar.selectbox("bundled corner", labels, key="demo_pick")
        return corners[labels.index(chosen)]

    upload = st.sidebar.file_uploader(
        "StatsBomb 360 JSON, setpiece JSON, or CSV (team,x,y,is_taker[,is_keeper])",
        type=["json", "csv"],
        key="upload",
    )
    if upload is None:
        st.sidebar.info("Waiting for a file. Coordinates are metres on 105x68, attacking +x.")
        return None
    try:
        return load_uploaded_corner(upload.name, upload.getvalue())
    except Exception as exc:
        st.sidebar.error(f"Could not read {upload.name}: {exc}")
        return None


def _sidebar_view() -> PitchView:
    st.sidebar.header("view")
    scale = st.sidebar.slider("pixels per metre", 8.0, 16.0, 11.0, step=0.5, key="scale")
    depth = st.sidebar.slider("metres of pitch shown", 25.0, 55.0, 41.0, step=1.0, key="depth")
    return PitchView(x_min=GOAL_X + GOAL_DEPTH - depth, px_per_metre=scale)


def _reset(frame: CornerFrame) -> None:
    st.session_state["frame"] = frame
    st.session_state["revision"] = int(st.session_state.get("revision", 0)) + 1
    st.session_state["suggestion"] = None


def main() -> None:
    """Entry point. Launched by ``setpiece app`` via ``streamlit run``.

    Streamlit re-executes this module top to bottom on every interaction, so the
    thread pin is re-applied here rather than at import: it is idempotent and
    costs nothing, and it has to happen before anything in the rerun scores a
    frame or runs the search.
    """
    pin_torch_threads()
    st.set_page_config(page_title="setpiece pitch editor", layout="wide")
    st.title("setpiece - corner pitch editor")
    st.caption(STATIC_NOTE)

    view = _sidebar_view()

    st.sidebar.header("model")
    model_arg = st.sidebar.text_input(
        "checkpoint file or directory (blank = bundled)",
        value=model_arg_from_argv() or "",
        key="model_path",
    )
    receiver_path, shot_path = resolve_model_paths(model_arg.strip() or None)
    bundle = _cached_bundle(receiver_path, shot_path)

    picked = _sidebar_frame()
    if picked is None:
        st.info("Pick a bundled corner or upload one to start.")
        return

    switched = st.session_state.get("corner_id") != picked.corner_id
    if switched or st.session_state.get("frame") is None:
        st.session_state["corner_id"] = picked.corner_id
        _reset(picked)

    base: CornerFrame = st.session_state["frame"]
    revision = int(st.session_state.get("revision", 0))

    if not bundle.ready:
        st.error(missing_model_message(bundle))

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("drag the players")
        st.caption(
            "Grab any marker and drop it. Attackers are red, defenders blue, "
            "the keeper yellow, the corner taker purple. The goal is at the top "
            "of the view, at x = 105 m."
        )
        has_background = ensure_canvas_background_support()
        result = st_canvas(
            background_image=_background_image(view) if has_background else None,
            background_color=GRASS_DARK,
            initial_drawing=initial_drawing(base, view),
            drawing_mode="transform",
            update_streamlit=True,
            display_toolbar=False,
            height=view.height_px,
            width=view.width_px,
            key=canvas_key(str(picked.corner_id), revision, view),
        )

        live = base
        if result is not None and result.json_data:
            try:
                live = base.with_players(players_from_canvas(base, result.json_data, view))
            except ValueError as exc:
                st.warning(str(exc))

        cols = st.columns(3)
        if cols[0].button("reset positions"):
            _reset(picked)
            st.rerun()
        if cols[1].button("commit drag"):
            _reset(live)
            st.rerun()

    started = time.perf_counter()
    scores = score_frame(bundle, live)
    scored_ms = 1000.0 * (time.perf_counter() - started)

    st.sidebar.header("suggest")
    movers = st.sidebar.slider("defenders moved", 1, 5, 3, key="movers")
    max_move = st.sidebar.slider("max move (m)", 1.0, 6.0, 3.0, step=0.5, key="maxmove")
    seed = st.sidebar.number_input("seed", value=0, step=1, key="seed")

    with right:
        st.subheader("scores")
        if scores is None:
            st.warning("Scoring is off until a trained model is available.")
        else:
            metric_cols = st.columns(2)
            metric_cols[0].metric("P(shot within 5 events)", format_probability(scores.shot_prob))
            metric_cols[1].metric(
                "P(defence wins first contact)", format_probability(scores.p_defence)
            )
            st.dataframe(
                [
                    {**row, "p": format_probability(float(row["p"]))}
                    for row in receiver_rows(live, scores)
                ],
                hide_index=True,
                **full_width_kwargs(),
            )
            st.caption(f"re-scored on drop in {scored_ms:.0f} ms (CPU)")

        st.subheader("defensive adjustment")
        disabled = not bundle.ready
        if st.button("suggest defender moves", disabled=disabled, type="primary"):
            with st.spinner("searching"):
                found = suggest(
                    live,
                    bundle.shot,
                    movers=int(movers),
                    max_move=float(max_move),
                    seed=int(seed),
                    defend=True,
                )
            st.session_state["suggestion"] = found[0] if found else None
            st.session_state["suggestion_positions"] = positions_signature(live)
            if not found:
                st.warning("No admissible move found for this shape.")

        suggestion: Suggestion | None = st.session_state.get("suggestion")
        arrows: list[Arrow] = []
        if suggestion is not None:
            arrows = move_arrows(live, suggestion.moves)
            if st.session_state.get("suggestion_positions") != positions_signature(live):
                st.info("The shape has changed since this suggestion. Run it again.")
            st.metric(
                "predicted change in P(shot)",
                format_delta(suggestion.delta),
                delta=format_delta(suggestion.delta),
                delta_color="inverse",
            )
            st.caption(
                f"{format_probability(suggestion.base_prob)} -> "
                f"{format_probability(suggestion.new_prob)} moving "
                f"{len(suggestion.moves)} defender(s), each at most "
                f"{float(max_move):.1f} m. Static frame: the search does not know "
                "whether anyone could get there."
            )
            st.dataframe(
                [
                    {
                        "player": player_label(live, m.player_idx),
                        "dx": round(m.dx, 2),
                        "dy": round(m.dy, 2),
                        "metres": round(m.length, 2),
                    }
                    for m in suggestion.moves
                ],
                hide_index=True,
                **full_width_kwargs(),
            )
            if st.button("apply these moves"):
                _reset(apply_moves(live, suggestion.moves))
                st.rerun()

    st.subheader("scored view")
    st.plotly_chart(
        build_figure(live, view=view, scores=scores, arrows=arrows),
        config={"displayModeBar": False},
        **full_width_kwargs(),
    )

    st.download_button(
        "download PNG",
        data=render_png(
            live,
            view=view,
            scores=scores,
            arrows=arrows,
            title=f"setpiece - corner {picked.corner_id}",
        ),
        file_name=f"setpiece-{picked.corner_id}.png",
        mime="image/png",
    )

    with st.expander("model cards"):
        st.json({"receiver": bundle.receiver_card, "shot": bundle.shot_card})
    st.caption(STATIC_NOTE)


if __name__ == "__main__":
    main()
