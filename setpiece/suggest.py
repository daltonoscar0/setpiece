"""Defender adjustment search.

Given a corner freeze frame and a shot model, find small repositionings of the
defending team that lower (or, with ``defend=False``, raise) the modelled
probability of a shot in the five events after the delivery.

The search is a greedy beam search over a fixed, deterministic set of candidate
displacements. Every hard constraint is enforced *inside* the search — an
inadmissible displacement is never scored, let alone returned:

* only defenders move, at most ``movers`` of them, each at most once;
* every displacement has euclidean length ``<= max_move``;
* a moved player never ends up closer than ``min_sep`` to any other player,
  attackers and unmoved defenders included;
* moved players stay inside the 105x68 pitch.

Freeze frames carry no velocities, so a "move" here is a static repositioning of
the defensive shape at the moment the corner is struck, not a run. That is the
``--assume-static`` caveat: the model has no notion of whether a defender could
actually cover the ground.

Note on pre-existing overlaps: real StatsBomb 360 frames sometimes place two
players less than ``min_sep`` apart. The search never *creates* such a pair, but
it cannot repair one between two players it does not move.

Cost. The number of shot-model evaluations is bounded by ``max_evals``, plus one
for the base frame, and the bound is enforced by a counter in the search loop
rather than argued for. The branching factor is laid out so that the default
budget is spent rather than truncated: 16 candidate displacements, 6 defenders
tried for the first move and 2 for each move after that, and a beam that narrows
by half at each level. For the default ``movers=3`` and ``beam=8`` that is

    6*16  +  4*2*16  +  2*2*16  =  96 + 128 + 64 = 288

evaluations, which at the ~3 ms an untrained ``ShotModel`` costs per corner on
one CPU thread is a little under a second — the round trip the pitch editor
allows. :func:`required_evals` computes that sum for any layout, and 288 is
exactly ``required_evals()`` at the defaults.

Widening the search and the budget are two separate knobs, and they interact:
``max_candidates=None`` on a frame with ten movable defenders wants
``10*16 + 4*2*16 + 2*2*16 = 352`` evaluations, so leaving ``max_evals`` at 288
truncates the search rather than widening it. The search degrades gracefully —
expansions are enumerated in beam rank order, best states first, and stop when
the budget runs out — but the caller asked for a wider search and did not get
one. :func:`suggest` therefore warns whenever the branching was widened past
:data:`MAX_CANDIDATE_DEFENDERS` / :data:`DEEP_CANDIDATE_DEFENDERS` and the
budget cannot cover it, naming the number of evaluations needed. A deliberately
small ``max_evals`` at the default branching is a supported, quiet use — that is
what ``setpiece demo`` does — so it is not warned about. ``setpiece suggest``
sizes ``max_evals`` from the layout with :func:`required_evals` unless
``--max-evals`` says otherwise, so widening the flags there really does widen
the search.
"""

from __future__ import annotations

import math
import random
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from setpiece.types import GOAL_X, GOAL_Y, PITCH_LENGTH, PITCH_WIDTH, CornerFrame

#: Floating point slack for the length, separation and bounds checks.
EPS: float = 1e-9

#: Compass directions per radius ring in the candidate displacement set.
N_DIRECTIONS: int = 8

#: Radii of the candidate rings, as fractions of ``max_move``.
RADIUS_FRACTIONS: tuple[float, ...] = (1.0, 0.5)

#: Default number of defenders considered when expanding the base frame, ranked
#: by how close they already are to the goal being attacked. Bounds the
#: branching factor; lift it with ``suggest(..., max_candidates=...)`` or
#: ``setpiece suggest --max-candidates``.
MAX_CANDIDATE_DEFENDERS: int = 6

#: The same, for expansions below the first move. Deeper levels are narrower
#: because the beam is already carrying the promising first moves.
DEEP_CANDIDATE_DEFENDERS: int = 2

#: Default ceiling on shot-model evaluations for one :func:`suggest` call.
#: Equal to ``required_evals()`` at the default layout.
DEFAULT_MAX_EVALS: int = 288


@dataclass(frozen=True)
class Move:
    """A single defender displacement, in metres."""

    player_idx: int
    dx: float
    dy: float

    @property
    def length(self) -> float:
        return math.hypot(self.dx, self.dy)


@dataclass
class Suggestion:
    """A set of moves and what the shot model makes of them.

    ``delta`` is ``new_prob - base_prob``, so a negative delta is a defensive
    improvement.
    """

    moves: list[Move]
    base_prob: float
    new_prob: float
    delta: float


ShotScorer = Callable[[CornerFrame], float]


# --------------------------------------------------------------------------- #
# shot model plumbing
# --------------------------------------------------------------------------- #
def _is_torch_module(obj: Any) -> bool:
    """True if ``obj`` is a ``torch.nn.Module``, without importing torch."""
    for cls in type(obj).__mro__:
        if f"{cls.__module__}.{cls.__qualname__}" == "torch.nn.modules.module.Module":
            return True
    return False


def _resolve_scorer(shot_model: Any) -> ShotScorer:
    """Adapt whatever we were handed into ``frame -> probability``.

    Accepted, in order: an object exposing ``shot_probability(frame)``; a
    ``torch.nn.Module``, which is run through :func:`setpiece.models.shot_probability`;
    any other callable, which is called directly with the frame. The last form is
    what tests and notebooks use.
    """
    method = getattr(shot_model, "shot_probability", None)
    if callable(method):
        return lambda frame: float(method(frame))
    if _is_torch_module(shot_model):
        from setpiece.models import shot_probability

        return lambda frame: float(shot_probability(shot_model, frame))
    if callable(shot_model):
        return lambda frame: float(shot_model(frame))
    raise TypeError(
        "shot_model must be callable, expose shot_probability(frame), "
        f"or be a torch Module; got {type(shot_model).__name__}"
    )


# --------------------------------------------------------------------------- #
# candidate displacements
# --------------------------------------------------------------------------- #
def candidate_displacements(max_move: float, seed: int = 0) -> tuple[tuple[float, float], ...]:
    """The fixed candidate displacement set: two rings of compass directions.

    The rings sit at ``max_move`` and ``max_move / 2``; the inner ring is rotated
    by half a step so the two rings do not share bearings. ``seed`` only sets a
    global phase offset, so the set is deterministic for a given seed and every
    member has length ``<= max_move``.

    A ``max_move`` that is not a positive finite number has no admissible
    displacement at all, so the set is empty. Returning a ring anyway would hand
    back either no-op moves (``max_move == 0``) or, for a negative budget, moves
    of length ``abs(max_move)`` that breach the very bound they are meant to
    respect.
    """
    if not math.isfinite(max_move) or max_move <= 0.0:
        return ()
    rng = random.Random(seed)
    step = 2.0 * math.pi / N_DIRECTIONS
    phase = rng.random() * step
    out: list[tuple[float, float]] = []
    for ring, frac in enumerate(RADIUS_FRACTIONS):
        radius = max_move * frac
        for i in range(N_DIRECTIONS):
            angle = phase + step * i + 0.5 * step * ring
            dx = radius * math.cos(angle)
            dy = radius * math.sin(angle)
            length = math.hypot(dx, dy)
            if length > max_move:  # trim the rounding fuzz off the outer ring
                scale = max_move / length
                dx, dy = dx * scale, dy * scale
            out.append((dx, dy))
    return tuple(out)


# --------------------------------------------------------------------------- #
# search state
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _State:
    moves: tuple[Move, ...]
    xs: tuple[float, ...]
    ys: tuple[float, ...]
    prob: float

    @property
    def moved(self) -> frozenset[int]:
        return frozenset(m.player_idx for m in self.moves)


def _move_key(moves: tuple[Move, ...]) -> tuple[tuple[int, float, float], ...]:
    """Order-independent, hashable identity for a set of moves."""
    return tuple(sorted((m.player_idx, round(m.dx, 9), round(m.dy, 9)) for m in moves))


def _admissible(
    state: _State,
    idx: int,
    nx: float,
    ny: float,
    min_sep: float,
) -> bool:
    """Pitch bounds and separation against every other player in ``state``."""
    if not (0.0 <= nx <= PITCH_LENGTH):
        return False
    if not (0.0 <= ny <= PITCH_WIDTH):
        return False
    for j in range(len(state.xs)):
        if j == idx:
            continue
        if math.hypot(nx - state.xs[j], ny - state.ys[j]) < min_sep - EPS:
            return False
    return True


def _beam_width(beam: int, depth: int) -> int:
    """How many states survive the expansion at ``depth``.

    The beam narrows geometrically — ``beam/2`` states after the first move,
    ``beam/4`` after the second and so on — which is what keeps the total number
    of shot-model evaluations inside the default budget while still letting the
    search spend all of ``movers``. ``beam`` itself remains the number of
    suggestions handed back.
    """
    return max(1, beam >> (depth + 1))


#: Stand-in width for ``max_candidates=None`` when the frame is not to hand. A
#: freeze frame holds at most 22 players, so no side can field more than this
#: many movable outfield defenders.
_UNBOUNDED_CANDIDATES: int = 10


def required_evals(
    *,
    movers: int = 3,
    beam: int = 8,
    max_candidates: int | None = MAX_CANDIDATE_DEFENDERS,
    deep_candidates: int | None = DEEP_CANDIDATE_DEFENDERS,
    n_displacements: int | None = None,
    n_movable: int | None = None,
) -> int:
    """Evaluations :func:`suggest` would spend at this layout if nothing were pruned.

    This is the exact arithmetic the search loop performs, not an estimate: one
    root state expanded over ``max_candidates`` defenders, then
    ``_beam_width(beam, depth-1)`` states expanded over ``deep_candidates``
    defenders at every level below, each expansion trying every candidate
    displacement. It is an *upper* bound on the real count, because a
    displacement that leaves the pitch, breaches ``min_sep`` or repeats a state
    already scored is skipped without costing an evaluation.

    ``required_evals()`` with no arguments is :data:`DEFAULT_MAX_EVALS`, which is
    how that constant was derived. Pass ``n_movable`` to tighten the bound for a
    specific frame: a frame with four movable defenders cannot spend a candidate
    width of ten.

    A ``None`` candidate width means "every movable defender", which is only a
    finite number when ``n_movable`` is given; without it, the width falls back
    to the number of defenders a full freeze frame could plausibly carry
    (:data:`_UNBOUNDED_CANDIDATES`), so the answer stays finite and generous.
    """
    if n_displacements is None:
        n_displacements = N_DIRECTIONS * len(RADIUS_FRACTIONS)
    depth = movers if n_movable is None else min(movers, n_movable)
    total = 0
    for d in range(max(0, depth)):
        width = max_candidates if d == 0 else deep_candidates
        if width is None:
            width = n_movable if n_movable is not None else _UNBOUNDED_CANDIDATES
        if n_movable is not None:
            width = min(width, n_movable - d)
        if width <= 0:
            break
        states = 1 if d == 0 else max(1, beam >> d)
        total += states * width * n_displacements
    return total


def _movable(frame: CornerFrame, move_keeper: bool) -> list[int]:
    """Defender indices, ranked by distance to the goal being attacked.

    Attackers and the taker are never movable. The keeper is a defender but is
    excluded by default: repositioning the goalkeeper is not the kind of advice
    this tool is for.
    """
    idxs = [
        i
        for i, p in enumerate(frame.players)
        if not p.is_attacker and not p.is_taker and (move_keeper or not p.is_keeper)
    ]

    def rank(i: int) -> tuple[float, int]:
        p = frame.players[i]
        return (math.hypot(GOAL_X - p.x, GOAL_Y - p.y), i)

    idxs.sort(key=rank)
    return idxs


def _warn_if_budget_truncates_a_widened_search(
    *,
    movers: int,
    beam: int,
    max_candidates: int | None,
    deep_candidates: int | None,
    n_displacements: int,
    n_movable: int,
    max_evals: int,
) -> None:
    """Say so when a caller widened the branching but left the budget behind.

    Only fires for a *widened* search. Shrinking ``max_evals`` at the default
    branching is a deliberate latency trade — ``setpiece demo`` does exactly
    that — and warning about it would train people to ignore the warning.
    """
    widened = (
        max_candidates is None
        or max_candidates > MAX_CANDIDATE_DEFENDERS
        or deep_candidates is None
        or deep_candidates > DEEP_CANDIDATE_DEFENDERS
    )
    if not widened:
        return
    need = required_evals(
        movers=movers,
        beam=beam,
        max_candidates=max_candidates,
        deep_candidates=deep_candidates,
        n_displacements=n_displacements,
        n_movable=n_movable,
    )
    if need <= max_evals:
        return
    warnings.warn(
        f"max_evals={max_evals} cannot cover max_candidates="
        f"{'all' if max_candidates is None else max_candidates}, deep_candidates="
        f"{'all' if deep_candidates is None else deep_candidates} on a frame with "
        f"{n_movable} movable defenders: that layout needs up to {need} evaluations, "
        f"so the search will be truncated rather than widened. Pass "
        f"max_evals=required_evals(...) to spend the wider branching.",
        RuntimeWarning,
        stacklevel=3,
    )


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def suggest(
    frame: CornerFrame,
    shot_model: Any,
    *,
    movers: int = 3,
    max_move: float = 3.0,
    min_sep: float = 0.5,
    beam: int = 8,
    seed: int = 0,
    defend: bool = True,
    max_evals: int = DEFAULT_MAX_EVALS,
    move_keeper: bool = False,
    top_k: int | None = None,
    max_candidates: int | None = MAX_CANDIDATE_DEFENDERS,
    deep_candidates: int | None = DEEP_CANDIDATE_DEFENDERS,
) -> list[Suggestion]:
    """Search for defender repositionings that move the modelled shot probability.

    Args:
        frame: the corner to work on. Left untouched.
        shot_model: anything :func:`_resolve_scorer` accepts — a trained
            ``ShotModel``, an object with ``shot_probability(frame)``, or a plain
            ``frame -> float`` callable.
        movers: maximum number of defenders repositioned. Each moves at most once.
        max_move: maximum euclidean length of one displacement, in metres.
        min_sep: minimum distance between any two players after the moves.
        beam: beam width, and the default number of suggestions returned.
        seed: fixes the phase of the candidate displacement set. Identical inputs
            and seed give a byte-identical result, ordering included.
        defend: ``True`` minimises the shot probability, ``False`` maximises it
            (useful for showing an attacking coach where the shape breaks).
        max_evals: ceiling on shot-model evaluations, excluding the base frame.
            It does not grow with ``max_candidates``; see the note below.
        move_keeper: allow the goalkeeper to be repositioned.
        top_k: how many suggestions to return. Defaults to ``beam``.
        max_candidates: how many defenders, nearest the attacked goal first, are
            tried for the *first* move. ``None`` means every movable defender;
            ``0`` means none, so no suggestion exists.
        deep_candidates: the same for every move below the first, where the beam
            is already carrying the promising openings. ``0`` restricts the
            search to single-defender adjustments.

    Returns:
        Suggestions ranked best first: by ``delta`` ascending when ``defend`` is
        True, descending when it is False. Every returned suggestion has at least
        one move, and every move has non-zero length. An empty list means no
        admissible move exists — a frame with no movable defender, ``movers <= 0``,
        ``max_candidates == 0``, a ``max_move`` that is not positive and finite,
        or a shape so tight that every candidate displacement breaches
        ``min_sep`` or the touchline.

    Budget and branching: ``max_evals`` is an independent ceiling, so widening
    ``max_candidates`` or ``deep_candidates`` without raising it truncates the
    search instead of widening it — the extra defenders are enumerated but the
    counter runs out before they are reached. Size the budget with
    :func:`required_evals`, which returns exactly what a layout can spend. When
    the branching is widened past :data:`MAX_CANDIDATE_DEFENDERS` /
    :data:`DEEP_CANDIDATE_DEFENDERS` and ``max_evals`` cannot cover it, a
    ``RuntimeWarning`` says so and names the figure; a small ``max_evals`` at the
    default branching is a supported, quiet way to buy latency back.

    Pre-existing overlaps: if two players are already closer than ``min_sep`` in
    ``frame``, the search will not repair the pair unless it happens to move one
    of them. What it guarantees is the stronger, checkable half — every *moved*
    player ends at least ``min_sep`` from every other player, so no returned
    suggestion ever creates a tight pair or leaves one touching a mover.
    """
    scorer = _resolve_scorer(shot_model)
    base_prob = scorer(frame)

    limit = beam if top_k is None else top_k
    if limit <= 0 or movers <= 0 or beam <= 0 or max_evals <= 0:
        return []
    if max_candidates is not None and max_candidates <= 0:
        return []
    if not math.isfinite(max_move) or max_move <= 0.0:
        return []

    movable = _movable(frame, move_keeper)
    if not movable:
        return []

    displacements = candidate_displacements(max_move, seed)
    _warn_if_budget_truncates_a_widened_search(
        movers=movers,
        beam=beam,
        max_candidates=max_candidates,
        deep_candidates=deep_candidates,
        n_displacements=len(displacements),
        n_movable=len(movable),
        max_evals=max_evals,
    )
    sign = 1.0 if defend else -1.0

    root = _State(
        moves=(),
        xs=tuple(p.x for p in frame.players),
        ys=tuple(p.y for p in frame.players),
        prob=base_prob,
    )

    frontier: list[_State] = [root]
    results: dict[tuple[tuple[int, float, float], ...], _State] = {}
    evals = 0

    for depth in range(min(movers, len(movable))):
        width = max_candidates if depth == 0 else deep_candidates
        if width is not None and width <= 0:
            break
        children: list[_State] = []
        for state in frontier:
            if evals >= max_evals:
                break
            already = state.moved
            available = [i for i in movable if i not in already]
            pool = available if width is None else available[:width]
            for idx in pool:
                if evals >= max_evals:
                    break
                for dx, dy in displacements:
                    if evals >= max_evals:
                        break
                    nx = state.xs[idx] + dx
                    ny = state.ys[idx] + dy
                    if not _admissible(state, idx, nx, ny, min_sep):
                        continue
                    moves = (*state.moves, Move(player_idx=idx, dx=dx, dy=dy))
                    key = _move_key(moves)
                    if key in results:
                        continue
                    xs = state.xs[:idx] + (nx,) + state.xs[idx + 1 :]
                    ys = state.ys[:idx] + (ny,) + state.ys[idx + 1 :]
                    # score the frame the caller would get by replaying `moves`,
                    # so the returned moves and the scored shape cannot drift
                    prob = scorer(apply_moves(frame, list(moves)))
                    evals += 1
                    child = _State(moves=moves, xs=xs, ys=ys, prob=prob)
                    results[key] = child
                    children.append(child)
        if not children:
            break
        children.sort(key=lambda s: (sign * s.prob, _move_key(s.moves)))
        frontier = children[: _beam_width(beam, depth)]

    ranked = sorted(results.values(), key=lambda s: (sign * s.prob, _move_key(s.moves)))
    return [
        Suggestion(
            moves=list(s.moves),
            base_prob=base_prob,
            new_prob=s.prob,
            delta=s.prob - base_prob,
        )
        for s in ranked[:limit]
    ]


def apply_moves(frame: CornerFrame, moves: list[Move]) -> CornerFrame:
    """Return a copy of ``frame`` with ``moves`` applied. Used by the CLI and app."""
    players = list(frame.players)
    for m in moves:
        players[m.player_idx] = players[m.player_idx].moved(m.dx, m.dy)
    return frame.with_players(players)
