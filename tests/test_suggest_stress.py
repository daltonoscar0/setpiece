"""Adversarial stress tests for :mod:`setpiece.suggest`.

:mod:`tests.test_suggest` checks the contract on well-formed frames. This file
tries to break it: hundreds of frames across several seeds, mixing real
StatsBomb corners with shapes chosen to be as hostile as possible — one
defender, none at all, every player stacked on a single point, players already
closer together than ``min_sep`` before anything moves, and players pinned
against the touchline and into the pitch corners.

Two properties get more care here than a straight assertion gives them.

**The pre-existing overlap.** Real 360 frames sometimes place two players 0.2 m
apart. No set of three defender moves can repair every such pair, so "no two
players end up closer than ``min_sep``" cannot be asserted literally against
such a frame. What :func:`setpiece.suggest.suggest` promises instead —
and what :func:`assert_constraints` checks — is the stronger half that *is*
achievable: the search never creates a tight pair, and never leaves one
touching a player it moved. Any pair still inside ``min_sep`` afterwards was
inside it beforehand and involves no mover.

**Determinism.** Checked twice in one process *and* in two subprocesses run
under different ``PYTHONHASHSEED`` values, comparing floats by their exact hex
representation. Set and dict iteration order and any unseeded RNG only show up
across processes.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from setpiece._io import read_corners
from setpiece.suggest import (
    Move,
    Suggestion,
    apply_moves,
    candidate_displacements,
    required_evals,
    suggest,
)
from setpiece.types import PITCH_LENGTH, PITCH_WIDTH, CornerFrame, Player

REPO = Path(__file__).resolve().parents[1]
DATASETS = (
    REPO / "data" / "interim" / "corners.jsonl",
    REPO / "data" / "interim" / "smoke.jsonl",
)

#: Slack allowed on the length, separation and bounds assertions. Wide enough to
#: absorb double rounding, far tighter than any displacement the search returns.
TOL = 1e-6

#: How many real corners to fold into the stress sweep, when a dataset exists.
N_REAL = 60


# --------------------------------------------------------------------------- #
# a deterministic stand-in for the shot model
# --------------------------------------------------------------------------- #
def analytic_scorer(frame: CornerFrame) -> float:
    """A smooth, position-sensitive probability, with no torch and no state.

    The search only needs a total order over shapes. Using a closed-form
    function instead of the real network keeps thousands of calls affordable and
    makes every result reproducible from the source alone.
    """
    total = 0.0
    for p in frame.players:
        total += math.sin(p.x * 0.37) * math.cos(p.y * 0.53)
        total += 0.3 if p.is_attacker else -0.2
    return 1.0 / (1.0 + math.exp(-total / 7.0))


# --------------------------------------------------------------------------- #
# hostile frames
# --------------------------------------------------------------------------- #
def _frame(name: str, players: list[Player]) -> CornerFrame:
    return CornerFrame(corner_id=name, players=players)


def degenerate_frames() -> dict[str, CornerFrame]:
    """Shapes picked to break an assumption rather than to look like football."""
    out: dict[str, CornerFrame] = {}

    # every player on one point: every candidate displacement collides
    out["stacked"] = _frame(
        "stacked", [Player(50.0, 34.0, i < 3, is_taker=(i == 0)) for i in range(8)]
    )
    # the same, jammed into the corner flag
    out["stacked_corner"] = _frame(
        "stacked_corner", [Player(105.0, 68.0, i < 2, is_taker=(i == 0)) for i in range(6)]
    )
    # exactly one defender
    out["one_defender"] = _frame(
        "one_defender",
        [
            Player(105.0, 0.0, True, is_taker=True),
            Player(90.0, 30.0, True),
            Player(95.0, 34.0, False),
        ],
    )
    # no defenders at all
    out["zero_defenders"] = _frame(
        "zero_defenders",
        [Player(105.0, 0.0, True, is_taker=True), Player(90.0, 30.0, True)],
    )
    # the only defender is the keeper, who does not move by default
    out["keeper_only"] = _frame(
        "keeper_only",
        [
            Player(105.0, 0.0, True, is_taker=True),
            Player(90.0, 30.0, True),
            Player(104.0, 34.0, False, is_keeper=True),
        ],
    )
    # the input already breaches min_sep, repeatedly, on both teams
    out["pre_violating"] = _frame(
        "pre_violating",
        [Player(105.0, 0.0, True, is_taker=True)]
        + [Player(95.0 + 0.1 * i, 34.0 + 0.1 * i, i % 2 == 0) for i in range(10)],
    )
    # two defenders exactly coincident
    out["coincident_defenders"] = _frame(
        "coincident_defenders",
        [
            Player(105.0, 0.0, True, is_taker=True),
            Player(96.0, 34.0, True),
            Player(99.0, 30.0, False),
            Player(99.0, 30.0, False),
            Player(101.0, 38.0, False),
        ],
    )
    # hard against every edge and into all four corners
    out["pitch_corners"] = _frame(
        "pitch_corners",
        [
            Player(105.0, 0.0, True, is_taker=True),
            Player(0.0, 0.0, False),
            Player(105.0, 68.0, False),
            Player(0.0, 68.0, False),
            Player(105.0, 34.0, False, is_keeper=True),
            Player(104.9, 67.9, False),
            Player(0.05, 0.05, False),
            Player(1.0, 1.0, True),
            Player(50.0, 34.0, True),
            Player(60.0, 10.0, True),
            Player(60.0, 60.0, False),
        ],
    )
    # a whole defensive line flat on the goal line
    out["goal_line"] = _frame(
        "goal_line",
        [Player(105.0, 68.0, True, is_taker=True)]
        + [Player(105.0, 26.0 + 1.2 * i, False) for i in range(9)]
        + [Player(98.0 + 0.7 * i, 34.0, True) for i in range(6)],
    )
    # a defender flagged as the taker: still never movable
    out["defending_taker"] = _frame(
        "defending_taker",
        [
            Player(105.0, 0.0, False, is_taker=True),
            Player(94.0, 34.0, True),
            Player(97.0, 30.0, False),
            Player(99.0, 40.0, False),
        ],
    )
    return out


def random_hostile_frames(n: int, seed: int = 20240) -> dict[str, CornerFrame]:
    """Randomised frames with wildly varying sizes, densities and clustering."""
    rng = random.Random(seed)
    out: dict[str, CornerFrame] = {}
    for k in range(n):
        spread = rng.choice([0.3, 1.0, 3.0, 12.0, 30.0])
        cx = rng.uniform(88.0, 105.0)
        cy = rng.uniform(20.0, 48.0)
        players = [Player(105.0, 0.0 if k % 2 else 68.0, True, is_taker=True)]
        for _ in range(rng.randint(1, 10)):
            players.append(
                Player(
                    min(max(rng.gauss(cx, spread), 0.0), PITCH_LENGTH),
                    min(max(rng.gauss(cy, spread), 0.0), PITCH_WIDTH),
                    True,
                )
            )
        for i in range(rng.randint(0, 11)):
            players.append(
                Player(
                    min(max(rng.gauss(cx, spread), 0.0), PITCH_LENGTH),
                    min(max(rng.gauss(cy, spread), 0.0), PITCH_WIDTH),
                    False,
                    is_keeper=(i == 0),
                )
            )
        out[f"random{k:03d}"] = _frame(f"random{k:03d}", players)
    return out


def real_frames() -> dict[str, CornerFrame]:
    """Real corners, if a dataset has been built. Empty otherwise."""
    for path in DATASETS:
        if path.exists():
            return {
                f"real{i:03d}": f for i, f in enumerate(read_corners(path)[:N_REAL])
            }
    return {}


ALL_FRAMES: dict[str, CornerFrame] = {
    **degenerate_frames(),
    **random_hostile_frames(90),
    **real_frames(),
}
FRAME_NAMES = sorted(ALL_FRAMES)


def test_the_sweep_covers_more_than_a_hundred_frames() -> None:
    assert len(ALL_FRAMES) >= 100, f"only {len(ALL_FRAMES)} frames in the sweep"


# --------------------------------------------------------------------------- #
# the constraint checker
# --------------------------------------------------------------------------- #
def _tight_pairs(frame: CornerFrame, min_sep: float) -> set[tuple[int, int]]:
    """Index pairs closer than ``min_sep``, using a tolerance the search cannot hide in."""
    out: set[tuple[int, int]] = set()
    ps = frame.players
    for i, j in itertools.combinations(range(len(ps)), 2):
        if math.hypot(ps[i].x - ps[j].x, ps[i].y - ps[j].y) < min_sep - TOL:
            out.add((i, j))
    return out


def assert_constraints(
    frame: CornerFrame,
    suggestions: list[Suggestion],
    *,
    movers: int,
    max_move: float,
    min_sep: float,
    tag: str,
) -> None:
    """Every hard constraint, checked against the frame the caller would build."""
    before = _tight_pairs(frame, min_sep)

    for rank, s in enumerate(suggestions):
        where = f"{tag} rank {rank}"
        assert s.moves, f"{where}: empty suggestion"

        idxs = [m.player_idx for m in s.moves]
        assert len(idxs) == len(set(idxs)), f"{where}: a player moved twice"
        assert len(idxs) <= movers, f"{where}: {len(idxs)} movers, budget {movers}"

        for m in s.moves:
            assert 0 <= m.player_idx < len(frame.players), f"{where}: index {m.player_idx}"
            p = frame.players[m.player_idx]
            assert not p.is_attacker, f"{where}: moved attacker {m.player_idx}"
            assert not p.is_taker, f"{where}: moved the taker"
            length = math.hypot(m.dx, m.dy)
            assert length <= max_move + TOL, f"{where}: displacement {length!r} > {max_move}"
            assert length > 0.0, f"{where}: zero-length move"

        moved = apply_moves(frame, list(s.moves))
        assert len(moved.players) == len(frame.players), f"{where}: player count changed"
        for i, p in enumerate(moved.players):
            if i in idxs:
                assert -TOL <= p.x <= PITCH_LENGTH + TOL, f"{where}: x={p.x} off pitch"
                assert -TOL <= p.y <= PITCH_WIDTH + TOL, f"{where}: y={p.y} off pitch"
            else:
                assert (p.x, p.y) == (frame.players[i].x, frame.players[i].y), (
                    f"{where}: unmoved player {i} shifted"
                )

        after = _tight_pairs(moved, min_sep)
        assert not (after - before), f"{where}: created tight pairs {sorted(after - before)}"
        touching = sorted(pr for pr in after if pr[0] in idxs or pr[1] in idxs)
        assert not touching, f"{where}: a moved player sits inside min_sep: {touching}"

        assert s.delta == pytest.approx(s.new_prob - s.base_prob, abs=1e-12)


# --------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------- #
#: (movers, max_move, min_sep, beam, seed) settings the sweep runs at every
#: frame. Kept to four rows: the sweep is already several hundred searches wide,
#: and the seeds vary the phase of the candidate ring, not the search itself.
SETTINGS = (
    (3, 3.0, 0.5, 8, 0),
    (3, 3.0, 0.5, 8, 17),
    (5, 1.0, 0.5, 4, 0),
    (2, 6.0, 2.0, 8, 17),
)


@pytest.mark.parametrize("movers,max_move,min_sep,beam,seed", SETTINGS)
def test_hard_constraints_hold_across_every_hostile_frame(
    movers: int, max_move: float, min_sep: float, beam: int, seed: int
) -> None:
    for name in FRAME_NAMES:
        frame = ALL_FRAMES[name]
        out = suggest(
            frame,
            analytic_scorer,
            movers=movers,
            max_move=max_move,
            min_sep=min_sep,
            beam=beam,
            seed=seed,
        )
        assert_constraints(
            frame,
            out,
            movers=movers,
            max_move=max_move,
            min_sep=min_sep,
            tag=f"{name} seed={seed} movers={movers}",
        )


# --------------------------------------------------------------------------- #
# the same sweep, with the candidate restriction lifted
# --------------------------------------------------------------------------- #
def n_movable(frame: CornerFrame, move_keeper: bool = False) -> int:
    """How many defenders the search is allowed to consider on this frame."""
    return sum(
        1
        for p in frame.players
        if not p.is_attacker and not p.is_taker and (move_keeper or not p.is_keeper)
    )


def wide_budget(
    frame: CornerFrame, movers: int, beam: int, deep: int | None, move_keeper: bool = False
) -> int:
    """A budget that cannot bind, so the widened search really is widened."""
    return max(
        1,
        required_evals(
            movers=movers,
            beam=beam,
            max_candidates=None,
            deep_candidates=deep,
            n_movable=n_movable(frame, move_keeper),
        ),
    )


#: (movers, max_move, min_sep, beam, seed, deep_candidates) for the widened
#: sweep. ``max_candidates`` is always ``None`` here, every defender on the
#: frame is a candidate for the first move, which is the setting the default of
#: six exists to avoid paying for.
WIDE_SETTINGS = (
    (3, 3.0, 0.5, 8, 0, None),
    (3, 3.0, 0.5, 8, 17, 4),
    (2, 6.0, 2.0, 4, 0, None),
)


@pytest.mark.parametrize("movers,max_move,min_sep,beam,seed,deep", WIDE_SETTINGS)
def test_hard_constraints_hold_with_the_candidate_restriction_lifted(
    movers: int, max_move: float, min_sep: float, beam: int, seed: int, deep: int | None
) -> None:
    """Every guarantee has to survive a search that is an order of magnitude wider.

    A wider pool means more chances to place a defender on top of someone, off
    the touchline, or twice; none of the constraints are enforced by the
    branching bound, and this is what proves it.
    """
    for name in FRAME_NAMES:
        frame = ALL_FRAMES[name]
        out = suggest(
            frame,
            analytic_scorer,
            movers=movers,
            max_move=max_move,
            min_sep=min_sep,
            beam=beam,
            seed=seed,
            max_candidates=None,
            deep_candidates=deep,
            max_evals=wide_budget(frame, movers, beam, deep),
        )
        assert_constraints(
            frame,
            out,
            movers=movers,
            max_move=max_move,
            min_sep=min_sep,
            tag=f"{name} wide seed={seed} movers={movers} deep={deep}",
        )


def test_widening_never_loses_a_single_move_the_narrow_search_found() -> None:
    """At one mover the wide pool is a strict superset, so it cannot do worse.

    Run over the whole sweep because the interesting cases are the hostile ones:
    a frame where most of the pool is inadmissible is exactly where a truncated
    budget would quietly hand back the narrow answer.
    """
    for name in FRAME_NAMES:
        frame = ALL_FRAMES[name]
        narrow = suggest(frame, analytic_scorer, movers=1, seed=0)
        wide = suggest(
            frame,
            analytic_scorer,
            movers=1,
            seed=0,
            max_candidates=None,
            max_evals=wide_budget(frame, 1, 8, None),
        )
        if not narrow:
            continue
        assert wide, f"{name}: narrow found a move and wide found none"
        assert wide[0].delta <= narrow[0].delta + 1e-12, name


def test_a_widened_search_that_outruns_its_budget_warns_rather_than_lying() -> None:
    frame = ALL_FRAMES["goal_line"]
    assert n_movable(frame) >= 8, "fixture needs more defenders than the default width"
    with pytest.warns(RuntimeWarning, match=r"truncated rather than widened"):
        out = suggest(frame, analytic_scorer, max_candidates=None, max_evals=288)
    # and the truncated search is still a legal one
    assert_constraints(frame, out, movers=3, max_move=3.0, min_sep=0.5, tag="goal_line truncated")


def test_the_keeper_joins_the_pool_only_when_asked_across_the_sweep() -> None:
    for name in FRAME_NAMES:
        frame = ALL_FRAMES[name]
        keepers = {i for i, p in enumerate(frame.players) if p.is_keeper and not p.is_attacker}
        if not keepers:
            continue
        default = suggest(
            frame,
            analytic_scorer,
            seed=0,
            max_candidates=None,
            max_evals=wide_budget(frame, 3, 8, 2),
        )
        assert not {m.player_idx for s in default for m in s.moves} & keepers, name

        out = suggest(
            frame,
            analytic_scorer,
            seed=0,
            move_keeper=True,
            max_candidates=None,
            max_evals=wide_budget(frame, 3, 8, 2, move_keeper=True),
        )
        assert_constraints(
            frame, out, movers=3, max_move=3.0, min_sep=0.5, tag=f"{name} move_keeper"
        )


def test_frames_without_a_movable_defender_yield_nothing() -> None:
    for name in ("zero_defenders", "keeper_only"):
        assert suggest(ALL_FRAMES[name], analytic_scorer, seed=0) == []
    # the defending taker is the only defender, and takers never move
    assert all(
        m.player_idx != 0
        for s in suggest(ALL_FRAMES["defending_taker"], analytic_scorer, seed=0)
        for m in s.moves
    )


def test_a_single_defender_never_exceeds_one_move() -> None:
    out = suggest(ALL_FRAMES["one_defender"], analytic_scorer, movers=3, seed=0)
    assert out
    assert all(len(s.moves) == 1 for s in out)
    assert all(s.moves[0].player_idx == 2 for s in out)


def test_stacked_players_still_never_breach_min_sep() -> None:
    frame = ALL_FRAMES["stacked"]
    out = suggest(frame, analytic_scorer, movers=3, seed=0)
    assert out, "a stacked shape should still admit an outward move"
    assert_constraints(frame, out, movers=3, max_move=3.0, min_sep=0.5, tag="stacked")


def test_a_pre_existing_overlap_is_never_made_worse() -> None:
    """The documented semantics when the input already breaks ``min_sep``.

    The search cannot separate a pair it does not touch, so tight pairs survive.
    It must never add one, and never leave a player it moved inside ``min_sep``.
    """
    frame = ALL_FRAMES["pre_violating"]
    before = _tight_pairs(frame, 0.5)
    assert before, "fixture is meant to violate min_sep before anything moves"

    out = suggest(frame, analytic_scorer, movers=3, seed=0)
    assert out
    assert_constraints(frame, out, movers=3, max_move=3.0, min_sep=0.5, tag="pre_violating")

    # and the survivors are exactly pre-existing pairs of untouched players
    for s in out:
        idxs = {m.player_idx for m in s.moves}
        after = _tight_pairs(apply_moves(frame, list(s.moves)), 0.5)
        assert after <= before
        assert all(i not in idxs and j not in idxs for i, j in after)


# --------------------------------------------------------------------------- #
# the displacement bound, at the floating point boundary
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "max_move",
    [
        3.0,
        1.0 / 3.0,
        7.0 / 3.0,
        math.nextafter(3.0, 0.0),
        math.nextafter(3.0, math.inf),
        1e-9,
        1e9,
        0.1 + 0.2,
    ],
)
def test_no_candidate_displacement_exceeds_max_move_by_a_single_ulp(max_move: float) -> None:
    for seed in range(40):
        for dx, dy in candidate_displacements(max_move, seed):
            length = math.hypot(dx, dy)
            assert length <= max_move, (
                f"seed {seed}: |({dx!r}, {dy!r})| = {length!r} > {max_move!r} "
                f"by {length - max_move!r}"
            )


@pytest.mark.parametrize("max_move", [0.0, -0.0, -1.0, -3.0, math.nan, math.inf, -math.inf])
def test_a_non_positive_or_non_finite_max_move_admits_no_move(max_move: float) -> None:
    """A negative budget used to produce moves of length ``abs(max_move)``.

    ``candidate_displacements`` rescales any ring member longer than ``max_move``
    back onto the bound; with a negative bound that comparison is true for every
    member and the rescale flips the sign instead of shrinking it, so the search
    happily returned 1 m moves under ``--max-move -1``. Both the generator and
    :func:`suggest` now refuse.
    """
    assert candidate_displacements(max_move, 0) == ()
    frame = ALL_FRAMES["pitch_corners"]
    assert suggest(frame, analytic_scorer, max_move=max_move, seed=0) == []


def test_moves_land_exactly_where_the_reported_displacement_says() -> None:
    """The scored shape and the returned moves cannot drift apart."""
    for name in ("pitch_corners", "goal_line", "stacked"):
        frame = ALL_FRAMES[name]
        for s in suggest(frame, analytic_scorer, seed=3):
            moved = apply_moves(frame, list(s.moves))
            assert analytic_scorer(moved) == pytest.approx(s.new_prob, abs=1e-12)


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def _fingerprint(out: list[Suggestion]) -> str:
    """An exact, ordering-sensitive identity: hex floats, no rounding."""
    return json.dumps(
        [
            {
                "moves": [[m.player_idx, m.dx.hex(), m.dy.hex()] for m in s.moves],
                "base": s.base_prob.hex(),
                "new": s.new_prob.hex(),
                "delta": s.delta.hex(),
            }
            for s in out
        ]
    )


#: A cross section of the sweep, wide enough to catch an ordering wobble and
#: narrow enough to run five times over (twice here, twice in subprocesses).
DETERMINISM_FRAMES = sorted(degenerate_frames()) + FRAME_NAMES[::10]


def _sweep_fingerprint() -> str:
    parts = []
    for name in DETERMINISM_FRAMES:
        frame = ALL_FRAMES[name]
        for seed in (0, 5):
            parts.append(_fingerprint(suggest(frame, analytic_scorer, seed=seed)))
        # the widened search enumerates a longer, less pruned pool, so it has
        # more room for an ordering wobble than the default one does
        parts.append(
            _fingerprint(
                suggest(
                    frame,
                    analytic_scorer,
                    seed=0,
                    max_candidates=None,
                    deep_candidates=None,
                    max_evals=wide_budget(frame, 3, 8, None),
                )
            )
        )
    return json.dumps(parts)


def test_identical_within_one_process() -> None:
    first = _sweep_fingerprint()
    second = _sweep_fingerprint()
    assert first == second


_CHILD = textwrap.dedent(
    """
    import hashlib, sys
    sys.path.insert(0, {tests!r})
    from test_suggest_stress import _sweep_fingerprint
    print(hashlib.sha256(_sweep_fingerprint().encode()).hexdigest())
    """
)


@pytest.mark.parametrize("hash_seeds", [("0", "987654")])
def test_identical_across_separate_processes(hash_seeds: tuple[str, ...]) -> None:
    """Unseeded RNG and set/dict iteration order only surface across processes."""
    import os

    script = _CHILD.format(tests=str(Path(__file__).resolve().parent))
    digests = []
    for hs in hash_seeds:
        env = {**os.environ, "PYTHONHASHSEED": hs}
        env.pop("PYTHONDONTWRITEBYTECODE", None)
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO),
            timeout=300,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        digests.append(proc.stdout.strip())
    assert len(set(digests)) == 1, f"PYTHONHASHSEED changed the result: {digests}"
    assert digests[0] == __import__("hashlib").sha256(
        _sweep_fingerprint().encode()
    ).hexdigest()


# --------------------------------------------------------------------------- #
# the frame is never mutated
# --------------------------------------------------------------------------- #
def test_no_frame_in_the_sweep_is_mutated() -> None:
    for name in FRAME_NAMES:
        frame = ALL_FRAMES[name]
        snapshot = [(p.x, p.y, p.is_attacker, p.is_taker, p.is_keeper) for p in frame.players]
        suggest(frame, analytic_scorer, seed=0)
        assert [
            (p.x, p.y, p.is_attacker, p.is_taker, p.is_keeper) for p in frame.players
        ] == snapshot, f"{name} was mutated by the search"


# --------------------------------------------------------------------------- #
# latency, against the real bundled checkpoint
# --------------------------------------------------------------------------- #
def test_a_full_search_on_a_22_player_frame_takes_about_a_second() -> None:
    """The pitch editor re-scores on drop, so the search has roughly a second.

    Single threaded on purpose. A corner graph has twenty-odd nodes and torch's
    default four threads lose more to synchronisation than they gain: the same
    search measures ~0.6 s pinned to one thread and ~2.6 s at four, which is why
    the CLI pins them. Anything launching the search outside the CLI has to do
    the same or it will miss the app's one second round trip.
    """
    path = next((p for p in DATASETS if p.exists()), None)
    if path is None:
        pytest.skip("no corner dataset on disk")
    torch = pytest.importorskip("torch")
    from setpiece.models import load_bundled

    try:
        model, _ = load_bundled("shot")
    except FileNotFoundError:
        pytest.skip("no bundled shot checkpoint")

    torch.set_num_threads(1)
    frames = read_corners(path)
    biggest = max(frames, key=lambda f: len(f.players))
    assert len(biggest.players) >= 18, "expected a near-full freeze frame in the dataset"

    import time

    suggest(biggest, model, seed=0)  # warm the graph builder and the kernels
    started = time.perf_counter()
    out = suggest(biggest, model, seed=0)
    elapsed = time.perf_counter() - started

    assert out
    assert_constraints(biggest, out, movers=3, max_move=3.0, min_sep=0.5, tag="bundled")
    # measured single-thread cost is ~0.6 s; the margin is for a busy machine,
    # not for a search that has quietly got three times slower
    assert elapsed < 2.0, f"{len(biggest.players)}-player search took {elapsed:.2f}s"


def test_move_and_suggestion_are_the_documented_shape() -> None:
    out = suggest(ALL_FRAMES["pitch_corners"], analytic_scorer, seed=0)
    assert out
    assert all(isinstance(s, Suggestion) for s in out)
    assert all(isinstance(m, Move) for s in out for m in s.moves)
    assert all(s.delta <= TOL for s in out[:1]), "the best defensive move should not be worse"
