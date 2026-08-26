"""Tests for the defender adjustment search.

These are standalone: the shot model is a hand-written stand-in, so nothing here
depends on setpiece.models, setpiece.features or a trained checkpoint.
"""

from __future__ import annotations

import math
import random
import warnings
from pathlib import Path

import pytest

from setpiece._io import read_corners
from setpiece.suggest import (
    DEEP_CANDIDATE_DEFENDERS,
    DEFAULT_MAX_EVALS,
    MAX_CANDIDATE_DEFENDERS,
    Move,
    Suggestion,
    apply_moves,
    candidate_displacements,
    required_evals,
    suggest,
)
from setpiece.types import GOAL_X, GOAL_Y, PITCH_LENGTH, PITCH_WIDTH, CornerFrame, Player

TOL = 1e-9

#: Optional: the real Euro 2024 corners, if a dataset has been extracted.
SMOKE = Path(__file__).resolve().parents[1] / "data" / "interim" / "smoke.jsonl"


# --------------------------------------------------------------------------- #
# stand-in shot model
# --------------------------------------------------------------------------- #
class FakeShotModel:
    """A smooth, deterministic pure function of the freeze frame.

    Every attacker contributes a threat that decays with its distance to goal and
    grows as the nearest defender gets further away; the threats are combined as
    independent chances. That gives the search a real gradient to climb — closing
    down the attackers near the six-yard box lowers the number — without any of
    the machinery of the actual model.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, frame: CornerFrame) -> float:
        self.calls += 1
        defenders = [p for p in frame.players if not p.is_attacker]
        miss = 1.0
        for a in frame.players:
            if not a.is_attacker or a.is_taker:
                continue
            d_goal = math.hypot(GOAL_X - a.x, GOAL_Y - a.y)
            if defenders:
                marking = min(math.hypot(d.x - a.x, d.y - a.y) for d in defenders)
            else:
                marking = 20.0
            threat = math.exp(-d_goal / 12.0) * (1.0 - math.exp(-marking / 4.0))
            miss *= 1.0 - 0.9 * threat
        return 1.0 - miss


class MethodShotModel:
    """Same numbers, exposed as ``shot_probability`` rather than ``__call__``."""

    def __init__(self) -> None:
        self._inner = FakeShotModel()

    def shot_probability(self, frame: CornerFrame) -> float:
        return self._inner(frame)


# --------------------------------------------------------------------------- #
# frame builders
# --------------------------------------------------------------------------- #
def random_frame(rng: random.Random, *, n_attackers: int = 8, n_defenders: int = 9) -> CornerFrame:
    """A plausible, well-separated corner: everyone packed into the attacking box.

    Positions are rejection-sampled so the base frame respects a 0.7 m minimum
    separation, comfortably above the 0.5 m the search defends. Real StatsBomb
    frames occasionally violate that, which the search tolerates but cannot fix.
    """
    players: list[Player] = [Player(x=104.6, y=0.4, is_attacker=True, is_taker=True)]
    placed: list[tuple[float, float]] = [(104.6, 0.4)]

    def place(is_attacker: bool, is_keeper: bool = False) -> None:
        for _ in range(500):
            x = rng.uniform(88.0, 104.0)
            y = rng.uniform(20.0, 48.0)
            if all(math.hypot(x - px, y - py) >= 0.7 for px, py in placed):
                placed.append((x, y))
                players.append(Player(x=x, y=y, is_attacker=is_attacker, is_keeper=is_keeper))
                return
        raise RuntimeError("could not place a player")

    place(False, is_keeper=True)
    for _ in range(n_attackers):
        place(True)
    for _ in range(n_defenders - 1):
        place(False)
    rng.shuffle(players)
    return CornerFrame(corner_id="rnd", players=players)


def tiny_frame(n_defenders: int) -> CornerFrame:
    players = [
        Player(x=104.6, y=67.6, is_attacker=True, is_taker=True),
        Player(x=104.0, y=34.0, is_attacker=False, is_keeper=True),
        Player(x=98.0, y=32.0, is_attacker=True),
        Player(x=97.0, y=36.0, is_attacker=True),
    ]
    for k in range(n_defenders):
        players.append(Player(x=95.0, y=28.0 + 3.0 * k, is_attacker=False))
    return CornerFrame(corner_id="tiny", players=players)


# --------------------------------------------------------------------------- #
# constraint checking
# --------------------------------------------------------------------------- #
def check_constraints(
    frame: CornerFrame,
    sug: Suggestion,
    *,
    movers: int,
    max_move: float,
    min_sep: float,
) -> None:
    assert 1 <= len(sug.moves) <= movers

    seen: set[int] = set()
    for m in sug.moves:
        assert 0 <= m.player_idx < len(frame.players)
        p = frame.players[m.player_idx]
        assert not p.is_attacker, "an attacker was moved"
        assert not p.is_taker, "the taker was moved"
        assert m.player_idx not in seen, "a player was moved twice"
        seen.add(m.player_idx)
        assert math.hypot(m.dx, m.dy) <= max_move + TOL

    moved = apply_moves(frame, sug.moves)
    assert len(moved.players) == len(frame.players)

    for m in sug.moves:
        src, dst = frame.players[m.player_idx], moved.players[m.player_idx]
        assert 0.0 <= dst.x <= PITCH_LENGTH
        assert 0.0 <= dst.y <= PITCH_WIDTH
        assert dst.is_attacker == src.is_attacker
        assert math.hypot(dst.x - src.x, dst.y - src.y) <= max_move + TOL

    # every pair, attackers and unmoved defenders included
    for i, a in enumerate(moved.players):
        for b in moved.players[i + 1 :]:
            assert math.hypot(a.x - b.x, a.y - b.y) >= min_sep - TOL

    # unmoved players are untouched
    for i, (before, after) in enumerate(zip(frame.players, moved.players, strict=True)):
        if i not in seen:
            assert (before.x, before.y) == (after.x, after.y)


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_candidate_set_respects_max_move() -> None:
    for seed in range(5):
        disps = candidate_displacements(3.0, seed)
        assert len(disps) == 16
        assert len(set(disps)) == 16
        for dx, dy in disps:
            assert math.hypot(dx, dy) <= 3.0 + TOL
        assert candidate_displacements(3.0, seed) == disps


@pytest.mark.parametrize("trial", range(12))
def test_hard_constraints_on_random_frames(trial: int) -> None:
    rng = random.Random(1000 + trial)
    frame = random_frame(rng)
    model = FakeShotModel()
    out = suggest(frame, model, movers=3, seed=trial)

    assert out, "expected at least one admissible suggestion on an open frame"
    for sug in out:
        check_constraints(frame, sug, movers=3, max_move=3.0, min_sep=0.5)


@pytest.mark.parametrize("movers", [1, 2, 3, 5])
def test_mover_budget_is_respected(movers: int) -> None:
    frame = random_frame(random.Random(7))
    out = suggest(frame, FakeShotModel(), movers=movers)
    assert out
    assert max(len(s.moves) for s in out) <= movers
    for sug in out:
        check_constraints(frame, sug, movers=movers, max_move=3.0, min_sep=0.5)


@pytest.mark.parametrize("max_move", [0.5, 1.0, 3.0, 6.0])
def test_max_move_is_respected(max_move: float) -> None:
    frame = random_frame(random.Random(11))
    out = suggest(frame, FakeShotModel(), max_move=max_move)
    for sug in out:
        check_constraints(frame, sug, movers=3, max_move=max_move, min_sep=0.5)


@pytest.mark.parametrize("min_sep", [0.5, 1.5, 3.0])
def test_min_sep_is_respected_against_everyone(min_sep: float) -> None:
    # a frame whose own spacing already clears the tightest min_sep we ask for
    rng = random.Random(23)
    players: list[Player] = [Player(x=104.5, y=0.5, is_attacker=True, is_taker=True)]
    placed = [(104.5, 0.5)]
    while len(players) < 15:
        x, y = rng.uniform(86.0, 104.0), rng.uniform(16.0, 52.0)
        if all(math.hypot(x - px, y - py) >= 3.2 for px, py in placed):
            placed.append((x, y))
            players.append(Player(x=x, y=y, is_attacker=len(players) % 2 == 0))
    frame = CornerFrame(corner_id="spaced", players=players)

    out = suggest(frame, FakeShotModel(), min_sep=min_sep)
    for sug in out:
        check_constraints(frame, sug, movers=3, max_move=3.0, min_sep=min_sep)


def test_moved_players_stay_on_the_pitch() -> None:
    # defenders parked on the goal line and the touchlines: most of the compass
    # rose points off the pitch and must be pruned by the search
    players = [
        Player(x=104.9, y=0.2, is_attacker=True, is_taker=True),
        Player(x=104.9, y=34.0, is_attacker=False, is_keeper=True),
        Player(x=104.8, y=20.0, is_attacker=False),
        Player(x=100.0, y=0.1, is_attacker=False),
        Player(x=100.0, y=67.9, is_attacker=False),
        Player(x=0.1, y=34.0, is_attacker=False),
        Player(x=99.0, y=30.0, is_attacker=True),
        Player(x=99.0, y=38.0, is_attacker=True),
    ]
    frame = CornerFrame(corner_id="edges", players=players)
    out = suggest(frame, FakeShotModel(), movers=3)
    assert out
    for sug in out:
        check_constraints(frame, sug, movers=3, max_move=3.0, min_sep=0.5)


def test_deterministic_for_a_fixed_seed() -> None:
    frame = random_frame(random.Random(3))
    first = suggest(frame, FakeShotModel(), seed=5)
    second = suggest(frame, FakeShotModel(), seed=5)
    assert first == second
    # identical ordering and identical floats, not merely equal sets
    assert [(m.player_idx, m.dx, m.dy) for s in first for m in s.moves] == [
        (m.player_idx, m.dx, m.dy) for s in second for m in s.moves
    ]
    assert [s.delta for s in first] == [s.delta for s in second]
    assert all(
        a.new_prob == b.new_prob and a.base_prob == b.base_prob
        for a, b in zip(first, second, strict=True)
    )


def test_different_seeds_still_produce_valid_output() -> None:
    frame = random_frame(random.Random(4))
    a = suggest(frame, FakeShotModel(), seed=0)
    b = suggest(frame, FakeShotModel(), seed=1)
    for sug in a + b:
        check_constraints(frame, sug, movers=3, max_move=3.0, min_sep=0.5)


def test_ranking_and_delta_bookkeeping() -> None:
    frame = random_frame(random.Random(9))
    model = FakeShotModel()
    base = model(frame)
    out = suggest(frame, model, movers=3)
    assert out

    deltas = [s.delta for s in out]
    assert deltas == sorted(deltas), "defend=True must rank by delta ascending"
    assert deltas[0] < 0.0, "the best suggestion should lower the shot probability"
    for sug in out:
        assert sug.base_prob == pytest.approx(base)
        assert sug.delta == pytest.approx(sug.new_prob - sug.base_prob)
        replayed = FakeShotModel()(apply_moves(frame, sug.moves))
        assert replayed == pytest.approx(sug.new_prob, abs=1e-12)


def test_attacking_mode_maximises() -> None:
    frame = random_frame(random.Random(13))
    out = suggest(frame, FakeShotModel(), defend=False)
    assert out
    deltas = [s.delta for s in out]
    assert deltas == sorted(deltas, reverse=True), "defend=False ranks best-first, descending"
    assert deltas[0] > 0.0
    for sug in out:
        check_constraints(frame, sug, movers=3, max_move=3.0, min_sep=0.5)

    best_defend = suggest(frame, FakeShotModel(), defend=True)[0]
    assert best_defend.delta < 0.0 < out[0].delta


def test_accepts_a_shot_probability_method() -> None:
    frame = random_frame(random.Random(17))
    out = suggest(frame, MethodShotModel(), movers=2)
    assert out
    for sug in out:
        check_constraints(frame, sug, movers=2, max_move=3.0, min_sep=0.5)


def test_evaluation_budget_is_bounded() -> None:
    frame = random_frame(random.Random(21))
    model = FakeShotModel()
    suggest(frame, model, movers=3)
    assert model.calls <= DEFAULT_MAX_EVALS + 1

    tight = FakeShotModel()
    suggest(frame, tight, movers=3, max_evals=40)
    assert tight.calls <= 41
    assert tight.calls > 1


def test_fewer_defenders_than_movers() -> None:
    frame = tiny_frame(n_defenders=2)  # plus a keeper, which is not moved by default
    out = suggest(frame, FakeShotModel(), movers=3)
    assert out
    assert max(len(s.moves) for s in out) <= 2
    for sug in out:
        check_constraints(frame, sug, movers=2, max_move=3.0, min_sep=0.5)

    one = suggest(tiny_frame(n_defenders=1), FakeShotModel(), movers=3)
    assert one
    assert all(len(s.moves) == 1 for s in one)


def test_no_movable_defenders_returns_empty() -> None:
    players = [
        Player(x=104.6, y=0.4, is_attacker=True, is_taker=True),
        Player(x=104.0, y=34.0, is_attacker=False, is_keeper=True),
        Player(x=98.0, y=33.0, is_attacker=True),
    ]
    frame = CornerFrame(corner_id="nodef", players=players)
    model = FakeShotModel()
    assert suggest(frame, model, movers=3) == []
    assert model.calls == 1  # only the base frame was scored


def test_every_candidate_move_blocked_returns_empty() -> None:
    """When min_sep cannot be satisfied anywhere, we return an empty list."""
    frame = random_frame(random.Random(31))
    # every player sits inside the box, so no displacement of 3 m or less can put
    # a defender 25 m clear of everyone else
    out = suggest(frame, FakeShotModel(), min_sep=25.0)
    assert out == []


def test_degenerate_budgets_return_empty() -> None:
    frame = random_frame(random.Random(33))
    model = FakeShotModel()
    assert suggest(frame, model, movers=0) == []
    assert suggest(frame, model, beam=0) == []
    assert suggest(frame, model, top_k=0) == []
    assert suggest(frame, model, max_evals=0) == []


def test_top_k_and_result_shape() -> None:
    frame = random_frame(random.Random(37))
    out = suggest(frame, FakeShotModel(), beam=8)
    assert 0 < len(out) <= 8
    assert all(isinstance(s, Suggestion) for s in out)
    assert all(isinstance(m, Move) for s in out for m in s.moves)

    three = suggest(frame, FakeShotModel(), top_k=3)
    assert len(three) == 3
    assert [s.delta for s in three] == [s.delta for s in out[:3]]


def test_no_duplicate_suggestions() -> None:
    frame = random_frame(random.Random(41))
    out = suggest(frame, FakeShotModel(), movers=3)
    keys = {
        tuple(sorted((m.player_idx, round(m.dx, 9), round(m.dy, 9)) for m in s.moves)) for s in out
    }
    assert len(keys) == len(out)


def test_input_frame_is_not_mutated() -> None:
    frame = random_frame(random.Random(43))
    before = [(p.x, p.y, p.is_attacker, p.is_taker, p.is_keeper) for p in frame.players]
    suggest(frame, FakeShotModel(), movers=3)
    after = [(p.x, p.y, p.is_attacker, p.is_taker, p.is_keeper) for p in frame.players]
    assert before == after


def test_keeper_moves_only_when_asked() -> None:
    frame = random_frame(random.Random(47))
    keeper_idx = next(i for i, p in enumerate(frame.players) if p.is_keeper)

    default = suggest(frame, FakeShotModel(), movers=3)
    assert all(m.player_idx != keeper_idx for s in default for m in s.moves)

    with_keeper = suggest(frame, FakeShotModel(), movers=3, move_keeper=True)
    for sug in with_keeper:
        check_constraints(frame, sug, movers=3, max_move=3.0, min_sep=0.5)


def test_rejects_a_non_callable_model() -> None:
    frame = random_frame(random.Random(53))
    with pytest.raises(TypeError):
        suggest(frame, object())


def test_search_reaches_the_full_move_budget_on_an_open_frame() -> None:
    """The eval budget must be wide enough to actually spend all three movers."""
    for trial in range(6):
        frame = random_frame(random.Random(500 + trial))
        model = FakeShotModel()
        out = suggest(frame, model, movers=3, seed=trial)
        assert max(len(s.moves) for s in out) == 3
        assert model.calls <= DEFAULT_MAX_EVALS + 1


@pytest.mark.skipif(not SMOKE.exists(), reason="no extracted corners on disk")
def test_real_corners() -> None:
    """Real frames, which sometimes already have two players inside min_sep.

    The search must never make such a pair worse or invent a new one, and must
    still return usable suggestions.
    """
    corners = read_corners(SMOKE)
    assert corners
    for frame in corners[:6]:
        out = suggest(frame, FakeShotModel(), movers=3)
        assert out
        before = _tight_pairs(frame, 0.5)
        for sug in out:
            for m in sug.moves:
                p = frame.players[m.player_idx]
                assert not p.is_attacker and not p.is_taker and not p.is_keeper
                assert math.hypot(m.dx, m.dy) <= 3.0 + TOL
            moved = apply_moves(frame, sug.moves)
            assert _tight_pairs(moved, 0.5) <= before
            for p in moved.players:
                assert 0.0 <= p.x <= PITCH_LENGTH
                assert 0.0 <= p.y <= PITCH_WIDTH


# --------------------------------------------------------------------------- #
# candidate width: how far into the defence the search is allowed to look
# --------------------------------------------------------------------------- #
class FurthestDefenderScorer:
    """Minimised by pulling the defender *furthest* from goal back towards it.

    Deliberately the opposite of what the candidate ranking prefers. The search
    ranks defenders nearest the goal first, so this scorer's only improving move
    lives at the far end of that ranking — exactly the region the default
    ``max_candidates`` of six cannot reach on a ten-defender frame.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, frame: CornerFrame) -> float:
        self.calls += 1
        far = max(
            math.hypot(GOAL_X - p.x, GOAL_Y - p.y)
            for p in frame.players
            if not p.is_attacker and not p.is_taker
        )
        return 1.0 / (1.0 + math.exp(-(far - 20.0)))


def ten_defender_frame() -> CornerFrame:
    """A taker, two attackers, a keeper and ten outfield defenders in a line.

    The defenders sit 4, 6, ... 22 m from the goal centre, so their distance
    ranking — the one :func:`setpiece.suggest._movable` sorts by — is exactly
    their index order and the four furthest are unambiguous.
    """
    players = [
        Player(x=104.6, y=0.4, is_attacker=True, is_taker=True),
        Player(x=104.5, y=34.0, is_attacker=False, is_keeper=True),
        Player(x=100.0, y=20.0, is_attacker=True),
        Player(x=100.0, y=48.0, is_attacker=True),
    ]
    players += [Player(x=GOAL_X - r, y=GOAL_Y, is_attacker=False) for r in range(4, 24, 2)]
    return CornerFrame(corner_id="ten_defenders", players=players)


def _outfield_defenders_by_distance(frame: CornerFrame) -> list[int]:
    idxs = [
        i
        for i, p in enumerate(frame.players)
        if not p.is_attacker and not p.is_taker and not p.is_keeper
    ]
    return sorted(idxs, key=lambda i: math.hypot(GOAL_X - frame.players[i].x, GOAL_Y - frame.players[i].y))


def test_required_evals_reproduces_the_default_budget() -> None:
    """288 is the layout's own arithmetic, not a number someone liked."""
    assert required_evals() == DEFAULT_MAX_EVALS == 288
    # 6*16 first move, then beam/2 = 4 states over 2 defenders, then beam/4 = 2
    assert required_evals() == 6 * 16 + 4 * 2 * 16 + 2 * 2 * 16
    # widening the first level costs exactly the extra defenders
    assert required_evals(max_candidates=10) == 10 * 16 + 4 * 2 * 16 + 2 * 2 * 16
    assert required_evals(max_candidates=None, n_movable=10) == required_evals(max_candidates=10)
    # a frame cannot spend a width it has no defenders for
    assert required_evals(n_movable=1) == 1 * 16
    assert required_evals(deep_candidates=0) == 6 * 16
    assert required_evals(movers=0) == 0


@pytest.mark.parametrize(
    "max_candidates,deep_candidates,movers,beam",
    [(6, 2, 3, 8), (10, 4, 3, 8), (None, None, 3, 8), (None, None, 4, 4), (3, 1, 2, 8)],
)
def test_required_evals_is_an_upper_bound_on_the_real_call_count(
    max_candidates: int | None, deep_candidates: int | None, movers: int, beam: int
) -> None:
    """The budget arithmetic must never under-count, or a search would truncate."""
    frame = ten_defender_frame()
    n_movable = len(_outfield_defenders_by_distance(frame))
    predicted = required_evals(
        movers=movers,
        beam=beam,
        max_candidates=max_candidates,
        deep_candidates=deep_candidates,
        n_movable=n_movable,
    )
    model = FakeShotModel()
    suggest(
        frame,
        model,
        movers=movers,
        beam=beam,
        max_candidates=max_candidates,
        deep_candidates=deep_candidates,
        max_evals=10_000,
    )
    assert model.calls - 1 <= predicted, "required_evals under-counted the real search"


def test_the_default_width_cannot_reach_the_furthest_defenders() -> None:
    """The restriction this flag exists to lift, stated as a measurement.

    Ten outfield defenders, and the only improving move belongs to the one
    furthest from goal. At the default width the search cannot see it and returns
    nothing better than the base frame; lifting the width finds it.
    """
    frame = ten_defender_frame()
    ranked = _outfield_defenders_by_distance(frame)
    assert len(ranked) == 10
    near_six, far_four = set(ranked[:MAX_CANDIDATE_DEFENDERS]), set(ranked[MAX_CANDIDATE_DEFENDERS:])
    assert len(far_four) == 4

    narrow = suggest(frame, FurthestDefenderScorer(), movers=3, seed=0)
    assert narrow
    moved_narrow = {m.player_idx for s in narrow for m in s.moves}
    assert moved_narrow <= near_six
    assert not (moved_narrow & far_four), "the default width reached a far defender"
    assert narrow[0].delta == pytest.approx(0.0, abs=1e-12), (
        "no move inside the default width can improve this shape"
    )

    wide = suggest(
        frame,
        FurthestDefenderScorer(),
        movers=3,
        seed=0,
        max_candidates=None,
        deep_candidates=None,
        max_evals=required_evals(max_candidates=None, deep_candidates=None, n_movable=10),
    )
    assert wide
    moved_wide = {m.player_idx for s in wide for m in s.moves}
    assert moved_wide & far_four, "a widened search still never reached the far defenders"
    assert wide[0].delta < narrow[0].delta - 1e-9
    for sug in wide:
        check_constraints(frame, sug, movers=3, max_move=3.0, min_sep=0.5)


def test_max_candidates_none_puts_every_defender_in_the_first_move_pool() -> None:
    frame = ten_defender_frame()
    ranked = _outfield_defenders_by_distance(frame)
    seen: set[int] = set()

    class Counter:
        def __call__(self, f: CornerFrame) -> float:
            moved = [
                i
                for i, (a, b) in enumerate(zip(f.players, frame.players, strict=True))
                if (a.x, a.y) != (b.x, b.y)
            ]
            seen.update(moved)
            return 0.5

    suggest(
        frame,
        Counter(),
        movers=1,
        seed=0,
        max_candidates=None,
        max_evals=required_evals(movers=1, max_candidates=None, n_movable=10),
    )
    assert seen == set(ranked), f"pool missed {sorted(set(ranked) - seen)}"


@pytest.mark.parametrize("width", [1, 2, 3, 6, 10, None])
def test_hard_constraints_survive_every_candidate_width(width: int | None) -> None:
    frame = ten_defender_frame()
    out = suggest(
        frame,
        FakeShotModel(),
        movers=3,
        seed=1,
        max_candidates=width,
        deep_candidates=width,
        max_evals=required_evals(
            max_candidates=width, deep_candidates=width, n_movable=10
        ),
    )
    assert out
    for sug in out:
        check_constraints(frame, sug, movers=3, max_move=3.0, min_sep=0.5)


def test_zero_candidate_widths_are_coherent() -> None:
    frame = ten_defender_frame()
    assert suggest(frame, FakeShotModel(), max_candidates=0) == []
    only_one = suggest(frame, FakeShotModel(), deep_candidates=0, movers=3)
    assert only_one
    assert all(len(s.moves) == 1 for s in only_one), "deep_candidates=0 must stop at one mover"


def test_widening_the_search_without_the_budget_warns() -> None:
    """The trap the flags would otherwise set: a wider search that is not wider."""
    frame = ten_defender_frame()
    with pytest.warns(RuntimeWarning, match=r"truncated rather than widened"):
        suggest(frame, FakeShotModel(), max_candidates=None, max_evals=DEFAULT_MAX_EVALS)
    with pytest.warns(RuntimeWarning, match=r"needs up to 352 evaluations"):
        suggest(frame, FakeShotModel(), max_candidates=10, max_evals=DEFAULT_MAX_EVALS)


def test_the_default_layout_and_a_deliberately_small_budget_stay_quiet() -> None:
    """Buying latency back at the default width is supported, not a mistake."""
    frame = ten_defender_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        suggest(frame, FakeShotModel(), movers=3)
        suggest(frame, FakeShotModel(), movers=3, max_evals=40)  # what `demo` does
        suggest(
            frame,
            FakeShotModel(),
            max_candidates=None,
            deep_candidates=None,
            max_evals=required_evals(max_candidates=None, deep_candidates=None, n_movable=10),
        )


def test_a_widened_search_is_a_superset_at_the_first_move() -> None:
    """Widening must not lose a first move the narrow search already had."""
    for trial in range(4):
        frame = random_frame(random.Random(700 + trial), n_attackers=6, n_defenders=10)
        narrow = suggest(frame, FakeShotModel(), movers=1, seed=trial)
        wide = suggest(
            frame,
            FakeShotModel(),
            movers=1,
            seed=trial,
            max_candidates=None,
            max_evals=required_evals(movers=1, max_candidates=None, n_movable=10),
        )
        assert wide
        assert wide[0].delta <= narrow[0].delta + 1e-12


def test_widening_is_still_deterministic() -> None:
    frame = ten_defender_frame()
    kwargs = {
        "max_candidates": None,
        "deep_candidates": None,
        "max_evals": required_evals(max_candidates=None, deep_candidates=None, n_movable=10),
        "seed": 5,
    }
    first = suggest(frame, FakeShotModel(), **kwargs)
    second = suggest(frame, FakeShotModel(), **kwargs)
    assert first == second
    assert [(m.player_idx, m.dx.hex(), m.dy.hex()) for s in first for m in s.moves] == [
        (m.player_idx, m.dx.hex(), m.dy.hex()) for s in second for m in s.moves
    ]


# --------------------------------------------------------------------------- #
# the CLI surface of the same three knobs
# --------------------------------------------------------------------------- #
def test_cli_exposes_the_search_width_flags() -> None:
    from setpiece import cli

    args = cli.build_parser().parse_args(["suggest", "c.json"])
    assert args.max_candidates == MAX_CANDIDATE_DEFENDERS
    assert args.deep_candidates == DEEP_CANDIDATE_DEFENDERS
    assert args.move_keeper is False
    assert args.max_evals is None  # sized from the layout, not fixed

    wide = cli.build_parser().parse_args(
        [
            "suggest",
            "c.json",
            "--max-candidates",
            "all",
            "--deep-candidates",
            "4",
            "--move-keeper",
            "--max-evals",
            "900",
        ]
    )
    assert wide.max_candidates is None
    assert wide.deep_candidates == 4
    assert wide.move_keeper is True
    assert wide.max_evals == 900


@pytest.mark.parametrize("bad", ["-1", "banana", "3.5"])
def test_cli_rejects_a_nonsense_candidate_width(bad: str) -> None:
    from setpiece import cli

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["suggest", "c.json", "--max-candidates", bad])


def test_cli_sizes_the_budget_from_the_layout() -> None:
    """``--max-candidates all`` must widen the search, not truncate it."""
    from setpiece import cli

    frame = ten_defender_frame()
    default = cli.suggest_frame(
        frame, FakeShotModel(), movers=3, max_move=3.0, seed=0, defend=True, beam=8, top_k=8
    )
    assert default["params"]["max_evals"] == DEFAULT_MAX_EVALS
    assert default["params"]["max_candidates"] == MAX_CANDIDATE_DEFENDERS

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a sized budget must not trip the truncation warning
        wide = cli.suggest_frame(
            frame,
            FurthestDefenderScorer(),
            movers=3,
            max_move=3.0,
            seed=0,
            defend=True,
            beam=8,
            top_k=8,
            max_candidates=None,
            deep_candidates=None,
        )
    assert wide["params"]["max_evals"] == required_evals(
        max_candidates=None, deep_candidates=None, n_movable=10
    )
    far_four = set(_outfield_defenders_by_distance(frame)[MAX_CANDIDATE_DEFENDERS:])
    moved = {m["player_idx"] for s in wide["suggestions"] for m in s["moves"]}
    assert moved & far_four


def test_cli_move_keeper_reaches_the_keeper() -> None:
    from setpiece import cli

    frame = ten_defender_frame()
    keeper_idx = next(i for i, p in enumerate(frame.players) if p.is_keeper)
    plain = cli.suggest_frame(
        frame, FakeShotModel(), movers=3, max_move=3.0, seed=0, defend=True, beam=8, top_k=8
    )
    assert all(
        m["player_idx"] != keeper_idx for s in plain["suggestions"] for m in s["moves"]
    )

    with_keeper = cli.suggest_frame(
        frame,
        FakeShotModel(),
        movers=1,
        max_move=3.0,
        seed=0,
        defend=True,
        beam=8,
        top_k=500,  # every scored single move, so "in the pool" is what is tested
        move_keeper=True,
        max_candidates=None,
    )
    assert with_keeper["params"]["move_keeper"] is True
    moved = {m["player_idx"] for s in with_keeper["suggestions"] for m in s["moves"]}
    assert keeper_idx in moved


def test_pretty_output_states_the_search_width() -> None:
    """The bound used to be invisible; now every suggestion carries it."""
    from setpiece import cli

    frame = ten_defender_frame()
    result = cli.suggest_frame(
        frame, FakeShotModel(), movers=3, max_move=3.0, seed=0, defend=True, beam=8, top_k=3
    )
    text = cli.format_suggestions([result], "pretty")
    assert "search width: 6 defenders for the first move, 2 for later moves" in text
    assert "keeper excluded" in text
    assert f"budget {DEFAULT_MAX_EVALS} evaluations" in text
    assert "--max-candidates" in text

    wide = cli.suggest_frame(
        frame,
        FakeShotModel(),
        movers=3,
        max_move=3.0,
        seed=0,
        defend=True,
        beam=8,
        top_k=3,
        max_candidates=None,
        deep_candidates=None,
        move_keeper=True,
    )
    wide_text = cli.format_suggestions([wide], "pretty")
    assert "all defenders for the first move, all for later moves" in wide_text
    assert "keeper movable" in wide_text


def _tight_pairs(frame: CornerFrame, min_sep: float) -> set[tuple[int, int]]:
    return {
        (i, j)
        for i, a in enumerate(frame.players)
        for j, b in enumerate(frame.players)
        if i < j and math.hypot(a.x - b.x, a.y - b.y) < min_sep - TOL
    }
