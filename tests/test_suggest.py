"""Tests for the defender adjustment search.

These are standalone: the shot model is a hand-written stand-in, so nothing here
depends on setpiece.models, setpiece.features or a trained checkpoint.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from setpiece._io import read_corners
from setpiece.suggest import (
    DEFAULT_MAX_EVALS,
    Move,
    Suggestion,
    apply_moves,
    candidate_displacements,
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


def _tight_pairs(frame: CornerFrame, min_sep: float) -> set[tuple[int, int]]:
    return {
        (i, j)
        for i, a in enumerate(frame.players)
        for j, b in enumerate(frame.players)
        if i < j and math.hypot(a.x - b.x, a.y - b.y) < min_sep - TOL
    }
