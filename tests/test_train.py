"""Tests for training, splitting and evaluation.

The three things that would quietly destroy the credibility of the metrics table
are a split that leaks a match across folds, a receiver metric computed over rows
that carry no receiver, and a home-grown AUC that is subtly wrong. Those get the
most attention here. The end-to-end run is exercised on a handful of synthetic
corners so it stays fast; the real smoke dataset is used when it is present.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from setpiece._io import read_corners, write_corners
from setpiece.models import load_model
from setpiece.train import (
    Row,
    build_parser,
    load_rows,
    purged_fold_assignment,
    receiver_metrics,
    roc_auc,
    set_seed,
    shot_metrics,
    train_all,
)
from setpiece.types import PITCH_LENGTH, CornerFrame, Player

REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / "data" / "interim" / "smoke.jsonl"


# --------------------------------------------------------------------------- #
# synthetic corners
# --------------------------------------------------------------------------- #
def make_frame(
    corner_id: str,
    match_id: int,
    *,
    seed: int = 0,
    first_touch: str = "attacker",
    receiver_slot: int | None = 0,
    shot: bool = False,
    n_attackers: int = 5,
    n_defenders: int = 6,
) -> CornerFrame:
    """A plausible corner: attackers and defenders scattered around the six-yard box."""
    rng = np.random.default_rng(seed)
    players: list[Player] = [Player(x=PITCH_LENGTH - 0.4, y=0.4, is_attacker=True, is_taker=True)]
    for _ in range(n_attackers):
        players.append(
            Player(
                x=float(rng.uniform(88.0, 103.0)),
                y=float(rng.uniform(24.0, 44.0)),
                is_attacker=True,
            )
        )
    for i in range(n_defenders):
        players.append(
            Player(
                x=float(rng.uniform(90.0, 104.0)),
                y=float(rng.uniform(24.0, 44.0)),
                is_attacker=False,
                is_keeper=(i == 0),
            )
        )
    # candidate attackers are indices 1 .. n_attackers (the taker is index 0)
    receiver_idx = None if receiver_slot is None else 1 + receiver_slot
    return CornerFrame(
        corner_id=corner_id,
        players=players,
        match_id=match_id,
        competition="Test League",
        season="2024",
        first_touch=first_touch,
        receiver_idx=receiver_idx if first_touch == "attacker" else None,
        receiver_match_dist=1.0 if receiver_idx is not None else None,
        shot_within_5=shot,
        meta={"end_xy": [100.0, 34.0], "pass_outcome": None},
    )


def make_dataset(path: Path, *, matches: int = 6, per_match: int = 3) -> Path:
    frames: list[CornerFrame] = []
    k = 0
    for m in range(matches):
        for c in range(per_match):
            k += 1
            attacker = k % 3 != 0
            frames.append(
                make_frame(
                    f"c{m}_{c}",
                    match_id=1000 + m,
                    seed=k,
                    first_touch="attacker" if attacker else "defender",
                    receiver_slot=(k % 5) if attacker else None,
                    shot=(k % 4 == 0),
                )
            )
    write_corners(frames, path)
    return path


# --------------------------------------------------------------------------- #
# AUC
# --------------------------------------------------------------------------- #
def test_auc_hand_computed_example() -> None:
    """Four points, no ties: 2 positives, 2 negatives.

    Scores: neg 0.1, pos 0.4, neg 0.35, pos 0.8. Of the four positive/negative
    pairs, (0.4, 0.1), (0.4, 0.35), (0.8, 0.1) and (0.8, 0.35) all have the
    positive ranked above the negative, so the AUC is 4/4 = 1.0.
    """
    assert roc_auc([0, 1, 0, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(1.0)


def test_auc_hand_computed_partial() -> None:
    """Scores: pos 0.3, neg 0.5, pos 0.6, neg 0.2.

    Pairs (pos, neg): (0.3, 0.5) loses, (0.3, 0.2) wins, (0.6, 0.5) wins,
    (0.6, 0.2) wins. Three of four, so 0.75.
    """
    assert roc_auc([1, 0, 1, 0], [0.3, 0.5, 0.6, 0.2]) == pytest.approx(0.75)


def test_auc_ties_count_half() -> None:
    """One positive and one negative sharing a score is exactly half a win."""
    assert roc_auc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    # pos 0.9, pos 0.4, neg 0.4, neg 0.1 -> pairs: 1, 1, 0.5, 1 out of 4 = 0.875
    assert roc_auc([1, 1, 0, 0], [0.9, 0.4, 0.4, 0.1]) == pytest.approx(0.875)


def test_auc_constant_score_is_chance() -> None:
    assert roc_auc([1, 0, 1, 0, 0], [2.0] * 5) == pytest.approx(0.5)


def test_auc_perfectly_wrong_is_zero() -> None:
    assert roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.0)


def test_auc_matches_brute_force_on_random_data() -> None:
    rng = np.random.default_rng(7)
    for _ in range(20):
        y = rng.integers(0, 2, size=40)
        s = rng.normal(size=40).round(1)  # rounding forces plenty of ties
        pos = s[y == 1]
        neg = s[y == 0]
        if pos.size == 0 or neg.size == 0:
            continue
        wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        assert roc_auc(y, s) == pytest.approx(wins / (pos.size * neg.size))


def test_auc_single_class_is_nan() -> None:
    assert math.isnan(roc_auc([1, 1, 1], [0.1, 0.5, 0.9]))
    assert math.isnan(roc_auc([0, 0], [0.1, 0.5]))


def test_auc_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        roc_auc([1, 0, 1], [0.1, 0.2])


def test_auc_rejects_non_finite_scores() -> None:
    """A missing prediction must fail loudly, not sort to the top.

    ``np.argsort`` puts ``nan`` last, so an unscored row would silently take the
    best rank: without the guard ``roc_auc([0, 1, 1], [0.1, nan, 0.9])`` returns
    a perfect 1.0. The out-of-fold arrays are pre-filled with ``nan``, so a hole
    in them has to raise rather than read as skill.
    """
    with pytest.raises(ValueError):
        roc_auc([0, 1, 1], [0.1, float("nan"), 0.9])
    with pytest.raises(ValueError):
        roc_auc([0, 1], [0.1, float("inf")])


# --------------------------------------------------------------------------- #
# purged splits
# --------------------------------------------------------------------------- #
def test_no_match_appears_in_two_folds() -> None:
    groups = [str(1000 + (i % 17)) for i in range(200)]
    assignment = purged_fold_assignment(groups, 5, seed=3)
    by_group: dict[str, set[int]] = {}
    for group, fold in zip(groups, assignment, strict=True):
        by_group.setdefault(group, set()).add(int(fold))
    offenders = {g: folds for g, folds in by_group.items() if len(folds) > 1}
    assert offenders == {}, f"matches split across folds: {offenders}"


def test_every_row_assigned_and_all_folds_used() -> None:
    groups = [str(i // 4) for i in range(80)]
    assignment = purged_fold_assignment(groups, 4, seed=1)
    assert assignment.shape == (80,)
    assert set(assignment.tolist()) == {0, 1, 2, 3}
    assert (assignment >= 0).all()


def test_folds_are_balanced() -> None:
    groups = [str(i // 3) for i in range(90)]
    assignment = purged_fold_assignment(groups, 5, seed=0)
    sizes = [int((assignment == f).sum()) for f in range(5)]
    assert max(sizes) - min(sizes) <= 3


def test_split_is_deterministic_given_seed() -> None:
    groups = [str(i % 11) for i in range(60)]
    a = purged_fold_assignment(groups, 4, seed=42)
    b = purged_fold_assignment(groups, 4, seed=42)
    c = purged_fold_assignment(groups, 4, seed=43)
    assert (a == b).all()
    assert isinstance(c, np.ndarray)


def test_fold_count_clamped_to_group_count() -> None:
    groups = [str(i % 3) for i in range(30)]
    assignment = purged_fold_assignment(groups, 10, seed=0)
    assert int(assignment.max()) + 1 == 3


def test_split_needs_at_least_two_groups() -> None:
    with pytest.raises(ValueError):
        purged_fold_assignment(["only"] * 10, 5)
    with pytest.raises(ValueError):
        purged_fold_assignment(["a", "b"], 1)


def test_real_dataset_splits_are_purged() -> None:
    if not SMOKE.exists():
        pytest.skip("smoke dataset not built")
    corners = read_corners(SMOKE)
    groups = [str(c.match_id) for c in corners]
    assignment = purged_fold_assignment(groups, 3, seed=0)
    seen: dict[str, int] = {}
    for group, fold in zip(groups, assignment, strict=True):
        assert seen.setdefault(group, int(fold)) == int(fold)


# --------------------------------------------------------------------------- #
# valid-row filtering
# --------------------------------------------------------------------------- #
def test_off_camera_receiver_row_is_invalid(tmp_path: Path) -> None:
    """attacker-first but no matched receiver -> dropped from the receiver task."""
    frame = make_frame("x", 1, first_touch="attacker", receiver_slot=None)
    path = tmp_path / "one.jsonl"
    write_corners([frame], path)
    row = load_rows(path)[0]
    assert row.receiver_valid is False
    assert row.target_pos == -1


def test_defender_first_touch_row_is_kept(tmp_path: Path) -> None:
    """The defence winning first contact is a class, not a missing label."""
    frame = make_frame("x", 1, first_touch="defender", receiver_slot=None)
    path = tmp_path / "one.jsonl"
    write_corners([frame], path)
    row = load_rows(path)[0]
    assert row.receiver_valid is True
    assert row.target_pos == -1


def test_unknown_first_touch_row_is_invalid(tmp_path: Path) -> None:
    frame = make_frame("x", 1, first_touch="unknown", receiver_slot=None)
    path = tmp_path / "one.jsonl"
    write_corners([frame], path)
    assert load_rows(path)[0].receiver_valid is False


def test_target_pos_indexes_candidates_not_players(tmp_path: Path) -> None:
    """The taker is player 0 but is not a candidate, so the slots shift by one."""
    frame = make_frame("x", 1, first_touch="attacker", receiver_slot=2)
    path = tmp_path / "one.jsonl"
    write_corners([frame], path)
    row = load_rows(path)[0]
    assert row.receiver_valid is True
    assert row.target_pos == 2
    assert int(row.cand_nodes[row.target_pos]) == 3


def test_invalid_rows_excluded_from_evaluation(tmp_path: Path) -> None:
    """Only valid rows reach the receiver metrics; the smoke set has one drop."""
    path = make_dataset(tmp_path / "d.jsonl", matches=4, per_match=3)
    frames = read_corners(path)
    frames.append(make_frame("dropme", 1003, first_touch="attacker", receiver_slot=None))
    write_corners(frames, path)

    rows = load_rows(path)
    assert sum(1 for r in rows if not r.receiver_valid) == 1

    report = train_all(
        path,
        tmp_path / "out",
        folds=2,
        epochs=1,
        hidden=8,
        layers=2,
        batch_size=8,
        baseline_iters=5,
        metrics_path=tmp_path / "metrics.md",
        quiet=True,
    )
    n_valid = sum(1 for r in rows if r.receiver_valid)
    assert report["receiver"]["gnn"]["n_rows"] == n_valid
    assert report["receiver"]["logreg"]["n_rows"] == n_valid
    assert report["receiver"]["heuristic"]["n_rows"] == n_valid
    # the shot head keeps every row
    assert report["shot"]["gnn"]["n_rows"] == len(rows)


# --------------------------------------------------------------------------- #
# receiver metrics
# --------------------------------------------------------------------------- #
def test_receiver_metrics_hand_computed() -> None:
    cand_probs = [
        np.array([0.6, 0.2, 0.1, 0.1]),  # target 0 -> top-1 hit
        np.array([0.1, 0.2, 0.5, 0.2]),  # target 3 -> rank 1 (tie with idx1 at 0.2)
        np.array([0.4, 0.3, 0.2, 0.1]),  # defence row
        np.array([0.05, 0.05, 0.05, 0.05]),  # target 3 -> rank 3, top-3 miss
    ]
    p_defence = np.array([0.0, 0.0, 0.8, 0.8])
    target = np.array([0, 3, -1, 3])
    m = receiver_metrics(cand_probs, p_defence, target)

    assert m["n_rows"] == 4
    assert m["n_attacker_first"] == 3
    assert m["top1_attacker"] == pytest.approx(1 / 3)
    # rows 0 and 1 are inside the top three; row 3's target ranks last of four
    assert m["top3_attacker"] == pytest.approx(2 / 3)
    # row 0 correct, row 1 wrong, row 2 defence predicted and correct, row 3 defence wrongly
    assert m["overall_accuracy"] == pytest.approx(2 / 4)
    # one positive (row 2, p 0.8) vs three negatives (0.0, 0.0, 0.8) -> (1 + 1 + 0.5) / 3
    assert m["defence_auc"] == pytest.approx(2.5 / 3)


def test_receiver_metrics_ties_do_not_flatter() -> None:
    """A target tied with a lower-indexed candidate must not count as top-1."""
    m = receiver_metrics([np.array([0.5, 0.5])], [0.0], [1])
    assert m["top1_attacker"] == pytest.approx(0.0)
    assert m["top3_attacker"] == pytest.approx(1.0)


def test_receiver_metrics_rejects_ragged_input() -> None:
    with pytest.raises(ValueError):
        receiver_metrics([np.array([1.0])], [0.0, 0.0], [0, 0])


def test_shot_metrics_reports_base_rate() -> None:
    m = shot_metrics(np.array([1.0, 0.0, 0.0, 1.0]), np.array([0.9, 0.1, 0.2, 0.8]))
    assert m["auc"] == pytest.approx(1.0)
    assert m["base_rate"] == pytest.approx(0.5)
    assert m["n_rows"] == 4


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def test_train_all_end_to_end(tmp_path: Path) -> None:
    path = make_dataset(tmp_path / "d.jsonl", matches=6, per_match=3)
    out = tmp_path / "run"
    metrics = tmp_path / "metrics.md"
    report = train_all(
        path,
        out,
        folds=3,
        epochs=2,
        hidden=8,
        layers=2,
        batch_size=8,
        baseline_iters=10,
        metrics_path=metrics,
        quiet=True,
    )

    assert report["splits"]["n_folds"] == 3
    assert sum(report["splits"]["rows_per_fold"]) == 18
    assert report["data"]["n_matches"] == 6

    for name in ("receiver", "shot"):
        checkpoint = out / f"{name}.pt"
        assert checkpoint.exists()
        model, card = load_model(checkpoint)
        assert model.training is False
        for key in ("data", "splits", "metrics", "date", "features", "params", "assume_static"):
            assert key in card, f"{name} card is missing {key}"
        assert card["assume_static"] is True
        assert card["params"] > 0
        assert card["splits"]["n_folds"] == 3
        assert card["features"]["n_node_features"] > 0

    text = metrics.read_text()
    assert "receiver head" in text
    assert "shot head" in text
    assert "| GNN |" in text
    assert "purged" in text
    assert (out / "report.json").exists()

    # Top-3 has a floor: a corner with three or fewer candidates is a free hit for
    # every model, so the table has to print the chance level next to the column
    # or 0.7 looks like skill it may not be.
    chance3 = report["receiver"]["chance"]["top3_attacker"]
    assert f"{chance3:.3f}" in text, "the top-3 chance level is computed but not reported"
    assert "free hit" in text
    # a fully grouped dataset must not carry the ungrouped caveat
    assert report["splits"]["n_ungrouped_corners"] == 0
    assert "carry no `match_id`" not in text


def test_corners_without_match_id_are_flagged_as_unpurged(tmp_path: Path) -> None:
    """A corner with no ``match_id`` is its own group, so it is not purged.

    That is the safe default, but the metrics table claims corners are grouped by
    match, and for these rows that claim is unverifiable. It has to say so.
    """
    frames = [
        make_frame(f"c{i}", match_id=1000 + i // 3, seed=i, shot=(i % 3 == 0))
        for i in range(18)
    ]
    for frame in frames[:5]:
        frame.match_id = None
    path = tmp_path / "d.jsonl"
    write_corners(frames, path)

    metrics = tmp_path / "metrics.md"
    report = train_all(
        path,
        tmp_path / "run",
        folds=2,
        epochs=1,
        hidden=8,
        layers=2,
        batch_size=8,
        baseline_iters=5,
        metrics_path=metrics,
        quiet=True,
    )
    assert report["splits"]["n_ungrouped_corners"] == 5
    text = metrics.read_text()
    assert "carry no `match_id`" in text
    assert "*not* purged" in text


def test_metrics_are_in_range(tmp_path: Path) -> None:
    path = make_dataset(tmp_path / "d.jsonl", matches=4, per_match=3)
    report = train_all(
        path,
        tmp_path / "run",
        folds=2,
        epochs=1,
        hidden=8,
        layers=2,
        batch_size=8,
        baseline_iters=5,
        metrics_path=None,
        quiet=True,
    )
    for family in ("receiver", "shot"):
        for name in ("gnn", "heuristic", "logreg"):
            for key, value in report[family][name].items():
                if key.startswith("n_"):
                    continue
                assert math.isnan(value) or 0.0 <= value <= 1.0, f"{family}/{name}/{key}={value}"


def test_training_is_reproducible(tmp_path: Path) -> None:
    path = make_dataset(tmp_path / "d.jsonl", matches=4, per_match=3)
    kwargs = dict(
        folds=2, epochs=2, hidden=8, layers=2, batch_size=8, baseline_iters=10,
        metrics_path=None, quiet=True, seed=11,
    )
    a = train_all(path, tmp_path / "a", **kwargs)
    b = train_all(path, tmp_path / "b", **kwargs)
    assert a["receiver"]["gnn"] == b["receiver"]["gnn"]
    assert a["shot"]["gnn"] == b["shot"]["gnn"]


def test_set_seed_is_deterministic() -> None:
    set_seed(5)
    a = torch.randn(4)
    set_seed(5)
    assert torch.equal(a, torch.randn(4))


def test_parser_defaults_and_overrides() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.folds >= 2
    assert args.epochs >= 1
    assert args.metrics_path.endswith("metrics.md")

    args = parser.parse_args(["--epochs", "3", "--hidden", "16", "--limit", "7", "--quiet"])
    assert (args.epochs, args.hidden, args.limit, args.quiet) == (3, 16, 7, True)


def test_load_rows_respects_limit(tmp_path: Path) -> None:
    path = make_dataset(tmp_path / "d.jsonl", matches=3, per_match=4)
    assert len(load_rows(path, limit=5)) == 5
    assert len(load_rows(path)) == 12


def test_load_rows_rejects_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError):
        load_rows(empty)


def test_rows_carry_pooled_features(tmp_path: Path) -> None:
    from setpiece import features as featmod

    path = make_dataset(tmp_path / "d.jsonl", matches=2, per_match=2)
    row = load_rows(path)[0]
    assert isinstance(row, Row)
    assert row.flat.shape[1] == featmod.N_NODE_FEATURES
    assert row.pooled.shape == (4 * featmod.N_NODE_FEATURES,)
    assert np.isfinite(row.pooled).all()


@pytest.mark.slow
def test_smoke_dataset_trains(tmp_path: Path) -> None:
    if not SMOKE.exists():
        pytest.skip("smoke dataset not built")
    report = train_all(
        SMOKE,
        tmp_path / "run",
        folds=3,
        epochs=2,
        hidden=16,
        layers=2,
        batch_size=8,
        baseline_iters=20,
        metrics_path=tmp_path / "metrics.md",
        quiet=True,
    )
    assert report["data"]["n_corners"] == 18
    assert (tmp_path / "run" / "receiver.pt").exists()
