"""Tests for training, splitting and evaluation.

The three things that would quietly destroy the credibility of the metrics table
are a split that leaks a match across folds, a receiver metric computed over rows
that carry no receiver, and a home-grown AUC that is subtly wrong. Those get the
most attention here. The end-to-end run is exercised on a handful of synthetic
corners so it stays fast; the real smoke dataset is used when it is present.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from setpiece import train as trainmod
from setpiece._io import read_corners, write_corners
from setpiece.calibration import calibrator_from_card, calibrator_from_dict
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
        seeds=2,
        metrics_path=metrics,
        quiet=True,
    )

    assert report["splits"]["n_folds"] == 3
    assert sum(report["splits"]["rows_per_fold"]) == 18
    assert report["data"]["n_matches"] == 6
    assert len(report["seeds"]) == 2
    assert len(report["per_seed"]) == 2

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
        # a card that quotes a mean over repeats has to carry the spread with it,
        # or the reader cannot tell a result from drift
        assert card["seeds"] == report["seeds"]
        assert card["metrics_spread"]["gnn"]

    text = metrics.read_text()
    assert "receiver head" in text
    assert "shot head" in text
    assert "| GNN |" in text
    assert "purged" in text
    assert (out / "report.json").exists()
    assert (out / "oof.npz").exists()

    # Top-3 has a floor: a corner with three or fewer candidates is a free hit for
    # every model, so the table has to print the chance level next to the column
    # or 0.7 looks like skill it may not be.
    chance3 = report["receiver"]["chance"]["top3_attacker"]
    assert f"{chance3:.3f}" in text, "the top-3 chance level is computed but not reported"
    assert "free hit" in text
    # a fully grouped dataset must not carry the ungrouped caveat
    assert report["splits"]["n_ungrouped_corners"] == 0
    assert "carry no `match_id`" not in text
    # every cell of a multi-repeat table has to show its own spread
    assert "±" in text
    assert "mean over 2 repeats" in text
    assert "## calibration" in text
    assert "## receiver label threshold" in text
    # the thread count is part of the recipe, not decoration
    assert f"torch.set_num_threads({report['training']['torch_threads']})" in text


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
        seeds=1,
        calibrate=False,
        sweep_threshold=False,
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
        seeds=1,
        calibrate=False,
        sweep_threshold=False,
        metrics_path=None,
        quiet=True,
    )
    for family in ("receiver", "shot"):
        for name in ("gnn", "heuristic", "logreg"):
            for key, value in report[family][name].items():
                if key.startswith("n_"):
                    continue
                assert math.isnan(value) or 0.0 <= value <= 1.0, f"{family}/{name}/{key}={value}"


@pytest.mark.slow
def test_training_is_reproducible(tmp_path: Path) -> None:
    """Two identical runs must agree.

    Marked slow because it is the only test that runs ``train_all`` twice over
    the same rows purely to compare the results, so the second cross-validation
    is duplicated work by construction. It is also the test most sensitive to
    machine state: per ``TODO.md`` the GNN is only bit-reproducible at a fixed
    torch thread count, so it belongs in the deliberate run rather than in the
    edit loop. ``test_train_all_end_to_end`` keeps the pipeline itself covered
    in the fast subset.
    """
    path = make_dataset(tmp_path / "d.jsonl", matches=4, per_match=3)
    kwargs = dict(
        folds=2, epochs=2, hidden=8, layers=2, batch_size=8, baseline_iters=10,
        metrics_path=None, quiet=True, seed=11, seeds=2,
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


# --------------------------------------------------------------------------- #
# what the dataset is: label windows and 360 coverage
# --------------------------------------------------------------------------- #
def guarded_dataset(path: Path) -> tuple[Path, Path]:
    """A pair of builds of the same corners, one before the guards and one after.

    The two differ only in the way the real pair does: a first touch recorded far
    outside ``FIRST_TOUCH_MAX_S`` is a class in the unguarded build and
    ``"unknown"`` in the guarded one, and a shot recorded outside ``SHOT_MAX_S``
    is dropped. Nothing here re-derives the label from the timing; both files are
    written the way ``_io`` would have written them.
    """
    before: list[CornerFrame] = []
    after: list[CornerFrame] = []
    for i in range(8):
        late = i < 3  # three corners whose "first touch" is a different phase of play
        dt = 120.0 if late else 1.5
        base = make_frame(
            f"c{i}", match_id=100 + i // 2, seed=i, first_touch="defender", receiver_slot=None
        )
        before.append(
            CornerFrame(
                **{
                    **base.__dict__,
                    "shot_within_5": late,
                    "meta": {**base.meta, "first_touch_dt": dt, "shot_dt": dt},
                }
            )
        )
        after.append(
            CornerFrame(
                **{
                    **base.__dict__,
                    "first_touch": "unknown" if late else "defender",
                    "shot_within_5": False,
                    "meta": {**base.meta, "first_touch_dt": dt, "shot_dt": dt},
                }
            )
        )
    guarded, unguarded = path / "after.jsonl", path / "before.jsonl"
    write_corners(after, guarded)
    write_corners(before, unguarded)
    return guarded, unguarded


def test_label_summary_counts_what_the_guard_caught(tmp_path: Path) -> None:
    guarded, _ = guarded_dataset(tmp_path)
    summary = trainmod.label_window_summary(load_rows(guarded))
    assert summary["first_touch"] == {"defender": 5, "attacker": 0, "unknown": 3}
    assert summary["late_first_touch"]["n"] == 3
    assert summary["late_first_touch"]["now_unknown"] == 3
    assert summary["late_first_touch"]["max_s"] == pytest.approx(120.0)
    assert summary["first_touch_dt"]["median"] == pytest.approx(1.5)
    assert "baseline" not in summary


def test_only_a_pre_guard_build_can_say_which_class_a_guarded_row_used_to_carry(
    tmp_path: Path,
) -> None:
    """The guarded dataset records *that* a touch was implausible, not whose it was."""
    guarded, unguarded = guarded_dataset(tmp_path)
    rows = load_rows(guarded)
    assert "baseline" not in trainmod.label_window_summary(rows)

    summary = trainmod.label_window_summary(rows, baseline_path=unguarded)
    assert summary["baseline"]["first_touch"] == {"defender": 8, "attacker": 0, "unknown": 0}
    assert summary["baseline"]["late_first_touch_was"]["defender"] == 3
    assert summary["baseline"]["shot_flips"] == 3
    assert summary["baseline"]["shot_base_rate"] == pytest.approx(3 / 8)


def test_the_report_prints_the_guarded_labels_and_names_the_before_and_after(
    tmp_path: Path,
) -> None:
    guarded, unguarded = guarded_dataset(tmp_path)
    report = train_all(
        guarded, tmp_path / "out", folds=2, epochs=1, hidden=8, layers=2, batch_size=8,
        baseline_iters=5, seeds=1, calibrate=False, sweep_threshold=False,
        metrics_path=None, quiet=True, label_baseline=unguarded,
    )
    markdown = trainmod.format_metrics_markdown(report)
    assert "3 unknown" in markdown
    assert "plausibility guards" in markdown
    assert "| first touch: unknown | 0 | 3 |" in markdown
    # the guards are the reason the unknown class exists, and the report says so
    assert "never relabelled" in markdown
    # the card a checkpoint ships carries the same composition
    _, card = load_model(tmp_path / "out" / "shot.pt")
    assert card["data"]["labels"]["first_touch"]["unknown"] == 3


def test_the_report_and_the_cards_record_the_coordinate_convention(tmp_path: Path) -> None:
    """A checkpoint from before the reflection has identical feature names.

    Nothing about the shape of the input distinguishes the two conventions, so
    the only thing that can is a written record travelling with the weights.
    """
    path = make_dataset(tmp_path / "d.jsonl", matches=4, per_match=2)
    report = train_all(
        path, tmp_path / "out", folds=2, epochs=1, hidden=8, layers=2, batch_size=8,
        baseline_iters=5, seeds=1, calibrate=False, sweep_threshold=False,
        metrics_path=None, quiet=True,
    )
    assert "y = GOAL_Y" in report["features"]["coordinates"]
    assert "mirrored coordinates" in trainmod.format_metrics_markdown(report)
    for name in ("receiver", "shot"):
        _, card = load_model(tmp_path / "out" / f"{name}.pt")
        assert "y = GOAL_Y" in card["features"]["coordinates"]
        assert card["features"]["n_node_features"] == len(card["features"]["node_features"])


def test_the_report_omits_the_before_and_after_when_no_baseline_was_given(
    tmp_path: Path,
) -> None:
    guarded, _ = guarded_dataset(tmp_path)
    report = train_all(
        guarded, tmp_path / "out", folds=2, epochs=1, hidden=8, layers=2, batch_size=8,
        baseline_iters=5, seeds=1, calibrate=False, sweep_threshold=False,
        metrics_path=None, quiet=True,
    )
    markdown = trainmod.format_metrics_markdown(report)
    assert "3 unknown" in markdown
    assert "before the guards" not in markdown


def test_coverage_folds_seasons_into_competitions_and_finds_the_empty_ones(
    tmp_path: Path,
) -> None:
    import json

    dataset = tmp_path / "corners.jsonl"
    make_dataset(dataset, matches=2, per_match=2)
    (tmp_path / "corners.state.json").write_text(
        json.dumps(
            {
                "coverage": {
                    "Ligue 1 2021/2022": {"corners_seen": 100, "corners_with_frame": 40,
                                          "dropped_too_few_players": 1},
                    "Ligue 1 2022/2023": {"corners_seen": 100, "corners_with_frame": 60,
                                          "dropped_too_few_players": 0},
                    "La Liga 2020/2021": {"corners_seen": 50, "corners_with_frame": 40,
                                          "dropped_too_few_players": 0},
                    "African Cup of Nations 2023": {"corners_seen": 60,
                                                    "corners_with_frame": 0,
                                                    "dropped_too_few_players": 0},
                }
            }
        )
    )
    cov = trainmod.coverage_summary(dataset)
    assert (cov["corners_seen"], cov["corners_with_frame"]) == (310, 140)
    assert cov["rate"] == pytest.approx(140 / 310)
    assert cov["dropped_too_few_players"] == 1
    # the two Ligue 1 seasons are one row, at their pooled rate
    assert {e["competition"]: e["with_frame"] for e in cov["entries"]}["Ligue 1"] == 100
    # a competition that ships no frames is named, and never becomes the "worst covered"
    assert cov["uncovered"] == ["African Cup of Nations"]
    assert cov["min"]["competition"] == "Ligue 1"
    assert cov["max"]["competition"] == "La Liga"


def test_the_ablation_section_separates_a_step_that_changed_the_rows(tmp_path: Path) -> None:
    """Two columns with different labels are not scored on the same rows.

    Reporting such a step as though it were a model improvement is the exact
    mistake the section exists to prevent, so it has to be marked in the output.
    """
    import json

    payload = {
        "model": "gnn",
        "table": {
            "receiver top-1": {"before": "0.4254 ±0.0035", "mirrored": "0.4300 ±0.0040"},
            "shot AUC": {"before": "0.5768 ±0.0149", "mirrored": "0.5800 ±0.0150"},
        },
        "steps": [
            {"from": "before", "to": "mirrored", "metric": "receiver top-1",
             "mean": 0.0046, "sd": 0.0010, "ratio": 4.6, "same_rows": False},
            {"from": "before", "to": "mirrored", "metric": "shot AUC",
             "mean": 0.0032, "sd": 0.0090, "ratio": 0.4, "same_rows": False},
        ],
    }
    section = "\n".join(trainmod._ablation_section({"ablation": [payload]}))
    assert "## what moved, and why" in section
    assert "### `gnn`" in section
    assert "| metric | before | mirrored |" in section
    assert "| receiver top-1 | +0.0046 | ±0.0010 | 4.6x | outside the noise floor |" in section
    assert "| shot AUC | +0.0032 | ±0.0090 | 0.4x | **inside the noise floor** |" in section
    assert "not scored on the same rows" in section

    # a step that kept the rows carries no such warning, and so is readable as a
    # model comparison rather than as a change of question
    same = {**payload, "steps": [{**s, "same_rows": True} for s in payload["steps"]]}
    assert "not scored on the same rows" not in "\n".join(trainmod._ablation_section(
        {"ablation": [same]}
    ))

    # a differenced pair is labelled as the gap it is, not as a third model
    gap = {**payload, "model": "gnn-logreg"}
    assert "the gap itself" in "\n".join(trainmod._ablation_section({"ablation": [gap]}))

    # and it reaches the report from the flag rather than being recomputed there
    path = tmp_path / "ablation.json"
    path.write_text(json.dumps(payload))
    gap_path = tmp_path / "ablation-gap.json"
    gap_path.write_text(json.dumps({**payload, "model": "gnn-logreg"}))
    dataset = make_dataset(tmp_path / "d.jsonl", matches=4, per_match=2)
    report = train_all(
        dataset, tmp_path / "out", folds=2, epochs=1, hidden=8, layers=2, batch_size=8,
        baseline_iters=5, seeds=1, calibrate=False, sweep_threshold=False,
        metrics_path=None, quiet=True, ablation=[path, gap_path],
    )
    assert [a["model"] for a in report["ablation"]] == ["gnn", "gnn-logreg"]
    markdown = trainmod.format_metrics_markdown(report)
    assert "### `gnn`" in markdown and "### `gnn-logreg`" in markdown


def test_the_ablation_section_is_absent_when_no_attribution_was_supplied() -> None:
    assert trainmod._ablation_section({}) == []
    empty = {"model": "gnn", "table": {}, "steps": []}
    assert trainmod._ablation_section({"ablation": [empty]}) == []


def test_coverage_is_absent_rather_than_guessed_when_there_is_no_ledger(
    tmp_path: Path,
) -> None:
    """A corner with no frame never became a row, so nothing can reconstruct this."""
    dataset = make_dataset(tmp_path / "corners.jsonl", matches=2, per_match=2)
    assert trainmod.coverage_summary(dataset) is None
    (tmp_path / "corners.state.json").write_text('{"n_corners": 4}')
    assert trainmod.coverage_summary(dataset) is None


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
        seeds=1,
        calibrate=False,
        sweep_threshold=False,
        metrics_path=tmp_path / "metrics.md",
        quiet=True,
    )
    assert report["data"]["n_corners"] == 18
    assert (tmp_path / "run" / "receiver.pt").exists()


# --------------------------------------------------------------------------- #
# repeated seeds
# --------------------------------------------------------------------------- #
def test_seed_list_separates_repeats_far_enough_not_to_collide() -> None:
    """Each repeat uses ``seed + fold`` per fold, so repeats must not overlap.

    Two repeats that shared a per-fold seed would share an initialisation, and a
    spread computed across them would understate the very drift it exists to
    measure.
    """
    seeds = trainmod.seed_list(0, 5)
    assert seeds == [0, 1000, 2000, 3000, 4000]
    per_fold = [s + fold for s in seeds for fold in range(20)]
    assert len(set(per_fold)) == len(per_fold)


def test_seed_list_rejects_zero_repeats() -> None:
    with pytest.raises(ValueError):
        trainmod.seed_list(0, 0)


def test_aggregate_reports_mean_spread_and_the_raw_repeats() -> None:
    agg = trainmod._aggregate([0.4, 0.6, 0.5])
    assert agg["mean"] == pytest.approx(0.5)
    assert agg["sd"] == pytest.approx(np.std([0.4, 0.6, 0.5], ddof=1))
    assert (agg["min"], agg["max"], agg["n"]) == (0.4, 0.6, 3)
    assert agg["values"] == [0.4, 0.6, 0.5]


def test_a_single_repeat_has_no_spread_rather_than_a_missing_one() -> None:
    """``sd`` of one sample is undefined; reporting ``nan`` there reads as a bug.

    "We ran this once" is said by ``n``, so the spread is 0.0 and the count is 1.
    """
    agg = trainmod._aggregate([0.42])
    assert agg["sd"] == 0.0
    assert agg["n"] == 1


def test_aggregate_drops_non_finite_repeats_instead_of_poisoning_the_mean() -> None:
    agg = trainmod._aggregate([0.4, float("nan"), 0.6])
    assert agg["mean"] == pytest.approx(0.5)
    assert agg["n"] == 2
    assert math.isnan(trainmod._aggregate([float("nan")])["mean"])


def test_summarise_seeds_keeps_the_shape_of_a_single_seed_report() -> None:
    """``means`` must look exactly like one seed's report, or every consumer breaks."""
    per_seed = [
        {
            "receiver": {"gnn": {"top1_attacker": 0.40}, "always_defence_accuracy": 0.6},
            "shot": {"gnn": {"auc": 0.55}},
        },
        {
            "receiver": {"gnn": {"top1_attacker": 0.44}, "always_defence_accuracy": 0.6},
            "shot": {"gnn": {"auc": 0.59}},
        },
    ]
    means, spread = trainmod.summarise_seeds(per_seed)
    assert means["receiver"]["gnn"]["top1_attacker"] == pytest.approx(0.42)
    assert means["receiver"]["always_defence_accuracy"] == pytest.approx(0.6)
    assert means["shot"]["gnn"]["auc"] == pytest.approx(0.57)
    assert spread["receiver"]["gnn"]["top1_attacker"]["values"] == [0.40, 0.44]
    assert spread["shot"]["gnn"]["auc"]["n"] == 2


def test_per_seed_summary_does_not_carry_the_row_id_array_into_a_card() -> None:
    per_seed = [
        {
            "receiver": {m: dict.fromkeys(trainmod.RECEIVER_METRIC_KEYS, 0.5) for m in trainmod.MODEL_KEYS},
            "shot": {m: {"auc": 0.5} for m in trainmod.MODEL_KEYS},
            "receiver_row_ids": np.arange(2464),
        }
    ]
    summary = trainmod.per_seed_summary(per_seed)
    assert summary == [
        {
            "receiver": {m: dict.fromkeys(trainmod.RECEIVER_METRIC_KEYS, 0.5) for m in trainmod.MODEL_KEYS},
            "shot": dict.fromkeys(trainmod.MODEL_KEYS, 0.5),
        }
    ]


# --------------------------------------------------------------------------- #
# reporting the spread
# --------------------------------------------------------------------------- #
def _report_with_spread(values: dict[str, dict[str, list[float]]]) -> dict:
    """A minimal report carrying ``shot.auc`` repeats for each named model."""
    per_seed = [
        {
            "receiver": {
                m: dict.fromkeys(trainmod.RECEIVER_METRIC_KEYS, 0.5) for m in trainmod.MODEL_KEYS
            },
            "shot": {m: {"auc": values[m]["auc"][i]} for m in trainmod.MODEL_KEYS},
        }
        for i in range(len(values["gnn"]["auc"]))
    ]
    means, spread = trainmod.summarise_seeds(per_seed)
    return {"receiver": means["receiver"], "shot": means["shot"], "spread": spread}


def test_cell_shows_a_spread_only_when_there_was_more_than_one_repeat() -> None:
    one = _report_with_spread({m: {"auc": [0.5]} for m in trainmod.MODEL_KEYS})
    assert trainmod._cell(one, "shot", "gnn", "auc") == "0.500"
    many = _report_with_spread(
        {m: {"auc": [0.50, 0.54, 0.52]} for m in trainmod.MODEL_KEYS}
    )
    assert trainmod._cell(many, "shot", "gnn", "auc").startswith("0.520 ±")


def test_cell_falls_back_to_the_bare_mean_when_no_spread_was_recorded() -> None:
    report = {"shot": {"gnn": {"auc": 0.61}}}
    assert trainmod._cell(report, "shot", "gnn", "auc") == "0.610"


def test_paired_gap_cancels_the_split_noise_the_columns_share() -> None:
    """Two models that move together across repeats have a tight *difference*.

    Each repeat re-draws the split, so both columns swing; differencing them
    inside a repeat removes that shared swing. The point of the paired gap is
    that it is far tighter than either column, which is exactly what makes a
    0.02 difference readable at all.
    """
    report = _report_with_spread(
        {
            "gnn": {"auc": [0.50, 0.60, 0.55]},
            "logreg": {"auc": [0.52, 0.62, 0.57]},
            "heuristic": {"auc": [0.40, 0.50, 0.45]},
        }
    )
    column_sd = trainmod._agg(report, "shot", "gnn", "auc")["sd"]
    gap = trainmod._paired_gap(report, "shot", "auc", "logreg", "gnn")
    assert gap["mean"] == pytest.approx(0.02)
    assert gap["sd"] == pytest.approx(0.0, abs=1e-9)
    assert column_sd > 0.04


def test_a_gap_inside_its_own_spread_is_labelled_as_noise() -> None:
    noisy = {"mean": 0.01, "sd": 0.02, "n": 5, "ratio": 0.5}
    clear = {"mean": 0.04, "sd": 0.005, "n": 5, "ratio": 8.0}
    assert "inside the noise floor" in trainmod._describe_gap(noisy, "the GNN", "the baseline", "AUC")
    assert "clear of the noise floor" in trainmod._describe_gap(clear, "the GNN", "the baseline", "AUC")
    # a single repeat has no spread to compare against, so it claims nothing
    assert trainmod._describe_gap({"mean": 0.04, "sd": 0.0, "n": 1, "ratio": 0.0}, "a", "b", "AUC") == ""


def test_the_table_says_plainly_when_a_baseline_beats_the_gnn() -> None:
    """The one thing this table must never do is flatter the model it ships.

    A report where the logistic regression wins the shot head has to print that
    in the reading section, in those words.
    """
    report = _report_with_spread(
        {
            "gnn": {"auc": [0.56, 0.58]},
            "logreg": {"auc": [0.59, 0.60]},
            "heuristic": {"auc": [0.48, 0.49]},
        }
    )
    report.update(
        {
            "date": "2026-01-01",
            "data": {"path": "d.jsonl", "n_corners": 10, "n_matches": 3, "competitions": []},
            "splits": {"n_folds": 2, "rows_per_fold": [5, 5], "matches_per_fold": [2, 1]},
            "params": {"receiver": 1, "shot": 1},
            "training": {"seed": 0, "torch_threads": 1},
            "seeds": [0, 1000],
        }
    )
    for model in trainmod.MODEL_KEYS:
        report["receiver"][model].update({"n_rows": 10.0, "n_attacker_first": 4.0})
        report["shot"][model].update({"n_rows": 10.0, "base_rate": 0.3})
    report["receiver"]["chance"] = {"top1_attacker": 0.2, "top3_attacker": 0.5}
    report["receiver"]["always_defence_accuracy"] = 0.6

    text = trainmod.format_metrics_markdown(report)
    assert "The logistic regression baseline **beats** the GNN on shot AUC" in text
    assert "0.595 against 0.570" in text
    assert "mean over 2 repeats" in text


def test_a_lead_too_small_to_print_is_called_level_rather_than_a_win() -> None:
    """Two models can differ in a decimal the table does not show.

    "The GNN wins (0.432), ahead of the baseline at 0.432" reads as a typo and
    invites the reader to believe in a lead the numbers beside it do not support.
    """
    report = _report_with_spread(
        {
            "gnn": {"auc": [0.5802, 0.5806]},
            "logreg": {"auc": [0.5799, 0.5801]},
            "heuristic": {"auc": [0.48, 0.49]},
        }
    )
    report.update(
        {
            "date": "2026-01-01",
            "data": {"path": "d.jsonl", "n_corners": 10, "n_matches": 3, "competitions": []},
            "splits": {"n_folds": 2, "rows_per_fold": [5, 5], "matches_per_fold": [2, 1]},
            "params": {"receiver": 1, "shot": 1},
            "training": {"seed": 0, "torch_threads": 1},
            "seeds": [0, 1000],
        }
    )
    for model in trainmod.MODEL_KEYS:
        report["receiver"][model].update({"n_rows": 10.0, "n_attacker_first": 4.0})
        report["shot"][model].update({"n_rows": 10.0, "base_rate": 0.3})
    report["receiver"]["chance"] = {"top1_attacker": 0.2, "top3_attacker": 0.5}
    report["receiver"]["always_defence_accuracy"] = 0.6

    text = trainmod.format_metrics_markdown(report)
    assert "GNN and logistic regression are level on shot AUC at 0.580" in text
    assert "wins on shot AUC" not in text
    assert "**beats** the GNN on shot AUC" not in text
    # the paired gap is still reported, and is still the honest reading
    assert "Seed by seed, the GNN minus the logistic regression on shot AUC" in text


# --------------------------------------------------------------------------- #
# cross-validation results and the out-of-fold cache
# --------------------------------------------------------------------------- #
def fake_cv(rows, *, seed: int = 0, folds: int = 2, shift: float = 0.0) -> trainmod.CVResult:
    """A ``CVResult`` with deterministic predictions and no training at all."""
    assignment = purged_fold_assignment([r.group for r in rows], folds, seed=seed)
    rng = np.random.default_rng(seed)
    cand: dict[str, list[np.ndarray]] = {}
    p_def: dict[str, np.ndarray] = {}
    shot: dict[str, np.ndarray] = {}
    for offset, key in enumerate(trainmod.MODEL_KEYS):
        rows_probs = []
        for r in rows:
            n = max(int(r.cand_nodes.size), 0)
            w = rng.random(n) if n else np.zeros(0)
            total = w.sum() + 1.0
            rows_probs.append(w / total)
        cand[key] = rows_probs
        p_def[key] = np.clip(rng.random(len(rows)) * 0.5 + 0.25 + shift, 0.0, 1.0)
        shot[key] = np.clip(rng.random(len(rows)) * 0.4 + 0.1 * offset, 0.0, 1.0)
    return trainmod.CVResult(
        seed=seed,
        assignment=assignment,
        splits=trainmod._split_summary(rows, assignment),
        cand_probs=cand,
        p_defence=p_def,
        shot=shot,
    )


def test_cvresult_check_catches_an_unscored_row(tmp_path: Path) -> None:
    """A hole in the out-of-fold arrays must raise, never reach a metric."""
    rows = load_rows(make_dataset(tmp_path / "d.jsonl", matches=4, per_match=2))
    cv = fake_cv(rows)
    cv.check(len(rows))
    cv.shot["gnn"][3] = np.nan
    with pytest.raises(RuntimeError, match="incomplete"):
        cv.check(len(rows))
    cv = fake_cv(rows)
    cv.cand_probs["logreg"] = cv.cand_probs["logreg"][:-1]
    with pytest.raises(RuntimeError, match="candidate distributions"):
        cv.check(len(rows))


def test_save_and_load_oof_round_trip_exactly(tmp_path: Path) -> None:
    rows = load_rows(make_dataset(tmp_path / "d.jsonl", matches=4, per_match=2))
    cv = fake_cv(rows, seed=3)
    path = trainmod.save_oof(rows, cv, tmp_path / "oof.npz", signature="sig")
    back = trainmod.load_oof(path, signature="sig")
    assert back.seed == cv.seed
    assert back.splits == cv.splits
    assert np.array_equal(back.assignment, cv.assignment)
    for key in trainmod.MODEL_KEYS:
        assert np.array_equal(back.shot[key], cv.shot[key])
        assert np.array_equal(back.p_defence[key], cv.p_defence[key])
        assert len(back.cand_probs[key]) == len(rows)
        for a, b in zip(back.cand_probs[key], cv.cand_probs[key], strict=True):
            assert np.array_equal(a, b)


def test_load_oof_rejects_a_cache_from_another_configuration(tmp_path: Path) -> None:
    """A stale cache would put numbers from an abandoned run into the docs."""
    rows = load_rows(make_dataset(tmp_path / "d.jsonl", matches=4, per_match=2))
    path = trainmod.save_oof(rows, fake_cv(rows), tmp_path / "oof.npz", signature="a")
    with pytest.raises(ValueError, match="different configuration"):
        trainmod.load_oof(path, signature="b")
    assert trainmod.load_oof(path).seed == 0  # no signature asked for, no check made


def test_oof_signature_moves_with_anything_that_changes_the_predictions(
    tmp_path: Path,
) -> None:
    rows = load_rows(make_dataset(tmp_path / "d.jsonl", matches=4, per_match=2))
    hp = dict(trainmod.DEFAULTS)
    base = trainmod.oof_signature(rows, "d.jsonl", 5, hp)
    assert base == trainmod.oof_signature(rows, "d.jsonl", 5, dict(hp))
    assert base != trainmod.oof_signature(rows, "other.jsonl", 5, hp)
    assert base != trainmod.oof_signature(rows, "d.jsonl", 3, hp)
    assert base != trainmod.oof_signature(rows[:-1], "d.jsonl", 5, hp)
    for key in ("epochs", "hidden", "lr", "batch_size", "baseline_iters"):
        bumped = dict(hp)
        bumped[key] = hp[key] * 2
        assert base != trainmod.oof_signature(rows, "d.jsonl", 5, bumped), key


def test_oof_signature_moves_when_the_labels_change_under_a_fixed_row_count(
    tmp_path: Path,
) -> None:
    """The label windows grew guards without the row count or feature width moving.

    A shape-only signature calls that cache reusable and reports numbers measured
    against labels nobody is using any more, which is the worst kind of stale:
    silent and plausible.
    """
    rows = load_rows(make_dataset(tmp_path / "d.jsonl", matches=4, per_match=2))
    hp = dict(trainmod.DEFAULTS)
    base = trainmod.oof_signature(rows, "d.jsonl", 5, hp)

    relabelled = [*rows]
    relabelled[0] = replace(rows[0], y_shot=1.0 - rows[0].y_shot)
    assert len(relabelled) == len(rows)
    assert trainmod.oof_signature(relabelled, "d.jsonl", 5, hp) != base

    moved = [*rows]
    moved[0] = replace(rows[0], target_pos=-1, receiver_valid=False)
    assert trainmod.oof_signature(moved, "d.jsonl", 5, hp) != base


def test_oof_signature_moves_when_the_features_change_under_a_fixed_column_count(
    tmp_path: Path,
) -> None:
    """The mirror change altered every signed ``y`` feature and no column count."""
    rows = load_rows(make_dataset(tmp_path / "d.jsonl", matches=4, per_match=2))
    hp = dict(trainmod.DEFAULTS)
    base = trainmod.oof_signature(rows, "d.jsonl", 5, hp)

    refeaturised = [*rows]
    graph = rows[0].graph.clone()
    graph.x = graph.x + 0.5
    refeaturised[0] = replace(rows[0], graph=graph)
    assert refeaturised[0].graph.x.shape == rows[0].graph.x.shape
    assert trainmod.oof_signature(refeaturised, "d.jsonl", 5, hp) != base

    edged = [*rows]
    graph = rows[0].graph.clone()
    graph.edge_attr = graph.edge_attr + 0.5
    edged[0] = replace(rows[0], graph=graph)
    assert trainmod.oof_signature(edged, "d.jsonl", 5, hp) != base


def test_cached_cross_validate_trains_once_and_then_reuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = load_rows(make_dataset(tmp_path / "d.jsonl", matches=4, per_match=2))
    hp = dict(trainmod.DEFAULTS)
    calls = {"n": 0}

    def counting(rows_, **kwargs):
        calls["n"] += 1
        return fake_cv(rows_, seed=kwargs["seed"], folds=kwargs["folds"])

    monkeypatch.setattr(trainmod, "cross_validate", counting)
    kwargs = dict(dataset_path="d.jsonl", folds=2, seed=7, hp=hp, cache_dir=tmp_path / "c")
    first = trainmod.cached_cross_validate(rows, **kwargs)
    second = trainmod.cached_cross_validate(rows, **kwargs)
    assert calls["n"] == 1
    assert np.array_equal(first.shot["gnn"], second.shot["gnn"])
    # with no cache directory nothing is stored and every call retrains
    trainmod.cached_cross_validate(rows, dataset_path="d.jsonl", folds=2, seed=7, hp=hp)
    assert calls["n"] == 2


def test_cross_validate_scores_every_row_exactly_once(tmp_path: Path) -> None:
    """The point of the protocol: no row is unscored and none is scored twice."""
    rows = load_rows(make_dataset(tmp_path / "d.jsonl", matches=6, per_match=2))
    hp = {**trainmod.DEFAULTS, "epochs": 1, "hidden": 8, "layers": 2, "batch_size": 8,
          "baseline_iters": 5}
    cv = trainmod.cross_validate(rows, folds=3, seed=0, hp=hp)
    cv.check(len(rows))
    assert sorted(set(cv.assignment.tolist())) == [0, 1, 2]
    for key in trainmod.MODEL_KEYS:
        assert np.isfinite(cv.shot[key]).all()
        assert np.isfinite(cv.p_defence[key]).all()
        for row, probs in zip(rows, cv.cand_probs[key], strict=True):
            assert probs.shape == (row.cand_nodes.size,)


def test_cross_validate_keeps_predictions_for_rows_the_receiver_table_drops(
    tmp_path: Path,
) -> None:
    """An excluded row is still scored, which is what makes the label audit possible.

    A corner whose receiver was off camera has no usable receiver label, but the
    fold model that never saw its match can still rank its candidates. Keeping
    that prediction is what lets a different threshold be evaluated later without
    retraining.
    """
    frames = [
        make_frame(f"c{i}", match_id=1000 + i // 2, seed=i, receiver_slot=i % 4)
        for i in range(12)
    ]
    for frame in frames[:3]:
        frame.receiver_idx = None
        frame.receiver_match_dist = 11.0
    path = tmp_path / "d.jsonl"
    write_corners(frames, path)
    rows = load_rows(path)
    assert [r.receiver_valid for r in rows[:3]] == [False, False, False]

    hp = {**trainmod.DEFAULTS, "epochs": 1, "hidden": 8, "layers": 2, "batch_size": 8,
          "baseline_iters": 5}
    cv = trainmod.cross_validate(rows, folds=3, seed=0, hp=hp)
    for i in range(3):
        assert cv.cand_probs["gnn"][i].size == rows[i].cand_nodes.size
        assert math.isfinite(float(cv.p_defence["gnn"][i]))


def test_metrics_from_oof_keep_restricts_the_receiver_table_only(tmp_path: Path) -> None:
    """The shot label has no match threshold, so ``keep`` must not touch it."""
    rows = load_rows(make_dataset(tmp_path / "d.jsonl", matches=6, per_match=2))
    cv = fake_cv(rows, folds=3)
    everything = trainmod.metrics_from_oof(rows, cv)
    keep = np.zeros(len(rows), dtype=bool)
    keep[: len(rows) // 2] = True
    restricted = trainmod.metrics_from_oof(rows, cv, keep=keep)
    assert restricted["receiver"]["gnn"]["n_rows"] < everything["receiver"]["gnn"]["n_rows"]
    assert restricted["shot"]["gnn"] == everything["shot"]["gnn"]


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def test_binary_targets_are_the_two_that_are_really_binary(tmp_path: Path) -> None:
    """Shot is defined everywhere; defence-wins only where the receiver label is usable."""
    frames = [
        make_frame(f"c{i}", match_id=1000 + i // 2, seed=i, shot=(i % 2 == 0))
        for i in range(8)
    ]
    frames[0].receiver_idx = None
    frames[0].receiver_match_dist = 11.0
    path = tmp_path / "d.jsonl"
    write_corners(frames, path)
    rows = load_rows(path)

    targets = trainmod._binary_targets(rows)
    y_shot, shot_mask = targets["shot"]
    y_def, def_mask = targets["defence"]
    assert shot_mask.all()
    assert y_shot.tolist() == [float(r.y_shot) for r in rows]
    assert def_mask.tolist() == [r.receiver_valid for r in rows]
    assert not def_mask[0]


def test_calibration_from_oof_covers_both_heads_and_every_model(tmp_path: Path) -> None:
    rows = load_rows(make_dataset(tmp_path / "d.jsonl", matches=6, per_match=3))
    report = trainmod.calibration_from_oof(rows, fake_cv(rows, folds=3), bins=4)
    assert set(report) == {"shot", "defence"}
    for head in ("shot", "defence"):
        assert set(report[head]) == set(trainmod.MODEL_KEYS)
        entry = report[head]["gnn"]
        assert entry["raw"]["n"] > 0
        for kind in trainmod.CALIBRATORS:
            assert entry[kind]["params"]["kind"] == kind
            assert math.isfinite(entry[kind]["in_sample"]["ece_uniform"])


def test_pick_calibrator_returns_identity_when_nothing_beats_the_raw_head() -> None:
    """Shipping a calibrator that makes calibration worse is worse than shipping none."""
    nested = {
        "shot": {
            "raw_ece": 0.02,
            "platt": {"ece": 0.05},
            "isotonic": {"ece": 0.04},
        }
    }
    kind, raw, chosen = trainmod._pick_calibrator(nested, "shot")
    assert kind == "identity"
    assert raw == chosen == 0.02


def test_pick_calibrator_prefers_platt_on_a_near_tie() -> None:
    """Isotonic has to earn its extra freedom on a few thousand rows.

    A staircase that beats a two-parameter logistic by a hair is chasing noise,
    so isotonic only wins when it is ahead by more than a tenth of the raw error.
    """
    near = {"shot": {"raw_ece": 0.10, "platt": {"ece": 0.061}, "isotonic": {"ece": 0.060}}}
    assert trainmod._pick_calibrator(near, "shot")[0] == "platt"
    clear = {"shot": {"raw_ece": 0.10, "platt": {"ece": 0.090}, "isotonic": {"ece": 0.050}}}
    assert trainmod._pick_calibrator(clear, "shot")[0] == "isotonic"


def test_fit_shipped_calibrators_records_the_choice_and_the_numbers(
    tmp_path: Path,
) -> None:
    rows = load_rows(make_dataset(tmp_path / "d.jsonl", matches=6, per_match=3))
    cv = fake_cv(rows, folds=3)
    nested = {
        head: {"raw_ece": 0.20, "platt": {"ece": 0.10}, "isotonic": {"ece": 0.19}}
        for head in ("shot", "defence")
    }
    shipped = trainmod.fit_shipped_calibrators(rows, cv, nested)
    for head in ("shot", "defence"):
        assert shipped[head]["kind"] == "platt"
        assert shipped[head]["ece_raw"] == 0.20
        assert shipped[head]["ece_calibrated_nested"] == 0.10
        assert shipped[head]["n_rows"] > 0
        # the map has to survive the trip through a model card
        rebuilt = calibrator_from_dict(shipped[head]["calibrator"])
        assert rebuilt.predict([0.3]).shape == (1,)


def test_summarise_calibration_quotes_the_nested_number_not_the_in_sample_one() -> None:
    """An ECE measured where the calibrator was fitted is a bound, not a result."""
    in_sample = {
        "shot": {
            "gnn": {
                "raw": {
                    "n": 100,
                    "base_rate": 0.3,
                    "mean_prediction": 0.25,
                    "min_prediction": 0.1,
                    "max_prediction": 0.4,
                    "brier": 0.2,
                    "curve": [],
                },
                "platt": {"in_sample": {"ece_uniform": 0.01}},
            }
        }
    }
    nested = {
        "shot": {
            "raw_ece": 0.09,
            "raw_ece_quantile": 0.11,
            "raw_auc": 0.58,
            "platt": {"ece": 0.05, "ece_quantile": 0.06, "auc": 0.58},
        }
    }
    shipped = {"shot": {"kind": "platt"}}
    out = trainmod.summarise_calibration(in_sample, nested, shipped)["shot"]
    assert out["ece_raw"] == 0.09
    assert out["ece_calibrated"] == 0.05  # nested, not the flattering 0.01
    assert out["ece_calibrated_quantile"] == 0.06
    assert out["chosen"] == "platt"


def test_summarise_calibration_leaves_an_identity_head_at_its_raw_error() -> None:
    in_sample = {
        "shot": {
            "gnn": {
                "raw": {
                    "n": 10, "base_rate": 0.3, "mean_prediction": 0.3,
                    "min_prediction": 0.1, "max_prediction": 0.5, "brier": 0.2, "curve": [],
                }
            }
        }
    }
    nested = {"shot": {"raw_ece": 0.03, "raw_ece_quantile": 0.04, "raw_auc": 0.55}}
    out = trainmod.summarise_calibration(in_sample, nested, {"shot": {"kind": "identity"}})
    assert out["shot"]["ece_calibrated"] == 0.03
    assert out["shot"]["ece_calibrated_quantile"] == 0.04


def test_calibration_section_never_claims_an_identity_head_ships_a_map() -> None:
    """"identity" means no calibrator ships, and the prose has to say that."""
    report = {
        "calibration": {
            "measured": {
                "shot": {
                    "n": 100, "base_rate": 0.3, "mean_prediction": 0.28,
                    "min_prediction": 0.1, "max_prediction": 0.5,
                    "ece_raw": 0.03, "ece_calibrated": 0.03,
                    "ece_raw_quantile": 0.04, "ece_calibrated_quantile": 0.04,
                    "brier_raw": 0.2, "chosen": "identity", "curve": [],
                },
                "defence": {
                    "n": 90, "base_rate": 0.6, "mean_prediction": 0.5,
                    "min_prediction": 0.3, "max_prediction": 0.7,
                    "ece_raw": 0.12, "ece_calibrated": 0.05,
                    "ece_raw_quantile": 0.14, "ece_calibrated_quantile": 0.06,
                    "brier_raw": 0.24, "chosen": "platt", "curve": [],
                },
            },
            "shipped": {"shot": {"kind": "identity"}, "defence": {"kind": "platt"}},
        }
    }
    text = "\n".join(trainmod._calibration_section(report))
    assert "stores a `platt` map for its defence-wins probability" in text
    assert "`identity` map" not in text
    assert "ships **no** calibrator" in text
    assert "not** applied automatically" in text


def test_calibration_section_is_absent_when_calibration_was_not_measured() -> None:
    assert trainmod._calibration_section({}) == []


@pytest.mark.slow
def test_nested_calibration_never_fits_a_calibrator_on_the_rows_it_scores(
    tmp_path: Path,
) -> None:
    """The whole point of the nested protocol, asserted on the fold bookkeeping.

    Each outer fold's calibrator is fitted on a *partner* fold, scored by an
    extra model that saw neither. If the fit ever touched the fold it is applied
    to, the reported calibration would be a lower bound dressed up as a result.
    """
    rows = load_rows(make_dataset(tmp_path / "d.jsonl", matches=8, per_match=2))
    hp = {**trainmod.DEFAULTS, "epochs": 1, "hidden": 8, "layers": 2, "batch_size": 8,
          "baseline_iters": 5}
    cv = trainmod.cross_validate(rows, folds=4, seed=0, hp=hp)

    out = trainmod.nested_calibration(rows, cv, hp=hp)
    for head in ("shot", "defence"):
        assert math.isfinite(out[head]["raw_ece"])
        assert math.isfinite(out[head]["raw_auc"])
        for kind in trainmod.CALIBRATORS:
            assert math.isfinite(out[head][kind]["ece"])
            # The pooled column is five different per-fold maps stitched
            # together, so it is *not* the raw AUC and must not be named as if
            # it were. A single global calibrator is the thing that preserves a
            # ranking, and that is asserted in tests/test_calibration.py.
            assert "auc" not in out[head][kind]
            assert math.isfinite(out[head][kind]["auc_pooled_per_fold_maps"])

    # Every calibrator is fitted on a fold other than the one it is applied to,
    # and by a model that saw neither. Assert the pairing itself.
    n_folds = int(cv.assignment.max()) + 1
    for fold in range(n_folds):
        partner = fold + 1 if fold % 2 == 0 else fold - 1
        partner = partner if partner < n_folds else 0
        assert partner != fold, "a fold used as its own calibration set"


# --------------------------------------------------------------------------- #
# receiver label threshold
# --------------------------------------------------------------------------- #
def make_threshold_dataset(path: Path) -> Path:
    """Twelve corners whose receiver matches sit at known distances.

    Four defence-wins rows, and eight attacker-first rows matched at 2, 3, 5, 7,
    9, 11, 13 m and one with no match at all, so every sweep point has something
    to move.
    """
    distances = [2.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, None]
    frames: list[CornerFrame] = []
    for i, d in enumerate(distances):
        frame = make_frame(f"a{i}", match_id=2000 + i // 2, seed=i, receiver_slot=i % 4)
        frame.receiver_match_dist = d
        # _io drops the index past its own threshold; mirror that here
        if d is None or d > 8.0:
            frame.receiver_idx = None
        frames.append(frame)
    for i in range(4):
        frames.append(
            make_frame(
                f"d{i}", match_id=2100 + i // 2, seed=100 + i,
                first_touch="defender", receiver_slot=None,
            )
        )
    write_corners(frames, path)
    return path


def test_threshold_sweep_counts_the_rows_that_survive_each_threshold(
    tmp_path: Path,
) -> None:
    rows = load_rows(make_threshold_dataset(tmp_path / "d.jsonl"))
    sweep = trainmod.label_threshold_sweep(rows, [fake_cv(rows, folds=3)])
    counts = {e["threshold"]: e["n_attacker_first"] for e in sweep["entries"]}
    # matches at 2, 3, 5, 7, 9, 11, 13; one row has no match at any threshold
    assert counts[4.0] == 2
    assert counts[6.0] == 3
    assert counts[8.0] == 4
    assert counts[10.0] == 5
    assert counts[12.0] == 6
    assert counts[None] == 7
    for entry in sweep["entries"]:
        assert entry["n_defence"] == 4
        assert entry["n_rows"] == entry["n_attacker_first"] + 4


def test_threshold_sweep_is_exact_up_to_the_shipped_threshold_and_bracketed_above(
    tmp_path: Path,
) -> None:
    """Above 8 m the dataset knows a match existed but not which player it was.

    ``_io`` keeps ``receiver_match_dist`` and drops ``receiver_idx``, so those
    rows can be counted and bounded but not scored. Presenting a bound as a point
    estimate would be the dishonest option.
    """
    rows = load_rows(make_threshold_dataset(tmp_path / "d.jsonl"))
    sweep = trainmod.label_threshold_sweep(rows, [fake_cv(rows, folds=3)])
    by_threshold = {e["threshold"]: e for e in sweep["entries"]}
    for t in (4.0, 6.0, 8.0):
        assert by_threshold[t]["scoreable"] is True
        assert "metrics" in by_threshold[t]
        assert "bounds" not in by_threshold[t]
    for t in (10.0, 12.0, None):
        assert by_threshold[t]["scoreable"] is False
        assert "metrics" not in by_threshold[t]
        low, high = by_threshold[t]["bounds"]["gnn"]["top1_attacker"]
        assert 0.0 <= low <= high <= 1.0

    # the bracket has to contain the shipped number, and widen as rows are added
    base = by_threshold[8.0]["metrics"]["gnn"]["top1_attacker"]["mean"]
    for t in (10.0, 12.0, None):
        low, high = by_threshold[t]["bounds"]["gnn"]["top1_attacker"]
        assert low <= base <= high or math.isclose(low, base) or math.isclose(high, base)
    w10 = np.subtract(*reversed(by_threshold[10.0]["bounds"]["gnn"]["top1_attacker"]))
    w_all = np.subtract(*reversed(by_threshold[None]["bounds"]["gnn"]["top1_attacker"]))
    assert w_all > w10


def test_threshold_sweep_scores_every_model_on_the_identical_rows(
    tmp_path: Path,
) -> None:
    rows = load_rows(make_threshold_dataset(tmp_path / "d.jsonl"))
    sweep = trainmod.label_threshold_sweep(rows, [fake_cv(rows, folds=3)])
    for entry in sweep["entries"]:
        if not entry["scoreable"]:
            continue
        counts = {entry["metrics"][m]["top1_attacker"]["n"] for m in trainmod.MODEL_KEYS}
        assert counts == {1}


def test_threshold_sweep_averages_over_repeats_like_the_main_table(
    tmp_path: Path,
) -> None:
    rows = load_rows(make_threshold_dataset(tmp_path / "d.jsonl"))
    cvs = [fake_cv(rows, seed=s, folds=3) for s in (0, 1000, 2000)]
    sweep = trainmod.label_threshold_sweep(rows, cvs)
    assert sweep["n_seeds"] == 3
    entry = next(e for e in sweep["entries"] if e["threshold"] == 8.0)
    assert entry["metrics"]["gnn"]["top1_attacker"]["n"] == 3
    assert "over the same 3 repeats" in " ".join(sweep["notes"])


def test_threshold_sweep_prose_states_that_loosening_needs_a_refetch(
    tmp_path: Path,
) -> None:
    rows = load_rows(make_threshold_dataset(tmp_path / "d.jsonl"))
    sweep = trainmod.label_threshold_sweep(rows, [fake_cv(rows, folds=3)])
    notes = " ".join(sweep["notes"])
    assert "drops `receiver_idx`" in notes
    assert "re-fetch" in notes
    assert "RECEIVER_MATCH_MAX_M = 8.0" in sweep["summary"]
    assert "*(shipped)*" in sweep["table"]


def test_a_bracket_that_adds_no_rows_prints_as_a_single_number() -> None:
    assert trainmod._range_cell(0.42, 0.42) == "0.420"
    assert trainmod._range_cell(0.40, 0.50) == "0.400-0.500"
    assert trainmod._range_cell(float("nan"), 0.5) == "n/a"


def test_threshold_section_is_absent_when_the_sweep_was_not_run() -> None:
    assert trainmod._threshold_section({}) == []


# --------------------------------------------------------------------------- #
# what actually ships
# --------------------------------------------------------------------------- #
def test_a_calibrated_checkpoint_carries_a_working_calibrator(tmp_path: Path) -> None:
    """A card that names a calibrator must let a reader rebuild and apply it.

    Each checkpoint carries only its own head's calibrator: the shot head's shot
    probability, the receiver head's ``P(defence wins first contact)``.
    """
    path = make_dataset(tmp_path / "d.jsonl", matches=8, per_match=2)
    report = train_all(
        path,
        tmp_path / "run",
        folds=4,
        epochs=1,
        hidden=8,
        layers=2,
        batch_size=8,
        baseline_iters=5,
        seeds=1,
        calibrate=True,
        sweep_threshold=True,
        metrics_path=None,
        quiet=True,
    )
    assert set(report["calibration"]["shipped"]) == {"shot", "defence"}

    for name, head in (("shot", "shot"), ("receiver", "defence")):
        _, card = load_model(tmp_path / "run" / f"{name}.pt")
        assert set(card["calibration"]) == {head}, "a card carries only its own head"
        kind = card["calibration"][head]["kind"]
        rebuilt = calibrator_from_card(card, head)
        if kind == "identity":
            # "measured and left alone" must be distinguishable from "never measured"
            assert rebuilt is None or rebuilt.kind == "identity"
        else:
            assert rebuilt is not None
            out = rebuilt.predict([0.05, 0.5, 0.95])
            assert out.shape == (3,)
            assert np.all((out >= 0.0) & (out <= 1.0))
            # monotone, so it can never reverse a ranking
            assert np.all(np.diff(out) >= -1e-12)

    # the receiver card records the label-threshold audit, without the metrics blob
    _, receiver_card = load_model(tmp_path / "run" / "receiver.pt")
    assert receiver_card["label_threshold"]["shipped_threshold_m"] == 8.0
    assert all("metrics" not in e for e in receiver_card["label_threshold"]["entries"])
    _, shot_card = load_model(tmp_path / "run" / "shot.pt")
    assert "label_threshold" not in shot_card


def test_calibration_and_the_sweep_can_both_be_switched_off(tmp_path: Path) -> None:
    path = make_dataset(tmp_path / "d.jsonl", matches=4, per_match=2)
    report = train_all(
        path,
        tmp_path / "run",
        folds=2,
        epochs=1,
        hidden=8,
        layers=2,
        batch_size=8,
        baseline_iters=5,
        seeds=1,
        calibrate=False,
        sweep_threshold=False,
        metrics_path=tmp_path / "m.md",
        quiet=True,
    )
    assert "calibration" not in report
    assert "label_threshold" not in report
    text = (tmp_path / "m.md").read_text()
    assert "## calibration" not in text
    assert "## receiver label threshold" not in text
    _, card = load_model(tmp_path / "run" / "shot.pt")
    assert "calibration" not in card


def test_train_all_reuses_a_cache_instead_of_retraining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regenerating the report must not cost another five fits per head."""
    path = make_dataset(tmp_path / "d.jsonl", matches=4, per_match=2)
    cache = tmp_path / "cache"
    kwargs = dict(
        folds=2, epochs=1, hidden=8, layers=2, batch_size=8, baseline_iters=5,
        seeds=2, calibrate=False, sweep_threshold=False, metrics_path=None,
        quiet=True, cache_dir=cache,
    )
    first = train_all(path, tmp_path / "a", **kwargs)
    assert sorted(p.name for p in cache.glob("*.npz")) == ["oof-seed0.npz", "oof-seed1000.npz"]

    calls = {"n": 0}
    real = trainmod.cross_validate

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(trainmod, "cross_validate", counting)
    second = train_all(path, tmp_path / "b", **kwargs)
    assert calls["n"] == 0, "a warm cache still retrained"
    assert second["shot"]["gnn"]["auc"] == pytest.approx(first["shot"]["gnn"]["auc"])
    assert second["receiver"]["gnn"] == first["receiver"]["gnn"]


def test_parser_exposes_the_repeat_count_and_the_cache(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.seeds >= 1
    assert args.cache_dir is None
    assert args.no_calibrate is False

    assert args.label_baseline is None
    assert args.ablation is None

    args = parser.parse_args(
        ["--seeds", "3", "--cache-dir", "runs/oof", "--no-calibrate", "--no-threshold-sweep",
         "--label-baseline", "data/interim/before.jsonl", "--ablation", "runs/ablation.json"]
    )
    assert (args.seeds, args.cache_dir) == (3, "runs/oof")
    assert args.no_calibrate is True
    assert args.no_threshold_sweep is True
    assert args.label_baseline == "data/interim/before.jsonl"
    assert args.ablation == ["runs/ablation.json"]
