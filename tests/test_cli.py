"""Tests for the command line interface.

The checkpoints used here are randomly initialised models written through
:func:`setpiece.models.save_model`. Nothing in this file asserts anything about
*what* the numbers are — only that the plumbing, the formats and the exit codes
behave — because an untrained network's probabilities are meaningless.
"""

from __future__ import annotations

import csv
import io
import json
import socket
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from setpiece import cli
from setpiece._io import load_corner

CARD = {
    "data": {"competitions": ["UEFA Euro"], "n_corners": 0},
    "splits": {"note": "untrained fixture"},
    "metrics": {},
    "date": "2026-08-25",
    "features": [],
    "assume_static": True,
}


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory holding an untrained receiver.pt and shot.pt."""
    import torch

    from setpiece.models import ReceiverModel, ShotModel, save_model

    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("checkpoints")
    save_model(ReceiverModel(), d / "receiver.pt", CARD)
    save_model(ShotModel(), d / "shot.pt", CARD)
    return d


@pytest.fixture
def corner_path() -> Path:
    paths = sorted(cli.DEMO_DIR.glob("*.json"))
    assert paths, "the bundled demo corners are missing from setpiece/assets/demo"
    return paths[0]


def run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# --------------------------------------------------------------------------- #
# bundled demo corners
# --------------------------------------------------------------------------- #
def test_three_demo_corners_are_bundled() -> None:
    paths = sorted(cli.DEMO_DIR.glob("*.json"))
    assert len(paths) == 3


def test_demo_corners_are_real_euro_2024_frames() -> None:
    frames = cli.demo_frames()
    assert len(frames) == 3
    for f in frames:
        assert f.competition == "UEFA Euro"
        assert f.season == "2024"
        assert f.match_id is not None
        assert 12 <= len(f.players) <= 22
        assert f.taker_index is not None
        assert f.n_attackers() >= 6
        assert f.n_defenders() >= 6
        # labels survived the round trip, so the demo can print the real outcome
        assert f.first_touch in {"attacker", "defender", "unknown"}
        assert isinstance(f.shot_within_5, bool)


def test_demo_corners_cover_more_than_one_match() -> None:
    frames = cli.demo_frames()
    assert len({f.match_id for f in frames}) == 3
    assert {f.first_touch for f in frames} == {"attacker", "defender"}


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def test_parser_exposes_every_documented_subcommand() -> None:
    parser = cli.build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions, "no subparsers registered"
    assert set(actions[0].choices) == {"score", "suggest", "demo", "app", "fetch", "train"}


def test_score_defaults() -> None:
    args = cli.build_parser().parse_args(["score", "c.json"])
    assert args.format == "pretty"
    assert args.model is None
    assert args.corner == "c.json"


def test_suggest_defaults_match_the_contract() -> None:
    args = cli.build_parser().parse_args(["suggest", "c.json"])
    assert args.movers == 3
    assert args.max_move == 3.0
    assert args.seed == 0
    assert args.beam == 8
    assert args.attack is False


def test_suggest_flags_parse() -> None:
    args = cli.build_parser().parse_args(
        ["suggest", "c.json", "--movers", "2", "--max-move", "1.5", "--seed", "7", "--defend"]
    )
    assert (args.movers, args.max_move, args.seed, args.defend) == (2, 1.5, 7, True)


def test_defend_and_attack_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.build_parser().parse_args(["suggest", "c.json", "--defend", "--attack"])
    assert excinfo.value.code == cli.EXIT_USAGE


def test_bare_invocation_prints_help_and_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run([], capsys)
    assert code == cli.EXIT_USAGE
    assert "score" in out and "suggest" in out


def test_unknown_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["wibble"])
    assert excinfo.value.code == cli.EXIT_USAGE


def test_bad_format_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["score", "c.json", "--format", "yaml"])
    assert excinfo.value.code == cli.EXIT_USAGE


def test_version_flag() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0


# --------------------------------------------------------------------------- #
# score, in all three formats
# --------------------------------------------------------------------------- #
def test_score_pretty(
    corner_path: Path, checkpoints: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(["score", str(corner_path), "--model", str(checkpoints)], capsys)
    assert code == cli.EXIT_OK
    assert "P(first contact)" in out
    assert "defence" in out
    assert "P(shot within 5 events)" in out
    # the pretty table is hand rolled: no line runs away
    assert max(len(line) for line in out.splitlines()) < 120


def test_score_json_is_valid_and_normalised(
    corner_path: Path, checkpoints: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(
        ["score", str(corner_path), "--model", str(checkpoints), "--format", "json"], capsys
    )
    assert code == cli.EXIT_OK
    payload = json.loads(out)

    frame = load_corner(corner_path)
    assert payload["corner_id"] == frame.corner_id
    assert payload["n_players"] == len(frame.players)
    assert payload["assume_static"] is True
    assert 0.0 <= payload["p_shot"] <= 1.0

    total = sum(p["p_receive"] for p in payload["players"])
    assert total + payload["p_defence_first_contact"] == pytest.approx(1.0, abs=1e-6)

    # only candidates - attackers who are not the taker - carry any mass
    for p, player in zip(payload["players"], frame.players, strict=True):
        if not player.is_attacker or player.is_taker:
            assert p["p_receive"] == 0.0


def test_score_json_never_leaks_label_adjacent_fields(
    corner_path: Path, checkpoints: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, out, _ = run(
        ["score", str(corner_path), "--model", str(checkpoints), "--format", "json"], capsys
    )
    assert "end_xy" not in out
    assert "pass_outcome" not in out


def test_score_csv_parses_and_carries_every_player(
    corner_path: Path, checkpoints: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(
        ["score", str(corner_path), "--model", str(checkpoints), "--format", "csv"], capsys
    )
    assert code == cli.EXIT_OK
    rows = list(csv.DictReader(io.StringIO(out)))
    frame = load_corner(corner_path)
    assert len(rows) == len(frame.players) + 1  # one row per player, plus the defence class
    assert rows[-1]["team"] == "defence_first_contact"
    assert rows[-1]["idx"] == "-1"
    assert {r["corner_id"] for r in rows} == {frame.corner_id}
    assert float(rows[0]["p_shot"]) == float(rows[-1]["p_shot"])


def test_score_accepts_a_jsonl_dataset(
    checkpoints: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from setpiece._io import write_corners

    dataset = tmp_path / "corners.jsonl"
    write_corners(cli.demo_frames(), dataset)
    code, out, _ = run(
        ["score", str(dataset), "--model", str(checkpoints), "--format", "json"], capsys
    )
    assert code == cli.EXIT_OK
    payload = json.loads(out)
    assert isinstance(payload, list)
    assert len(payload) == 3


def test_score_is_deterministic(
    corner_path: Path, checkpoints: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["score", str(corner_path), "--model", str(checkpoints), "--format", "json"]
    _, first, _ = run(argv, capsys)
    _, second, _ = run(argv, capsys)
    assert first == second


# --------------------------------------------------------------------------- #
# suggest
# --------------------------------------------------------------------------- #
def test_suggest_pretty_ranks_by_delta(
    corner_path: Path, checkpoints: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(
        [
            "suggest",
            str(corner_path),
            "--model",
            str(checkpoints),
            "--movers",
            "1",
            "--top-k",
            "3",
        ],
        capsys,
    )
    assert code == cli.EXIT_OK
    assert "base P(shot)" in out
    assert "delta" in out


def test_suggest_json_honours_movers_max_move_and_seed(
    corner_path: Path, checkpoints: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "suggest",
        str(corner_path),
        "--model",
        str(checkpoints),
        "--format",
        "json",
        "--movers",
        "2",
        "--max-move",
        "2.0",
        "--seed",
        "3",
        "--top-k",
        "4",
    ]
    code, out, _ = run(argv, capsys)
    assert code == cli.EXIT_OK
    payload = json.loads(out)
    assert payload["objective"] == "minimise_shot"
    assert payload["params"]["seed"] == 3
    assert payload["suggestions"]

    frame = load_corner(corner_path)
    deltas = [s["delta"] for s in payload["suggestions"]]
    assert deltas == sorted(deltas)
    for s in payload["suggestions"]:
        assert 1 <= len(s["moves"]) <= 2
        assert len({m["player_idx"] for m in s["moves"]}) == len(s["moves"])
        for m in s["moves"]:
            assert m["length"] <= 2.0 + 1e-9
            # defenders only, and never the taker
            assert not frame.players[m["player_idx"]].is_attacker
            assert not frame.players[m["player_idx"]].is_taker
        assert s["new_prob"] - s["base_prob"] == pytest.approx(s["delta"], abs=1e-9)

    # the same seed reproduces the run exactly
    _, again, _ = run(argv, capsys)
    assert again == out


def test_suggest_attack_flips_the_objective(
    corner_path: Path, checkpoints: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(
        [
            "suggest",
            str(corner_path),
            "--model",
            str(checkpoints),
            "--format",
            "json",
            "--attack",
            "--movers",
            "1",
        ],
        capsys,
    )
    assert code == cli.EXIT_OK
    payload = json.loads(out)
    assert payload["objective"] == "maximise_shot"
    deltas = [s["delta"] for s in payload["suggestions"]]
    assert deltas == sorted(deltas, reverse=True)


def test_suggest_csv_one_row_per_move(
    corner_path: Path, checkpoints: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(
        [
            "suggest",
            str(corner_path),
            "--model",
            str(checkpoints),
            "--format",
            "csv",
            "--movers",
            "1",
            "--top-k",
            "2",
        ],
        capsys,
    )
    assert code == cli.EXIT_OK
    rows = list(csv.DictReader(io.StringIO(out)))
    assert len(rows) == 2
    assert {r["rank"] for r in rows} == {"1", "2"}
    for r in rows:
        assert float(r["new_prob"]) - float(r["base_prob"]) == pytest.approx(
            float(r["delta"]), abs=1e-6
        )


# --------------------------------------------------------------------------- #
# missing checkpoints
# --------------------------------------------------------------------------- #
def test_score_without_a_model_is_actionable(
    corner_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    code, _, err = run(["score", str(corner_path), "--model", str(empty)], capsys)
    assert code == cli.EXIT_NO_MODEL
    assert "setpiece train" in err


def test_missing_model_path_is_reported(
    corner_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = run(
        ["score", str(corner_path), "--model", str(tmp_path / "nope.pt")], capsys
    )
    assert code == cli.EXIT_NO_MODEL
    assert "does not exist" in err


def test_bundled_models_are_used_when_no_flag_is_given(
    corner_path: Path, checkpoints: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Once checkpoints land in setpiece/assets/, score works with no extra flags."""
    from setpiece import models

    monkeypatch.setattr(models, "ASSETS_DIR", checkpoints)
    code, out, _ = run(["score", str(corner_path), "--format", "json"], capsys)
    assert code == cli.EXIT_OK
    assert json.loads(out)["p_shot"] >= 0.0


def test_demo_without_a_model_exits_three(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    code, _, err = run(["demo", "--model", str(empty)], capsys)
    assert code == cli.EXIT_NO_MODEL
    assert "setpiece train" in err


def test_missing_corner_file_exits_one(
    checkpoints: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = run(
        ["score", str(tmp_path / "absent.json"), "--model", str(checkpoints)], capsys
    )
    assert code == cli.EXIT_ERROR
    assert "no such corner file" in err


# --------------------------------------------------------------------------- #
# demo: offline, fast, three corners
# --------------------------------------------------------------------------- #
@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make any network use fail loudly, at both our fetch layer and the socket."""
    from setpiece import _io

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the demo attempted a network fetch")

    monkeypatch.setattr(_io, "fetch_json", forbidden)
    monkeypatch.setattr(_io, "cached_json", forbidden)

    real_socket = socket.socket

    class BlockedSocket(real_socket):  # type: ignore[misc, valid-type]
        def connect(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("the demo attempted to open a socket")

        def connect_ex(self, *args: Any, **kwargs: Any) -> int:
            raise AssertionError("the demo attempted to open a socket")

    monkeypatch.setattr(socket, "socket", BlockedSocket)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    yield


def test_demo_runs_offline(
    checkpoints: Path, no_network: None, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run(["demo", "--model", str(checkpoints)], capsys)
    assert code == cli.EXIT_OK, err
    assert out.count("P(shot within 5 events)") == 3
    assert "no network access" in out


def test_demo_scores_three_corners_under_ten_seconds(
    checkpoints: Path, no_network: None, capsys: pytest.CaptureFixture[str]
) -> None:
    import time

    started = time.perf_counter()
    code, out, _ = run(["demo", "--model", str(checkpoints), "--format", "json"], capsys)
    elapsed = time.perf_counter() - started
    assert code == cli.EXIT_OK

    payload = json.loads(out)
    assert len(payload["corners"]) == 3
    for entry in payload["corners"]:
        assert entry["score"]["n_players"] >= 12
        assert entry["suggest"]["suggestions"]
    assert elapsed < 10.0, f"demo took {elapsed:.1f}s"

    # the scoring and searching itself is the part we control; importing torch
    # and reading the checkpoints is measured separately so a slow, loaded
    # machine cannot silently eat the whole budget
    compute = payload["seconds"] - payload["startup_seconds"]
    assert compute < 5.0, f"demo spent {compute:.1f}s scoring three corners"


def test_demo_csv_holds_both_tables(
    checkpoints: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(["demo", "--model", str(checkpoints), "--format", "csv"], capsys)
    assert code == cli.EXIT_OK
    blocks = out.strip().split("\n\n")
    assert len(blocks) == 2
    assert blocks[0].splitlines()[0].startswith("corner_id,idx,team")
    assert blocks[1].splitlines()[0].startswith("corner_id,rank,base_prob")


# --------------------------------------------------------------------------- #
# app and train, without launching either
# --------------------------------------------------------------------------- #
def test_app_launches_streamlit_as_a_subprocess(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import subprocess

    seen: dict[str, Any] = {}

    def fake_call(cmd: list[str], *args: Any, **kwargs: Any) -> int:
        seen["cmd"] = cmd
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)
    app_path = Path(cli.__file__).resolve().parent / "app.py"
    if not app_path.exists():
        pytest.skip("setpiece/app.py is not present in this checkout")

    code, _, _ = run(["app", "--port", "8899"], capsys)
    assert code == cli.EXIT_OK
    cmd = seen["cmd"]
    assert cmd[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert cmd[4].endswith("app.py")
    assert "8899" in cmd


def test_importing_the_cli_pulls_in_nothing_heavy() -> None:
    """streamlit and torch stay lazy, so ``setpiece --help`` is instant."""
    import subprocess

    probe = (
        "import setpiece.cli, sys; "
        "print([m for m in ('streamlit', 'torch', 'torch_geometric') if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_train_forwards_its_arguments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, Any] = {}
    stub = types.ModuleType("setpiece.train")

    def fake_main(argv: list[str]) -> int:
        seen["argv"] = list(argv)
        return 0

    stub.main = fake_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "setpiece.train", stub)

    code, _, _ = run(["train", "--epochs", "2", "--data", "x.jsonl"], capsys)
    assert code == cli.EXIT_OK
    assert seen["argv"] == ["--epochs", "2", "--data", "x.jsonl"]


def test_train_propagates_a_failing_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stub = types.ModuleType("setpiece.train")
    stub.main = lambda argv: 7  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "setpiece.train", stub)
    code, _, _ = run(["train"], capsys)
    assert code == 7


def test_train_reports_a_module_with_no_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stub = types.ModuleType("setpiece.train")
    monkeypatch.setitem(sys.modules, "setpiece.train", stub)
    code, _, err = run(["train"], capsys)
    assert code == cli.EXIT_ERROR
    assert "entry point" in err


def test_fetch_is_wired_to_build_dataset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from setpiece import fetch as fetch_mod

    seen: dict[str, Any] = {}

    def fake_build(out_path: Path, raw_dir: Path, **kwargs: Any) -> dict:
        seen["out"] = Path(out_path)
        seen["raw"] = Path(raw_dir)
        seen.update(kwargs)
        return {"matches": 2, "corners": 9, "failures": []}

    monkeypatch.setattr(fetch_mod, "build_dataset", fake_build)
    code, out, _ = run(
        [
            "fetch",
            "--competition",
            "UEFA Euro",
            "--max-matches",
            "2",
            "--out",
            str(tmp_path / "c.jsonl"),
            "--raw-dir",
            str(tmp_path / "raw"),
        ],
        capsys,
    )
    assert code == cli.EXIT_OK
    assert seen["only_competitions"] == ["UEFA Euro"]
    assert seen["max_matches"] == 2
    assert "9 corners" in out
