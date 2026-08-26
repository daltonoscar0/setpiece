"""Command line interface.

    setpiece score   <corner> [--format json|csv|pretty] [--model PATH]
    setpiece suggest <corner> [--defend] [--movers 3] [--max-move 3.0] [--seed 0]
    setpiece demo
    setpiece app
    setpiece fetch   [--competition NAME] [--max-matches N]
    setpiece train   [...]

Two things shape the code below.

**Nothing heavy is imported at module import time.** ``torch``,
``torch_geometric``, ``streamlit`` and :mod:`setpiece.train` are all pulled in
inside the subcommand that needs them, so ``setpiece --help`` stays instant and
a half-written optional module cannot break the rest of the tool.

**Checkpoints are optional at every entry point.** The bundled models live in
``setpiece/assets/`` and are picked up with no flags at all; when they are
absent, or when ``--model`` points somewhere empty, the command exits with
:data:`EXIT_NO_MODEL` and a message telling the user to run ``setpiece train``.
No command ever scores a corner with an untrained network behind the user's
back.

Exit codes:

===  ==========================================================
0    success
1    a runtime failure (bad corner file, fetch error, ...)
2    argparse usage error
3    a required model checkpoint is missing
4    an optional dependency is missing (``streamlit`` for ``app``)
===  ==========================================================
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import sys
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

from setpiece import __version__
from setpiece.types import CornerFrame

__all__ = [
    "DEMO_DIR",
    "EXIT_ERROR",
    "EXIT_NO_DEPENDENCY",
    "EXIT_NO_MODEL",
    "EXIT_OK",
    "EXIT_USAGE",
    "ModelUnavailableError",
    "build_parser",
    "demo_frames",
    "format_scores",
    "format_suggestions",
    "load_head",
    "main",
    "run_train",
    "score_frame",
    "suggest_frame",
]

log = logging.getLogger("setpiece")

EXIT_OK: int = 0
EXIT_ERROR: int = 1
EXIT_USAGE: int = 2
EXIT_NO_MODEL: int = 3
EXIT_NO_DEPENDENCY: int = 4

#: The three bundled Euro 2024 corners the ``demo`` command runs on.
DEMO_DIR: Path = Path(__file__).resolve().parent / "assets" / "demo"

#: Suggestion search budget for ``demo``, trimmed so three corners are scored,
#: searched and printed well inside the ten second promise.
DEMO_MOVERS: int = 2
DEMO_BEAM: int = 4
DEMO_MAX_EVALS: int = 64
DEMO_TOP_K: int = 3

_FORMATS = ("pretty", "json", "csv")


class ModelUnavailableError(RuntimeError):
    """Raised when a checkpoint a command needs cannot be found or loaded."""


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def _train_hint(kind: str, path: Path) -> str:
    return (
        f"no trained '{kind}' model available (looked for {path}).\n"
        f"Train one and install it as the bundled checkpoint:\n"
        f"    uv run setpiece train --install\n"
        f"(plain 'setpiece train' only writes to --out-dir, default runs/latest)\n"
        f"or point at an existing checkpoint:\n"
        f"    uv run setpiece <command> --model path/to/{kind}.pt\n"
        f"(--model also accepts a directory holding receiver.pt and shot.pt)."
    )


def load_head(kind: str, model_arg: str | Path | None = None) -> tuple[Any, dict]:
    """Load the ``'receiver'`` or ``'shot'`` checkpoint, and its model card.

    ``model_arg`` may be ``None`` (use the checkpoint bundled in
    ``setpiece/assets/``), a directory holding ``receiver.pt`` / ``shot.pt``, or
    a single checkpoint file. A single file is only used for the head whose
    class it actually contains; asking for the other head then falls back to the
    bundled asset, so ``--model my_shot.pt`` still scores a full corner once a
    receiver model is bundled.

    Raises:
        ModelUnavailableError: if nothing suitable exists, carrying a message that
            tells the user exactly what to run.
    """
    if kind not in ("receiver", "shot"):
        raise ValueError(f"unknown model head {kind!r}")

    from setpiece import models

    path: Path | None = None
    if model_arg is not None:
        given = Path(model_arg)
        if given.is_dir():
            path = given / f"{kind}.pt"
        elif given.exists():
            path = given
        else:
            raise ModelUnavailableError(f"--model path does not exist: {given}")

    if path is not None:
        try:
            model, card = models.load_model(path)
        except FileNotFoundError as exc:
            raise ModelUnavailableError(_train_hint(kind, path)) from exc
        except Exception as exc:
            raise ModelUnavailableError(f"could not load checkpoint {path}: {exc}") from exc
        expected = "ReceiverModel" if kind == "receiver" else "ShotModel"
        if type(model).__name__ == expected:
            return model, card
        # a single file of the other kind: fall through to the bundled asset
        if Path(model_arg).is_dir():
            raise ModelUnavailableError(
                f"{path} holds a {type(model).__name__}, not a {expected}"
            )

    try:
        return models.load_bundled(kind)
    except FileNotFoundError as exc:
        raise ModelUnavailableError(_train_hint(kind, models.ASSETS_DIR / f"{kind}.pt")) from exc
    except Exception as exc:
        raise ModelUnavailableError(f"could not load the bundled '{kind}' model: {exc}") from exc


# --------------------------------------------------------------------------- #
# corner loading
# --------------------------------------------------------------------------- #
def _tune_torch_threads() -> None:
    """Score single threaded unless the user asked otherwise.

    A corner graph has twenty-odd nodes; spreading that over the machine's cores
    costs more in synchronisation than it saves, and measurably doubles the time
    per evaluation, which the suggestion search pays hundreds of times over.
    """
    import os

    if any(os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")):
        return
    import torch

    torch.set_num_threads(1)


def _load_frames(path: str | Path) -> list[CornerFrame]:
    """Load one corner file, or every corner in a ``.jsonl`` dataset.

    Every parse failure is normalised to :class:`ValueError` so ``main`` reports
    it as ``error: ...`` and :data:`EXIT_ERROR`. A hand-written corner file can
    fail in ways the loader does not anticipate — a player dict missing ``y``, an
    unknown top-level key — and those surface as ``TypeError`` or ``KeyError``
    from the dataclass constructors, which would otherwise escape as a traceback.

    A corner with no players is rejected rather than scored. The models happily
    consume an empty graph and return a probability, and that number is a
    fabrication: it describes no one.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such corner file: {path}")
    from setpiece import _io

    try:
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            frames = _io.read_corners(path)
        else:
            frames = [_io.load_corner(path)]
    except (ValueError, OSError):
        raise
    except Exception as exc:
        raise ValueError(f"malformed corner file {path}: {exc}") from exc

    if not frames:
        raise ValueError(f"{path} contains no corners")
    for frame in frames:
        if not frame.players:
            raise ValueError(f"corner {frame.corner_id!r} in {path} has no players")
    return frames


def demo_frames() -> list[CornerFrame]:
    """The bundled Euro 2024 corners, in a stable order. No network, no dataset."""
    from setpiece import _io

    paths = sorted(DEMO_DIR.glob("*.json"))
    if not paths:
        raise FileNotFoundError(
            f"no bundled demo corners in {DEMO_DIR}; the package install looks incomplete"
        )
    return [_io.load_corner(p) for p in paths]


def _short_id(corner_id: str) -> str:
    return corner_id.split("-")[0][:8] if corner_id else "corner"


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score_frame(frame: CornerFrame, receiver_model: Any, shot_model: Any) -> dict:
    """Score one corner into a plain dict, ready for any of the output formats."""
    from setpiece.models import receiver_probabilities, shot_probability

    probs, p_defence = receiver_probabilities(receiver_model, frame)
    p_shot = shot_probability(shot_model, frame)

    players: list[dict] = []
    for i, p in enumerate(frame.players):
        players.append(
            {
                "idx": i,
                "team": "attack" if p.is_attacker else "defend",
                "x": round(float(p.x), 2),
                "y": round(float(p.y), 2),
                "is_taker": bool(p.is_taker),
                "is_keeper": bool(p.is_keeper),
                "p_receive": float(probs[i]),
            }
        )

    result: dict[str, Any] = {
        "corner_id": frame.corner_id,
        "competition": frame.competition,
        "season": frame.season,
        "match_id": frame.match_id,
        "minute": frame.minute,
        "n_players": len(frame.players),
        "n_attackers": frame.n_attackers(),
        "n_defenders": frame.n_defenders(),
        "players": players,
        "p_defence_first_contact": float(p_defence),
        "p_shot": float(p_shot),
        "assume_static": True,
    }
    if frame.first_touch is not None or frame.shot_within_5 is not None:
        result["actual"] = {
            "first_touch": frame.first_touch,
            "receiver_idx": frame.receiver_idx,
            "shot_within_5": frame.shot_within_5,
        }
    return result


# --------------------------------------------------------------------------- #
# pretty printing
# --------------------------------------------------------------------------- #
def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], right: Sequence[bool]) -> str:
    """A plain fixed-width table. Hand rolled so the tool has no extra dependency."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells: Sequence[str]) -> str:
        out = []
        for i, cell in enumerate(cells):
            out.append(cell.rjust(widths[i]) if right[i] else cell.ljust(widths[i]))
        return "  ".join(out).rstrip()

    parts = [line(headers), line(["-" * w for w in widths])]
    parts.extend(line(r) for r in rows)
    return "\n".join(parts)


def _bar(p: float, width: int = 16) -> str:
    filled = int(round(max(0.0, min(1.0, p)) * width))
    return "#" * filled + "." * (width - filled)


def _pretty_score(result: dict) -> str:
    head = f"corner {_short_id(result['corner_id'])}"
    where = _describe_result(result)
    if where:
        head = f"{head}  ({where})"

    rows: list[list[str]] = []
    ranked = sorted(
        result["players"], key=lambda p: (-p["p_receive"], p["idx"])
    )
    for p in ranked:
        if p["p_receive"] <= 0.0:
            continue
        role = "attack"
        if p["is_taker"]:
            role = "taker"
        elif p["is_keeper"]:
            role = "keeper"
        rows.append(
            [
                str(p["idx"]),
                role,
                f"{p['x']:.1f}",
                f"{p['y']:.1f}",
                f"{p['p_receive']:.3f}",
                _bar(p["p_receive"]),
            ]
        )
    rows.append(
        [
            "-",
            "defence",
            "",
            "",
            f"{result['p_defence_first_contact']:.3f}",
            _bar(result["p_defence_first_contact"]),
        ]
    )
    table = _table(
        ["#", "role", "x", "y", "P(first contact)", ""],
        rows,
        [True, False, True, True, True, False],
    )

    lines = [
        head,
        f"{result['n_players']} players: {result['n_attackers']} attacking, "
        f"{result['n_defenders']} defending",
        "",
        table,
        "",
        f"P(shot within 5 events)  {result['p_shot']:.3f}  {_bar(result['p_shot'])}",
    ]
    actual = result.get("actual")
    if actual and actual.get("first_touch"):
        recv = actual.get("receiver_idx")
        who = f"attacker #{recv}" if recv is not None else str(actual["first_touch"])
        lines.append(
            f"actual: first contact {actual['first_touch']} ({who}), "
            f"shot within 5: {bool(actual.get('shot_within_5'))}"
        )
    lines.append("positions only; freeze frames carry no velocities (--assume-static)")
    return "\n".join(lines)


def _describe_result(result: dict) -> str:
    bits = [
        str(result[k])
        for k in ("competition", "season")
        if result.get(k)
    ]
    if result.get("match_id") is not None:
        bits.append(f"match {result['match_id']}")
    if result.get("minute") is not None:
        bits.append(f"{result['minute']}'")
    return ", ".join(bits)


def format_scores(results: Sequence[dict], fmt: str) -> str:
    """Render scored corners as ``pretty``, ``json`` or ``csv``."""
    if fmt == "json":
        payload: Any = results[0] if len(results) == 1 else list(results)
        return json.dumps(payload, indent=2)
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(
            [
                "corner_id",
                "idx",
                "team",
                "x",
                "y",
                "is_taker",
                "is_keeper",
                "p_receive",
                "p_shot",
            ]
        )
        for r in results:
            for p in r["players"]:
                writer.writerow(
                    [
                        r["corner_id"],
                        p["idx"],
                        p["team"],
                        f"{p['x']:.2f}",
                        f"{p['y']:.2f}",
                        int(p["is_taker"]),
                        int(p["is_keeper"]),
                        f"{p['p_receive']:.6f}",
                        f"{r['p_shot']:.6f}",
                    ]
                )
            writer.writerow(
                [
                    r["corner_id"],
                    -1,
                    "defence_first_contact",
                    "",
                    "",
                    0,
                    0,
                    f"{r['p_defence_first_contact']:.6f}",
                    f"{r['p_shot']:.6f}",
                ]
            )
        return buf.getvalue().rstrip("\n")
    if fmt == "pretty":
        return "\n\n".join(_pretty_score(r) for r in results)
    raise ValueError(f"unknown format {fmt!r}")


# --------------------------------------------------------------------------- #
# suggestions
# --------------------------------------------------------------------------- #
def suggest_frame(
    frame: CornerFrame,
    shot_model: Any,
    *,
    movers: int,
    max_move: float,
    seed: int,
    defend: bool,
    beam: int,
    top_k: int,
    max_evals: int | None = None,
) -> dict:
    """Run the adjustment search and flatten the suggestions into a dict."""
    from setpiece import suggest as suggest_mod

    kwargs: dict[str, Any] = {
        "movers": movers,
        "max_move": max_move,
        "seed": seed,
        "defend": defend,
        "beam": beam,
        "top_k": top_k,
    }
    if max_evals is not None:
        kwargs["max_evals"] = max_evals
    suggestions = suggest_mod.suggest(frame, shot_model, **kwargs)

    out: list[dict] = []
    for rank, s in enumerate(suggestions, 1):
        moves = []
        for m in s.moves:
            p = frame.players[m.player_idx]
            moves.append(
                {
                    "player_idx": m.player_idx,
                    "dx": round(float(m.dx), 3),
                    "dy": round(float(m.dy), 3),
                    "length": round(math.hypot(m.dx, m.dy), 3),
                    "from": [round(float(p.x), 2), round(float(p.y), 2)],
                    "to": [round(float(p.x + m.dx), 2), round(float(p.y + m.dy), 2)],
                }
            )
        out.append(
            {
                "rank": rank,
                "moves": moves,
                "base_prob": float(s.base_prob),
                "new_prob": float(s.new_prob),
                "delta": float(s.delta),
            }
        )

    base = suggestions[0].base_prob if suggestions else None
    return {
        "corner_id": frame.corner_id,
        "competition": frame.competition,
        "season": frame.season,
        "match_id": frame.match_id,
        "minute": frame.minute,
        "objective": "minimise_shot" if defend else "maximise_shot",
        "params": {
            "movers": movers,
            "max_move": max_move,
            "seed": seed,
            "beam": beam,
            "top_k": top_k,
        },
        "base_p_shot": base,
        "suggestions": out,
        "assume_static": True,
    }


def _moves_text(moves: Sequence[dict]) -> str:
    return "; ".join(
        f"#{m['player_idx']} -> {m['to'][0]:.1f},{m['to'][1]:.1f} ({m['length']:.1f}m)"
        for m in moves
    )


def _pretty_suggestion(result: dict) -> str:
    head = f"corner {_short_id(result['corner_id'])}"
    where = _describe_result(result)
    if where:
        head = f"{head}  ({where})"
    goal = (
        "lower the shot probability"
        if result["objective"] == "minimise_shot"
        else "raise the shot probability"
    )
    lines = [head, f"moving up to {result['params']['movers']} defenders to {goal}"]
    if result["base_p_shot"] is None:
        lines.append("")
        lines.append("no admissible adjustment found for this shape")
        return "\n".join(lines)

    lines.append(f"base P(shot) {result['base_p_shot']:.3f}")
    lines.append("")
    rows = [
        [
            str(s["rank"]),
            f"{s['new_prob']:.3f}",
            f"{s['delta']:+.3f}",
            _moves_text(s["moves"]),
        ]
        for s in result["suggestions"]
    ]
    lines.append(
        _table(
            ["#", "P(shot)", "delta", "moves"],
            rows,
            [True, True, True, False],
        )
    )
    lines.append("")
    lines.append(
        "static repositioning at the moment of the delivery; no player is assumed "
        "to cover ground (--assume-static)"
    )
    return "\n".join(lines)


def format_suggestions(results: Sequence[dict], fmt: str) -> str:
    """Render suggestion sets as ``pretty``, ``json`` or ``csv``."""
    if fmt == "json":
        payload: Any = results[0] if len(results) == 1 else list(results)
        return json.dumps(payload, indent=2)
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(
            [
                "corner_id",
                "rank",
                "base_prob",
                "new_prob",
                "delta",
                "player_idx",
                "dx",
                "dy",
                "to_x",
                "to_y",
            ]
        )
        for r in results:
            for s in r["suggestions"]:
                for m in s["moves"]:
                    writer.writerow(
                        [
                            r["corner_id"],
                            s["rank"],
                            f"{s['base_prob']:.6f}",
                            f"{s['new_prob']:.6f}",
                            f"{s['delta']:.6f}",
                            m["player_idx"],
                            f"{m['dx']:.3f}",
                            f"{m['dy']:.3f}",
                            f"{m['to'][0]:.2f}",
                            f"{m['to'][1]:.2f}",
                        ]
                    )
        return buf.getvalue().rstrip("\n")
    if fmt == "pretty":
        return "\n\n".join(_pretty_suggestion(r) for r in results)
    raise ValueError(f"unknown format {fmt!r}")


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #
def _cmd_score(args: argparse.Namespace, out: Any) -> int:
    frames = _load_frames(args.corner)
    _tune_torch_threads()
    receiver_model, _ = load_head("receiver", args.model)
    shot_model, _ = load_head("shot", args.model)
    results = [score_frame(f, receiver_model, shot_model) for f in frames]
    print(format_scores(results, args.format), file=out)
    return EXIT_OK


def _cmd_suggest(args: argparse.Namespace, out: Any) -> int:
    frames = _load_frames(args.corner)
    _tune_torch_threads()
    shot_model, _ = load_head("shot", args.model)
    results = [
        suggest_frame(
            f,
            shot_model,
            movers=args.movers,
            max_move=args.max_move,
            seed=args.seed,
            defend=not args.attack,
            beam=args.beam,
            top_k=args.top_k if args.top_k is not None else args.beam,
        )
        for f in frames
    ]
    print(format_suggestions(results, args.format), file=out)
    return EXIT_OK


def _cmd_demo(args: argparse.Namespace, out: Any) -> int:
    import time

    started = time.perf_counter()
    frames = demo_frames()
    _tune_torch_threads()
    receiver_model, receiver_card = load_head("receiver", args.model)
    shot_model, _ = load_head("shot", args.model)
    loaded = time.perf_counter()

    scores = [score_frame(f, receiver_model, shot_model) for f in frames]
    plans = [
        suggest_frame(
            f,
            shot_model,
            movers=DEMO_MOVERS,
            max_move=3.0,
            seed=0,
            defend=True,
            beam=DEMO_BEAM,
            top_k=DEMO_TOP_K,
            max_evals=DEMO_MAX_EVALS,
        )
        for f in frames
    ]

    if args.format == "json":
        payload = {
            "corners": [
                {"score": s, "suggest": p} for s, p in zip(scores, plans, strict=True)
            ],
            "model_card_date": receiver_card.get("date"),
            "seconds": round(time.perf_counter() - started, 2),
            "startup_seconds": round(loaded - started, 2),
        }
        print(json.dumps(payload, indent=2), file=out)
        return EXIT_OK
    if args.format == "csv":
        print(format_scores(scores, "csv"), file=out)
        print(file=out)
        print(format_suggestions(plans, "csv"), file=out)
        return EXIT_OK

    print("setpiece demo: three real Euro 2024 corners, scored offline", file=out)
    print("=" * 70, file=out)
    for score, plan in zip(scores, plans, strict=True):
        print(file=out)
        print(_pretty_score(score), file=out)
        print(file=out)
        print(_pretty_suggestion(plan), file=out)
        print("-" * 70, file=out)
    print(
        f"scored and searched {len(frames)} corners in "
        f"{time.perf_counter() - loaded:.1f}s "
        f"(plus {loaded - started:.1f}s to import torch and load the checkpoints), "
        f"no network access",
        file=out,
    )
    return EXIT_OK


def _cmd_app(args: argparse.Namespace, out: Any) -> int:
    import shutil
    import subprocess

    app_path = Path(__file__).resolve().parent / "app.py"
    if not app_path.exists():
        print(f"error: the Streamlit app is not installed at {app_path}", file=sys.stderr)
        return EXIT_NO_DEPENDENCY
    if shutil.which("streamlit") is None:
        try:
            import importlib.util

            if importlib.util.find_spec("streamlit") is None:
                raise ImportError
        except ImportError:
            print(
                "error: streamlit is not installed. Install the app extra:\n"
                "    uv sync --extra app",
                file=sys.stderr,
            )
            return EXIT_NO_DEPENDENCY

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
    ]
    if args.headless:
        cmd += ["--server.headless", "true"]
    if args.model:
        cmd += ["--", "--model", str(args.model)]
    print(f"launching: {' '.join(cmd)}", file=out)
    return int(subprocess.call(cmd))


def _cmd_fetch(args: argparse.Namespace, out: Any) -> int:
    from setpiece.fetch import build_dataset

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = build_dataset(
        Path(args.out),
        Path(args.raw_dir),
        only_competitions=list(args.competition) if args.competition else None,
        max_matches=args.max_matches,
        keep_raw=args.keep_raw,
    )
    print(
        f"{summary['corners']} corners from {summary['matches']} matches -> {args.out}",
        file=out,
    )
    if summary["failures"]:
        print(f"{len(summary['failures'])} matches failed", file=out)
    return EXIT_OK


def _cmd_train(args: argparse.Namespace, out: Any) -> int:
    return run_train(list(args.train_args))


def run_train(extra: Sequence[str]) -> int:
    """Hand off to :mod:`setpiece.train`, imported lazily so the CLI never needs it.

    Arguments are forwarded verbatim; this wrapper deliberately knows nothing
    about the training module's own flags.
    """
    try:
        from setpiece import train as train_mod
    except ImportError as exc:
        print(f"error: setpiece.train is unavailable: {exc}", file=sys.stderr)
        return EXIT_ERROR

    forwarded = list(extra)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    entry = None
    for name in ("main", "cli", "run"):
        candidate = getattr(train_mod, name, None)
        if callable(candidate):
            entry = candidate
            break
    if entry is None:
        print(
            "error: setpiece.train exposes no main()/cli()/run() entry point",
            file=sys.stderr,
        )
        return EXIT_ERROR
    result = entry(forwarded)
    return EXIT_OK if result is None else int(result)


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def _add_model_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        metavar="PATH",
        default=None,
        help=(
            "checkpoint file, or a directory holding receiver.pt and shot.pt. "
            "Defaults to the models bundled in setpiece/assets/."
        ),
    )


def _add_format_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=_FORMATS,
        default="pretty",
        help="output format (default: pretty)",
    )


def build_parser() -> argparse.ArgumentParser:
    """The full argument parser. Exposed so tests and docs can introspect it."""
    parser = argparse.ArgumentParser(
        prog="setpiece",
        description="Corner-kick receiver and shot scoring, with defender adjustment search.",
    )
    parser.add_argument("--version", action="version", version=f"setpiece {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress to stderr")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_score = sub.add_parser("score", help="score one corner: receiver, defence, shot")
    p_score.add_argument("corner", help="corner JSON/CSV file, or a .jsonl dataset")
    _add_format_flag(p_score)
    _add_model_flag(p_score)
    p_score.set_defaults(func=_cmd_score)

    p_suggest = sub.add_parser("suggest", help="rank small defensive adjustments")
    p_suggest.add_argument("corner", help="corner JSON/CSV file, or a .jsonl dataset")
    objective = p_suggest.add_mutually_exclusive_group()
    objective.add_argument(
        "--defend",
        action="store_true",
        help="minimise the shot probability (the default)",
    )
    objective.add_argument(
        "--attack",
        action="store_true",
        help="maximise it instead, to show where the shape breaks",
    )
    p_suggest.add_argument("--movers", type=int, default=3, help="defenders moved (default: 3)")
    p_suggest.add_argument(
        "--max-move",
        type=float,
        default=3.0,
        dest="max_move",
        help="maximum displacement in metres (default: 3.0)",
    )
    p_suggest.add_argument("--seed", type=int, default=0, help="search seed (default: 0)")
    p_suggest.add_argument("--beam", type=int, default=8, help="beam width (default: 8)")
    p_suggest.add_argument(
        "--top-k",
        type=int,
        default=None,
        dest="top_k",
        help="suggestions returned (default: the beam width)",
    )
    _add_format_flag(p_suggest)
    _add_model_flag(p_suggest)
    p_suggest.set_defaults(func=_cmd_suggest)

    p_demo = sub.add_parser("demo", help="score the three bundled Euro 2024 corners, offline")
    _add_format_flag(p_demo)
    _add_model_flag(p_demo)
    p_demo.set_defaults(func=_cmd_demo)

    p_app = sub.add_parser("app", help="launch the Streamlit pitch editor")
    p_app.add_argument("--port", type=int, default=8501, help="server port (default: 8501)")
    p_app.add_argument(
        "--headless",
        action="store_true",
        help="do not open a browser window",
    )
    _add_model_flag(p_app)
    p_app.set_defaults(func=_cmd_app)

    p_fetch = sub.add_parser("fetch", help="build the corner dataset from StatsBomb open data")
    p_fetch.add_argument(
        "--competition",
        action="append",
        metavar="NAME",
        help="restrict to a competition name; repeatable",
    )
    p_fetch.add_argument(
        "--max-matches",
        type=int,
        default=None,
        dest="max_matches",
        help="stop after N matches",
    )
    p_fetch.add_argument(
        "--out",
        default="data/interim/corners.jsonl",
        help="dataset path (default: data/interim/corners.jsonl)",
    )
    p_fetch.add_argument(
        "--raw-dir",
        default="data/raw",
        dest="raw_dir",
        help="download cache (default: data/raw)",
    )
    p_fetch.add_argument(
        "--keep-raw",
        action="store_true",
        dest="keep_raw",
        help="keep the bulky raw event and 360 files",
    )
    p_fetch.set_defaults(func=_cmd_fetch)

    p_train = sub.add_parser(
        "train",
        help="train the receiver and shot models (wraps setpiece.train)",
    )
    p_train.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        metavar="...",
        help="arguments forwarded verbatim to setpiece.train",
    )
    p_train.set_defaults(func=_cmd_train)

    return parser


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def _train_passthrough(tokens: Sequence[str]) -> tuple[list[str], bool] | None:
    """Split ``train`` off before argparse sees it, or return ``None``.

    ``argparse.REMAINDER`` drops the first argument when it starts with a dash,
    so ``setpiece train --epochs 20`` would lose ``--epochs``. Peeling the
    subcommand off by hand keeps the forwarding verbatim and lets
    ``setpiece train --help`` reach the training module's own help.
    """
    verbose = False
    for i, tok in enumerate(tokens):
        if tok in ("-v", "--verbose"):
            verbose = True
            continue
        return (list(tokens[i + 1 :]), verbose) if tok == "train" else None
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand. Returns the process exit code; never raises for user error."""
    tokens = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    passthrough = _train_passthrough(tokens)
    if passthrough is not None:
        forwarded, verbose = passthrough
        if verbose:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        action: Any = partial(run_train, forwarded)
    else:
        args = parser.parse_args(tokens)
        if getattr(args, "command", None) is None:
            parser.print_help()
            return EXIT_USAGE
        if args.verbose:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        action = partial(args.func, args, sys.stdout)

    try:
        return int(action())
    except ModelUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NO_MODEL
    except BrokenPipeError:
        return EXIT_OK
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_ERROR
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
