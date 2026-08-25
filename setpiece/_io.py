"""StatsBomb 360 corner loading, plus the user-facing input formats.

This module is the minimal corner-only substitute for the shared ``sportio``
data layer. If ``sportio`` is installed it is preferred for the raw event and
freeze-frame fetch; otherwise everything here is self-contained. See TODO.md.

Downloads are cached under ``data/raw/``, capped at one request per second per
host with exponential backoff. Freeze-frame files are large (~7 MB per match)
and are discarded after corner extraction unless ``keep_raw`` is set, which
keeps the working tree small enough to build the full dataset on a laptop.
"""

from __future__ import annotations

import json
import logging
import math
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from setpiece.types import (
    SB_LENGTH,
    SB_WIDTH,
    CornerFrame,
    Player,
)

log = logging.getLogger(__name__)

SB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
USER_AGENT = "setpiece/0.1 (+https://github.com/daltonoscar0/setpiece)"

_MIN_INTERVAL = 1.0  # seconds between requests to the same host
_last_request_at: float = 0.0

#: Event types that represent a genuine touch of the ball.
TOUCH_TYPES = frozenset(
    {
        "Ball Receipt*",
        "Shot",
        "Clearance",
        "Miscontrol",
        "Interception",
        "Block",
        "Goal Keeper",
        "Pass",
    }
)

#: A receiver match further than this (metres) from the touch location is treated
#: as ambiguous and the label is dropped.
RECEIVER_MATCH_MAX_M = 8.0

#: How many events after the corner are scanned for a shot.
SHOT_WINDOW = 5


# --------------------------------------------------------------------------- #
# coordinates
# --------------------------------------------------------------------------- #
def sb_to_metres(x: float, y: float) -> tuple[float, float]:
    """Rescale StatsBomb 120x80 units onto the canonical 105x68 metre pitch."""
    from setpiece.types import PITCH_LENGTH, PITCH_WIDTH

    return (x * PITCH_LENGTH / SB_LENGTH, y * PITCH_WIDTH / SB_WIDTH)


# --------------------------------------------------------------------------- #
# polite fetching
# --------------------------------------------------------------------------- #
def _throttle() -> None:
    global _last_request_at
    wait = _MIN_INTERVAL - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def fetch_json(url: str, *, retries: int = 4, timeout: int = 120) -> Any:
    """GET and parse JSON, rate limited with exponential backoff."""
    delay = 2.0
    last: Exception | None = None
    for attempt in range(retries):
        _throttle()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError(url) from exc
            last = exc
        except Exception as exc:  # network flake, timeout, truncated body
            last = exc
        if attempt < retries - 1:
            log.warning("fetch failed (%s), retrying in %.0fs: %s", last, delay, url)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"failed to fetch {url}: {last}")


def cached_json(url: str, cache: Path, *, keep: bool = True) -> Any:
    """Fetch ``url``, reusing ``cache`` if present. Writes the cache when ``keep``."""
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except json.JSONDecodeError:
            log.warning("corrupt cache %s, refetching", cache)
            cache.unlink(missing_ok=True)
    payload = fetch_json(url)
    if keep:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload))
    return payload


# --------------------------------------------------------------------------- #
# StatsBomb catalogue
# --------------------------------------------------------------------------- #
def list_360_seasons(raw_dir: Path) -> list[dict]:
    """Every open competition-season that ships 360 freeze frames."""
    comps = cached_json(f"{SB_BASE}/competitions.json", raw_dir / "competitions.json")
    seen: set[tuple[int, int]] = set()
    out: list[dict] = []
    for c in comps:
        if not c.get("match_available_360"):
            continue
        key = (c["competition_id"], c["season_id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "competition_id": c["competition_id"],
                "season_id": c["season_id"],
                "competition": c["competition_name"],
                "season": c["season_name"],
            }
        )
    return out


def list_matches(competition_id: int, season_id: int, raw_dir: Path) -> list[dict]:
    return cached_json(
        f"{SB_BASE}/matches/{competition_id}/{season_id}.json",
        raw_dir / "matches" / f"{competition_id}_{season_id}.json",
    )


# --------------------------------------------------------------------------- #
# corner extraction
# --------------------------------------------------------------------------- #
def _is_corner(ev: dict) -> bool:
    return ev.get("type", {}).get("name") == "Pass" and ev.get("pass", {}).get("type", {}).get(
        "name"
    ) == "Corner"


def _first_touch(events: list[dict], idx: int) -> dict | None:
    """First genuine touch event strictly after ``events[idx]``."""
    for ev in events[idx + 1 : idx + 1 + 12]:
        name = ev.get("type", {}).get("name")
        if name not in TOUCH_TYPES:
            continue
        if name == "Ball Receipt*":
            outcome = ev.get("ball_receipt", {}).get("outcome", {}).get("name")
            if outcome == "Incomplete":
                continue
        if ev.get("location") is None or ev.get("team") is None:
            continue
        return ev
    return None


def _shot_within(events: list[dict], idx: int, attacking_team_id: int) -> bool:
    for ev in events[idx + 1 : idx + 1 + SHOT_WINDOW]:
        if (
            ev.get("type", {}).get("name") == "Shot"
            and ev.get("team", {}).get("id") == attacking_team_id
        ):
            return True
    return False


def _players_from_freeze(frame: list[dict]) -> list[Player]:
    players: list[Player] = []
    for p in frame:
        loc = p.get("location")
        if not loc or len(loc) < 2:
            continue
        x, y = sb_to_metres(float(loc[0]), float(loc[1]))
        players.append(
            Player(
                x=x,
                y=y,
                is_attacker=bool(p.get("teammate")),
                is_taker=bool(p.get("actor")),
                is_keeper=bool(p.get("keeper")),
            )
        )
    return players


def corners_from_match(
    events: list[dict],
    three_sixty: list[dict],
    *,
    match_id: int | None = None,
    competition: str | None = None,
    season: str | None = None,
) -> list[CornerFrame]:
    """Extract every corner that carries a freeze frame, with labels attached.

    Roughly half of open-data corners have no 360 frame; those are skipped.
    """
    frames = {f["event_uuid"]: f for f in three_sixty if f.get("event_uuid")}
    out: list[CornerFrame] = []

    for idx, ev in enumerate(events):
        if not _is_corner(ev):
            continue
        frame = frames.get(ev["id"])
        if frame is None or not frame.get("freeze_frame"):
            continue
        players = _players_from_freeze(frame["freeze_frame"])
        if len(players) < 8:
            continue

        attacking_team_id = ev.get("team", {}).get("id")
        shot = _shot_within(events, idx, attacking_team_id)

        receiver_idx: int | None = None
        match_dist: float | None = None
        touch = _first_touch(events, idx)
        if touch is None:
            first_touch = "unknown"
        elif touch.get("team", {}).get("id") != attacking_team_id:
            first_touch = "defender"
        elif touch.get("type", {}).get("name") == "Goal Keeper":
            # the attacking team's own keeper never receives their own corner
            first_touch = "defender"
        else:
            first_touch = "attacker"
            tx, ty = sb_to_metres(float(touch["location"][0]), float(touch["location"][1]))
            best_i, best_d = None, math.inf
            for i, p in enumerate(players):
                if not p.is_attacker or p.is_taker:
                    continue
                d = math.hypot(p.x - tx, p.y - ty)
                if d < best_d:
                    best_i, best_d = i, d
            if best_i is not None:
                match_dist = best_d
                # a far match means the receiver was off-camera; keep the distance
                # for auditing but drop the index so training can filter on it
                receiver_idx = best_i if best_d <= RECEIVER_MATCH_MAX_M else None

        pass_info = ev.get("pass", {})
        end_loc = pass_info.get("end_location")
        out.append(
            CornerFrame(
                corner_id=ev["id"],
                players=players,
                match_id=match_id,
                competition=competition,
                season=season,
                period=ev.get("period"),
                minute=ev.get("minute"),
                first_touch=first_touch,
                receiver_idx=receiver_idx,
                receiver_match_dist=match_dist,
                shot_within_5=shot,
                meta={
                    "play_pattern": ev.get("play_pattern", {}).get("name"),
                    "pass_length": pass_info.get("length"),
                    "pass_height": pass_info.get("height", {}).get("name"),
                    "pass_outcome": pass_info.get("outcome", {}).get("name"),
                    # Delivery destination. LABEL-ADJACENT: this is the outcome of
                    # the corner, recorded for diagnostics only. It must never be
                    # used as a model feature.
                    "end_xy": list(sb_to_metres(float(end_loc[0]), float(end_loc[1])))
                    if end_loc
                    else None,
                },
            )
        )
    return out


def fetch_match_corners(
    match_id: int,
    raw_dir: Path,
    *,
    competition: str | None = None,
    season: str | None = None,
    keep_raw: bool = False,
) -> list[CornerFrame]:
    """Download one match, extract its corners, then drop the bulky raw files."""
    ev_cache = raw_dir / "events" / f"{match_id}.json"
    ts_cache = raw_dir / "three-sixty" / f"{match_id}.json"
    events = cached_json(f"{SB_BASE}/events/{match_id}.json", ev_cache, keep=keep_raw)
    try:
        three_sixty = cached_json(
            f"{SB_BASE}/three-sixty/{match_id}.json", ts_cache, keep=keep_raw
        )
    except FileNotFoundError:
        log.info("no 360 file for match %s", match_id)
        three_sixty = []
    return corners_from_match(
        events, three_sixty, match_id=match_id, competition=competition, season=season
    )


# --------------------------------------------------------------------------- #
# dataset serialisation
# --------------------------------------------------------------------------- #
def corner_to_dict(c: CornerFrame) -> dict:
    d = asdict(c)
    d["players"] = [asdict(p) for p in c.players]
    return d


def corner_from_dict(d: dict) -> CornerFrame:
    players = [Player(**p) for p in d["players"]]
    return CornerFrame(**{**d, "players": players})


def write_corners(corners: Iterable[CornerFrame], path: Path) -> int:
    """Write corners as JSON lines. Returns the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as fh:
        for c in corners:
            fh.write(json.dumps(corner_to_dict(c)) + "\n")
            n += 1
    return n


def read_corners(path: Path) -> list[CornerFrame]:
    with Path(path).open() as fh:
        return [corner_from_dict(json.loads(line)) for line in fh if line.strip()]


def iter_corners(path: Path) -> Iterator[CornerFrame]:
    with Path(path).open() as fh:
        for line in fh:
            if line.strip():
                yield corner_from_dict(json.loads(line))


# --------------------------------------------------------------------------- #
# user-facing single-corner inputs
# --------------------------------------------------------------------------- #
def load_corner(path: str | Path) -> CornerFrame:
    """Load one corner from a StatsBomb 360 JSON export, our JSON, or a CSV.

    CSV columns: ``team,x,y,is_taker`` with an optional ``is_keeper``. ``team``
    accepts ``attack``/``defend`` (or ``att``/``def``, ``1``/``0``). Coordinates
    are metres on a 105x68 pitch, attacking towards +x.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return _load_corner_csv(path)
    payload = json.loads(path.read_text())
    return _load_corner_json(payload, source=path.stem)


def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def _load_corner_csv(path: Path) -> CornerFrame:
    import csv

    players: list[Player] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            row = {(k or "").strip().lower(): v for k, v in row.items()}
            team = str(row.get("team", "")).strip().lower()
            if team in {"attack", "att", "a", "1", "attacking"}:
                is_attacker = True
            elif team in {"defend", "def", "d", "0", "defending"}:
                is_attacker = False
            else:
                raise ValueError(f"unrecognised team value {team!r} in {path}")
            players.append(
                Player(
                    x=float(row["x"]),
                    y=float(row["y"]),
                    is_attacker=is_attacker,
                    is_taker=_truthy(row.get("is_taker", 0)),
                    is_keeper=_truthy(row.get("is_keeper", 0)),
                )
            )
    if not players:
        raise ValueError(f"no players parsed from {path}")
    return CornerFrame(corner_id=path.stem, players=players)


def _load_corner_json(payload: Any, *, source: str) -> CornerFrame:
    # our own serialisation
    if isinstance(payload, dict) and "players" in payload:
        return corner_from_dict(payload)
    # a raw StatsBomb 360 entry, or a bare freeze_frame list
    if isinstance(payload, dict) and "freeze_frame" in payload:
        players = _players_from_freeze(payload["freeze_frame"])
        return CornerFrame(corner_id=payload.get("event_uuid", source), players=players)
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        if "teammate" in payload[0]:
            return CornerFrame(corner_id=source, players=_players_from_freeze(payload))
        if "freeze_frame" in payload[0]:
            entry = payload[0]
            return CornerFrame(
                corner_id=entry.get("event_uuid", source),
                players=_players_from_freeze(entry["freeze_frame"]),
            )
    raise ValueError(f"unrecognised corner JSON structure in {source}")


def save_corner(corner: CornerFrame, path: str | Path) -> None:
    Path(path).write_text(json.dumps(corner_to_dict(corner), indent=1))
