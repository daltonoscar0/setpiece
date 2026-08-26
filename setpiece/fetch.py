"""Build the corner dataset from StatsBomb open data.

Downloads every open competition-season that ships 360 freeze frames, extracts
the corners that carry a frame, and appends them to a JSON-lines dataset. Raw
event and freeze-frame files are discarded after extraction by default, so the
full build needs well under a gigabyte of disk.

Progress is checkpointed per match, so an interrupted build resumes where it
stopped rather than re-downloading.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from setpiece._io import (
    corner_to_dict,
    fetch_match_corners,
    list_360_seasons,
    list_matches,
)

log = logging.getLogger(__name__)


def build_dataset(
    out_path: Path,
    raw_dir: Path,
    *,
    only_competitions: list[str] | None = None,
    max_matches: int | None = None,
    keep_raw: bool = False,
    max_attempts: int = 3,
) -> dict:
    """Fetch and extract corners into ``out_path`` (JSON lines). Returns a summary."""
    out_path = Path(out_path)
    raw_dir = Path(raw_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    state_path = out_path.with_suffix(".state.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    done: set[int] = set(state.get("done_matches", []))
    failures: list[dict] = state.get("failures", [])
    #: seasons flagged as having 360 data whose freeze-frame files are absent upstream
    no_360: dict[str, int] = state.get("no_360", {})
    #: match_id -> consecutive failure count, so a permanently broken match is not
    #: re-downloaded on every run. Match 3845506 has malformed JSON upstream and
    #: would otherwise cost ~7 MB and a failure row on each pass.
    attempts: dict[str, int] = {str(k): int(v) for k, v in state.get("attempts", {}).items()}

    seasons = list_360_seasons(raw_dir)
    if only_competitions:
        wanted = {c.lower() for c in only_competitions}
        seasons = [s for s in seasons if s["competition"].lower() in wanted]

    targets: list[tuple[int, dict]] = []
    for s in seasons:
        try:
            matches = list_matches(s["competition_id"], s["season_id"], raw_dir)
        except Exception as exc:
            failures.append({"season": f"{s['competition']} {s['season']}", "error": str(exc)})
            log.warning("season listing failed: %s", exc)
            continue
        for m in matches:
            targets.append((m["match_id"], s))

    targets = [t for t in targets if t[0] not in done]
    gave_up = [t for t in targets if attempts.get(str(t[0]), 0) >= max_attempts]
    if gave_up:
        log.warning(
            "skipping %d match(es) that failed %d times already: %s",
            len(gave_up),
            max_attempts,
            ", ".join(str(t[0]) for t in gave_up[:10]),
        )
    targets = [t for t in targets if attempts.get(str(t[0]), 0) < max_attempts]
    if max_matches is not None:
        targets = targets[:max_matches]

    n_corners = int(state.get("n_corners", 0))
    mode = "a" if out_path.exists() and done else "w"
    if mode == "w":
        n_corners = 0

    log.info("building dataset: %d matches to fetch", len(targets))
    with out_path.open(mode) as fh:
        for i, (match_id, season) in enumerate(targets, 1):
            report: dict = {}
            try:
                corners = fetch_match_corners(
                    match_id,
                    raw_dir,
                    competition=season["competition"],
                    season=season["season"],
                    keep_raw=keep_raw,
                    report=report,
                )
            except Exception as exc:
                n = attempts.get(str(match_id), 0) + 1
                attempts[str(match_id)] = n
                # one row per match, updated in place, rather than one per attempt
                for row in failures:
                    if row.get("match_id") == match_id:
                        row.update({"error": str(exc), "attempts": n})
                        break
                else:
                    failures.append({"match_id": match_id, "error": str(exc), "attempts": n})
                if n >= max_attempts:
                    log.warning(
                        "match %s failed %d times, giving up on it: %s", match_id, n, exc
                    )
                else:
                    log.warning("match %s failed (attempt %d): %s", match_id, n, exc)
                continue
            attempts.pop(str(match_id), None)
            if not report.get("has_360", True):
                key = f"{season['competition']} {season['season']}"
                no_360[key] = no_360.get(key, 0) + 1
            for c in corners:
                fh.write(json.dumps(corner_to_dict(c)) + "\n")
            fh.flush()
            n_corners += len(corners)
            done.add(match_id)

            if i % 10 == 0 or i == len(targets):
                state_path.write_text(
                    json.dumps(
                        {
                            "done_matches": sorted(done),
                            "n_corners": n_corners,
                            "failures": failures,
                            "no_360": no_360,
                            "attempts": attempts,
                        }
                    )
                )
                log.info("%d/%d matches, %d corners", i, len(targets), n_corners)

    state_path.write_text(
        json.dumps(
            {
                "done_matches": sorted(done),
                "n_corners": n_corners,
                "failures": failures,
                "no_360": no_360,
                "attempts": attempts,
            }
        )
    )
    if no_360:
        for season, n in sorted(no_360.items()):
            log.warning("%s: %d matches flagged for 360 but no freeze-frame file upstream", season, n)
    return {
        "matches": len(done),
        "corners": n_corners,
        "failures": failures,
        "no_360": no_360,
        "attempts": attempts,
        "gave_up": sorted(int(k) for k, v in attempts.items() if v >= max_attempts),
    }
