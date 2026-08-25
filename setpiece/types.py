"""Core data types and pitch constants shared by every module.

Coordinates are metres on a 105x68 pitch, matching the canonical schema used by
the rest of the tool suite. The attacking team always attacks towards +x, so the
goal being attacked is centred at ``(PITCH_LENGTH, PITCH_WIDTH / 2)``.

StatsBomb ships 120x80 units with the attacking team already normalised
left-to-right; :func:`setpiece._io.sb_to_metres` performs the rescale.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

PITCH_LENGTH: float = 105.0
PITCH_WIDTH: float = 68.0

GOAL_X: float = PITCH_LENGTH
GOAL_Y: float = PITCH_WIDTH / 2.0
GOAL_WIDTH: float = 7.32
POST_LEFT_Y: float = GOAL_Y - GOAL_WIDTH / 2.0
POST_RIGHT_Y: float = GOAL_Y + GOAL_WIDTH / 2.0

# StatsBomb native pitch, used only at the loader boundary.
SB_LENGTH: float = 120.0
SB_WIDTH: float = 80.0


@dataclass(frozen=True)
class Player:
    """One player node in a corner freeze frame.

    Freeze frames carry no player identity, so nodes are positional only.
    Velocities are not available in freeze frames and are never fabricated.
    """

    x: float
    y: float
    is_attacker: bool
    is_taker: bool = False
    is_keeper: bool = False

    def moved(self, dx: float, dy: float) -> Player:
        """Return a copy displaced by ``(dx, dy)`` metres."""
        return replace(self, x=self.x + dx, y=self.y + dy)


@dataclass
class CornerFrame:
    """A single corner kick: the freeze frame plus optional supervision labels.

    ``receiver_idx`` indexes into :attr:`players`. It is recovered by matching the
    first post-corner touch location to the nearest attacking node, so
    ``receiver_match_dist`` is retained as a label-quality flag: large values mean
    the match is ambiguous and the row should be dropped from training.
    """

    corner_id: str
    players: list[Player]
    match_id: int | None = None
    competition: str | None = None
    season: str | None = None
    period: int | None = None
    minute: int | None = None

    # Labels; all None at inference time.
    #: Which side made first contact: "attacker", "defender", or "unknown".
    #: This is a label in its own right - the receiver head carries an explicit
    #: "defence wins first contact" class, because on open data the defence wins
    #: it slightly more often than not.
    first_touch: str | None = None
    #: Index of the receiving attacker, set only when ``first_touch == "attacker"``
    #: and the positional match was unambiguous.
    receiver_idx: int | None = None
    receiver_match_dist: float | None = None
    shot_within_5: bool | None = None

    meta: dict = field(default_factory=dict)

    @property
    def attacker_indices(self) -> list[int]:
        return [i for i, p in enumerate(self.players) if p.is_attacker]

    @property
    def defender_indices(self) -> list[int]:
        return [i for i, p in enumerate(self.players) if not p.is_attacker]

    @property
    def taker_index(self) -> int | None:
        for i, p in enumerate(self.players):
            if p.is_taker:
                return i
        return None

    def n_attackers(self) -> int:
        return len(self.attacker_indices)

    def n_defenders(self) -> int:
        return len(self.defender_indices)

    def with_players(self, players: list[Player]) -> CornerFrame:
        """Return a shallow copy carrying a different player list."""
        return replace(self, players=players)
