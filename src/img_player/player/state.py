"""Immutable playback state and its enum companion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class LoopMode(StrEnum):
    ONCE = "once"
    LOOP = "loop"
    PING_PONG = "ping_pong"


@dataclass(frozen=True)
class PlaybackState:
    """Snapshot of the player's runtime state.

    Kept immutable so we can signal-emit it safely; transitions go through
    :func:`dataclasses.replace`.
    """

    is_playing: bool = False
    current_frame: int = 0
    fps: float = 24.0
    loop_mode: LoopMode = LoopMode.LOOP
    direction: int = 1  # +1 (forward) or -1 (reverse)
    in_frame: int | None = None  # None = use sequence.first_frame
    out_frame: int | None = None  # None = use sequence.last_frame
    dropped_frames: int = 0
