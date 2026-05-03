"""Time-based media abstractions.

The ``media`` package introduces a clip-level model that complements the
frame-indexed image-sequence model in ``img_player.sequence``. A *clip*
is anything you can ask "give me the picture at time T (in seconds)" —
an image sequence at a known FPS, a video container with its own PTS
clock, or eventually a still image with a duration.

This layer is the foundation for video playback: image sequences keep
their existing path through ``FrameCache`` / ``MasterFrameCache``;
video files plug in via :class:`VideoSource` and a dedicated PyAV-based
decoder. The session FPS stays as the *display cadence* (how often the
viewport refreshes); the master timeline becomes a continuous time
axis, with each layer mapping master-time → its own native frame /
PTS.
"""

from img_player.media.video_probe import (
    VIDEO_EXTENSIONS,
    VideoMetadata,
    is_video_file,
    probe_video,
)
from img_player.media.video_source import VideoSource

__all__ = [
    "VIDEO_EXTENSIONS",
    "VideoMetadata",
    "VideoSource",
    "is_video_file",
    "probe_video",
]
