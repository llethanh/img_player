"""Group raw EXR channel names into layer-level entries.

Multichannel EXRs typically expose channels named like a flat list:

    R, G, B, A, volume_Z,
    albedo.R, albedo.G, albedo.B,
    diffuse.R, diffuse.G, diffuse.B,
    normal.X, normal.Y, normal.Z,
    crypto00.r, crypto00.g, crypto00.b, crypto00.a,
    Z

Showing each one as a separate entry in the channel selector floods
the user with a hundred options and breaks the natural workflow —
artists think in *layers* (= passes), not in individual channels.
This module collapses the flat list into a UI-friendly representation
where ``albedo.R``/``.G``/``.B`` becomes a single ``"albedo"`` entry
that loads the three channels as an RGB composite.

The output is ordered to match the user's mental model:
1. The default ``"RGB"`` (or ``"RGBA"``) entry first — the beauty pass.
2. Other RGB-shaped layers in their original order (``albedo``,
   ``diffuse``, …).
3. Anything that wasn't grouped (``Z``, ``volume_Z``, single-channel
   masks, normals if not grouped) — listed last as ``layer.sub``
   so the user can still reach them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# RGB-like sub-channel names recognised when grouping. Some renderers
# write lower-case (Cryptomatte does), so we normalise.
_RGB_SUBS = ("r", "g", "b")
_ALPHA_SUBS = ("a",)


@dataclass(frozen=True)
class ChannelGroup:
    """One entry in the UI selector.

    ``label`` is what we show in the combo box. ``channels`` is the
    raw OIIO channel list passed to ``read_frame`` (preserves OIIO
    naming case).
    """

    label: str
    channels: tuple[str, ...]


def _split_layer(name: str) -> tuple[str, str | None]:
    """Split ``"albedo.R"`` → ``("albedo", "R")``; ``"Z"`` → ``("Z", None)``.

    EXR uses ``.`` as the layer separator. Names without one are
    bare (R, G, B, A at the root, or single-channel masks).
    """
    if "." not in name:
        return name, None
    head, _, tail = name.rpartition(".")
    return head, tail


def group_channels(raw: Iterable[str]) -> list[ChannelGroup]:
    """Convert a flat list of EXR channel names into UI groups.

    Rules:

    * The bare ``R``/``G``/``B`` (+ optional ``A``) at the root
      become a single ``"RGB"`` (or ``"RGBA"``) entry — the beauty
      pass, always first.
    * Any layer that has at least R, G and B sub-channels becomes a
      single layer entry (e.g. ``"albedo"``) loading the three (or
      four with alpha) channels as a composite.
    * Everything else stays individual: ``"Z"``, ``"volume_Z"``,
      ``"normal.X"`` (if normal didn't have R/G/B), single masks…

    The list preserves the original ordering of the input so that
    layer order from the renderer is respected (diffuse before
    specular, etc.).
    """
    raw_list = list(raw)

    # Per-layer accumulator of {sub_lower: original_name}. We index
    # by lowercase for matching but keep the original case in the
    # output channels.
    by_layer: dict[str, dict[str, str]] = {}
    # First-seen index for each layer — drives the output order.
    layer_order: list[str] = []
    # Channels with no "." (bare names like "R", "Z").
    bare_channels: list[str] = []

    for ch in raw_list:
        layer, sub = _split_layer(ch)
        if sub is None:
            bare_channels.append(ch)
            continue
        if layer not in by_layer:
            by_layer[layer] = {}
            layer_order.append(layer)
        by_layer[layer][sub.lower()] = ch

    groups: list[ChannelGroup] = []

    # 1. Root RGB(A) — the beauty pass.
    bare_lower = {b.lower(): b for b in bare_channels}
    if all(s in bare_lower for s in _RGB_SUBS):
        chans = tuple(bare_lower[s] for s in _RGB_SUBS)
        if "a" in bare_lower:
            chans = chans + (bare_lower["a"],)
            groups.append(ChannelGroup("RGBA", chans))
        else:
            groups.append(ChannelGroup("RGB", chans))
        # Mark these as "consumed" so they don't reappear later.
        consumed = set(chans)
        bare_channels = [b for b in bare_channels if b not in consumed]

    # 2. RGB-shaped layers (albedo, diffuse, specular…).
    for layer in layer_order:
        subs = by_layer[layer]
        if all(s in subs for s in _RGB_SUBS):
            chans = tuple(subs[s] for s in _RGB_SUBS)
            if "a" in subs:
                chans = chans + (subs["a"],)
            groups.append(ChannelGroup(layer, chans))
            # Remove the consumed sub-channels so they don't
            # re-appear individually below.
            for s in _RGB_SUBS + _ALPHA_SUBS:
                subs.pop(s, None)

    # 3. Leftover sub-channels (e.g. ``normal.X`` if normal had no
    # R/G/B, or AOVs that only have one component). Listed in the
    # original order.
    for layer in layer_order:
        subs = by_layer[layer]
        for sub_name, original in subs.items():
            groups.append(ChannelGroup(original, (original,)))

    # 4. Bare leftovers (Z, masks…). Often the most useful ones for
    # inspection, but listed last because they're the "secondary"
    # channels.
    for ch in bare_channels:
        groups.append(ChannelGroup(ch, (ch,)))

    return groups
