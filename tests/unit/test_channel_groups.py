"""Pure-function tests for the channel grouper."""

from __future__ import annotations

from img_player.sequence.channels import ChannelGroup, group_channels


def _labels(groups: list[ChannelGroup]) -> list[str]:
    return [g.label for g in groups]


class TestRgbBeauty:
    def test_plain_rgb(self) -> None:
        groups = group_channels(["R", "G", "B"])
        assert _labels(groups) == ["RGB"]
        assert groups[0].channels == ("R", "G", "B")

    def test_rgba(self) -> None:
        groups = group_channels(["R", "G", "B", "A"])
        assert _labels(groups) == ["RGBA"]
        assert groups[0].channels == ("R", "G", "B", "A")


class TestLayerGrouping:
    def test_albedo_rgb_collapses(self) -> None:
        groups = group_channels(
            ["R", "G", "B", "A", "albedo.R", "albedo.G", "albedo.B"]
        )
        # RGBA first, then albedo as a single composite entry.
        assert _labels(groups) == ["RGBA", "albedo"]
        albedo = next(g for g in groups if g.label == "albedo")
        assert albedo.channels == ("albedo.R", "albedo.G", "albedo.B")

    def test_layer_with_alpha_loads_four(self) -> None:
        groups = group_channels([
            "albedo.R", "albedo.G", "albedo.B", "albedo.A",
        ])
        assert len(groups) == 1
        assert groups[0].channels == (
            "albedo.R", "albedo.G", "albedo.B", "albedo.A",
        )

    def test_multiple_layers_keep_order(self) -> None:
        groups = group_channels([
            "R", "G", "B",
            "diffuse.R", "diffuse.G", "diffuse.B",
            "specular.R", "specular.G", "specular.B",
        ])
        assert _labels(groups) == ["RGB", "diffuse", "specular"]


class TestNonGroupedChannels:
    def test_bare_z_is_kept(self) -> None:
        groups = group_channels(["R", "G", "B", "Z"])
        assert _labels(groups) == ["RGB", "Z"]
        z = next(g for g in groups if g.label == "Z")
        assert z.channels == ("Z",)

    def test_normal_xyz_not_collapsed(self) -> None:
        # normal.X / .Y / .Z is NOT R/G/B, so the layer doesn't get
        # collapsed — each component appears individually.
        groups = group_channels(["normal.X", "normal.Y", "normal.Z"])
        labels = _labels(groups)
        assert "normal.X" in labels
        assert "normal.Y" in labels
        assert "normal.Z" in labels

    def test_volume_z_bare(self) -> None:
        groups = group_channels(["R", "G", "B", "volume_Z"])
        assert _labels(groups) == ["RGB", "volume_Z"]


class TestRealWorldExr:
    def test_lighting_render(self) -> None:
        # A typical Arnold/Karma lighting render with multiple AOVs.
        raw = [
            "R", "G", "B", "A",
            "diffuse.R", "diffuse.G", "diffuse.B",
            "specular.R", "specular.G", "specular.B",
            "albedo.R", "albedo.G", "albedo.B",
            "Z",
            "N.X", "N.Y", "N.Z",
        ]
        groups = group_channels(raw)
        labels = _labels(groups)
        # The beauty pass is first.
        assert labels[0] == "RGBA"
        # Then the RGB-shaped layers in their original order.
        assert labels.index("diffuse") < labels.index("specular")
        assert labels.index("specular") < labels.index("albedo")
        # Z is reachable.
        assert "Z" in labels
        # Normals aren't collapsed (no R/G/B sub-names) — listed
        # individually.
        assert "N.X" in labels and "N.Y" in labels and "N.Z" in labels


class TestEdgeCases:
    def test_empty_input(self) -> None:
        assert group_channels([]) == []

    def test_only_alpha_no_rgb(self) -> None:
        # A single bare A channel doesn't form an RGBA group.
        groups = group_channels(["A"])
        assert _labels(groups) == ["A"]

    def test_layer_with_only_two_components(self) -> None:
        # albedo.R + albedo.G (no .B) → not collapsed, listed solo.
        groups = group_channels(["albedo.R", "albedo.G"])
        labels = _labels(groups)
        assert "albedo" not in labels
        assert "albedo.R" in labels
        assert "albedo.G" in labels

    def test_case_insensitive_subs(self) -> None:
        # Cryptomatte writes lower-case sub-names.
        groups = group_channels([
            "crypto00.r", "crypto00.g", "crypto00.b", "crypto00.a",
        ])
        assert _labels(groups) == ["crypto00"]
        assert groups[0].channels == (
            "crypto00.r", "crypto00.g", "crypto00.b", "crypto00.a",
        )
