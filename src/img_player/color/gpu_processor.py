"""Builds GLSL shader source + LUT textures for an OCIO color transform."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from typing import Literal

import numpy as np
import PyOpenColorIO as ocio

from img_player.color.ocio_manager import OCIOManager

# Load shader templates from the package at import time. These are the
# "skeleton" shaders that the OCIO-generated function gets injected into.
_VERTEX_SRC = (
    resources.files("img_player.render.shaders").joinpath("vertex.glsl").read_text(encoding="utf-8")
)
_FRAGMENT_TEMPLATE = (
    resources.files("img_player.render.shaders")
    .joinpath("fragment_template.glsl")
    .read_text(encoding="utf-8")
)

# Sentinel that the template uses to splice in OCIO's generated GLSL.
# Chosen to look obviously non-GLSL so it can't accidentally appear in
# OCIO's output or our comments.
_OCIO_PLACEHOLDER = "@@OCIO_INJECT@@"


@dataclass(frozen=True)
class Texture1D:
    name: str
    values: np.ndarray  # shape (N,) float32 for RED, or (N, 3) for RGB
    channel: Literal["RED", "RGB"]


@dataclass(frozen=True)
class Texture3D:
    name: str
    values: np.ndarray  # shape (edge, edge, edge, 3) float32
    edge: int


@dataclass(frozen=True)
class ShaderBundle:
    """Everything the GL viewport needs to apply an OCIO color transform.

    ``vertex_source`` / ``fragment_source`` are compile-ready GLSL. LUT
    textures are the 1D/3D samplers OCIO declared in the fragment source —
    the GL viewport uploads and binds them to the matching sampler uniform
    names.
    """

    vertex_source: str
    fragment_source: str
    textures_1d: tuple[Texture1D, ...] = field(default_factory=tuple)
    textures_3d: tuple[Texture3D, ...] = field(default_factory=tuple)
    source_colorspace: str = ""
    display: str = ""
    view: str = ""


def build_shader_bundle(
    manager: OCIOManager,
    *,
    source_colorspace: str,
    display: str,
    view: str,
    language: ocio.GpuLanguage = ocio.GPU_LANGUAGE_GLSL_4_0,
) -> ShaderBundle:
    """Build a compile-ready ``ShaderBundle`` for the given transform.

    Raises :class:`PyOpenColorIO.Exception` if the transform cannot be built
    (invalid colorspace / display / view).
    """
    processor = manager.get_display_view_processor(source_colorspace, display, view)
    gpu_proc = processor.getDefaultGPUProcessor()

    desc = ocio.GpuShaderDesc.CreateShaderDesc()
    desc.setLanguage(language)
    desc.setFunctionName("OCIOMain")
    desc.setResourcePrefix("ocio_")
    gpu_proc.extractGpuShaderInfo(desc)

    ocio_glsl = desc.getShaderText()
    fragment_source = _FRAGMENT_TEMPLATE.replace(_OCIO_PLACEHOLDER, ocio_glsl)

    textures_1d = _collect_textures_1d(desc)
    textures_3d = _collect_textures_3d(desc)

    return ShaderBundle(
        vertex_source=_VERTEX_SRC,
        fragment_source=fragment_source,
        textures_1d=textures_1d,
        textures_3d=textures_3d,
        source_colorspace=source_colorspace,
        display=display,
        view=view,
    )


def _collect_textures_1d(desc: ocio.GpuShaderDesc) -> tuple[Texture1D, ...]:
    textures: list[Texture1D] = []
    for tex in desc.getTextures():
        values_flat = np.asarray(tex.getValues(), dtype=np.float32)
        if tex.channel == desc.TEXTURE_RED_CHANNEL:
            values = values_flat.reshape(tex.width)
            chan_label: Literal["RED", "RGB"] = "RED"
        else:
            values = values_flat.reshape(tex.width, 3)
            chan_label = "RGB"
        textures.append(Texture1D(name=tex.samplerName, values=values, channel=chan_label))
    return tuple(textures)


def _collect_textures_3d(desc: ocio.GpuShaderDesc) -> tuple[Texture3D, ...]:
    textures: list[Texture3D] = []
    for tex in desc.get3DTextures():
        edge = tex.edgeLen
        values_flat = np.asarray(tex.getValues(), dtype=np.float32)
        values = values_flat.reshape(edge, edge, edge, 3)
        textures.append(Texture3D(name=tex.samplerName, values=values, edge=edge))
    return tuple(textures)
