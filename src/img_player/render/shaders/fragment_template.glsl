#version 410 core

// Fragment shader template. The placeholder line below is replaced at
// runtime with GLSL that OCIO generates (a function named OCIOMain plus
// any helpers, samplers and uniforms it needs). It is kept on its own
// line so the splicing cannot accidentally break surrounding comments.

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D uImage;
uniform float uExposure;  // stops, additive in log2 space
uniform float uGamma;     // user display gamma, defaults to 1.0
// Per-channel mask: 1.0 = channel visible, 0.0 = muted. Lets the UI
// expose four toggles (R / G / B / A) without re-uploading the
// texture or invalidating the cache. The mask is applied *after* the
// OCIO transform so toggling a channel only changes what reaches the
// screen, not what we're tone-mapping.
uniform vec4 uChannelMask;

// When isolating a single colour channel (mask is one of the basis
// vectors) we render it as luminance instead of leaving the other
// two channels black — that's how Nuke / RV behave and what users
// expect from a "show only red" toggle. 1.0 = isolate-as-luminance,
// 0.0 = show as a coloured channel (matches the Nuke "channel" mode
// in pixel-pick popouts).
uniform float uChannelIsolateLuminance;

// Checker pattern cell size in screen pixels. Standard VFX viewers
// use 8–16 px squares; we expose it as a uniform so the host can
// tweak via Preferences without recompiling the shader. The checker
// is only drawn where the cached buffer's alpha is < 1, so opaque
// content never sees it — no mode flag needed.
uniform float uCheckerScale;

// ---- Compare overlay (v1.2) -----------------------------------------
// Two-layer A/B compare lives in the shader so dragging the seam is
// a uniform-update, not a full numpy compose + GL upload. ``uImage``
// holds layer A (the existing single-image texture); ``uImageB``
// holds layer B (uploaded once per frame_changed via
// ``GLViewport.set_compare_b``). ``uCompareMode`` selects the wipe
// shape:
//   0 = compare off (= use uImage as before, ignore uImageB)
//   1 = vertical wipe — left half from A, right half from B at seam
//   2 = horizontal wipe — top half from A, bottom half from B
//   3 = opacity blend — linear mix(A, B, seam)
//   4 = solo B — show only B regardless of seam
// ``uCompareSeam`` is the wipe / blend position in [0..1] (texture
// coords). ``uCompareSeamLineAlpha`` is the seam line's blend
// strength against an accent-orange tint; 0 disables the line.
uniform sampler2D uImageB;
uniform int uCompareMode;
uniform float uCompareSeam;
uniform float uCompareSeamLineAlpha;

@@OCIO_INJECT@@

// Pick the compare-mode source pixel for the current fragment. When
// compare is off this is just texture(uImage, …); for wipes it's a
// branch on vTexCoord; for opacity it's a linear mix.
vec4 compare_pick() {
    vec4 pa = texture(uImage, vTexCoord);
    if (uCompareMode == 0) {
        return pa;
    }
    vec4 pb = texture(uImageB, vTexCoord);
    if (uCompareMode == 4) {
        // Solo B — used by the band's A↔B toggle.
        return pb;
    }
    if (uCompareMode == 3) {
        // Opacity blend — same math as compose.MODE_OPACITY.
        return mix(pa, pb, clamp(uCompareSeam, 0.0, 1.0));
    }
    if (uCompareMode == 1) {
        // Vertical wipe. Left half (vTexCoord.x < seam) is A, right is B.
        return vTexCoord.x < uCompareSeam ? pa : pb;
    }
    if (uCompareMode == 2) {
        // Horizontal wipe. Top half is A, bottom is B. The vertex
        // shader already flipped vTexCoord.y so y=0 is the top of
        // the image (= row 0 of the numpy array). seam < y means
        // we're below the seam line → B.
        return vTexCoord.y < uCompareSeam ? pa : pb;
    }
    return pa;
}

// Apply a thin accent-orange tint at the wipe seam so the user can
// see the boundary. Width is one screen pixel via fwidth(); when
// compare mode isn't a wipe (= no spatial seam) we no-op.
vec3 paint_seam_tint(vec3 rgb) {
    if (uCompareSeamLineAlpha <= 0.0) {
        return rgb;
    }
    if (uCompareMode != 1 && uCompareMode != 2) {
        return rgb;
    }
    float coord = (uCompareMode == 1) ? vTexCoord.x : vTexCoord.y;
    float pixel_w = (uCompareMode == 1) ? fwidth(vTexCoord.x)
                                        : fwidth(vTexCoord.y);
    float dist = abs(coord - uCompareSeam);
    if (dist <= pixel_w) {
        // Accent orange (= H.ACCENT) blended at the configured alpha.
        vec3 accent = vec3(232.0 / 255.0, 144.0 / 255.0, 28.0 / 255.0);
        return mix(rgb, accent, uCompareSeamLineAlpha);
    }
    return rgb;
}

void main() {
    vec4 pixel = compare_pick();
    // Capture the raw alpha BEFORE OCIO. Most OCIO transforms leave
    // alpha untouched but some configs route it through display
    // tone-mapping curves — would corrupt the transparency composite
    // below. Using the raw value keeps the matte exact.
    float raw_alpha = pixel.a;

    // Exposure: multiplicative scale in scene-linear.
    pixel.rgb *= pow(2.0, uExposure);

    // Color transform: scene-linear input -> display-encoded output.
    pixel = OCIOMain(pixel);

    // Optional user gamma adjustment applied to the display-encoded output.
    pixel.rgb = pow(max(pixel.rgb, vec3(0.0)), vec3(1.0 / uGamma));

    // Channel masking — done last, in display space, so the result
    // is exactly "what the user said to show". A single-channel
    // isolation can promote that channel to luminance if the user
    // wants it on grey rather than coloured.
    vec3 masked = pixel.rgb * uChannelMask.rgb;
    float rgb_count = dot(uChannelMask.rgb, vec3(1.0));
    if (uChannelIsolateLuminance > 0.5 && rgb_count > 0.0 && rgb_count < 2.5) {
        // Single (or two) channel isolation: collapse to grey by
        // averaging the visible components.
        float lum = dot(masked, vec3(1.0)) / rgb_count;
        masked = vec3(lum);
    }
    // Always-on checker composite where the buffer's alpha < 1.
    // The cache produces premultiplied buffers (composite path
    // converts straight contributors to premult before the over
    // operator) — so a single ``masked + checker * (1 - a)`` formula
    // is correct here. Opaque content (alpha = 1) renders as before;
    // alpha < 1 reveals the checker. The uChannelMask.a toggle still
    // dims alpha-driven content for the legacy "show alpha as
    // brightness multiplier" workflow.
    vec2 cell = floor(gl_FragCoord.xy / max(uCheckerScale, 1.0));
    float c = mod(cell.x + cell.y, 2.0);
    vec3 checker = mix(vec3(0.40), vec3(0.55), c);
    float a = clamp(raw_alpha, 0.0, 1.0);
    vec3 final_rgb = masked * uChannelMask.a + checker * (1.0 - a);
    // Compare seam line — painted in display space so it stays a
    // consistent thickness regardless of zoom or OCIO output range.
    final_rgb = paint_seam_tint(final_rgb);
    fragColor = vec4(final_rgb, 1.0);
}
