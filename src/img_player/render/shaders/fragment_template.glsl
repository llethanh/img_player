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

@@OCIO_INJECT@@

void main() {
    vec4 pixel = texture(uImage, vTexCoord);
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
    fragColor = vec4(final_rgb, 1.0);
}
