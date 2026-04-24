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

@@OCIO_INJECT@@

void main() {
    vec4 pixel = texture(uImage, vTexCoord);

    // Exposure: multiplicative scale in scene-linear.
    pixel.rgb *= pow(2.0, uExposure);

    // Color transform: scene-linear input -> display-encoded output.
    pixel = OCIOMain(pixel);

    // Optional user gamma adjustment applied to the display-encoded output.
    pixel.rgb = pow(max(pixel.rgb, vec3(0.0)), vec3(1.0 / uGamma));

    fragColor = vec4(pixel.rgb, 1.0);
}
