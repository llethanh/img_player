#version 410 core

// Fullscreen-quad vertex shader. The app draws a quad covering clip-space
// [-1, 1] in both X and Y, with texture coordinates [0, 1]. We flip the
// Y coordinate here because numpy arrays are top-to-bottom but GL textures
// are bottom-to-top by default.

layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aTexCoord;

out vec2 vTexCoord;

uniform mat4 uTransform;  // model * view * projection (letterboxing / pan / zoom)

void main() {
    gl_Position = uTransform * vec4(aPosition, 0.0, 1.0);
    vTexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
}
