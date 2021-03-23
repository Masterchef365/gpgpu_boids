#version 450

layout(location = 0) in vec3 fragColor;

layout(binding = 1) uniform Animation {
    float anim;
};

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(abs(fragColor), 1.0);
}
