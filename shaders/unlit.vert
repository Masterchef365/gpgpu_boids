#version 450
#extension GL_EXT_multiview : require

struct Boid {
    vec3 position;
    float unused0;
    vec3 heading;
    float unused1;
};

layout(binding = 0) uniform Scene {
    mat4 camera[2];
    uint frame_idx;
};

layout(std430, binding = 1) buffer Boids {
    Boid boids[];
};

layout(location = 0) out vec3 fragColor;

void main() {
    uint boid_idx = gl_VertexIndex / 2;
    Boid boid = boids[boid_idx];

    vec3 pos;
    if ((gl_VertexIndex & 1) == 0) {
        pos = boid.position;
    } else {
        pos = boid.position + boid.heading;
    }

    gl_Position = camera[gl_ViewIndex] * vec4(pos, 1.0);
    fragColor = boid.heading;
}
