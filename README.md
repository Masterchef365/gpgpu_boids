# Le Fishe
Boid-like contraptions
Vertex shader reads directly from the boids array...

Draws `N*2` vertices, where `N` is the number of boids. So it would access like:

```glsl
struct Boid {
    vec3 position;
    float unused0;
    vec3 heading;
    float unused1;
}

layout(std430, binding = 0) uniform Scene {
    mat4 camera[2];
    uint frame_idx;
};

layout(std430, binding = 1) buffer Boids {
    Boid boids[];
}

layout(location = 0) out vec3 fragColor;

void main() {
    uint boid_idx = gl_VertexIndex / 2;
    Boid boid = boids[boid_idx];

    vec3 pos;
    if (gl_VertexIndex & 1 == 0) {
        pos = boid.position;
    } else {
        pos = boid.position + boid.heading;
    }

    gl_Position = camera[gl_ViewIndex] * model * vec4(pos, 1.0);
    fragColor = boid.heading;
}
```
N frames in flight UBOs
* Camera matrices
* Frame idx
N frames in flight descriptor sets for camera:
* Pointer to appropriate boid buffer
* Pointer to appropriate UBO


Rendering loop:
* ~~Barrier: Wait for vertex shader to stop before starting motion compute~~ Nope! If we're double buffered, we're readings from the same buf as the vertex
* ~~Barrier: Wait for motion compute... Nope! Don't need to, it will already have stopped.~~
* Update camera desriptor
* Bind appopriate motion descriptor set
* Start motion compute
* Barrier: Wait for motion compute to end before vertex shader start
* Start graphics pipeline 

A -> B
Wait for B before (Vertex, Compute)
Read B
B -> A
Wait for A before (Vertex, Compute)
Read A
A -> B
