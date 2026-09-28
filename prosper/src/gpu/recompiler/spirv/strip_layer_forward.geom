#version 450

// The admitted guest ES stores POS0, PARAM0.xy and a layer in one record per
// logical vertex. Its GS copies three records into each primitive and exports
// their shared layer. The host strip already supplies those triangles; this
// stage relays their current per-vertex values to the layered attachment.
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
layout(location = 0) in vec4 source_param0[];
layout(location = 1) in vec4 source_layer_bits[];
layout(location = 0) out vec4 forwarded_param0;
// The recompiled vertex stage's output block carries only gl_Position. Vulkan requires a
// Block-decorated interface shared between two stages to match member for member, so redeclare
// both sides of this stage with exactly that member rather than the implicit four-member block.
in gl_PerVertex { vec4 gl_Position; } gl_in[];
out gl_PerVertex { vec4 gl_Position; };

void main() {
    int layer = floatBitsToInt(source_layer_bits[0].x);
    for (int vertex = 0; vertex < 3; ++vertex) {
        gl_Position = gl_in[vertex].gl_Position;
        gl_Layer = layer;
        forwarded_param0 = source_param0[vertex];
        EmitVertex();
    }
    EndPrimitive();
}
