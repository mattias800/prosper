#version 460
layout(location = 0) out vec4 color;
#if defined(PARAM_MRT2)
// Two colour targets from one layered draw (#4643): slot 1 carries the layer in GREEN and a
// constant blue, so a swapped, dropped or slot-0-aliased attachment cannot pass the readback.
layout(location = 0) in vec4 layer_param;
layout(location = 1) out vec4 color1;
void main() {
    color = layer_param;
    color1 = vec4(0.0, layer_param.r, 1.0, 1.0);
}
#elif defined(PARAM_COLOR)
layout(location = 0) in vec4 layer_param;
void main() { color = layer_param; }
#elif defined(SAMPLE_VOLUME)
layout(set = 1, binding = 4) uniform sampler3D source_volume;
void main() {
    int slice = clamp(int(gl_FragCoord.x / 16.0), 0, 3);
    color = texelFetch(source_volume, ivec3(0, 0, slice), 0);
}
#elif defined(GREEN)
void main() { color = vec4(0.0, 1.0, 0.0, 1.0); }
#else
void main() { color = vec4(1.0, 0.0, 0.0, 1.0); }
#endif
