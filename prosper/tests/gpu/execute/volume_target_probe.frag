#version 460
layout(location = 0) out vec4 color;
#if defined(SAMPLE_VOLUME)
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
