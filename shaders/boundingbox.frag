//shadowMap.frag
#version 460
layout (depth_less) out float gl_FragDepth;

in vec4 col;

out vec4 c;

void main() {
	c = vec4((0.5 * col.xyz) + 0.5, 1.0);
	gl_FragDepth = 1.0 - (gl_FragCoord.w);
}