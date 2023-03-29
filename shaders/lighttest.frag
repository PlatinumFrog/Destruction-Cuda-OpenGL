//Spheres.frag

#version 460

uniform mat4 icam_matrix;
uniform mat4 proj_matrix;
uniform float time;
in vec4 vpos;
in vec4 wpos;

void main() {
	float t = time * 62.83;
	vec3 lipos = vec3(500.0 * cos(t), 0.0, 500.0 * sin(t));
	vec3 rd = normalize(vec3((0.00185185 * gl_FragCoord.xy) - vec2(1.7777778, 1.0), -proj_matrix[1][1]));
	float b = dot(vpos.xyz, rd);
	float h = (b * b) - dot(vpos.xyz, vpos.xyz) + (vpos.w * vpos.w);
	if(h < 0.0) discard;
	h = sqrt(h);
	float rl = b - h;
	bool inside = (rl < 0.1);
	if(inside) rl = b + h;
	vec3 fragwpos = (icam_matrix * vec4(rl * rd, 1.0)).xyz;
	vec3 no = normalize(fragwpos - wpos.xyz);
	if(inside) no = -no;
	if((-1000.0 * (dot(no, fragwpos)/dot(fragwpos, fragwpos))) < 0.0) discard;
}