//Spheres.geom

#version 460
#pragma optionNV (unroll all)
//#define TEXTUREANDLIGHTING

layout (points) in;

const int sides = 6;
const float ds = 1.0 / float(sides);
const float sc = 1.0 / cos(3.14159265359 * ds);
layout(triangle_strip, max_vertices = sides) out;

layout(location = 2) uniform mat4 proj_matrix;
layout(location = 6) uniform mat4 cam_matrix;

in v_out {
	vec4 col;
	//vec4 wpos;
	//vec4 vpos;
	//vec4 v[6];
} i[];

out vec4 color;
out vec3 viewray;
out vec4 vpos;
//out vec4 wpos;

void main() {
	color = i[0].col;
	//wpos = gl_in[0].gl_Position;
	vec3 p = (cam_matrix * vec4(gl_in[0].gl_Position.xyz, 1.0)).xyz;
	vpos = vec4(p, gl_in[0].gl_Position.w);
	const float l2 = dot(p.xyz, p.xyz);
	const float r2 = (gl_in[0].gl_Position.w * gl_in[0].gl_Position.w);
	float k = 1.0 - (r2/l2);
	float r = gl_in[0].gl_Position.w * sqrt(k);
	if(l2 <= r2 * 1.001 && gl_in[0].gl_Position.w > 8.0) {
		k = 0.245 * proj_matrix[1][1];
		p = vec3(0.0, 0.0, -1.0);
		r = 1.0;
	}
	const mat2x3 hm = mat2x3(normalize(vec3(-p.z, 0.0, p.x)), normalize(vec3(-p.x * p.y, p.x * p.x + p.z * p.z, -p.y * p.z)));
	p *= k;
	for(int l = 0; l < sides; l++) {
		const float u = 6.28318530718 * float(((l & 1) == 1) ? sides - ((l + 1) >> 1) : (l >> 1)) * ds;
		const vec2 v = sc * vec2(cos(u), sin(u));
		const vec3 vertex = hm * vec2(r * v) + p;
		viewray = vertex;
		gl_Position = proj_matrix * vec4(vertex, 1.0);
		EmitVertex();
	}
	EndPrimitive();
}