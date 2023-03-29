#version 460

//#define TEXTUREANDLIGHTING

layout(location = 0) in vec4 position; //x, y, z, radius
layout(location = 1) in vec4 color; //red, green, blue, alpha

layout(location = 2) uniform mat4 proj_matrix;

layout(location = 6) uniform mat4 cam_matrix;

#ifdef TEXTUREANDLIGHTING
layout(location = 10) uniform mat4 icam_matrix;
layout(location = 14) uniform float time;
#endif

out v_out {
	vec4 col;
#ifdef TEXTUREANDLIGHTING
	vec4 wpos;
#endif
	vec4 vpos;
	vec4 v[6];
} o;

void main() {
	o.col = color;
#ifdef TEXTUREANDLIGHTING
	o.wpos = position;
#endif
	vec4 p = vec4((cam_matrix * vec4(position.xyz, 1.0)).xyz, position.w);
	o.vpos = p;

	float l2 = dot(p.xyz, p.xyz);
	float r2 = (p.w * p.w);
	float k = 1.0 - (r2/l2);
	float radius = p.w * sqrt(k);
	if(l2 < r2) {
		p = vec4(0.0, 0.0, -1.0, p.w);
		radius = 1.0;
		k = 0.245 * proj_matrix[1][1];
	}
	vec3 hx = radius * normalize(vec3(-p.z, 0.0, p.x));
	vec3 hy = radius * normalize(vec3(-p.x * p.y, p.z * p.z + p.x * p.x, -p.z * p.y));
	p.xyz *= k;
	hx *= 0.57735026919;
	vec3 hv[6];
	hv[0] = vec3(2.0 * hx);
	hv[1] = vec3(hx + hy);
	hv[2] = vec3(hx - hy);
	hv[3] = vec3(p.xyz - hv[2]);
	hv[4] = vec3(p.xyz - hv[1]);
	hv[5] = vec3(p.xyz - hv[0]);
	hv[0] += p.xyz;
	hv[1] += p.xyz;
	hv[2] += p.xyz;
	o.v[0] = proj_matrix * vec4(hv[0], p.w);
	o.v[1] = proj_matrix * vec4(hv[1], p.w);
	o.v[2] = proj_matrix * vec4(hv[2], p.w);
	o.v[3] = proj_matrix * vec4(hv[3], p.w);
	o.v[4] = proj_matrix * vec4(hv[4], p.w);
	o.v[5] = proj_matrix * vec4(hv[5], p.w);
	gl_Position = p;
}
