#version 460

layout(location = 0) in vec4 position; //x, y, z, radius
layout(location = 1) in vec4 color; //red, green, blue, alpha

layout(location = 2) uniform mat4 view_matrix;

out v_out {
	vec4 col;
	vec4 box[8];
} o;

void main() {

	o.col = color;

	vec4 p = position;
	vec3 v = color.xyz;

	vec3 hz = p.w * normalize(v);
	vec3 hx = p.w * normalize(vec3(-v.z, 0.0, v.x));
	vec3 hy = p.w * normalize(vec3(-v.x * v.y, v.z * v.z + v.x * v.x, -v.z * v.y));
	
	vec3 ur = hy + hx;
	vec3 bl = p.xyz - ur;
	ur += p.xyz;
	vec3 ul = hy - hx;
	vec3 br = p.xyz - ul;
	ul += p.xyz;
	vec3 vf = hz + v;
	vec3 vur = ur + vf;
	ur -= hz;
	vec3 vul = ul + vf;
	ul -= hz;
	vec3 vbr = br + vf;
	br -= hz;
	vec3 vbl = bl + vf;
	bl -= hz;
	o.box[0] = view_matrix * vec4(ur, 1.0);
	o.box[1] = view_matrix * vec4(ul, 1.0);
	o.box[2] = view_matrix * vec4(br, 1.0);
	o.box[3] = view_matrix * vec4(bl, 1.0);
	o.box[4] = view_matrix * vec4(vur, 1.0);
	o.box[5] = view_matrix * vec4(vul, 1.0);
	o.box[6] = view_matrix * vec4(vbr, 1.0);
	o.box[7] = view_matrix * vec4(vbl, 1.0);

	gl_Position = p;

}
