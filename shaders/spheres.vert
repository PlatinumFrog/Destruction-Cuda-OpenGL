#version 460

//#define TEXTUREANDLIGHTING

layout(location = 0) in vec4 position; //x, y, z, radius
layout(location = 1) in vec4 color; //red, green, blue, alpha

layout(location = 2) uniform mat4 proj_matrix;
layout(location = 6) uniform mat4 cam_matrix;
//layout(location = 10) uniform mat4 icam_matrix;
//layout(location = 14) uniform float time;

out v_out {
	vec4 col;
	//vec4 wpos;
	//vec4 vpos;
	//vec4 v[6];
} o;

void main() {
	o.col = color;
	gl_Position = position;
}
