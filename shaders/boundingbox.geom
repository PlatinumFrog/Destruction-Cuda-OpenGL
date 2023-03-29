//shadowMap.geom
#version 460

layout (points) in;
layout(triangle_strip, max_vertices = 10) out;

in v_out {
	vec4 col;
	vec4 box[8];
} i[];

out vec4 col;

void main() {
	
	col = i[0].col;
	gl_Position = i[0].box[0];
	EmitVertex();
	gl_Position = i[0].box[4];
	EmitVertex();

	gl_Position = i[0].box[1];
	EmitVertex();
	gl_Position = i[0].box[5];
	EmitVertex();

	gl_Position = i[0].box[3];
	EmitVertex();
	gl_Position = i[0].box[7];
	EmitVertex();

	gl_Position = i[0].box[2];
	EmitVertex();
	gl_Position = i[0].box[6];
	EmitVertex();

	gl_Position = i[0].box[0];
	EmitVertex();
	gl_Position = i[0].box[4];
	EmitVertex();

	EndPrimitive();
}