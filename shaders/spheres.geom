//Spheres.geom

#version 460

//#define TEXTUREANDLIGHTING

layout (points) in;

layout(triangle_strip, max_vertices = 6) out;

in v_out {
	vec4 col;
#ifdef TEXTUREANDLIGHTING
	vec4 wpos;
#endif
	vec4 vpos;
	vec4 v[6];
} i[];

out vec4 color;
out vec4 vpos;

#ifdef TEXTUREANDLIGHTING
out vec4 wpos;
#endif

void main() {
	vpos = i[0].vpos;

#ifdef TEXTUREANDLIGHTING
	wpos = i[0].wpos;
#else
	color = i[0].col;
#endif

	gl_Position = i[0].v[0];
	EmitVertex();
	gl_Position = i[0].v[1];
	EmitVertex();
	gl_Position = i[0].v[2];
	EmitVertex();
	gl_Position = i[0].v[3];
	EmitVertex();
	gl_Position = i[0].v[4];
	EmitVertex();
	gl_Position = i[0].v[5];
	EmitVertex();
	EndPrimitive();
}