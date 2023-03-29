//Spheres.frag

#version 460
//#define DRAWSHAPE
//#define TEXTUREANDLIGHTING

layout (depth_less) out float gl_FragDepth;

#ifdef TEXTUREANDLIGHTING
uniform sampler2D mainTexture;
#endif


uniform mat4 proj_matrix;

#ifdef TEXTUREANDLIGHTING
uniform mat4 icam_matrix;
uniform float time;
#endif

in vec4 color;
in vec4 vpos;
#ifdef TEXTUREANDLIGHTING
in vec4 wpos;
#endif
out vec4 fcolor;

void main() {

#ifdef TEXTUREANDLIGHTING
	float t = time * 62.83;//vec3(100.0 * cos(t), 0.0, 100.0 * sin(t))
	vec3 lipos = vec3(500.0 * cos(t), 0.0, 500.0 * sin(t));
#endif
	//ray direction in view space
	vec3 rd = normalize(vec3((0.00185185 * gl_FragCoord.xy) - vec2(1.7777778, 1.0), -proj_matrix[1][1]));
	
	//find intersection between sphere and ray direction
	float b = dot(vpos.xyz, rd);
	float h = (b * b) - dot(vpos.xyz, vpos.xyz) + (vpos.w * vpos.w);

	//if ray misses the sphere, discard
	bool miss = (h < 0.0);

#ifndef DRAWSHAPE
	if(miss) discard;
#endif
	
	h = sqrt(h);

	float rl = b - h;
	//if ray is short enough, calculate interior of sphere instead
	bool inside = (rl < 0.0001);
	if(inside) rl = b + h;
	if(rl < 0.0001) discard;

	
#ifdef TEXTUREANDLIGHTING
	//world space position of fragment on sphere surface
	vec3 fragpos = (icam_matrix * vec4(rl * rd, 1.0)).xyz;
	//normal of sphere at the surface.
	vec3 no = normalize(fragpos - wpos.xyz);
	if(inside) no = -no;
	vec3 fraglpos = fragwpos - lipos;
	//get texture and lighting
	vec2 uv = vec2(1.0 - ((sign(no.x) < 0) ? 0.0 : 0.5) - (0.5 * atan(no.z/no.x) / 3.141592) + time, 0.5 + (asin(no.y) / 3.141592));
	fcolor = vec4((-1000.0 * (dot(no, fraglpos)/dot(fraglpos, fraglpos))) * texture(mainTexture, uv).rgb, 1.0);
#else
	//world space position of fragment on sphere surfacedot(no, vpos.xyz)/dot(vpos.xyz, vpos.xyz)
	vec3 fragpos = rl * rd;
	vec3 no = normalize(vpos.xyz - fragpos);
	if(inside) no = -no;
	fcolor = vec4(clamp(color.xyz,0.0,1.0) * dot(no, normalize(fragpos)), color.w);
#endif
	

	

	gl_FragDepth = 0.0001 * rl;
#ifdef DRAWSHAPE
	if(miss) {
		fcolor = vec4(1.0);
		gl_FragDepth = 0.0001 * gl_FragCoord.z;
	}
#endif
	
}