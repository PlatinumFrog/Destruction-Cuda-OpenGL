//Spheres.frag

#version 460
//#define DRAWSHAPE
//#define TEXTUREANDLIGHTING

layout (depth_less) out float gl_FragDepth;

#ifdef TEXTUREANDLIGHTING
uniform sampler2D mainTexture;
#endif


//uniform mat4 proj_matrix;
//uniform mat4 icam_matrix;

//uniform float time;

in vec4 color;
in vec3 viewray;
in vec4 vpos;
#ifdef TEXTUREANDLIGHTING
in vec4 wpos;
#endif
out vec4 fcolor;

#ifdef TEXTUREANDLIGHTING
void main() {
	float t = time * 62.83;
	vec3 lipos = vec3(500.0 * cos(t), 0.0, 500.0 * sin(t));
	vec3 rd = normalize(viewray);
	float b = dot(vpos.xyz, rd);
	float h = (b * b) - dot(vpos.xyz, vpos.xyz) + (vpos.w * vpos.w);
	bool miss = (h < 0.0);
	if(miss) discard;
	h = sqrt(h);
	float rl = b - h;
	bool inside = (rl < 0.01);
	if(inside) rl = b + h;
	if(rl < 0.01) discard;
	vec3 fragpos = (icam_matrix * vec4(rl * rd, 1.0)).xyz;
	vec3 no = normalize(fragpos - wpos.xyz);
	if(inside) no = -no;
	vec3 fraglpos = fragpos - lipos;
	vec2 uv = vec2(1.0 - ((sign(no.x) < 0) ? 0.0 : 0.5) - (0.5 * atan(no.z/no.x) / 3.141592) + time, 0.5 + (asin(no.y) / 3.141592));
	fcolor = vec4((-1000.0 * (dot(no, fraglpos)/dot(fraglpos, fraglpos))) * texture(mainTexture, uv).rgb, 1.0);
}
#else
void main() {
	const vec3 rd = normalize(viewray);
	const float b = dot(vpos.xyz, rd);
	float h = (b * b) - dot(vpos.xyz, vpos.xyz) + (vpos.w * vpos.w);
	if(h < 0.0) discard;
	h = sqrt(h);
	float rl = b - h;
	const float near = -8.0 / rd.z;
	const bool inside = rl < near;
	if(inside) rl = b + h;
	if(rl < near) discard;
    const vec3 rayhit = rl * rd;
	const vec3 raydir = vpos.xyz - rayhit;
    const vec3 no = normalize(raydir);
	const float light = 0.5 + (0.5 * dot(no, normalize(rayhit)));
	fcolor = vec4(color.xyz * (inside ? 0.85 - light : light), color.w);
	gl_FragDepth = rl*0.0001;
}
#endif
	
