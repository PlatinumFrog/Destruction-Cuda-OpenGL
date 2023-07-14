#pragma once

#include "SmartGraphicsArray.cuh"
#include "Shader.h"
#include "Input.h"
#include "SDL.h"
#include "SDL_image.h"

//#define USE_TEXTURE
//#define USE_LIGHTING

//constexpr GLuint shadowTexSize = 4096;

template<size_t c>
class VAOSphere {
	GLuint vaoID;
	GLuint shaderID;

	//GLuint shaderOBBID;

	
	GLuint imgID;


	bool enabled;
	bool shaderCompileToggle;
	float shaderAnim;
public:
	
	d_gArray<4> position;
	d_gArray<4> color;

	VAOSphere();
	~VAOSphere();

	void enable();
	void disable();

	void draw(size_t s);

};