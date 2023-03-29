#pragma once
#include "SmartGraphicsArray.cuh"
#include "Shader.h"
#include "Input.h"
#include "SDL.h"
#include "SDL_image.h"
//constexpr size_t mainPointSize = 32;
//constexpr size_t mainPointSize = 2048;
//constexpr size_t mainPointSize = 65536;
//constexpr size_t mainPointSize = 1000000;
//constexpr size_t mainPointSize = 2073600;
//constexpr size_t mainPointSize = 2097152;
//constexpr size_t mainPointSize = 4194304;
//constexpr size_t mainPointSize = 8388608;
constexpr size_t mainPointSize = 16777216;

//constexpr GLuint shadowTexSize = 4096;

template<size_t c>
class VAOSphere {
	GLuint vaoID;
	GLuint shaderID;

	GLuint shaderOBBID;

	
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